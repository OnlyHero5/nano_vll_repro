# 02. 补齐 Qwen3 模型主干与权重映射（按 HF 语义对齐）

## 1. 本篇目标

这一篇的任务不是“把模型文件写长一点”，而是把你现在的 Qwen3 骨架改成：

1. 配置字段能正确承接 HF `Qwen3Config`
2. 注意力层顺序和 HF 真正一致
3. `from_pretrained()`、`packed_modules_mapping`、`compute_logits()` 这些边界清晰可用
4. 后续 Day4 / Day5 / Day6 不需要再回头重构模型主干

做完以后，至少应达到下面的状态：

- `config.py` 可以稳定暴露 `head_dim / num_key_value_heads / rope_parameters / attention_bias`
- `layers/rotary_embedding.py` 不再只认死板的 `rope_theta`
- `models/qwen3.py` 里的 `Qwen3Attention / Qwen3DecoderLayer / Qwen3ForCausalLM` 语义与 HF 对齐
- `tests/test_Day2.py` 不再调用已经不存在或即将废弃的旧接口

---

## 2. 权威参考

本篇强制对照 4 个文件：

1. 当前仓库：
   - `nano_vll_repro/config.py`
   - `nano_vll_repro/layers/rotary_embedding.py`
   - `nano_vll_repro/models/qwen3.py`
   - `nano_vll_repro/tests/test_Day2.py`
2. HF 配置：
   - `transformers/.../configuration_qwen3.py`
3. HF 模型：
   - `transformers/.../modeling_qwen3.py`
4. 上游教学实现：
   - `nano-vllm/nanovllm/models/qwen3.py`

本篇最重要的判断标准不是“像不像上游教学代码”，而是：

> 你的模型语义是否真的和 HF `Qwen3` 一致。

---

## 3. 先看当前仓库到底缺什么

### 3.1 `config.py` 的问题

当前 `Config` 只有最基础的路径和显存参数，但后续链路至少还要依赖这些字段：

- `dtype`
- `kv_cache_dtype`
- `max_cudagraph_batch_size`
- `head_dim`
- `num_key_value_heads`
- `attention_bias`
- `hidden_act`
- `tie_word_embeddings`
- `rope_parameters`
- `layer_types`
- `use_sliding_window / sliding_window`

如果这一步不补，后面你每写一篇都会继续在各处手搓 `getattr(...)`。

### 3.2 `rotary_embedding.py` 的问题

当前文件基本还是上游早期简化版，只支持：

- `base=rope_theta`
- `rotary_dim == head_size`

这在“只想跑一个最小 demo”时没问题，但你现在已经明确要按 HF `Qwen3` 语义继续走，所以至少要能：

- 接受 `rope_parameters`
- 明确写出当前仓库“只支持默认 RoPE，不支持完整动态外推”的边界

### 3.3 `models/qwen3.py` 的问题

这里有 7 个关键缺口：

1. `Qwen3Attention` 里头数与 `head_dim` 的语义还是“能跑就行”的状态。
2. `q_norm / k_norm` 虽然写了，但没有明确说明它们必须在 head 维度上做、且在 RoPE 之前做。
3. `Qwen3DecoderLayer` 现在用了 fused `RMSNorm`，但文档没解释它和 HF residual 顺序为什么等价。
4. 线性层构造参数还停留在旧版骨架风格，和 `01` 里补齐后的 `QKVLinear / MergedLinear` 签名对不上。
5. `Qwen3ForCausalLM.forward()` 直接返回 logits，后面 Day4 / Day6 做 runner 和 graph 时不够灵活。
6. `compute_logits()` 缺失，导致 logits head 和主干边界不清楚。
7. `from_pretrained()` 只是打印信息，没有把“结构创建”和“权重加载”边界解释清楚。

### 3.4 `tests/test_Day2.py` 的问题

这个测试文件已经暴露出至少两处接口漂移：

1. `test_gqa()` 还在给 `Qwen3Attention` 传 `attention_mask=None`
2. `test_qwen3_model()` 默认认为 `model(input_ids)` 直接返回 logits

如果你本篇把模型接口收口，这两个测试都要同步改。

---

## 4. 本篇修改原则

### 4.1 不把模型重写成 HF 原样

HF 用的是通用 batch 维接口、cache 抽象、mask 系统；你的仓库当前是单维 token 流 + 全局 `Context`。

所以本篇只对齐“语义”，不照搬“外壳”：

- 保留你当前 `positions + hidden_states` 的调用方式
- 保留 fused `RMSNorm` API
- 但让层内顺序、字段来源、权重映射都和 HF 对齐

### 4.2 现在就把后续需要的配置字段补出来

尤其是：

- `dtype / kv_cache_dtype / max_cudagraph_batch_size`

这三个字段如果现在不补，后面 04~06 会反复来回改 `Config`。

### 4.3 把 logits head 和主干拆开

后面做 `ModelRunner.run_model()` 和 `CUDA Graph` 时，最干净的边界是：

- `Qwen3ForCausalLM.forward()` 返回 hidden states
- `compute_logits(hidden_states)` 单独负责 vocab 投影

这是本篇最值得提前做的接口收口。

---

## 5. 逐步修改

## 5.1 先扩 `config.py`，但保留 `model_path` 这个本地命名

修改位置：

- 文件：`nano_vll_repro/config.py`
- 锚点 1：定位到 `@dataclass class Config:` 的字段定义区，在 `enforce_eager` 后、`hf_config` 前插入新的 dtype / cudagraph 字段
- 锚点 2：定位到 `class Config` 内已有 property 区，把下面给出的 property 逐个补进去；如果同名 property 已存在，就整段替换

不要为了对齐上游把字段名直接改成 `model`。你当前仓库大量地方已经在用 `model_path`，现在没有必要为了“更像上游”引入一轮无收益重命名。

推荐做法是：

1. 继续保留 `model_path`
2. 补一个 `model` property 作为别名
3. 把后续文档会用到的字段一次补齐

建议你在 dataclass 里新增这些字段：

```python
dtype: str = "auto"
kv_cache_dtype: str = "auto"
max_cudagraph_batch_size: int = 32
```

然后把属性访问面扩成下面这种风格：

```python
@property
def head_dim(self) -> int:
    """
    输入：无。
    输出：单头维度。

    为什么不能简单写 hidden_size // num_attention_heads：
    因为 HF 的 Qwen3Config 允许显式给出 head_dim；
    只有配置里没有该字段时，才回退到 hidden_size // num_attention_heads。
    """
    return getattr(self.hf_config, "head_dim", self.hidden_size // self.num_attention_heads)


@property
def rope_parameters(self):
    """
    输入：无。
    输出：HF 配置里的 rope_parameters；如果模型配置没有该字段，则返回 None。

    后续 `rotary_embedding.py` 只从这里拿数据，不要让每个调用点都重复自己解析 HF config。
    """
    return getattr(self.hf_config, "rope_parameters", None)


@property
def rope_theta(self) -> float:
    """
    输入：无。
    输出：当前仓库实际使用的 rope theta。

    兼容策略：
    1. 如果 rope_parameters 是 dict，优先取其中的 rope_theta；
    2. 如果没有，再回退到老式的 hf_config.rope_theta；
    3. 最后才给默认值。
    """
    rope_parameters = self.rope_parameters
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_theta", rope_parameters.get("base", 1000000.0))
    return getattr(self.hf_config, "rope_theta", 1000000.0)
```

还要补两个 dtype property，因为 04 / 05 / 06 都会直接依赖：

```python
@property
def torch_dtype(self) -> torch.dtype:
    """
    输入：无。
    输出：模型权重与主干计算使用的 torch.dtype。

    这里故意支持 "auto"：
    - 有 CUDA 且支持 BF16 时优先 BF16；
    - 否则回退 FP16 / FP32；
    - 不把 dtype 决策写死在 ModelRunner 里。
    """
    if self.dtype == "bfloat16":
        return torch.bfloat16
    if self.dtype == "float16":
        return torch.float16
    if self.dtype == "float32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


@property
def kv_torch_dtype(self) -> torch.dtype:
    """
    输入：无。
    输出：KV Cache 使用的 dtype。

    单独拆出来的原因：
    后续很容易出现“模型用 BF16，但 KV Cache 暂时仍想用 FP16”的场景。
    """
    if self.kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if self.kv_cache_dtype == "float16":
        return torch.float16
    if self.kv_cache_dtype == "float32":
        return torch.float32
    if self.kv_cache_dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"不支持的 kv_cache_dtype: {self.kv_cache_dtype}")
```

---

## 5.2 在 `rotary_embedding.py` 里把“支持范围”和“暂不支持范围”都写清楚

修改位置：

- 文件：`nano_vll_repro/layers/rotary_embedding.py`
- 锚点：定位到文件底部现有的 `get_rope(...)` 工厂函数，从函数签名开始到 `return RotaryEmbedding(...)` 结束，整段替换为下面这份实现

这一步不要追求一次支持 HF 所有 RoPE 变体。对你当前仓库最现实的目标是：

1. 支持从 `rope_parameters` 里解析 `rope_theta`
2. 如果有人传入动态缩放 / 外推配置，直接明确报错

建议你只改 `get_rope()` 的参数面和边界检查，不要把整个 `RotaryEmbedding` 重写一遍。

重点片段可以写成：

```python
@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """
    输入：
    - head_size / rotary_dim / max_position / base: 当前仓库真正需要的最小 RoPE 参数集
    - rope_scaling: 预留给 HF 配置的兼容入口

    输出：RotaryEmbedding 实例

    当前策略写死说明：
    - 这份教学仓库只支持默认 RoPE；
    - 如果你后面要做 yarn / longrope / dynamic_ntk，再单独开一篇，不要在这里偷塞半套逻辑。
    """
    assert rope_scaling is None, "当前教学仓库只支持默认 RoPE，暂不支持 rope_scaling"
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
```

这样做的好处是：

- 接口已经和 HF 接上了
- 但实现边界仍然明确，不会制造“看起来支持，实际上偷偷忽略参数”的假象

---

## 5.3 重写 `Qwen3Attention` 的“头数语义”，不是重写整层

当前 `Qwen3Attention` 最大的问题不是 forward 写错，而是“局部头数 / 总头数 / head_dim”关系没有被说清楚。

### 先把线性层构造参数改到和 `01` 一致

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 锚点 1：定位到 `class Qwen3Attention.__init__`
- 锚点 2：定位到 `class Qwen3MLP.__init__`
- 要求：下面给出的构造器片段是“替代代码”，不是新增注释

在真正改 forward 之前，先把构造器用法对齐到 `01` 里补齐后的线性层签名。

`qkv_proj` 不应该再写成旧的：

```python
self.qkv_proj = QKVLinear(
    hidden_size=hidden_size,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    head_dim=self.head_dim,
    bias=qkv_bias,
)
```

而应该改成：

```python
self.qkv_proj = QKVLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=self.total_num_heads,
    total_num_kv_heads=self.total_num_kv_heads,
    bias=qkv_bias,
)
```

`Qwen3MLP` 里的 `gate_up_proj` 也一样，旧的：

```python
self.gate_up_proj = MergedLinear(
    input_size=hidden_size,
    output_size=intermediate_size,
    num_shards=2,
    bias=False,
)
```

要改成：

```python
self.gate_up_proj = MergedLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size, intermediate_size],
    bias=False,
)
```

这一步看起来像“只是改参数名”，但本质上是在让模型主干和 `01` 的融合线性层契约重新对齐。

这里建议你按下面顺序收口：

1. 先保留 `hidden_size / num_heads / num_kv_heads / head_dim`
2. 显式区分 `total_num_heads` 和 `num_heads`
3. 先让单卡时两者相等，为 Day5 的 TP 留口子
4. `q_size / kv_size` 全部基于“当前模块真正输出的本地头数”计算

建议的局部结构如下：

```python
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        max_position: int = 4096 * 32,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 1000000.0,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()

        # 这里先把“总头数”保留下来；单卡时 total == local，
        # 但 Day5 做 TP 时就会把这两个概念拆开。
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        # 当前阶段先按单卡处理，所以 local 头数先等于 total。
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # head_dim 一定优先信任配置，而不是永远自己算。
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
```

然后把投影顺序固定成 HF 的语义：

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 锚点：定位到 `class Qwen3Attention.forward` 内部，从 `qkv = self.qkv_proj(...)` 开始，到 `q, k = self.rotary_emb(...)` 结束，替换为下面这段

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

# 先 reshape 成多头，再在 head_dim 上做 q_norm / k_norm。
q = q.view(num_tokens, self.num_heads, self.head_dim)
k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

# 注意：Qwen3 的 q_norm / k_norm 是作用在 head_dim 上，而不是 hidden_size 上。
q = self.q_norm(q)
k = self.k_norm(k)

# RoPE 一定在 q/k norm 之后。
q, k = self.rotary_emb(positions, q, k)
```

这里不要再保留旧版大段注释掉的“手写 attention fallback”逻辑。那段代码会模糊当前 Attention 层的真实职责：

- `Qwen3Attention` 负责投影、norm、RoPE、调 `Attention`
- 真正的 prefill / decode 分流由 `layers/attention.py` + `Context` 决定

---

## 5.4 在 `Qwen3DecoderLayer` 里把 fused RMSNorm 和 HF residual 顺序讲明白

你当前仓库有个很容易让后续读者误解的点：

- 代码是 fused `RMSNorm(x, residual)`
- 但 HF 是显式的 `residual + norm + sublayer`

这两者不是冲突关系，而是“语义等价、实现不同”。

建议你保留 fused API，但把 forward 改成下面这种可读性更强的形态：

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 锚点：定位到 `class Qwen3DecoderLayer.forward`，从方法签名开始到 `return hidden_states, residual` 结束，整段替换

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
    1. positions: 当前 token 位置。
    2. hidden_states: 当前层输入。
    3. residual: 累积残差；首层为 None。

    输出：
    1. hidden_states: 当前层子模块输出。
    2. residual: 供下一层继续复用的残差。

    语义说明：
    这份写法虽然用了 fused RMSNorm，但它等价于 HF 的：
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states = self_attn(hidden_states)
        hidden_states = residual + hidden_states
    """
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

你这里真正要教会后续自己的不是 fused API，而是：

- residual 何时更新
- 为什么更新后还要把 residual 单独往下传

---

## 5.5 把 `Qwen3ForCausalLM` 改成“主干输出 hidden states，lm_head 单独算 logits”

这是本篇最重要的接口重构。

推荐目标结构：

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 锚点 1：定位到 `class Qwen3ForCausalLM`
- 锚点 2：把 `forward(...)` 与 `from_pretrained(...)` 改成下面这份完整替代代码
- 锚点 3：如果当前类里没有 `compute_logits(...)`，就在 `forward(...)` 后面直接插入

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        输入：token ids 与可选位置。
        输出：最后一层 hidden states，而不是 logits。

        为什么现在就拆开：
        - Day4 的 ModelRunner 更适合显式控制“先主干、再 logits、再 sampler”；
        - Day6 的 CUDA Graph 更适合捕获 hidden states 主干，而不是超大的 vocab logits。
        """
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        输入：主干输出的 hidden states。
        输出：词表维 logits。
        """
        return self.lm_head(hidden_states)
```

`from_pretrained()` 则只负责“根据 HF 配置实例化结构”，不要在这里偷偷加载权重：

```python
@classmethod
def from_pretrained(cls, model_path: str):
    """
    输入：HF 模型目录。
    输出：仅已创建结构、尚未加载 safetensors 的模型实例。

    这样做的原因：
    - 结构创建属于模型类职责；
    - 实际权重写入由 utils.loader.load_model() 负责；
    - 这两个阶段分开后，后面的测试和调试都更清楚。
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return cls(config)
```

---

## 5.6 同步修 `tests/test_Day2.py`，不要让测试继续绑死旧接口

本篇最少要改两处测试。

### 改动 1：`test_qwen3_model()`

修改位置：

- 文件：`nano_vll_repro/tests/test_Day2.py`
- 锚点：定位到 `def test_qwen3_model()` 里第一次对 `model(...)` 的调用，把原来的 `logits = model(input_ids)` 调用段替换为下面这段

旧写法：

```python
logits = model(input_ids)
```

改成：

```python
positions = torch.arange(num_tokens)
hidden_states = model(input_ids, positions)
logits = model.compute_logits(hidden_states)
```

原因：

- 这一步不是“多写几行”，而是让测试和你后续 runner 的真实调用方式统一

### 改动 2：`test_gqa()`

修改位置：

- 文件：`nano_vll_repro/tests/test_Day2.py`
- 锚点：定位到 `def test_gqa()` 里的 `attn(...)` 调用，把旧的三参数调用替换为下面这条两参数调用

把旧的：

```python
output = attn(positions, hidden_states, attention_mask=None)
```

改成：

```python
output = attn(positions, hidden_states)
```

原因：

- 当前仓库的 `Qwen3Attention.forward()` 不吃 `attention_mask`
- mask / prefill / decode 分流是在 `layers/attention.py` 里通过 `Context` 做的

---

## 6. 本篇结束后的最小验收

先做语法级检查：

```bash
cd nano_vll_repro
python -m py_compile config.py layers/rotary_embedding.py models/qwen3.py
```

再跑 Day2：

```bash
python tests/test_Day2.py
```

如果你想单独验证模型接口边界，可以手动执行：

```bash
python - <<'PY'
import torch
from models.qwen3 import Qwen3ForCausalLM

model = Qwen3ForCausalLM.from_pretrained("models/Qwen3-0.6B")
input_ids = torch.tensor([1, 2, 3, 4])
positions = torch.arange(4)
hidden = model(input_ids, positions)
logits = model.compute_logits(hidden)
print(hidden.shape, logits.shape)
PY
```

---

## 7. 常见错误

### 7.1 继续把 `forward()` 当成“必须直接吐 logits”

后果：

- 后面 `ModelRunner` 不得不把“主干前向 + vocab 投影 + sampler”揉在一起
- `CUDA Graph` 时更难控制捕获边界

### 7.2 把 `q_norm / k_norm` 放在 reshape 之前

后果：

- 你做的是对 `hidden_size` 维归一化，不是 HF 的 `head_dim` 归一化
- 数值语义已经不是 Qwen3 了

### 7.3 `rope_parameters` 表面接了，实际完全忽略

后果：

- 文档会给人一种“已经兼容 HF 配置”的错觉
- 还不如明确断言“当前只支持默认 RoPE”

### 7.4 测试文件不改

后果：

- 你会误以为模型实现有 bug
- 实际上只是测试脚本仍在调用旧接口

---

## 8. 本篇真正学到的东西

这一篇你真正要掌握的是：

1. HF `Qwen3` 的语义边界到底在哪里。
2. 为什么模型主干和 logits head 最好拆开。
3. 为什么“能跑的教学骨架”和“可继续演进的仓库主干”不是一回事。

完成后进入下一篇：

- [03-补全Sampler与SamplingParams.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/03-补全Sampler与SamplingParams.md)

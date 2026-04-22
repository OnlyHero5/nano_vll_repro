# 02. 补齐 Qwen3 模型主干与权重映射（先收口单卡接口，再为 Day4/Day5 留边界）

## 1. 本篇目标

这一篇的任务不是“把 `models/qwen3.py` 改得更像上游 TP 版本”，而是先把你当前仓库里真正已经存在的单卡模型骨架收口成一条清晰、可继续演进的主干。

本篇完成后，至少应达到下面 5 个状态：

1. `config.py` 能统一暴露模型语义字段，而不是让每个调用点都自己 `getattr(...)`。
2. `layers/rotary_embedding.py` 明确写出“当前只支持默认 RoPE”的边界，并给 `rope_scaling` 留兼容入口。
3. `models/qwen3.py` 继续使用当前仓库真实存在的 `QKVLinear / MergedLinear / RowLinear`，不提前切换到上游 TP 专用类。
4. `Qwen3ForCausalLM.forward()` 返回 hidden states，`compute_logits()` 单独负责 vocab projection。
5. `tests/test_Day2.py` 的接口调用方式与新主干一致，不再继续绑死旧返回值约定。

这里先把阶段边界写死：

> 本篇仍然是“单卡语义 + 当前本地线性层接口”。上游 `nano-vllm` 当前主干已经使用 `QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear`，但那属于 [05-实现Tensor-Parallel基础版.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/05-实现Tensor-Parallel基础版.md) 的升级范围，不应该提前偷带进来。

---

## 2. 权威参考

本篇只对照下面 4 组来源：

1. 当前仓库：
   - `nano_vll_repro/config.py`
   - `nano_vll_repro/layers/rotary_embedding.py`
   - `nano_vll_repro/models/qwen3.py`
   - `nano_vll_repro/tests/test_Day2.py`
2. 上游主仓库：
   - `https://github.com/GeeeekExplorer/nano-vllm`
   - `nanovllm/config.py`
   - `nanovllm/models/qwen3.py`
   - `nanovllm/layers/rotary_embedding.py`
3. Hugging Face：
   - `transformers` 中的 `Qwen3Config`
   - `transformers` 中的 `Qwen3` 模型实现
4. 公开变体：
   - `qqtang-code/nano-vllm`
   - `wangyuzhuo116/nano-vllm`
   - `DIOYF/nano-vllm-dio`

我这次重新核对后的结论要先说明白：

1. 上游主仓库当前确实已经是 TP 友好的实现，根目录公开页面显示有 `bench.py`、`example.py`、`nanovllm/` 主包，并在 README 中把 Tensor Parallelism / CUDA graph 写进特性列表。
2. 你文档里之前引用的几个变体，目前公开页面仍然是 fork 形态，根目录结构、README 和提交数都与上游主仓库保持同一代际，没有给你提供一套新的单卡接口标准。
3. 因此本篇真正应该做的是：
   - 参考上游和 HF 的“语义边界”
   - 但继续基于你当前仓库真实存在的单卡类名与签名落地

换句话说：

> 这一篇要“学上游语义”，但不能“抄上游 TP 接口”。

---

## 3. 先看当前仓库真正缺什么

### 3.1 `config.py` 的真实问题

当前 [config.py](/home/psx/nano_vllm_repro/nano_vll_repro/config.py:1) 只有最基础的路径、批处理和显存参数，问题不在“类不存在”，而在“字段语义散落在外面”：

1. 只有 `model_path` 和 `model` 别名，没有统一的 HF 配置桥接字段。
2. 后面 `ModelRunner`、`Qwen3Attention`、`CUDA Graph` 都会用到的 dtype / rope / head 相关信息，现在只能在各处手搓 `getattr(...)`。
3. 当前文件还没有 `torch_dtype`、`kv_torch_dtype`、`max_cudagraph_batch_size` 这类后续篇章会直接依赖的统一出口。

### 3.2 `rotary_embedding.py` 的真实问题

当前 [layers/rotary_embedding.py](/home/psx/nano_vllm_repro/nano_vll_repro/layers/rotary_embedding.py:140) 已经比旧文档说得更接近目标：

1. 它已经有 `rope_scaling` 形参。
2. 它已经会对“暂不支持的位置缩放配置”直接断言。

所以这一步真正要做的不是“重写整个 RoPE 文件”，而是：

1. 把模型 / 配置层传参方式和它对齐。
2. 把“当前只支持默认 RoPE”的边界写得更明确。

### 3.3 `models/qwen3.py` 的真实问题

当前 [models/qwen3.py](/home/psx/nano_vllm_repro/nano_vll_repro/models/qwen3.py:338) 最大的问题不是层顺序彻底错了，而是“接口边界还没收口”：

1. `Qwen3Attention` 现在使用的线性层签名，其实已经和当前本地 `QKVLinear / MergedLinear / RowLinear` 对齐；旧文档把它误写成“旧接口”，这是错的。
2. `Qwen3ForCausalLM.forward()` 当前直接返回 logits，不利于 Day4 的 `ModelRunner.run_model()` 和 Day6 的 CUDA Graph。
3. `compute_logits()` 当前缺失，导致“主干 hidden states”和“lm_head 投影”耦合在一起。
4. `from_pretrained()` 当前只负责创建结构并打印信息，这个方向本身没错，但文档没有把“结构创建”和“权重加载”的职责边界讲清楚。
5. 文件里还保留了大段已经注释掉的手写 attention fallback，这会让后续读者误判当前真实执行路径。

### 3.4 `tests/test_Day2.py` 的真实问题

当前 [tests/test_Day2.py](/home/psx/nano_vllm_repro/nano_vll_repro/tests/test_Day2.py:165) 已经暴露出两处真实接口漂移：

1. `test_qwen3_model()` 还假定 `model(input_ids)` 直接返回 logits。
2. `test_gqa()` 还在给 `Qwen3Attention` 传 `attention_mask=None`，但当前 [Qwen3Attention.forward](/home/psx/nano_vllm_repro/nano_vll_repro/models/qwen3.py:101) 根本不接受这个参数。

---

## 4. 本篇修改原则

### 4.1 单卡线性层接口继续沿用当前本地实现

这一篇统一使用当前仓库真实存在的 3 个类：

- `QKVLinear`
- `MergedLinear`
- `RowLinear`

不要在这里提前引入：

- `QKVParallelLinear`
- `MergedColumnParallelLinear`
- `RowParallelLinear`

原因不是“上游不好”，而是：

1. 你当前仓库的 `layers/linear.py` 还不是 TP 版本。
2. 如果这里先把模型代码和文档改到 TP 版签名，后面读者会发现 `models/qwen3.py` 和 `layers/linear.py` 当场脱节。

### 4.2 先把模型主干和 logits head 拆开

这一条是本篇最关键的接口收口：

- `Qwen3ForCausalLM.forward()` 只返回 hidden states
- `compute_logits(hidden_states)` 单独负责 lm head

这么做不是为了“更抽象”，而是因为后面：

1. Day4 的 `ModelRunner` 需要显式控制“前向 -> logits -> sampler”。
2. Day6 的 CUDA Graph 更适合只捕获主干 hidden states 路径。

### 4.3 语义向 HF / 上游靠拢，但实现仍按当前仓库落地

这一篇真正要对齐的是：

1. `q_norm / k_norm` 的顺序
2. `RoPE` 的传参边界
3. `packed_modules_mapping` 的含义
4. `from_pretrained()` 与 `utils.loader.load_model()` 的职责划分

而不是对齐：

1. 上游 TP 类名
2. 上游 worker 多进程结构
3. 上游并行 embedding / parallel lm head

---

## 5. 逐步修改

## 5.1 先收口 `config.py`，把 HF 语义桥接放到一个地方

修改位置：

- 文件：`nano_vll_repro/config.py`
- 操作：这里作为少数例外，允许整文件替换为下面这份完整实现

为什么这里允许整文件替换：

1. 当前文件很短，零散补 property 反而更难读。
2. 这份配置类后面会被 `ModelRunner`、`Qwen3Attention`、`bench.py` 和 `CUDA Graph` 共用，单点收口更稳。

完整替代代码如下。注意：代码块里的注释故意写得很密，因为这份文件本身就是“配置语义说明书”，不是只图跑通的占位类。

```python
import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass
class Config:
    """
    nano-vllm 教学仓库的核心配置对象。

    输入：
    1. 用户在 `LLM(...)` / `LLMEngine(...)` 初始化时传入的运行参数。
    2. 本地模型目录中的 Hugging Face 配置文件。

    输出：
    1. 供运行时直接使用的基础参数。
    2. 一组统一的 property，用来屏蔽 HF 配置字段名和本地调用点之间的差异。

    设计边界：
    - 这里不直接持有模型权重，也不做推理。
    - 这里做的事情是“把模型元信息整理成后续模块好用的形状”。
    """

    # ===== 模型路径 =====
    # 本地仓库当前统一使用 `model_path`，先不要为了“更像上游”改名成 `model`。
    model_path: str

    # ===== 连续批处理基础参数 =====
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096

    # ===== 显存相关参数 =====
    gpu_memory_utilization: float = 0.7

    # ===== 并行与图优化参数 =====
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    max_cudagraph_batch_size: int = 32

    # ===== dtype 配置 =====
    # `auto` 的意义是：由运行时根据设备能力选择默认 dtype。
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"

    # ===== 运行时自动填充字段 =====
    hf_config: AutoConfig | None = None
    eos: int = -1

    # ===== PagedAttention 参数 =====
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self) -> None:
        """
        dataclass 初始化后执行的参数校验与 HF 配置加载逻辑。

        这里故意只做“确定性、无副作用”的准备工作：
        - 检查路径是否存在
        - 读取 HF config
        - 对基础数值约束做断言

        不在这里做的事情：
        - 不初始化 CUDA
        - 不加载模型权重
        - 不分配 KV Cache
        """

        # 模型目录必须存在；否则后面所有路径都会继续报连锁错误。
        assert os.path.isdir(self.model_path), f"模型路径不存在：{self.model_path}"

        # 你当前仓库的 Triton / FlashAttention 路径默认按 256 block size 在写。
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数"

        # 这里仍然保留单机教学版范围，不提前放开到任意 world size。
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size 必须在 1 到 8 之间"

        # 加载 HF 配置，这是后面所有 property 的统一来源。
        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        # 运行时最大上下文不能超过模型本身支持的最大位置长度。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 单批 token 上限至少要能覆盖一条最大长度序列。
        assert self.max_num_batched_tokens >= self.max_model_len, (
            "max_num_batched_tokens 必须 >= max_model_len"
        )

    @property
    def model(self) -> str:
        """
        给上游风格保留一个只读别名。

        这样做的原因不是鼓励混用，而是为了让后续参考上游代码时，
        你能少做一次“字段名翻译”。
        """
        return self.model_path

    @property
    def hidden_size(self) -> int:
        # 这一层 property 的意义不是省几个字，而是让调用点不要直接碰 hf_config。
        return self.hf_config.hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.hf_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        # Qwen3Config 通常会给这个字段；如果没有，再回退到全量 attention heads。
        return getattr(self.hf_config, "num_key_value_heads", self.num_attention_heads)

    @property
    def head_dim(self) -> int:
        """
        单头维度。

        注意：
        - 先信任 HF 配置里显式给出的 `head_dim`
        - 只有字段不存在时，才用 hidden_size // num_attention_heads 回退
        """
        return getattr(self.hf_config, "head_dim", self.hidden_size // self.num_attention_heads)

    @property
    def attention_bias(self) -> bool:
        # 当前仓库统一从这里取 attention bias，而不是在模型里散写 getattr。
        return getattr(self.hf_config, "attention_bias", False)

    @property
    def hidden_act(self) -> str:
        return getattr(self.hf_config, "hidden_act", "silu")

    @property
    def tie_word_embeddings(self) -> bool:
        return getattr(self.hf_config, "tie_word_embeddings", False)

    @property
    def rope_parameters(self):
        """
        HF Qwen3 新式 RoPE 配置入口。

        当前仓库暂时不会在这里完整支持 longrope / yarn / dynamic_ntk，
        但要把原始配置保留下来，供模型层和文档层做边界判断。
        """
        return getattr(self.hf_config, "rope_parameters", None)

    @property
    def rope_theta(self) -> float:
        """
        当前仓库实际使用的 rope theta。

        兼容顺序：
        1. 如果 `rope_parameters` 是 dict，优先取其中显式给出的 `rope_theta`
        2. 若没有，再看旧式 `hf_config.rope_theta`
        3. 最后才回退到 Qwen3 常见默认值
        """
        rope_parameters = self.rope_parameters
        if isinstance(rope_parameters, dict):
            return rope_parameters.get("rope_theta", rope_parameters.get("base", 1_000_000.0))
        return getattr(self.hf_config, "rope_theta", 1_000_000.0)

    @property
    def rope_scaling(self):
        """
        给 `get_rope(...)` 保留一个统一入口。

        当前仓库如果不支持具体缩放策略，应该在 RoPE 工厂函数里显式报错，
        而不是在这里静默吞掉配置。
        """
        rope_parameters = self.rope_parameters
        if isinstance(rope_parameters, dict):
            return rope_parameters.get("rope_scaling", None)
        return getattr(self.hf_config, "rope_scaling", None)

    @property
    def layer_types(self):
        # 这个字段后面做 sliding window / layer pattern 时可能会用到，先统一暴露。
        return getattr(self.hf_config, "layer_types", None)

    @property
    def sliding_window(self):
        return getattr(self.hf_config, "sliding_window", None)

    @property
    def use_sliding_window(self) -> bool:
        return bool(getattr(self.hf_config, "use_sliding_window", False))

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        主干计算与模型权重默认使用的 dtype。

        这里故意支持 `auto`：
        - CUDA + BF16 能力存在时优先 BF16
        - 否则在 CUDA 下回退 FP16
        - CPU 路径回退 FP32
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
        KV Cache 使用的 dtype。

        之所以单独拆开，是为了允许：
        - 模型主干走 BF16
        - KV Cache 暂时继续走 FP16
        """
        if self.kv_cache_dtype == "bfloat16":
            return torch.bfloat16
        if self.kv_cache_dtype == "float16":
            return torch.float16
        if self.kv_cache_dtype == "float32":
            return torch.float32

        # `auto` 时仍然优先使用较省显存的 float16。
        if self.kv_cache_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32

        raise ValueError(f"不支持的 kv_cache_dtype: {self.kv_cache_dtype}")
```

这一段真正要学到的是：

1. `Config` 不是“存参数的袋子”，而是“HF 配置桥接层”。
2. 调用点越少直接碰 `hf_config`，后续越容易收口。

---

## 5.2 只微调 `get_rope()` 的边界，不重写 `RotaryEmbedding`

修改位置：

- 文件：`nano_vll_repro/layers/rotary_embedding.py`
- 操作：只替换底部 `get_rope(...)` 工厂函数

当前文件主体实现已经够用。真正要修的是“参数入口写清楚”，而不是重写旋转数学本身。

完整替代函数如下：

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
    获取当前仓库使用的 RoPE 实例。

    输入：
    1. head_size: 单头维度。
    2. rotary_dim: 参与旋转的位置编码维度；当前仓库要求它等于 head_size。
    3. max_position: 最大位置长度。
    4. base: 当前实际使用的 rope theta。
    5. rope_scaling: 预留给 HF 配置的兼容入口。

    输出：
    - 一个可缓存复用的 `RotaryEmbedding` 实例。

    当前边界必须写死：
    - 这份教学仓库当前只支持默认 RoPE。
    - 如果传入了 yarn / longrope / dynamic_ntk 等扩展配置，
      不应该静默忽略，而应该立即报错。
    """

    # 这里故意显式拒绝扩展缩放配置。
    # 原因不是“以后永远不支持”，而是当前仓库还没有实现这些数学语义。
    if rope_scaling is not None:
        raise AssertionError("当前教学仓库只支持默认 RoPE，暂不支持 rope_scaling / rope_parameters 扩展")

    # 这里保留当前实现最简单、最明确的构造路径。
    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
    )
```

这里最重要的不是函数体，而是“不要制造假兼容”：

1. 接口上可以先留入口。
2. 语义上不支持时必须显式报错。

---

## 5.3 重写 `Qwen3Attention`，但继续使用当前本地单卡线性层

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 操作：把 `Qwen3Attention` 整个类替换为下面这份完整实现

这一步最重要的纠偏只有一条：

> 你要学的是上游和 HF 的“层内顺序”，不是上游 TP 版本的“线性层签名”。

完整替代代码如下：

```python
class Qwen3Attention(nn.Module):
    """
    Qwen3 注意力层。

    当前阶段的实现边界：
    1. 仍然是单卡语义。
    2. 仍然使用本地 `QKVLinear / RowLinear`。
    3. 通过 `layers.attention.Attention` + `Context` 处理 prefill / decode 分流。

    这意味着：
    - 本层负责投影、Q/K norm、RoPE、调用 Attention、输出投影。
    - 本层不再内嵌一套注释掉的手写 attention fallback。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        max_position: int = 4096 * 32,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 1_000_000.0,
        rope_scaling: dict | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()

        # ===== 头数与维度语义 =====
        # 当前阶段先保持单卡，所以 total heads 和 local heads 先不拆。
        # Day5 做 TP 时，才会把这两个概念显式拆开。
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # head_dim 必须优先信任配置，而不是永远自己算。
        self.head_dim = head_dim or hidden_size // self.num_heads

        # Q / K / V 拼接后的切分尺寸。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # attention scale 始终按单头维度计算。
        self.scaling = self.head_dim ** -0.5

        # ===== 线性投影层 =====
        # 注意：这里继续使用当前仓库真实存在的单卡 QKVLinear。
        # 不要在本篇里提前切到上游 TP 签名。
        self.qkv_proj = QKVLinear(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bias=qkv_bias,
        )

        # o_proj 也继续使用当前本地 RowLinear。
        self.o_proj = RowLinear(
            input_size=self.q_size,
            output_size=hidden_size,
            bias=False,
        )

        # ===== RoPE =====
        # 当前仓库如果收到 rope_scaling，会在 get_rope 里显式报错；
        # 这样接口兼容和实现边界都清楚。
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # ===== Q / K norm =====
        # Qwen3 的 q_norm / k_norm 是作用在 head_dim 上，而不是 hidden_size 上。
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # ===== PagedAttention 外壳 =====
        # 真正的 prefill / decode 分流由 Attention + Context 完成。
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
        1. positions: `[num_tokens]`
        2. hidden_states: `[num_tokens, hidden_size]`

        输出：
        - `[num_tokens, hidden_size]`

        执行顺序必须固定：
        1. QKV 融合投影
        2. 按本层定义的 q_size / kv_size 切分
        3. reshape 成多头张量
        4. 在 head_dim 上做 q_norm / k_norm
        5. 对 q / k 应用 RoPE
        6. 调用底层 Attention
        7. 输出投影
        """

        num_tokens = hidden_states.shape[0]

        # 先做融合投影。
        qkv = self.qkv_proj(hidden_states)

        # 再按当前层显式维护的尺寸切分。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 只有先 reshape 成 `[num_tokens, num_heads, head_dim]`，
        # 后面的 q_norm / k_norm 才是真正按单头维做归一化。
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # Qwen3 的 q_norm / k_norm 在 RoPE 之前做。
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 位置编码只作用在 q / k 上，不作用在 v 上。
        q, k = self.rotary_emb(positions, q, k)

        # 这里不再保留另一套注释掉的手写 attention 路径。
        # 当前仓库的真实执行语义应该只有这一条。
        attn_output = self.attn(q, k, v)

        # 最后把多头输出 flatten 回 hidden_size 再做 o_proj。
        output = self.o_proj(attn_output.reshape(num_tokens, -1))
        return output
```

这段代码里最值得记住的是：

1. `q_norm / k_norm` 的位置比“是否长得像上游”更重要。
2. 本篇不改 TP 签名，本篇只改层内语义与对外接口边界。

---

## 5.4 `Qwen3MLP` 继续沿用当前本地 `MergedLinear` 语义

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 操作：只替换 `Qwen3MLP` 类

这一段专门纠正旧文档里的一个错误方向：

> 当前本地 `MergedLinear` 的真实签名仍然是 `output_size + num_shards`，不是上游 TP 版的 `output_sizes=[...]`。

完整替代代码如下：

```python
class Qwen3MLP(nn.Module):
    """
    Qwen3 前馈层。

    当前阶段继续使用本地单卡 `MergedLinear`：
    - gate_proj 和 up_proj 沿输出维拼接
    - `num_shards=2` 明确表示拼了两个等尺寸分片
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        # gate_proj / up_proj 融合成一个输出维为 2 * intermediate_size 的线性层。
        self.gate_up_proj = MergedLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            num_shards=2,
            bias=False,
        )

        # down_proj 继续使用当前仓库的单卡 RowLinear。
        self.down_proj = RowLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
        )

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - `[num_tokens, hidden_size]`

        输出：
        - `[num_tokens, hidden_size]`

        执行顺序：
        1. gate_up_proj 得到 `[num_tokens, 2 * intermediate_size]`
        2. `SiluAndMul` 做 SwiGLU
        3. down_proj 回到 hidden_size
        """

        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

---

## 5.5 重写 `Qwen3ForCausalLM` 的边界：主干输出 hidden states，lm_head 单独算 logits

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 操作：把 `Qwen3ForCausalLM` 整个类替换为下面这份完整实现

这一步是本篇最重要的接口收口。你后面如果还想做：

1. `ModelRunner.run_model()`
2. CUDA Graph
3. benchmark 时单独统计 logits 阶段

都必须先把这层边界拆开。

完整替代代码如下：

```python
class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型外壳。

    当前职责只有 3 件事：
    1. 持有主干 `Qwen3Model`
    2. 持有 `lm_head`
    3. 暴露 packed 模块映射，供 `utils.loader.load_model()` 使用

    这里故意不在 `forward()` 里直接返回 logits，
    因为后面的运行时更适合显式控制“主干 -> logits -> sampler”。
    """

    # 这些映射告诉 loader：
    # HF 的分离权重，应该如何写入当前仓库的融合参数。
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

        # 词嵌入和 lm_head 是否共享权重，完全交给配置决定。
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        输入：
        1. input_ids
        2. 可选 positions

        输出：
        - 最后一层 hidden states，而不是 logits

        这样做的原因：
        - `ModelRunner` 可以明确控制 logits 计算时机
        - Day6 的 CUDA Graph 更适合只 capture 主干路径
        """

        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - 主干输出的 hidden states

        输出：
        - vocab 维 logits

        这里单独拆函数，是为了让运行时边界更清楚。
        """

        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        输入：
        - HF 模型目录

        输出：
        - 只创建了结构、尚未加载 safetensors 权重的模型实例

        这里故意不直接加载权重，原因是：
        - 结构创建属于模型类职责
        - 权重写入属于 `utils.loader.load_model()` 职责
        - 这两个阶段拆开后，测试和调试都更清楚
        """

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)

        # 打印信息不是必须，但对教学仓库有价值；
        # 它可以帮助读者确认当前结构创建的是哪一份配置。
        print("[Info] 模型结构已创建，权重尚未加载")
        print(f"[Info] hidden_size: {config.hidden_size}")
        print(f"[Info] num_layers: {config.num_hidden_layers}")
        print(f"[Info] num_heads: {config.num_attention_heads}")
        print(f"[Info] num_kv_heads: {config.num_key_value_heads}")
        return model
```

这里一定要理解：

1. 这不是“多写一个函数”，而是“把运行时边界前置收口”。
2. 结构创建和权重加载分开，是为了让 `loader.py` 的 packed weight 协议保持清晰。

---

## 5.6 同步修 `tests/test_Day2.py`

本篇至少要改 3 处测试。

### 改动 1：`test_qwen3_model()`

修改位置：

- 文件：`nano_vll_repro/tests/test_Day2.py`
- 操作：把 `test_qwen3_model()` 整个函数替换为下面这份完整实现

```python
@torch.inference_mode()
def test_qwen3_model():
    """测试 Qwen3 模型的“主干 -> logits”新边界。"""

    print("=" * 50)
    print("测试 Qwen3 模型")
    print("=" * 50)

    from dataclasses import dataclass
    from models.qwen3 import Qwen3ForCausalLM

    @dataclass
    class TestConfig:
        # ===== 最小可运行配置 =====
        vocab_size: int = 1000
        hidden_size: int = 128
        num_hidden_layers: int = 2
        num_attention_heads: int = 4
        num_key_value_heads: int = 2
        intermediate_size: int = 256
        max_position_embeddings: int = 512
        rms_norm_eps: float = 1e-6
        attention_bias: bool = False
        rope_theta: float = 10000.0
        tie_word_embeddings: bool = False

    config = TestConfig()
    model = Qwen3ForCausalLM(config)
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 前向传播 =====
    num_tokens = 10
    input_ids = torch.randint(0, config.vocab_size, (num_tokens,))
    positions = torch.arange(num_tokens)

    # forward 现在返回的是 hidden states。
    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)

    print(f"输入 token 数: {num_tokens}")
    print(f"hidden_states 形状: {hidden_states.shape}")
    print(f"logits 形状: {logits.shape}")

    assert hidden_states.shape == (num_tokens, config.hidden_size)
    assert logits.shape == (num_tokens, config.vocab_size)

    # ===== 简单自回归模拟 =====
    # 这里仍然保留一个最小生成循环，
    # 但显式走“先主干、再 logits”的新接口。
    print("\n模拟自回归生成:")
    generated = input_ids.tolist()
    for _ in range(3):
        cur_input_ids = torch.tensor(generated)
        cur_positions = torch.arange(len(generated))
        cur_hidden = model(cur_input_ids, cur_positions)
        cur_logits = model.compute_logits(cur_hidden)
        next_token = cur_logits[-1].argmax().item()
        generated.append(next_token)
        print(f"  生成 token: {next_token}")

    print("✅ Qwen3 模型测试通过!\n")
```

### 改动 2：`test_gqa()`

修改位置：

- 文件：`nano_vll_repro/tests/test_Day2.py`
- 操作：只替换 `attn(...)` 调用和周边说明

完整替代代码如下：

```python
@torch.inference_mode()
def test_gqa():
    """测试当前单卡语义下的 Grouped Query Attention。"""

    print("=" * 50)
    print("测试 GQA (Grouped Query Attention)")
    print("=" * 50)

    from models.qwen3 import Qwen3Attention

    hidden_size = 128
    num_heads = 8
    num_kv_heads = 2

    attn = Qwen3Attention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qkv_bias=False,
    )
    attn.eval()

    num_tokens = 5
    hidden_states = torch.randn(num_tokens, hidden_size)
    positions = torch.arange(num_tokens)

    # 当前仓库的 Qwen3Attention.forward 不接 attention_mask。
    # prefill / decode 的 mask 分流由 Attention + Context 处理。
    output = attn(positions, hidden_states)

    print(f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}")
    print(f"每个 KV head 被 {num_heads // num_kv_heads} 个 Q head 共享")
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")

    assert output.shape == hidden_states.shape
    print("✅ GQA 测试通过!\n")
```

### 改动 3：建议删掉模型文件里的大段注释式 fallback

这个不是测试代码修改，但这里要明确写给你：

1. `models/qwen3.py` 里那一大段注释掉的手写 attention fallback，建议在本篇直接删掉。
2. 原因不是洁癖，而是它会让读者误判当前真实执行路径。

本篇之后，模型文件里应该只保留一条清晰主线：

- 投影
- q/k norm
- RoPE
- `self.attn(...)`
- 输出投影

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

如果你只想快速确认模型接口边界，也可以手动执行：

```bash
python - <<'PY'
import torch
from models.qwen3 import Qwen3ForCausalLM

model = Qwen3ForCausalLM.from_pretrained("models/Qwen3-0.6B")
input_ids = torch.tensor([1, 2, 3, 4])
positions = torch.arange(4)

hidden_states = model(input_ids, positions)
logits = model.compute_logits(hidden_states)

print(hidden_states.shape, logits.shape)
PY
```

---

## 7. 常见错误

### 7.1 在本篇提前改成上游 TP 线性层签名

后果：

- `models/qwen3.py` 和当前本地 `layers/linear.py` 当场对不上
- 你会误以为是模型逻辑错了，实际上是接口被提前切了

### 7.2 继续让 `forward()` 直接返回 logits

后果：

- Day4 的 `ModelRunner` 很难把“主干前向”和“lm_head 投影”拆开
- Day6 的 graph capture 边界会更乱

### 7.3 表面接入 `rope_scaling`，实际偷偷忽略

后果：

- 文档会给出“已经兼容 HF 扩展 RoPE”的假象
- 真正排查长上下文行为时会非常痛苦

### 7.4 测试仍然调用旧接口

后果：

- 你会把测试失败误判成实现失败
- 实际上只是测试脚本仍停在旧边界上

---

## 8. 本篇真正学到的东西

这一篇真正要掌握的是下面 4 件事：

1. 如何把 HF 配置语义统一收口到 `Config`。
2. 为什么当前阶段要继续使用本地单卡 `QKVLinear / MergedLinear / RowLinear`。
3. 为什么模型主干和 logits head 应该尽早拆开。
4. 为什么“参考上游”不等于“照抄上游当前所有接口”。

完成后进入下一篇：

- [03-补全Sampler与SamplingParams.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/03-补全Sampler与SamplingParams.md)

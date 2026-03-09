# 02. 补齐 Qwen3 模型主干与权重映射

## 1. 学习目标

这一篇的目标是把“模型骨架”升级成“和 Qwen3 主流实现一致的可加载主干”。

你要完成的核心事情有三件：

1. 让 `config.py` 真正承接 Hugging Face `Qwen3Config` 里的关键字段
2. 让 `models/qwen3.py` 的结构顺序对齐 HF `Qwen3`
3. 让 `layers/rotary_embedding.py` 至少能兼容 `rope_parameters`

这一篇完成后，仓库应该具备：

- 正确的 `Qwen3Attention / Qwen3MLP / Qwen3DecoderLayer`
- 正确的 residual / norm 顺序
- 与 HF 命名兼容的权重映射入口

## 2. 先修知识

## 2.1 为什么 Qwen3 容易“看起来写对了，实际上写错了”

Qwen3 不是“普通 Transformer + RoPE”这么简单。

最容易写错的地方有 4 个：

1. `q_norm / k_norm` 的位置
2. `head_dim` 的来源
3. `num_key_value_heads` 对 GQA 的影响
4. decoder layer 的 residual 顺序

如果你把这 4 个点写错，即使代码能跑，也可能：

- 权重加载失败
- logits 明显异常
- FlashAttention 输入形状不匹配
- 多卡切分后 head 维度错位

## 2.2 HF `Qwen3Attention` 的正确顺序

根据 Hugging Face `transformers` 的 `Qwen3Attention`，顺序应该是：

```text
q_proj / k_proj / v_proj
-> reshape 成多头
-> q_norm / k_norm
-> RoPE
-> attention
-> o_proj
```

注意：

- `q_norm / k_norm` 是做在 `head_dim` 上，不是在整个 `hidden_size` 上
- RoPE 要在 norm 之后
- `num_key_value_heads` 不等于 `num_attention_heads` 时，就是 GQA

## 2.3 decoder layer 的正确顺序

Qwen3 的 decoder layer 是标准的 **pre-norm + residual**：

```text
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states
```

这里最容易犯的错，是把你当前仓库里 `RMSNorm` 的 fused residual 写法直接套进来，结果把语义写歪。

我的建议是：

- 先按 HF 语义写正确
- 后面真要做 fused norm，再作为优化层补

## 3. 本仓库当前缺口

当前仓库在这一块有 5 个问题：

1. `config.py` 只承接了很少的 HF 配置字段
2. `models/qwen3.py` 虽然长得像 Qwen3，但 residual 语义还没完全贴齐
3. `from_pretrained()` 里的行为还不够干净
4. `layers/rotary_embedding.py` 只支持旧式 `rope_theta`
5. 还没有把“HF 命名 -> 仓库内部融合命名”的边界说清楚

## 4. 最终应修改的文件

- `config.py`
- `layers/rotary_embedding.py`
- `models/qwen3.py`

## 5. 完整代码

### 5.1 替换 `config.py`

```python
import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass
class Config:
    model_path: str

    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096

    gpu_memory_utilization: float = 0.7
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    max_cudagraph_batch_size: int = 32

    hf_config: AutoConfig | None = None
    eos: int = -1

    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self) -> None:
        assert os.path.isdir(self.model_path), f"模型路径不存在：{self.model_path}"
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数"
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size 必须在 1 到 8 之间"

        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.max_model_len = min(
            self.max_model_len,
            getattr(self.hf_config, "max_position_embeddings", self.max_model_len),
        )
        assert self.max_num_batched_tokens >= self.max_model_len

    @property
    def model(self) -> str:
        return self.model_path

    @property
    def hidden_size(self) -> int:
        return self.hf_config.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.hf_config.intermediate_size

    @property
    def num_hidden_layers(self) -> int:
        return self.hf_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.hf_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.hf_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return getattr(self.hf_config, "head_dim", self.hidden_size // self.num_attention_heads)

    @property
    def rms_norm_eps(self) -> float:
        return self.hf_config.rms_norm_eps

    @property
    def max_position_embeddings(self) -> int:
        return self.hf_config.max_position_embeddings

    @property
    def rope_parameters(self):
        return getattr(self.hf_config, "rope_parameters", None)

    @property
    def rope_theta(self) -> float:
        rope_parameters = self.rope_parameters
        if isinstance(rope_parameters, dict):
            return rope_parameters.get("rope_theta", rope_parameters.get("base", 1000000.0))
        return getattr(self.hf_config, "rope_theta", 1000000.0)

    @property
    def attention_bias(self) -> bool:
        return getattr(self.hf_config, "attention_bias", False)

    @property
    def attention_dropout(self) -> float:
        return getattr(self.hf_config, "attention_dropout", 0.0)

    @property
    def hidden_act(self) -> str:
        return getattr(self.hf_config, "hidden_act", "silu")

    @property
    def tie_word_embeddings(self) -> bool:
        return getattr(self.hf_config, "tie_word_embeddings", False)

    @property
    def layer_types(self):
        return getattr(self.hf_config, "layer_types", None)

    @property
    def use_sliding_window(self) -> bool:
        return getattr(self.hf_config, "use_sliding_window", False)

    @property
    def sliding_window(self):
        return getattr(self.hf_config, "sliding_window", None)

    @property
    def torch_dtype(self) -> torch.dtype:
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
        if self.kv_cache_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if self.kv_cache_dtype == "bfloat16":
            return torch.bfloat16
        if self.kv_cache_dtype == "float16":
            return torch.float16
        return torch.float32
```

### 5.2 替换 `layers/rotary_embedding.py`

```python
from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        assert rotary_dim == head_size, "当前实现要求 rotary_dim == head_size"
        self.head_size = head_size
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_parameters: dict | None = None,
) -> RotaryEmbedding:
    if rope_parameters is not None:
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise NotImplementedError(
                f"当前教案只实现 default RoPE，暂不支持 rope_type={rope_type}"
            )
        base = rope_parameters.get("rope_theta", rope_parameters.get("base", base))

    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
    )
```

### 5.3 替换 `models/qwen3.py`

```python
import torch
from torch import nn
from transformers import AutoConfig

from layers.activation import SiluAndMul
from layers.attention import Attention
from layers.layernorm import RMSNorm
from layers.linear import QKVLinear, MergedLinear, RowLinear, divide, get_tp_world_size
from layers.rotary_embedding import get_rope


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int = 0) -> None:
        super().__init__()
        tp_size = get_tp_world_size()

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVLinear(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = RowLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", rope_parameters.get("base", rope_theta))

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_parameters=rope_parameters,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            layer_idx=layer_idx,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output.reshape(num_tokens, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_act = getattr(config, "hidden_act", "silu")
        assert hidden_act in {"silu", "swiglu"}, f"暂不支持 hidden_act={hidden_act}"

        self.gate_up_proj = MergedLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            num_shards=2,
            bias=False,
        )
        self.down_proj = RowLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(cls, model_path: str):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return cls(config)
```

## 6. 手敲顺序

建议顺序：

1. 先重写 `config.py`
2. 再重写 `layers/rotary_embedding.py`
3. 最后重写 `models/qwen3.py`

为什么这样排：

- `models/qwen3.py` 会同时依赖前两个文件
- 如果你先写模型文件，手敲过程中很容易来回改配置与 RoPE

## 7. 最小验收方法

### 7.1 语法校验

```bash
python -m py_compile config.py layers/rotary_embedding.py models/qwen3.py
```

### 7.2 Day2 测试入口

如果你的环境有 `torch`：

```bash
python tests/test_Day2.py
```

### 7.3 手动自测问题

如果你能自己回答下面 4 个问题，就说明这一篇真正学会了：

1. 为什么 `q_norm / k_norm` 要放在 RoPE 前面
2. 为什么 `head_dim` 不能总写死成 `hidden_size // num_attention_heads`
3. 为什么 `num_key_value_heads` 会直接影响 QKV 的 split 形状
4. 为什么 decoder layer 里不要急着上 fused residual 写法

下一篇进入：

- [03-补全Sampler与SamplingParams.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/03-补全Sampler与SamplingParams.md)

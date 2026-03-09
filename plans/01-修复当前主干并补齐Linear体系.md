# 01. 修复当前主干并补齐 Linear 体系

## 1. 学习目标

这一篇做三件事：

1. 修掉当前 `HEAD` 最明显的断裂点：`layers/linear.py`
2. 建立单卡与 Tensor Parallelism 共用的一套线性层抽象
3. 补齐权重加载器依赖的 `weight_loader` 机制

这一篇完成后，你的仓库至少应当满足：

- `layers/linear.py` 可以正常导入
- `models/qwen3.py` 依赖的 `QKVLinear`、`MergedLinear`、`RowLinear` 都存在
- `utils/loader.py` 可以正确把 Hugging Face 的分离权重映射到融合层

## 2. 先修知识

### 2.1 为什么要做 QKV 融合

Hugging Face 里的 Qwen3 注意力层通常是：

- `q_proj`
- `k_proj`
- `v_proj`

但推理框架里更喜欢把它们合成一个大线性层，原因是：

1. 只做一次 GEMM，减少 kernel launch
2. 更容易做张量并行切分
3. 权重加载时可以按分片写入目标大矩阵

### 2.2 为什么 MLP 也要融合

Qwen3 的 MLP 结构本质上是：

```text
down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

所以 `gate_proj + up_proj` 很适合融合成一个 `MergedLinear`，先一次矩阵乘法拿到 `2 * intermediate_size`，再拆成两半做 SwiGLU。

### 2.3 Column Parallel 和 Row Parallel 的区别

这是 Day6 TP 的基础，所以这一篇就要把抽象搭好。

#### Column Parallel

按输出维度切分权重：

```text
[out, in] -> 每张卡持有 [out / tp, in]
```

特点：

- 输入是完整的
- 每张卡输出自己的那一段
- forward 后通常不需要立刻通信

适合：

- `qkv_proj`
- `gate_up_proj`

#### Row Parallel

按输入维度切分权重：

```text
[out, in] -> 每张卡持有 [out, in / tp]
```

特点：

- 输入本身就是分片的
- 每张卡先算出部分贡献
- forward 后需要 `all_reduce`

适合：

- `o_proj`
- `down_proj`

## 3. 本仓库当前缺口

当前仓库里最严重的问题不是“TP 没做完”，而是 `layers/linear.py` 已经断尾，导致整个主干都不稳定。

你已经看到这些现象：

- `class RowParallelLinear(LinearBase):` 后面没有完整实现
- `models/qwen3.py` 在 import `QKVLinear`、`MergedLinear`、`RowLinear`
- 当前 `layers/linear.py` 并没有把这些名字完整导出

所以这一步不是优化，而是**修主干**。

## 4. 最终应修改的文件

这一篇只动 3 个文件：

- `layers/linear.py`
- `utils/loader.py`
- `tests/test_Day4.py`

## 5. 完整代码

### 5.1 替换 `layers/linear.py`

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator: int, denominator: int) -> int:
    assert denominator > 0, "denominator must be > 0"
    assert numerator % denominator == 0, f"{numerator} 不能被 {denominator} 整除"
    return numerator // denominator


def get_tp_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_tp_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.tp_dim = tp_dim

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (self.weight.size(1) ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__(input_size, output_size, bias=bias, tp_dim=None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
        default_weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        self.output_size = output_size
        self.output_size_per_partition = divide(output_size, get_tp_world_size())
        super().__init__(input_size, self.output_size_per_partition, bias=bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
        shard_size = param.data.size(0)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(0, start_idx, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        target = param.data.narrow(0, shard_offset, shard_size)
        source = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        target.copy_(source.to(device=param.device, dtype=param.dtype))


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.num_heads = divide(self.total_num_heads, get_tp_world_size())
        self.num_kv_heads = divide(self.total_num_kv_heads, get_tp_world_size())

        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias=bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        assert loaded_shard_id in {"q", "k", "v"}

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size

        target = param.data.narrow(0, shard_offset, shard_size)
        source = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        target.copy_(source.to(device=param.device, dtype=param.dtype))


class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        self.input_size = input_size
        self.input_size_per_partition = divide(input_size, get_tp_world_size())
        super().__init__(self.input_size_per_partition, output_size, bias=bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
        if param.ndim == 1:
            default_weight_loader(param, loaded_weight)
            return

        shard_size = param.data.size(1)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(1, start_idx, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, None)
        if self.tp_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(output)
        if self.bias is not None:
            output = output + self.bias
        return output


class QKVLinear(QKVParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=bias,
        )


class MergedLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_shards: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_sizes=[output_size] * num_shards,
            bias=bias,
        )


class RowLinear(RowParallelLinear):
    pass
```

### 5.2 替换 `utils/loader.py`

```python
import os
from glob import glob

import torch
from torch import nn

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from layers.linear import default_weight_loader


def load_model(model: nn.Module, model_path: str) -> None:
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors is required for loading weights")

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    loaded_count = 0
    skipped_count = 0

    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                is_packed = False

                for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
                    if original_name not in weight_name:
                        continue

                    param_name = weight_name.replace(original_name, packed_name)
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        skipped_count += 1
                        is_packed = True
                        break

                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is None:
                        raise RuntimeError(f"{param_name} 没有绑定 weight_loader")

                    weight_loader(param, loaded_weight, shard_id)
                    loaded_count += 1
                    is_packed = True
                    break

                if is_packed:
                    continue

                try:
                    param = model.get_parameter(weight_name)
                except AttributeError:
                    skipped_count += 1
                    continue

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_count += 1

    print(f"[Loader] 完成：加载 {loaded_count} 个权重，跳过 {skipped_count} 个")


def load_model_weights(model: nn.Module, model_path: str) -> None:
    load_model(model, model_path)
```

### 5.3 替换 `tests/test_Day4.py`

```python
"""Day 4 线性层与采样器测试"""

import sys
sys.path.insert(0, '.')

import torch


@torch.inference_mode()
def test_linear_layers():
    from layers.linear import QKVLinear, MergedLinear, RowLinear

    qkv = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64)
    q_weight = torch.randn(8 * 64, 512)
    k_weight = torch.randn(2 * 64, 512)
    v_weight = torch.randn(2 * 64, 512)

    qkv.weight.weight_loader(qkv.weight, q_weight, "q")
    qkv.weight.weight_loader(qkv.weight, k_weight, "k")
    qkv.weight.weight_loader(qkv.weight, v_weight, "v")

    assert qkv.weight.shape == (768, 512)
    assert torch.allclose(qkv.weight.data[:512], q_weight)
    assert torch.allclose(qkv.weight.data[512:640], k_weight)
    assert torch.allclose(qkv.weight.data[640:], v_weight)

    merged = MergedLinear(512, 1024, num_shards=2)
    gate = torch.randn(1024, 512)
    up = torch.randn(1024, 512)
    merged.weight.weight_loader(merged.weight, gate, 0)
    merged.weight.weight_loader(merged.weight, up, 1)

    assert merged.weight.shape == (2048, 512)
    assert torch.allclose(merged.weight.data[:1024], gate)
    assert torch.allclose(merged.weight.data[1024:], up)

    row = RowLinear(512, 256, bias=False)
    hidden = torch.randn(3, 512)
    output = row(hidden)
    assert output.shape == (3, 256)

    print("✅ Linear 层测试通过")


@torch.inference_mode()
def test_sampler():
    from layers.sampler import Sampler

    sampler = Sampler()
    logits = torch.randn(4, 1000)
    temps = torch.tensor([0.0, 0.5, 1.0, 2.0])
    top_ks = torch.tensor([0, 0, 0, 0])
    top_ps = torch.tensor([1.0, 1.0, 1.0, 1.0])

    tokens = sampler(logits, temps, top_ks, top_ps)
    assert tokens.shape == (4,)
    assert tokens[0] == logits[0].argmax()

    print("✅ Sampler 测试通过")


@torch.inference_mode()
def test_sequence_attributes():
    from engine.sequence import Sequence
    from sampling_params import SamplingParams

    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0.7))
    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.last_token == 5
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.prompt_token_ids == [1, 2, 3, 4, 5]
    assert seq.completion_token_ids == []

    seq.append_token(6)
    assert seq.token_ids == [1, 2, 3, 4, 5, 6]
    assert seq.last_token == 6
    assert seq.num_tokens == 6
    assert seq.num_completion_tokens == 1

    print("✅ Sequence 属性测试通过")


if __name__ == "__main__":
    test_linear_layers()
    test_sampler()
    test_sequence_attributes()
    print("\n🎉 Day 4 所有测试通过！")
```

## 6. 手敲顺序

请严格按下面顺序来：

1. 先完整重写 `layers/linear.py`
2. 再重写 `utils/loader.py`
3. 最后重写 `tests/test_Day4.py`

不要先改测试。因为你现在的主干断裂点就在 `layers/linear.py`。

## 7. 最小验收方法

### 7.1 先做语法验收

```bash
python -m py_compile layers/linear.py utils/loader.py
```

预期结果：

- 没有任何报错

### 7.2 再做 Day4 入口验收

如果你的环境里已经安装了 `torch`：

```bash
python tests/test_Day4.py
```

预期结果：

- 能成功 import `QKVLinear`、`MergedLinear`、`RowLinear`
- 线性层测试通过

## 8. 这一篇完成后你学到了什么

如果你能不看文档，自己回答下面 3 个问题，就说明这一篇真的学会了：

1. 为什么 `q_proj/k_proj/v_proj` 适合融合成一个 `QKVLinear`
2. 为什么 `o_proj` 和 `down_proj` 更适合 `RowParallelLinear`
3. 为什么权重加载器必须绑定到参数对象上，而不是单独写一个“外部转换脚本”

下一篇进入：

- [02-补齐Qwen3模型主干与权重映射.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/02-补齐Qwen3模型主干与权重映射.md)


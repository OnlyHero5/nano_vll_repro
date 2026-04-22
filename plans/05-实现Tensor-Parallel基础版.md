# 05. 实现 Tensor Parallel 基础版

## 1. 本篇目标

这一篇的目标不是把仓库一下子变成上游那种完整多进程系统。

这一篇只做一件事：

> 在不打碎当前公开接口的前提下，把单卡线性层和模型主干升级成教学版 SPMD Tensor Parallel。

这里的“教学版 SPMD”有两个关键词：

1. **SPMD**：每个 rank 跑同一套 Python 主循环。
2. **教学版**：先把张量切分规则和运行时边界讲清楚，不上 worker 池，不上 RPC。

本篇完成后，你应该得到下面 5 个结果：

1. `layers/linear.py` 具备 TP helper 和 TP-aware 线性层。
2. 公开名字 `QKVLinear / MergedLinear / RowLinear` 继续可用。
3. `models/qwen3.py` 能区分全局 heads 和本地 heads。
4. `ModelRunner` 能在 `torchrun` 环境下完成最基础的分布式初始化。
5. 仓库里新增一个 `tests/test_Day6_tp.py`，锁住教学版 TP 烟雾路径。

---

## 2. 前置条件

如果你准备真的照本文改代码，默认前面几篇已经做到下面两件事：

1. `04` 已经把单卡主循环收口好。
2. `02` 已经把 `Qwen3ForCausalLM.forward()` / `compute_logits()` 边界拆开。

如果这两个前提还没落地，这一篇先只看概念，不要急着把代码直接抄进去。

---

## 3. 先看当前仓库的真实状态

### 3.1 `layers/linear.py`

当前 [layers/linear.py](/home/psx/nano_vllm_repro/nano_vll_repro/layers/linear.py:20) 只有单卡版本：

1. `QKVLinear`
2. `MergedLinear`
3. `RowLinear`

它现在还没有：

1. world size / rank helper
2. 输出切分规则
3. 输入切分规则
4. `all_reduce` 规约逻辑

### 3.2 `models/qwen3.py`

当前 [models/qwen3.py](/home/psx/nano_vllm_repro/nano_vll_repro/models/qwen3.py:53) 还是纯单卡头数语义：

1. `self.num_heads = num_heads`
2. `self.num_kv_heads = num_kv_heads`
3. `self.q_size = self.num_heads * self.head_dim`
4. `self.kv_size = self.num_kv_heads * self.head_dim`

如果线性层开始只返回本地 shard，这套写法就不够用了。

### 3.3 `engine/model_runner.py`

当前 [engine/model_runner.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/model_runner.py:44) 还是单卡设备初始化：

1. 没读取 `RANK / LOCAL_RANK / WORLD_SIZE`
2. 没有 `dist.init_process_group()`
3. 没有 `torch.cuda.set_device(local_rank)`
4. KV Cache 还是按全局 `num_key_value_heads` 分配

---

## 4. 本篇修改原则

### 4.1 先改算子层，再改模型层

TP 的切分规则首先属于线性层，不属于模型层。

所以顺序必须是：

1. `layers/linear.py`
2. `models/qwen3.py`
3. `engine/model_runner.py`

### 4.2 保留当前公开类名

这一点非常重要。

前面几篇和现有代码都在用：

- `QKVLinear`
- `MergedLinear`
- `RowLinear`

所以本篇不要求全仓库突然切换到上游类名。

更稳的做法是：

> 新增 TP-aware 内部类，文件末尾继续用 alias 暴露当前公开名字。

### 4.3 分布式未初始化时必须安全退化到单卡

这是你当前仓库和上游最不一样的地方之一。

因为 Day1 到 Day5 的很多脚本仍然会直接单卡运行。

所以本篇所有 helper 都要遵守这条规则：

- 没初始化 `dist` 时，`world_size = 1`
- 没初始化 `dist` 时，`rank = 0`

---

## 5. 逐步修改

## 5.1 在 `layers/linear.py` 里补 TP helper 和 TP-aware 线性层

修改位置：

- 文件：`layers/linear.py`

这一步是本篇少数允许“按模块完整替换”的地方。

原因很简单：

当前单卡版和 TP 版的抽象层次已经不一样了。继续零碎补丁，最后反而更乱。

把下面这段代码放到文件中部，替换当前单卡 `QKVLinear / MergedLinear / RowLinear` 定义区域即可。
文件开头保留 import，文件结尾保留 `default_weight_loader`。

完整代码如下：

```python
import torch.distributed as dist
import torch.nn.functional as F


def divide(numerator: int, denominator: int) -> int:
    """必须整除的整数切分。"""
    assert denominator > 0, "denominator 必须 > 0"
    assert numerator % denominator == 0, f"{numerator} 不能被 {denominator} 整除"
    return numerator // denominator


def get_tp_world_size() -> int:
    """未初始化分布式时安全返回 1。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tp_rank() -> int:
    """未初始化分布式时安全返回 0。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)


class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, divide(output_size, get_tp_world_size()), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int,
    ):
        local_shard_size = divide(self.output_sizes[shard_id], self.tp_size)
        local_shard_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        shard = loaded_weight.chunk(self.tp_size, dim=self.tp_dim)[self.tp_rank]
        param.data[local_shard_offset: local_shard_offset + local_shard_size].copy_(
            shard.to(device=param.device, dtype=param.dtype)
        )


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
    ):
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, get_tp_world_size())
        self.num_kv_heads = divide(total_num_kv_heads, get_tp_world_size())
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_dim
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ):
        assert shard_id in {"q", "k", "v"}
        if shard_id == "q":
            local_size = self.num_heads * self.head_dim
            local_offset = 0
        elif shard_id == "k":
            local_size = self.num_kv_heads * self.head_dim
            local_offset = self.num_heads * self.head_dim
        else:
            local_size = self.num_kv_heads * self.head_dim
            local_offset = self.num_heads * self.head_dim + self.num_kv_heads * self.head_dim

        shard = loaded_weight.chunk(self.tp_size, dim=self.tp_dim)[self.tp_rank]
        param.data[local_offset: local_offset + local_size].copy_(
            shard.to(device=param.device, dtype=param.dtype)
        )


class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(divide(input_size, get_tp_world_size()), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.data.ndim == 1:
            param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            return
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


QKVLinear = QKVParallelLinear
MergedLinear = MergedColumnParallelLinear
RowLinear = RowParallelLinear
```

这里最关键的不是类名，而是下面两条切分纪律：

1. `ColumnParallel` 按输出维切
2. `RowParallel` 按输入维切，然后 `all_reduce`

---

## 5.2 用下面这两份完整方法替换 `Qwen3Attention`

修改位置：

- 文件：`models/qwen3.py`
- 锚点 1：`Qwen3Attention.__init__`
- 锚点 2：`Qwen3Attention.forward`

完整替代代码如下：

```python
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

    tp_size = get_tp_world_size()
    self.total_num_heads = num_heads
    self.total_num_kv_heads = num_kv_heads
    self.num_heads = divide(self.total_num_heads, tp_size)
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.head_dim = head_dim or hidden_size // self.total_num_heads

    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim ** (-0.5)

    self.qkv_proj = QKVLinear(
        hidden_size=hidden_size,
        head_dim=self.head_dim,
        total_num_heads=self.total_num_heads,
        total_num_kv_heads=self.total_num_kv_heads,
        bias=qkv_bias,
    )
    self.o_proj = RowLinear(
        self.total_num_heads * self.head_dim,
        hidden_size,
        bias=False,
    )

    self.rotary_emb = get_rope(
        head_size=self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position,
        base=rope_theta,
    )
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.attn = Attention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        scale=self.scaling,
        num_kv_heads=self.num_kv_heads,
        layer_idx=layer_idx,
    )
```

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
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
```

这里最容易错的是：

> `q_size / kv_size` 现在必须基于本地 head 数，不是全局 head 数。

---

## 5.3 在 `ModelRunner` 里新增 TP 运行时初始化

修改位置：

- 文件：`engine/model_runner.py`
- 锚点：`__init__()` 前后

先新增下面这两个完整方法：

```python
def setup_tp_runtime(self) -> None:
    """
    初始化教学版 TP 运行时。

    规则：
    1. `tensor_parallel_size == 1` 时直接退化为单卡
    2. TP 路径要求 CUDA
    """
    self.tp_size = self.config.tensor_parallel_size
    self.rank = 0
    self.local_rank = 0
    self.is_distributed = False

    if self.tp_size == 1:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Tensor Parallelism 需要 CUDA 环境")

    self.rank = int(os.environ["RANK"])
    self.local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == self.tp_size, (
        f"WORLD_SIZE={world_size} 与 tensor_parallel_size={self.tp_size} 不一致"
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.cuda.set_device(self.local_rank)
    self.device = torch.device("cuda", self.local_rank)
    self.is_distributed = True
```

```python
def is_main_process(self) -> bool:
    return self.rank == 0
```

然后把 `__init__()` 开头改成：

```python
self.config = config
self.setup_tp_runtime()
```

最后，把 `allocate_kv_cache()` 里 `self.num_kv_heads` 的来源改成**本地 KV 头数**。

完整替代片段如下：

```python
self.num_kv_heads = self.model.config.num_key_value_heads // self.config.tensor_parallel_size
```

这里一定不要继续按全局 KV 头数分 cache。

否则多卡时每个 rank 都会多分一份不属于自己的 cache。

---

## 5.4 新建 `tests/test_Day6_tp.py`

新建文件：

- `tests/test_Day6_tp.py`

完整代码如下：

```python
"""Day 6 TP 测试脚本 - 教学版 Tensor Parallelism 验收"""

import os
import sys
sys.path.insert(0, ".")

import torch

from config import Config
from layers.linear import get_tp_world_size, get_tp_rank, divide


def test_tp_helpers_fallback():
    assert divide(8, 2) == 4
    assert get_tp_world_size() >= 1
    assert get_tp_rank() >= 0


def test_config_accepts_tp_size():
    config = Config(
        model_path="models/Qwen3-0.6B",
        tensor_parallel_size=1,
    )
    assert config.tensor_parallel_size == 1


@torch.inference_mode()
def test_qwen3_attention_single_rank_fallback():
    from models.qwen3 import Qwen3Attention

    attn = Qwen3Attention(
        hidden_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
    )
    positions = torch.arange(4)
    hidden_states = torch.randn(4, 128)
    output = attn(positions, hidden_states)
    assert output.shape == hidden_states.shape


if __name__ == "__main__":
    print("Day 6 TP 测试开始")

    test_tp_helpers_fallback()
    test_config_accepts_tp_size()
    test_qwen3_attention_single_rank_fallback()

    if torch.cuda.device_count() >= 2:
        print("检测到至少 2 张 GPU，可以继续做 torchrun 烟雾测试。")
    else:
        print("当前不足 2 张 GPU，只完成单进程 fallback 验证。")

    print("🎉 Day 6 TP 测试执行完成")
```

这份测试故意先锁两件事：

1. 单进程 fallback 不能炸
2. 单 rank 下 TP-aware 模型代码仍然要能工作

真正的双卡烟雾测试，再用下面命令跑：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

---

## 6. 本篇结束后的最小验收

```bash
cd nano_vll_repro
python -m py_compile layers/linear.py models/qwen3.py engine/model_runner.py tests/test_Day6_tp.py
python tests/test_Day6_tp.py
```

如果你有两张 GPU，再补：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

---

## 7. 常见错误

### 7.1 在模型层手搓切分，不先改线性层

这样最后一定会同时出现：

- 模型里一套切分规则
- loader 里另一套切分规则

### 7.2 `world_size > 1` 才能 import 成功

这会把前面所有单卡回归都打碎。

### 7.3 KV Cache 继续按全局 KV 头数分配

这是 TP 初学者最常见的坑之一。

### 7.4 `RowParallelLinear` 忘了 `all_reduce`

这样每个 rank 都只看到自己的局部输出，最后结果一定不对。

---

## 8. 本篇真正学到的东西

TP 的第一层改造对象不是调度器，也不是 `LLMEngine`。

而是：

1. 线性层切分规则
2. 模型里全局头数和本地头数的关系
3. 运行时的 rank / device 初始化

只要这 3 层清楚了，教学版 SPMD TP 就已经成立了一大半。

# 05. 实现 Tensor Parallel 基础版（先升级线性层，再改模型与运行时）

## 1. 本篇目标

这一篇要把仓库从“单卡融合线性层版本”推进到“教学版 TP 可运行版本”，但顺序必须改正确：

1. 先把 `layers/linear.py` 从单卡包装升级成 TP-aware 实现。
2. 再让 `models/qwen3.py` 正确区分 total heads 和 local heads。
3. 最后才让 `ModelRunner`、`example.py` 和 TP smoke test 接上分布式运行时。

本篇完成后，至少要满足下面 5 个状态：

1. `layers/linear.py` 具备 `divide / get_tp_world_size / get_tp_rank` 这些 TP helper。
2. 当前公开名字 `QKVLinear / MergedLinear / RowLinear` 继续可用，但其底层实现已经支持 TP。
3. `Qwen3Attention` 会显式区分全局头数和本地头数。
4. `ModelRunner` 能在 `torchrun` 环境里完成 rank / device 初始化，并按本地 `num_kv_heads` 分配 KV Cache。
5. `tests/test_Day6_tp.py` 能锁住这条教学版 SPMD TP 链路。

这里也先把边界写死：

> 本篇只做“教学版 SPMD TP”。也就是每个 rank 都执行同一套 Python 主循环，只让 rank0 打印用户可见输出。不上 worker 进程池，不上 RPC，不上请求分发协议。

---

## 2. 权威参考

本篇对照下面 4 组来源：

1. 当前仓库：
   - `nano_vll_repro/layers/linear.py`
   - `nano_vll_repro/models/qwen3.py`
   - `nano_vll_repro/engine/model_runner.py`
   - `nano_vll_repro/example.py`
2. 上游主仓库：
   - `https://github.com/GeeeekExplorer/nano-vllm`
   - `nanovllm/layers/linear.py`
   - `nanovllm/models/qwen3.py`
   - `nanovllm/engine/model_runner.py`
3. 公开变体：
   - `qqtang-code/nano-vllm`
   - `wangyuzhuo116/nano-vllm`
   - `DIOYF/nano-vllm-dio`
4. 你前面已经收口好的本地前提：
   - [01-修复当前主干并补齐Linear体系.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/01-修复当前主干并补齐Linear体系.md)
   - [02-补齐Qwen3模型主干与权重映射.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/02-补齐Qwen3模型主干与权重映射.md)
   - [04-补齐单卡推理链路与Day5测试.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/04-补齐单卡推理链路与Day5测试.md)

这次重新核对后，最重要的现实结论是：

1. 上游和公开 fork 当前都已经把 TP-aware 线性层放在 `layers/linear.py` 里，而不是在 `models/qwen3.py` 临时手搓分片。
2. 你当前本地仓库还没有这层基础设施，只有单卡 `QKVLinear / MergedLinear / RowLinear`。
3. 所以本篇最关键的第一步不是改 `ModelRunner`，而是先升级 `layers/linear.py`。

换句话说：

> 旧版 `05` 的根本错误，是把“上游已经做完的 TP 线性层能力”误写成“你当前仓库已经具备的前提”。

---

## 3. 先看当前仓库为什么还不能直接开 TP

### 3.1 `layers/linear.py`

当前 [layers/linear.py](/home/psx/nano_vllm_repro/nano_vll_repro/layers/linear.py:20) 只有：

1. 单卡 `QKVLinear`
2. 单卡 `MergedLinear`
3. 单卡 `RowLinear`

它还没有：

1. `divide()`
2. `get_tp_world_size()`
3. `get_tp_rank()`
4. `ColumnParallelLinear / MergedColumnParallelLinear / RowParallelLinear`
5. “未初始化分布式时退化回单卡”的安全逻辑

### 3.2 `models/qwen3.py`

当前 [models/qwen3.py](/home/psx/nano_vllm_repro/nano_vll_repro/models/qwen3.py:53) 还是纯单卡语义：

1. `self.num_heads = num_heads`
2. `self.num_kv_heads = num_kv_heads`
3. `self.q_size = self.num_heads * self.head_dim`
4. `self.kv_size = self.num_kv_heads * self.head_dim`

一旦 `QKVLinear` 开始只返回本地 shard，这里就必须改成：

1. 先区分 total heads
2. 再计算 local heads
3. `q_size / kv_size` 必须基于 local heads

### 3.3 `engine/model_runner.py`

当前 [engine/model_runner.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/model_runner.py:37) 没有任何统一的分布式初始化：

1. 没读 `RANK / LOCAL_RANK / WORLD_SIZE`
2. 没 `dist.init_process_group()`
3. 没 `torch.cuda.set_device(local_rank)`
4. 没有只在主进程打印日志的约束
5. KV Cache 还是按全局 `num_key_value_heads` 分配

### 3.4 `example.py`

当前 [example.py](/home/psx/nano_vllm_repro/nano_vll_repro/example.py:1) 完全是单卡思路：

1. 没写 `torchrun` 启动方式
2. 没限制只有 rank0 打印
3. 没明确这是 SPMD 教学版 TP 入口

---

## 4. 本篇修改原则

### 4.1 先升级线性层，再升级模型

原因非常直接：

1. TP 的切分规则首先属于“算子定义”。
2. 模型层只是消费这些 TP-aware 算子。
3. 如果算子层还没准备好，模型层再怎么改都会停留在伪接口上。

### 4.2 保留当前公开类名，底层换成 TP-aware 实现

这一条是为了减少前面几篇对调用面的冲击：

1. 文档、模型、测试当前都在用 `QKVLinear / MergedLinear / RowLinear`。
2. 本篇不强迫你把所有调用点改成上游 TP 类名。
3. 更合理的做法是：
   - 新增 TP-aware 的内部类
   - 文件末尾用 alias 保留当前公开名字

也就是说，本篇推荐的暴露方式是：

```python
QKVLinear = QKVParallelLinear
MergedLinear = MergedColumnParallelLinear
RowLinear = RowParallelLinear
```

这样一来：

1. 单卡路径仍然继续工作
2. 模型与测试的 import 面不需要大改
3. TP 能力从底层自然升级进来

### 4.3 教学版 TP 必须允许“未初始化分布式时安全退化为单卡”

这一点和上游主仓库的默认前提不同，但对你当前仓库很重要：

1. Day1 ~ Day5 的很多脚本并不会先 init process group。
2. 如果本篇把 `dist.get_rank()` / `dist.get_world_size()` 直接写死，前面的单卡回归会全部炸掉。

所以本篇必须统一提供：

1. `get_tp_world_size() -> 1`
2. `get_tp_rank() -> 0`

作为未初始化分布式时的安全回退。

---

## 5. 逐步修改

## 5.1 先整文件重写 `layers/linear.py`

修改位置：

- 文件：`nano_vll_repro/layers/linear.py`
- 操作：建议整文件替换为下面这份完整实现

为什么这里建议整文件替换而不是零碎补丁：

1. 当前单卡版和 TP 版的抽象层次已经不同。
2. 如果继续缝缝补补，很容易留下“单卡字段名”和“TP 字段名”并存的混乱状态。

完整替代代码如下。注意：这份代码块的注释密度故意很高，因为你后面调 TP shape 问题时，最先回看的就是这里。

```python
"""融合 Linear 层（TP 基础版）

这份文件在教学仓库里承担两个职责：
1. 单卡时继续提供与之前相同的公开类名。
2. 多卡 TP 时把这些公开类名升级成真正的并行线性层实现。

设计原则：
1. 公开类名保持稳定：QKVLinear / MergedLinear / RowLinear
2. 底层实现换成 TP-aware：QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear
3. 未初始化分布式时安全退化到 world_size=1、rank=0
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator: int, denominator: int) -> int:
    """
    做“必须整除”的整数切分。

    这里不允许静默 floor division，
    因为 TP 头数和维度一旦不能整除，后面 shape 会在更深处才爆炸。
    """

    assert denominator > 0, "denominator 必须 > 0"
    assert numerator % denominator == 0, f"{numerator} 不能被 {denominator} 整除"
    return numerator // denominator


def get_tp_world_size() -> int:
    """
    返回当前 TP world size。

    关键兼容策略：
    - 分布式未初始化时返回 1
    - 这样单卡脚本依旧能把并行层退化成普通线性层
    """

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tp_rank() -> int:
    """
    返回当前 TP rank。

    关键兼容策略：
    - 分布式未初始化时返回 0
    - 这样单卡脚本依旧能走 rank0 路径
    """

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def copy_weight_to_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    把外部权重安全复制到目标参数。

    这里统一做 device / dtype 对齐，避免：
    - CPU 读到的权重直接拷到 CUDA 参数时报错
    - FP32 权重直接拷到 BF16 / FP16 参数时出现不一致
    """

    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    默认权重加载器。

    用于没有自定义分片逻辑的普通参数：
    - embedding
    - norm
    - 其他未融合参数
    """

    assert param.data.shape == loaded_weight.shape, (
        f"Shape mismatch: {param.data.shape} vs {loaded_weight.shape}"
    )
    copy_weight_to_param(param, loaded_weight)


class LinearBase(nn.Module):
    """
    线性层公共基类。

    统一负责：
    1. 保存 tp_size / tp_rank
    2. 创建 weight / bias
    3. 绑定默认初始化

    子类只需要关心：
    - 如何分片
    - 如何加载对应 shard
    - forward 是否需要 all_reduce
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int = 0,
    ) -> None:
        super().__init__()

        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.tp_dim = tp_dim

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # 默认先把 weight_loader 绑定到参数对象上。
        # 子类如果需要特殊分片逻辑，直接覆盖 self.weight_loader 即可。
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def reset_parameters(self) -> None:
        """
        统一初始化。

        这里沿用 PyTorch Linear 常见初始化，不追求训练最优，
        只保证在“未加载真实权重”的测试路径下数值稳定。
        """

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (self.weight.size(1) ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args) -> None:
        """
        默认加载行为。

        对普通参数来说直接 copy 即可；
        对并行层子类来说，会覆盖成“先切 shard 再 copy”的行为。
        """

        copy_weight_to_param(param, loaded_weight)


class ColumnParallelLinear(LinearBase):
    """
    列并行 Linear。

    语义：
    - 按输出维切分
    - 每个 rank 持有输出矩阵的一段
    - forward 后不需要 all_reduce，因为每个 rank 的输出本来就是局部输出
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        tp_size = get_tp_world_size()
        super().__init__(
            input_size=input_size,
            output_size=divide(output_size, tp_size),
            bias=bias,
            tp_dim=0,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args) -> None:
        """
        加载列并行 shard。

        列并行切的是输出维，所以 weight 的第 0 维就是切分维。
        bias 也是沿第 0 维自然切分。
        """

        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        copy_weight_to_param(param, shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """
    多个等尺寸列并行线性层的融合实现。

    当前仓库主要用于：
    - gate_proj
    - up_proj

    这里故意保留和旧版 `MergedLinear` 一致的公开构造器：
    - input_size
    - output_size
    - num_shards

    这样前面几篇文档和当前模型代码不需要立刻改签名。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_shards: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.input_size = input_size
        self.output_size = output_size
        self.num_shards = num_shards
        self.local_output_size = divide(output_size, self.tp_size)
        self.total_size = self.local_output_size * self.num_shards

        self.weight = nn.Parameter(torch.empty(self.total_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int) -> None:
        """
        加载融合列并行权重。

        这里有两层偏移：
        1. 先在单个逻辑分片内部按 TP rank 取本地输出 shard
        2. 再把它写到融合大矩阵里的对应 shard_id 区间
        """

        local_shard_size = self.local_output_size
        source_start = self.tp_rank * local_shard_size
        source_shard = loaded_weight.narrow(0, source_start, local_shard_size)

        target_start = shard_id * local_shard_size
        target_end = target_start + local_shard_size
        param.data[target_start:target_end].copy_(
            source_shard.to(device=param.device, dtype=param.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """
    QKV 融合列并行线性层。

    公开构造器刻意保持与旧版 `QKVLinear` 一致：
    - hidden_size
    - num_heads
    - num_kv_heads
    - head_dim

    这样可以在不破坏前面单卡调用面的情况下，把底层实现升级成 TP-aware。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.num_heads = divide(self.total_num_heads, self.tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, self.tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.total_size = self.q_size + 2 * self.kv_size

        self.weight = nn.Parameter(torch.empty(self.total_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str) -> None:
        """
        加载 Q / K / V 三种分离权重到本地融合参数。

        这里要同时处理：
        1. 逻辑分片类型：q / k / v
        2. TP rank 对应的本地切片
        3. 融合矩阵中的目标偏移
        """

        if shard_id == "q":
            global_size = self.total_num_heads * self.head_dim
            local_size = self.q_size
            target_start = 0
        elif shard_id == "k":
            global_size = self.total_num_kv_heads * self.head_dim
            local_size = self.kv_size
            target_start = self.q_size
        elif shard_id == "v":
            global_size = self.total_num_kv_heads * self.head_dim
            local_size = self.kv_size
            target_start = self.q_size + self.kv_size
        else:
            raise ValueError(f"Unknown shard_id: {shard_id}")

        # 全局分片尺寸必须能被 TP world size 整除。
        assert global_size == loaded_weight.shape[0], (
            f"加载的权重首维 {loaded_weight.shape[0]} 与预期 {global_size} 不一致"
        )

        source_start = self.tp_rank * local_size
        source_shard = loaded_weight.narrow(0, source_start, local_size)

        target_end = target_start + local_size
        param.data[target_start:target_end].copy_(
            source_shard.to(device=param.device, dtype=param.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    """
    行并行 Linear。

    语义：
    - 按输入维切分
    - 每个 rank 只拿到自己那部分输入通道
    - 局部 matmul 后再 all_reduce 聚合输出

    这类层主要用于：
    - attention 的 o_proj
    - MLP 的 down_proj
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        tp_size = get_tp_world_size()
        super().__init__(
            input_size=divide(input_size, tp_size),
            output_size=output_size,
            bias=bias,
            tp_dim=1,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args) -> None:
        """
        加载行并行 shard。

        对 weight：
        - 沿输入维，也就是第 1 维切分

        对 bias：
        - bias 不切分，保持完整
        """

        # bias 只有一维，不做输入维切分。
        if param.data.ndim == 1:
            copy_weight_to_param(param, loaded_weight)
            return

        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        copy_weight_to_param(param, shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行前向。

        关键细节：
        - 多卡时只有 rank0 在 matmul 时加 bias
        - 否则 all_reduce 后 bias 会被重复累加 tp_size 次
        """

        if self.tp_size > 1:
            local_bias = self.bias if self.tp_rank == 0 else None
        else:
            local_bias = self.bias

        y = F.linear(x, self.weight, local_bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


# ===== 兼容当前仓库公开类名 =====
# 这一层 alias 的意义非常重要：
# - 前面几篇文档、模型代码、测试代码都已经在用这些名字
# - 本篇要升级的是“底层实现”，不是强迫所有调用点当天改名
QKVLinear = QKVParallelLinear
MergedLinear = MergedColumnParallelLinear
RowLinear = RowParallelLinear
```

这一整步真正要理解的是：

1. 先升级“算子层”，比先改模型和 runner 更稳。
2. alias 不是多此一举，它是在保护前面几篇文档已经收口好的调用面。

---

## 5.2 再改 `models/qwen3.py`，显式拆出 total heads 和 local heads

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 操作：只替换 `Qwen3Attention` 的 import、`__init__()` 和 `forward()` 里头数相关部分

这一步不用推翻整层，只改“头数语义”和 `split / view` 逻辑。

### 第一步：把 import 改成下面这样

```python
from layers.linear import (
    QKVLinear,
    MergedLinear,
    RowLinear,
    divide,
    get_tp_world_size,
)
```

### 第二步：替换 `Qwen3Attention.__init__()` 中头数与投影初始化部分

完整替代代码如下：

```python
# ===== 头数与维度语义 =====
# 这里先保留“全局头数”，因为 TP 下 config 代表的是模型全局规格。
self.total_num_heads = num_heads
self.total_num_kv_heads = num_kv_heads

# 读取当前 TP world size；单卡未初始化分布式时会安全返回 1。
tp_size = get_tp_world_size()

# 当前 rank 真正持有的是 local heads。
self.num_heads = divide(self.total_num_heads, tp_size)
self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)

# head_dim 仍然优先信任配置。
self.head_dim = head_dim or hidden_size // self.total_num_heads

# q_size / kv_size 现在必须基于“本地头数”计算。
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
self.scaling = self.head_dim ** -0.5

# ===== 投影层 =====
# 注意：公开类名不变，但底层已经是 TP-aware alias。
self.qkv_proj = QKVLinear(
    hidden_size=hidden_size,
    num_heads=self.total_num_heads,
    num_kv_heads=self.total_num_kv_heads,
    head_dim=self.head_dim,
    bias=qkv_bias,
)

# RowLinear 内部会自己按输入维切分，所以这里传全局 q 输出维度。
self.o_proj = RowLinear(
    input_size=self.total_num_heads * self.head_dim,
    output_size=hidden_size,
    bias=False,
)
```

这里最容易写错的地方是 `o_proj`：

1. 不要传本地 `self.q_size`
2. 要传全局 `self.total_num_heads * self.head_dim`

原因是：

1. `RowLinear` 自己会按输入维做分片。
2. 如果你这里已经传了本地尺寸，内部还会再切一次，输入维就被切重了。

### 第三步：替换 `Qwen3Attention.forward()` 里 `split / view` 部分

完整替代代码如下：

```python
# ===== QKV 融合投影 =====
qkv = self.qkv_proj(hidden_states)

# 这里切分的尺寸必须是“本地尺寸”。
# 因为 TP-aware QKVLinear 返回的已经是当前 rank 的局部输出 shard。
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

# reshape 时也必须使用本地头数。
q = q.view(num_tokens, self.num_heads, self.head_dim)
k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

# q_norm / k_norm 的语义不变，依旧作用在 head_dim 上。
q = self.q_norm(q)
k = self.k_norm(k)

# 位置编码仍然只作用在 q / k 上。
q, k = self.rotary_emb(positions, q, k)
```

这一步最重要的理解是：

1. 全局头数属于“模型配置语义”
2. 本地头数属于“当前 rank 张量 shape 语义”
3. `split / view` 一定要跟着本地头数走

---

## 5.3 在 `ModelRunner` 里新增统一的分布式初始化

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：
  - 先把 import 区补成下面这样
  - 在 `class ModelRunner` 内、`__init__` 之前插入 `init_distributed_env()`
  - 在 `__init__()` 最开始调用它

先补 import：

```python
import os

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer
from typing import Optional

from config import Config
from engine.sequence import Sequence
from utils.context import Context, set_context, get_context
from utils.loader import load_model
from layers.sampler import Sampler
from layers.linear import divide
```

完整新增方法如下：

```python
def init_distributed_env(self) -> None:
    """
    初始化教学版 TP 运行时环境。

    输入：
    - 无；直接读取 config 和 torchrun 注入的环境变量

    输出：
    - 无；就地设置 self.rank / self.local_rank / self.tp_size / self.device

    当前策略：
    1. `tensor_parallel_size == 1` 时直接退化为单卡
    2. 多卡时要求 CUDA 可用
    3. 多卡时要求通过 `torchrun` 启动
    """

    # 先把最保守的默认值写上，这样单卡路径不用再重复设置。
    self.rank = 0
    self.local_rank = 0
    self.tp_size = self.config.tensor_parallel_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 单卡路径到此为止，不做任何分布式初始化。
    if self.config.tensor_parallel_size == 1:
        return

    # TP 教学版明确要求 CUDA。
    if not torch.cuda.is_available():
        raise RuntimeError("Tensor Parallelism 需要 CUDA 环境")

    # 多卡路径下，如果还没初始化 process group，这里统一初始化。
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    self.rank = dist.get_rank()
    self.tp_size = dist.get_world_size()
    self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))

    # 设备绑定必须发生在模型构建之前。
    torch.cuda.set_device(self.local_rank)
    self.device = torch.device("cuda", self.local_rank)

    # 配置里的期望 TP size 和真实 world size 不一致时必须立即报错。
    assert self.config.tensor_parallel_size == self.tp_size, (
        f"config.tensor_parallel_size={self.config.tensor_parallel_size} "
        f"但实际 world_size={self.tp_size}"
    )
```

然后在 `__init__()` 最开头加：

```python
self.config = config
self.init_distributed_env()
```

这里的关键不是“多几行初始化代码”，而是：

1. 模型加载前必须先知道 device
2. KV Cache 分配前必须先知道本地 rank

---

## 5.4 统一主进程判定，并把日志收敛到 rank0

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：在 `init_distributed_env()` 后、`_load_model()` 前插入下面这个 property

```python
@property
def is_main_process(self) -> bool:
    """
    教学版 TP 下的主进程判定。

    当前约定非常简单：
    - 只有 rank0 对用户打印主要日志和最终文本输出
    """

    return self.rank == 0
```

然后把 `_load_model()` 和 `allocate_kv_cache()` 里的 `print(...)` 全部包成：

```python
if self.is_main_process:
    print(...)
```

原因不是为了“日志好看”，而是：

1. SPMD 下每个 rank 都会执行同一段 Python。
2. 如果不做 rank0 约束，终端会同时出现 N 份重复日志。

---

## 5.5 让 KV Cache 按本地 `num_kv_heads` 分配

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：
  - 替换 `__init__()` 里 `self.num_kv_heads = ...`
  - 替换 `allocate_kv_cache()` 里的 dtype 和形状

完整替代代码如下：

```python
# ===== 模型配置缓存 =====
self.num_layers = self.model.config.num_hidden_layers

# 这里必须按 TP world size 计算“当前 rank 的本地 KV 头数”。
self.num_kv_heads = divide(self.model.config.num_key_value_heads, self.tp_size)

self.head_dim = getattr(
    self.model.config,
    "head_dim",
    self.model.config.hidden_size // self.model.config.num_attention_heads,
)

self.block_size = Sequence.block_size
```

`allocate_kv_cache()` 里则替换成：

```python
cache = torch.zeros(
    2,
    num_blocks,
    self.block_size,
    self.num_kv_heads,
    self.head_dim,
    dtype=self.config.kv_torch_dtype,
    device=self.device,
)
```

这里的核心理解是：

1. 每个 rank 只持有本地 KV 头。
2. 所以每个 rank 也只该为本地 KV 头分配 cache。

如果你这里继续用全局 `num_key_value_heads`，后果就是：

1. 每张卡都重复分配整份 KV Cache
2. 显存直接浪费掉一倍甚至更多

---

## 5.6 `example.py` 改成教学版 TP 入口

修改位置：

- 文件：`nano_vll_repro/example.py`
- 操作：
  - 在文件顶部补启动方式说明
  - 在打印结果处加 rank0 保护

建议新增的文件头说明如下：

```python
# 单卡：
#   python example.py
#
# 双卡 TP：
#   torchrun --nproc_per_node=2 example.py
```

然后在真正打印回答前，补：

```python
import os


is_rank0 = int(os.environ.get("RANK", "0")) == 0

if is_rank0:
    for prompt, output in zip(raw_prompts, outputs):
        print(f"\n[问题] {prompt}")
        print(f"[回答] {output['text']}")
        print(f"[生成 token 数] {len(output['token_ids'])}")
```

这里你不需要把 `LLMEngine` 改成复杂 worker 架构。教学版 TP 的前提就是：

1. 所有 rank 跑同一份 Python 控制流
2. 模型内部并行算子负责同步数值
3. 只有 rank0 暴露用户可见输出

---

## 5.7 新增 `tests/test_Day6_tp.py`

直接新建：

- `nano_vll_repro/tests/test_Day6_tp.py`

完整代码如下。注释密度依旧故意很高，因为这份文件本身就是 TP 行为边界的可执行说明。

```python
"""Day 6 TP 测试脚本 - 教学版 Tensor Parallelism 验收

本文件只验证两件事情：
1. Qwen3Attention 是否正确计算本地头数。
2. ModelRunner 是否按本地 KV 头数分配 KV Cache。

注意：
1. 这份文件必须用 `torchrun` 启动。
2. 这份测试不是生产级多进程调度测试，而是教学版 SPMD smoke test。
"""

import os
import sys

import torch
import torch.distributed as dist


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


from config import Config
from engine.model_runner import ModelRunner
from models.qwen3 import Qwen3Attention


def ensure_distributed_initialized() -> tuple[int, int, int]:
    """
    保证当前 `torchrun` 环境已初始化。

    这里单独抽 helper 的原因是：
    - Qwen3Attention 构造时会读取 TP world size
    - 如果不先初始化，它会退化成单卡语义，测试就失去意义
    """

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


@torch.inference_mode()
def test_qwen3_attention_local_heads() -> None:
    """验证 Qwen3Attention 的本地头数计算。"""

    rank, _, world_size = ensure_distributed_initialized()

    # 这里故意选一个可整除的头数配置，让断言重点只落在“切分结果”。
    attn = Qwen3Attention(
        hidden_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
    )

    # total heads 代表模型配置语义，不因 rank 改变。
    assert attn.total_num_heads == 8
    assert attn.total_num_kv_heads == 2

    # local heads 才代表当前 rank 的真实持有量。
    assert attn.num_heads == 8 // world_size
    assert attn.num_kv_heads == 2 // world_size
    assert attn.q_size == attn.num_heads * attn.head_dim
    assert attn.kv_size == attn.num_kv_heads * attn.head_dim

    print(f"[rank {rank}] ✅ Qwen3Attention 本地头数测试通过")


@torch.inference_mode()
def test_model_runner_local_kv_cache_layout() -> None:
    """验证 ModelRunner 的本地 KV Cache 形状。"""

    rank, _, world_size = ensure_distributed_initialized()

    model_path = os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B")
    if not os.path.isdir(model_path):
        print(f"[rank {rank}] skip: 模型路径不存在 {model_path}")
        return

    config = Config(
        model_path=model_path,
        tensor_parallel_size=world_size,
        max_num_seqs=8,
        max_num_batched_tokens=256,
    )

    runner = ModelRunner(config)
    runner.allocate_kv_cache(2)

    # rank / tp_size 基本一致性。
    assert runner.rank == rank
    assert runner.tp_size == world_size

    # 当前 rank 的本地 KV 头数必须等于全局 KV 头数 / world_size。
    assert runner.num_kv_heads == runner.model.config.num_key_value_heads // world_size

    # KV Cache 结构：[2, num_blocks, block_size, num_kv_heads, head_dim]
    assert runner.kv_cache is not None
    assert len(runner.kv_cache) == runner.num_layers

    first_layer_cache = runner.kv_cache[0]
    assert first_layer_cache.shape[0] == 2
    assert first_layer_cache.shape[1] == 2
    assert first_layer_cache.shape[3] == runner.num_kv_heads
    assert first_layer_cache.shape[4] == runner.head_dim

    print(f"[rank {rank}] ✅ ModelRunner 本地 KV Cache 形状测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 6 TP 测试开始")
    print("=" * 60)

    # 教学版 TP 明确要求至少 2 张 GPU。
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("skip: TP smoke test 需要至少 2 张 CUDA GPU")
        sys.exit(0)

    test_qwen3_attention_local_heads()
    test_model_runner_local_kv_cache_layout()

    # 所有 rank 在退出前同步一次，避免某些 rank 提前结束导致其他 rank 报错。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print("=" * 60)
    print("🎉 Day 6 TP 测试执行完成")
    print("=" * 60)
```

运行方式也要在文档里写死：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

---

## 6. 本篇结束后的最小验收

先做语法检查：

```bash
cd nano_vll_repro
python -m py_compile layers/linear.py models/qwen3.py engine/model_runner.py example.py
```

有 2 张及以上 GPU 时，再执行：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
torchrun --nproc_per_node=2 example.py
```

---

## 7. 常见错误

### 7.1 只改 `ModelRunner`，不改 `layers/linear.py`

后果：

- 你会在模型层里写出一套“想象中的 TP 头数逻辑”
- 但底层线性层仍然只会返回单卡全量输出

### 7.2 `o_proj` 误传本地 `q_size`

后果：

- `RowLinear` 会再做一次输入维切分
- 输入维被切两次，shape 会在更深处才爆

### 7.3 KV Cache 还按全局 `num_kv_heads` 分配

后果：

- 每张卡都在重复存整份 KV
- 显存开销直接失真

### 7.4 所有 rank 都打印结果

后果：

- 终端输出混乱
- 很难区分到底是数值错误还是只是日志重复

---

## 8. 本篇真正学到的东西

这一篇真正重要的是下面 4 件事：

1. TP 的第一层改造对象是线性层，不是 runner。
2. 公开调用面可以稳定，底层实现可以升级，这正是 alias 的价值。
3. global heads 和 local heads 必须同时写清楚。
4. 教学版 SPMD TP 的核心不是复杂进程架构，而是“所有 rank 跑同一条控制流、算子层内部做同步”。

完成后进入下一篇：

- [06-实现CUDA-Graph基础版.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/06-实现CUDA-Graph基础版.md)

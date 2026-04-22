# 05. 实现 Tensor Parallel 基础版（先改模型头数语义，再接运行时）

## 1. 本篇目标

这一篇要把你的仓库从“单卡稳定版”推进到“基础版 TP 可运行版”，但必须按正确顺序做：

1. 先把 `models/qwen3.py` 改成 TP 友好的头数语义
2. 再让 `ModelRunner` 正确初始化分布式环境
3. 最后再处理示例脚本和 TP smoke test

如果你跳过第一步、直接改 runner，结果通常是：

- `qkv_proj` 输出分片了，但 `q_size / kv_size` 还按全局头数在切
- 注意力层一开始就把张量拆错

本篇完成后，应至少满足：

- `Qwen3Attention` 明确区分 total heads 和 local heads
- `ModelRunner` 能在 `torchrun` 环境下完成 rank / device 初始化
- KV Cache 按“本地 `num_kv_heads`”分配，而不是按全局头数重复分配
- `example.py` 可以作为教学版 SPMD TP 入口

---

## 2. 权威参考

本篇对照：

1. 当前仓库：
   - `models/qwen3.py`
   - `engine/model_runner.py`
   - `example.py`
2. 上游：
   - `nano-vllm/nanovllm/models/qwen3.py`
   - `nano-vllm/nanovllm/engine/model_runner.py`
3. 已在本地准备好的基础设施：
   - `layers/linear.py`
   - `config.py`

注意一个关键差异：

> 上游 TP 是“主进程 + worker 进程”架构；你当前这一篇只做“教学版 SPMD”，即每个 rank 跑同一套 Python 主循环，只让 rank0 打印结果。

---

## 3. 先看当前仓库为什么还不能直接开 TP

### 3.1 `models/qwen3.py`

这是本篇最核心的问题源头。

当前 `Qwen3Attention` 还是按“单卡头数”在写：

- `self.num_heads = num_heads`
- `self.num_kv_heads = num_kv_heads`
- `self.q_size = self.num_heads * self.head_dim`

但在 TP 场景里，`QKVLinear` 输出的是当前 rank 的局部分片，所以：

- `self.q_size` 必须是本地 Q 头数乘以 `head_dim`
- `self.kv_size` 必须是本地 KV 头数乘以 `head_dim`

### 3.2 `engine/model_runner.py`

当前 runner 没有统一处理：

- `dist.init_process_group()`
- `LOCAL_RANK`
- `torch.cuda.set_device(local_rank)`
- rank0 日志输出控制
- 本地 `num_kv_heads` 的 KV Cache 分配

### 3.3 `example.py`

当前示例是单卡思路，没有说明：

- 如何用 `torchrun` 启动
- 为什么所有 rank 都要执行同一段 Python 逻辑
- 为什么只允许 rank0 打印输出

---

## 4. 本篇修改原则

### 4.1 先改模型，再改 runner

因为 TP 的数学语义属于模型定义的一部分，不属于运行时策略。

如果模型里这些关系没先写清楚：

- 总头数
- 本地头数
- `q_size / kv_size`
- `o_proj` 的 global input size

那 runner 再怎么 init 分布式都没用。

### 4.2 这一步只做“教学版 TP”

当前范围限定为：

- `torchrun --nproc_per_node=N`
- 每个 rank 跑同一份 `LLMEngine`
- 所有 rank 使用同一批 prompts
- 只让 rank0 打印最终结果

现在不要引入：

- 共享内存 worker
- 控制进程 / 执行进程分离
- request 分发协议

---

## 5. 逐步修改

## 5.1 先改 `models/qwen3.py`：显式区分 total heads 和 local heads

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 锚点：定位到 `class Qwen3Attention.__init__`
- 操作：把头数相关字段、`qkv_proj` 初始化、`o_proj` 初始化替换为下面这份代码

在 `Qwen3Attention.__init__()` 开头，新增对 TP world size 的读取：

```python
from layers.linear import QKVLinear, MergedLinear, RowLinear, divide, get_tp_world_size
```

然后把头数相关逻辑收口成下面这种结构：

```python
tp_size = get_tp_world_size()

# total_* 永远代表模型全局配置，不因为 rank 改变。
self.total_num_heads = num_heads
self.total_num_kv_heads = num_kv_heads

# local 头数才是当前 rank 真实持有的张量分片。
self.num_heads = divide(self.total_num_heads, tp_size)
self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)

self.head_dim = head_dim or hidden_size // self.total_num_heads
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
```

这几行看起来简单，但它们决定了后面所有张量 shape 是否正确。

### 同步改 `qkv_proj` 和 `o_proj` 的初始化

`qkv_proj` 仍然必须按“全局头数”构造，因为底层 `QKVLinear` 自己会做分片：

```python
self.qkv_proj = QKVLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=self.total_num_heads,
    total_num_kv_heads=self.total_num_kv_heads,
    bias=qkv_bias,
)
```

`o_proj` 的 `input_size` 也要继续用全局 heads：

```python
self.o_proj = RowLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    bias=False,
)
```

为什么这里不是 `self.q_size`：

- 因为 `RowLinear` 内部会自己按输入维度切分
- 传全局 input size，它才能正确计算每个 rank 的本地输入分片尺寸

---

## 5.2 再改 `Qwen3Attention.forward()`：split 尺寸必须是 local 尺寸

修改位置：

- 文件：`nano_vll_repro/models/qwen3.py`
- 锚点：定位到 `class Qwen3Attention.forward` 内部，从 `qkv = self.qkv_proj(...)` 开始，到 `v = v.view(...)` 结束，替换为下面这段

这一点经常被漏。

当前如果你还写：

```python
q, k, v = qkv.split([global_q_size, global_kv_size, global_kv_size], dim=-1)
```

那在 TP 下会直接崩，因为 `qkv_proj` 返回的是本地 shard，不是全量输出。

正确做法是继续使用刚刚定义好的：

```python
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

并且 reshape 时也必须使用本地头数：

```python
q = q.view(num_tokens, self.num_heads, self.head_dim)
k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
```

这样 `Attention` 层拿到的就是“当前 rank 的局部多头张量”。

---

## 5.3 在 `ModelRunner` 里补分布式初始化和 rank 信息

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点 1：定位到 `class ModelRunner` 内、`__init__` 前，插入下面这份 `init_distributed_env()` 方法
- 锚点 2：定位到 `__init__()` 开头，在加载模型之前调用 `self.init_distributed_env()`

推荐新增一个 `init_distributed_env()`，并在 `__init__()` 早期调用。

结构建议：

```python
def init_distributed_env(self) -> None:
    """
    输入：无；直接读取 config 与 torchrun 注入的环境变量。
    输出：无；更新 self.rank / self.local_rank / self.device。

    当前策略：
    - tensor_parallel_size == 1 时直接返回；
    - 只有多卡时才初始化 NCCL；
    - 设备绑定必须在模型构建前完成。
    """
    if self.config.tensor_parallel_size == 1:
        self.rank = 0
        self.local_rank = 0
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Tensor Parallelism 需要 CUDA 环境")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    self.rank = dist.get_rank()
    self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
    torch.cuda.set_device(self.local_rank)
    self.device = torch.device("cuda", self.local_rank)
```

这一步一定要发生在：

- 构建模型之前
- 分配 KV Cache 之前

否则权重和 cache 很容易落到错误设备上。

---

## 5.4 让 `ModelRunner` 的 KV Cache 按本地 `num_kv_heads` 分配

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点 1：定位到 `__init__()` 里 `self.num_kv_heads = ...`
- 锚点 2：定位到 `allocate_kv_cache()` 里 cache shape 的 `num_kv_heads` 维度
- 操作：都替换为下面这套“本地 KV 头数”写法

这是 TP 下最容易浪费显存的地方。

你不能再沿用单卡逻辑里的：

```python
self.num_kv_heads = self.model.config.num_key_value_heads
```

而应该改成：

```python
self.tp_size = get_tp_world_size()
self.num_kv_heads = divide(self.model.config.num_key_value_heads, self.tp_size)
```

然后 `allocate_kv_cache()` 里使用的 shape 才能变成：

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

- 每个 rank 只持有自己那部分 KV 头
- 所以每个 rank 只需要为本地 KV 头分配 cache

---

## 5.5 统一 `is_main_process`，把打印和用户可见输出都收敛到 rank0

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 `init_distributed_env()` 后、`_load_model()` 前，插入下面这份 `is_main_process` property

在 `ModelRunner` 里补一个 property：

```python
@property
def is_main_process(self) -> bool:
    return self.rank == 0
```

之后所有这些输出都应该受它控制：

- 模型加载日志
- KV Cache 分配日志
- example.py 最终回答打印

原因不是“日志少一点”，而是：

- SPMD 模式下每个 rank 都会执行同一段 Python
- 如果不管控输出，终端会出现 N 份重复信息，调试价值极低

---

## 5.6 把 `example.py` 改成教学版 TP 入口，而不是只适用于单卡

修改位置：

- 文件：`nano_vll_repro/example.py`
- 锚点 1：定位到文件顶部模块注释区，补上单卡 / torchrun 两种启动方式说明
- 锚点 2：定位到最终打印回答的循环，在循环外加入 `is_rank0` 判断，并把循环包进 `if is_rank0:` 中

建议在文件头注释里明确写出两种启动方式：

```python
# 单卡：
#   python example.py
#
# 双卡 TP：
#   torchrun --nproc_per_node=2 example.py
```

然后在真正打印回答的地方，加一层 rank0 保护：

```python
import os

is_rank0 = int(os.environ.get("RANK", "0")) == 0

if is_rank0:
    for prompt, output in zip(raw_prompts, outputs):
        print(f"[问题] {prompt}")
        print(f"[回答] {output['text']}")
```

你这里不需要把 `LLMEngine` 改成复杂的多进程框架，因为教学版 TP 的前提就是：

- 各 rank 执行完全相同的 Python 控制流
- 模型层内部通过 `all_reduce` 保持结果一致

---

## 5.7 新增 `tests/test_Day6_tp.py`，这里也必须直接给完整文件

直接新建：

- `nano_vll_repro/tests/test_Day6_tp.py`

完整代码如下。这份文件默认用 `torchrun` 启动，因此它的写法故意保持“每个 rank 执行同一份脚本、只在最后统一退出”的教学风格。

```python
"""Day 6 TP 测试脚本 - 教学版 Tensor Parallelism 验收

这份测试文件只验证教学版 TP 的两个核心事实：

1. Qwen3Attention 会按照 world size 正确计算本地 Q / KV 头数。
2. ModelRunner 会按照本地 KV 头数分配每个 rank 的 KV Cache。

注意：
1. 这份文件必须用 `torchrun` 执行，而不是普通 `python`。
2. 这不是生产级多进程 worker 测试，而是 SPMD 教学版 smoke test。
3. 为了和仓库当前风格一致，注释会写得比较密。
"""

import os
import sys

import torch
import torch.distributed as dist


# 让测试脚本可以直接导入仓库模块。
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


# 这些导入覆盖了 TP 验收真正关心的模块：
# 1. Config：用于构造带 tensor_parallel_size 的运行配置
# 2. ModelRunner：用于验证 rank / device / KV Cache 本地形状
# 3. Qwen3Attention：用于验证 local head 计算
from config import Config
from engine.model_runner import ModelRunner
from models.qwen3 import Qwen3Attention


def _ensure_distributed_initialized() -> tuple[int, int, int]:
    """确保当前 `torchrun` 环境已经初始化分布式。

    输入：无，直接读取 torchrun 注入的环境变量。
    输出：
    1. rank
    2. local_rank
    3. world_size

    之所以单独抽成 helper：
    - Qwen3Attention 在构造时会读取 TP world size；
    - 如果不先初始化分布式，它会退化成单卡语义，测试就失去意义。
    """

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # 设备绑定必须和 local_rank 对齐。
    # 这一步不是装饰动作，而是所有后续 CUDA 张量分配的前提。
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


@torch.inference_mode()
def test_qwen3_attention_local_heads() -> None:
    """测试 Qwen3Attention 的本地头数计算。

    这一步只看模型数学语义，不看真正前向。
    因为 forward 依赖更多 runtime 上下文，而这里的目标只是先把“头数切分规则”锁住。
    """

    rank, _, world_size = _ensure_distributed_initialized()

    # 这里故意选一个可以被 world_size 整除的头数配置。
    # 这样断言的重点就只剩“切分结果是否正确”，而不是配置本身是否合法。
    attn = Qwen3Attention(
        hidden_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
    )

    assert attn.total_num_heads == 8
    assert attn.total_num_kv_heads == 2
    assert attn.num_heads == 8 // world_size
    assert attn.num_kv_heads == 2 // world_size
    assert attn.q_size == attn.num_heads * attn.head_dim
    assert attn.kv_size == attn.num_kv_heads * attn.head_dim

    print(f"[rank {rank}] ✅ Qwen3Attention 本地头数测试通过")


@torch.inference_mode()
def test_model_runner_local_kv_cache_layout() -> None:
    """测试 ModelRunner 的本地 KV Cache 布局。

    这里会真正初始化一份 runner，并分配少量 KV Cache。
    验证重点是：
    - 当前 rank 记录的本地 `num_kv_heads` 是否正确
    - 分配出来的 cache tensor 第 4 维是否等于本地 KV 头数
    """

    rank, _, world_size = _ensure_distributed_initialized()

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

    # 这里不需要分配很多块，2 块就足够验证形状。
    runner.allocate_kv_cache(2)

    assert runner.tp_size == world_size
    assert runner.rank == rank
    assert runner.num_kv_heads == runner.model.config.num_key_value_heads // world_size
    assert runner.kv_cache is not None
    assert len(runner.kv_cache) == runner.num_layers

    # KV Cache 形状：[2, num_blocks, block_size, num_kv_heads, head_dim]
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

    # 依次执行两个核心测试。
    test_qwen3_attention_local_heads()
    test_model_runner_local_kv_cache_layout()

    # 所有 rank 在退出前做一次 barrier，避免某些 rank 提前退出导致其他 rank 报错。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print("=" * 60)
    print("🎉 Day 6 TP 测试执行完成")
    print("=" * 60)
```

运行方式也必须在文档里写死，不要省略：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

---

## 6. 本篇结束后的最小验收

先做语法检查：

```bash
cd nano_vll_repro
python -m py_compile models/qwen3.py engine/model_runner.py example.py
```

有 2 张及以上 GPU 时，再执行：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
torchrun --nproc_per_node=2 example.py
```

---

## 7. 常见错误

### 7.1 只改 runner，不改 `Qwen3Attention` 的 local head 语义

后果：

- `qkv_proj` 的输出分片与 `split()` 尺寸不匹配
- 错误通常在 reshape 时才暴露

### 7.2 `o_proj` 误传本地 `q_size`

后果：

- `RowLinear` 会再次分片，最终输入维度被切了两次
- 这类错误非常隐蔽，因为单卡下可能看不出来

### 7.3 KV Cache 还按全局 `num_kv_heads` 分配

后果：

- 每张卡都在重复存整份 KV
- 显存直接翻倍甚至更多

### 7.4 所有 rank 都打印答案

后果：

- 终端输出混乱
- 很难分辨到底是计算错误还是只是日志重复

---

## 8. 本篇真正学到的东西

这一篇真正重要的是：

1. TP 的数学切分规则属于模型定义，不属于 runner 的附属细节。
2. SPMD 教学版 TP 的核心是“所有 rank 同步执行同一条控制流”。
3. local heads、global heads、KV Cache 本地形状，这三件事必须一起改。

完成后进入下一篇：

- [06-实现CUDA-Graph基础版.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/06-实现CUDA-Graph基础版.md)

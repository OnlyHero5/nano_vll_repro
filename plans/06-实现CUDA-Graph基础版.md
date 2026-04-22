# 06. 实现 CUDA Graph 基础版（只图化 decode，且只做 exact-match replay）

## 1. 本篇目标

这一篇的任务不是“把整个系统都图化”，而是给你当前这份教学仓库加一版边界清晰、能解释、能验收的 decode-only CUDA Graph。

本篇完成后，目标行为应当是：

1. 只在 CUDA 环境下启用
2. 只在 `enforce_eager=False` 时启用
3. 只图化 decode 路径
4. 只在 batch size 命中已录制图时 replay
5. 其他情况全部回退 eager

这一步的核心不是“追求极限快”，而是把：

- 静态输入 buffer
- graph capture
- graph replay
- eager fallback

这四件事讲清楚并落到你当前仓库里。

---

## 2. 权威参考

本篇只对照：

1. 当前仓库：
   - `utils/context.py`
   - `engine/model_runner.py`
2. 上游参考：
   - `nano-vllm/nanovllm/engine/model_runner.py`
   - `nano-vllm/nanovllm/utils/context.py`

还要明确一个前提：

> 你当前仓库在 `04` 已经把 `run_model()` 抽出来了；这一篇默认你已经按 `04` 的写法完成接口收口，否则这里的图捕获边界会不清晰。

---

## 3. 先看当前仓库的真实缺口

### 3.1 `utils/context.py`

当前其实已经有：

- `Context` dataclass
- `set_context()`
- `get_context()`
- `reset_context()`

但还没有把“graph replay 时的静态 context”这个需求写清楚。

### 3.2 `engine/model_runner.py`

当前 runner 已经具备两个非常好的前置条件：

1. `prepare_decode()` 已经能把 decode 所需元数据收集出来
2. `run_model()` 已经是单独的边界

但还缺 5 个核心部件：

1. decode graph runner 的静态 buffer 容器
2. graph capture 入口
3. graph replay 入口
4. 命中 / fallback 分发逻辑
5. `allocate_kv_cache()` 完成后触发 capture 的时机

---

## 4. 本篇修改原则

### 4.1 capture 时机必须放在 KV Cache 分配之后

原因非常实际：

- decode 图里 attention 会访问真实 KV Cache
- 如果先 capture、后分配 cache，图里引用的对象就不稳定

### 4.2 sampler 不进 graph

这一步只 capture：

- `model(input_ids, positions)` 主干
- 对应的静态 decode `Context`

采样仍然在 graph 外做，原因有两个：

1. sampler 本身包含随机数与动态参数，不适合先塞进基础版 graph
2. 你现在真正要学的是“主干 replay”，不是“把所有东西都图化”

### 4.3 先做 exact-match，不做 padding 命中更大图

教学版的策略是：

- batch size 精确命中某个已录制图时 replay
- 否则 eager

现在不要提前引入：

- padding 到更大 batch
- graph 档位合并策略
- 图缓存淘汰策略

---

## 5. 逐步修改

## 5.1 先把 `Context` 保持成“纯数据”，并明确 reset 语义

修改位置：

- 文件：`nano_vll_repro/utils/context.py`
- 锚点 1：定位到 `@dataclass class Context`
- 锚点 2：定位到 `reset_context()`
- 操作：用下面给出的类注释和函数注释替换原有简写说明；如果字段定义与下面代码不同，以替代代码为准

`utils/context.py` 本篇不需要大改结构，但要把注释和字段职责写清楚，尤其是下面这 4 个字段：

- `slot_mapping`
- `context_lens`
- `block_tables`
- `kv_cache`

推荐你把 `Context` 的类注释收口成下面这种风格：

```python
@dataclass
class Context:
    """
    全局推理上下文。

    输入：由 ModelRunner 在 prepare_prefill / prepare_decode / graph replay 前设置。
    输出：由 Attention 层在 forward 内读取。

    设计边界：
    - 这里不做任何计算，只保存当前 step 所需的静态元数据；
    - 之所以保留全局单例，而不是层层传参，是因为当前仓库是教学版 token 流模型，修改所有中间层签名的成本过高。
    """
```

然后确保 `reset_context()` 的注释明确写出一句：

> 每次 run 结束后、每次 graph capture 结束后，都必须恢复为一个新的空 `Context()`。

这句话是后面排查 graph 污染问题的关键。

---

## 5.2 在 `ModelRunner` 里新增 `DecodeGraphRunner` 容器

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 import 区之后、`class ModelRunner` 定义之前，插入下面这份 `@dataclass DecodeGraphRunner`

不要把所有静态 buffer 散落成十几个 `self.xxx` 变量，后面会很难维护。建议直接在 `engine/model_runner.py` 顶部加一个 dataclass：

```python
@dataclass
class DecodeGraphRunner:
    """
    单个 decode batch size 对应的一组静态图资源。

    输入：batch_size 档位、静态输入 buffer、CUDA Graph 实例。
    输出：graph replay 所需的全部状态容器。

    为什么要单独做 dataclass：
    - 你后面会同时持有多个 batch size 的图；
    - 如果不把资源收进一个结构体，self 上的状态会迅速失控。
    """

    batch_size: int
    max_num_blocks: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    hidden_states: torch.Tensor
    graph: torch.cuda.CUDAGraph
```

这里刻意只保留主干输出 `hidden_states`，不把 logits 放进图里。原因是：

- vocab 维太大
- `compute_logits()` 放在 graph 外更灵活

---

## 5.3 在 `__init__()` 里先声明 graph 开关和图表容器

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 `ModelRunner.__init__()` 里 `self.sampler` 与 `self.kv_cache` 定义附近
- 操作：把下面这两行 graph 初始化代码插入到相同区域

在 `ModelRunner.__init__()` 里，`self.sampler` 和 `self.kv_cache` 附近新增：

```python
self.decode_graphs: dict[int, DecodeGraphRunner] = {}
self.use_cuda_graph = (
    torch.cuda.is_available()
    and self.device.type == "cuda"
    and not self.config.enforce_eager
)
```

这里不要立刻 capture，原因前面已经说过：

- `__init__()` 时 KV Cache 还没分配

所以真正的 capture 时机放到 `allocate_kv_cache()` 末尾。

---

## 5.4 在 `allocate_kv_cache()` 末尾触发图捕获

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 `allocate_kv_cache()` 的最后一个 `print(...)` 之后、方法返回前
- 操作：插入下面这段 capture 触发逻辑

在当前分配逻辑结束后补：

```python
if self.use_cuda_graph and not self.decode_graphs:
    self.capture_decode_graphs()
```

加 `not self.decode_graphs` 的原因很简单：

- `allocate_kv_cache()` 可能被重复调用
- 不要每次都重复 capture 一套图

---

## 5.5 新增 `capture_decode_graphs()`：先确定档位，再逐个 capture

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 `allocate_kv_cache()` 后、`run_model()` 前
- 操作：插入下面这份完整 `capture_decode_graphs()` 方法

教学版建议直接录制：

```python
capture_batch_sizes = list(range(1, self.config.max_cudagraph_batch_size + 1))
```

因为它最好理解。虽然不是最省显存，但对 0.6B 教学仓库足够清楚。

方法骨架建议写成：

```python
@torch.inference_mode()
def capture_decode_graphs(self) -> None:
    """
    输入：无；依赖已分配好的 self.kv_cache 与模型。
    输出：无；在 self.decode_graphs 里缓存各 batch size 的图。

    当前策略：
    - 只捕获 decode
    - 只捕获 exact-match 档位
    - 每个档位拥有独立静态 buffer
    """
    if not self.use_cuda_graph:
        return

    max_num_blocks = max(1, self.config.max_model_len // self.block_size + 1)
    hidden_size = self.model.config.hidden_size

    for batch_size in range(1, self.config.max_cudagraph_batch_size + 1):
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        block_tables = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32, device=self.device)
        hidden_states = torch.zeros(batch_size, hidden_size, dtype=self.config.torch_dtype, device=self.device)

        graph = torch.cuda.CUDAGraph()

        # 先做一次 warmup，让相关 kernel 与缓存状态稳定下来。
        set_context(
            Context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                max_context_len=0,
                max_num_blocks=max_num_blocks,
                kv_cache=self.kv_cache,
            )
        )
        hidden_states.copy_(self.model(input_ids, positions))
        torch.cuda.synchronize()

        with torch.cuda.graph(graph):
            set_context(
                Context(
                    is_prefill=False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                    max_context_len=0,
                    max_num_blocks=max_num_blocks,
                    kv_cache=self.kv_cache,
                )
            )
            hidden_states.copy_(self.model(input_ids, positions))

        reset_context()
        self.decode_graphs[batch_size] = DecodeGraphRunner(
            batch_size=batch_size,
            max_num_blocks=max_num_blocks,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            hidden_states=hidden_states,
            graph=graph,
        )
```

这里最重要的两个细节：

1. warmup 不是可有可无，它能减少 capture 时第一次运行带来的不稳定因素
2. capture 块内也要重新 `set_context(...)`，因为 graph 记录的是对这些静态张量的访问关系

---

## 5.6 在 `run_model()` 里加“命中图则 replay，否则 eager”的分发逻辑

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 `def run_model(...)`；如果这个方法是 Day4 新增的，就直接从方法签名开始整段替换为下面这份 graph-aware 版本

这一步建议把 `run_model()` 的开头改成：

```python
@torch.inference_mode()
def run_model(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_prefill: bool = False,
) -> torch.Tensor:
    """
    输入：
    - input_ids / positions: 当前 step 的输入
    - is_prefill: 当前是否为 prefill

    输出：logits

    分发规则：
    - prefill 永远 eager
    - decode 若命中已录制 batch size，则 replay
    - 否则 eager fallback
    """
    if (
        is_prefill
        or not self.use_cuda_graph
        or input_ids.size(0) not in self.decode_graphs
    ):
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)

    graph_runner = self.decode_graphs[input_ids.size(0)]
    live_context = get_context()

    graph_runner.input_ids.copy_(input_ids)
    graph_runner.positions.copy_(positions)
    graph_runner.slot_mapping.copy_(live_context.slot_mapping)
    graph_runner.context_lens.copy_(live_context.context_lens)
    graph_runner.block_tables.zero_()
    graph_runner.block_tables[:, : live_context.block_tables.size(1)].copy_(live_context.block_tables)

    set_context(
        Context(
            is_prefill=False,
            slot_mapping=graph_runner.slot_mapping,
            context_lens=graph_runner.context_lens,
            block_tables=graph_runner.block_tables,
            max_context_len=int(graph_runner.context_lens.max().item()),
            max_num_blocks=graph_runner.max_num_blocks,
            kv_cache=self.kv_cache,
        )
    )
    graph_runner.graph.replay()
    logits = self.model.compute_logits(graph_runner.hidden_states)
    return logits
```

然后在 `run()` 里把调用改成：

```python
logits = self.run_model(input_ids, positions, is_prefill=is_prefill)
```

这里的关键理解是：

- graph replay 使用的是静态 buffer
- 但每次 replay 前必须把“这次 decode 的真实输入与 context”拷进去

---

## 5.7 让 `run()` 继续负责 sampler，并在最后统一 `reset_context()`

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 锚点：定位到 `def run(...)` 的尾部 sampler 调用段
- 操作：把尾部替换成下面这 3 行，确保 graph 与 eager 都统一清理 Context

这一点故意保持和 `04` 一致，不要把 graph 逻辑扩散到 sampler。

`run()` 的尾部建议仍然保持：

```python
next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
reset_context()
return next_tokens.tolist()
```

不要因为 graph path 已经在 `run_model()` 里改过 context，就省掉这里的 reset。原因是：

- eager path 和 graph path 都会经过 `run()`
- 统一在外层 reset 更不容易漏

---

## 5.8 新增 `tests/test_Day6_cudagraph.py`，这里必须给完整文件

直接新建：

- `nano_vll_repro/tests/test_Day6_cudagraph.py`

完整代码如下。注释会比普通测试脚本更密，因为这份文件本身就是 Day6 行为边界的“可执行说明”。

```python
"""Day 6 CUDA Graph 测试脚本 - decode graph 行为验收

本文件只验证逻辑边界，不做真实性能评估。

它要锁住的行为有三类：

1. `enforce_eager=True` 时不会录制任何 decode graph。
2. `enforce_eager=False` 且 CUDA 可用时，会录制 `decode_graphs`。
3. `run_model()` 在命中图与未命中图时都能返回合法 logits。

注意：
1. 这份测试依赖 CUDA 与本地模型权重。
2. 这里测试的是“行为正确性”，不是“速度提升多少”。
3. 注释密度故意较高，便于后续回看时快速理解 graph 的边界。
"""

import os
import sys

import torch


# 让测试脚本可以直接导入项目模块。
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


# 下面这些导入覆盖了 Day6 真正关心的模块：
# 1. Config / ModelRunner：图开关、capture、replay 分发逻辑
# 2. Context / set_context / reset_context：decode replay 前的静态上下文准备
from config import Config
from engine.model_runner import ModelRunner
from utils.context import Context, reset_context, set_context


def _get_model_path() -> str:
    """返回仓库约定的本地模型路径。"""
    return os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B")


def _build_decode_context(runner: ModelRunner, batch_size: int) -> Context:
    """构造一份最小可运行的 decode Context。

    输入：
    - runner: 当前测试用的 ModelRunner
    - batch_size: 当前 decode batch 大小

    输出：
    - 一份足以驱动 decode attention 的最小 Context

    这里故意用最小值：
    - context_lens 全部设为 1
    - block_tables 只给 1 个 block
    - slot_mapping 从 0 开始递增

    这样做的目的不是模拟真实业务，而是尽量降低测试变量。
    """

    max_num_blocks = 1

    return Context(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        slot_mapping=torch.arange(batch_size, dtype=torch.long, device=runner.device),
        context_lens=torch.ones(batch_size, dtype=torch.int32, device=runner.device),
        block_tables=torch.zeros(batch_size, max_num_blocks, dtype=torch.int32, device=runner.device),
        max_context_len=1,
        max_num_blocks=max_num_blocks,
        kv_cache=runner.kv_cache,
    )


@torch.inference_mode()
def test_cudagraph_disabled_by_enforce_eager() -> None:
    """验证 eager 强制开关会关闭 CUDA Graph。"""

    model_path = _get_model_path()

    if not torch.cuda.is_available():
        print("skip: CUDA Graph 测试需要 CUDA")
        return

    if not os.path.isdir(model_path):
        print(f"skip: 模型路径不存在 {model_path}")
        return

    config = Config(
        model_path=model_path,
        enforce_eager=True,
        max_cudagraph_batch_size=2,
    )

    runner = ModelRunner(config)
    runner.allocate_kv_cache(2)

    # eager 强制打开时，runner 允许完全不启用 graph。
    assert runner.use_cuda_graph is False or runner.decode_graphs == {}

    print("✅ enforce_eager 关闭 CUDA Graph 的测试通过")


@torch.inference_mode()
def test_cudagraph_capture_decode_graphs() -> None:
    """验证 decode graph 会按配置档位生成。"""

    model_path = _get_model_path()

    if not torch.cuda.is_available():
        print("skip: CUDA Graph 测试需要 CUDA")
        return

    if not os.path.isdir(model_path):
        print(f"skip: 模型路径不存在 {model_path}")
        return

    config = Config(
        model_path=model_path,
        enforce_eager=False,
        max_cudagraph_batch_size=2,
    )

    runner = ModelRunner(config)
    runner.allocate_kv_cache(2)

    # 这里要求至少捕获 batch size 1 和 2 两个档位。
    assert 1 in runner.decode_graphs
    assert 2 in runner.decode_graphs

    print("✅ decode graph capture 测试通过")


@torch.inference_mode()
def test_cudagraph_exact_match_and_fallback() -> None:
    """验证命中图与未命中图的两条路径都可运行。"""

    model_path = _get_model_path()

    if not torch.cuda.is_available():
        print("skip: CUDA Graph 测试需要 CUDA")
        return

    if not os.path.isdir(model_path):
        print(f"skip: 模型路径不存在 {model_path}")
        return

    config = Config(
        model_path=model_path,
        enforce_eager=False,
        max_cudagraph_batch_size=2,
    )

    runner = ModelRunner(config)
    runner.allocate_kv_cache(2)

    # ===== 分支 A：batch size = 1，命中已录制图 =====
    input_ids = torch.zeros(1, dtype=torch.long, device=runner.device)
    positions = torch.zeros(1, dtype=torch.long, device=runner.device)

    set_context(_build_decode_context(runner, batch_size=1))
    logits_hit = runner.run_model(input_ids, positions, is_prefill=False)
    reset_context()

    assert logits_hit.shape[0] == 1
    assert logits_hit.shape[1] == runner.model.config.vocab_size

    # ===== 分支 B：batch size = 3，不命中图，应回退 eager =====
    input_ids = torch.zeros(3, dtype=torch.long, device=runner.device)
    positions = torch.arange(3, dtype=torch.long, device=runner.device)

    # 这里显式断言 batch size 3 不在图表里，避免测试失去意义。
    assert 3 not in runner.decode_graphs

    set_context(_build_decode_context(runner, batch_size=3))
    logits_fallback = runner.run_model(input_ids, positions, is_prefill=False)
    reset_context()

    assert logits_fallback.shape[0] == 3
    assert logits_fallback.shape[1] == runner.model.config.vocab_size

    print("✅ exact-match replay 与 eager fallback 测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 6 CUDA Graph 测试开始")
    print("=" * 60)

    test_cudagraph_disabled_by_enforce_eager()
    test_cudagraph_capture_decode_graphs()
    test_cudagraph_exact_match_and_fallback()

    print("=" * 60)
    print("🎉 Day 6 CUDA Graph 测试执行完成")
    print("=" * 60)
```

这份文件必须保证是“可直接复制为测试文件”的完整源码，不允许再退回到函数名占位。

---

## 6. 本篇结束后的最小验收

先做语法检查：

```bash
cd nano_vll_repro
python -m py_compile utils/context.py engine/model_runner.py
```

有 CUDA 时再跑：

```bash
python tests/test_Day6_cudagraph.py
```

另外建议手动再做两次真实验证：

1. `enforce_eager=False`、batch size 命中图
2. `enforce_eager=False`、batch size 超出 `max_cudagraph_batch_size`，确认会回退 eager

---

## 7. 常见错误

### 7.1 在 KV Cache 分配前 capture

后果：

- graph 里引用的 cache 对象不稳定
- 回放时行为不可预测

### 7.2 把 sampler 也塞进 graph

后果：

- 随机数与动态采样参数会让问题复杂度陡增
- 你会在一个基础版教学仓库里一次引入两类新变量

### 7.3 replay 前忘记把 live context 拷进静态 buffer

后果：

- graph 虽然 replay 成功
- 但 attention 读到的是上一次 batch 的 slot/block 信息

### 7.4 `run()` 末尾忘记 `reset_context()`

后果：

- eager 与 graph 路径互相污染
- 问题通常在第二次调用后才出现

---

## 8. 本篇真正学到的东西

这一篇最重要的不是“会写 `torch.cuda.CUDAGraph()`”，而是你要真正理解：

1. graph 记录的是“对静态 tensor 的访问图”，不是“自动帮你处理动态输入”。
2. decode 比 prefill 更适合先图化，因为输入 shape 稳定。
3. 基础版 graph 的价值在于把边界讲清楚，而不是一开始就追求最复杂的档位策略。

完成后进入最后一篇：

- [07-补齐Benchmark与Day7验收.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/07-补齐Benchmark与Day7验收.md)

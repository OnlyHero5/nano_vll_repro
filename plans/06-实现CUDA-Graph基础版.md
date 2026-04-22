# 06. 实现 CUDA Graph 基础版

## 1. 本篇目标

这一篇不是“把整个系统都图化”。

这一篇只做一件事：

> 给当前教学仓库加上一版单卡、decode-only、exact-match 命中的 CUDA Graph。

范围先写死：

1. 只支持 CUDA
2. 只图化 decode
3. 只做 batch size 精确命中
4. `tensor_parallel_size > 1` 时直接回 eager
5. sampler 不进 graph

本篇完成后，你应该至少得到下面 5 个结果：

1. `ModelRunner` 里有单独的 graph 资源容器。
2. graph 的 capture 时机放在 KV Cache 分配之后。
3. 命中图时 replay，没命中时 eager。
4. `run()` 不管 eager 还是 replay，结束后都统一 `reset_context()`。
5. 仓库里新增一个 `tests/test_Day6_cudagraph.py`，锁住这套行为边界。

---

## 2. 前置条件

如果你准备照本文改代码，默认前面已经做到：

1. `04` 已经有 `run_model()` 边界
2. `04` 的 `run()` 已经把 `sampler` 放到主干前向外面
3. `05` 的 TP fallback 逻辑已经存在

如果这些边界还没建好，这一篇先不要直接抄代码。

---

## 3. 先看当前仓库的真实状态

### 3.1 `utils/context.py`

当前 [utils/context.py](/home/psx/nano_vllm_repro/nano_vll_repro/utils/context.py:24) 已经有 `Context`、`set_context()`、`get_context()`、`reset_context()`。

所以本篇真正缺的不是“上下文机制不存在”。

真正缺的是：

1. 哪些 graph 静态 buffer 要挂在哪
2. capture / replay 时怎么重建 decode `Context`
3. 为什么 `reset_context()` 必须变成强纪律

### 3.2 `engine/model_runner.py`

当前 [engine/model_runner.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/model_runner.py:73) 还没有任何 graph 相关状态：

1. 没有 decode graph 资源结构体
2. 没有 graph 开关
3. 没有 capture 入口
4. 没有 replay 分发逻辑

---

## 4. 本篇修改原则

### 4.1 capture 必须放在 KV Cache 分配之后

原因很直接。

decode graph 里的 attention 会直接读写真实 KV Cache。

如果 graph 在 cache 分配前就 capture，图里引用的对象根本不稳定。

### 4.2 sampler 不进 graph

sampler 暂时放在 graph 外，原因有两个：

1. 它有随机性
2. 它的输入参数是 per-sequence 动态变化的

这一篇真正要讲透的是：

> 主干 replay，而不是所有逻辑一起图化。

### 4.3 基础版只做 exact-match

现在不要急着上：

- padding 到更大 batch 档位
- graph 淘汰策略
- TP + graph 联动

基础版先把最小闭环讲清楚就够了。

---

## 5. 逐步修改

## 5.1 在 `engine/model_runner.py` 顶部新增 `DecodeGraphRunner`

修改位置：

- 文件：`engine/model_runner.py`
- 锚点：import 区之后、`class ModelRunner` 之前

完整新增代码如下：

```python
from dataclasses import dataclass


@dataclass
class DecodeGraphRunner:
    """
    单个 batch size 档位的一组静态 CUDA Graph 资源。
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

这里把静态 buffer 集中收进一个结构体，是为了避免后面 graph 状态散成一地 `self.xxx`。

---

## 5.2 在 `ModelRunner.__init__()` 里补 graph 状态

修改位置：

- 文件：`engine/model_runner.py`
- 锚点：`self.sampler` 和 `self.kv_cache` 附近

完整新增代码如下：

```python
# ===== CUDA Graph 相关状态 =====
self.decode_graphs: dict[int, DecodeGraphRunner] = {}
self.use_cuda_graph = (
    torch.cuda.is_available()
    and self.device.type == "cuda"
    and not self.config.enforce_eager
    and self.config.tensor_parallel_size == 1
)
```

然后在 `allocate_kv_cache()` 末尾追加：

```python
if self.use_cuda_graph:
    self.capture_decode_graphs()
```

这样 graph capture 的时机就固定在：

> 模型已经加载，KV Cache 已经分配，decode 依赖的静态对象已经稳定下来之后。

---

## 5.3 新增 `capture_decode_graphs()`

修改位置：

- 文件：`engine/model_runner.py`
- 锚点：插在 `allocate_kv_cache()` 后面

完整新增代码如下：

```python
@torch.inference_mode()
def capture_decode_graphs(self) -> None:
    """
    预先录制一组 decode batch size 档位的 CUDA Graph。
    当前基础版只录 1/2/4/8/16 这些小档位。
    """
    capture_batch_sizes = [1, 2, 4, 8, 16]
    max_num_blocks = max(1, self.config.max_model_len // self.block_size + 1)

    for batch_size in capture_batch_sizes:
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        context_lens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        block_tables = torch.zeros(
            batch_size,
            max_num_blocks,
            dtype=torch.int32,
            device=self.device,
        )

        # 先用一份静态 decode Context 预热，确保 graph 捕获时依赖对象已经就位。
        context = Context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            max_context_len=1,
            max_num_blocks=max_num_blocks,
            kv_cache=self.kv_cache,
        )
        set_context(context)

        # warmup
        hidden_states = self.model(input_ids, positions)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden_states = self.model(input_ids, positions)

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

        reset_context()
```

注意这里的核心点只有两个：

1. 录的是 **decode 主干**
2. 录完一个档位就 `reset_context()`

---

## 5.4 用下面这份完整 `run_model()` 替换当前实现

修改位置：

- 文件：`engine/model_runner.py`
- 锚点：`def run_model(...)`

完整替代代码如下：

```python
@torch.inference_mode()
def run_model(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_prefill: bool,
) -> torch.Tensor:
    """
    graph-aware 的模型执行入口。
    Prefill 直接 eager。
    Decode 命中 graph 档位时 replay，否则 eager。
    """
    if is_prefill or not self.use_cuda_graph:
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)

    batch_size = input_ids.shape[0]
    runner = self.decode_graphs.get(batch_size)
    if runner is None:
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)

    context = get_context()

    runner.input_ids.copy_(input_ids)
    runner.positions.copy_(positions)
    runner.slot_mapping.zero_()
    runner.slot_mapping[:batch_size].copy_(context.slot_mapping)
    runner.context_lens.zero_()
    runner.context_lens[:batch_size].copy_(context.context_lens)
    runner.block_tables.zero_()
    runner.block_tables[:batch_size, : context.block_tables.shape[1]].copy_(context.block_tables)

    replay_context = Context(
        is_prefill=False,
        slot_mapping=runner.slot_mapping,
        context_lens=runner.context_lens,
        block_tables=runner.block_tables,
        max_context_len=int(runner.context_lens.max().item()),
        max_num_blocks=runner.max_num_blocks,
        kv_cache=self.kv_cache,
    )
    set_context(replay_context)
    runner.graph.replay()
    return self.model.compute_logits(runner.hidden_states)
```

这段代码要记住一条总原则：

> graph replay 前，必须把静态 buffer 全部写成当前 batch 的真实值，再重建当前 decode `Context`。

---

## 5.5 `run()` 继续只负责 sampler 和 `reset_context()`

如果你已经按 `04` 改过 `run()`，这里不要再把 graph 分支塞进去。

`run()` 仍然维持下面这份完整实现：

```python
@torch.inference_mode()
def run(
    self,
    sequences: list[Sequence],
    is_prefill: bool,
) -> list[int]:
    if not sequences:
        return []

    if is_prefill:
        input_ids, positions = self.prepare_prefill(sequences)
    else:
        input_ids, positions = self.prepare_decode(sequences)

    temperatures, top_ks, top_ps = self.prepare_sampling_tensors(sequences)

    try:
        logits = self.run_model(input_ids, positions, is_prefill)
        next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
        return next_tokens.tolist()
    finally:
        reset_context()
```

不要把 `reset_context()` 从这里删掉。

因为 eager 和 replay 两条路径最后都要靠它收尾。

---

## 5.6 新建 `tests/test_Day6_cudagraph.py`

新建文件：

- `tests/test_Day6_cudagraph.py`

完整代码如下：

```python
"""Day 6 CUDA Graph 测试"""

import sys
sys.path.insert(0, ".")

import torch

from utils.context import Context, get_context, reset_context, set_context


def test_context_reset_is_hard_boundary():
    set_context(Context(is_prefill=False, max_num_blocks=8))
    assert get_context().max_num_blocks == 8
    reset_context()
    assert get_context().max_num_blocks is None


def test_decode_graph_runner_dataclass_exists():
    from engine.model_runner import DecodeGraphRunner
    assert DecodeGraphRunner is not None


@torch.inference_mode()
def test_graph_is_disabled_on_cpu_or_enforce_eager():
    from config import Config
    from engine.model_runner import ModelRunner

    config = Config(model_path="models/Qwen3-0.6B", enforce_eager=True)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.device = torch.device("cpu")
    runner.decode_graphs = {}
    runner.kv_cache = None
    runner.use_cuda_graph = (
        torch.cuda.is_available()
        and runner.device.type == "cuda"
        and not runner.config.enforce_eager
        and runner.config.tensor_parallel_size == 1
    )
    assert runner.use_cuda_graph is False


if __name__ == "__main__":
    test_context_reset_is_hard_boundary()
    test_decode_graph_runner_dataclass_exists()
    test_graph_is_disabled_on_cpu_or_enforce_eager()
    print("🎉 Day 6 CUDA Graph 测试执行完成")
```

这份测试仍然刻意保持轻量。

它先锁边界，不先锁性能。

---

## 6. 本篇结束后的最小验收

```bash
cd nano_vll_repro
python -m py_compile engine/model_runner.py utils/context.py tests/test_Day6_cudagraph.py
python tests/test_Day6_cudagraph.py
```

如果你有 CUDA 环境，并且前面 `04` 已经真的落地，再补一个手动 smoke test：

```bash
python example.py
```

然后看 decode 小 batch 是否能命中 graph 档位。

---

## 7. 常见错误

### 7.1 在 KV Cache 分配前 capture

这是 graph 最容易埋雷的地方之一。

### 7.2 把 sampler 也塞进 graph

基础版没必要，复杂度会直接上升。

### 7.3 graph replay 前忘了重写静态 buffer

这样图虽然 replay 了，但吃到的是上一个 batch 的数据。

### 7.4 eager 和 replay 走了两套不同的 `Context` 协议

后面 debug 会非常痛苦。

### 7.5 以为 graph 目标是“所有 batch 都命中”

基础版先追求可解释，不追求覆盖所有动态形状。

---

## 8. 本篇真正学到的东西

CUDA Graph 这一篇最重要的不是 `torch.cuda.CUDAGraph` 这个类名。

而是下面 4 句话：

1. 先把主干边界拆出来，再做 graph。
2. 先做 decode-only，再谈更复杂档位。
3. graph 命中靠静态 buffer，不靠魔法。
4. eager fallback 不是失败，而是基础版设计的一部分。

# 06. 实现 CUDA Graph 基础版（先做单卡 decode-only graph，再谈更复杂档位）

## 1. 本篇目标

这一篇的任务不是“把整个系统都图化”，而是先给你当前这份教学仓库加上一版：

1. 边界清晰
2. 行为可解释
3. 测试可验收
4. 不和前面单卡主循环打架

的 decode-only CUDA Graph。

本篇完成后，目标行为应当是：

1. 只在 CUDA 环境下启用
2. 只在 `enforce_eager=False` 时启用
3. 当前基础版只支持单卡 graph，不和 TP 交叉
4. 只图化 decode 路径
5. 只有 batch size 精确命中已录制档位时才 replay
6. 其他情况全部回退 eager

这里先把范围写死：

> 本篇先不做“TP + CUDA Graph 联动”。也就是 `tensor_parallel_size > 1` 时，当前基础版直接退回 eager。这样你一次只面对一类新复杂度，而不是把通信问题和 graph 问题绑在一起调。

---

## 2. 权威参考

本篇对照下面 3 组来源：

1. 当前仓库：
   - `nano_vll_repro/utils/context.py`
   - `nano_vll_repro/engine/model_runner.py`
2. 上游主仓库：
   - `https://github.com/GeeeekExplorer/nano-vllm`
   - `nanovllm/engine/model_runner.py`
   - `nanovllm/utils/context.py`
3. 你前面已经补齐的本地前提：
   - [02-补齐Qwen3模型主干与权重映射.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/02-补齐Qwen3模型主干与权重映射.md)
   - [04-补齐单卡推理链路与Day5测试.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/04-补齐单卡推理链路与Day5测试.md)
   - [05-实现Tensor-Parallel基础版.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/05-实现Tensor-Parallel基础版.md)

这里有一个前提一定要说清楚：

> 本篇默认你已经按 `04` 把 `run_model()`、`compute_logits()`、`reset_context()` 这些运行时边界收口好了。没有这些边界，graph capture 会直接变成一团混在一起的大函数。

---

## 3. 先看当前仓库的真实缺口

### 3.1 `utils/context.py`

当前 [utils/context.py](/home/psx/nano_vllm_repro/nano_vll_repro/utils/context.py:24) 已经有：

1. `Context` dataclass
2. `set_context()`
3. `get_context()`
4. `reset_context()`

所以这一步真正缺的不是“上下文机制不存在”，而是：

1. 图录制和图回放时，哪些字段必须是静态 buffer
2. 为什么每次 `run()` 末尾仍然必须统一 `reset_context()`
3. 为什么 `Context` 要继续保持成纯数据容器，而不是把 graph 状态塞进去

### 3.2 `engine/model_runner.py`

当前如果你已经按 04 收口，`ModelRunner` 应该具备：

1. `prepare_prefill()`
2. `prepare_decode()`
3. `run_model()`
4. `run()`

但它还缺少 5 个核心部件：

1. decode graph 资源容器
2. graph 开关与图表缓存
3. capture 入口
4. replay 入口
5. “命中图则 replay，否则 eager” 的统一分发逻辑

---

## 4. 本篇修改原则

### 4.1 capture 时机必须放在 KV Cache 分配之后

原因非常实际：

1. decode 图里 attention 会访问真实 KV Cache
2. 如果先 capture、后分配 cache，图里引用的对象就不稳定

### 4.2 sampler 不进 graph

这一步只 capture：

1. `model(input_ids, positions)` 主干
2. 对应的 decode `Context`

采样仍然在 graph 外，原因很简单：

1. sampler 本身包含随机数和动态采样参数
2. 当前教学仓库真正要讲清楚的是“主干 replay”，不是“把所有东西都图化”

### 4.3 先做 exact-match，不做 padding 命中更大档位

基础版策略统一为：

1. batch size 精确命中某个档位时 replay
2. 否则 eager

现在不要提前引入：

1. padding 到更大 batch
2. 图档位合并
3. 图缓存淘汰策略

### 4.4 当前基础版先只支持单卡 graph

这里刻意再强调一次：

1. `tensor_parallel_size > 1` 时，当前基础版直接禁用 graph
2. 这是为了先把 decode-only graph 的数据流讲清楚
3. TP 与 graph 的交叉优化可以以后单开一篇

---

## 5. 逐步修改

## 5.1 保持 `Context` 为纯数据容器，并补清楚 reset 语义

修改位置：

- 文件：`nano_vll_repro/utils/context.py`
- 操作：不改字段结构，只改类注释和 `reset_context()` 注释

推荐替代注释如下：

```python
@dataclass
class Context:
    """
    全局推理上下文。

    输入：
    - 由 ModelRunner 在 prepare_prefill / prepare_decode / graph replay 前设置。

    输出：
    - 由 Attention 层在 forward 中读取。

    设计边界：
    1. 这里只保存“当前 step 已经整理好的静态元数据”。
    2. 这里不做任何计算，也不持有 graph 状态对象。
    3. 保持成纯数据容器，是为了让 eager 路径和 graph 路径共享同一套读取协议。
    """
```

`reset_context()` 的注释建议改成：

```python
def reset_context():
    """
    重置全局 Context。

    这一步必须在两个场景统一调用：
    1. 每次 `run()` 结束后
    2. 每次 graph capture 完成后

    原因：
    - Context 是“当前 step 的元数据快照”
    - 如果不在 step 边界清空，下一轮 eager / replay 都可能读到上一个 batch 的旧数据
    """

    global _current_context
    _current_context = Context()
```

这里真正重要的不是文案，而是你后面排查 graph 污染问题时，会反复依赖这条纪律。

---

## 5.2 在 `ModelRunner` 顶部新增 `DecodeGraphRunner` 容器

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：在 import 区之后、`class ModelRunner` 之前插入下面这份 dataclass

完整新增代码如下：

```python
from dataclasses import dataclass


@dataclass
class DecodeGraphRunner:
    """
    单个 decode batch size 档位对应的一组静态 graph 资源。

    这里把所有静态 buffer 收进一个结构体，原因有两个：
    1. 后面你会同时持有多个 batch size 的图
    2. 如果把这些资源都散成 `self.xxx`，状态很快就会失控

    字段说明：
    - batch_size: 这个 graph 档位对应的 decode batch 大小
    - max_num_blocks: 当前静态 block table 的列数上限
    - input_ids / positions: replay 前每次都会写入最新 decode 输入
    - slot_mapping / context_lens / block_tables: replay 前每次都会写入最新 decode Context
    - hidden_states: graph replay 后产出的主干输出
    - graph: 当前档位对应的 CUDA Graph 实例
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

这里刻意只缓存 `hidden_states`，不缓存 logits。原因是：

1. `compute_logits()` 放在 graph 外更灵活
2. vocab 维通常很大，基础版没必要急着把 lm head 也塞进去

---

## 5.3 在 `ModelRunner.__init__()` 里补 graph 开关和图表容器

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：在 `self.sampler` 与 `self.kv_cache` 附近插入下面这段

完整新增代码如下：

```python
# ===== CUDA Graph 相关状态 =====
# decode_graphs 按 batch size 存图，key 就是 exact-match 档位。
self.decode_graphs: dict[int, DecodeGraphRunner] = {}

# 当前基础版只在这些条件都满足时才启用 graph：
# 1. CUDA 可用
# 2. 当前 device 确实是 CUDA
# 3. 用户没有强制 eager
# 4. 当前不是 TP 模式（教学版先不处理 TP + graph 交叉复杂度）
self.use_cuda_graph = (
    torch.cuda.is_available()
    and self.device.type == "cuda"
    and not self.config.enforce_eager
    and self.config.tensor_parallel_size == 1
)
```

这里不要在 `__init__()` 里立刻 capture。原因前面已经说过：

1. `__init__()` 时 KV Cache 还没分配
2. graph capture 必须建立在最终的 cache 对象之上

---

## 5.4 在 `allocate_kv_cache()` 末尾触发图捕获

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：在 `allocate_kv_cache()` 最后一个 `print(...)` 之后插入下面这段

完整新增代码如下：

```python
# KV Cache 分配完成之后，graph 所依赖的静态 cache 引用才真正稳定。
# 因此 capture 触发时机必须放在这里，而不是更早。
if self.use_cuda_graph and not self.decode_graphs:
    self.capture_decode_graphs()
```

这里保留 `not self.decode_graphs` 是为了避免：

1. 重复调用 `allocate_kv_cache()`
2. 每次都把整套 decode 图重新录一遍

---

## 5.5 新增 `capture_decode_graphs()`

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：在 `allocate_kv_cache()` 后、`run_model()` 前插入下面这份完整方法

完整新增代码如下：

```python
@torch.inference_mode()
def capture_decode_graphs(self) -> None:
    """
    录制当前基础版 decode graphs。

    输入：
    - 无；依赖已经稳定存在的 `self.model`、`self.kv_cache` 和 `self.device`

    输出：
    - 无；把不同 batch size 档位的图写入 `self.decode_graphs`

    当前策略固定为：
    1. 只录 decode
    2. 只录 exact-match 档位
    3. batch size 从 1 录到 `max_cudagraph_batch_size`
    """

    # graph 开关没打开时，这个方法应该是空操作。
    if not self.use_cuda_graph:
        return

    # block table 静态 buffer 的列数上限按最大上下文长度推。
    max_num_blocks = max(1, self.config.max_model_len // self.block_size + 1)
    hidden_size = self.model.config.hidden_size

    for batch_size in range(1, self.config.max_cudagraph_batch_size + 1):
        # ===== 为当前档位创建静态输入 buffer =====
        # 这些张量会在每次 replay 前被最新 decode 数据覆盖。
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        context_lens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        block_tables = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32, device=self.device)

        # hidden_states 是 graph 的输出 buffer。
        hidden_states = torch.zeros(
            batch_size,
            hidden_size,
            dtype=self.config.torch_dtype,
            device=self.device,
        )

        graph = torch.cuda.CUDAGraph()

        # ===== warmup =====
        # 先用一轮普通 eager 让相关 kernel 和 cache 状态稳定下来。
        set_context(
            Context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                max_context_len=1,
                max_num_blocks=max_num_blocks,
                kv_cache=self.kv_cache,
            )
        )
        hidden_states.copy_(self.model(input_ids, positions))
        torch.cuda.synchronize()

        # ===== 正式 capture =====
        # capture 块内部也要重新 set_context，
        # 因为 graph 记录的是对这些静态 tensor 的访问关系。
        with torch.cuda.graph(graph):
            set_context(
                Context(
                    is_prefill=False,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=0,
                    max_seqlen_k=0,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                    max_context_len=1,
                    max_num_blocks=max_num_blocks,
                    kv_cache=self.kv_cache,
                )
            )
            hidden_states.copy_(self.model(input_ids, positions))

        # capture 完一个档位之后，立即清空全局 Context，
        # 避免下一个档位沿用上一个档位的静态对象引用。
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

这里最重要的两个细节是：

1. warmup 不是可有可无，它能减少第一次 capture 的不稳定因素。
2. capture 块内也必须重新 `set_context(...)`，因为 graph 记录的是静态张量访问图，而不是抽象逻辑。

---

## 5.6 让 `run_model()` 变成 graph-aware 分发入口

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：把 `run_model()` 整个方法替换为下面这份完整实现

完整替代代码如下：

```python
@torch.inference_mode()
def run_model(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_prefill: bool = False,
) -> torch.Tensor:
    """
    运行模型主干并返回 logits。

    输入：
    1. input_ids
    2. positions
    3. is_prefill

    输出：
    - vocab 维 logits

    分发规则：
    1. prefill 永远 eager
    2. graph 开关关闭时 eager
    3. batch size 没命中图档位时 eager
    4. 其他情况 replay
    """

    # ===== eager 路径 =====
    if (
        is_prefill
        or not self.use_cuda_graph
        or input_ids.size(0) not in self.decode_graphs
    ):
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)

    # ===== graph replay 路径 =====
    graph_runner = self.decode_graphs[input_ids.size(0)]
    live_context = get_context()

    # replay 前把这次 decode 的真实输入写进静态 buffer。
    graph_runner.input_ids.copy_(input_ids)
    graph_runner.positions.copy_(positions)
    graph_runner.slot_mapping.copy_(live_context.slot_mapping)
    graph_runner.context_lens.copy_(live_context.context_lens)

    # block_tables 的列数是静态上限，live block table 可能更短。
    # 因此先清零，再只覆盖真实有效区间。
    graph_runner.block_tables.zero_()
    graph_runner.block_tables[:, : live_context.block_tables.size(1)].copy_(live_context.block_tables)

    # replay 前，把全局 Context 切到“引用静态 buffer 的版本”。
    set_context(
        Context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=graph_runner.slot_mapping,
            context_lens=graph_runner.context_lens,
            block_tables=graph_runner.block_tables,
            max_context_len=int(graph_runner.context_lens.max().item()),
            max_num_blocks=graph_runner.max_num_blocks,
            kv_cache=self.kv_cache,
        )
    )

    # graph replay 后，hidden_states 会写到静态输出 buffer。
    graph_runner.graph.replay()

    # logits 仍然放在 graph 外计算，保持基础版边界简单清楚。
    logits = self.model.compute_logits(graph_runner.hidden_states)
    return logits
```

这一步真正要理解的是：

1. graph replay 用的是静态 buffer
2. 但每次 replay 前，必须先把“这次 decode 的真实输入与 context”拷进去

---

## 5.7 `run()` 仍然只负责 sampler 和统一 `reset_context()`

修改位置：

- 文件：`nano_vll_repro/engine/model_runner.py`
- 操作：只改 `run()` 里 `run_model(...)` 的调用签名，并保持尾部 reset 纪律

你在 `run()` 里应当保持下面这种结构：

```python
logits = self.run_model(input_ids, positions, is_prefill=is_prefill)

if is_prefill:
    context = get_context()
    last_token_indices = context.cu_seqlens_q[1:] - 1
    logits = logits[last_token_indices.long()]

next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)

# 这条 reset 纪律不能删。
# 原因是 eager 和 replay 两条路径最终都会走到 run() 外层。
reset_context()
return next_tokens.tolist()
```

这里不要把 sampler 或 reset 逻辑塞回 `run_model()`，否则：

1. graph 分发逻辑会和采样逻辑重新耦合
2. eager / replay 的清理边界会再次变乱

---

## 5.8 新增 `tests/test_Day6_cudagraph.py`

直接新建：

- `nano_vll_repro/tests/test_Day6_cudagraph.py`

完整代码如下。依旧保持“测试文件也是行为说明书”的写法。

```python
"""Day 6 CUDA Graph 测试脚本 - decode graph 行为验收

这份文件只验证行为边界，不做真实性能评估。

要锁住的行为有三类：
1. `enforce_eager=True` 时不会录制任何 decode graph
2. `enforce_eager=False` 且单卡 CUDA 环境可用时，会录制 decode graph
3. `run_model()` 在命中图与未命中图时都能返回合法 logits
"""

import os
import sys

import torch


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


from config import Config
from engine.model_runner import ModelRunner
from utils.context import Context, reset_context, set_context


def get_model_path() -> str:
    """返回仓库约定的本地模型路径。"""
    return os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B")


def build_decode_context(runner: ModelRunner, batch_size: int) -> Context:
    """
    构造一份最小可运行的 decode Context。

    这里故意把内容压到最小：
    - context_lens 全部设为 1
    - block_tables 只给 1 列
    - slot_mapping 从 0 开始递增

    这样测试的重点就只剩：
    - replay 路径是否会读取正确的静态 buffer
    """

    return Context(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        slot_mapping=torch.arange(batch_size, dtype=torch.long, device=runner.device),
        context_lens=torch.ones(batch_size, dtype=torch.int32, device=runner.device),
        block_tables=torch.zeros(batch_size, 1, dtype=torch.int32, device=runner.device),
        max_context_len=1,
        max_num_blocks=1,
        kv_cache=runner.kv_cache,
    )


@torch.inference_mode()
def test_cudagraph_disabled_by_enforce_eager() -> None:
    """验证 eager 强制开关会关闭 graph。"""

    model_path = get_model_path()

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

    # eager 强制打开时，decode_graphs 应保持为空。
    assert runner.use_cuda_graph is False
    assert runner.decode_graphs == {}

    print("✅ enforce_eager 关闭 CUDA Graph 的测试通过")


@torch.inference_mode()
def test_cudagraph_capture_decode_graphs() -> None:
    """验证 decode graph 会按配置录制档位。"""

    model_path = get_model_path()

    if not torch.cuda.is_available():
        print("skip: CUDA Graph 测试需要 CUDA")
        return

    if not os.path.isdir(model_path):
        print(f"skip: 模型路径不存在 {model_path}")
        return

    config = Config(
        model_path=model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_cudagraph_batch_size=2,
    )

    runner = ModelRunner(config)
    runner.allocate_kv_cache(2)

    # 当前基础版至少要录 batch size 1 和 2 两个档位。
    assert 1 in runner.decode_graphs
    assert 2 in runner.decode_graphs

    print("✅ decode graph capture 测试通过")


@torch.inference_mode()
def test_cudagraph_exact_match_and_fallback() -> None:
    """验证命中图与未命中图两条路径都可运行。"""

    model_path = get_model_path()

    if not torch.cuda.is_available():
        print("skip: CUDA Graph 测试需要 CUDA")
        return

    if not os.path.isdir(model_path):
        print(f"skip: 模型路径不存在 {model_path}")
        return

    config = Config(
        model_path=model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_cudagraph_batch_size=2,
    )

    runner = ModelRunner(config)
    runner.allocate_kv_cache(2)

    # ===== 分支 A：batch size = 1，命中已录制图 =====
    input_ids = torch.zeros(1, dtype=torch.long, device=runner.device)
    positions = torch.zeros(1, dtype=torch.long, device=runner.device)

    set_context(build_decode_context(runner, batch_size=1))
    logits_hit = runner.run_model(input_ids, positions, is_prefill=False)
    reset_context()

    assert logits_hit.shape[0] == 1
    assert logits_hit.shape[1] == runner.model.config.vocab_size

    # ===== 分支 B：batch size = 3，不命中图，应回退 eager =====
    input_ids = torch.zeros(3, dtype=torch.long, device=runner.device)
    positions = torch.arange(3, dtype=torch.long, device=runner.device)

    assert 3 not in runner.decode_graphs

    set_context(build_decode_context(runner, batch_size=3))
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

1. `enforce_eager=False` 且 batch size 命中图
2. `enforce_eager=False` 且 batch size 超出 `max_cudagraph_batch_size`，确认会回退 eager

---

## 7. 常见错误

### 7.1 在 KV Cache 分配前 capture

后果：

- graph 里引用的 cache 对象不稳定
- 回放时行为不可预测

### 7.2 把 sampler 也塞进 graph

后果：

- 采样随机性和动态参数会把基础版复杂度直接拉高
- 你会失去对“主干 replay 边界”的清晰认识

### 7.3 replay 前忘记把 live context 写进静态 buffer

后果：

- graph 虽然 replay 成功
- 但 attention 读到的是上一次 batch 的 slot / block 信息

### 7.4 TP 模式下也硬开当前基础版 graph

后果：

- 你会同时面对通信与 graph 两类问题
- 诊断成本远高于教学收益

### 7.5 `run()` 末尾忘记 `reset_context()`

后果：

- eager 和 replay 两条路径会互相污染
- 问题通常在第二个 step 才出现，更难查

---

## 8. 本篇真正学到的东西

这一篇最重要的不是“会写 `torch.cuda.CUDAGraph()`”，而是下面 4 个边界：

1. graph 记录的是“对静态张量的访问图”，不是自动处理动态输入。
2. decode 比 prefill 更适合先图化，因为输入 shape 更稳定。
3. sampler 保持在 graph 外，可以显著降低基础版复杂度。
4. TP 和 graph 都重要，但教学仓库最好一次只引入一类新变量。

完成后进入最后一篇：

- [07-补齐Benchmark与Day7验收.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/07-补齐Benchmark与Day7验收.md)

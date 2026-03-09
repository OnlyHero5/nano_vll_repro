# 06. 实现 CUDA Graph 基础版

## 1. 学习目标

这一篇的目标是把 Decode 阶段接上基础版 CUDA Graph。

你要完成的不是“把所有路径都图化”，而是做一版边界明确、行为稳定的教学版实现：

- 只图化 decode
- 只在 CUDA 上启用
- 只在 `enforce_eager=False` 时启用
- batch size 命中已录制档位才 replay，否则回退 eager

## 2. 先修知识

## 2.1 CUDA Graph 到底解决什么问题

Decode 阶段的特点是：

- 每步输入 shape 很小
- 但调用次数很多

这时瓶颈不一定是矩阵乘法本身，而可能是：

- Python 调度开销
- kernel launch 开销
- 框架层重复组图开销

CUDA Graph 的核心思路是：

1. 先把一段固定 shape 的 GPU 计算录下来
2. 之后只改输入 tensor 内容
3. 直接 replay，不再重复发射那一串 kernel

## 2.2 为什么 Prefill 不适合先做 Graph

Prefill 每次长度差异太大，batch 也很不稳定，shape 很难固定。

而 decode 的 shape 规律非常强：

- 每条序列一次只输入一个 token
- `input_ids.shape == [batch_size]`
- `positions.shape == [batch_size]`

所以 graph 先做 decode 是最合理的。

## 2.3 这篇为什么只做“基础版”

生产级实现通常会：

- 维护一组捕获好的 batch size 档位
- 对小 batch 做 padding 命中更大图
- 处理更多 fallback 分支

这篇为了教学清晰，采用更容易手敲和理解的版本：

- 录制 `1 ~ max_cudagraph_batch_size` 的每个 batch size
- 仅 exact match replay
- 其余一律 eager fallback

## 3. 本仓库当前缺口

当前仓库虽然已经有：

- `Config.enforce_eager`
- `utils/context.reset_context()`

但还没有真正的：

- graph capture
- graph replay
- decode 静态 buffer
- graph 命中与 eager fallback 分发逻辑

## 4. 最终应修改的文件

- `engine/model_runner.py`
- `utils/context.py`
- `tests/test_Day6_cudagraph.py`

## 5. 完整代码

### 5.1 替换 `utils/context.py`

```python
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Context:
    is_prefill: bool = False

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int | None = None
    max_num_blocks: int | None = None

    kv_cache: Optional[list[torch.Tensor]] = None


_current_context = Context()


def get_context() -> Context:
    return _current_context


def set_context(context: Context) -> None:
    global _current_context
    _current_context = context


def clear_context() -> None:
    reset_context()


def reset_context() -> None:
    global _current_context
    _current_context = Context()
```

### 5.2 替换 `engine/model_runner.py`

```python
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from config import Config
from engine.sequence import Sequence
from layers.linear import divide, get_tp_rank, get_tp_world_size
from layers.sampler import Sampler
from utils.context import Context, get_context, reset_context, set_context
from utils.loader import load_model


@dataclass
class DecodeGraphRunner:
    batch_size: int
    max_num_blocks: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor
    next_tokens: torch.Tensor


class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        self.rank = 0
        self.local_rank = 0
        self.tp_size = config.tensor_parallel_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_distributed_env()

        self.model_dtype = config.torch_dtype
        self.kv_cache_dtype = config.kv_torch_dtype

        Sequence.block_size = config.kvcache_block_size
        self.block_size = Sequence.block_size

        self.model = self._load_model()
        self.sampler = Sampler()
        self.kv_cache: Optional[list[torch.Tensor]] = None

        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.num_layers = self.model.config.num_hidden_layers
        self.head_dim = getattr(
            self.model.config,
            "head_dim",
            self.model.config.hidden_size // self.model.config.num_attention_heads,
        )
        self.num_kv_heads = divide(self.model.config.num_key_value_heads, self.tp_size)

        self.decode_graphs: dict[int, DecodeGraphRunner] = {}
        self.use_cuda_graph = (
            torch.cuda.is_available()
            and not self.config.enforce_eager
            and self.device.type == "cuda"
        )

    def init_distributed_env(self) -> None:
        if self.config.tensor_parallel_size == 1:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("Tensor Parallelism 需要 CUDA 环境")

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        assert world_size == self.config.tensor_parallel_size
        torch.cuda.set_device(local_rank)

        self.rank = rank
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _load_model(self) -> nn.Module:
        from models.qwen3 import Qwen3ForCausalLM

        if self.is_main_process:
            print(f"[ModelRunner] 加载模型：{self.config.model_path}")

        model = Qwen3ForCausalLM.from_pretrained(self.config.model_path)
        load_model(model, self.config.model_path)
        model = model.to(self.device, dtype=self.model_dtype)
        model.eval()
        return model

    def get_num_free_gpu_blocks(self) -> int:
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks

        if not torch.cuda.is_available():
            return 128

        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device)
        free_memory = total_memory - allocated_memory
        available_memory = int(free_memory * self.config.gpu_memory_utilization)

        bytes_per_block_per_layer = (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.kv_cache_dtype).element_size()
        )
        bytes_per_block = bytes_per_block_per_layer * self.num_layers
        return max(1, available_memory // bytes_per_block)

    def allocate_kv_cache(self, num_blocks: int) -> None:
        self.kv_cache = []
        for _ in range(self.num_layers):
            cache = torch.zeros(
                2,
                num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.kv_cache_dtype,
                device=self.device,
            )
            self.kv_cache.append(cache)

        if self.is_main_process:
            print(f"[ModelRunner] KV Cache 分配完成：{num_blocks} blocks x {self.num_layers} layers")

        if self.use_cuda_graph:
            self.capture_decode_graphs()

    def prepare_prefill(self, sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        all_token_ids = []
        all_positions = []
        cu_seqlens = [0]
        slot_mapping = []

        for seq in sequences:
            seq_len = len(seq.token_ids)
            all_token_ids.extend(seq.token_ids)
            all_positions.extend(range(seq_len))
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

            for i in range(seq_len):
                block_idx = i // self.block_size
                offset = i % self.block_size
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)

        input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
        max_seqlen = max(len(seq.token_ids) for seq in sequences)

        context = Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            context_lens=None,
            block_tables=None,
            max_context_len=None,
            max_num_blocks=None,
            kv_cache=self.kv_cache,
        )
        set_context(context)
        return input_ids, positions

    def prepare_decode(self, sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        context_lens = []
        block_tables = []
        slot_mapping = []

        max_num_blocks = max(len(seq.block_table) for seq in sequences)

        for seq in sequences:
            input_ids.append(seq.last_token)
            positions.append(seq.num_tokens - 1)
            context_lens.append(seq.num_tokens)

            padded_block_table = seq.block_table.copy()
            while len(padded_block_table) < max_num_blocks:
                padded_block_table.append(0)
            block_tables.append(padded_block_table)

            pos = seq.num_tokens - 1
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = seq.block_table[block_idx]
            slot_mapping.append(block_id * self.block_size + offset)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)

        context = Context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),
            max_context_len=max(context_lens),
            max_num_blocks=max_num_blocks,
            kv_cache=self.kv_cache,
        )
        set_context(context)
        return input_ids, positions

    def capture_decode_graphs(self) -> None:
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size

        for batch_size in range(1, self.config.max_cudagraph_batch_size + 1):
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
            temperatures = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            top_ks = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            top_ps = torch.ones(batch_size, dtype=torch.float32, device=self.device)

            context = Context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=None,
                max_seqlen_k=None,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                max_context_len=self.config.max_model_len,
                max_num_blocks=max_num_blocks,
                kv_cache=self.kv_cache,
            )
            set_context(context)

            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    logits = self.model(input_ids, positions)
                    self.sampler(logits, temperatures, top_ks, top_ps)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits = self.model(input_ids, positions)
                next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)

            self.decode_graphs[batch_size] = DecodeGraphRunner(
                batch_size=batch_size,
                max_num_blocks=max_num_blocks,
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                graph=graph,
                logits=logits,
                next_tokens=next_tokens,
            )

        reset_context()

    def replay_decode_graph(self, sequences: list[Sequence]) -> list[int]:
        batch_size = len(sequences)
        runner = self.decode_graphs.get(batch_size)
        if runner is None:
            raise KeyError(f"batch_size={batch_size} 没有对应的 CUDA Graph")

        runner.block_tables.zero_()
        for i, seq in enumerate(sequences):
            runner.input_ids[i] = seq.last_token
            runner.positions[i] = seq.num_tokens - 1
            runner.context_lens[i] = seq.num_tokens
            runner.temperatures[i] = seq.temperature
            runner.top_ks[i] = getattr(seq, "top_k", 0)
            runner.top_ps[i] = getattr(seq, "top_p", 1.0)

            pos = seq.num_tokens - 1
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = seq.block_table[block_idx]
            runner.slot_mapping[i] = block_id * self.block_size + offset

            for j, block_id in enumerate(seq.block_table):
                runner.block_tables[i, j] = block_id

        context = Context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            slot_mapping=runner.slot_mapping,
            context_lens=runner.context_lens,
            block_tables=runner.block_tables,
            max_context_len=self.config.max_model_len,
            max_num_blocks=runner.max_num_blocks,
            kv_cache=self.kv_cache,
        )
        set_context(context)

        runner.graph.replay()
        return runner.next_tokens[:batch_size].clone().tolist()

    @torch.inference_mode()
    def run(self, sequences: list[Sequence], is_prefill: bool) -> list[int]:
        if not sequences:
            return []

        if is_prefill:
            input_ids, positions = self.prepare_prefill(sequences)
            logits = self.model(input_ids, positions)
            context = get_context()
            last_token_indices = (context.cu_seqlens_q[1:] - 1).long()
            logits = logits[last_token_indices]

            temperatures = torch.tensor(
                [seq.temperature for seq in sequences],
                dtype=torch.float32,
                device=self.device,
            )
            top_ks = torch.tensor(
                [getattr(seq, "top_k", 0) for seq in sequences],
                dtype=torch.long,
                device=self.device,
            )
            top_ps = torch.tensor(
                [getattr(seq, "top_p", 1.0) for seq in sequences],
                dtype=torch.float32,
                device=self.device,
            )
            next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
            return next_tokens.tolist()

        if self.use_cuda_graph and len(sequences) in self.decode_graphs:
            return self.replay_decode_graph(sequences)

        input_ids, positions = self.prepare_decode(sequences)
        logits = self.model(input_ids, positions)
        temperatures = torch.tensor(
            [seq.temperature for seq in sequences],
            dtype=torch.float32,
            device=self.device,
        )
        top_ks = torch.tensor(
            [getattr(seq, "top_k", 0) for seq in sequences],
            dtype=torch.long,
            device=self.device,
        )
        top_ps = torch.tensor(
            [getattr(seq, "top_p", 1.0) for seq in sequences],
            dtype=torch.float32,
            device=self.device,
        )
        next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
        return next_tokens.tolist()
```

### 5.3 新增 `tests/test_Day6_cudagraph.py`

```python
"""Day 6: CUDA Graph 基础验证

运行方式：
python tests/test_Day6_cudagraph.py
"""

import os
import sys

sys.path.insert(0, '.')

import torch

from config import Config
from engine.model_runner import ModelRunner


def main():
    if not torch.cuda.is_available():
        print("跳过：CUDA 不可用")
        return

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-0.6B")
    config = Config(
        model_path=model_path,
        enforce_eager=False,
        max_cudagraph_batch_size=4,
    )
    runner = ModelRunner(config)
    num_blocks = runner.get_num_free_gpu_blocks()
    runner.allocate_kv_cache(min(num_blocks, 32))

    assert runner.use_cuda_graph is True
    assert len(runner.decode_graphs) == 4
    print("✅ CUDA Graph 捕获测试通过")


if __name__ == "__main__":
    main()
```

## 6. 手敲顺序

这一篇一定要按下面顺序来：

1. 先改 `utils/context.py`
2. 再完整替换 `engine/model_runner.py`
3. 最后写 `tests/test_Day6_cudagraph.py`

## 7. 验收方法

### 7.1 语法校验

```bash
python -m py_compile utils/context.py engine/model_runner.py tests/test_Day6_cudagraph.py
```

### 7.2 CUDA Graph 测试

```bash
python tests/test_Day6_cudagraph.py
```

## 8. 你必须真正弄懂的 4 个点

1. 为什么 decode 比 prefill 更适合做 graph
2. 为什么 graph replay 前不能重新分配输入 tensor，而必须原地改静态 buffer
3. 为什么这篇只在 exact batch size 命中时 replay
4. 为什么 graph 逻辑应该留在 `ModelRunner`，而不是写进 `LLMEngine`

下一篇进入：

- [07-补齐Benchmark与Day7验收.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/07-补齐Benchmark与Day7验收.md)


# 05. 实现 Tensor Parallel 基础版

## 1. 学习目标

这一篇要把仓库从“单卡系统”推进到“基础版多卡 TP 系统”。

重点不是把所有并行细节都塞进 `LLMEngine`，而是把职责分清：

- `layers/linear.py`：负责数学切分与 all-reduce
- `models/qwen3.py`：负责把 attention / MLP 接到并行线性层
- `engine/model_runner.py`：负责分布式初始化、设备绑定、KV Cache 尺寸决策

## 2. 先修知识

## 2.1 为什么 TP 的“切分规则”不应该写在 ModelRunner 里

因为线性层的切分规则本身属于**模型数学定义**的一部分，而不是运行时策略。

例如：

- `QKVLinear` 为什么按列切
- `RowLinear` 为什么需要 all-reduce
- `MergedLinear` 为什么切完后还能对应到 `gate_proj/up_proj`

这些都应该固化在 `layers/linear.py` 里。

`ModelRunner` 真正需要管的是：

- 启动几个 rank
- 每个 rank 用哪张卡
- KV Cache 每层分配多少本地 `num_kv_heads`

## 2.2 这篇做的是“基础版 TP”

这里我们做的是**教学版、基础版** TP：

- 使用 `torch.distributed`
- 默认用 `torchrun`
- 所有 rank 运行同一套引擎主循环
- 只让 rank0 打印输出

它还不是生产级别的“控制进程 + worker 进程”架构，但足够帮助你理解 TP 如何接进这个仓库。

## 3. 本仓库当前缺口

你在 `01` 里已经把并行线性层抽象搭起来了，但还缺运行时入口：

1. 什么时候 `init_process_group`
2. 什么时候 `torch.cuda.set_device`
3. 什么时候构建模型
4. 什么时候按本地 head 数分配 KV Cache

顺序错了，TP 就会直接失效。

## 4. 最终应修改的文件

- `engine/model_runner.py`
- `example.py`
- `tests/test_Day6_tp.py`

## 5. 完整代码

### 5.1 替换 `engine/model_runner.py`

```python
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from config import Config
from engine.sequence import Sequence
from layers.linear import divide, get_tp_rank, get_tp_world_size
from layers.sampler import Sampler
from utils.context import Context, get_context, set_context
from utils.loader import load_model


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

        assert world_size == self.config.tensor_parallel_size, (
            f"world_size={world_size} 与 tensor_parallel_size={self.config.tensor_parallel_size} 不一致"
        )

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

    @torch.inference_mode()
    def run(self, sequences: list[Sequence], is_prefill: bool) -> list[int]:
        if not sequences:
            return []

        if is_prefill:
            input_ids, positions = self.prepare_prefill(sequences)
        else:
            input_ids, positions = self.prepare_decode(sequences)

        logits = self.model(input_ids, positions)
        if is_prefill:
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
```

### 5.2 替换 `example.py`

```python
import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from llm import LLM
from sampling_params import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="nano-vLLM Tensor Parallel 示例")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def main():
    args = parse_args()
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)

    llm = LLM(
        model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    raw_prompts = [
        "请解释一下 Tensor Parallelism 的核心思想。",
        "请解释一下 Row Parallel 为什么需要 all_reduce。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=is_main_process())

    if is_main_process():
        for prompt, output in zip(raw_prompts, outputs):
            print(f"\n[问题] {prompt}")
            print(f"[回答] {output['text']}")


if __name__ == "__main__":
    if torch.cuda.is_available() and is_main_process():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    main()
```

### 5.3 新增 `tests/test_Day6_tp.py`

```python
"""Day 6: Tensor Parallel 基础验证

运行方式：
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
"""

import os
import sys

sys.path.insert(0, '.')

import torch
import torch.distributed as dist

from layers.linear import MergedLinear, QKVLinear, RowLinear


def init_dist():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def main():
    rank, world_size, local_rank = init_dist()

    qkv = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64).cuda()
    merged = MergedLinear(512, 1024, num_shards=2).cuda()
    row = RowLinear(512, 256, bias=False).cuda()

    assert qkv.weight.shape == ((8 + 2 + 2) * 64 // world_size, 512)
    assert merged.weight.shape == (2048 // world_size, 512)
    assert row.weight.shape == (256, 512 // world_size)

    hidden = torch.randn(4, 512 // world_size, device="cuda")
    output = row(hidden)
    assert output.shape == (4, 256)

    if rank == 0:
        print("✅ Tensor Parallel 基础测试通过")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## 6. 手敲顺序

这篇的关键顺序是：

1. 先改 `engine/model_runner.py`
2. 再改 `example.py`
3. 最后写 `tests/test_Day6_tp.py`

因为 TP 是否真正生效，取决于**模型构建前**有没有完成分布式初始化。

## 7. 验收方法

### 7.1 语法校验

```bash
python -m py_compile engine/model_runner.py example.py tests/test_Day6_tp.py
```

### 7.2 双卡 TP 测试

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

### 7.3 双卡示例运行

```bash
torchrun --nproc_per_node=2 example.py --tensor_parallel_size 2 --model_path models/Qwen3-0.6B
```

## 8. 这一篇必须真正理解的 3 个点

1. 为什么 `init_process_group` 必须发生在模型构建之前
2. 为什么 `QKVLinear` 的 head 切分逻辑属于 `layers/linear.py`
3. 为什么在这个教学版系统里，所有 rank 都跑一遍 generate，但只有 rank0 打印输出

下一篇进入：

- [06-实现CUDA-Graph基础版.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/06-实现CUDA-Graph基础版.md)


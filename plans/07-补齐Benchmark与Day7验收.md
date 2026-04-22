# 07. 补齐 Benchmark 与 Day7 验收

## 1. 本篇目标

前面几篇把功能链路补齐以后，最后这一篇解决的是：

> 这套仓库到底怎么用真实数据收口，而不是继续停留在“看起来快写完了”。

本篇完成后，至少要做到下面 5 件事：

1. 仓库里有一个真正可运行的 `bench.py`
2. 仓库里有一个轻量的 `tests/test_Day7.py`
3. `readme.md` 不再写错命令和错目录树
4. `todo_list.md` 的勾选状态和真实代码状态一致
5. benchmark 数据和 README 回写顺序明确，不再先写文档后找数据

---

## 2. 前置条件

本篇默认你前面的实现已经走到：

1. 单卡主循环能跑
2. `LLM.generate()` 返回 `list[dict]`
3. `SamplingParams` 已经有 `temperature / top_k / top_p`

如果这些前提还没成立，先不要急着跑 benchmark。

因为那样测出来的不是性能结论，只是半成品状态。

---

## 3. 先看当前仓库的真实问题

### 3.1 `readme.md`

当前 [readme.md](/home/psx/nano_vllm_repro/nano_vll_repro/readme.md:72) 至少有 4 个问题：

1. 示例命令还是旧 CLI 形态
2. 目录树写成 `nanovllm/` 风格，不是当前仓库真实布局
3. Day6 / Day7 已勾选，但当前仓库并没有对应文件
4. 性能表还是占位词

### 3.2 `todo_list.md`

当前 [todo_list.md](/home/psx/nano_vllm_repro/nano_vll_repro/todo_list.md:20) 更像“早期计划草稿”，不是当前真实待办：

1. 目标目录树已经和现状不一致
2. 采样参数等条目和当前实现状态有冲突
3. Day6 / Day7 的完成状态没有和代码同步

### 3.3 当前仓库还没有 benchmark 入口

这意味着现在还缺下面 4 件事：

1. `bench.py`
2. `nano` / `hf` 共用同一组输入条件
3. 结构化 benchmark 结果对象
4. README 可直接回写的 Markdown 表格

---

## 4. 本篇修改原则

### 4.1 benchmark 脚本负责“测量”

它可以比较重，因为它真的要跑模型。

### 4.2 测试文件负责“约束结构”

它必须轻量。

不要在 `tests/test_Day7.py` 里直接跑大模型。

### 4.3 README / TODO 只回写事实

顺序固定如下：

1. 先写 `bench.py`
2. 再跑 benchmark
3. 保存真实结果
4. 最后回写 `readme.md`
5. 最后回写 `todo_list.md`

---

## 5. 逐步修改

## 5.1 新建 `bench.py`

新建文件：

- `bench.py`

完整代码如下：

```python
"""Day 7 benchmark 脚本"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from llm import LLM
from sampling_params import SamplingParams


@dataclass
class BenchmarkResult:
    backend: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tpot_ms: float
    throughput_tps: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano-vllm / HF benchmark")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--backend", choices=["nano", "hf", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser


def build_prompts(batch_size: int) -> list[str]:
    seed_prompts = [
        "请用一句话解释一下 PagedAttention。",
        "请用一句话解释一下 Continuous Batching。",
        "请用一句话解释一下 Prefix Cache。",
        "请用一句话解释一下 CUDA Graph 为什么常用于 decode。",
    ]
    prompts = []
    while len(prompts) < batch_size:
        prompts.extend(seed_prompts)
    return prompts[:batch_size]


def count_prompt_tokens(tokenizer, prompts: list[str]) -> int:
    encoded = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    return int(encoded["attention_mask"].sum().item())


def make_sampling_params(args) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def format_markdown_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Backend | Batch | Prompt Tokens | Output Tokens | TTFT (ms) | Total (ms) | TPOT (ms) | Throughput (tok/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item.backend} | {item.batch_size} | {item.prompt_tokens} | {item.output_tokens} | "
            f"{item.ttft_ms:.2f} | {item.total_latency_ms:.2f} | {item.tpot_ms:.2f} | {item.throughput_tps:.2f} |"
        )
    return "\n".join(lines)


def aggregate_result(
    backend: str,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    ttft_list: list[float],
    total_list: list[float],
) -> BenchmarkResult:
    avg_ttft = mean(ttft_list)
    avg_total = mean(total_list)
    tpot_ms = 0.0 if output_tokens == 0 else avg_total / output_tokens * 1000.0
    throughput = 0.0 if avg_total == 0 else output_tokens / avg_total
    return BenchmarkResult(
        backend=backend,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=avg_ttft * 1000.0,
        total_latency_ms=avg_total * 1000.0,
        tpot_ms=tpot_ms,
        throughput_tps=throughput,
    )


def run_nano_backend(args, prompts: list[str], tokenizer) -> BenchmarkResult:
    llm = LLM(args.model_path)
    sampling_params = make_sampling_params(args)

    ttft_list = []
    total_list = []
    output_tokens = 0

    for _ in range(args.warmup):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    for _ in range(args.repeat):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        total = time.perf_counter() - start
        total_list.append(total)

        # 当前教学仓库没有单独暴露 TTFT 事件，基础版先用第一轮 step 的近似值替代。
        ttft_list.append(total / max(1, args.max_tokens))
        output_tokens = sum(len(item["token_ids"]) for item in outputs)

    prompt_tokens = count_prompt_tokens(tokenizer, prompts)
    return aggregate_result("nano", args.batch_size, prompt_tokens, output_tokens, ttft_list, total_list)


@torch.inference_mode()
def run_hf_backend(args, prompts: list[str], tokenizer) -> BenchmarkResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    ttft_list = []
    total_list = []
    output_tokens = 0

    for _ in range(args.warmup):
        model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
        )

    for _ in range(args.repeat):
        start = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
        )
        total = time.perf_counter() - start
        total_list.append(total)
        ttft_list.append(total / max(1, args.max_tokens))
        output_tokens = int(outputs.shape[1] - inputs["input_ids"].shape[1]) * args.batch_size

    prompt_tokens = int(inputs["attention_mask"].sum().item())
    return aggregate_result("hf", args.batch_size, prompt_tokens, output_tokens, ttft_list, total_list)


def main():
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = build_prompts(args.batch_size)

    results: list[BenchmarkResult] = []
    if args.backend in {"nano", "both"}:
        results.append(run_nano_backend(args, prompts, tokenizer))
    if args.backend in {"hf", "both"}:
        results.append(run_hf_backend(args, prompts, tokenizer))

    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        print(format_markdown_table(results))


if __name__ == "__main__":
    main()
```

这份脚本的重点不是“绝对精准的论文级 benchmark”。

它的重点是：

1. 两个后端共用同一组输入条件
2. 结果有结构化对象
3. 可以直接生成 README 用的表格

---

## 5.2 新建 `tests/test_Day7.py`

新建文件：

- `tests/test_Day7.py`

完整代码如下：

```python
"""Day 7 benchmark 结构测试"""

import sys
sys.path.insert(0, ".")

from bench import BenchmarkResult, aggregate_result, build_parser, format_markdown_table


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.model_path == "models/Qwen3-0.6B"
    assert args.backend == "both"
    assert args.batch_size == 4
    assert args.max_tokens == 64


def test_aggregate_result_metrics():
    result = aggregate_result(
        backend="nano",
        batch_size=4,
        prompt_tokens=120,
        output_tokens=40,
        ttft_list=[0.02, 0.03],
        total_list=[1.0, 1.2],
    )
    assert isinstance(result, BenchmarkResult)
    assert result.backend == "nano"
    assert result.batch_size == 4
    assert result.prompt_tokens == 120
    assert result.output_tokens == 40
    assert result.ttft_ms == 25.0
    assert result.total_latency_ms == 1100.0
    assert round(result.tpot_ms, 2) == 27.5


def test_markdown_table_format():
    result = BenchmarkResult(
        backend="hf",
        batch_size=2,
        prompt_tokens=64,
        output_tokens=32,
        ttft_ms=12.5,
        total_latency_ms=400.0,
        tpot_ms=12.5,
        throughput_tps=80.0,
    )
    table = format_markdown_table([result])
    assert "| Backend | Batch | Prompt Tokens | Output Tokens |" in table
    assert "| hf | 2 | 64 | 32 | 12.50 | 400.00 | 12.50 | 80.00 |" in table


if __name__ == "__main__":
    test_parser_defaults()
    test_aggregate_result_metrics()
    test_markdown_table_format()
    print("🎉 Day 7 测试执行完成")
```

这份测试只锁结构，不锁真实模型推理。

这样它才适合常跑。

---

## 5.3 回写 `readme.md`

这里不要整文件重写。

按下面 4 条回写就够了：

1. 把目录树从 `nanovllm/` 风格改成当前真实仓库布局。
2. 把示例命令改成：

```bash
python example.py
```

3. 把性能表替换成 `bench.py` 的真实输出。
4. 在性能表上方补一句环境说明：

```markdown
> 以下数据来自本机实际测试，仅代表当前硬件、模型和配置，不代表通用结论。
```

---

## 5.4 回写 `todo_list.md`

这里也不要继续把它当“最初冲刺计划表”。

更稳的做法是把它改成：

1. **当前已完成**
2. **当前未完成**
3. **下一步建议**

三段式。

至少要把下面几件事写实：

1. 当前真实存在的测试只到 `test_Day4.py`
2. `bench.py / test_Day7.py` 是本文新增目标，不是当前 HEAD 已存在事实
3. TP / CUDA Graph 只有在对应文档真正落地后才能勾选

---

## 6. 本篇结束后的最小验收

先做结构层验收：

```bash
cd nano_vll_repro
python -m py_compile bench.py tests/test_Day7.py
python tests/test_Day7.py
```

再做真实 benchmark：

```bash
python bench.py --backend both --batch_size 4 --max_tokens 64 --repeat 3
```

最后再回写：

```text
readme.md
todo_list.md
```

顺序不要反过来。

---

## 7. 常见错误

### 7.1 先改 README，再跑 benchmark

这样最后很容易又回到“文档先行，数据补不齐”的老问题。

### 7.2 在 `tests/test_Day7.py` 里直接跑大模型

这样测试会变重，也会把结构校验和真实 benchmark 混在一起。

### 7.3 `nano` 和 `hf` 用了不同采样参数

这会让对比结果失去解释性。

### 7.4 README 继续保留旧命令和旧目录树

这类错误很低级，但读者第一眼就会踩坑。

---

## 8. 本篇真正学到的东西

Day7 真正重要的，不是“补了一个 benchmark 脚本”。

而是下面 4 件事：

1. 结果要可测量
2. 结论要可回写
3. 文档要只写事实
4. 测试和 benchmark 要分层

只有这样，这套仓库才算从“教学草稿”真正进入“可验证工程”状态。

# 07. 补齐 Benchmark 与 Day7 验收

## 1. 学习目标

这一篇是 Day7 的收口篇。

你要完成的不是再加一个“新功能”，而是回答下面这三个问题：

1. 这个系统到底能不能稳定运行
2. 性能到底怎么样
3. README 和 `todo_list.md` 该如何更新成“真实完成状态”

## 2. 先修知识

## 2.1 Day7 真正要测什么

Day7 不只是“跑一次 example.py 就结束”。

至少要收集这三类指标：

- 吞吐量：`tokens / second`
- 首 token 延迟：TTFT（Time To First Token）
- 输出 token 延迟：TPOT（Time Per Output Token）

## 2.2 为什么要同时测 nano-vllm 版和 Hugging Face 版

因为你最终要回答的问题是：

> 你这个手写系统，相比“直接用 Hugging Face”，到底差在哪里、快在哪里、学到了什么。

所以 benchmark 脚本应该至少支持：

- `nano` 后端
- `hf` 后端
- 二者对比输出

## 2.3 为什么 Day7 还要有测试脚本

性能是性能，正确性是正确性。

Day7 至少还要有一个“最终验收脚本”，帮助你确认：

- benchmark 结果结构完整
- 输出表格正常
- 关键字段不会丢

## 3. 本仓库当前缺口

当前仓库在 Day7 上几乎还是空的：

1. 没有 `bench.py`
2. 没有 Day7 的验收测试
3. README 虽然写了 Day7 已完成，但性能表还是“待测试”
4. `todo_list.md` 的勾选状态和尾部进度表自相矛盾

## 4. 最终应修改的文件

- `bench.py`
- `tests/test_Day7.py`
- `readme.md`
- `todo_list.md`

这一篇里我只给前两个文件的完整代码；后两个文档文件给出明确回写模板。

## 5. 完整代码

### 5.1 新增 `bench.py`

```python
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def build_parser():
    parser = argparse.ArgumentParser(description="nano-vLLM / HF benchmark")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--backend", type=str, choices=["nano", "hf", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser


def build_chat_prompts(tokenizer, batch_size: int) -> list[str]:
    raw_prompts = [
        "请解释一下 PagedAttention 的核心思想。",
        "请解释一下 Continuous Batching 的优势。",
        "请解释一下 Tensor Parallelism 中 Row Parallel 的作用。",
        "请解释一下 CUDA Graph 为什么更适合 decode 阶段。",
    ]
    prompts = []
    for i in range(batch_size):
        raw_prompt = raw_prompts[i % len(raw_prompts)]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def summarize_runs(backend: str, batch_size: int, prompt_tokens: int, output_tokens: int, ttft_runs, total_runs):
    ttft_ms = mean(ttft_runs) * 1000
    total_ms = mean(total_runs) * 1000
    tpot_ms = (mean(total_runs) / max(output_tokens, 1)) * 1000
    throughput = output_tokens / max(mean(total_runs), 1e-6)
    return BenchmarkResult(
        backend=backend,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        total_latency_ms=total_ms,
        tpot_ms=tpot_ms,
        throughput_tps=throughput,
    )


def format_results_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Backend | Batch | Prompt Tokens | Output Tokens | TTFT (ms) | Total (ms) | TPOT (ms) | Throughput (tok/s) |",
        "|---------|-------|---------------|---------------|-----------|------------|-----------|--------------------|",
    ]
    for result in results:
        lines.append(
            f"| {result.backend} | {result.batch_size} | {result.prompt_tokens} | {result.output_tokens} | "
            f"{result.ttft_ms:.2f} | {result.total_latency_ms:.2f} | {result.tpot_ms:.2f} | {result.throughput_tps:.2f} |"
        )
    return "\n".join(lines)


def run_nano_benchmark(args) -> BenchmarkResult:
    import torch
    from transformers import AutoTokenizer

    from llm import LLM
    from sampling_params import SamplingParams

    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, args.batch_size)
    prompt_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)

    llm = LLM(model_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for _ in range(args.warmup):
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    ttft_runs = []
    total_runs = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        first_token_outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=1,
            ),
            use_tqdm=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_runs.append(time.perf_counter() - start)
        output_tokens = sum(len(item["token_ids"]) for item in outputs)

        del first_token_outputs

    return summarize_runs("nano", args.batch_size, prompt_tokens, output_tokens, ttft_runs, total_runs)


def run_hf_benchmark(args) -> BenchmarkResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, args.batch_size)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].numel())

    for _ in range(args.warmup):
        _ = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=True)

    ttft_runs = []
    total_runs = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        one_token = model.generate(**inputs, max_new_tokens=1, do_sample=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_runs.append(time.perf_counter() - start)

        output_tokens = int(outputs.numel() - inputs["input_ids"].numel())
        del one_token

    return summarize_runs("hf", args.batch_size, prompt_tokens, output_tokens, ttft_runs, total_runs)


def main():
    args = build_parser().parse_args()
    results = []

    if args.backend in {"nano", "both"}:
        results.append(run_nano_benchmark(args))
    if args.backend in {"hf", "both"}:
        results.append(run_hf_benchmark(args))

    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        print(format_results_table(results))


if __name__ == "__main__":
    main()
```

### 5.2 新增 `tests/test_Day7.py`

```python
"""Day 7: benchmark 输出与最终收口测试"""

import sys

sys.path.insert(0, '.')

from bench import BenchmarkResult, format_results_table, summarize_runs


def test_summarize_runs():
    result = summarize_runs(
        backend="nano",
        batch_size=4,
        prompt_tokens=128,
        output_tokens=256,
        ttft_runs=[0.05, 0.06, 0.07],
        total_runs=[0.5, 0.55, 0.6],
    )

    assert isinstance(result, BenchmarkResult)
    assert result.backend == "nano"
    assert result.batch_size == 4
    assert result.output_tokens == 256
    assert result.ttft_ms > 0
    assert result.throughput_tps > 0


def test_format_results_table():
    table = format_results_table(
        [
            BenchmarkResult(
                backend="nano",
                batch_size=4,
                prompt_tokens=128,
                output_tokens=256,
                ttft_ms=50.0,
                total_latency_ms=500.0,
                tpot_ms=2.0,
                throughput_tps=512.0,
            )
        ]
    )

    assert "Backend" in table
    assert "nano" in table
    assert "512.00" in table


if __name__ == "__main__":
    test_summarize_runs()
    test_format_results_table()
    print("✅ Day7 收口测试通过")
```

## 6. README 与 `todo_list.md` 的回写模板

这一步你不要再“凭感觉打勾”，而要按真实状态回写。

### 6.1 README 的性能数据表

当你第一次跑完 benchmark 后，把 README 里这一段：

```markdown
| 吞吐量 (tokens/s) | 待测试 | 待测试 | 待测试 |
| 首 Token 延迟 | 待测试 | 待测试 | 待测试 |
| 显存占用 | 待测试 | 待测试 | 待测试 |
```

替换成真实数据，例如：

```markdown
| 吞吐量 (tokens/s) | 512.4 | 401.8 | +27.5% |
| 首 Token 延迟 | 58.2 ms | 71.4 ms | -18.5% |
| 显存占用 | 8.6 GB | 9.8 GB | -12.2% |
```

### 6.2 `todo_list.md` 的 Day6 / Day7 回写原则

只有在下面条件都满足时，才把 Day6 / Day7 勾掉：

- Day6：
  - `torchrun --nproc_per_node=2 tests/test_Day6_tp.py` 通过
  - `python tests/test_Day6_cudagraph.py` 通过
  - `example.py` 在 TP 模式下可跑
- Day7：
  - `python tests/test_Day7.py` 通过
  - `python bench.py --backend both ...` 能输出结果
  - README / `todo_list.md` 状态一致

## 7. 手敲顺序

1. 先写 `bench.py`
2. 再写 `tests/test_Day7.py`
3. 最后根据真实 benchmark 结果更新 README 和 `todo_list.md`

## 8. Day7 验收命令

### 8.1 语法校验

```bash
python -m py_compile bench.py tests/test_Day7.py
```

### 8.2 Day7 测试

```bash
python tests/test_Day7.py
```

### 8.3 Benchmark 示例

```bash
python bench.py --backend both --model_path models/Qwen3-0.6B --batch_size 4 --max_tokens 64
```

## 9. 全套教案结束后的最终问题

如果你已经跟着所有文档敲完，你应该可以自己完整回答下面这些问题：

1. 为什么 PagedAttention 需要 block table
2. Continuous Batching 的优势到底来自哪里
3. Qwen3 的 `q_norm / k_norm` 和普通 Transformer 有什么区别
4. 为什么 TP 的切分数学逻辑应该写在线性层，而不是写在引擎里
5. 为什么 CUDA Graph 优先接 decode
6. 怎么定义一个真正有意义的 LLM 推理 benchmark

到这一步，这套仓库才算真正完成了你 `todo_list.md` 里的 Day7 目标。

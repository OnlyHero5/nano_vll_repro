# 07. 补齐 Benchmark 与 Day7 验收（只回写真实数据，不回写想象中的完成状态）

## 1. 本篇目标

前面几篇把功能链路补齐以后，最后这一篇要解决的不是“再加一个新特性”，而是把仓库收口成：

1. 可验证
2. 可对外说明
3. 文档与代码状态一致

的状态。

你最后要回答 4 个问题：

1. 这套系统现在到底能不能稳定跑
2. 相比 Hugging Face，性能差异到底是多少
3. `readme.md` 和 `todo_list.md` 到底该如何回写成真实状态
4. benchmark 输出能不能被测试文件稳定约束住

本篇完成后，至少应满足下面 5 个状态：

1. 仓库里有一个真正可运行的 `bench.py`
2. 仓库里有一个轻量但有约束力的 `tests/test_Day7.py`
3. `readme.md` 里的性能表不再是“待测试”
4. `todo_list.md` 的勾选状态和真实结果一致
5. 文档会明确区分“真实 benchmark 输出”和“还没有做的后续工作”

---

## 2. 权威参考

本篇对照下面 4 组来源：

1. 当前仓库：
   - `nano_vll_repro/readme.md`
   - `nano_vll_repro/todo_list.md`
   - `nano_vll_repro/llm.py`
   - `nano_vll_repro/sampling_params.py`
2. 上游主仓库：
   - `https://github.com/GeeeekExplorer/nano-vllm`
   - 根目录 `README.md`
   - 根目录 `bench.py`
3. Hugging Face 对照对象：
   - `transformers.AutoModelForCausalLM`
   - `transformers.AutoTokenizer`
4. 公开变体：
   - `qqtang-code/nano-vllm`
   - `wangyuzhuo116/nano-vllm`
   - `DIOYF/nano-vllm-dio`

这里先把这次联网核对后的结论说清楚：

1. 上游主仓库根目录公开页面确实已经有 `bench.py`，说明“最终给一个基准脚本”是合理方向。
2. 但上游 benchmark 的目标更偏吞吐量对比，不是你当前仓库要的“结构化 Day7 验收脚本”。
3. 因此本篇的正确做法是：
   - 参考上游的 benchmark 入口思路
   - 但扩成适合当前教学仓库的结构化输出、JSON 输出和 README 回写流程

换句话说：

> 本篇是“参考上游 benchmark 的方向”，但不是“原样照抄上游 bench.py”。

---

## 3. 先看当前仓库的真实问题

### 3.1 `readme.md`

当前 [readme.md](/home/psx/nano_vllm_repro/nano_vll_repro/readme.md:72) 至少有 4 个真实问题：

1. 示例命令还是 `python example.py --model qwen3 --device cuda --max_tokens 128`，但当前 `example.py` 根本不是这个 CLI 形态。
2. Day6 / Day7 被打勾。
3. 性能表还是“待测试”。
4. 文档目录树还写成 `nanovllm/` 风格，和当前仓库实际目录结构不完全一致。

### 3.2 `todo_list.md`

当前 [todo_list.md](/home/psx/nano_vllm_repro/nano_vll_repro/todo_list.md:72) 的问题和 README 类似，但更严重：

1. 前面很多任务被勾选成完成。
2. Day6 / Day7 仍然大量未勾选。
3. 文中又混入一些已经和当前代码状态不一致的目标结构表述。

换句话说，它既不是“真实完成状态”，也不是“当前待办清单”，而是两种状态混在一起。

### 3.3 缺少真正的 benchmark 入口

目前仓库还没有：

1. `bench.py`
2. `nano` 与 `hf` 共用一套 prompt / sampling 配置的对照逻辑
3. 结构化 benchmark 输出对象
4. 统一的 Markdown 表格格式化逻辑

### 3.4 缺少最终验收测试

Day7 不应该再新增一个“重的大模型全流程测试”，而应该新增一个：

1. 只验证 benchmark 结构层
2. 不依赖真实模型和 GPU
3. 能快速回归

的测试文件。

---

## 4. 本篇修改原则

### 4.1 benchmark 脚本负责“测量”，测试文件负责“约束结构”

这两件事情必须分开：

1. `bench.py` 可以较重，因为它真的要跑模型
2. `tests/test_Day7.py` 必须轻量，因为它要常跑

### 4.2 `nano` 和 `hf` 的输入条件必须尽量对齐

至少要统一这些项：

1. prompt
2. batch size
3. `max_tokens`
4. `temperature`
5. `top_k`
6. `top_p`

即便两侧最终输出文本不可能完全一样，也必须保证“输入条件尽量一致”，否则对比结果没有解释性。

### 4.3 README / TODO 只允许回写事实

建议工作顺序永远是：

1. 写 `bench.py`
2. 跑真实数据
3. 保存结果
4. 再回写 README
5. 再回写 TODO

不要倒过来。

### 4.4 benchmark 输出最好既有人类可读版本，也有机器可读版本

这就是为什么本篇会同时提供：

1. Markdown 表格输出
2. JSON 输出

这样你既可以：

1. 把表格直接贴进 README
2. 也可以把 JSON 喂给测试或日志系统

---

## 5. 逐步修改

## 5.1 新建 `bench.py`

直接新建：

- `nano_vll_repro/bench.py`

完整代码如下。注释密度故意写得很高，因为这份脚本会同时承担：

1. benchmark 入口
2. 结果格式化器
3. README 数据来源

三种职责。

```python
"""Day 7 benchmark 脚本

这个脚本只做一件事：
对比当前教学仓库自己的 `nano` 后端与 Hugging Face `hf` 后端，
在同一组输入条件下的推理指标。

输出指标包括：
1. TTFT（Time To First Token）
2. Total Latency
3. TPOT（Time Per Output Token）
4. Throughput（tokens / second）

边界说明：
1. 这不是论文级 benchmark，而是工程验收脚本。
2. `nano` 与 `hf` 的输出文本不要求逐 token 完全一致。
3. benchmark 只比较性能指标，不比较文本质量。
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from statistics import mean


# 让脚本可以从仓库根目录直接导入本地模块。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class BenchmarkResult:
    """
    单个后端的一次聚合 benchmark 结果。

    这些字段既会被：
    - Markdown 表格输出使用
    - JSON 输出使用

    也会被：
    - `tests/test_Day7.py`
    - README 回写流程

    共同依赖。

    因此这里的字段名和语义必须保持稳定。
    """

    backend: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tpot_ms: float
    throughput_tps: float


def build_parser() -> argparse.ArgumentParser:
    """
    构造 CLI 参数解析器。

    这里故意把所有可调参数统一收口到一个 parser，
    这样后面的 `tests/test_Day7.py` 就能稳定验证默认值不漂移。
    """

    parser = argparse.ArgumentParser(description="nano-vllm / HF benchmark")

    # 模型路径默认沿用当前仓库本地模型目录约定。
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")

    # 可以只跑一个后端，也可以两者都跑。
    parser.add_argument("--backend", type=str, choices=["nano", "hf", "both"], default="both")

    # 输入规模与采样配置。
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)

    # warmup 与 repeat 分开设置，便于规避首次初始化扰动。
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)

    # `--json` 让结果既能给人看，也能给别的脚本处理。
    parser.add_argument("--json", action="store_true")
    return parser


def build_chat_prompts(tokenizer, batch_size: int) -> list[str]:
    """
    构造一组固定聊天 prompt。

    这里故意不用随机 prompt，原因是：
    - Day7 的目标是可复现的工程对比
    - 不是随机输入压测
    """

    raw_prompts = [
        "请解释一下 PagedAttention 的核心思想。",
        "请解释一下 Continuous Batching 的优势。",
        "请解释一下 Tensor Parallelism 中 Row Parallel 的作用。",
        "请解释一下 CUDA Graph 为什么更适合 decode 阶段。",
    ]

    prompts: list[str] = []
    for index in range(batch_size):
        raw_prompt = raw_prompts[index % len(raw_prompts)]

        # 沿用 Qwen chat template，尽量让 nano / hf 两侧输入形式一致。
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    return prompts


def summarize_runs(
    backend: str,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    ttft_runs: list[float],
    total_runs: list[float],
) -> BenchmarkResult:
    """
    把多次运行的原始耗时聚合成最终结果对象。

    这里故意把“统计聚合”单独抽出来，
    是为了让测试文件可以只验证公式，而不用真的跑模型。
    """

    mean_ttft = mean(ttft_runs)
    mean_total = mean(total_runs)

    # TTFT / Total 用毫秒展示，更适合 README 表格阅读。
    ttft_ms = mean_ttft * 1000.0
    total_ms = mean_total * 1000.0

    # TPOT 用“平均总时延 / 平均输出 token 数”定义。
    # 如果 output_tokens 为 0，用 max(..., 1) 防止除零。
    tpot_ms = (mean_total / max(output_tokens, 1)) * 1000.0

    # Throughput 按总输出 token 数 / 总耗时来算。
    throughput_tps = output_tokens / max(mean_total, 1e-6)

    return BenchmarkResult(
        backend=backend,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        total_latency_ms=total_ms,
        tpot_ms=tpot_ms,
        throughput_tps=throughput_tps,
    )


def format_results_table(results: list[BenchmarkResult]) -> str:
    """
    把结果格式化成 Markdown 表格。

    这样你跑完 benchmark 以后，可以直接把输出复制进 README。
    """

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


def sync_if_cuda(torch_module) -> None:
    """
    在 CUDA 环境下显式同步 GPU。

    这一步是 benchmark 基本功：
    - 如果不做同步
    - 计时会被 CUDA 异步执行污染
    """

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def build_hf_generate_kwargs(args, max_new_tokens: int) -> dict:
    """
    构造 Hugging Face generate 的参数字典。

    这里把：
    - greedy
    - do_sample
    - top_k / top_p

    的分支逻辑单独抽出来，是为了让结构测试可以直接锁住这层行为。
    """

    kwargs = {"max_new_tokens": max_new_tokens}

    # temperature <= 0 统一视为 greedy。
    if args.temperature <= 0:
        kwargs["do_sample"] = False
        return kwargs

    kwargs["do_sample"] = True
    kwargs["temperature"] = args.temperature

    # HF 里 top_k=0 通常表示不启用 top-k。
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k

    kwargs["top_p"] = args.top_p
    return kwargs


def run_nano_benchmark(args) -> BenchmarkResult:
    """
    运行当前教学仓库自己的 `nano` 后端 benchmark。

    这里故意直接用本地 `LLM` 用户侧 API，而不是旁路内部模块，
    因为 Day7 关注的是“对外可用状态”，不是局部函数吞吐量。
    """

    import torch
    from transformers import AutoTokenizer

    from llm import LLM
    from sampling_params import SamplingParams

    model_path = os.path.join(PROJECT_ROOT, args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, args.batch_size)

    # prompt token 数按 tokenizer 编码后的总长度统计。
    prompt_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)

    llm = LLM(model_path)

    full_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # TTFT 在这里用“只生成 1 个 token 的总耗时”近似。
    ttft_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=1,
    )

    # 先 warmup，尽量把冷启动代价排除在正式测量之外。
    for _ in range(args.warmup):
        _ = llm.generate(prompts, sampling_params=full_sampling_params, use_tqdm=False)
        sync_if_cuda(torch)

    ttft_runs: list[float] = []
    total_runs: list[float] = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        first_token_outputs = llm.generate(
            prompts,
            sampling_params=ttft_sampling_params,
            use_tqdm=False,
        )
        sync_if_cuda(torch)
        ttft_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling_params=full_sampling_params,
            use_tqdm=False,
        )
        sync_if_cuda(torch)
        total_runs.append(time.perf_counter() - start)

        # 当前这轮真实输出 token 总数。
        output_tokens = sum(len(item["token_ids"]) for item in outputs)

        # 删除临时结果，避免长 benchmark 中保留无用引用。
        del first_token_outputs

    return summarize_runs(
        backend="nano",
        batch_size=args.batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_runs=ttft_runs,
        total_runs=total_runs,
    )


def run_hf_benchmark(args) -> BenchmarkResult:
    """
    运行 Hugging Face 对照 benchmark。

    这里的目标不是把 HF 路径包装得多漂亮，
    而是让它尽量和本地 nano 路径共享同一批 prompt 和同一组采样参数。
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(PROJECT_ROOT, args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, args.batch_size)

    # HF 路径直接走 padding batch。
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype 选择保持简单：
    # - CUDA + BF16 支持时优先 BF16
    # - 否则让 HF 自己决定默认 dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # prompt token 数按 attention_mask 统计，更接近真实输入 token 数。
    prompt_tokens = int(inputs["attention_mask"].sum().item())

    ttft_kwargs = build_hf_generate_kwargs(args, max_new_tokens=1)
    total_kwargs = build_hf_generate_kwargs(args, max_new_tokens=args.max_tokens)

    for _ in range(args.warmup):
        _ = model.generate(**inputs, **total_kwargs)
        sync_if_cuda(torch)

    ttft_runs: list[float] = []
    total_runs: list[float] = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        _ = model.generate(**inputs, **ttft_kwargs)
        sync_if_cuda(torch)
        ttft_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        generated = model.generate(**inputs, **total_kwargs)
        sync_if_cuda(torch)
        total_runs.append(time.perf_counter() - start)

        # 输出 token 数 = 生成后长度减输入长度。
        prompt_len = inputs["input_ids"].shape[1]
        output_tokens = int(generated[:, prompt_len:].numel())

    return summarize_runs(
        backend="hf",
        batch_size=args.batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_runs=ttft_runs,
        total_runs=total_runs,
    )


def main() -> None:
    """
    benchmark 脚本入口。

    输出分两种：
    1. JSON：适合日志系统或二次处理
    2. Markdown 表格：适合直接复制到 README
    """

    parser = build_parser()
    args = parser.parse_args()

    results: list[BenchmarkResult] = []

    if args.backend in {"nano", "both"}:
        results.append(run_nano_benchmark(args))

    if args.backend in {"hf", "both"}:
        results.append(run_hf_benchmark(args))

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(format_results_table(results))


if __name__ == "__main__":
    main()
```

---

## 5.2 新建 `tests/test_Day7.py`

直接新建：

- `nano_vll_repro/tests/test_Day7.py`

完整代码如下。注意这份测试文件故意不跑真实模型，它只锁住 benchmark 的结构层。

```python
"""Day 7 测试脚本 - benchmark 结构层验收

这份测试文件只验证 `bench.py` 的结构化输出层，不跑真实模型。

它要锁住的东西包括：
1. parser 默认值
2. summarize_runs 的统计公式
3. Markdown 表格输出格式
4. BenchmarkResult 的 JSON 序列化能力

这样做的原因很明确：
- 真实 benchmark 已经由 `bench.py` 负责
- 测试文件必须保持轻量、快速、无额外硬件依赖
"""

import json
import os
import sys


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


from bench import BenchmarkResult, build_parser, format_results_table, summarize_runs


def test_build_parser_defaults() -> None:
    """验证 CLI 默认值不会漂移。"""

    parser = build_parser()
    args = parser.parse_args([])

    assert args.model_path == "models/Qwen3-0.6B"
    assert args.backend == "both"
    assert args.batch_size == 4
    assert args.max_tokens == 64
    assert args.temperature == 0.8
    assert args.top_k == 20
    assert args.top_p == 0.95
    assert args.warmup == 1
    assert args.repeat == 3
    assert args.json is False


def test_summarize_runs() -> None:
    """验证聚合统计逻辑。"""

    result = summarize_runs(
        backend="nano",
        batch_size=4,
        prompt_tokens=120,
        output_tokens=40,
        ttft_runs=[0.10, 0.20],
        total_runs=[1.00, 1.20],
    )

    # mean(ttft_runs) = 0.15s -> 150ms
    assert abs(result.ttft_ms - 150.0) < 1e-6

    # mean(total_runs) = 1.10s -> 1100ms
    assert abs(result.total_latency_ms - 1100.0) < 1e-6

    # TPOT = 1.10 / 40 * 1000 = 27.5ms
    assert abs(result.tpot_ms - 27.5) < 1e-6

    # Throughput = 40 / 1.10
    assert abs(result.throughput_tps - (40 / 1.10)) < 1e-6


def test_format_results_table() -> None:
    """验证 Markdown 表格输出包含关键字段。"""

    result = BenchmarkResult(
        backend="nano",
        batch_size=4,
        prompt_tokens=100,
        output_tokens=32,
        ttft_ms=123.45,
        total_latency_ms=678.90,
        tpot_ms=21.22,
        throughput_tps=47.89,
    )

    table = format_results_table([result])

    # 先锁住表头。
    assert "| Backend | Batch | Prompt Tokens |" in table

    # 再锁住结果行。
    assert "| nano | 4 | 100 | 32 | 123.45 | 678.90 | 21.22 | 47.89 |" in table


def test_benchmark_result_json_serializable() -> None:
    """验证结果对象可以安全转成 JSON。"""

    result = BenchmarkResult(
        backend="hf",
        batch_size=2,
        prompt_tokens=64,
        output_tokens=16,
        ttft_ms=88.0,
        total_latency_ms=400.0,
        tpot_ms=25.0,
        throughput_tps=40.0,
    )

    payload = json.dumps(result.__dict__, ensure_ascii=False)

    assert '"backend": "hf"' in payload
    assert '"batch_size": 2' in payload
    assert '"throughput_tps": 40.0' in payload


if __name__ == "__main__":
    print("=" * 60)
    print("Day 7 benchmark 结构测试开始")
    print("=" * 60)

    test_build_parser_defaults()
    test_summarize_runs()
    test_format_results_table()
    test_benchmark_result_json_serializable()

    print("=" * 60)
    print("🎉 Day 7 benchmark 结构测试执行完成")
    print("=" * 60)
```

---

## 5.3 最后再回写 `readme.md`

修改位置：

- 文件：`nano_vll_repro/readme.md`
- 操作：
  - 修正错误示例命令
  - 把 Day6 / Day7 勾选状态改成真实状态
  - 用 `bench.py` 的真实输出替换性能表
  - 在性能表下补测试环境说明

README 至少要改 4 类内容。

### 第一类：删掉明显错误的启动命令

当前文档里的：

```bash
python example.py --model qwen3 --device cuda --max_tokens 128
```

和当前 `example.py` 真实形态不符，必须删掉。

如果你已经按前面文档把 `example.py` 改成单卡烟雾测试入口，那么 README 里最少应该写成：

```bash
python example.py
```

### 第二类：把“已完成但待测试”的自相矛盾清掉

例如当前 README 里：

1. Day6 被打勾
2. Day7 被打勾
3. 性能表却还是“待测试”

这三者不能同时成立。

### 第三类：把性能表替换成真实 benchmark 数据

推荐表头直接复用 `format_results_table()` 的列：

| Backend | Batch | Prompt Tokens | Output Tokens | TTFT (ms) | Total (ms) | TPOT (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|
| nano | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 |
| hf | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 |

### 第四类：补“测试环境说明”

至少写：

1. GPU 型号
2. CUDA 版本
3. PyTorch 版本
4. 测试模型
5. `batch_size / max_tokens / repeat`

否则别人无法理解你的结果。

---

## 5.4 再回写 `todo_list.md`

修改位置：

- 文件：`nano_vll_repro/todo_list.md`
- 操作：
  - 把“真实已完成”和“仍是目标”的状态拆开
  - 在文档尾部新增“最终验收”小节

`todo_list.md` 现在的问题不是“有点旧”，而是同时混着：

1. 已完成的历史记录
2. 还没做的目标
3. 和当前代码状态已经不一致的描述

建议回写原则：

1. 只有真正跑过并验证过的内容才勾选
2. Day7 相关项增加一行“benchmark 已完成，结果见 README”
3. 对仍然没做的项直接保留未勾选，不要写模糊状态

可以新增一个“最终验收”小节：

```markdown
### 最终验收

- [x] 单卡 smoke test 通过
- [x] Day1 ~ Day7 对应测试脚本通过
- [x] benchmark 已完成并回写 README
- [ ] 生产级多进程 worker 架构（当前仓库仍未实现）
- [ ] 更完整的 prefix-cache / chunked-prefill 对齐
```

这样会比“形式上全部勾满”真实得多。

---

## 6. 本篇结束后的最小验收

先做语法检查：

```bash
cd nano_vll_repro
python -m py_compile bench.py
python -m pytest tests/test_Day7.py -q
```

然后跑一次真实 benchmark：

```bash
python bench.py --backend both --batch_size 4 --max_tokens 64 --repeat 3
```

确认结果可信后，再回写：

```bash
# 手动根据真实输出更新 readme.md 和 todo_list.md
```

---

## 7. 常见错误

### 7.1 先改 README，再跑 benchmark

后果：

- 你会再次回到“文档宣称已完成，数据却还是假的”这种状态

### 7.2 在 `tests/test_Day7.py` 里直接跑大模型

后果：

- 测试耗时和硬件依赖暴涨
- 无法作为日常回归入口

### 7.3 `nano` 和 `hf` 路径使用不同 prompt / 不同采样参数

后果：

- 性能对比失去解释性
- 你根本不知道差异来自实现还是输入条件

### 7.4 benchmark 只输出总耗时，不输出 TTFT / TPOT

后果：

- 你看不到 decode 优化到底有没有收益
- Day6 的 CUDA Graph 收益很难被解释

### 7.5 继续保留 README 里的错误命令行

后果：

- 用户第一步就会照着旧命令报错
- 这类问题会直接伤害外部可用性判断

---

## 8. 本篇真正学到的东西

Day7 真正重要的不是“补了一个 bench.py”，而是下面 4 件事：

1. 工程收口和功能实现是两回事。
2. benchmark 脚本与回归测试脚本的职责必须分开。
3. 文档只能回写事实，不能回写希望。
4. 对外 README 的可信度，取决于它是否严格来自可重复执行的输出。

全套教案到这里结束。下一步如果你继续扩仓库，建议优先做：

1. prefix-cache 驱动的 partial prefill
2. 更完整的 TP worker 架构
3. 更系统的 benchmark 数据采样与可视化

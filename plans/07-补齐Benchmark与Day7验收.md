# 07. 补齐 Benchmark 与 Day7 验收（只回写真实数据，不回写想象中的完成状态）

## 1. 本篇目标

前面几篇把功能链路补齐以后，最后这一篇要解决的不是“再加一个新特性”，而是把仓库收口成可验证、可对外说明的状态。

你最后要回答 3 个问题：

1. 这套系统现在到底能不能稳定跑
2. 相比 Hugging Face，性能差异到底是多少
3. `readme.md` 和 `todo_list.md` 到底该如何回写成真实状态

本篇完成后，应至少满足：

- 有一个可运行的 `bench.py`
- 有一个轻量但有约束力的 `tests/test_Day7.py`
- `readme.md` 里的性能表来自真实测试，不再是占位词
- `todo_list.md` 的勾选状态和真实功能状态一致

---

## 2. 权威参考

本篇参考：

1. 当前仓库：
   - `readme.md`
   - `todo_list.md`
   - `llm.py`
   - `sampling_params.py`
2. 上游参考：
   - `nano-vllm/README.md`
3. HF 对照对象：
   - `transformers.AutoModelForCausalLM`
   - `transformers.AutoTokenizer`

这里要先把一个纪律写清楚：

> `README` 和 `todo_list` 只允许回写真实跑出来的结果，不允许继续保留“已完成但待测试”的伪完成状态。

---

## 3. 先看当前仓库的真实问题

### 3.1 `readme.md`

当前 README 存在明显自相矛盾：

- Day6 / Day7 被打勾
- 但性能表还是“待测试”

### 3.2 `todo_list.md`

问题和 README 类似：

- 勾选状态很激进
- 但没有真实 benchmark 数据支撑

### 3.3 缺少真正的 benchmark 入口

目前仓库还没有：

- `bench.py`
- TTFT / TPOT / throughput 的统一统计逻辑
- `nano` 与 `hf` 的可比对输出

### 3.4 缺少最终验收测试

Day7 不应该再新增一个“非常重的大模型全流程测试”，而应该新增一个：

- CLI / 输出结构 / 表格格式 / JSON 字段完整性

都能快速验证的收口测试文件。

---

## 4. 本篇修改原则

### 4.1 benchmark 脚本负责“测量”，测试文件负责“约束输出结构”

职责必须分开：

- `bench.py` 可以较重，因为它真的要跑模型
- `tests/test_Day7.py` 必须轻量，因为它要常跑

### 4.2 `nano` 和 `hf` 的采样参数必须尽量对齐

至少要统一这些项：

- prompt
- batch size
- `max_tokens`
- `temperature`
- `top_k`
- `top_p`

即便 HF 侧和你自写的 sampler 无法做到字节级完全一致，也要把差异写在输出说明里，而不是默认忽略。

### 4.3 文档回写只接受“跑完后的事实”

建议工作顺序永远是：

1. 写 `bench.py`
2. 跑真实数据
3. 保存结果
4. 再回写 `README`
5. 再回写 `todo_list`

不要倒过来。

---

## 5. 逐步修改

## 5.1 新建 `bench.py`，这里必须直接给完整文件

直接新建：

- `nano_vll_repro/bench.py`

完整代码如下。由于你要求新文件必须“完整可运行 + 高密度注释”，下面这份代码按这个标准来写。

```python
"""Day 7 benchmark 脚本

这个脚本只做一件事：
对比当前教学仓库的 `nano` 后端和 Hugging Face `hf` 后端在同一组输入条件下的推理指标。

输出指标包括：
1. TTFT（Time To First Token）
2. Total Latency
3. TPOT（Time Per Output Token）
4. Throughput（tokens / second）

注意：
1. 这不是精密论文 benchmark，只是教学仓库的工程验收脚本。
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


# 保证可以从仓库根目录直接导入项目模块。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class BenchmarkResult:
    """单个后端的一次聚合 benchmark 结果。

    这些字段既会被表格输出复用，也会被 JSON 输出复用。
    因此这里必须保持字段名稳定、语义清晰。
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
    """构造命令行参数解析器。

    这里先把所有可调参数收口到同一个 parser，
    后面的 `tests/test_Day7.py` 也会直接验证这些默认值。
    """

    parser = argparse.ArgumentParser(description="nano-vLLM / HF benchmark")

    # 模型路径默认沿用仓库当前约定。
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")

    # backend 可以只跑一个，也可以两者都跑。
    parser.add_argument("--backend", type=str, choices=["nano", "hf", "both"], default="both")

    # 下面这些参数控制输入规模与采样配置。
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)

    # warmup 与 repeat 分开设置，方便避开首次运行的冷启动扰动。
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)

    # `--json` 用于把结果直接喂给别的脚本或复制到日志里。
    parser.add_argument("--json", action="store_true")

    return parser


def build_chat_prompts(tokenizer, batch_size: int) -> list[str]:
    """构造一组固定的聊天 prompt。

    这里故意使用固定 prompt，而不是随机生成输入。
    原因是 Day7 的目标是“可复现的工程对比”，不是随机数据压测。
    """

    raw_prompts = [
        "请解释一下 PagedAttention 的核心思想。",
        "请解释一下 Continuous Batching 的优势。",
        "请解释一下 Tensor Parallelism 中 Row Parallel 的作用。",
        "请解释一下 CUDA Graph 为什么更适合 decode 阶段。",
    ]

    prompts: list[str] = []

    # 当 batch_size 大于预设 prompt 数量时，直接循环复用。
    for index in range(batch_size):
        raw_prompt = raw_prompts[index % len(raw_prompts)]

        # 沿用 Qwen chat template，保证 nano / hf 两侧输入形式一致。
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
    """把多次运行的原始耗时聚合成最终结果对象。"""

    mean_ttft = mean(ttft_runs)
    mean_total = mean(total_runs)

    # TTFT / Total 用毫秒展示，阅读更直观。
    ttft_ms = mean_ttft * 1000.0
    total_ms = mean_total * 1000.0

    # TPOT 定义为平均总时延除以平均输出 token 数。
    # 如果 output_tokens 为 0，就强制给一个极小值避免除零。
    tpot_ms = (mean_total / max(output_tokens, 1)) * 1000.0

    # Throughput 这里按总输出 token 数 / 总耗时来算。
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
    """把 benchmark 结果格式化成 Markdown 表格。

    这样你运行完 `bench.py` 后，结果可以直接复制进 README。
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


def _sync_if_cuda(torch_module) -> None:
    """在 CUDA 环境下同步 GPU，避免异步执行污染计时。"""

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _build_hf_generate_kwargs(args, max_new_tokens: int) -> dict:
    """构造 Hugging Face generate 的参数字典。

    这里故意单独抽函数，是为了把：
    - greedy
    - do_sample
    - top_k / top_p

    的分支逻辑写清楚，也便于 Day7 的结构化测试复用。
    """

    kwargs = {"max_new_tokens": max_new_tokens}

    # temperature = 0 代表 greedy。
    if args.temperature <= 0:
        kwargs["do_sample"] = False
        return kwargs

    kwargs["do_sample"] = True
    kwargs["temperature"] = args.temperature

    # HF 里 top_k=0 通常表示不启用该裁剪。
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k

    kwargs["top_p"] = args.top_p
    return kwargs


def run_nano_benchmark(args) -> BenchmarkResult:
    """运行当前教学仓库自己的 `nano` 后端 benchmark。"""

    import torch
    from transformers import AutoTokenizer

    from llm import LLM
    from sampling_params import SamplingParams

    model_path = os.path.join(PROJECT_ROOT, args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, args.batch_size)

    # prompt token 数直接按 tokenizer 编码后的总长度统计。
    prompt_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)

    llm = LLM(model_path)

    full_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # TTFT 近似为“只生成 1 个 token 的总耗时”。
    ttft_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=1,
    )

    # 先 warmup，避免首轮初始化、kernel 编译、缓存建立污染正式计时。
    for _ in range(args.warmup):
        _ = llm.generate(prompts, sampling_params=full_sampling_params, use_tqdm=False)
        _sync_if_cuda(torch)

    ttft_runs: list[float] = []
    total_runs: list[float] = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        first_token_outputs = llm.generate(prompts, sampling_params=ttft_sampling_params, use_tqdm=False)
        _sync_if_cuda(torch)
        ttft_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=full_sampling_params, use_tqdm=False)
        _sync_if_cuda(torch)
        total_runs.append(time.perf_counter() - start)

        # output_tokens 取这轮真实生成 token 总数。
        output_tokens = sum(len(item["token_ids"]) for item in outputs)

        # 明确删除临时结果，避免在长 benchmark 中保留无用引用。
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
    """运行 Hugging Face 对照 benchmark。"""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(PROJECT_ROOT, args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, args.batch_size)

    # HF 路径使用 padding 后的 batch 输入。
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 这里故意让 dtype 选择保持简单：
    # - CUDA + BF16 支持时优先 BF16
    # - 否则交给 HF 默认行为
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # prompt token 数按 attention_mask 统计，更接近真实输入 token 数。
    prompt_tokens = int(inputs["attention_mask"].sum().item())

    ttft_kwargs = _build_hf_generate_kwargs(args, max_new_tokens=1)
    total_kwargs = _build_hf_generate_kwargs(args, max_new_tokens=args.max_tokens)

    for _ in range(args.warmup):
        _ = model.generate(**inputs, **total_kwargs)
        _sync_if_cuda(torch)

    ttft_runs: list[float] = []
    total_runs: list[float] = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        _ = model.generate(**inputs, **ttft_kwargs)
        _sync_if_cuda(torch)
        ttft_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        generated = model.generate(**inputs, **total_kwargs)
        _sync_if_cuda(torch)
        total_runs.append(time.perf_counter() - start)

        # 输出 token 数取“生成后长度 - 输入长度”的总和。
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
    """脚本入口。"""

    parser = build_parser()
    args = parser.parse_args()

    results: list[BenchmarkResult] = []

    if args.backend in {"nano", "both"}:
        results.append(run_nano_benchmark(args))

    if args.backend in {"hf", "both"}:
        results.append(run_hf_benchmark(args))

    # JSON 输出适合日志系统或二次处理；
    # Markdown 表格输出适合直接复制进 README。
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(format_results_table(results))


if __name__ == "__main__":
    main()
```

## 5.2 新建 `tests/test_Day7.py`，也必须直接给完整文件

直接新建：

- `nano_vll_repro/tests/test_Day7.py`

完整代码如下。注意这份测试文件故意不跑真实模型，它只验证 benchmark 脚本的结构层是否稳定。

```python
"""Day 7 测试脚本 - benchmark 结构层验收

这份测试文件只验证 `bench.py` 的结构化输出层，不跑真实模型。

它要锁住的东西包括：
1. parser 默认值
2. summarize_runs 的统计公式
3. Markdown 表格输出格式
4. BenchmarkResult 的 JSON 序列化能力

这样做的原因很明确：
真实模型 benchmark 已经由 `bench.py` 负责；
测试文件本身必须保持轻量、快速、无额外硬件依赖。
"""

import json
import os
import sys


# 让测试文件能够直接导入 bench.py。
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


# 这里只导入结构层函数，不导入真实模型。
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

    # 这里故意给两组简单样本，便于人工核对。
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

    # 先断言表头存在。
    assert "| Backend | Batch | Prompt Tokens |" in table

    # 再断言结果行存在。
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

## 5.7 最后再回写 `readme.md`

修改位置：

- 文件：`nano_vll_repro/readme.md`
- 锚点 1：定位到 “Day 6 / Day 7 路线图勾选状态” 所在小节，把与真实状态不符的勾选项改掉
- 锚点 2：定位到 “性能数据” 表格，整张表替换为 `bench.py` 的真实输出结果
- 锚点 3：在性能表下方新增一段“测试环境说明”

README 至少要改 3 类内容。

### 一类：把“已完成但待测试”的自相矛盾删掉

例如原来的：

- Day6 已完成
- Day7 已完成
- 性能表全是“待测试”

这种状态必须清掉。

### 二类：把 benchmark 结果表替换成真实数据

推荐表头直接复用 `format_results_table()` 的列：

| Backend | Batch | Prompt Tokens | Output Tokens | TTFT (ms) | Total (ms) | TPOT (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|
| nano | 4 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 |
| hf | 4 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 | 实测值 |

### 三类：补一段“测试环境说明”

至少写：

- GPU 型号
- CUDA 版本
- PyTorch 版本
- 测试模型
- batch size / max_tokens / repeat

否则别人没法理解你的结果。

---

## 5.8 再回写 `todo_list.md`

修改位置：

- 文件：`nano_vll_repro/todo_list.md`
- 锚点 1：定位到 Day6 / Day7 的勾选项，把未跑过的项目取消勾选
- 锚点 2：在文档尾部新增一个“最终验收”小节，使用下面给出的替代文本

`todo_list.md` 不要再写成“形式上全部勾满、实际没有证据”。

建议回写原则：

1. 只有真正跑过的内容才勾选
2. Day7 相关项增加一行“已完成 benchmark，结果见 README”
3. 对仍然没做的项直接保留未勾选，而不是模糊写法

可以增加一个“最终收口”小节：

```markdown
### 最终验收

- [x] 单卡 smoke test 通过
- [x] Day1 ~ Day7 对应测试脚本通过
- [x] benchmark 已完成并回写 README
- [ ] 生产级多进程 worker 架构（当前仓库仍未实现）
- [ ] 更完整的 prefix-cache / chunked-prefill 对齐
```

这会比“全部打勾”真实得多。

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

- 你会再次得到“文档宣称已完成，数据却还是假的”这种状态

### 7.2 在 `tests/test_Day7.py` 里直接跑大模型

后果：

- 测试耗时和硬件依赖暴涨
- 无法作为日常回归入口

### 7.3 `nano` 和 `hf` 路径使用不同 prompt / 不同采样参数

后果：

- 性能对比失去解释性
- 你不知道差异来自实现，还是来自输入条件

### 7.4 benchmark 只输出总耗时，不输出 TTFT / TPOT

后果：

- 你看不到 decode 优化到底有没有收益
- Day6 的 CUDA Graph 收益很难被解释

---

## 8. 本篇真正学到的东西

Day7 真正重要的不是“补了一个 bench.py”，而是你要理解：

1. 工程收口和功能实现是两回事。
2. benchmark 脚本与回归测试脚本的职责必须分开。
3. 文档只能回写事实，不能回写希望。

全套教案到这里结束。下一步如果你继续扩仓库，建议优先做：

1. prefix-cache 驱动的 partial prefill
2. 更完整的 TP worker 架构
3. 更系统的 benchmark 数据采样与可视化

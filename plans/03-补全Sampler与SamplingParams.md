# 03. 补全 Sampler 与 SamplingParams（先做兼容迁移，再做功能升级）

## 1. 本篇目标

当前仓库的采样侧最大问题不是“少了几个算法名”，而是接口本身还停在只支持 `temperature` 的状态，导致：

1. `SamplingParams` 不足以表达真实生成策略
2. `Sampler.forward()` 的签名对后续 `top_k / top_p` 不友好
3. `Sequence` 还不会把 `top_k / top_p` 带进运行期状态
4. Day1 与 Day4 的测试假设已经和后续目标冲突

本篇完成后，你的仓库应该具备：

- `temperature`
- `top_k`
- `top_p`
- greedy / temperature / top-k / top-p 的统一入口
- 对旧调用方的过渡兼容能力

---

## 2. 权威参考

本篇只看 5 个文件：

1. 当前仓库：
   - `sampling_params.py`
   - `layers/sampler.py`
   - `engine/sequence.py`
   - `tests/test_Day1.py`
   - `tests/test_Day4.py`
2. 上游：
   - `nano-vllm/nanovllm/sampling_params.py`
   - `nano-vllm/nanovllm/layers/sampler.py`

这里要特别注意：

> 上游当前还只有 `temperature`，所以这一篇不是“抄上游”，而是在上游最小实现之上，为你当前仓库继续往 Day5 / Day7 推进补真实采样能力。

---

## 3. 先看当前仓库为什么已经不够用了

### 3.1 `sampling_params.py`

当前只有：

- `temperature`
- `max_tokens`
- `ignore_eos`

缺少：

- `top_k`
- `top_p`

而且 `__post_init__()` 还强制 `temperature > 0`，这会直接和“`temperature=0` 等价 greedy”这条常见推理路径冲突。

### 3.2 `layers/sampler.py`

当前实现已经做了 `temperature == 0` 的 greedy 特判，但：

- 接口签名只接收 `temperatures`
- `top_k / top_p` 完全没有入口
- 所有调用方以后都要被迫一起重写

### 3.3 `engine/sequence.py`

当前 `Sequence.__init__()` 只保存了：

- `temperature`
- `max_tokens`
- `ignore_eos`

如果这一步不把 `top_k / top_p` 也存进去，后面即使 `SamplingParams` 和 `Sampler` 都写好了，`ModelRunner` 仍然拿不到每条序列各自的采样配置。

### 3.4 `tests/test_Day1.py`

Day1 现在还在把 `SamplingParams(temperature=0)` 当成非法输入，这和你当前 `Sampler` 的设计已经矛盾了。

### 3.5 `tests/test_Day4.py`

Day4 只测了旧接口：

```python
tokens = sampler(logits, temps)
```

如果你这一步把 `Sampler.forward()` 直接硬改成四参数强制调用，旧测试会先全部炸掉。

---

## 4. 本篇修改原则

### 4.1 先做“兼容升级”，不要一步把所有调用点都改爆

推荐策略：

1. `Sampler.forward()` 新增 `top_ks / top_ps` 参数
2. 但让它们保持可选
3. 旧调用仍然合法，新调用可以逐步接入

这样 Day4 先不需要立刻全量迁移，Day5 的 `ModelRunner` 再正式把 `top_k / top_p` 带上。

### 4.2 `temperature=0` 在当前仓库里应被视为合法 greedy 配置

理由很简单：

- 你当前 `Sampler` 已经支持 greedy mask
- 推理框架里把 `temperature=0` 视为 greedy 是非常常见的约定

所以本篇必须同步改掉 Day1 的旧断言。

---

## 5. 逐步修改

## 5.1 先扩 `SamplingParams`，并修正校验规则

修改位置：

- 文件：`nano_vll_repro/sampling_params.py`
- 锚点：定位到 `@dataclass class SamplingParams:`，从类定义开始到 `__post_init__()` 结束，整段替换为下面这份完整类定义

推荐把 dataclass 调整成下面这类字段集合：

```python
@dataclass
class SamplingParams:
    """
    采样参数配置。

    输入：由用户在 generate 时传入的采样控制项。
    输出：一个经过校验的、可被 Sequence / ModelRunner / Sampler 统一消费的配置对象。

    这里要明确区分：
    - temperature: 控制分布锐度
    - top_k: 限制候选 token 个数
    - top_p: 限制累计概率覆盖范围
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 4096
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        # temperature 允许等于 0，表示 greedy。
        assert self.temperature >= 0.0, "temperature 必须 >= 0"

        # top_k = 0 约定为“不启用 top-k 裁剪”。
        assert self.top_k >= 0, "top_k 必须 >= 0"

        # top_p 必须在 (0, 1]，因为 0 没有实际语义。
        assert 0.0 < self.top_p <= 1.0, "top_p 必须在 (0, 1] 内"

        assert self.max_tokens > 0, "max_tokens 必须 > 0"
```

这里的关键不是字段本身，而是 3 条语义约定要写清楚：

1. `temperature=0` 合法，表示 greedy
2. `top_k=0` 合法，表示不启用 top-k
3. `top_p=1.0` 合法，表示不启用 top-p

---

## 5.2 同步扩 `engine/sequence.py`，把采样参数真正带进运行期

修改位置：

- 文件：`nano_vll_repro/engine/sequence.py`
- 锚点：定位到 `class Sequence.__init__()` 里 `#===采样参数===` 这一段，把现有的 3 行字段赋值替换为下面这 5 行

这一小步虽然代码量不大，但它决定了你后面能不能做 per-sequence sampling。

在 `Sequence.__init__()` 里，把现有采样字段：

```python
self.temperature = sampling_params.temperature
self.max_tokens = sampling_params.max_tokens
self.ignore_eos = sampling_params.ignore_eos
```

扩成：

```python
# 这些字段都来自 SamplingParams，但 Sequence 必须把它们固化成运行期状态。
# 原因是调度器、ModelRunner、后处理逻辑都只拿 Sequence，不会重新回看用户原始入参。
self.temperature = sampling_params.temperature
self.top_k = sampling_params.top_k
self.top_p = sampling_params.top_p
self.max_tokens = sampling_params.max_tokens
self.ignore_eos = sampling_params.ignore_eos
```

如果你愿意再多做一步，可以在 `Sequence` 类注释里补一句：

> 采样参数不是“外部配置的只读影子”，而是请求生命周期的一部分状态。

---

## 5.3 重写 `Sampler.forward()` 的签名，但保留旧调用兼容

修改位置：

- 文件：`nano_vll_repro/layers/sampler.py`
- 锚点 1：定位到 `class Sampler.forward`
- 锚点 2：如果类里还没有 `_apply_top_k` / `_apply_top_p`，就在 `forward` 之前插入
- 要求：下面给出的函数签名和函数体都是替代代码，不能只改参数名而保留旧逻辑

推荐签名：

```python
def forward(
    self,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor | None = None,
    top_ps: torch.Tensor | None = None,
) -> torch.Tensor:
```

为什么不要直接强制四参数：

- 你当前 `ModelRunner`、Day4 测试、手动调试片段都还在用旧接口
- 兼容迁移比“一刀切全部重写”更适合教学仓库

进入 forward 后，第一步先补默认值：

```python
if top_ks is None:
    top_ks = torch.zeros_like(temperatures, dtype=torch.long)

if top_ps is None:
    top_ps = torch.ones_like(temperatures, dtype=torch.float32)
```

这样旧调用方仍然等价于：

- 不启用 top-k
- 不启用 top-p

---

## 5.4 先写两个局部裁剪函数，再在 `forward()` 里串起来

不要把所有逻辑塞进一个巨大的 `forward()`。建议先拆两个 helper：

```python
def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    输入：
    - logits: 单条样本或批量样本的 logits
    - top_k: 保留前 K 个 token；0 表示不启用

    输出：裁剪后的 logits；未保留的位置填 -inf
    """
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    输入：
    - logits: 当前样本 logits
    - top_p: nucleus sampling 的累计概率阈值

    输出：只保留累计概率达到 top_p 前的最小集合
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    masked_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
    restored = torch.full_like(masked_sorted_logits, float("-inf"))
    restored.scatter_(dim=-1, index=sorted_indices, src=masked_sorted_logits)
    return restored
```

这里最容易漏掉的细节是 top-p 的“首 token 强制保留”：

- 如果累计概率第一项就超过了 `top_p`
- 仍然必须保留概率最高的那个 token

这就是：

```python
sorted_mask[..., 0] = False
```

存在的原因。

---

## 5.5 在 `forward()` 里先做温度，再做 top-k / top-p，最后统一采样

推荐流程写成下面这样：

```python
greedy_mask = temperatures == 0

safe_temperatures = temperatures.clone()
safe_temperatures[greedy_mask] = 1.0

scaled_logits = logits.float() / safe_temperatures.unsqueeze(dim=1)

# 对 batch 里的每一条样本分别应用 top-k / top-p；
# 这里不要偷懒写成一个全局超参数，否则 per-sequence sampling 参数就失效了。
filtered_logits = []
for row_logits, top_k, top_p in zip(scaled_logits, top_ks.tolist(), top_ps.tolist()):
    row_logits = self._apply_top_k(row_logits, int(top_k))
    row_logits = self._apply_top_p(row_logits, float(top_p))
    filtered_logits.append(row_logits)

filtered_logits = torch.stack(filtered_logits, dim=0)
probs = torch.softmax(filtered_logits, dim=-1)

# 仍然保留 Gumbel-Max，因为这比 multinomial 更接近你当前实现风格。
gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
sampled_tokens = (probs / gumbel_noise).argmax(dim=-1)

greedy_tokens = filtered_logits.argmax(dim=-1)
sampled_tokens = torch.where(greedy_mask, greedy_tokens, sampled_tokens)
return sampled_tokens
```

这里的教学重点有两个：

1. `temperature` 是对 logits 做缩放，不是对 softmax 概率再缩放
2. `top_k / top_p` 都应该作用在“温度缩放后的 logits”上

---

## 5.6 同步修 `tests/test_Day1.py`

修改位置：

- 文件：`nano_vll_repro/tests/test_Day1.py`
- 锚点：定位到 `def test_sampling_params()` 里“temperature=0 应该失败”的旧断言段，整段替换为下面这组新断言

这一步很多人会忘。你现在必须把 Day1 里关于 `temperature=0` 的旧断言改掉。

旧逻辑：

```python
try:
    SamplingParams(top_p=0.0)
    print("❌ 应该抛出异常但没有")
except AssertionError as e:
    print(f"✅ 正确拒绝 top_p=0: {e}")
```

改成检查新默认值和新字段：

```python
sp = SamplingParams()
assert sp.top_k == 0
assert sp.top_p == 1.0

sp2 = SamplingParams(temperature=0.0, top_k=20, top_p=0.9, max_tokens=128)
assert sp2.temperature == 0.0
assert sp2.top_k == 20
assert sp2.top_p == 0.9
```

如果你还想保留“非法参数会报错”的测试，请把目标改成真正非法的组合：

- `top_k=-1`
- `top_p=0`
- `max_tokens=0`

---

## 5.7 同步修 `tests/test_Day4.py`

修改位置：

- 文件：`nano_vll_repro/tests/test_Day4.py`
- 锚点：定位到 `def test_sampler()`，从 `sampler = Sampler()` 开始到函数结束，替换为下面这份新测试代码

这一篇建议把 Day4 的 sampler 测试分成两部分：

1. 旧接口兼容测试
2. 新接口能力测试

结构示例：

```python
@torch.inference_mode()
def test_sampler():
    from layers.sampler import Sampler

    sampler = Sampler()
    logits = torch.randn(4, 1000)
    temps = torch.tensor([0.0, 0.5, 1.0, 2.0])

    # 先验证旧接口仍然可用，避免这一步把所有调用方一次性改爆。
    tokens = sampler(logits, temps)
    assert tokens.shape == (4,)
    assert tokens[0] == logits[0].argmax()

    # 再验证新接口也已经接通。
    top_ks = torch.tensor([0, 10, 20, 50])
    top_ps = torch.tensor([1.0, 0.9, 0.8, 0.95])
    tokens2 = sampler(logits, temps, top_ks, top_ps)
    assert tokens2.shape == (4,)
```

这样做的好处是：

- 你没有把迁移成本全部压到同一篇
- 但采样器升级已经被测试真正锁住了

---

## 6. 本篇结束后的最小验收

先做语法检查：

```bash
cd nano_vll_repro
python -m py_compile sampling_params.py engine/sequence.py layers/sampler.py
```

再跑两个回归入口：

```bash
python tests/test_Day1.py
python tests/test_Day4.py
```

如果你只想先快速验证 sampler 行为，也可以执行：

```bash
python - <<'PY'
import torch
from layers.sampler import Sampler

sampler = Sampler()
logits = torch.randn(2, 100)
temps = torch.tensor([0.0, 0.8])
top_ks = torch.tensor([0, 10])
top_ps = torch.tensor([1.0, 0.9])
print(sampler(logits, temps, top_ks, top_ps))
PY
```

---

## 7. 常见错误

### 7.1 继续把 `temperature=0` 当非法输入

后果：

- `SamplingParams` 和 `Sampler` 语义互相打架
- 用户配置 greedy 时要绕一圈写 `1e-6` 这种脏技巧

### 7.2 把 `top_k / top_p` 写成全局单值，而不是 per-sequence 参数

后果：

- batch 内不同请求不能用不同采样参数
- 后面 `LLM.generate()` 的接口就失去意义

### 7.3 top-p 裁剪后忘了恢复原始 token 顺序

后果：

- 你是在排序后的索引空间里采样，不是在原始 vocab 空间里采样
- 采样结果会错位

### 7.4 一步强制所有调用点都改成四参数

后果：

- 还没开始做 Day5，你的所有旧测试就已经大面积报错
- 不利于定位“是 sampler 逻辑错了，还是调用方没迁移完”

---

## 8. 本篇真正学到的东西

这一篇最重要的不是“实现了 top-k / top-p”，而是你要理解：

1. 为什么接口迁移应该先做兼容，再做全面切换。
2. `temperature / top_k / top_p` 各自控制的是哪一层语义。
3. 为什么采样参数必须允许 per-sequence 生效。

完成后进入下一篇：

- [04-补齐单卡推理链路与Day5测试.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/04-补齐单卡推理链路与Day5测试.md)

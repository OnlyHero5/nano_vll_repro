# 03. 补全 Sampler 与 SamplingParams

## 1. 学习目标

这一篇解决两个问题：

1. `SamplingParams` 现在字段太少，无法支撑真正的文本采样
2. `layers/sampler.py` 的文档宣称支持 `top_k / top_p`，但实际并没有实现

这一篇完成后，你的仓库应当具备：

- `temperature`
- `top_k`
- `top_p`
- greedy / temperature / top-k / top-p 的统一采样入口

## 2. 先修知识

## 2.1 为什么不能只做 temperature

`temperature` 只是在缩放 logits 的尖锐程度。

它能控制“随机不随机”，但不能直接限制候选空间：

- `top_k`：只允许最高概率的前 K 个 token 参与采样
- `top_p`：只允许累计概率达到 P 的最小 token 集合参与采样

如果你只有 `temperature`，那模型仍可能从很长尾的 token 里采样，结果通常更飘。

## 2.2 Top-K 的原理

Top-K 很直接：

1. 找到概率最高的前 K 个 token
2. 其余 token 的 logits 置为 `-inf`
3. 再 softmax / 采样

好处：

- 约束强
- 容易理解

坏处：

- 候选数固定，不够自适应

## 2.3 Top-P 的原理

Top-P（nucleus sampling）更灵活：

1. 把 token 按概率排序
2. 取累计概率刚好超过 `p` 的最小集合
3. 只在这个集合里采样

好处：

- 候选空间会随分布形状变化
- 在尖锐分布和扁平分布下都更合理

## 3. 本仓库当前缺口

当前仓库在采样侧存在 3 个具体缺口：

1. `sampling_params.py` 里没有 `top_k / top_p`
2. `layers/sampler.py` 只做了 greedy 和 temperature
3. `tests/test_Day4.py` 里已经默认 `Sampler.forward` 可以接受 `top_k / top_p`

所以这一篇是必须做的。

## 4. 最终应修改的文件

- `sampling_params.py`
- `layers/sampler.py`

## 5. 完整代码

### 5.1 替换 `sampling_params.py`

```python
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 4096
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        assert self.temperature >= 0.0, "temperature 必须 >= 0"
        assert self.top_k >= 0, "top_k 必须 >= 0"
        assert 0.0 < self.top_p <= 1.0, "top_p 必须在 (0, 1] 内"
        assert self.max_tokens > 0, "max_tokens 必须 > 0"
```

### 5.2 替换 `layers/sampler.py`

```python
import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return logits

        values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 0] = False

        filtered_logits = logits.clone()
        filtered_logits.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits.masked_fill(sorted_mask, float("-inf")),
        )
        return filtered_logits

    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        batch_size = logits.shape[0]

        greedy_mask = temperatures == 0
        safe_temperatures = temperatures.clone().float()
        safe_temperatures[greedy_mask] = 1.0

        scaled_logits = logits / safe_temperatures.unsqueeze(-1)
        filtered_logits = torch.empty_like(scaled_logits)

        for i in range(batch_size):
            row = scaled_logits[i]
            row = self._apply_top_k(row, int(top_ks[i].item()))
            row = self._apply_top_p(row, float(top_ps[i].item()))
            filtered_logits[i] = row

        probs = torch.softmax(filtered_logits, dim=-1)
        gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sampled_tokens = (probs / gumbel_noise).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)

    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def sample_with_temperature(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        temperatures = torch.full(
            (batch_size,),
            temperature,
            device=logits.device,
            dtype=torch.float32,
        )
        top_ks = torch.zeros(batch_size, device=logits.device, dtype=torch.long)
        top_ps = torch.ones(batch_size, device=logits.device, dtype=torch.float32)
        return self.forward(logits, temperatures, top_ks, top_ps)
```

## 6. 手敲顺序

这一篇很简单：

1. 先改 `sampling_params.py`
2. 再改 `layers/sampler.py`

因为 `Sampler` 的调用协议是由 `SamplingParams` 决定的。

## 7. 最小验收方法

### 7.1 语法校验

```bash
python -m py_compile sampling_params.py layers/sampler.py
```

### 7.2 先跑 Day4

```bash
python tests/test_Day4.py
```

### 7.3 你应该能解释的 3 个问题

1. 为什么 `top_k` 和 `top_p` 不是一回事
2. 为什么先做 temperature，再做 top-k/top-p 过滤
3. 为什么 greedy 需要单独走 `temperature == 0` 分支

下一篇进入：

- [04-补齐单卡推理链路与Day5测试.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/04-补齐单卡推理链路与Day5测试.md)


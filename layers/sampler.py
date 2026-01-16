"""采样器

实现 LLM 的 token 采样策略。

采样方法：
1. Greedy: 直接取 argmax（temperature=0）
2. Temperature Sampling: 缩放 logits 后采样
3. Top-K: 只从概率最高的 K 个 token 中采样
4. Top-P (Nucleus): 从累积概率达到 P 的最小集合中采样

本实现使用 Gumbel-Max Trick 进行高效采样：
- 传统方法: softmax -> multinomial（两次 kernel 调用）
- Gumbel-Max: logits/temp -> add_gumbel -> argmax（可融合）

Gumbel-Max Trick 原理：
如果 G ~ Gumbel(0,1)，则 argmax(logits + G) 等价于从 softmax(logits) 采样
Gumbel(0,1) 可通过 -log(-log(U)) 生成，U ~ Uniform(0,1)
或者等价地：argmax(probs / Exp(1)) 其中 Exp(1) 是指数分布
"""

import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
            self,
            logits: torch.Tensor,
            temperatures: torch.Tensor
    ) -> torch.Tensor:
        """从 logits 采样下一个 token
        
        Args:
            logits: [batch_size, vocab_size] 模型输出的 logits
            temperatures: [batch_size] 每个序列的温度参数
                - temperature < 1: 更确定性（分布更尖锐）
                - temperature = 1: 原始分布
                - temperature > 1: 更随机（分布更平坦）
                - temperature = 0: 等价于 greedy（需特殊处理）
        
        Returns:
            [batch_size] 采样的 token ID
        """
        # 处理 temperature = 0 ，贪婪编码的情况
        # 创建一个mask记录一个batch里有哪些需要贪婪编码
        greedy_mask = (temperatures == 0)

        # 避免除0，
        safe_temperatures = temperatures.clone()
        safe_temperatures[greedy_mask] = 1.0

        # 温度缩放
        # logits:[batch, vocab], temperatures: [batch] -> [batch, 1]
        scaled_logits = logits.float() / safe_temperatures.unsqueeze(dim=1)

        # softmax得到概率分布
        probs = torch.softmax(scaled_logits, dim=-1)

        # Gumbel-Max Trick 采样
        # 等价于 torch.multinomial(probs, 1)，但更高效
        # Exp(1) 分布: torch.empty_like(probs).exponential_(1)
        # 为避免 log(0)，加上一个小量 clamp_min
        gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sampled_tokens = (probs / gumbel_noise).argmax(dim=-1)

        # 对于greedy的位置，直接argmax
        greedy_tokens = logits.argmax(dim=-1)
        # (避免出现if else 判断语句)
        sampled_tokens = torch.where(greedy_mask, greedy_tokens, sampled_tokens)

        return sampled_tokens
    
    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """贪婪编码"""
        return logits.argmax(dim=-1)
    
    def sample_with_temperature(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.Tensor:
        """单温度值采样值
        """
        batch_size = logits.shape[0]
        temperatures = torch.full(
            (batch_size,),
            temperature,
            device=logits.device,
            dtype=logits.dtype
        )
        return self.forward(logits, temperatures)


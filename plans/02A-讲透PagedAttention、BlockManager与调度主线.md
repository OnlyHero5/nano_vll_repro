# 02A. 讲透 PagedAttention、BlockManager 与调度主线

## 1. 本篇目标

这一篇不急着改代码，先把整条主线讲清楚。

你如果只记住函数名，不理解下面这些概念，后面几篇一定会越改越乱：

1. 什么是逻辑块，什么是物理块
2. `block_table` 到底在映射什么
3. `slot_mapping` 到底是给谁用的
4. Prefix Cache 为什么要靠块级哈希
5. Prefill 和 Decode 为什么不是一回事
6. 调度器为什么总是围着“KV Cache 够不够”在转

本篇读完以后，你应该至少能用自己的话说明：

1. 一个请求从 `Sequence` 开始，最后怎么走到 `Attention`
2. 为什么 vLLM 的关键不是“有 cache”，而是“cache 怎么分块、怎么复用、怎么调度”
3. 当前仓库里这些概念分别落在哪些文件里

---

## 2. 权威参考

本篇主要对照下面 6 个文件：

1. 当前仓库：
   - `engine/sequence.py`
   - `engine/block_manager.py`
   - `layers/attention.py`
   - `engine/scheduler.py`
   - `engine/model_runner.py`
   - `utils/context.py`
2. 上游主仓库：
   - `nanovllm/engine/sequence.py`
   - `nanovllm/engine/block_manager.py`
   - `nanovllm/layers/attention.py`
   - `nanovllm/engine/scheduler.py`
   - `nanovllm/engine/model_runner.py`

这里先说一个最关键的判断：

> vLLM 最核心的东西，不是某个线性层，也不是某个 sampler，而是“怎么把 KV Cache 变成可分页、可复用、可调度的资源”。

---

## 3. 先看一条请求到底怎么走

先别看源码，先看全流程。

用户发来一句话，比如：

```text
"你好，请解释一下 PagedAttention。"
```

在这套系统里，它大致会经历下面 8 步：

1. 文本被 tokenizer 变成 token 序列。
2. token 序列被封装成一个 `Sequence`。
3. `Scheduler` 决定这条 `Sequence` 现在能不能上车。
4. `BlockManager` 给它分配 KV Cache 对应的物理块。
5. `ModelRunner` 把一批 `Sequence` 拼成当前 step 的输入。
6. `Context` 把这批输入对应的元数据传给 `Attention`。
7. `Attention` 一边算注意力，一边把 K/V 写进 cache。
8. decode 阶段每多生成一个 token，就继续重复“调度 -> 准备输入 -> 写 cache -> 再算一步”。

如果只看这一串名字，你会觉得它们很多。

但其实核心只是在回答两个问题：

1. **这一批请求现在该不该算**
2. **这一批请求的 KV 应该写到显存的哪里**

---

## 4. 先把 4 个最容易混的概念讲明白

### 4.1 `Sequence` 不是“字符串”，而是“请求运行时状态”

在当前仓库里，[engine/sequence.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/sequence.py:33) 的 `Sequence` 里至少放了这些东西：

1. `token_ids`
2. `status`
3. `block_table`
4. `num_cached_tokens`
5. 采样参数

直白地说：

> `Sequence` 不是“用户原始输入”，而是“这个请求活到当前这一刻的全部运行时状态”。

所以它必须同时知道：

- prompt 有多长
- 现在总共有多少 token
- 生成到哪了
- KV Cache 映射到哪些物理块

### 4.2 逻辑块和物理块不是一回事

先看一个非常简单的例子。

假设：

- `block_size = 4`
- 某条序列现在有 10 个 token

那从这条序列自己的视角，它需要 3 个逻辑块：

```text
逻辑块 0: token 0 1 2 3
逻辑块 1: token 4 5 6 7
逻辑块 2: token 8 9
```

但 GPU 显存里真正分给它的物理块，可能是：

```text
逻辑块 0 -> 物理块 17
逻辑块 1 -> 物理块 203
逻辑块 2 -> 物理块 41
```

这张映射表，就是 `block_table`。

所以一句话记忆：

> `block_table` 不是“有哪些 token”，而是“这条序列的逻辑块现在落在哪些物理块上”。

### 4.3 `slot_mapping` 比 `block_table` 更细

`block_table` 管到“块”。

`slot_mapping` 管到“某个 token 最终写进 cache 的具体槽位”。

还是上面的例子。

如果：

- 物理块 17 对应 slots `68 69 70 71`
- 物理块 203 对应 slots `812 813 814 815`

那 token 的具体写入位置就是：

```text
token 0 -> slot 68
token 1 -> slot 69
token 2 -> slot 70
token 3 -> slot 71
token 4 -> slot 812
...
```

`layers/attention.py` 里的 Triton 写 cache kernel 真正需要的，就是这种“写到哪一个 slot”的信息。

所以再记一句：

> `block_table` 给调度和 decode 用，`slot_mapping` 给实际写 KV Cache 用。

### 4.4 Prefix Cache 复用的是“块”，不是“整条序列”

很多人第一次看 Prefix Cache，会误以为：

> “两条请求前缀相同，所以整条序列共享。”

这不对。

真正共享的是**已经填满、已经稳定、已经可以做内容哈希的块**。

也就是说：

1. 只有完整块适合进 prefix cache。
2. 最后一个没填满的块通常不能直接当稳定缓存复用。
3. 命中缓存时，复用的是现成物理块，而不是重新计算这部分 token。

---

## 5. Prefill 和 Decode 到底差在哪

这两个词很常见，但经常被说得太玄。

直接说人话：

### 5.1 Prefill

Prefill 处理的是：

> “这条请求已经有的那一整段 prompt。”

比如 prompt 有 120 个 token，那 Prefill 会一次把这 120 个 token 都喂进去。

这时最重要的事情是：

1. 把整段 prompt 对应的 K/V 写进 cache
2. 算出最后一个位置的 logits，给下一步生成用

### 5.2 Decode

Decode 处理的是：

> “这条请求最新生成出来的那个 token，再往后推一步。”

它不是再把整段 prompt 全算一遍。

它只会：

1. 取每条序列当前最后一个 token
2. 带着已有 cache 做下一步注意力
3. 把新 token 的 K/V 继续写进 cache

一句话总结：

> Prefill 是“第一次把上下文灌进去”，Decode 是“沿着已有 cache 一步一步往前走”。

---

## 6. 当前仓库里这几层是怎么接起来的

### 6.1 `Sequence` 负责“请求状态”

[engine/sequence.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/sequence.py:33)

它关心：

- 这条请求有哪些 token
- 当前状态是不是 `WAITING / RUNNING / FINISHED`
- 当前需要多少块
- 当前块表是什么

### 6.2 `BlockManager` 负责“物理块资源”

[engine/block_manager.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/block_manager.py:50)

它关心：

- 还有多少空闲块
- 某条序列现在能不能分配
- 追加 token 时要不要新开块
- 哪些完整块可以进入 prefix cache

### 6.3 `Scheduler` 负责“这一步算谁”

[engine/scheduler.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/scheduler.py:17)

它关心：

- waiting 队列
- running 队列
- 这一步是做 prefill 还是 decode
- 显存不够时要不要抢占

### 6.4 `ModelRunner` 负责“把这一批序列整理成模型能吃的输入”

[engine/model_runner.py](/home/psx/nano_vllm_repro/nano_vll_repro/engine/model_runner.py:32)

它关心：

- 当前 batch 的 `input_ids`
- `positions`
- `cu_seqlens`
- `slot_mapping`
- `context_lens`
- `block_tables`

### 6.5 `Context` 负责“把元数据传给 Attention”

[utils/context.py](/home/psx/nano_vllm_repro/nano_vll_repro/utils/context.py:24)

它不是业务对象，也不是调度器。

它只是一个全局数据容器，专门把当前 step 的 attention 元数据传下去。

### 6.6 `Attention` 负责“真正读写 KV Cache”

[layers/attention.py](/home/psx/nano_vllm_repro/nano_vll_repro/layers/attention.py:112)

它会做两件事：

1. 根据 `slot_mapping` 把当前 step 的 K/V 写进 cache
2. 根据当前是 prefill 还是 decode，选不同的 attention 路径

---

## 7. 三段关键代码，必须读懂

这一节不要求你马上改代码。

但你至少要看懂这些代码为什么长这样。

### 7.1 `Sequence.num_blocks`

文件：

- `engine/sequence.py`

这段代码的作用很简单：

> 计算当前序列一共需要几个逻辑块。

完整代码如下：

```python
@property
def num_blocks(self):
    """当前需要的总块数 = ceil(num_tokens / block_size)"""
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

为什么它重要？

因为后面几乎所有“要不要分配新块”的判断，都绕不过这个值。

### 7.2 `BlockManager.get_slot_mapping()`

文件：

- `engine/block_manager.py`

这段代码把“逻辑位置”翻译成“真实 cache slot”。

完整代码如下：

```python
def get_slot_mapping(self, seq: Sequence, start_pos: int = 0) -> list[int]:
    """计算 slot mapping（从 start_pos 开始的所有 token）"""
    slots = []
    for pos in range(start_pos, len(seq)):
        block_idx = pos // self.block_size
        offset = pos % self.block_size
        block_id = seq.block_table[block_idx]
        slots.append(block_id * self.block_size + offset)
    return slots
```

为什么这段代码一定要会看？

因为 `slot_mapping` 一旦算错，KV Cache 写入位置就会乱，后面的输出通常会直接坏掉。

### 7.3 `Scheduler.schedule()`

文件：

- `engine/scheduler.py`

这段代码是“当前 step 到底做 prefill 还是 decode”的入口。

完整代码如下：

```python
def schedule(self) -> Tuple[List[Sequence], bool]:
    """核心调度方法"""
    scheduled_seqs: List[Sequence] = []
    num_seqs = 0
    num_batched_tokens = 0

    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        new_tokens = len(seq) - seq.num_cached_tokens
        if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
            break
        if not self.block_manager.can_allocate(seq):
            break

        self.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)

        num_seqs += 1
        num_batched_tokens += new_tokens

    if scheduled_seqs:
        return scheduled_seqs, True

    decoded_seqs: List[Sequence] = []
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                victim = self.running.pop()
                self.__preempt(victim)
            else:
                self.__preempt(seq)
                break
        else:
            self.block_manager.append_slot(seq)
            decoded_seqs.append(seq)
            num_seqs += 1

    for seq in reversed(decoded_seqs):
        self.running.appendleft(seq)

    return decoded_seqs, False
```

你不用一遍就全记住。

但至少要看懂这 3 个判断：

1. waiting 能塞进去，就优先做 prefill
2. waiting 塞不进去，才轮到 decode
3. decode 不够块时，会触发 preempt

---

## 8. 最容易误解的 5 个点

### 8.1 “有 KV Cache”不等于“已经是 vLLM”

很多实现也有 KV Cache。

vLLM 真正特别的地方是：

1. 分块
2. 复用
3. 调度

也就是：

> cache 不是一个大数组，而是一种可以按块管理的资源。

### 8.2 Prefix Cache 不是简单字符串缓存

它不是“前缀一样就直接返回答案”。

它复用的是：

- 已经算过
- 已经填满
- 已经做过内容校验

的 block。

### 8.3 `block_table` 不是 attention mask

它只管：

- 某条序列的逻辑块映射到哪些物理块

它不直接表达因果关系，也不替代 mask 语义。

### 8.4 `slot_mapping` 不是只在 decode 才需要

prefill 也需要写 cache。

所以 prefill 一样要知道：

- 当前 step 这批 token 分别写到哪个 slot

### 8.5 调度器调的不是“字符串”，而是“显存预算”

`Scheduler` 看起来在排请求。

但本质上，它真正紧盯的是：

1. 当前 batch token 预算
2. 当前空闲块数量
3. 哪些序列继续跑是划算的

所以 Continuous Batching 的核心不是“把请求拼成一批”。

而是：

> 在有限 KV Cache 资源下，持续让 GPU 保持有活干。

---

## 9. 本篇结束后的最小验收

这一篇是理解篇，不是改代码篇。

最小验收不是跑测试，而是你自己能不能回答下面 6 个问题：

1. `Sequence` 为什么必须持有 `block_table`
2. 逻辑块和物理块有什么区别
3. `slot_mapping` 和 `block_table` 有什么区别
4. Prefix Cache 为什么按块复用，而不是按整条序列复用
5. Prefill 和 Decode 的输入为什么不同
6. `Scheduler` 为什么要和 `BlockManager` 强绑定

如果你 6 个里有 2 个以上答不清，先别急着进 `04`。

先回头把这篇重新过一遍。

---

## 10. 本篇真正学到的东西

这一篇最重要的不是记住某个类名。

而是记住下面 4 句话：

1. `Sequence` 管请求状态。
2. `BlockManager` 管物理块资源。
3. `Scheduler` 管“这一步算谁”。
4. `ModelRunner + Context + Attention` 管“这一批 token 怎么写进 cache、怎么继续往前走”。

只要这 4 句话在你脑子里是通的，后面再看单卡主循环、TP、CUDA Graph，就不会只剩接口记忆。

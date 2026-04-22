# 01. 修复当前主干并补齐 Linear 体系（按现有文件逐段修改）

## 1. 本篇目标

这一篇不是“重写整个线性层文件”，而是先把当前主干从“会导入失败 / 结构不完整”修到“后续 Day2 ~ Day6 都能继续接”的状态。

本篇完成后，至少要满足 4 个条件：

1. `layers/linear.py` 可以稳定导入，不再有断尾类定义。
2. `models/qwen3.py` 当前 import 的 `QKVLinear`、`MergedLinear`、`RowLinear` 都能在本地找到。
3. `utils/loader.py` 能把 HF 的分离权重写入你仓库里的融合参数。
4. `tests/test_Day4.py` 至少能覆盖 `QKV / Merged / Row` 这三类线性层的基本加载逻辑。

---

## 2. 权威参考

本篇只看 3 个来源：

1. 你的当前文件：
   - `nano_vll_repro/layers/linear.py`
   - `nano_vll_repro/utils/loader.py`
   - `nano_vll_repro/tests/test_Day4.py`
2. 上游参考：
   - `GeeeekExplorer/nano-vllm/nanovllm/layers/linear.py`
   - `GeeeekExplorer/nano-vllm/nanovllm/utils/loader.py`
3. 你的本地依赖方：
   - `nano_vll_repro/models/qwen3.py`

这里必须强调：

> 上游 `linear.py` 假设 `torch.distributed` 已初始化；你的仓库当前并没有这个前提，所以不能直接照抄。

---

## 3. 先看当前仓库到底断在哪里

先打开 `layers/linear.py`，你会看到 3 个核心问题：

1. `RowParallelLinear` 只有类头，没有完整实现。
2. 代码里定义的是 `QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear`，但 `models/qwen3.py` import 的却是 `QKVLinear / MergedLinear / RowLinear`。
3. 当前实现大量直接写 `dist.get_rank()` / `dist.get_world_size()`，这会让未初始化分布式的单卡测试直接炸掉。

再看 `utils/loader.py`，会发现它的主逻辑已经接近上游，但还差 3 个细节：

1. 缺一个统一的 `default_weight_loader` 兜底。
2. 对 packed weight 的参数不存在分支，错误处理不够清晰。
3. 普通参数和 packed 参数的边界没有在文档里说明白。

最后看 `tests/test_Day4.py`，你会发现它已经在帮你暴露接口漂移：

- 它只测了 `QKVLinear` 和 `MergedLinear`
- 还没有把 `RowLinear` 的 `all_reduce / bias` 路径纳入 smoke test
- sampler 仍然是旧接口，这一篇先不要升级到 `top_k / top_p`

---

## 4. 本篇修改原则

### 4.1 不整文件推翻，只补 3 类能力

本篇只改：

- `layers/linear.py`
- `utils/loader.py`
- `tests/test_Day4.py`

### 4.2 允许“单卡默认值”，因为现在仓库必须先能跑

你当前仓库大量 Day1 ~ Day5 测试都不会先调用 `init_process_group()`，因此这里一定要做“安全降级”：

- world size 未初始化时返回 `1`
- rank 未初始化时返回 `0`

### 4.3 先提供 alias，再谈完全改名

为了不连带改坏 `models/qwen3.py`，本篇先保留当前底层类名，同时新增上层 alias：

- `QKVLinear = QKVParallelLinear`
- `MergedLinear = MergedColumnParallelLinear`
- `RowLinear = RowParallelLinear`

这样后面 Day2/Day5 再继续改模型时，接口边界更稳。

---

## 5. 逐步修改

## 5.1 先在 `linear.py` 文件顶部补“单卡安全”的 TP 辅助函数

放置位置：

- `import` 之后
- `LinearBase` 之前

为什么先改这里：

- 你后面所有并行线性层都依赖 rank / world size
- 如果每个类各自写 fallback，会把逻辑分散到整个文件

把现有的 `divide()` 保留，然后在它下面补这三个辅助函数：

```python
def get_tp_world_size() -> int:
    """
    输入：无。
    输出：当前张量并行 world size。

    这里不能直接写 dist.get_world_size()，因为你当前仓库的 Day1~Day5
    绝大多数测试都不会先初始化分布式环境；未初始化时必须安全返回 1，
    才能让单卡路径把“并行层”退化成普通线性层。
    """
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_tp_rank() -> int:
    """
    输入：无。
    输出：当前张量并行 rank。

    同样地，单卡 / 未初始化分布式环境下必须返回 0；
    否则 import 阶段就会因为 get_rank() 抛错，后面的任何测试都无法开始。
    """
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    输入：
    1. param: 目标参数，来自当前模型。
    2. loaded_weight: 从 safetensors / HF 权重里读出来的张量。

    输出：无；就地把 loaded_weight 拷到目标参数里。

    注意这里显式转到 param 的 device 和 dtype。
    原因不是“好看”，而是为了让 CPU 读入的权重在单卡、BF16、FP16 等路径下都能统一工作。
    """
    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
```

改完以后，再把 `LinearBase.__init__()` 里的：

- `self.tp_rank = dist.get_rank() if ...`
- `self.tp_size = dist.get_world_size() if ...`

改成统一调用刚才的辅助函数。

---

## 5.2 把 `LinearBase` 补成真正的抽象基类，而不是半成品容器

当前文件已经有 `LinearBase`，但还有两个教学上必须补的点：

1. 需要统一初始化权重，保证 smoke test 的行为稳定。
2. 需要把 `weight_loader` 绑定到参数对象上，后面 `utils/loader.py` 才能无分支调用。

你不用重写整个类，只要在现有类里补一个 `reset_parameters()`，并在 `__init__()` 最后调用它。

建议加成下面这样：

```python
def reset_parameters(self) -> None:
    """
    输入：无。
    输出：无；原地初始化 self.weight / self.bias。

    这里沿用 PyTorch Linear 常见的 kaiming_uniform_ 初始化，
    目的不是追求训练最优，而是让当前仓库里的层在“未加载真实权重”的测试场景下
    也能有稳定、可解释的数值范围。
    """
    nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
    if self.bias is not None:
        bound = 1 / (self.weight.size(1) ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
```

这里的重点不是初始化公式本身，而是：

- 以后任何子类都不必自己各写一份初始化
- 没加载权重时，模型骨架测试不会因为 `torch.empty()` 的脏值而出现随机错误

---

## 5.3 把 `ColumnParallelLinear` 和 `MergedColumnParallelLinear` 改成“单卡也能走”的写法

你当前 `ColumnParallelLinear.__init__()` 里直接从 `dist.get_world_size()` 取 `tp_size`，这要改成统一调用 `get_tp_world_size()`。

只需要改 3 个点：

1. `tp_size` 来源改成 `get_tp_world_size()`
2. `weight_loader()` 里分片逻辑保留，但 copy 前转 device / dtype
3. `MergedColumnParallelLinear.weight_loader()` 里的 `chunk()` 也用同样的安全 copy

可以参考下面这段局部补丁：

```python
class ColumnParallelLinear(LinearBase):
    """列并行 Linear。

    输入：完整输入 hidden states。
    输出：当前 rank 对应的输出分片。

    教学上要注意：
    - 单卡时它会自然退化成普通 Linear；
    - 多卡时它按输出维度切分，所以 forward 本身通常不需要 all_reduce。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = get_tp_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias=bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # 这里切的是输出维度，对应 weight 的 dim=0。
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))
```

`MergedColumnParallelLinear` 的核心不是“重新发明一种并行层”，而是：

- 先把多个逻辑线性层沿输出维度拼起来
- 再在拼好的大矩阵内部，给每个 shard 找到自己的 offset

所以你要重点保留的是：

- `self.output_sizes`
- `shard_offset`
- `shard_size`

而不是盲目修改 forward。

---

## 5.4 把 `QKVParallelLinear` 写完整，同时补上 alias

这一步的目标不是让它“看起来像上游”，而是让本地 `Qwen3Attention` 真的能依赖它。

你应该在现有 `QKVParallelLinear` 上检查并修正 4 件事：

1. `total_num_kv_heads` 为空时要回退到 `total_num_heads`
2. `self.num_heads / self.num_kv_heads` 必须是“当前 rank 持有的本地头数”
3. `weight_loader(..., loaded_shard_id)` 要支持 `"q" / "k" / "v"`
4. 文件末尾必须补 alias，解决 `models/qwen3.py` 当前 import 失败的问题

文件末尾直接补：

```python
# 这三个 alias 的意义不是“多此一举”，而是为了兼容你当前模型代码已经写死的导入名。
# 先让主干稳定，再在后续文档里决定是否统一改名。
QKVLinear = QKVParallelLinear
MergedLinear = MergedColumnParallelLinear
RowLinear = RowParallelLinear
```

---

## 5.5 把 `RowParallelLinear` 补完整，这是当前主干真正断掉的地方

你当前文件在这里直接断尾，所以这一段必须完整补上。

为什么 `RowParallelLinear` 不能照着 `ColumnParallelLinear` 写？

- `ColumnParallelLinear` 按输出切，输入是完整的
- `RowParallelLinear` 按输入切，输入本身已经是分片
- 因此 `RowParallelLinear.forward()` 之后需要把各 rank 的部分和做 `all_reduce`

建议补成下面这种结构：

```python
class RowParallelLinear(LinearBase):
    """行并行 Linear。

    输入：当前 rank 持有的输入分片。
    输出：聚合后的完整输出。

    设计原因：
    - 这类层通常用于 attention 的 o_proj 和 MLP 的 down_proj；
    - 它们的输入本身已经被前面的列并行层切开了，所以这里按输入维度分片最自然。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = get_tp_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias=bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # bias 不分片，只有 weight 需要按输入维度切。
        if param.data.ndim == 1:
            param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            return

        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多卡时只让 rank0 保留 bias，可以避免每张卡都重复加一次 bias。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

这里最容易写错的是 bias：

- 如果每张卡都带 bias，再 `all_reduce`，bias 会被累加多次
- 所以只让 `rank0` 带 bias 是最简单、也最接近上游语义的做法

---

## 5.6 回到 `utils/loader.py`，把 packed weight 和普通 weight 的边界写清楚

这一步不需要重写逻辑，只要把当前实现收口成“任何参数都走同一入口”。

建议你按下面顺序检查：

1. 继续保留 `packed_modules_mapping = getattr(model, "packed_modules_mapping", {})`
2. 命中 packed name 时，先把 `weight_name` 替换成融合参数名
3. 然后统一调用目标参数上的 `weight_loader`
4. 非 packed 参数则退回 `default_weight_loader`

局部片段建议写成这样：

```python
def load_model(model: nn.Module, model_path: str) -> None:
    """
    输入：
    1. model: 目标模型；要求 packed_modules_mapping 与参数上的 weight_loader 已就位。
    2. model_path: Hugging Face safetensors 所在目录。

    输出：无；直接把权重写入 model。

    这里最重要的不是“遍历文件”，而是把“普通参数”和“融合参数”统一为同一种调用协议：
    最终都是 param.weight_loader(param, loaded_weight, *extra_args)。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file_path in glob(os.path.join(model_path, "*.safetensors")):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)

                for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
                    if original_name not in weight_name:
                        continue

                    param_name = weight_name.replace(original_name, packed_name)
                    param = model.get_parameter(param_name)
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
```

为什么这里要保留 `for ... else ...`：

- 因为 packed 命中是一种“短路成功”分支
- 没命中时才应该退回普通加载
- 这个结构比额外写一层 `is_packed` 标志更清楚

---

## 5.7 把 `tests/test_Day4.py` 改成真正能卡住回归的 smoke test

本篇先不要升级 sampler 到 `top_k / top_p`。这里应该只做线性层 smoke test。

建议补两类断言：

1. `RowLinear` 的 weight 分片或单卡复制行为
2. alias 名称确实能 import 成功

可以把测试的主体调整成下面这种结构：

```python
@torch.inference_mode()
def test_linear_layers():
    """
    这里先只做三件事：
    1. 验证 alias 名称存在；
    2. 验证 QKV / Merged 的 weight_loader 会把各分片写到正确偏移；
    3. 验证 RowLinear 至少能在单卡下完成一次前向传播。
    """
    from layers.linear import QKVLinear, MergedLinear, RowLinear

    qkv = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64)
    merged = MergedLinear(512, [1024, 1024])
    row = RowLinear(512, 256, bias=False)

    x = torch.randn(4, row.weight.shape[1])
    y = row(x)

    assert y.shape == (4, 256)
```

如果你愿意多补一步，再加一个显式断言：

- `hasattr(qkv.weight, "weight_loader")`
- `hasattr(merged.weight, "weight_loader")`
- `hasattr(row.weight, "weight_loader")`

这样 `utils/loader.py` 的调用协议就被真正锁住了。

---

## 6. 本篇结束后的最小验收

先做语法级验收：

```bash
cd nano_vll_repro
python -m py_compile layers/linear.py utils/loader.py
```

再跑本篇最小 smoke test：

```bash
python tests/test_Day4.py
```

如果你只想先确认 import 层面稳定，也可以先手动跑：

```bash
python - <<'PY'
from layers.linear import QKVLinear, MergedLinear, RowLinear
print(QKVLinear, MergedLinear, RowLinear)
PY
```

---

## 7. 常见错误

### 7.1 直接把上游 `dist.get_rank()` 抄过来

后果：

- 单卡未初始化分布式时直接崩
- Day2 / Day4 测试还没开始就失败

### 7.2 忘记给 alias

后果：

- `models/qwen3.py` 会继续 import 失败
- 你以为是模型问题，其实是线性层暴露名不一致

### 7.3 `RowParallelLinear` 所有 rank 都加 bias

后果：

- 多卡路径下 bias 被重复累加
- 输出数值会系统性偏大，不是随机误差

### 7.4 `default_weight_loader` 不做 dtype / device 对齐

后果：

- CPU 读到的权重在 BF16 / FP16 路径容易报 dtype mismatch
- 后面一旦引入 `config.torch_dtype`，这个坑会更明显

---

## 8. 本篇真正学到的东西

这一篇真正的重点不是“会写几个线性层类”，而是你要弄懂下面 3 件事：

1. 为什么列并行和行并行的切分维度不同。
2. 为什么 `weight_loader` 要绑定在参数对象上，而不是塞进 `utils/loader.py` 的大 if-else。
3. 为什么当前仓库必须先支持“未初始化分布式时安全退化为单卡”。

完成后进入下一篇：

- [02-补齐Qwen3模型主干与权重映射.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/02-补齐Qwen3模型主干与权重映射.md)

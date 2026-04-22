# 01. 修复当前主干并收口 Linear/Loader 接口（按当前仓库逐段修改）

## 1. 本篇目标

这一篇的任务不是“把 `layers/linear.py` 重写成 Tensor Parallel 版本”，而是先把当前主干里已经存在的单卡线性层、权重加载协议和 Day4 验收口径整理到一致。

本篇完成后，至少要满足下面 4 个条件：

1. `layers/linear.py` 暴露的公开类名与 `models/qwen3.py`、`tests/test_Day4.py` 完全一致。
2. `QKVLinear / MergedLinear / RowLinear` 的参数都遵守同一套 `weight_loader` 调用协议。
3. `utils/loader.py` 能清楚区分 packed 参数和普通参数，并把权重安全写入目标参数。
4. `tests/test_Day4.py` 至少能覆盖 `QKV / Merged / Row` 三类线性层的基本加载与前向行为。

这里先把边界写死：

> 本篇不引入 `ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear` 这类 TP 线性层。真正的 Tensor Parallel 改造放到 [05-实现Tensor-Parallel基础版.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/05-实现Tensor-Parallel基础版.md)。

---

## 2. 权威参考

本篇只看 4 个来源：

1. 当前仓库：
   - `nano_vll_repro/layers/linear.py`
   - `nano_vll_repro/utils/loader.py`
   - `nano_vll_repro/tests/test_Day4.py`
   - `nano_vll_repro/models/qwen3.py`
2. 上游教学实现：
   - `GeeeekExplorer/nano-vllm/nanovllm/layers/linear.py`
   - `GeeeekExplorer/nano-vllm/nanovllm/utils/loader.py`
3. Hugging Face 权重命名习惯：
   - `q_proj / k_proj / v_proj`
   - `gate_proj / up_proj`
4. 你当前仓库的运行边界：
   - Day1 ~ Day4 以单卡、单进程优先
   - 是否安装 `flash_attn` 会影响 `tests/test_Day4.py` 能否直接启动

这一篇最重要的判断标准不是“像不像上游 TP 实现”，而是：

> 文档中的类名、签名和修改步骤，必须与当前仓库真实存在的代码一致。

---

## 3. 先看当前仓库到底是什么状态

先打开 `layers/linear.py`，你会看到当前文件的真实情况和旧文档描述并不一致：

1. 文件里已经存在的是 `QKVLinear`、`MergedLinear`、`RowLinear`。
2. `models/qwen3.py` 当前 import 的也是 `QKVLinear`、`MergedLinear`、`RowLinear`，导入名本身并没有漂移。
3. 当前 `layers/linear.py` 里没有 `RowParallelLinear`、`QKVParallelLinear`、`MergedColumnParallelLinear` 这些类头。
4. 当前文件里也没有直接写 `dist.get_rank()` / `dist.get_world_size()`，所以“先补单卡安全 fallback”并不是这份代码眼前的主问题。

换句话说，旧版 `plans/01` 开头那段“先看当前仓库到底断在哪里”的判断，本身就是错的，不能继续沿用。

### 3.1 `layers/linear.py` 当前真正的问题

当前文件真正值得修的点主要有 4 个：

1. `QKVLinear` 只给 `self.weight` 绑定了 `weight_loader`，如果启用 bias，`self.bias` 没有绑定同样的加载器。
2. `QKVLinear._weight_loader()`、`MergedLinear._weight_loader()`、`RowLinear._weight_loader()`、`default_weight_loader()` 都是直接 `copy_`，没有统一做 device / dtype 对齐。
3. `MergedLinear` 的真实签名是 `MergedLinear(input_size, output_size, num_shards=2, bias=False)`，不是“传一个 list 作为多个输出尺寸”。
4. `RowLinear` 当前是完整的单卡版本，不应该在这一篇里被误写成“只有类头、等待补完”。

### 3.2 `utils/loader.py` 当前真正的问题

`utils/loader.py` 的主流程已经能工作，但文档描述得太乱，导致读者很容易误判：

1. packed 参数和普通参数其实已经共用一套“在参数对象上找 `weight_loader`”的协议。
2. 当前代码里还有多余的分支注释，好像 `RowLinear` 需要单独走特殊通道，实际上它和普通参数一样，只是恰好也绑定了自定义 loader。
3. 文档没有明确告诉读者：“packed 参数额外多一个 `shard_id`，普通参数没有。”

### 3.3 `tests/test_Day4.py` 当前真正的问题

这个测试文件也和旧文档说的不一样：

1. 它目前只测了 `QKVLinear` 和 `MergedLinear` 的权重写入。
2. 它没有覆盖 `RowLinear` 的 `weight_loader` 和前向输出形状。
3. 它没有锁定“参数对象必须挂有 `weight_loader`”这一条协议。
4. 它通过 `from layers.linear import ...` 导入时，会先执行 `layers/__init__.py`；如果环境里没有安装 `flash_attn`，测试甚至还没跑到线性层就会先失败。

---

## 4. 本篇修改原则

### 4.1 以当前公开接口为准，不虚构并行类名

这一篇只围绕下面 3 个真实存在的类展开：

- `QKVLinear`
- `MergedLinear`
- `RowLinear`

不要在本篇中引入别名兜底，也不要把“未来可能存在的 TP 类名”写成“当前文件已经有，只是没补完”。

### 4.2 本篇只修单卡主干，不提前做 TP

为什么这里必须克制：

1. 当前模型代码和测试代码都还在单卡语义上。
2. 如果你现在就把 `RowLinear` 改成“按输入维切分 + all_reduce”的写法，会把 `models/qwen3.py`、`tests/test_Day4.py` 和 Day2/Day4 的认知边界全部打乱。
3. Day5 才是专门引入 TP 头数语义和并行线性层的阶段。

### 4.3 把 `weight_loader` 当作统一协议，而不是散落在 `loader.py` 里的特殊分支

这篇真正要教会你的不是“写出某几个 if-else”，而是：

- packed 参数通过 `weight_loader(param, loaded_weight, shard_id)` 写入；
- 普通参数通过 `weight_loader(param, loaded_weight)` 写入；
- `utils/loader.py` 只负责分发，不负责知道每一种线性层内部怎么拼。

---

## 5. 逐步修改

## 5.1 先修 `layers/linear.py` 里的真实缺口

修改位置：

- 文件：`nano_vll_repro/layers/linear.py`
- 重点类：`QKVLinear`、`MergedLinear`、`RowLinear`
- 重点函数：`default_weight_loader`

这一段不要重写类层次结构，只补当前文件确实缺失的行为。

### 第一步：补一个统一的安全复制辅助函数

建议在文件底部 `default_weight_loader` 附近，或者在几个 loader 之前，统一抽一个小工具函数，例如：

```python
def copy_weight_to_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    输入：
    1. param: 当前模型中的目标参数。
    2. loaded_weight: 从 safetensors / HF 权重读出的张量。

    输出：无；原地把权重写入 param。

    这里统一做 device / dtype 对齐，避免 CPU FP32 权重直接写入 CUDA BF16/FP16 参数时报错。
    """
    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
```

为什么建议抽这个函数：

1. 现在 4 个 loader 都各自 `copy_`，以后很容易只修一处漏三处。
2. 你的仓库后面会开始明确区分 `config.torch_dtype`、CPU 加载与 GPU 参数，这个对齐逻辑迟早要有。

### 第二步：给 `QKVLinear` 的 bias 也绑定 `weight_loader`

当前 `QKVLinear.__init__()` 里只做了：

```python
self.weight.weight_loader = self._weight_loader
```

如果后面 `attention_bias=True`，那么 `qkv_proj.bias` 会没有同样的加载协议。这里要补成：

```python
self.weight.weight_loader = self._weight_loader
if self.bias is not None:
    self.bias.weight_loader = self._weight_loader
```

为什么这是必须修的真实问题：

1. `QKVLinear._weight_loader()` 对 weight 和 bias 都成立，本身并不依赖二维张量。
2. `models/qwen3.py` 已经保留了 `qkv_bias` 开关，这个分支不应该在真正打开时才暴露问题。

### 第三步：把 4 个 loader 都改成统一走安全复制

这 4 个地方都要同步改：

1. `QKVLinear._weight_loader()`
2. `MergedLinear._weight_loader()`
3. `RowLinear._weight_loader()`
4. `default_weight_loader()`

以 `QKVLinear._weight_loader()` 为例，当前写法是：

```python
param.data[shard_offset:shard_offset+shard_size].copy_(loaded_weight)
```

这里应该改成“先对齐，再写入”：

```python
target = loaded_weight.to(device=param.device, dtype=param.dtype)
param.data[shard_offset:shard_offset + shard_size].copy_(target)
```

`MergedLinear` 和 `RowLinear` 也是同一个原则，不要继续裸 `copy_`。

### 第四步：明确 `RowLinear` 这一篇只保持单卡语义

`RowLinear` 当前的说明已经基本正确：

- 单卡版本；
- 用于 `o_proj` 和 `down_proj`；
- 暂时不切分输入。

这里要做的是把文档里的误导删掉，而不是把它改造成 TP 版本。你只需要保留这条边界：

> `RowLinear` 在 01 中就是普通线性层包装；等到 05 再把它升级成真正的 row parallel。

---

## 5.2 回到 `utils/loader.py`，把加载协议写清楚

修改位置：

- 文件：`nano_vll_repro/utils/loader.py`

这一段不要重写外层流程，重点是把“packed 参数”和“普通参数”的协议解释清楚，并顺手把实现收口到更容易读的结构。

建议按下面顺序整理：

1. 保留 `packed_modules_mapping = getattr(model, "packed_modules_mapping", {})`
2. 命中 packed 参数时：
   - 把原始权重名替换成融合参数名；
   - 通过 `model.get_parameter()` 找到目标参数；
   - 调用 `param.weight_loader(param, loaded_weight, shard_id)`。
3. 没命中 packed 参数时：
   - 直接查同名参数；
   - 调用参数上的 `weight_loader`，若不存在则退回 `default_weight_loader(param, loaded_weight)`。

更清晰的局部结构大致应是：

```python
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

这里最关键的是读者要看明白：

1. packed 参数的额外信息只有一个 `shard_id`；
2. 普通参数和自定义线性层，最终都还是落到“参数对象自己的 loader”；
3. `loader.py` 不负责理解 QKV 拼接布局，它只负责把权重送到正确的参数对象。

### 一个容易被忽视但很重要的细节

如果你打算顺手提高稳定性，建议把：

```python
safetensor_files = glob(os.path.join(model_path, "*.safetensors"))
```

改成排序后的结果：

```python
safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
```

原因不是功能正确性，而是：

- 调试日志顺序更稳定；
- 多文件模型目录下更方便复现问题。

---

## 5.3 把 `tests/test_Day4.py` 改成真正能卡回归的 smoke test

修改位置：

- 文件：`nano_vll_repro/tests/test_Day4.py`

这一篇先不要升级 sampler 接口，也不要把 Day4 写成全模型测试。这里的重点只有线性层。

建议补下面 3 类断言：

### 断言 1：线性层参数对象都真的带有 `weight_loader`

这一条能直接锁定 `utils/loader.py` 的调用协议：

```python
assert hasattr(qkv.weight, "weight_loader")
assert hasattr(merged.weight, "weight_loader")
assert hasattr(row.weight, "weight_loader")
```

如果你把 `QKVLinear` 的 bias loader 一并补了，也应该顺手测：

```python
qkv_with_bias = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64, bias=True)
assert hasattr(qkv_with_bias.bias, "weight_loader")
```

### 断言 2：`RowLinear` 至少要覆盖一次加载和前向

当前文件缺的正是这一块。最低限度可以补成：

```python
row = RowLinear(512, 256, bias=False)
row_weight = torch.randn(256, 512)
row.weight.weight_loader(row.weight, row_weight)

x = torch.randn(4, 512)
y = row(x)

assert torch.allclose(row.weight.data, row_weight)
assert y.shape == (4, 256)
```

### 断言 3：示例代码要和真实签名一致

这一篇文档和测试里都要统一写成：

```python
merged = MergedLinear(512, 1024, num_shards=2)
```

不要再写成：

```python
merged = MergedLinear(512, [1024, 1024])
```

因为当前仓库里根本不是这个接口。

---

## 6. 本篇结束后的最小验收

先做语法级检查：

```bash
cd nano_vll_repro
python -m py_compile layers/linear.py utils/loader.py tests/test_Day4.py
```

再跑本篇 smoke test：

```bash
python tests/test_Day4.py
```

如果这里一启动就因为 `flash_attn` 缺失失败，不是线性层逻辑的问题，而是环境前置条件没满足。按照仓库根目录说明先安装依赖：

```bash
pip install flash-attn
```

如果你只想先确认导入名和构造签名，可以用一段最小脚本手动看：

```bash
python - <<'PY'
from layers.linear import QKVLinear, MergedLinear, RowLinear

qkv = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64)
merged = MergedLinear(512, 1024, num_shards=2)
row = RowLinear(512, 256, bias=False)

print(type(qkv).__name__, type(merged).__name__, type(row).__name__)
PY
```

---

## 7. 常见错误

### 7.1 把未来 TP 阶段的类名写进当前主干文档

后果：

- 读者一打开 `layers/linear.py` 就发现文档和代码完全对不上；
- 还没开始改代码，就先怀疑自己是不是看错了文件。

### 7.2 在 01 里提前引入 `dist.get_rank()` / `dist.get_world_size()`

后果：

- 当前主干的单卡语义被打乱；
- 后续读者会误以为 Day1 ~ Day4 已经建立在分布式前提上。

### 7.3 忘记给 `QKVLinear.bias` 绑定 `weight_loader`

后果：

- 一旦 `attention_bias=True`，加载流程会只修好 weight、漏掉 bias；
- 这种错误在默认配置下可能长期不暴露，但一开开关就炸。

### 7.4 直接把 CPU FP32 权重裸 `copy_` 到 CUDA BF16/FP16 参数

后果：

- 轻则 dtype mismatch；
- 重则某些路径下静默发生不一致，后面更难排查。

### 7.5 文档示例继续使用错误签名

典型错误例子：

```python
MergedLinear(512, [1024, 1024])
```

后果：

- 读者照抄就会报错；
- 你以为是实现问题，实际上只是文档把接口写错了。

---

## 8. 本篇真正学到的东西

这一篇真正要吃透的是下面 3 件事：

1. 当前仓库已经有单卡线性层骨架，问题在于加载协议和测试覆盖，而不是“并行类没补完”。
2. `weight_loader` 应该绑定在参数对象上，让 `utils/loader.py` 只做分发，不做布局判断。
3. 文档必须明确阶段边界：01 收口单卡线性层，05 才引入 Tensor Parallel。

完成后进入下一篇：

- [02-补齐Qwen3模型主干与权重映射.md](/home/psx/nano_vllm_repro/nano_vll_repro/plans/02-补齐Qwen3模型主干与权重映射.md)

# Repository Guidelines（仓库指南）

## 项目结构与模块组织
这是一个精简的 Python 推理原型。根目录入口主要有 `llm.py`、`config.py`、`sampling_params.py` 和 `example.py`。核心执行流程位于 `engine/`，包括调度、执行、KV Cache 管理和序列状态；算子与层实现放在 `layers/`；模型定义放在 `models/`；通用辅助逻辑放在 `utils/`。测试目前按里程碑拆分在 `tests/test_Day1.py` 到 `tests/test_Day4.py`。本地模型权重放在 `models/Qwen3-0.6B/`，不要把权重或缓存文件作为常规源码改动提交。

## 构建、测试与开发命令
- `python -m venv .venv && source .venv/bin/activate`：创建并启用本地虚拟环境。
- `pip install torch transformers flash-attn triton pytest`：安装 `readme.md` 中提到的核心依赖。
- `python example.py`：运行端到端推理示例，默认读取本地 `models/Qwen3-0.6B`。
- `pytest -q tests`：以 `pytest` 方式执行全部测试。
- `python tests/test_Day3.py`：单独运行某一天的测试脚本，便于查看详细打印信息。

## 代码风格与命名约定
遵循现有 Python 风格：4 空格缩进，导入放在模块顶部，函数、变量、文件名使用 `snake_case`，类名使用 `PascalCase`，例如 `Config`、`LLMEngine`。新增代码尽量按子系统归档到 `engine/`、`layers/`、`models/`、`utils/`，避免把不相关逻辑堆到根目录。仓库中没有单独的格式化配置，默认贴近 PEP 8，并保持现有注释风格简洁直接。

## 测试规范
新增测试优先放到对应的 `tests/test_Day*.py`，如果后续测试增多，可补充更通用的 `tests/test_*.py`。测试应尽量使用确定性的张量形状和轻量配置，避免无必要地依赖大模型权重。涉及 CUDA、FlashAttention 或多卡路径时，要在测试说明或 PR 描述里写清硬件前提。提交前先跑与改动最相关的测试，再跑一次 `pytest -q tests`。

## 提交与合并请求规范
现有提交历史以简短单行消息为主，且大多使用中文，例如 `修复bug`、`适配计算图录制后的清除操作`。继续保持简洁，但应尽量写出具体模块或行为，避免只有 `debug` 这类低信息量描述。PR 应说明改动目的、主要行为变化、验证命令、模型或 GPU 前提，并关联相关任务或问题；只有输出界面发生变化时才需要附截图。

## 配置与资产说明
`Config` 会在运行时校验模型路径，因此本地调试前先确认模型目录存在。不要提交下载的模型权重、缓存目录或机器相关的绝对路径配置。

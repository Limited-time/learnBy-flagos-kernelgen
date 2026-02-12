```md
# Get Start With FlagGems

## Introduction

FlagGems 是一个使用 Triton 语言实现的高性能通用操作符库。
它的目标是为 LLM 训练和推理提供一组内核函数。

通过在 PyTorch 的 ATen 后端进行注册，FlagGems 可以实现无缝过渡，
使用户无需修改模型代码即可切换到 Triton 函数库。
FlagGems 由 [FlagTree 编译器](https://github.com/flagos-ai/flagtree/)
支持，适用于不同的 AI 芯片集，并且支持 OpenAI Triton 编译器（适用于 NVIDIA 和 AMD）。

## Quick Installation

FlagGems 可以作为纯 Python 包或带有 C 扩展的包进行安装，
以获得更好的运行时性能。
默认情况下，它不会构建 C 扩展，请参阅 [安装](./installation.md) 了解如何使用 C++ 运行时。

### Install Build Dependencies

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

### 安装

将仓库克隆到本地环境：

```shell
git clone https://github.com/flagos-ai/FlagGems.git
```

然后使用以下命令触发安装：

```shell
cd FlagGems
# 如果你想使用原生 Triton 而不是 FlagTree，请跳过此步骤。
# 其他后端：替换为 requirements_backendxxx.txt
pip install -r flag_tree_requirements/requirements_nvidia.txt
pip install --no-build-isolation .
```

你还可以使用以下命令进行可编辑安装：

```shell
cd FlagGems
pip install --no-build-isolation -e .
```

此外，你还可以构建一个 wheel 进行安装。

```shell
pip install -U build
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
python -m build --no-isolation --wheel .
```

## 如何使用 Gems

### 导入

```python
# 永久启用 flag_gems
import flag_gems
flag_gems.enable()

# 或临时启用 flag_gems
with flag_gems.use_gems():
    pass
```

例如：

```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
with flag_gems.use_gems():
    C = torch.mm(A, B)
```

## 如何使用 实验性 Gems

`experimental_ops` 模块为尚未准备好投入生产的 新操作符提供空间。
这些操作符可以通过 `flag_gems.experimental_ops.*` 访问。
这些操作符遵循与核心操作符相同的开发模式。

```python
import flag_gems

# 全局启用
flag_gems.enable()
result = flag_gems.experimental_ops.rmsnorm(*args)

# 或作用域使用
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```
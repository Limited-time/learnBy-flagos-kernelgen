# FlagGems 手册

[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)

[中文版|[English](./README.md)]

## 介绍

FlagGems 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能通用算子库，
旨在为大语言模型提供一系列可应用于 PyTorch 框架的算子，加速模型面向多种后端平台的推理与训练。

FlagGems 通过对 PyTorch 的后端 ATen 算子进行覆盖重写，实现算子库的无缝替换，
一方面使得模型开发者能够在无需修改底层 API 的情况下平稳地切换到 Triton 算子库，
使用其熟悉的 PyTorch API 同时享受新硬件带来的加速能力，
另一方面对 kernel 开发者而言，Triton 语言提供了更好的可读性和易用性，可媲美 CUDA 的性能，
因此开发者只需付出较低的学习成本，即可参与 FlagGems 的算子开发与算子库建设。

## 特性

- 支持的算子数量规模较大
- 部分算子已经过深度性能调优
- 可直接在 Eager 模式下使用, 无需通过 `torch.compile`
- Pointwise 自动代码生成，灵活支持多种输入类型和内存排布
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 代码库已集成十余种后端
- C++ Triton 函数派发 (开发中)

更多特性细节可参阅 [./docs/features.md] 文档。

## 快速入门指南

### 安装指南

FlagGems 可以作为纯 Python 包或带有 C 扩展的包安装，以获得更好的运行时性能。默认情况下，它不会构建 C 扩展，有关如何使用 C++ 运行时的详细信息，请参考 [installation](./docs/installation.md)。

#### 安装构建依赖

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

#### 安装步骤

克隆仓库到本地环境：

```shell
git clone https://github.com/flagos-ai/FlagGems.git
```

然后使用以下命令触发安装：

```shell
cd FlagGems
# 如果您想使用原生 Triton 而不是 FlagTree，请跳过此步骤。
# 其他后端：替换为 requirements_backendxxx.txt
pip install -r flag_tree_requirements/requirements_nvidia.txt
pip install --no-build-isolation .
```

您还可以使用以下命令进行可编辑安装：

```shell
cd FlagGems
pip install --no-build-isolation -e .
```

此外，您可以构建一个 wheel 进行安装：

```shell
pip install -U build
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
python -m build --no-isolation --wheel .
```

### 使用方法

FlagGems 支持三种常见的使用模式，您可以根据具体场景选择合适的方式：

#### 1. 全局启用（推荐）

**何时使用**：当你希望整个应用都使用 FlagGems 加速时

执行 `flag_gems.enable()` 后，支持的 `torch.*` / `torch.nn.functional.*` 调用将会自动分发（Dispatch）到 FlagGems 的实现上。

```python
import torch
import flag_gems

# 全局启用 FlagGems
flag_gems.enable()

# 正常使用 PyTorch API，底层会自动使用 FlagGems 加速
x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
y = torch.mm(x, x)
```

#### 2. 上下文管理器（临时启用）

**何时使用**：当你只希望部分代码使用 FlagGems 加速时（如基准测试、比较性能）

如果你只想在某个作用域内使用 FlagGems，请使用上下文管理器：

```python
import torch
import flag_gems

# 仅在特定代码块中使用 FlagGems
with flag_gems.use_gems():
    x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
    y = torch.mm(x, x)

# 此处代码不会使用 FlagGems 加速
z = torch.add(x, y)
```

#### 3. 显式调用

**何时使用**：当你需要精确控制哪些算子使用 FlagGems 时

你也可以绕过 PyTorch 的分发机制，直接从 `flag_gems.ops` 中调用算子，此时无需调用 `enable()`：

```python
import torch
from flag_gems import ops
import flag_gems

# 直接调用 FlagGems 算子，无需启用
a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

## 高级用法

### 基本用法选项

使用 `FlagGems` 算子库时，您可以在运行计算之前导入并启用加速。您可以全局启用、选择性启用或临时启用。

#### 选项 1：全局启用

**何时使用**：当你希望整个脚本或交互会话都应用 FlagGems 优化时

要在整个脚本或交互会话中应用 `FlagGems` 优化：

```python
import flag_gems

# 全局启用所有 FlagGems 操作
flag_gems.enable()
```

启用后，代码中所有受支持的算子将自动替换为优化的 `FlagGems` 实现 - 无需进一步更改。

#### 选项 2：选择性启用

**何时使用**：当你只想加速一部分操作时

要仅启用特定算子并跳过其余算子：

```python
import flag_gems

# 仅启用选定的操作
flag_gems.only_enable(include=["rms_norm", "softmax"])
```

这在你只想加速模型中的特定部分时非常有用。

#### 选项 3：作用域启用

**何时使用**：当你需要更精细的控制时

对于更精细的控制，你可以使用上下文管理器在特定代码块内启用 `FlagGems`：

```python
import flag_gems

# 临时启用 flag_gems
with flag_gems.use_gems():
    # 此块内的代码将使用 Gems 加速的算子
    ...
```

这种作用域使用在以下情况下很有帮助：
- 基准测试性能差异
- 比较实现之间的正确性
- 在复杂工作流中有选择地应用加速

你还可以在上下文管理器中使用选择性启用：

```python
# 在作用域内仅启用特定操作
with flag_gems.use_gems(include=["sum", "add"]):
    # 只有 sum 和 add 会被加速
    ...

# 或排除特定操作
with flag_gems.use_gems(exclude=["mul", "div"]):
    # 除了 mul 和 div 之外的所有操作都会被加速
    ...
```

注意：`include` 参数的优先级高于 `exclude`。如果同时提供两者，则会忽略 `exclude`。

### 参数概览

| 参数      | 类型      | 描述                                         |
| --------- | --------- | -------------------------------------------- |
| `unused`  | List[str] | 禁用特定算子（用于 `enable`）                |
| `include` | List[str] | 仅启用特定算子（用于 `only_enable`）         |
| `record`  | bool      | 记录算子调用以进行调试或分析                 |
| `path`    | str       | 日志文件路径（仅在 `record=True` 时使用）    |

### 示例：选择性禁用特定算子

**何时使用**：当某个特定算子在你的工作负载中表现不如预期，或者你看到性能不佳并想暂时回退到原始实现时

你可以在 `enable()` 中使用 `unused` 参数来排除某些算子被 `FlagGems` 加速：

```python
flag_gems.enable(unused=["sum", "add"])
```

使用此配置，`sum` 和 `add` 将继续使用原生 PyTorch 实现，而所有其他受支持的算子将使用 `FlagGems` 版本。

### 示例：启用调试日志

**何时使用**：当你需要查看哪些算子被 FlagGems 加速时

启用 `record=True` 以在运行时记录算子使用情况，并使用 `path` 指定输出路径：

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

运行脚本后，检查日志文件（例如 `gems_debug.log`）以查看通过 `flag_gems` 调用的算子列表。

示例日志内容：

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

### 示例：查询已注册的算子

**何时使用**：当你需要调试或验证哪些算子处于活动状态时

启用 `FlagGems` 后，你可以查询已注册的算子：

```python
import flag_gems

flag_gems.enable()

# 获取已注册的函数名称列表
registered_funcs = flag_gems.all_registered_ops()
print("Registered functions:", registered_funcs)

# 获取已注册的算子键列表
registered_keys = flag_gems.all_registered_keys()
print("Registered keys:", registered_keys)
```

## 多平台支持

### 支持的平台

FlagGems 支持 NVIDIA 以外的多种 AI 芯片。有关最新的已验证平台列表，请参阅 [Supported Platforms](./docs/features.md#platforms-supported)。

### 统一使用接口

**何时使用**：当你需要在不同硬件平台间切换时

无论底层硬件如何，`flag_gems` 的使用方式完全相同。从 NVIDIA 平台切换到非 NVIDIA 平台时，无需修改应用程序代码。

一旦你调用 `import flag_gems` 并通过 `flag_gems.enable()` 启用加速，算子分发将自动路由到正确的后端。这在异构环境中提供了一致的开发体验。

### 后端要求

虽然使用模式不变，但在非 NVIDIA 硬件上运行需要底层依赖项 —— **PyTorch** 和 **Triton 编译器** —— 可用于目标平台并正确配置。

有两种常见方法可以获得兼容的构建：

1. **向硬件供应商请求**

   硬件供应商通常维护为其芯片定制的 PyTorch 和 Triton 构建。联系供应商以请求适当的版本。

2. **探索 FlagTree 项目**

   [FlagTree](https://github.com/flagos-ai/flagtree) 项目提供了一个统一的 Triton 编译器，支持一系列 AI 芯片，包括 NVIDIA 和非 NVIDIA 平台。它将供应商特定的补丁和增强功能整合到一个共享的开源后端中，简化了编译器维护并实现了多平台兼容性。

   > [!Note]
   > FlagTree 仅提供 Triton。仍然需要单独的匹配 PyTorch 构建。

### 后端自动检测和手动设置

默认情况下，`flag_gems` 会在运行时自动检测当前硬件后端并选择相应的实现。在大多数情况下，不需要手动配置，一切开箱即用。

**何时使用**：当自动检测失败或与你的环境不兼容时

如果你需要手动设置目标后端以确保正确的运行时行为，请在运行代码之前设置以下环境变量：

```shell
export GEMS_VENDOR=<your_vendor_name>
```

> ⚠️  此设置应与实际硬件平台匹配。手动设置不正确的后端可能会导致运行时错误。

你可以在运行时使用以下命令验证活动后端：

```python
import flag_gems
print(flag_gems.vendor_name)
```

## 与流行框架的集成

为了帮助将 `flag_gems` 集成到实际场景中，我们提供了与广泛使用的深度学习框架的示例。这些集成需要最少的代码更改，并保留原始工作流结构。

### 示例 1：Hugging Face Transformers

**何时使用**：当你使用 Hugging Face 模型时

集成 with Hugging Face 的 `transformers` 库非常简单。你可以简单地遵循前面部分介绍的基本使用模式。

在推理过程中，你可以激活加速而无需修改模型或分词器逻辑。这里是一个最小示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

# 将模型移至正确的设备并设置为评估模式
device = flag_gems.device
model.to(device).eval()

# 准备输入并在启用 flag_gems 的情况下运行推理
prompt = "Tell me a joke."
inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
with flag_gems.use_gems():
    output = model.generate(**inputs, max_length=100, num_beams=5)
```

此模式确保生成过程中使用的所有兼容算子都会自动加速。你可以在以下文件中找到更多示例：

- `examples/model_llama_test.py`
- `examples/model_llava_test.py`

### 示例 2：vLLM

**何时使用**：当你使用 vLLM 进行高吞吐量推理时

[vLLM](https://github.com/vllm-project/vllm) 是一种高吞吐量推理引擎，专为高效服务大型语言模型而设计。它支持分页注意力、连续批处理和优化的内存管理等功能。

`flag_gems` 可以集成到 vLLM 中，替换标准 PyTorch (`aten`) 操作和 vLLM 的内部自定义内核。

#### 在 vLLM 中替换标准 PyTorch 算子

要加速 vLLM 中的标准 PyTorch 操作（例如 `add`、`masked_fill`），你可以简单地使用与其他框架相同的方法：

- 在任何模型初始化或推理之前调用 `flag_gems.enable()`。
- 这将覆盖所有兼容的 PyTorch `aten` 操作，包括 vLLM 中间接使用的操作。

#### 替换 vLLM 特定的自定义算子

为了进一步优化 vLLM 的内部内核，`flag_gems` 提供了一个额外的 API：

```python
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```

此函数使用 `flag_gems` 实现修补某些 vLLM 特定的 C++ 或 Triton 算子。当 `verbose=True` 时，它将记录哪些函数被替换：

```none
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
Patched SiluAndMul.forward_cuda with FLAGGEMS custom_gems_silu_and_mul
```

当需要更全面的 `flag_gems` 覆盖时使用此功能。

## 性能优化

虽然 `flag_gems` 内核设计用于高性能，但在完整模型部署中实现最佳端到端速度需要仔细集成和考虑运行时行为。特别是，两个常见的性能瓶颈是：

- **生产环境中的运行时自动调优开销**。
- **由于框架级内核注册或与 Triton 运行时的交互而导致的次优调度**。

这些问题有时会抵消高度优化内核的好处。为了解决这些问题，我们提供了两条互补的优化路径，旨在确保 `flag_gems` 在实际推理场景中以最高效率运行。

### 预调优模型形状

**何时使用**：在生产环境中，特别是对延迟敏感的场景

`flag_gems` 集成了 [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/libentry.py#L139)，这是对 Triton 自动调优系统的轻量级增强。`libtuner` 引入了一个 **持久的、每设备调优缓存**，有助于减轻 Triton 默认自动调优过程的运行时开销。

#### 为什么需要预调优？

Triton 通常在新输入形状的前几次执行期间执行自动调优，这可能会导致延迟峰值 —— 尤其是在对延迟敏感的推理系统中。`libtune` 通过以下方式解决了这个问题：

- 持久缓存：最佳自动调优配置在运行之间保存。
- 跨进程共享：缓存在同一设备上的进程之间共享。
- 减少运行时开销：一旦调优，算子在未来运行中跳过调优。

这对于 `mm` 和 `addmm` 等经常触发 Triton 自动调优逻辑的算子特别有用。

#### 如何使用预调优

要主动预热系统并填充缓存：

1. 确定生产工作负载中使用的关键输入形状。
2. 运行预调优脚本以基准测试和缓存最佳配置：`python examples/pretune.py`
3. 正常部署，`flag_gems` 将在推理期间自动从缓存中选择最佳配置。

### 使用 C++ 包装器

**何时使用**：在延迟敏感或高吞吐量场景中

`flag_gems` 中的另一个高级优化路径是使用 **C++ 包装器** 来处理选定的算子。虽然 Triton 内核提供了相当好的计算性能，但 Triton 本身是一种 Python 嵌入式 DSL。这意味着算子定义和运行时调度都依赖于 Python，这可能会在对延迟敏感或高吞吐量的场景中引入 **不可忽视的开销**。

为了解决这个问题，我们提供了一个 C++ 运行时解决方案，将算子的包装逻辑、注册机制和运行时管理完全封装在 C++ 中，同时仍然重用底层 Triton 内核进行实际计算。这种方法在保持 Triton 内核级效率的同时，显著减少了与 Python 相关的开销，实现了与低级 CUDA 工作流的更紧密集成，并提高了整体推理性能。

#### 安装 & 设置

要使用 C++ 算子包装器：

1. 按照 [installation guide](./installation.md) 编译并安装 `flag_gems` 的 C++ 版本。

2. 使用以下代码片段验证安装是否成功：

   ```python
   try:
       from flag_gems import c_operators
       has_c_extension = True
   except Exception as e:
       c_operators = None  # 避免导入错误
       has_c_extension = False
   ```

   如果 `has_c_extension` 为 `True`，则 C++ 运行时路径可用。

3. 安装成功后，C++ 包装器将在 **补丁模式** 下自动优先使用，以及在显式使用 `flag_gems` 定义的模块构建模型时。

## 性能加速效果

FlagGems 在各种模型和硬件上都表现出了显著的性能加速。以下是一些性能对比示例：

![性能加速对比](./docs/assets/speedup-20251225.png)

## 常见问题和解决方案

### Q: FlagGems 支持哪些硬件平台？
A: FlagGems 支持多种 AI 芯片，包括 NVIDIA 和非 NVIDIA 平台。有关最新的已验证平台列表，请参阅 [Supported Platforms](./docs/features.md#platforms-supported)。

### Q: 如何在多 GPU 环境中使用 FlagGems？
A: 在单节点部署中，集成非常简单。你可以在脚本开始时导入并调用 `flag_gems.enable()`。在多节点部署中，每个进程都必须单独初始化 `flag_gems`。

### Q: 如何解决 FlagGems 与某些算子不兼容的问题？
A: 你可以使用 `flag_gems.enable(unused=["operator_name"])` 来禁用特定算子的加速，回退到 PyTorch 的原生实现。

### Q: 如何查看 FlagGems 是否正在加速我的模型？
A: 你可以使用 `flag_gems.enable(record=True, path="./gems_debug.log")` 来记录所有通过 FlagGems 加速的算子调用。

### Q: FlagGems 如何与 vLLM 集成？
A: 你可以使用 `flag_gems.enable()` 来加速标准 PyTorch 算子，并使用 `flag_gems.apply_gems_patches_to_vllm(verbose=True)` 来替换 vLLM 特定的自定义算子。

## 供测试的模型

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## 贡献代码

- 欢迎大家参与 FlagGems 的算子开发并贡献代码，详情请参考[CONTRIBUTING.md](./CONTRIBUTING_cn.md)。
- 欢迎提交问题报告（Issue）或者特性请求（Feature Request）
- 关于项目的疑问或建议，可发送邮件至<a href="mailto:contact@flagos.io">contact@flagos.io</a>。
- 我们为 FlagGems 创建了微信群。扫描二维码即可加入群聊！第一时间了解我们的动态和信息和新版本发布，
  或者有任何问题或想法，请立即加入我们！

  <img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/4e9a8566-c91e-4120-a011-6b5577c1a53d" />

## 引用

欢迎引用我们的项目：

```bibtex
@misc{flaggems2024,
    title={FlagOpen/FlagGems: FlagGems is an operator library for large language models implemented in the Triton language.},
    url={https://github.com/FlagOpen/FlagGems},
    journal={GitHub},
    author={BAAI FlagOpen team},
    year={2024}
}
```

## 许可证

本项目采用 [Apache License (version 2.0)](./LICENSE) 授权许可。

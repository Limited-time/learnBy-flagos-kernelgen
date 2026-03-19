# FlagGems 官方手册

> **前置阅读**：[递进认识flagOS](../递进认识flagOS.md)
> **目标读者**：普通用户、模型开发者
> **文档定位**：FlagGems 的官方使用手册

## 简介

FlagGems 是一个使用 OpenAI 推出的 [Triton 编程语言](https://github.com/openai/triton) 实现的高性能通用算子库，旨在为大语言模型提供一系列可应用于 PyTorch 框架的算子，加速模型面向多种后端平台的推理与训练。

### 技术路线

FlagGems 通过对 PyTorch 的后端 ATen 算子进行覆盖重写，实现算子库的无缝替换，使用户能够在不修改模型代码的情况下平稳地切换到 Triton 算子库。FlagGems 不会影响 ATen 后端的正常使用。

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch 高层 API                          │
│                  (torch.nn, torch.optim...)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       ATen 库                                │
│              (张量计算、底层硬件通信)                          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  CUDA    │   │FlagGems  │   │  其他    │
        │ (原生)   │   │ (Triton) │   │  后端    │
        └──────────┘   └──────────┘   └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌──────────────┐
                    │   GPU 硬件   │
                    └──────────────┘
```

**核心原理**：
- 在 PyTorch 中，核心的张量操作以及底层硬件通信是由 ATen 库实现的
- 当 ATen 需要执行一些可以在 GPU 上加速的操作时，它会通过 CUDA 来调用 GPU 的资源
- FlagGems 通过注册机制替换 ATen 的算子实现，实现无缝加速

### 核心特性

- ✅ 支持的算子数量规模较大
- ✅ 部分算子已经过深度性能调优
- ✅ 可直接在 Eager 模式下使用，无需通过 `torch.compile`
- ✅ Pointwise 自动代码生成，灵活支持多种输入类型
- ✅ 灵活的多后端支持机制
- ✅ 代码库已集成十余种后端

### 文档导航

| 文档 | 内容 | 适用场景 |
|------|------|---------|
| [快速入门](quick_start.md) | 安装、配置、第一个示例 | 5分钟上手 |
| [用户指南](user_guide.md) | 高级用法、多平台支持、框架集成 | 日常使用 |
| [性能优化](performance.md) | 预调优、C++运行时、性能加速效果 | 追求极致性能 |

## 快速开始

### 安装

```bash
# 创建虚拟环境
conda create -n flaggems python=3.10
conda activate flaggems

# 安装PyTorch（根据硬件选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装FlagGems
pip install flag-gems

# 验证安装
python -c "import flag_gems; print(flag_gems.__version__)"
```

### 第一个示例

```python
import torch
import flag_gems

# 启用FlagGems（只需一行代码）
flag_gems.enable()

# 正常使用PyTorch，自动加速
x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
y = torch.mm(x, x)
print(f"矩阵乘法完成，结果形状: {y.shape}")
```

## 三种使用方式

### 方式1：全局启用（推荐新手）

```python
import flag_gems
flag_gems.enable()  # 全局启用

# 之后所有PyTorch操作自动使用FlagGems加速
x = torch.randn(1024, 1024).cuda()
y = torch.mm(x, x)  # 自动加速
```

### 方式2：局部启用（推荐测试）

```python
import flag_gems

with flag_gems.use_gems():
    x = torch.randn(1024, 1024).cuda()
    y = torch.mm(x, x)  # 加速

z = torch.mm(x, x)  # 不加速
```

### 方式3：选择性启用（推荐优化）

```python
import flag_gems

# 只加速特定算子
flag_gems.only_enable(include=["mm", "addmm", "rms_norm"])

# 或禁用特定算子
flag_gems.enable(unused=["sum", "add"])
```

## 多平台支持

### 支持的硬件平台

| 厂商 | 型号 | 支持状态 |
|------|------|---------|
| NVIDIA | A100, H100, V100 | ✅ 完全支持 |
| 华为 | 昇腾910B | ✅ 完全支持 |
| 天数 | T10, T20 | ✅ 完全支持 |
| 海光 | DCU | ✅ 完全支持 |
| 摩尔线程 | S80, S3000 | ✅ 完全支持 |

### 自动检测硬件

```python
import flag_gems
print(f"当前后端：{flag_gems.vendor_name}")
```

### 手动设置后端

```bash
export GEMS_VENDOR=nvidia  # 或 huawei、cambricon 等
```

## 参考资源

- [FlagGems GitHub](https://github.com/FlagOpen/FlagGems)
- [Triton 官方文档](https://triton-lang.org/main/index.html)
- [开发者实践指南](../developer/CONTRIBUTING.md)

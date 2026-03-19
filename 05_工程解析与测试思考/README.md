# FlagOS 文档中心

欢迎来到 FlagOS 文档中心！本文档库为不同角色的用户提供全面的 FlagOS 学习和使用指南。

## 📚 文档导航

### 🌐 生态深度解析系列 (ecosystem/)

从七个不同角色视角深入理解 FlagOS：

| 序号 | 文档 | 目标读者 | 内容概述 |
|------|------|----------|----------|
| 01 | [架构师视角](ecosystem/01-architect.md) | 系统架构师 | 系统设计理念、架构原则、技术选型 |
| 02 | [开发者视角](ecosystem/02-developer.md) | 软件开发工程师 | API 使用、开发工作流、调试技巧 |
| 03 | [运维工程师视角](ecosystem/03-ops.md) | 运维工程师 | 部署、监控、故障排查 |
| 04 | [评测工程师视角](ecosystem/04-benchmark.md) | 性能测试工程师 | 评测方法、性能分析、对比测试 |
| 05 | [硬件厂商视角](ecosystem/05-vendor.md) | 硬件厂商 | 硬件适配、驱动开发、认证流程 |
| 06 | [模型使用者视角](ecosystem/06-model-user.md) | AI 模型使用者 | 模型部署、推理优化、服务化 |
| 07 | [研究人员视角](ecosystem/07-researcher.md) | 学术研究人员 | 研究平台、实验环境、数据分析 |

### 💎 FlagGems 官方手册 (flaggems/)

FlagGems 是 FlagOS 的高性能算子库，支持多种 AI 芯片：

| 文档 | 说明 |
|------|------|
| [README](flaggems/README.md) | 概览与导航 |
| [快速开始](flaggems/quick_start.md) | 5分钟上手教程 |
| [用户指南](flaggems/user_guide.md) | 完整功能文档 |
| [性能优化](flaggems/performance.md) | 性能调优指南 |

**核心特性**：
- 支持 NVIDIA、昇腾、天数智芯、海光、摩尔线程等硬件
- 无需修改代码，开箱即用
- 部分算子经过深度性能调优

### 🛠️ 开发者实践指南 (developer/)

为 FlagOS 贡献者提供的开发文档：

| 文档 | 说明 |
|------|------|
| [CONTRIBUTING](developer/CONTRIBUTING.md) | 贡献指南与规范 |
| [算子开发模板](developer/operator_template.md) | 标准算子开发模板 |
| [KernelGen 集成](developer/kernelgen_integration.md) | 代码生成工具集成 |
| [故障排查](developer/troubleshooting.md) | 常见问题与解决方案 |

### 📝 个人实践案例 (examples/)

使用 KernelGen 进行算子开发的实践案例：

| 案例 | 说明 |
|------|------|
| [broadcast](examples/broadcast/) | Broadcast 算子优化实践 |
| [minimum](examples/minimum/) | Minimum 算子优化实践 |

### 📖 递进认识 FlagOS

[递进认识 FlagOS](递进认识flagOS.md) - 循序渐进的学习路径

## 🚀 快速开始

### 1. 安装 FlagGems

```bash
pip install flag-gems
```

### 2. 启用加速

```python
import flag_gems
flag_gems.enable()

# 正常使用 PyTorch，自动加速
import torch
x = torch.randn(4096, 4096, device='cuda')
y = torch.mm(x, x)
```

### 3. 验证安装

```python
import flag_gems
print(f"FlagGems 版本: {flag_gems.__version__}")
print(f"当前后端: {flag_gems.vendor_name}")
```

## 📖 学习路径

```
入门阶段
├── 阅读 [递进认识 FlagOS](递进认识flagOS.md)
├── 完成 [快速开始](flaggems/quick_start.md)
└── 了解 [用户指南](flaggems/user_guide.md)

进阶阶段
├── 选择角色阅读 [生态深度解析](ecosystem/)
├── 学习 [性能优化](flaggems/performance.md)
└── 参考 [实践案例](examples/)

贡献阶段
├── 阅读 [贡献指南](developer/CONTRIBUTING.md)
├── 学习 [算子开发模板](developer/operator_template.md)
└── 参考 [KernelGen 集成](developer/kernelgen_integration.md)
```

## 🤝 参与贡献

我们欢迎各种形式的贡献：

- 提交 Issue 报告问题
- 提交 PR 改进文档
- 分享您的使用案例

## 📞 联系我们

- FlagGems GitHub: [https://github.com/FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)
- KernelGen 平台: [https://kernelgen.flagos.io](https://kernelgen.flagos.io)
- Triton 官方文档: [https://triton-lang.org](https://triton-lang.org)

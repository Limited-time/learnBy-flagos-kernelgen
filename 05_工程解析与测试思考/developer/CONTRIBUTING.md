# FlagGems 贡献指南

> 本指南面向需要**研究源码、贡献算子、适配新硬件**的开发者。基础使用请参考 [FlagGems官方手册](../flaggems/README.md)

## 一、贡献流程概览

### 1.1 完整流程图

```
┌──────────────────┐
│ 1. Fork 仓库      │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 2. Clone 到本地   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 3. 创建开发分支   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 4. 安装 pre-commit│
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 5. 开发代码       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 6. 本地测试       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 7. 提交 PR       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 8. 通过 CI 测试   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 9. Code Review   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 10. Merge        │
└──────────────────┘
```

### 1.2 详细步骤

#### Step 1: Fork 仓库

打开 [FlagGems GitHub](https://github.com/FlagOpen/FlagGems)，点击 **Fork** 按钮创建仓库副本。

#### Step 2: Clone 到本地

```bash
git clone https://github.com/YOUR_USERNAME/FlagGems.git
cd FlagGems
```

#### Step 3: 创建开发分支

```bash
git checkout -b feature/my-new-op
```

#### Step 4: 安装 pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit  # 手动运行检查
```

> **重要**：`pre-commit` 是 CI 的一部分，不通过检查的 PR 无法合并。

#### Step 5: 开发代码

```bash
git status
git diff
```

#### Step 6: 本地测试

```bash
cd tests
pytest test_xx_ops.py                # CUDA
pytest test_xx_ops.py --device cpu   # CPU

cd examples
pytest model_xx_test.py

cd benchmark
pytest test_xx_perf.py -s            # kernel 性能
pytest test_xx_perf.py -s --mode cpu # e2e 性能

pytest program.py --log-cli-level debug
```

#### Step 7: 提交代码

```bash
git add <file>
git commit -m "feat: add new operator xxx"
git push origin feature/my-new-op
```

#### Step 8: 提交 PR

1. 打开你 Fork 的 FlagGems 页面
2. 切换到你的分支
3. 点击 **Compare & pull request** 按钮
4. 填写 PR 描述（说明做了什么、为什么做）

#### Step 9: 等待 CI 测试

| CI 流水线 | 检查内容 |
|----------|---------|
| 代码格式检查 | pre-commit 静态检查 |
| 算子单元测试 | pytest 测试正确性 |
| 模型测试 | 模型集成测试 |
| 代码覆盖率检查 | 测试覆盖率 |

#### Step 10: Code Review

CI 通过后，等待维护者 Review。根据反馈修改代码，回复评审意见。

#### Step 11: Merge

Review 通过后，维护者会合并你的 PR。

### 1.3 保持代码最新

```bash
git remote add upstream https://github.com/FlagOpen/FlagGems.git
git fetch upstream
git rebase upstream/master
```

## 二、FlagGems 架构概览

### 2.1 核心设计理念

| 理念 | 说明 | 代码体现 |
|------|------|----------|
| **算子复用** | 一套 Triton 代码适配多硬件 | `src/flag_gems/ops/` |
| **后端抽象** | 厂商差异隔离在后端层 | `src/flag_gems/runtime/backend/` |
| **自动调优** | LibTuner 缓存最优配置 | `src/flag_gems/utils/libentry.py` |

### 2.2 仓库目录结构

```
FlagGems/
├── src/flag_gems/
│   ├── __init__.py              # 入口：enable(), use_gems()
│   ├── ops/                     # 通用算子实现（Triton）
│   │   ├── mm.py               # 矩阵乘法
│   │   ├── rms_norm.py         # RMS归一化
│   │   ├── softmax.py          # Softmax
│   │   └── ...
│   │
│   ├── runtime/
│   │   ├── __init__.py
│   │   └── backend/             # 后端适配层
│   │       ├── _nvidia/        # NVIDIA GPU
│   │       ├── _cambricon/     # 寒武纪
│   │       ├── _ascend/        # 昇腾
│   │       └── ...
│   │
│   ├── fused/                   # 融合算子
│   ├── modules/                 # 高性能模块
│   └── utils/                   # 工具函数
│
├── tests/                       # 单元测试
├── benchmark/                   # 性能基准
├── examples/                    # 使用示例
└── docs/                        # 文档
```

### 2.3 调用链路

```
用户代码: torch.mm(x, y)
    ↓
PyTorch ATen 调度
    ↓
FlagGems 注册拦截 (flag_gems.enable())
    ↓
后端选择 (NVIDIA/AMD/寒武纪...)
    ↓
Triton 内核执行
    ↓
返回结果
```

## 三、适配新硬件后端

### 3.1 后端目录结构

```
src/flag_gems/runtime/backend/_newvendor/
├── __init__.py              # 后端初始化
├── ops/                     # 厂商定制算子（可选）
│   └── custom_op.py
└── tune_configs.yaml        # 调优参数
```

### 3.2 后端注册

```python
from flag_gems.runtime.backend import register_backend

class NewVendorBackend:
    name = "newvendor"
    
    def is_available(self):
        try:
            import newvendor_driver
            return True
        except ImportError:
            return False
    
    def get_tune_config(self, op_name):
        return {"BLOCK_SIZE": 1024}

register_backend("newvendor", NewVendorBackend())
```

## 四、参考资源

- [FlagGems GitHub](https://github.com/FlagOpen/FlagGems)
- [Triton 官方文档](https://triton-lang.org/main/index.html)
- [FlagTree 项目](https://github.com/flagos-ai/flagtree)
- [KernelGen 平台](https://kernelgen.flagos.io)

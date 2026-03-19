# 硬件厂商视角：FlagOS 适配指南

> **前置阅读**：[递进认识flagOS](../递进认识flagOS.md)
> **目标读者**：芯片厂商的软件工程师、编译器工程师
> **文档定位**：从硬件厂商角度提供 FlagOS 的适配和集成指南

## 1. 硬件特性分析

### 1.1 硬件架构对比

| 特性 | NVIDIA A100 | 昇腾910B | 寒武纪MLU | 天数智芯 |
|------|-------------|----------|----------|----------|
| 计算单元 | Tensor Core | Cube | MLU Core | Tensor Core |
| FP16算力 | 312 TFLOPS | 256 TFLOPS | - | - |
| 显存容量 | 40/80 GB | 64 GB | - | - |
| 显存带宽 | 1.6 TB/s | 1.2 TB/s | - | - |
| 互连带宽 | 600 GB/s | 392 GB/s | - | - |

### 1.2 关键特性提取

- **计算范式差异**：Tensor Core vs Cube vs MLU Core
- **存储层次差异**：Shared Memory vs UB vs L1 Cache
- **指令集差异**：CUDA指令 vs CANN指令 vs 厂商私有指令
- **性能特征参数**：带宽、延迟、并行度

## 2. Triton 后端开发

### 2.1 后端架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton Core                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Triton IR (TTIR)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│         ┌────────────────┼────────────────┐               │
│         ▼                ▼                ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ NVIDIA后端  │  │ 昇腾后端    │  │ 新硬件后端  │       │
│  │ (PTX)       │  │ (CANN)      │  │ (自定义)    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 后端开发步骤

```cpp
// Step 1: 实现Target接口
class CustomTarget : public mlir::triton::Target {
public:
  llvm::StringRef getName() const override { return "custom"; }
  
  std::string getTriple() const override {
    return "custom-unknown-unknown";
  }
  
  LogicalResult compileModule(
    ModuleOp module,
    llvm::raw_ostream &os
  ) const override {
    // 实现代码生成逻辑
  }
};

// Step 2: 注册后端
static TargetRegistration reg("custom", []() {
  return std::make_unique<CustomTarget>();
});
```

## 3. FlagGems 后端适配

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
# src/flag_gems/runtime/backend/_newvendor/__init__.py

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

### 3.3 硬件专属算子开发

```python
# runtime/backend/custom/ops/matmul.py
@triton.jit
def custom_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    # 硬件专属参数
    CUSTOM_PARAM: tl.constexpr,
):
    # 利用硬件专属特性优化
    pass
```

## 4. 性能验证

### 4.1 验证清单

```markdown
## 硬件适配验证清单

### 功能验证
- [ ] 所有测试用例通过
- [ ] 精度误差在允许范围内
- [ ] 边界情况处理正确

### 性能验证
- [ ] 核心算子性能达标
- [ ] 模型端到端性能达标
- [ ] 无明显性能回归

### 兼容性验证
- [ ] 与PyTorch兼容
- [ ] 与其他后端共存
- [ ] 版本兼容性
```

### 4.2 性能基准

```python
# 性能目标
PERFORMANCE_TARGETS = {
    'matmul': {
        'speedup_vs_baseline': 0.8,  # 相对厂商库
        'memory_efficiency': 0.7,    # 带宽利用率
    },
    'attention': {
        'speedup_vs_baseline': 0.85,
        'memory_efficiency': 0.75,
    },
}
```

## 5. 上游贡献

### 5.1 贡献流程

```
Fork仓库 → 开发分支 → 编写测试 → 提交PR → 代码审查 → 合并主分支
```

### 5.2 代码规范

```markdown
## 代码贡献规范

### 代码风格
- 遵循PEP 8
- 使用类型注解
- 添加文档字符串

### 测试要求
- 单元测试覆盖率 > 80%
- 性能测试通过
- 兼容性测试通过

### 文档要求
- README更新
- API文档更新
- 变更日志更新
```

## 6. 平台特定配置

### 6.1 昇腾平台

```yaml
# src/flag_gems/runtime/backend/_ascend/tune_configs.yaml

matmul:
  BLOCK_SIZE_M: 64
  BLOCK_SIZE_N: 64
  BLOCK_SIZE_K: 32
  num_warps: 4
  num_stages: 3

attention:
  BLOCK_M: 64
  BLOCK_N: 64
  num_warps: 8
```

### 6.2 天数智芯平台

```yaml
# src/flag_gems/runtime/backend/_iluvatar/tune_configs.yaml

matmul:
  BLOCK_SIZE_M: 128
  BLOCK_SIZE_N: 128
  BLOCK_SIZE_K: 64
```

## 7. 参考资源

- [FlagTree 项目](https://github.com/flagos-ai/flagtree) - 统一Triton编译器
- [Triton 后端开发指南](https://triton-lang.org/main/index.html)
- [开发者视角](02-developer.md) - 开发实践指南
- [评测视角](04-benchmark.md) - 性能验证方法

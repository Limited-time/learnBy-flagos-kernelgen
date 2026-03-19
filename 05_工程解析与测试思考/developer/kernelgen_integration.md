# KernelGen 集成指南

> 本指南结合实际案例，说明如何使用 KernelGen 生成算子并集成到 FlagGems

## 一、KernelGen 是什么？

[KernelGen](https://kernelgen.flagos.io) 是 FlagOS 生态的算子自动生成工具：

```
自然语言/数学公式 → KernelGen → Triton 内核代码
```

**核心价值**：
- 降低算子开发门槛
- 加速开发迭代
- 支持多平台验证

## 二、算子开发全流程

### 2.1 流程图

```
┌─────────────────┐
│ 1. 定义算子需求  │  ← 理解算子语义、输入输出
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. KernelGen生成 │  ← 输入描述，生成 Triton 代码
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. 正确性验证    │  ← 与 PyTorch 原生实现对比
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. 性能优化     │  ← 调整块大小、内存访问模式
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. 多平台测试   │  ← NVIDIA、昇腾、寒武纪等
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. 集成到FlagGems│  ← 注册算子、添加测试
└─────────────────┘
```

## 三、五阶段流程详解

### 3.1 阶段一：算子定义

**流程**：
```
收集需求 → 分析应用场景 → 定义算子信息 → 
定义输入输出 → 算法分析 → 制定优化策略 → 编写文档
```

**算子定义模板**：

```yaml
name: operator_name
description: 算子描述
type: Pointwise

inputs:
  - name: input_name
    type: torch.Tensor
    shape: [dim1, dim2]
    constraints:
      - must be on NPU/CUDA device

outputs:
  - name: output_name
    type: torch.Tensor
    shape: [dim1, dim2]

optimization_strategies:
  - grid_config_optimization
  - block_size_optimization
```

**算法分析要点**：

```
计算密度 = 计算操作数 / 内存访问操作数

对于Broadcast算子：
- 计算操作数：0（仅数据复制）
- 内存访问操作数：D + P×D
- 计算密度 ≈ 0
- 结论：内存带宽受限算子
```

### 3.2 阶段二：代码生成

**生成的四个文件**：

| 文件 | 作用 |
|------|------|
| `_triton.py` | Triton内核实现（核心代码） |
| `_baseline.py` | PyTorch基准实现（参考对比） |
| `_accuracy.py` | 正确性测试（验证功能） |
| `_performance.py` | 性能测试（评估性能） |

**文件关系**：

```
算子定义文档 → KernelGen工具 → 四个Python文件 → 测试优化 → 集成FlagGems
```

### 3.3 阶段三：测试验证

```bash
python *_accuracy.py
python *_performance.py
```

**验证标准**：

| 指标 | 目标值 |
|------|--------|
| 数值精度 | rtol=1e-3, atol=1e-3 |
| 加速比 | > 1.0 |
| 内存带宽利用率 | > 60% |

### 3.4 阶段四：性能优化

**修改类型判断**：

```
问题：我修改的是什么？
  ↓
├─→ BLOCK_M, BLOCK_N, num_warps, num_stages
│   → 【性能参数修改】只修改 _triton.py
│
├─→ 函数签名（参数个数、类型）
│   → 【接口修改】必须同步修改所有四个文件
│
└─→ 计算公式（数学运算）
    → 【逻辑修改】必须同步修改所有四个文件
```

**优化策略**：

| 策略 | 具体调整 | 预期提升 |
|------|---------|---------|
| 网格配置优化 | 一维网格 → 二维网格 | 10-20% |
| 块大小优化 | 调整BLOCK_M/BLOCK_N | 15-30% |
| Warp/Stage优化 | 调整num_warps/num_stages | 10-20% |
| 内存访问优化 | 对齐、向量化、Cache策略 | 10-15% |

**优化示例**：

```python
BLOCK_M = 32
BLOCK_N = 512
num_stages = 3

grid = lambda meta: (triton.cdiv(P, meta['BLOCK_M']), 
                     triton.cdiv(D, meta['BLOCK_N']))
```

### 3.5 阶段五：记档反馈

```markdown
## 优化过程
### 初始性能
- 加速比: 0.85x

### 第一轮优化：网格配置
- 修改: 一维网格 → 二维网格
- 结果: 加速比 0.85x → 1.05x

### 成功经验
1. 二维网格显著提高并行度
2. 适当增大块大小提高计算密度
```

## 四、实际案例：Broadcast 算子

### 4.1 算子定义

```python
# Broadcast: 将张量扩展到目标形状
# 输入: tensor of shape (...,)
# 输出: tensor of shape (target_shape)
```

### 4.2 Triton 实现

```python
import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    input_ptr, output_ptr,
    input_size, output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    input_offsets = offsets % input_size
    
    x = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


def broadcast_triton(input_tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    output = torch.empty(target_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    input_size = input_tensor.numel()
    output_size = output.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(output_size, BLOCK_SIZE),)
    
    broadcast_kernel[grid](
        input_tensor, output,
        input_size, output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
```

### 4.3 优化过程

| 阶段 | BLOCK_M | BLOCK_N | num_stages | 加速比 |
|------|---------|---------|------------|--------|
| 初始 | 64 | 256 | 3 | 0.30x |
| 第一轮 | 16 | 256 | 2 | 0.33x |
| 第二轮 | 32 | 512 | 3 | 1.41x |

## 五、集成到 FlagGems

### 5.1 文件放置

```
src/flag_gems/ops/
└── broadcast.py    # 你的算子实现
```

### 5.2 注册算子

```python
# src/flag_gems/ops/__init__.py

from .broadcast import broadcast

__all__ = [
    "broadcast",
]
```

### 5.3 添加测试

```
tests/
└── test_broadcast.py    # 你的测试文件
```

## 六、多平台适配

### 6.1 平台差异

| 平台 | 注意事项 |
|------|---------|
| NVIDIA | 标准 CUDA，块大小可大 |
| 昇腾 | 需要适配 Triton 后端 |
| 寒武纪 | 可能需要调整内存访问模式 |
| 天数智芯 | 注意缓存大小限制 |
| 海光 | 注意内存对齐要求 |
| 摩尔线程 | 检查驱动兼容性 |

### 6.2 平台特定配置

```yaml
# src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml

broadcast:
  BLOCK_SIZE: 1024
  
# src/flag_gems/runtime/backend/_ascend/tune_configs.yaml

broadcast:
  BLOCK_SIZE: 512
```

## 七、参考资源

- [KernelGen 平台](https://kernelgen.flagos.io)
- [FlagGems GitHub](https://github.com/FlagOpen/FlagGems)
- [FlagGems 官方文档](https://docs.flagos.io/projects/FlagGems/en/latest/)

# Broadcast 算子优化实践案例

> 本案例展示使用 KernelGen 生成 Broadcast 算子并进行性能优化的完整过程

## 案例概述

**算子名称**：Broadcast  
**算子类型**：Pointwise  
**计算密度**：≈0（纯数据复制）  
**性能瓶颈**：内存带宽

## 核心文件

| 文件 | 说明 |
|------|------|
| [broadcast_v1_triton.py](./broadcast_v1_triton.py) | Triton 内核实现 |
| [broadcast_v1_baseline.py](./broadcast_v1_baseline.py) | PyTorch 基准实现 |
| [broadcast_v1_test_relu_accuracy.py](./broadcast_v1_test_relu_accuracy.py) | 正确性测试 |
| [broadcast_v1_test_relu_performance.py](./broadcast_v1_test_relu_performance.py) | 性能测试 |

## 详细文档

| 文档 | 说明 |
|------|------|
| [KERNELGEN_FULL_PROCESS.md](./KERNELGEN_FULL_PROCESS.md) | KernelGen 算子生成与优化全流程 |
| [Broadcast算子完整定义与分析.md](./Broadcast算子完整定义与分析.md) | 算子定义与分析 |
| [optimization_log.md](./optimization_log.md) | 优化过程记录 |

## 优化过程

| 阶段 | BLOCK_M | BLOCK_N | num_stages | 加速比 |
|------|---------|---------|------------|--------|
| 初始 | 64 | 256 | 3 | 0.30x |
| 第一轮 | 16 | 256 | 2 | 0.33x |
| 第二轮 | 32 | 512 | 3 | 1.41x |

## 关键优化点

1. **网格配置**：一维网格 → 二维网格
2. **块大小调整**：增大 BLOCK_N 提高内存带宽利用率
3. **流水线深度**：调整 num_stages 隐藏内存延迟

## 运行测试

```bash
# 正确性测试
python broadcast_v1_test_relu_accuracy.py

# 性能测试
python broadcast_v1_test_relu_performance.py
```

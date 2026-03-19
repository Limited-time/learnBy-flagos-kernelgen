# FlagGems 性能优化

> **前置阅读**：[用户指南](user_guide.md)
> **目标读者**：追求极致性能的用户
> **文档定位**：FlagGems 的性能调优指南

## 性能优化概述

虽然 `flag_gems` 内核设计用于高性能，但在完整模型部署中实现最佳端到端速度需要仔细集成和考虑运行时行为。两个常见的性能瓶颈是：

1. **生产环境中的运行时自动调优开销**
2. **由于框架级内核注册导致的次优调度**

## 预调优模型形状

### 为什么需要预调优？

Triton 通常在新输入形状的前几次执行期间执行自动调优，这可能会导致延迟峰值。`LibTuner` 通过以下方式解决这个问题：

- **持久缓存**：最佳自动调优配置在运行之间保存
- **跨进程共享**：缓存在同一设备上的进程之间共享
- **减少运行时开销**：一旦调优，算子在未来运行中跳过调优

### 如何使用预调优

```python
import flag_gems

# 使用预调优配置
flag_gems.enable(tune_cache_dir="./tuned_configs")
```

### 预调优脚本

```bash
# 运行预调优脚本
python examples/pretune.py
```

预调优脚本会：
1. 确定生产工作负载中使用的关键输入形状
2. 运行基准测试和缓存最佳配置
3. 正常部署时自动从缓存中选择最佳配置

## 使用 C++ 包装器

### 为什么使用 C++ 包装器？

虽然 Triton 内核提供了相当好的计算性能，但 Triton 本身是一种 Python 嵌入式 DSL。这意味着算子定义和运行时调度都依赖于 Python，可能会在对延迟敏感或高吞吐量的场景中引入不可忽视的开销。

### 安装 C++ 运行时

```bash
# 安装构建依赖
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake

# 构建 C++ 扩展
cd FlagGems
pip install --no-build-isolation .
```

### 验证安装

```python
try:
    from flag_gems import c_operators
    has_c_extension = True
    print("C++ 运行时可用")
except Exception as e:
    c_operators = None
    has_c_extension = False
    print("C++ 运行时不可用")
```

如果 `has_c_extension` 为 `True`，则 C++ 运行时路径可用。

## 性能加速效果

### 性能对比

FlagGems 在各种模型和硬件上都表现出了显著的性能加速：

| 平台 | 模型 | 原生PyTorch | FlagGems | 加速比 |
|------|------|------------|---------|--------|
| NVIDIA A100 | Llama-2-7B | 650ms | 420ms | 1.55x |
| 昇腾910B | Llama-2-7B | 720ms | 480ms | 1.50x |

### 性能优化技巧

#### 1. 内存访问优化

```python
@triton.jit
def optimized_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # ✅ 合并访问：连续内存地址
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    
    # ❌ 避免随机访问
    # indices = tl.rand(BLOCK_SIZE) * n
    # x = tl.load(x_ptr + indices)  # 性能差
```

#### 2. 使用 LibTuner 缓存

```python
from flag_gems.utils.libentry import libtuner

@libtuner(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def tuned_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pass
```

#### 3. 性能参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `BLOCK_M` | 16-64 | 行方向块大小 |
| `BLOCK_N` | 256-512 | 列方向块大小（64的倍数） |
| `num_warps` | 4-8 | Warp数量 |
| `num_stages` | 2-4 | 流水线深度 |

## 性能分析工具

### 使用 PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    result = torch.mm(x, y)

print(prof.key_averages().table())
prof.export_chrome_trace("trace.json")
```

### 正确性对比

```python
import torch
import flag_gems

x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')

with flag_gems.use_gems():
    result_gems = torch.mm(x, y)

result_torch = torch.mm(x, y)

diff = (result_gems - result_torch).abs().max()
print(f"最大差异: {diff}")
```

## 参考资源

- [FlagGems GitHub](https://github.com/FlagOpen/FlagGems)
- [Triton 官方文档](https://triton-lang.org/main/index.html)
- [算子开发模板](../developer/operator_template.md)
- [故障排查](../developer/troubleshooting.md)

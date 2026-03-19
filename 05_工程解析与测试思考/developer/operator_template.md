# 算子开发模板

> 本文档提供 FlagGems 算子开发的标准模板和最佳实践

## 一、算子实现模板

### 1.1 Triton 内核实现

```python
# src/flag_gems/ops/my_op.py

import torch
import triton
import triton.language as tl

@triton.jit
def my_op_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    
    tl.store(output_ptr + offsets, output, mask=mask)


def my_op(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    my_op_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### 1.2 带自动调优的实现

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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 二、单元测试模板

### 2.1 正确性测试

```python
# tests/test_my_op.py

import torch
import pytest
from flag_gems.ops import my_op

class TestMyOp:
    
    @pytest.mark.parametrize("shape", [
        (1024,),
        (1024, 1024),
        (2, 1024, 1024),
    ])
    def test_shape(self, shape):
        x = torch.randn(shape, device='cuda')
        result = my_op(x)
        assert result.shape == x.shape
    
    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
    ])
    def test_dtype(self, dtype):
        x = torch.randn(1024, dtype=dtype, device='cuda')
        result = my_op(x)
        assert result.dtype == dtype
    
    def test_correctness(self):
        x = torch.randn(1024, device='cuda')
        result = my_op(x)
        expected = x * 2
        assert torch.allclose(result, expected, atol=1e-6)
```

### 2.2 性能基准测试

```python
# benchmark/test_my_op_perf.py

import torch
import pytest
from flag_gems.ops import my_op

class TestMyOpPerf:
    
    @pytest.mark.parametrize("shape", [
        (1024, 1024),
        (4096, 4096),
    ])
    def test_performance(self, shape, benchmark):
        x = torch.randn(shape, device='cuda')
        
        def op():
            return my_op(x)
        
        result = benchmark(op)
        assert result.shape == x.shape
```

## 三、性能优化技巧

### 3.1 内存访问优化

```python
@triton.jit
def optimized_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
```

### 3.2 二维网格配置

```python
@triton.jit
def kernel_2d(
    x_ptr, output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :])
    tl.store(output_ptr + offs_m[:, None] * N + offs_n[None, :], x, mask=mask_m[:, None] & mask_n[None, :])

grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
```

### 3.3 性能参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `BLOCK_M` | 16-64 | 行方向块大小 |
| `BLOCK_N` | 256-512 | 列方向块大小（64的倍数） |
| `num_warps` | 4-8 | Warp数量 |
| `num_stages` | 2-4 | 流水线深度 |

## 四、算子注册

### 4.1 添加到 __init__.py

```python
# src/flag_gems/ops/__init__.py

from .my_op import my_op

__all__ = [
    "my_op",
]
```

### 4.2 自定义算子注册

```python
from flag_gems.ops import register_op

register_op("custom_op", custom_op)

flag_gems.enable()
result = custom_op(input_tensor)
```

## 五、调试与验证

### 5.1 启用调试日志

```python
import flag_gems

flag_gems.enable(record=True, path="./debug.log")
```

### 5.2 性能分析

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    result = torch.mm(x, y)

print(prof.key_averages().table())
prof.export_chrome_trace("trace.json")
```

### 5.3 正确性对比

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

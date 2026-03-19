# 开发者视角：FlagOS 开发实践

> **前置阅读**：[递进认识flagOS](../递进认识flagOS.md)
> **目标读者**：软件开发工程师、算子开发者
> **文档定位**：从开发者角度提供 FlagOS 的开发指南和最佳实践

## 1. 开发环境搭建

### 1.1 基础环境要求

```yaml
硬件要求:
  GPU: NVIDIA A100/H100 或 昇腾910B
  CPU: 16核以上
  内存: 64GB以上
  存储: 500GB SSD

软件要求:
  OS: Ubuntu 20.04+ / CentOS 7+
  Python: 3.8+
  CUDA: 11.8+ (NVIDIA) / CANN 8.0+ (昇腾)
  PyTorch: 2.0+
```

### 1.2 环境配置步骤

```bash
# Step 1: 克隆代码
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems

# Step 2: 创建虚拟环境
conda create -n flaggems python=3.10
conda activate flaggems

# Step 3: 安装依赖
pip install -r requirements.txt
pip install -e .

# Step 4: 验证安装
python -c "import flag_gems; print(flag_gems.__version__)"
```

### 1.3 IDE配置

**VS Code配置**：
```json
{
  "python.linting.enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

**PyCharm配置**：
- 启用Python类型检查
- 配置Triton语法高亮
- 设置调试断点

## 2. 核心 API 介绍

### 2.1 FlagGems 启用方式

```python
import torch
import flag_gems

# 方式1：全局启用（推荐新手）
flag_gems.enable()

# 方式2：局部启用（推荐测试）
with flag_gems.use_gems():
    y = torch.mm(x, x)

# 方式3：选择性启用（推荐优化）
flag_gems.only_enable(include=["mm", "addmm", "rms_norm"])
```

### 2.2 后端选择

```python
# 自动检测硬件
print(f"当前后端：{flag_gems.vendor_name}")

# 手动设置后端
import os
os.environ["GEMS_VENDOR"] = "huawei"
```

### 2.3 调试接口

```python
# 启用调试日志
flag_gems.enable(record=True, path="./debug.log")

# 查看已注册算子
print(flag_gems.all_registered_ops())
```

## 3. 算子开发实战

### 3.1 第一个Triton算子

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 3.2 算子注册到FlagGems

```python
from flag_gems import libentry, register_op

@libentry()
@register_op("custom_add")
def custom_add(x, y):
    return add(x, y)
```

### 3.3 单元测试编写

```python
import pytest
import torch
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
    
    def test_correctness(self):
        x = torch.randn(1024, device='cuda')
        result = my_op(x)
        expected = x * 2
        assert torch.allclose(result, expected, atol=1e-6)
```

## 4. 调试技巧

### 4.1 常见问题排查

| 问题类型 | 现象 | 排查方法 | 解决方案 |
|---------|------|---------|---------|
| 精度问题 | 结果不一致 | 逐步打印中间值 | 使用FP32中间计算 |
| 性能问题 | 性能不达标 | Nsight Systems分析 | 优化内存访问模式 |
| 内存问题 | OOM | 内存分析工具 | 减小BLOCK_SIZE |

### 4.2 性能分析工具

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    result = torch.mm(x, y)

print(prof.key_averages().table())
prof.export_chrome_trace("trace.json")
```

### 4.3 正确性对比

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

## 5. 性能优化

### 5.1 内存访问优化

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

### 5.2 使用 LibTuner 缓存

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

### 5.3 性能参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `BLOCK_M` | 16-64 | 行方向块大小 |
| `BLOCK_N` | 256-512 | 列方向块大小（64的倍数） |
| `num_warps` | 4-8 | Warp数量 |
| `num_stages` | 2-4 | 流水线深度 |

## 6. 参考资源

- [FlagGems GitHub](https://github.com/FlagOpen/FlagGems)
- [Triton 官方文档](https://triton-lang.org/main/index.html)
- [算子开发模板](../developer/operator_template.md)
- [故障排查指南](../developer/troubleshooting.md)

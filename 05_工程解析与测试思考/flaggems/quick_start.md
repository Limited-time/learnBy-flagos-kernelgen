# FlagGems 快速入门

> **前置阅读**：[FlagGems 手册](README.md)
> **目标读者**：初次使用 FlagGems 的用户
> **预计时间**：5分钟

## 安装指南

### 系统要求

```yaml
硬件要求:
  GPU: NVIDIA A100/H100 或 昇腾910B
  CPU: 16核以上
  内存: 64GB以上

软件要求:
  OS: Ubuntu 20.04+ / CentOS 7+
  Python: 3.8+
  PyTorch: 2.0+
```

### 安装步骤

```bash
# Step 1: 创建虚拟环境
conda create -n flaggems python=3.10
conda activate flaggems

# Step 2: 安装PyTorch
# NVIDIA平台
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Step 3: 安装FlagGems
pip install flag-gems

# Step 4: 验证安装
python -c "import flag_gems; print(f'FlagGems版本: {flag_gems.__version__}')"
```

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems

# 安装依赖
pip install -r requirements.txt

# 可编辑安装
pip install --no-build-isolation -e .
```

## 第一个示例

### 基础使用

```python
import torch
import flag_gems

# 定义矩阵维度
M, N, K = 1024, 1024, 1024

# 创建测试数据
A = torch.randn((M, K), dtype=torch.float16, device="cuda")
print(f"矩阵A形状: {A.shape}")

B = torch.randn((K, N), dtype=torch.float16, device="cuda")
print(f"矩阵B形状: {B.shape}")

# 使用FlagGems进行矩阵乘法
with flag_gems.use_gems():
    C = torch.mm(A, B)
    print(f"结果矩阵C形状: {C.shape}")
    print(f"结果示例值: {C[0, :5]}")  # 显示前5个值
```

**预期输出**：
```
矩阵A形状: torch.Size([1024, 1024])
矩阵B形状: torch.Size([1024, 1024])
结果矩阵C形状: torch.Size([1024, 1024])
结果示例值: tensor([...], device='cuda:0', dtype=torch.float16)
```

### 性能对比测试

```python
import torch
import flag_gems
import time

# 创建测试数据
x = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
y = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)

# 执行矩阵乘法（自动使用FlagGems加速）
result = torch.mm(x, y)

print(f"结果形状: {result.shape}")
print(f"当前后端: {flag_gems.vendor_name}")
```

### 验证加速效果

```python
import torch
import time
import flag_gems

# 测试数据
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
y = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# 预热
for _ in range(10):
    _ = torch.mm(x, y)
torch.cuda.synchronize()

# 测试原生PyTorch
start = time.time()
for _ in range(100):
    _ = torch.mm(x, y)
torch.cuda.synchronize()
torch_time = (time.time() - start) / 100

# 启用FlagGems
flag_gems.enable()

# 测试FlagGems
start = time.time()
for _ in range(100):
    _ = torch.mm(x, y)
torch.cuda.synchronize()
gems_time = (time.time() - start) / 100

print(f"PyTorch时间: {torch_time*1000:.3f}ms")
print(f"FlagGems时间: {gems_time*1000:.3f}ms")
print(f"加速比: {torch_time/gems_time:.2f}x")
```

## 常见问题

### Q1: 如何确认FlagGems已启用？

```python
import flag_gems

# 启用后检查
flag_gems.enable()
print(f"已注册算子: {flag_gems.all_registered_ops()[:5]}...")  # 显示前5个
```

### Q2: 如何查看哪些算子被加速？

```python
# 启用调试日志
flag_gems.enable(record=True, path="./debug.log")

# 运行你的代码
# ...

# 查看日志
with open("./debug.log", "r") as f:
    print(f.read())
```

### Q3: 如何在特定硬件上使用？

```python
import os

# 方法1：环境变量
os.environ["GEMS_VENDOR"] = "huawei"

# 方法2：代码中设置
import flag_gems
print(f"当前后端: {flag_gems.vendor_name}")
```

### Q4: 如何禁用特定算子？

```python
import flag_gems

# 禁用特定算子
flag_gems.enable(unused=["sum", "add"])

# 或只启用特定算子
flag_gems.only_enable(include=["mm", "addmm"])
```

## 下一步

- 阅读 [用户指南](user_guide.md) 了解高级用法
- 查看 [性能优化](performance.md) 获取最佳性能
- 参考 [开发者实践指南](../developer/CONTRIBUTING.md) 参与贡献

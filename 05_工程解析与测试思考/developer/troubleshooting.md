# 故障排除指南

> 本文档汇总 FlagGems 和 KernelGen 使用过程中的常见问题及解决方案

## 一、FlagGems 常见问题

### 1.1 加速效果不明显

**诊断步骤**：

```python
# 1. 检查是否启用
print(f"FlagGems enabled: {flag_gems.is_enabled()}")

# 2. 查看加速的算子
flag_gems.enable(record=True, path="./debug.log")
```

**解决方案**：

```python
# 预调优
flag_gems.enable(tune_cache_dir="./tuned_configs")

# 选择性启用关键算子
flag_gems.only_enable(include=["mm", "addmm", "rms_norm"])
```

### 1.2 如何回退到 PyTorch 原生实现

```python
# 全局禁用
flag_gems.disable()

# 禁用特定算子
flag_gems.enable(unused=["mm"])

# 使用上下文管理器
with flag_gems.use_gems(exclude=["mm"]):
    pass
```

### 1.3 多平台兼容性问题

**解决方案**：

```python
import os
os.environ["GEMS_VENDOR"] = "huawei"  # 手动指定后端

# 或禁用不支持的算子
flag_gems.enable(unused=["unsupported_op"])
```

## 二、KernelGen 常见问题

### 2.1 生成的代码性能不佳

**解决方案**：

1. 检查网格配置（使用二维网格）
2. 调整块大小（BLOCK_M: 16-64, BLOCK_N: 256-512）
3. 增加自动调优

### 2.2 生成的代码有错误

**解决方案**：

- 手动修正，或提供更精确的描述
- 检查算子定义是否完整准确

## 三、开发流程问题

### 3.1 pre-commit 检查失败

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| pre-commit 检查失败 | 代码格式不符合规范 | 运行 `pre-commit run --all-files` 自动修复 |

```bash
pre-commit run --all-files
```

### 3.2 CI 测试失败

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CI 测试失败 | 测试用例未通过 | 本地运行 `pytest` 排查问题 |

```bash
cd tests
pytest test_xx_ops.py
```

### 3.3 Triton 编译失败

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Triton 编译失败 | 语法错误或版本不兼容 | 检查 Triton 语法，更新版本 |

```bash
pip install --upgrade triton
```

## 四、运行时错误

### 4.1 常见错误速查表

| 错误 | 解决方案 |
|------|---------|
| `RuntimeError: CUDA error` | 检查CUDA版本匹配 |
| `ImportError: cannot import name` | 重新安装：`pip install -e .` |
| `AssertionError: must be CUDA tensor` | 确保数据在GPU上：`x.cuda()` |
| 内存访问越界 | 检查 `grid` 和 `mask` |
| 性能不如预期 | 使用合并访问，优化块大小 |

### 4.2 CUDA 错误排查

```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4.3 导入错误排查

```bash
# 重新安装 FlagGems
pip install -e .

# 检查安装路径
python -c "import flag_gems; print(flag_gems.__file__)"
```

## 五、性能问题

### 5.1 首次执行延迟

**原因**：Triton 内核需要编译

**解决方案**：使用预调优配置

```python
flag_gems.enable(tune_cache_dir="./tuned_configs")
```

### 5.2 性能参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `BLOCK_M` | 16-64 | 行方向块大小 |
| `BLOCK_N` | 256-512 | 列方向块大小（64的倍数） |
| `num_warps` | 4-8 | Warp数量 |
| `num_stages` | 2-4 | 流水线深度 |

### 5.3 性能分析工具

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    result = torch.mm(x, y)

print(prof.key_averages().table())
prof.export_chrome_trace("trace.json")
```

## 六、集成问题

### 6.1 算子未被调用

**检查步骤**：

```python
# 查看已注册算子
print(flag_gems.all_registered_ops())

# 检查注册是否正确
flag_gems.enable(record=True, path="./debug.log")
```

### 6.2 多平台结果不一致

**解决方案**：

- 检查数值精度，添加容差
- 使用 `torch.allclose(result, expected, rtol=1e-3, atol=1e-3)`

## 七、调试技巧

### 7.1 启用调试日志

```python
import flag_gems

flag_gems.enable(record=True, path="./debug.log")
```

### 7.2 正确性对比

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

### 7.3 运行时日志

```bash
pytest program.py --log-cli-level debug
```

## 八、参考资源

- [FlagGems GitHub Issues](https://github.com/FlagOpen/FlagGems/issues)
- [Triton 官方文档](https://triton-lang.org/main/index.html)
- [FlagGems 官方文档](https://docs.flagos.io/projects/FlagGems/)

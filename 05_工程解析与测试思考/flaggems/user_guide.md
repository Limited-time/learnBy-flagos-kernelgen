# FlagGems 用户指南

> **前置阅读**：[快速入门](quick_start.md)
> **目标读者**：需要深入了解 FlagGems 的用户
> **文档定位**：FlagGems 的高级用法和完整功能说明

## 高级用法

### 选择性启用算子

```python
import flag_gems

# 方式1：只启用特定算子
flag_gems.only_enable(include=["mm", "addmm", "rms_norm", "softmax"])

# 方式2：启用所有但排除特定算子
flag_gems.enable(unused=["sum", "add"])

# 方式3：在上下文中选择性启用
with flag_gems.use_gems(include=["mm", "addmm"]):
    # 只有 mm 和 addmm 会被加速
    pass

with flag_gems.use_gems(exclude=["mul", "div"]):
    # 除了 mul 和 div 之外的所有操作都会被加速
    pass
```

### 参数概览

| 参数 | 类型 | 描述 |
|------|------|------|
| `unused` | List[str] | 禁用特定算子（用于 `enable`） |
| `include` | List[str] | 仅启用特定算子（用于 `only_enable`） |
| `record` | bool | 记录算子调用以进行调试或分析 |
| `path` | str | 日志文件路径（仅在 `record=True` 时使用） |

### 启用调试日志

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

运行后检查日志文件：

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

### 查询已注册算子

```python
import flag_gems

flag_gems.enable()

# 获取已注册的函数名称列表
registered_funcs = flag_gems.all_registered_ops()
print("Registered functions:", registered_funcs[:10])

# 获取已注册的算子键列表
registered_keys = flag_gems.all_registered_keys()
print("Registered keys:", registered_keys[:10])
```

## 多平台支持

### 支持的平台

| 厂商 | 型号 | 后端名称 |
|------|------|---------|
| NVIDIA | A100, H100, V100 | nvidia |
| 华为 | 昇腾910B | huawei |
| 天数 | T10, T20 | iluvatar |
| 海光 | DCU | hygon |
| 摩尔线程 | S80, S3000 | moore |
| 寒武纪 | MLU | cambricon |

### 后端自动检测

默认情况下，`flag_gems` 会在运行时自动检测当前硬件后端：

```python
import flag_gems
print(f"当前后端：{flag_gems.vendor_name}")
```

### 手动设置后端

```bash
# 通过环境变量设置
export GEMS_VENDOR=huawei
```

> ⚠️ 此设置应与实际硬件平台匹配。手动设置不正确的后端可能会导致运行时错误。

## 与流行框架集成

### Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 将模型移至正确的设备
device = flag_gems.device
model.to(device).eval()

# 启用FlagGems加速
with flag_gems.use_gems():
    inputs = tokenizer("Hello", return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=100)
```

### vLLM 集成

```python
from vllm import LLM, SamplingParams
import flag_gems

# 启用FlagGems
flag_gems.enable()

# 初始化vLLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="float16")

# 批量推理
prompts = ["Hello", "World"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

### 替换 vLLM 特定算子

```python
# 使用FlagGems替换vLLM内部算子
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```

输出示例：
```
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
```

## 显式调用

绕过 PyTorch 的分发机制，直接调用 FlagGems 算子：

```python
import torch
from flag_gems import ops
import flag_gems

# 直接调用FlagGems算子，无需启用
a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

## 回退到原生实现

```python
import flag_gems

# 全局禁用
flag_gems.disable()

# 禁用特定算子
flag_gems.enable(unused=["mm"])

# 使用上下文管理器排除特定算子
with flag_gems.use_gems(exclude=["mm"]):
    pass
```

## 参考资源

- [性能优化](performance.md) - 获取最佳性能
- [开发者实践指南](../developer/CONTRIBUTING.md) - 参与贡献
- [故障排查](../developer/troubleshooting.md) - 常见问题解决

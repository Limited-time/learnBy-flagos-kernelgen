# 模型使用者视角：FlagOS 模型部署

> **前置阅读**：[递进认识flagOS](../递进认识flagOS.md)
> **目标读者**：算法工程师、模型应用开发者
> **文档定位**：从模型使用者角度提供在 FlagOS 上部署和运行模型的指南

## 1. 模型选择与准备

### 1.1 支持模型列表

| 模型类型 | 模型名称 | 参数量 | 训练支持 | 推理支持 |
|---------|---------|--------|---------|---------|
| LLM | LLaMA2 | 7B-70B | ✓ | ✓ |
| LLM | Qwen2.5 | 0.5B-72B | ✓ | ✓ |
| LLM | DeepSeek-V3 | 671B | ✓ | ✓ |
| VLM | Qwen2.5-VL | 3B-72B | ✓ | ✓ |
| MoE | Mixtral | 8x7B | ✓ | ✓ |

### 1.2 模型获取方式

```bash
# 从ModelScope下载
pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct')

# 从HuggingFace下载
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# 从FlagRelease下载（已适配版本）
# 见 https://huggingface.co/FlagRelease
```

## 2. 环境配置

### 2.1 快速开始

```bash
# 安装FlagOS全家桶
pip install flag-gems flag-scale flag-cx

# 启用FlagGems加速
python -c "import flag_gems; flag_gems.enable()"
```

### 2.2 硬件配置

```yaml
# NVIDIA配置
hardware:
  vendor: nvidia
  devices: [0,1,2,3,4,5,6,7]

# 昇腾配置
hardware:
  vendor: ascend
  devices: [0,1,2,3,4,5,6,7]
```

## 3. 模型训练

### 3.1 训练配置

```yaml
# train_config.yaml
model:
  name: qwen2.5-7b
  path: /path/to/model

training:
  batch_size: 128
  learning_rate: 1e-4
  epochs: 3
  precision: bf16

parallel:
  tensor_parallel: 2
  pipeline_parallel: 2
  data_parallel: 2

hardware:
  vendor: nvidia  # 或 ascend
```

### 3.2 启动训练

```bash
# 使用FlagScale
flagscale train --config train_config.yaml

# 或使用原生PyTorch
torchrun --nproc_per_node=8 train.py --config train_config.yaml
```

### 3.3 训练监控

```python
# 监控训练性能
metrics = {
    'throughput': 'tokens/s',
    'loss': 'training_loss',
    'learning_rate': 'current_lr',
    'gpu_memory': 'GB',
    'gpu_utilization': '%',
}
```

## 4. 模型推理

### 4.1 推理配置

```yaml
# inference_config.yaml
model:
  name: qwen2.5-7b
  path: /path/to/model

inference:
  max_batch_size: 32
  max_seq_len: 4096
  precision: fp16

serving:
  port: 8000
  workers: 1
```

### 4.2 启动推理服务

```bash
# 使用FlagScale Serve
flagscale serve --config inference_config.yaml

# 或使用vLLM
python -m vllm.entrypoints.api_server \
    --model /path/to/model \
    --port 8000
```

### 4.3 推理性能测试

```python
# 推理性能测试
def benchmark_inference(model, prompts, batch_sizes):
    results = []
    for bs in batch_sizes:
        start = time.time()
        outputs = model.generate(prompts[:bs])
        end = time.time()
        
        results.append({
            'batch_size': bs,
            'latency': (end - start) / bs * 1000,  # ms
            'throughput': bs / (end - start),  # req/s
        })
    return results
```

## 5. 与流行框架集成

### 5.1 Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 启用FlagGems加速
with flag_gems.use_gems():
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=100)
```

### 5.2 vLLM 集成

```python
from vllm import LLM, SamplingParams
import flag_gems

# 启用FlagGems
flag_gems.enable()

# 初始化vLLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="float16")

# 批量推理
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

## 6. 生产部署

### 6.1 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Instance 1 │    │  Instance 2 │    │  Instance N │
│  (GPU Node) │    │  (GPU Node) │    │  (GPU Node) │
└─────────────┘    └─────────────┘    └─────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                          │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 部署检查清单

```markdown
## 生产部署检查清单

### 模型准备
- [ ] 模型权重已转换
- [ ] 模型精度验证通过
- [ ] 模型大小评估完成

### 环境准备
- [ ] 硬件资源充足
- [ ] 软件环境一致
- [ ] 网络配置正确

### 服务配置
- [ ] 端口配置正确
- [ ] 负载均衡配置
- [ ] 监控告警配置

### 安全配置
- [ ] 访问控制配置
- [ ] 数据加密配置
- [ ] 日志审计配置
```

## 7. 参考资源

- [FlagScale GitHub](https://github.com/flagos-ai/flagscale)
- [FlagRelease 平台](https://huggingface.co/FlagRelease)
- [运维视角](03-ops.md) - 生产运维指南
- [评测视角](04-benchmark.md) - 性能评测方法

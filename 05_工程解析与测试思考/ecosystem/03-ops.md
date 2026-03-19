# 运维工程师视角：FlagOS 运维指南

> **前置阅读**：[递进认识flagOS](../递进认识flagOS.md)
> **目标读者**：运维工程师、SRE
> **文档定位**：从运维角度提供 FlagOS 的部署、监控和维护指南

## 1. 系统部署

### 1.1 集群规划

```yaml
集群配置:
  master节点: 1台
    - CPU: 32核
    - 内存: 128GB
    - 存储: 2TB SSD
  
  worker节点: N台
    - CPU: 64核
    - 内存: 256GB
    - GPU: 8×A100/昇腾910B
    - 存储: 4TB NVMe
  
  网络要求:
    - 节点间: 100Gbps InfiniBand
    - 存储: 25Gbps Ethernet
```

### 1.2 部署方案

**方案一：Ansible自动化部署**
```bash
ansible-playbook -i inventory deploy.yml
```

**方案二：Kubernetes部署**
```bash
kubectl apply -f flagos-cluster.yaml
```

**方案三：Docker Compose**
```bash
docker-compose up -d
```

### 1.3 配置管理

```yaml
# config/cluster.yaml
cluster:
  name: flagos-prod
  nodes:
    - host: gpu-node-01
      devices: [0,1,2,3,4,5,6,7]
    - host: gpu-node-02
      devices: [0,1,2,3,4,5,6,7]

training:
  backend: nccl  # 或 hccl
  port: 29500

monitoring:
  enabled: true
  port: 9090
```

## 2. 监控与告警

### 2.1 监控指标设计

| 指标类型 | 指标名称 | 采集频率 | 告警阈值 |
|---------|---------|---------|---------|
| GPU | 利用率 | 10s | <30% 或 >95% |
| GPU | 显存使用 | 10s | >90% |
| GPU | 温度 | 30s | >85°C |
| GPU | 功耗 | 30s | >额定值95% |
| 网络 | 带宽使用 | 10s | >80% |
| 训练 | 吞吐 | 1min | 下降>20% |
| 训练 | Loss | 1min | NaN或发散 |

### 2.2 监控系统搭建

```yaml
# Prometheus配置
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['gpu-node-01:9400', 'gpu-node-02:9400']
  
  - job_name: 'training-metrics'
    static_configs:
      - targets: ['trainer:8000']
```

### 2.3 告警规则

```yaml
# alerting/rules.yml
groups:
  - name: gpu-alerts
    rules:
      - alert: GPUHighTemperature
        expr: gpu_temperature > 85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU温度过高"
      
      - alert: TrainingLossNaN
        expr: training_loss == NaN
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "训练Loss为NaN"
```

## 3. 日志管理

### 3.1 日志收集

```yaml
# 日志配置
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: file
      path: /var/log/flagos/app.log
      rotation: daily
      retention: 30 days
    - type: syslog
      address: logs.example.com:514
```

### 3.2 日志分析

```bash
# 查看最近错误
grep "ERROR" /var/log/flagos/app.log | tail -100

# 统计错误类型
grep "ERROR" /var/log/flagos/app.log | awk '{print $5}' | sort | uniq -c
```

## 4. 故障排查

### 4.1 常见故障案例

**案例1：训练Loss发散**
```
现象：训练过程中Loss突然变为NaN

排查步骤：
1. 检查输入数据是否正常
2. 检查学习率是否过大
3. 检查混合精度计算
4. 检查算子精度

解决方案：
- 启用FP32中间计算
- 降低学习率
- 添加梯度裁剪
```

**案例2：GPU利用率低**
```
现象：GPU利用率持续低于30%

排查步骤：
1. 分析数据加载瓶颈
2. 检查CPU预处理
3. 分析通信开销
4. 检查算子性能

解决方案：
- 增加DataLoader workers
- 启用数据预取
- 优化通信策略
- 替换低效算子
```

### 4.2 故障诊断工具

```bash
# GPU诊断
nvidia-smi dmon -s pucmet -i 0

# 性能分析
nsys profile -o report python train.py

# 内存分析
python -m torch.utils.bottleneck train.py
```

## 5. 性能调优

### 5.1 系统级调优

```bash
# GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 300  # 功耗限制

# CPU性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 内存大页
echo 1000 > /proc/sys/vm/nr_hugepages
```

### 5.2 训练调优

```python
# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 6. 容量规划

### 6.1 资源评估模型

```
所需GPU数 = 模型参数量 × 字节数 × 并行度 / 单卡显存

训练时间 = 数据量 / (batch_size × 吞吐 × GPU数)

存储需求 = 数据集 + 模型检查点 × 版本数 + 日志
```

### 6.2 扩容策略

| 策略 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| 横向扩容 | 数据并行 | 简单直接 | 通信开销 |
| 纵向扩容 | 模型并行 | 减少通信 | 硬件成本 |
| 混合扩容 | 大模型训练 | 灵活高效 | 架构复杂 |

## 7. 参考资源

- [FlagScale GitHub](https://github.com/flagos-ai/flagscale)
- [FlagCX GitHub](https://github.com/flagos-ai/flagcx)
- [架构师视角](01-architect.md) - 系统架构设计
- [评测视角](04-benchmark.md) - 性能评测方法

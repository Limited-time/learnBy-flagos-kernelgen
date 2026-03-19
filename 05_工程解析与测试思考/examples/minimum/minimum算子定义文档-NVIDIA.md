# Minimum算子定义与优化版本说明（NVIDIA GPU）

## 版本一：基础实现版本

### （1）原始需求
实现基础的Minimum功能，确保在低维和中维Shape下（如[512, 128, 512]）结果正确，能够处理逐元素比较和基本广播逻辑，暂不考虑极致的性能优化。

### （2）算子基本信息
- **算子名称**：Minimum
- **测评设备**：NVIDIA A100 / H100
- **算子类型**：Pointwise
- **功能描述**：逐元素比较两个输入张量，返回对应位置的最小值。支持广播机制，当两个张量shape不同但可广播时，自动扩展后进行比较。充分利用NVIDIA GPU的Tensor Cores特性。

### （3）输入参数
| 参数名称 | 数据类型 | 描述 |
| :--- | :--- | :--- |
| input | torch.Tensor | 第一个输入张量，支持任意维度，必须位于GPU设备上。 |
| other | torch.Tensor | 第二个输入张量或标量，将被与input逐元素比较，必须位于GPU设备上。 |

**输入约束**：
- `input` 和 `other` 必须位于GPU设备上（`cuda`）
- `input` 和 `other` 必须可广播（遵循NumPy广播规则）
- 支持的数据类型：float16, bfloat16, float32, int32
- `other` 可以是标量（Python数值或0维张量）

### （4）输出参数
| 数据类型 | 描述 |
| :--- | :--- |
| torch.Tensor | 输出张量，形状为广播后的shape，数据类型与输入相同，位于GPU设备上。 |

**输出特性**：
- 输出张量形状：`input` 和 `other` 广播后的shape
- 输出张量值：`output[i] = min(input[i], other[i])`

### （5）自动优化最大迭代轮次
10

### （6）针对性调整与KernelGen优化记档
**未达到测试目标情况**：
在处理高维Shape（如S1: [2048, 2048, 2048]）时，执行时间超过阈值，且NVIDIA Nsight显示Global Memory读写耗时占比过高（>80%），双输入导致内存访问量加倍。Tensor Cores利用率低（<20%）。

**Triton代码修改策略**：
1. **调整BLOCK_SIZE**：默认的BLOCK_SIZE可能较小，导致每个线程块处理的数据量不足，无法充分利用内存带宽。建议将`BLOCK_SIZE`从默认的64或128调整为512或1024，以增加每次数据搬运的粒度。同时需要考虑双输入的Shared Memory占用。
2. **向量化加载**：检查生成的Triton代码中是否使用了`tl.load`的掩码参数。对于连续内存区域，移除掩码或确保边界对齐，并尝试使用更大的向量宽度（如256-512B）进行加载。确保两个输入都使用向量化加载。
3. **向量化比较**：使用`tl.minimum`或`tl.where`进行向量化比较，确保比较操作充分向量化。对于FP16/BF16数据类型，使用Tensor Cores进行矩阵化比较，提高计算单元利用率。
4. **Shared Memory优化**：对于广播场景，使用Shared Memory复用广播数据，避免重复加载。

**KernelGen工具优化记档**：
- 记录在高维场景下，默认Tiling策略导致的带宽利用率不足问题。
- 建议KernelGen在后续版本中，针对大Shape输入，自动增大默认的Block Size，并优先尝试向量化加载指令。
- 建议KernelGen针对双输入算子，自动计算最优的Shared Memory分配策略，确保两个输入都能高效加载。
- 建议工具根据数据类型大小（FP16/BF16），自动计算最优的内存对齐参数（如32字节对齐）。
- 建议KernelGen针对FP16/BF16数据类型，自动利用Tensor Cores进行矩阵化比较，提高计算单元利用率。

---

## 版本二：内存访问优化版本

### （1）原始需求
针对高维Shape场景，解决内存访问非连续和带宽利用率低的问题。重点优化数据在Shared Memory中的复用，减少Global Memory的交互次数，并确保内存访问对齐。同时优化双输入数据的加载策略，提升缓存利用率。充分利用Tensor Cores加速比较计算。

### （2）算子基本信息
- **算子名称**：Minimum_Opt_Mem
- **测评设备**：NVIDIA A100 / H100
- **算子类型**：Pointwise
- **功能描述**：在基础功能之上，通过优化数据分块策略和内存访问模式，提升高维张量逐元素比较的内存带宽利用率。支持广播机制，避免广播数据的重复加载。利用Tensor Cores进行向量化比较。

### （3）输入参数
| 参数名称 | 数据类型 | 描述 |
| :--- | :--- | :--- |
| input | torch.Tensor | 第一个输入张量。 |
| other | torch.Tensor | 第二个输入张量或标量。 |

### （4）输出参数
| 数据类型 | 描述 |
| :--- | :--- |
| torch.Tensor | 输出张量。 |

### （5）自动优化最大迭代轮次
20

### （6）针对性调整与KernelGen优化记档
**未达到测试目标情况**：
虽然内存带宽利用率提升，但在非连续内存访问（如广播维度在中间）的场景下，性能提升不明显，且存在Bank Conflict风险。双输入导致Shared Memory占用增加，限制了块大小的选择。Tensor Cores利用率仍然较低（<30%）。

**Triton代码修改策略**：
1. **调整数据布局**：在Triton Kernel中，手动调整`strides`参数。对于需要广播的维度（stride=0），确保生成的代码能够正确处理，避免重复加载。对于广播维度，使用`tl.broadcast_to`指令或stride=0机制。
2. **多级缓存**：启用`num_stages`参数（例如设置为3或4），利用流水线技术，在计算当前Block的同时预取下一个Block的数据，掩盖内存延迟。对于双输入，需要更多的stage来隐藏延迟。
3. **Group Size调整**：如果存在Bank Conflict，尝试调整访问的Group大小或改变访问步长。
4. **双输入优化**：优化双输入数据的加载策略，使用双缓冲技术同时加载两个输入，减少等待时间。
5. **Tensor Cores优化**：对于FP16/BF16数据类型，使用Tensor Cores进行矩阵化比较，一次比较多个元素（如16个FP16元素）。使用`__hmin`指令（Tensor Cores优化）。

**KernelGen工具优化记档**：
- 记录非连续访问场景下的性能退化现象。
- 建议KernelGen引入自动分析Stride的逻辑，对于Stride为0的广播维度，自动生成广播逻辑（stride=0机制）而非重复加载逻辑。
- 建议工具根据数据类型大小（FP16/BF16），自动计算最优的内存对齐参数（如32字节对齐）。
- 建议KernelGen针对双输入算子，自动优化Shared Memory分配策略，确保两个输入都能高效加载，避免Shared Memory溢出。
- 建议KernelGen针对FP16/BF16数据类型，自动利用Tensor Cores进行矩阵化比较，使用`__hmin`指令，提高计算单元利用率。

---

## 版本三：高性能并行与动态Shape优化版本

### （1）原始需求
在内存优化的基础上，进一步挖掘算子性能上限。要求支持动态Shape输入（如shape=-1），消除动态分支开销；利用多核并行和向量化指令，最大化SM（Streaming Multiprocessor）的吞吐量，满足S1、S2等超高维Shape的严苛性能要求。同时优化比较指令序列，充分利用Tensor Cores和CUDA Cores特性。

### （2）算子基本信息
- **算子名称**：Minimum_Opt_HighPerf
- **测评设备**：NVIDIA A100 / H100
- **算子类型**：Pointwise
- **功能描述**：全功能的Minimum算子，支持动态Shape和广播机制，采用多核并行策略和自适应Tiling，针对NVIDIA GPU架构深度优化，实现极致推理性能。充分利用Tensor Cores、CUDA Cores和Shared Memory特性。

### （3）输入参数
| 参数名称 | 数据类型 | 描述 |
| :--- | :--- | :--- |
| input | torch.Tensor | 第一个输入张量，支持动态维度。 |
| other | torch.Tensor | 第二个输入张量或标量，支持动态维度。 |

### （4）输出参数
| 数据类型 | 描述 |
| :--- | :--- |
| torch.Tensor | 输出张量。 |

### （5）自动优化最大迭代轮次
50

### （6）针对性调整与KernelGen优化记档
**未达到测试目标情况**：
在动态Shape测试用例（S5-S8）中，性能波动大，且多核并行扩展性差（增加SM数后性能不线性增长）。计算单元利用率低（<40%），Tensor Cores利用率低（<30%），向量化比较指令未充分利用。

**Triton代码修改策略**：
1. **动态Shape处理**：确保生成的Triton代码使用`tl.program_id`动态计算索引，避免编译时常量限制。对于动态Shape，在Kernel启动时通过参数传入实际维度。
2. **并行度调整**：调整Grid的大小（`triton.cdiv`逻辑），确保生成的Block数量足够多，能够填满所有可用的SM。对于小Shape，减少Block数量以避免调度开销。
3. **Wave Scheduling**：在Triton调用中启用特定的调度策略（如`num_warps`），针对Minimum这种计算密度较低（0.25）、访存密度高的算子，适当增加每个Block的Warps数量以隐藏延迟。
4. **向量化比较优化**：使用`tl.minimum`或`tl.where`进行向量化比较，确保比较操作充分向量化，提高计算单元利用率。对于FP16/BF16数据类型，使用Tensor Cores进行矩阵化比较，一次比较多个元素（如16个FP16元素）。使用`__hmin`指令（Tensor Cores优化）。
5. **自动调优**：使用`@triton.autotune`自动调优功能，自动寻找最优的BLOCK_SIZE、num_warps、num_stages组合，适应不同Shape和硬件平台。
6. **Tensor Cores充分利用**：对于FP16/BF16数据类型，确保Tensor Cores利用率达到50-60%，使用矩阵化比较策略。

**KernelGen工具优化记档**：
- 记录动态Shape场景下的编译和运行时开销问题。
- 建议KernelGen实现自适应Tiling算法：根据输入Tensor的总大小，自动计算最优的`BLOCK_SIZE`和`num_warps`组合。
- 记录多核并行策略，建议工具在生成代码时，自动插入针对NVIDIA GPU架构的多核调度原语，确保负载均衡。
- 建议KernelGen针对双输入算子，自动优化比较指令序列，使用向量化比较指令和Tensor Cores，提高计算单元利用率。
- 建议工具实现自动调优功能（类似Triton的`@triton.autotune`），自动寻找最优配置，降低调优成本。
- 建议KernelGen根据硬件架构特性，自动选择最优的向量化宽度（256-512B），并充分利用Tensor Cores特性。
- 建议KernelGen针对FP16/BF16数据类型，自动利用Tensor Cores进行矩阵化比较，使用`__hmin`指令，确保Tensor Cores利用率达到50-60%。

---

## 附录：NVIDIA GPU优化特性

### NVIDIA GPU架构特性
| 特性 | A100 | H100 | 说明 |
|------|------|------|------|
| 理论带宽 | 2039 GB/s (HBM2e) | 3350 GB/s (HBM3) | H100带宽提升显著 |
| Tensor Cores | 3代 | 4代 | H100支持更多数据类型 |
| Shared Memory | 164 KB/SM | 228 KB/SM | H100容量更大 |
| 向量化宽度 | 128-512B | 256-512B | H100支持更宽向量 |

### NVIDIA GPU优化策略
1. **Tensor Cores利用**：FP16/BF16场景下使用Tensor Cores进行矩阵化比较
2. **Shared Memory优化**：使用Shared Memory复用广播数据，避免重复加载
3. **Coalesced访问**：确保内存访问合并，提高带宽利用率
4. **Warp调度**：优化Warp数量，隐藏内存延迟
5. **流水线优化**：使用多stage流水线，隐藏内存延迟

### Tensor Cores优化代码示例
```cpp
// 使用Tensor Cores进行向量化比较（FP16）
template <>
__global__ void minimum_kernel<half>(
    const half* __restrict__ input,
    const half* __restrict__ other,
    half* __restrict__ output,
    int64_t total_elements) {

    // 使用向量化加载（16个FP16元素 = 256位）
    using VecT = aligned_vector<half, 16>;
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;

    if (idx + 16 <= total_elements) {
        // 加载向量
        VecT input_vec = *reinterpret_cast<const VecT*>(input + idx);
        VecT other_vec = *reinterpret_cast<const VecT*>(other + idx);
        VecT output_vec;

        // 向量化比较（使用Tensor Cores）
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            output_vec.val[i] = __hmin(input_vec.val[i], other_vec.val[i]);
        }

        // 存储向量
        *reinterpret_cast<VecT*>(output + idx) = output_vec;
    }
}
```

### 性能预期
| 测试场景 | 硬件 | 理论带宽 | 预期利用率 | 预期加速比 |
|---------|------|---------|-----------|-----------|
| [2048, 2048, 2048] FP16 | A100 | 2039 GB/s | 75-85% | 2.0-3.0x |
| [4096, 1024, 4096] FP16 | A100 | 2039 GB/s | 75-85% | 2.0-3.0x |
| [1024, 512, 1024] FP16 | A100 | 2039 GB/s | 80-90% | 2.5-3.5x |
| [2048, 2048, 2048] FP16 | H100 | 3350 GB/s | 70-80% | 3.0-4.5x |

---

## 参考资料

1. NVIDIA CUDA编程指南
2. NVIDIA Tensor Cores文档
3. NVIDIA Nsight性能分析工具
4. Triton编程语言文档（NVIDIA版本）
5. NVIDIA GPU架构文档（Ampere、Hopper）
6. Roofline模型理论
7. 业界基准实现分析报告（PyTorch、TensorFlow）

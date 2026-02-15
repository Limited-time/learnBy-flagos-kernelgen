# Minimum算子定义与分析文档

## 目录
- [1. 算子定义](#1-算子定义)
- [2. 算法分析](#2-算法分析)
- [3. 性能优化策略](#3-性能优化策略)
- [4. 未达目标修改方案](#4-未达目标修改方案)

---

## 1. 算子定义

### （1）原始需求

在深度学习模型训练和推理过程中，经常需要进行逐元素比较操作。Minimum算子用于比较两个输入张量对应位置的元素，返回较小值。该算子广泛应用于激活函数限制、门控机制、梯度裁剪等场景。

**核心需求**：
- 支持高维Shape场景下的高效逐元素比较操作（如[batch, 1024, 1024, 256]）
- 在昇腾AI处理器（Ascend-snt9b / 910B）上实现高性能实现
- 确保数值精度符合IEEE FP16/BF16标准
- 支持动态Shape输入
- 支持广播机制（两个输入shape不同时可广播）
- 支持标量输入（其中一个输入可以是标量）

### （2）算子基本信息

| 属性 | 值 |
|------|-----|
| **算子名称** | Minimum |
| **测评设备** | 昇腾AI处理器（Ascend-snt9b / 910B） |
| **算子类型** | Pointwise |
| **功能描述** | 逐元素比较两个输入张量，返回对应位置的最小值。支持广播机制和标量输入，遵循NumPy广播语义：从尾部对齐维度，维度为1的可扩展为任意大小，缺失维度自动填充为1。 |

**典型应用场景**：
1. 激活函数限制：如ReLU6、Hardtanh等激活函数中限制输出范围，`min(max(x, 0), 6)`
2. 门控机制：LSTM、GRU等网络中的门控计算，通过minimum操作限制门控值范围
3. 裁剪操作：梯度裁剪、数值裁剪等防止数值溢出的场景，`min(clip_abs, abs(x))`
4. 注意力机制：Mask操作中的数值限制，防止注意力值过大
5. 条件计算：在条件分支中进行逐元素取最小值操作，实现条件逻辑

### （3）输入参数

| 参数名称 | 数据类型 | 描述 |
|---------|---------|------|
| **input** | torch.Tensor | 第一个输入张量，支持任意维度，必须位于NPU设备上 |
| **other** | torch.Tensor | 第二个输入张量或标量，将被与input逐元素比较，必须位于NPU设备上 |

**输入约束**：
- `input` 和 `other` 必须为NPU张量（`is_npu`）
- `input` 和 `other` 必须可广播（遵循NumPy广播规则）
- `input` 和 `other` 必须为连续内存格式（contiguous）
- 支持的数据类型：float16, bfloat16, float32, int32
- `other` 可以是标量（Python数值或0维张量）

**广播规则**：
1. 从尾部（最内层）开始对齐两个张量的shape
2. 若某维大小为1，可扩展为任意大小
3. 若某一侧维度缺失，自动填充为1

**广播示例**：
- `input` shape=[2,3], `other` shape=[2,3] → 直接逐元素比较
- `input` shape=[2,3], `other` shape=[1,3] → `other`在第0维广播
- `input` shape=[2,3], `other` shape=[3] → `other`在第0维填充为1后广播
- `input` shape=[2,3,4], `other` shape=[4] → `other`广播到[2,3,4]

### （4）输出参数

| 数据类型 | 描述 |
|---------|------|
| **torch.Tensor** | 输出张量，形状为广播后的shape，数据类型与输入相同 |

**输出特性**：
- 输出张量形状：`input` 和 `other` 广播后的shape
- 输出张量值：`output[i] = min(input[i], other[i])`
- 内存布局：与输入保持一致（ND格式）

### （5）自动优化最大迭代轮次

**建议最大迭代轮次：5轮**

| 轮次 | 优化重点 | 预期加速比提升 |
|-----|---------|---------------|
| 第1轮 | 网格配置优化（一维→二维） | 10-20% |
| 第2轮 | 块大小调整（BLOCK_M/BLOCK_N） | 15-30% |
| 第3轮 | Warp和Stage配置优化 | 10-20% |
| 第4轮 | 内存访问优化（对齐、Cache策略、双输入加载） | 10-15% |
| 第5轮 | 向量化比较指令优化或自动调优 | 5-10% |

**终止条件**：
- 加速比达到目标值（通常 > 1.0）
- 连续2轮优化加速比提升 < 5%
- 达到最大迭代轮次（5轮）

---

## 2. 算法分析

### 2.1 算法复杂度分析

#### 时间复杂度

**Minimum操作本质**：逐元素比较操作

设输入张量 `input` 的形状为 `(M, K, N)`，`other` 的形状为广播后与之相同：
- `T = M × K × N`：输出张量的总元素数

**时间复杂度**：O(T)

**分析**：
- 需要比较 `T` 对元素
- 每次比较包含：读取两个输入元素、比较、写入一个输出元素
- 总操作次数：`T` 次比较操作

#### 空间复杂度

**空间复杂度**：O(T)

**分析**：
- 输入张量 `input`：`T` 个元素
- 输入张量 `other`：根据广播规则，可能小于 `T` 个元素
- 输出张量 `out`：`T` 个元素
- 总内存占用：`T + other_size + T = 2T + other_size`

### 2.2 计算密度分析

**计算密度** = 计算操作数 / 总操作数

对于Minimum算子，每个元素的处理包含：
- **计算操作数**：1次比较操作
- **内存访问操作数**：
  - 读取 `input`：1次
  - 读取 `other`：1次（广播后）
  - 写入 `out`：1次
  - 总计：3次

**总操作数** = 1次计算 + 3次内存访问 = 4次

**计算密度** = 1 / 4 = 0.25

**业界基准对比**：
- PyTorch/CUDA/TensorFlow/Triton：计算密度均为0.25
- 广播算子（Broadcast）：计算密度≈0（纯数据复制）

**结论**：Minimum算子是**内存带宽受限**（Memory Bandwidth Bound）算子，但相比Broadcast算子（计算密度≈0），有较低的计算负载。优化重点应放在：
1. 减少内存访问次数（通过数据复用）
2. 提高内存带宽利用率（向量化加载/存储）
3. 优化内存访问模式（对齐、连续性）
4. 优化比较指令（向量化比较、SIMD）
5. 充分利用硬件特性（如昇腾的Vector计算单元、NVIDIA的Tensor Cores）

### 2.3 Roofline模型分析

**Roofline模型**用于分析算子性能上限：

```
性能上限 = min(计算上限, 内存带宽上限)
```

对于Minimum算子：
- **计算上限**：理论计算峰值 × 计算单元利用率
- **内存带宽上限**：理论内存带宽 × 带宽利用率

**性能瓶颈**：内存带宽（但计算也成为次要瓶颈）

**优化方向**：
1. 提高内存带宽利用率（从理论带宽的10-20%提升到60-80%）
2. 提高计算单元利用率（通过向量化比较指令）
3. 减少内存访问次数（通过数据复用）
4. 优化内存访问模式（缓存友好、对齐）

**昇腾AI处理器特性**：
- 理论内存带宽：约1200 GB/s（Ascend 910B）
- Local Memory（UB）：256KB，访问速度快，适合数据复用
- 数据对齐要求：32字节（8个float16元素）
- 向量化指令支持：Vector计算单元支持SIMD比较操作

### 2.4 性能瓶颈分析

#### 瓶颈1：内存访问模式

**问题**：
- 一维网格下，每个block需要处理完整的N维度
- 当N较大时，每个block的内存访问不连续
- 缓存命中率低
- 双输入导致内存访问量加倍

**解决方案**：
- 使用二维网格，分块处理
- 确保内存访问对齐（32字节对齐，即8个float16元素）
- 使用向量化加载指令（Ascend C的DataCopy指令）
- 优化双输入数据的加载策略

**昇腾AI处理器优化**：
- 使用DataCopy指令进行数据搬运
- 确保数据对齐到32字节边界
- 利用Vector计算单元的SIMD能力
- 使用双缓冲技术同时加载两个输入

#### 瓶颈2：数据重复加载

**问题**：
- 每个block都需要加载完整的input和other数据
- 数据在全局内存中被重复加载
- 浪费内存带宽
- 昇腾AI处理器的Local Memory（UB）未充分利用

**解决方案**：
- 在block内部复用数据
- 使用Local Memory（UB）缓存数据
- 优化块大小以减少重复加载
- 对于广播维度，避免重复加载广播数据

**昇腾AI处理器优化**：
- 使用UB（Unified Buffer）缓存input和other数据
- 通过DataCopy指令将数据从Global Memory搬运到UB
- 在单个AI Core内多次复用UB中的数据
- 对于广播维度，使用广播指令而非重复加载

#### 瓶颈3：并行度不足

**问题**：
- 一维网格下，并行度受限于 `T / (BLOCK_M × BLOCK_N)`
- 当块大小过大时，并行度不足
- 昇腾AI处理器的多核并行能力未充分利用

**解决方案**：
- 使用二维网格提高并行度
- 调整块大小以平衡并行度和计算密度
- 优化warp数量

**昇腾AI处理器优化**：
- 利用多核并行（Ascend 910B有多个AI Core）
- 设计合理的Tiling策略，充分利用多核
- 使用多核并行比较策略（分块并行+核内流水）

#### 瓶颈4：比较指令效率

**问题**：
- 逐元素比较未充分利用向量化计算
- 串行处理效率低
- 昇腾AI处理器的Vector计算单元未充分利用

**解决方案**：
- 应用向量化指令(SIMD)加速比较计算
- 优化比较指令序列，减少指令开销
- 使用流水线技术隐藏内存延迟

**昇腾AI处理器优化**：
- 使用Vector计算单元的SIMD比较指令
- 一次比较多个元素
- 使用Duplicate指令进行广播
- 使用DataCopy指令进行向量化搬运

#### 瓶颈5：内存延迟

**问题**：
- 全局内存访问延迟高（约400-800个时钟周期）
- 单线程或单warp无法有效隐藏延迟
- 昇腾AI处理器的DMA搬运单元未充分利用

**解决方案**：
- 使用流水线（pipelining）技术
- 增加stage数量（num_stages）
- 使用异步计算掩盖内存延迟

**昇腾AI处理器优化**：
- 使用DMA的双缓冲技术
- 实现数据搬运与计算的流水线
- 利用MTE2/MTE3单元进行异步数据搬运

### 2.5 数据流分析

**Minimum操作的数据流**：

```
输入数据流：
Global Memory (input) → DMA搬运 → Local Memory (UB) → Vector计算 (比较) → Global Memory (out)
Global Memory (other) → DMA搬运 → Local Memory (UB) → Vector计算 (比较) → Global Memory (out)

优化后的数据流：
Global Memory (input) → DMA搬运 (向量化) → Local Memory (UB) → 复用 → Vector计算 (SIMD比较) → DMA搬运 (向量化) → Global Memory (out)
Global Memory (other) → DMA搬运 (向量化) → Local Memory (UB) → 复用 → Vector计算 (SIMD比较) → DMA搬运 (向量化) → Global Memory (out)
```

**关键优化点**：
1. **Load优化**：使用DataCopy指令，确保内存对齐，同时加载两个输入
2. **比较优化**：在Vector计算单元使用SIMD比较指令
3. **Store优化**：使用DataCopy指令，确保内存对齐
4. **流水线优化**：使用多stage流水线隐藏内存延迟
5. **广播优化**：对于广播维度，使用广播指令而非重复加载

**昇腾AI处理器特性**：
- Scalar计算单元：执行地址计算、循环控制
- Vector计算单元：执行向量运算（SIMD比较）
- DMA搬运单元：负责数据搬运，支持双缓冲
- Local Memory（UB）：256KB，用于数据缓存和复用

---

## 3. 性能优化策略

### 3.1 网格配置优化

#### 策略1：二维网格 vs 一维网格

**一维网格**（初始方案）：
```python
grid = lambda meta: (triton.cdiv(M * K * N, meta['BLOCK_M'] * meta['BLOCK_N']),)
pid = tl.program_id(axis=0)
```

**问题**：
- 每个block需要处理完整的N维度
- 当N较大时，内存访问不连续
- 并行度不足

**二维网格**（优化方案）：
```python
grid = lambda meta: (triton.cdiv(M * K, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
```

**优势**：
- 提高并行度：`(M*K)/BLOCK_M × N/BLOCK_N` 个blocks
- 内存访问更连续：每个block处理一个tile
- 更好的负载均衡
- 更容易实现数据复用

**昇腾AI处理器实现**：
```cpp
// 使用多核并行策略
// 将任务分配给多个AI Core
uint32_t total_cores = GetBlockIdx() * GetBlockNum() + GetCoreIdx();
uint32_t core_num = GetBlockNum() * GetAICoreNum();

// 二维分块策略
uint32_t tile_m = (M * K + core_num - 1) / core_num;
uint32_t tile_n = (N + core_num - 1) / core_num;
```

**性能提升**：10-20%

### 3.2 块大小优化

#### 策略2：调整BLOCK_M和BLOCK_N

**初始配置**（性能差）：
```python
BLOCK_M = 64
BLOCK_N = 256
```

**问题**：
- BLOCK_M过大导致寄存器压力
- 负载不均衡
- Block数量过少
- UB容量受限

**第一轮优化**：
```python
BLOCK_M = 16
BLOCK_N = 256
```

**效果**：
- 减少寄存器压力
- 提高block数量
- 适合UB容量

**第二轮优化**：
```python
BLOCK_M = 32
BLOCK_N = 512
```

**效果**：
- 提高计算密度
- 减少grid启动开销
- 更好利用内存带宽
- 需要考虑双输入的UB占用

**昇腾AI处理器实现**：
```cpp
// 根据UB大小选择最优块大小
constexpr uint32_t UB_SIZE = 256 * 1024;  // 256KB
constexpr uint32_t ELEMENT_SIZE = sizeof(half);  // float16

// 计算最优块大小（考虑双输入）
uint32_t BLOCK_N = UB_SIZE / (ELEMENT_SIZE * 3);  // input + other + output
uint32_t BLOCK_M = UB_SIZE / (ELEMENT_SIZE * 6);  // 考虑多行复用
```

**选择原则**：
- BLOCK_M：16, 32, 64（根据寄存器容量和UB大小选择）
- BLOCK_N：256, 512, 1024（根据内存对齐要求选择，必须是8的倍数）
- 权衡：更大的块 → 更高的计算密度 vs 更少的并行度
- 需要考虑双输入的UB占用：`BLOCK_M * BLOCK_N * 3 * ELEMENT_SIZE <= UB_SIZE`

**性能提升**：15-30%

### 3.3 Warp和Stage配置优化

#### 策略3：调整num_warps和num_stages

**初始配置**（性能差）：
```python
num_warps = 8
num_stages = 3
```

**问题**：
- num_warps过大导致线程管理开销
- num_stages配置不匹配block大小
- 对于双输入，需要更多stage来隐藏延迟

**优化配置**：
```python
num_warps = 4
num_stages = 4
```

**选择原则**：
- num_warps：2, 4, 8（根据block大小选择，通常4适合中等块大小）
- num_stages：1, 2, 3, 4, 5（根据内存延迟和计算强度选择，双输入需要更多stage）

**昇腾AI处理器实现**：
```cpp
// 使用流水线技术隐藏内存延迟
// 双缓冲策略
LocalTensor<half> input_ub = in_queue.AllocTensor<half>();
LocalTensor<half> other_ub = in_queue.AllocTensor<half>();
LocalTensor<half> out_ub = out_queue.AllocTensor<half>();

// 流水线阶段
DataCopy(input_ub, input_ptr, BLOCK_N);  // 阶段1：数据搬运
DataCopy(other_ub, other_ptr, BLOCK_N);  // 阶段2：数据搬运
Mins(out_ub, input_ub, other_ub, BLOCK_N);  // 阶段3：比较计算
DataCopy(out_ptr, out_ub, BLOCK_N);  // 阶段4：结果存储
```

**性能提升**：10-20%

### 3.4 内存访问优化

#### 策略4：内存对齐优化

**优化前**：
```python
col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

**优化后**：
```python
col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
tl.multiple_of(col_ids, 64)  # 确保对齐到64个元素（256B）
```

**优势**：
- 提高内存访问效率
- 减少内存访问延迟
- 更好利用内存带宽

**昇腾AI处理器实现**：
```cpp
// 确保数据对齐到32字节边界
constexpr uint32_t ALIGNMENT = 32;  // 32字节对齐

// 使用对齐的指针
GlobalTensor<half> input_ptr;
input_ptr.SetGlobal((__gm__ half*)ALIGN_UP((uint64_t)input_addr, ALIGNMENT));

GlobalTensor<half> other_ptr;
other_ptr.SetGlobal((__gm__ half*)ALIGN_UP((uint64_t)other_addr, ALIGNMENT));

// 使用DataCopy进行对齐的数据搬运
DataCopy(input_ub, input_ptr, BLOCK_N, {1, 1}, {1, 1});
DataCopy(other_ub, other_ptr, BLOCK_N, {1, 1}, {1, 1});
```

**性能提升**：5-10%

#### 策略5：向量化加载/存储

**优化前**：
```python
input_tile = tl.load(input_ptr + col_ids, mask=mask_cols, other=0.0)
other_tile = tl.load(other_ptr + col_ids, mask=mask_cols, other=0.0)
```

**优化后**：
```python
input_tile = tl.load(
    input_ptr + col_ids,
    mask=mask_cols,
    other=0.0,
    eviction_policy='evict_last'  # 优化缓存策略
).to(tl.float32)

other_tile = tl.load(
    other_ptr + col_ids,
    mask=mask_cols,
    other=0.0,
    eviction_policy='evict_last'  # 优化缓存策略
).to(tl.float32)
```

**优势**：
- 减少内存访问次数
- 提高缓存命中率
- 更好利用内存带宽
- 同时加载两个输入

**昇腾AI处理器实现**：
```cpp
// 使用Vector计算单元的SIMD指令
// 一次处理多个元素
constexpr uint32_t BLOCK_SIZE = 256;  // 每次处理256个元素

// 使用DataCopy进行向量化搬运（同时加载两个输入）
DataCopy(input_ub, input_ptr, BLOCK_SIZE, {1, BLOCK_SIZE}, {1, BLOCK_SIZE});
DataCopy(other_ub, other_ptr, BLOCK_SIZE, {1, BLOCK_SIZE}, {1, BLOCK_SIZE});

// 使用Mins指令进行向量化比较
Mins(out_ub, input_ub, other_ub, BLOCK_SIZE, 1, 1, 1);

// 使用DataCopy进行向量化存储
DataCopy(out_ptr, out_ub, BLOCK_SIZE, {1, BLOCK_SIZE}, {1, BLOCK_SIZE});
```

**性能提升**：5-15%

### 3.5 数据复用优化

#### 策略6：Tile级数据复用

**核心思想**：在block内部复用数据

**实现**：
```python
# 每个程序加载一次input和other块
input_tile = tl.load(input_ptr + col_ids, mask=mask_cols, other=0.0).to(tl.float32)
other_tile = tl.load(other_ptr + col_ids, mask=mask_cols, other=0.0).to(tl.float32)

# 在BLOCK_M行中复用input和other
for i in range(BLOCK_M):
    row_ids = pid_m * BLOCK_M + i
    # 比较计算
    val = tl.minimum(input_tile, other_tile)
    tl.store(out_ptr + row_ids * stride + col_ids, val, mask=mask_cols)
```

**优势**：
- 减少全局内存访问次数
- 提高数据局部性
- 降低内存带宽压力

**昇腾AI处理器实现**：
```cpp
// 在UB中缓存input和other数据，并在多行中复用
LocalTensor<half> input_ub = in_queue.AllocTensor<half>();
LocalTensor<half> other_ub = in_queue.AllocTensor<half>();
LocalTensor<half> out_ub = out_queue.AllocTensor<half>();

// 只搬运一次input和other数据到UB
DataCopy(input_ub, input_ptr, BLOCK_N);
DataCopy(other_ub, other_ptr, BLOCK_N);

// 在BLOCK_M行中复用input和other数据
for (uint32_t i = 0; i < BLOCK_M; ++i) {
    Mins(out_ub[i], input_ub, other_ub, BLOCK_N);
}

// 一次性输出所有结果
DataCopy(out_ptr, out_ub, BLOCK_M * BLOCK_N);
```

**性能提升**：10-20%

### 3.6 广播优化

#### 策略7：广播维度优化

**核心思想**：对于广播维度，使用广播指令而非重复加载

**实现**：
```python
# 判断是否需要广播
if input_shape != other_shape:
    # 对于广播维度，使用广播指令
    if input_dim == 1:
        input_tile = tl.broadcast_to(input_tile, (BLOCK_M, BLOCK_N))
    elif other_dim == 1:
        other_tile = tl.broadcast_to(other_tile, (BLOCK_M, BLOCK_N))
```

**优势**：
- 避免重复加载广播数据
- 减少内存访问次数
- 提高计算效率

**昇腾AI处理器实现**：
```cpp
// 使用Duplicate指令进行广播
if (input_dim == 1) {
    Duplicate(input_ub, input_ub[0], BLOCK_N, 1, 1, 1);
}
if (other_dim == 1) {
    Duplicate(other_ub, other_ub[0], BLOCK_N, 1, 1, 1);
}
```

**性能提升**：5-10%

### 3.7 向量化比较优化

#### 策略8：向量化比较指令

**核心思想**：使用SIMD指令加速比较计算

**实现**：
```python
# 使用向量化比较指令
val = tl.minimum(input_tile, other_tile)
```

**优势**：
- 一次比较多个元素
- 提高计算单元利用率
- 减少指令开销

**昇腾AI处理器实现**：
```cpp
// 使用Vector计算单元的SIMD比较指令
constexpr uint32_t VECTOR_SIZE = 256;  // 向量化大小

// 使用Mins指令进行向量化比较
Mins(out_ub, input_ub, other_ub, VECTOR_SIZE, 1, 1, 1);
```

**性能提升**：5-15%

---

## 4. 未达目标修改方案

### 4.1 问题诊断框架

当加速比未达到目标时，按以下步骤诊断：

#### 步骤1：确认修改类型

```
问题：我修改的是什么？
├─ BLOCK_M, BLOCK_N, num_warps, num_stages → 性能参数修改
├─ 网格配置（一维/二维） → 性能参数修改
├─ 函数签名（参数个数、类型） → 接口修改（需同步所有文件）
├─ 数据类型约束 → 接口修改（需同步所有文件）
├─ 比较公式/数学运算 → 逻辑修改（需同步所有文件）
└─ 广播逻辑 → 逻辑修改（需同步所有文件）
```

#### 步骤2：性能瓶颈分析

使用以下工具分析性能瓶颈：
- **APROF**：昇腾性能分析工具
- **msprof**：昇腾性能分析工具
- **NVIDIA Nsight**：如果使用CUDA

**关注指标**：
- 内存带宽利用率（目标：>60%）
- Cache命中率（目标：>80%）
- 计算单元利用率（对于Minimum算子中等，目标：>40%）
- 内存访问延迟
- 指令吞吐量

#### 步骤3：确定优化方向

根据瓶颈确定优化方向：

| 瓶颈类型 | 优化方向 | 具体措施 |
|---------|---------|---------|
| 内存带宽利用率低 | 内存访问优化 | 对齐、向量化、Cache策略 |
| Cache命中率低 | 数据局部性优化 | 调整块大小、数据复用 |
| 并行度不足 | 网格配置优化 | 二维网格、调整块大小 |
| 内存延迟高 | 流水线优化 | 增加stage、异步计算 |
| 计算单元利用率低 | 向量化优化 | SIMD比较指令、指令序列优化 |
| 广播效率低 | 广播优化 | 广播指令、避免重复加载 |

### 4.2 具体修改方案

#### 方案1：动态调整BLOCK_M和BLOCK_N

**适用场景**：不同Shape下性能差异大

**实现方式**：
```python
def get_optimal_block_size(M, K, N):
    """
    根据输入张量形状动态选择最优块大小
    考虑双输入的UB占用
    """
    total_elements = M * K

    if N > 4096:
        BLOCK_N = 512  # 大N，增大BLOCK_N以减少列方向的grid数量
    elif N > 2048:
        BLOCK_N = 256
    else:
        BLOCK_N = 128

    if total_elements > 4096:
        BLOCK_M = 32  # 大M*K，适当增大BLOCK_M以减少行方向的grid数量
    elif total_elements > 2048:
        BLOCK_M = 16
    else:
        BLOCK_M = 8

    return BLOCK_M, BLOCK_N

# 在Python包装器中使用
BLOCK_M, BLOCK_N = get_optimal_block_size(M, K, N)
```

**预期效果**：
- 适应不同Shape场景
- 提高泛化能力
- 性能提升：5-15%

#### 方案2：使用自动调优功能

**适用场景**：难以手动确定最优参数

**实现方式**：
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 1024}, num_warps=4, num_stages=4),
    ],
    key=['M', 'K', 'N'],
)
def minimum_autotune(input_ptr, other_ptr, out_ptr, M, K, N, stride_input, stride_other, stride_out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 内核实现
    pass
```

**预期效果**：
- 自动找到最优配置
- 适应不同硬件平台
- 性能提升：5-20%

#### 方案3：内存访问优化

**适用场景**：内存带宽利用率低

**实现方式**：
```python
# 1. 使用向量化加载/存储
input_tile = tl.load(
    input_ptr + col_ids,
    mask=mask_cols,
    other=0.0,
    eviction_policy='evict_last'  # 优化缓存策略
).to(tl.float32)

other_tile = tl.load(
    other_ptr + col_ids,
    mask=mask_cols,
    other=0.0,
    eviction_policy='evict_last'  # 优化缓存策略
).to(tl.float32)

# 2. 确保内存对齐
tl.multiple_of(col_ids, 64)  # 对齐到64个元素（256B）

# 3. 使用向量化存储
tl.store(out_ptr + row_ids * stride + col_ids, val, mask=mask_cols)
```

**昇腾AI处理器实现**：
```cpp
// 1. 确保数据对齐
constexpr uint32_t ALIGNMENT = 32;  // 32字节对齐

// 2. 使用DataCopy进行向量化搬运
DataCopy(input_ub, input_ptr, BLOCK_N, {1, BLOCK_N}, {1, BLOCK_N});
DataCopy(other_ub, other_ptr, BLOCK_N, {1, BLOCK_N}, {1, BLOCK_N});

// 3. 使用Mins进行向量化比较
Mins(out_ub, input_ub, other_ub, BLOCK_N, 1, 1, 1);

// 4. 使用DataCopy进行向量化存储
DataCopy(out_ptr, out_ub, BLOCK_N, {1, BLOCK_N}, {1, BLOCK_N});
```

**预期效果**：
- 提高内存带宽利用率
- 减少内存访问延迟
- 性能提升：10-15%

#### 方案4：广播优化

**适用场景**：广播场景性能差

**实现方式**：
```python
# 判断是否需要广播
if input.shape != other.shape:
    # 计算广播后的shape
    broadcast_shape = np.broadcast_shapes(input.shape, other.shape)

    # 对于广播维度，使用广播指令
    if input.ndim < other.ndim:
        input = tl.expand_dims(input, axis=range(other.ndim - input.ndim))

    for i in range(len(broadcast_shape)):
        if input.shape[i] == 1:
            input = tl.broadcast_to(input, broadcast_shape)
        if other.shape[i] == 1:
            other = tl.broadcast_to(other, broadcast_shape)
```

**昇腾AI处理器实现**：
```cpp
// 使用Duplicate指令进行广播
if (input_dim == 1) {
    Duplicate(input_ub, input_ub[0], BLOCK_N, 1, 1, 1);
}
if (other_dim == 1) {
    Duplicate(other_ub, other_ub[0], BLOCK_N, 1, 1, 1);
}
```

**预期效果**：
- 避免重复加载广播数据
- 提高广播效率
- 性能提升：5-10%

#### 方案5：向量化比较优化

**适用场景**：计算单元利用率低

**实现方式**：
```python
# 使用向量化比较指令
val = tl.minimum(input_tile, other_tile)

# 或者使用更底层的向量化指令
val = tl.where(input_tile < other_tile, input_tile, other_tile)
```

**昇腾AI处理器实现**：
```cpp
// 使用Vector计算单元的SIMD比较指令
constexpr uint32_t VECTOR_SIZE = 256;  // 向量化大小

// 使用Mins指令进行向量化比较
Mins(out_ub, input_ub, other_ub, VECTOR_SIZE, 1, 1, 1);
```

**预期效果**：
- 提高计算单元利用率
- 减少指令开销
- 性能提升：5-15%

#### 方案6：多核并行优化

**适用场景**：并行度不足

**实现方式**：
```python
# 使用多核并行策略
# 将任务分配给多个AI Core
def minimum_multicore(input, other, out):
    # 获取可用的AI Core数量
    num_cores = get_num_cores()

    # 将任务分配给多个Core
    chunk_size = (input.numel() + num_cores - 1) // num_cores

    # 并行执行
    for core_id in range(num_cores):
        start = core_id * chunk_size
        end = min(start + chunk_size, input.numel())
        minimum_kernel[input[start:end], other[start:end], out[start:end]]()
```

**昇腾AI处理器实现**：
```cpp
// 使用多核并行策略
uint32_t total_cores = GetBlockIdx() * GetBlockNum() + GetCoreIdx();
uint32_t core_num = GetBlockNum() * GetAICoreNum();

// 计算每个Core处理的范围
uint32_t chunk_size = (total_elements + core_num - 1) / core_num;
uint32_t start = total_cores * chunk_size;
uint32_t end = min(start + chunk_size, total_elements);

// 在当前Core上处理
for (uint32_t i = start; i < end; ++i) {
    out[i] = min(input[i], other[i]);
}
```

**预期效果**：
- 提高并行度
- 充分利用多核
- 性能提升：10-20%

### 4.3 针对性调整与KernelGen优化记档

**未达到测试目标情况**：
在处理高维Shape（如S1: [2048, 2048, 2048]）时，执行时间超过阈值，且APROF显示Global Memory读写耗时占比过高（>80%），计算单元利用率低（<40%）。

**Triton代码修改策略**：
1. **调整BLOCK_SIZE**：默认的BLOCK_SIZE可能较小，导致每个线程块处理的数据量不足，无法充分利用内存带宽。建议将`BLOCK_SIZE`从默认的64或128调整为512或1024，以增加每次数据搬运的粒度。同时需要考虑双输入的UB占用。
2. **向量化加载**：检查生成的Triton代码中是否使用了`tl.load`的掩码参数。对于连续内存区域，移除掩码或确保边界对齐，并尝试使用更大的向量宽度（如`tl.vectorized`提示）进行加载。同时确保两个输入都使用向量化加载。
3. **向量化比较**：使用`tl.minimum`或`tl.where`进行向量化比较，确保比较操作充分向量化。
4. **广播优化**：对于广播场景，使用`tl.broadcast_to`指令而非重复加载。

**KernelGen工具优化记档**：
- 记录在高维场景下，默认Tiling策略导致的带宽利用率不足问题。
- 建议KernelGen在后续版本中，针对大Shape输入，自动增大默认的Block Size，并优先尝试向量化加载指令。
- 建议KernelGen针对双输入算子，自动计算最优的UB分配策略，确保两个输入都能高效加载。
- 建议工具根据数据类型大小（FP16/BF16），自动计算最优的内存对齐参数（如32字节对齐）。
- 建议KernelGen实现自动广播逻辑优化，对于广播维度，自动生成广播指令而非重复加载逻辑。

**未达到测试目标情况**：
虽然内存带宽利用率提升，但在非连续内存访问（如广播维度在中间）的场景下，性能提升不明显，且存在Bank Conflict风险，计算单元利用率仍然较低。

**Triton代码修改策略**：
1. **调整数据布局**：在Triton Kernel中，手动调整`strides`参数。对于需要广播的维度（stride=0），确保生成的代码能够正确处理，避免重复加载。
2. **多级缓存**：启用`num_stages`参数（例如设置为3或4），利用流水线技术，在计算当前Block的同时预取下一个Block的数据，掩盖内存延迟。对于双输入，需要更多的stage。
3. **Group Size调整**：如果存在Bank Conflict，尝试调整访问的Group大小或改变访问步长。
4. **向量化比较优化**：使用`tl.minimum`或`tl.where`进行向量化比较，确保比较操作充分向量化。

**KernelGen工具优化记档**：
- 记录非连续访问场景下的性能退化现象。
- 建议KernelGen引入自动分析Stride的逻辑，对于Stride为0的广播维度，自动生成广播逻辑而非重复加载逻辑。
- 建议工具根据数据类型大小（FP16/BF16），自动计算最优的内存对齐参数（如32字节对齐）。
- 建议KernelGen针对双输入算子，自动优化比较指令序列，使用向量化比较指令。

**未达到测试目标情况**：
在动态Shape测试用例（S5-S8）中，性能波动大，且多核并行扩展性差（增加核数后性能不线性增长），计算单元利用率低。

**Triton代码修改策略**：
1. **动态Shape处理**：确保生成的Triton代码使用`tl.program_id`动态计算索引，避免编译时常量限制。对于动态Shape，在Kernel启动时通过参数传入实际维度。
2. **并行度调整**：调整Grid的大小（`triton.cdiv`逻辑），确保生成的Block数量足够多，能够填满所有可用的AI Core。对于小Shape，减少Block数量以避免调度开销。
3. **Wave Scheduling**：如果硬件支持，在Triton调用中启用特定的调度策略（如`num_warps`），针对Minimum这种计算密度低、访存密度高的算子，适当增加每个Block的Warps数量以隐藏延迟。
4. **向量化比较优化**：确保比较操作充分向量化，提高计算单元利用率。

**KernelGen工具优化记档**：
- 记录动态Shape场景下的编译和运行时开销问题。
- 建议KernelGen实现自适应Tiling算法：根据输入Tensor的总大小，自动计算最优的`BLOCK_SIZE`和`num_warps`组合。
- 记录多核并行策略，建议工具在生成代码时，自动插入针对Ascend架构的多核调度原语，确保负载均衡。
- 建议KernelGen针对双输入算子，自动优化比较指令序列，使用向量化比较指令，提高计算单元利用率。

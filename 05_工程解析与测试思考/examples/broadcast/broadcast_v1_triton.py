import torch
import triton
import triton.language as tl


@triton.jit
def broadcast_v1(x_ptr,        # *Pointer* to high-dim input tensor (unused, shape reference).
                 bias_ptr,     # *Pointer* to low-dim bias tensor to be broadcasted along last dim.
                 out_ptr,      # *Pointer* to output tensor, same shape as x.
                 P,            # Number of rows after flattening all leading dims: P = prod(x.shape[:-1])
                 D,            # Last dimension size (must equal bias.numel()).
                 stride_x_row, # Stride between rows in x (elements) - kept for future extension.
                 stride_o_row, # Stride between rows in out (elements).
                 BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr):
    """
    Triton kernel实现张量广播操作

    将低维偏置张量沿最后一个维度广播到高维输出张量中

    参数:
        x_ptr: 高维输入张量指针（未使用，仅作形状参考）
        bias_ptr: 沿最后一个维度广播的低维偏置张量指针
        out_ptr: 输出张量指针，形状与x相同
        P: 展平所有前导维度后的行数：P = prod(x.shape[:-1])
        D: 最后一个维度大小（必须等于bias.numel()）
        stride_x_row: x中行之间的步长（元素）- 保留用于未来扩展
        stride_o_row: out中行之间的步长（元素）
        BLOCK_M: 行方向的块大小（编译时常量）
        BLOCK_N: 列方向的块大小（编译时常量）
    """
    # 使用二维网格以提高并行度和内存访问效率
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 计算当前程序处理的行和列范围
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 创建掩码
    mask_rows = row_ids < P
    mask_cols = col_ids < D
    mask = mask_rows[:, None] & mask_cols[None, :]

    # 确保列偏移量是64个元素的倍数（fp32为256B）以获得更好的内存对齐
    tl.multiple_of(col_ids, 64)

    # 每个程序加载一次bias块并在BLOCK_M行中重用
    bias_tile = tl.load(bias_ptr + col_ids, mask=mask_cols, other=0.0).to(tl.float32)

    # 计算展平(P, D)布局中的每行基础偏移量
    out_offsets = row_ids[:, None] * stride_o_row + col_ids[None, :]

    # 沿BLOCK_M行广播bias
    val_tile = tl.broadcast_to(bias_tile[None, :], (BLOCK_M, BLOCK_N))

    # 存储回输出；Triton会在需要时转换为指针元素的数据类型
    tl.store(out_ptr + out_offsets, val_tile, mask=mask)


# 在定义同名Python包装器之前保持对Triton内核的引用
_broadcast_v1_kernel = broadcast_v1


def broadcast_v1(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    将偏置张量沿最后一个维度广播到输入张量形状
    
    参数:
        x: 高维输入张量
        bias: 要广播的低维偏置张量
        
    返回:
        广播后的输出张量，形状与输入x相同
        
    异常:
        AssertionError: 当输入不是CUDA张量或维度不匹配时抛出
    """
    # 确保CUDA张量并保证ND格式的连续性
    assert x.is_npu and bias.is_npu, "x and bias must be CUDA tensors"
    x_c = x.contiguous()
    bias_c = bias.contiguous()
    assert x_c.dim() >= 1, "x must have at least 1 dimension"
    D = bias_c.numel()
    assert x_c.shape[-1] == D, "bias last dimension must match x's last dimension"
    P = x_c.numel() // D

    # 分配与x数据类型和形状匹配的输出张量
    out = torch.empty_like(x_c)

    # 优化块大小以提高性能
    # 尝试更大的块以提高计算密度和内存带宽利用率
    BLOCK_N = 512  # 增大BLOCK_N以减少列方向的grid数量
    BLOCK_M = 32   # 增大BLOCK_M以提高每行的计算密度

    # 对于连续的ND格式，展平行步长等于最后一个维度大小
    stride_row = D

    # 使用二维网格以提高并行度和内存访问效率
    grid = lambda meta: (triton.cdiv(P, meta['BLOCK_M']), triton.cdiv(D, meta['BLOCK_N']))

    # 调整num_warps和num_stages以优化性能
    # num_warps=4适合较大的块大小
    # num_stages=3以更好地隐藏内存延迟
    _broadcast_v1_kernel[grid](
        x_c, bias_c, out,
        P, D,
        stride_row, stride_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3,
    )
    return out
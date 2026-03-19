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
    Triton内核函数，用于将低维偏置张量广播到高维输入张量的最后一维。
    
    参数:
        x_ptr (tl.tensor): 指向高维输入张量的指针（未使用，仅作为形状参考）。
        bias_ptr (tl.tensor): 指向低维偏置张量的指针，需沿最后一维进行广播。
        out_ptr (tl.tensor): 指向输出张量的指针，形状与输入张量相同。
        P (int): 展平所有前导维度后的行数，即 P = prod(x.shape[:-1])。
        D (int): 最后一维的大小，必须等于偏置张量的元素数量。
        stride_x_row (int): 输入张量中行之间的步长（元素数），保留以供未来扩展。
        stride_o_row (int): 输出张量中行之间的步长（元素数）。
        BLOCK_M (tl.constexpr): 每个程序处理的行数块大小。
        BLOCK_N (tl.constexpr): 每个程序处理的列数块大小。
    """
    # 获取当前程序的行和列索引
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 计算当前块的行和列范围
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 生成掩码以避免越界访问
    mask_rows = row_ids < P
    mask_cols = col_ids < D
    mask = mask_rows[:, None] & mask_cols[None, :]

    # 确保列偏移量是64的倍数以优化内存对齐
    tl.multiple_of(col_ids, 64)

    # 加载偏置张量的一个tile，并在多个行中复用
    bias_tile = tl.load(bias_ptr + col_ids, mask=mask_cols, other=0.0).to(tl.float32)

    # 计算输出张量中每个行的基址偏移
    out_offsets = row_ids[:, None] * stride_o_row + col_ids[None, :]

    # 将偏置张量广播到BLOCK_M行
    val_tile = tl.broadcast_to(bias_tile[None, :], (BLOCK_M, BLOCK_N))

    # 将结果存储到输出张量中
    tl.store(out_ptr + out_offsets, val_tile, mask=mask)


# 保存Triton内核的句柄，以便后续定义同名Python包装函数
_broadcast_v1_kernel = broadcast_v1


def broadcast_v1(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Python包装函数，调用Triton内核实现偏置张量的广播操作。
    
    参数:
        x (torch.Tensor): 高维输入张量，必须位于CUDA设备上。
        bias (torch.Tensor): 低维偏置张量，必须位于CUDA设备上。
        
    返回:
        torch.Tensor: 广播后的输出张量，形状与输入张量相同。
    """
    # 确保输入张量位于CUDA设备上并转换为连续格式
    assert x.is_npu and bias.is_npu, "x and bias must be CUDA tensors"
    x_c = x.contiguous()
    bias_c = bias.contiguous()
    assert x_c.dim() >= 1, "x must have at least 1 dimension"
    D = bias_c.numel()
    assert x_c.shape[-1] == D, "bias last dimension must match x's last dimension"
    P = x_c.numel() // D

    # 分配与输入张量类型和形状匹配的输出张量
    out = torch.empty_like(x_c)

    # 设置块大小：BLOCK_N为64的倍数以优化内存对齐
    BLOCK_N = 256
    # 复用加载的偏置张量以减少全局内存访问
    BLOCK_M = 16

    # 计算展平后的行步长
    stride_row = D

    # 定义网格划分策略
    grid = lambda meta: (triton.cdiv(P, meta['BLOCK_M']), triton.cdiv(D, meta['BLOCK_N']))

    # 调用Triton内核执行广播操作
    _broadcast_v1_kernel[grid](
        x_c, bias_c, out,
        P, D,
        stride_row, stride_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out
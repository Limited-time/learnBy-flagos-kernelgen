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
        bias_ptr (tl.tensor): 指向低维偏置张量的指针，将在最后一维进行广播。
        out_ptr (tl.tensor): 指向输出张量的指针，形状与输入张量相同。
        P (int): 展平所有前导维度后的行数：P = prod(x.shape[:-1])。
        D (int): 最后一维的大小（必须等于bias.numel()）。
        stride_x_row (int): 输入张量中行之间的步长（元素数）——保留供未来扩展使用。
        stride_o_row (int): 输出张量中行之间的步长（元素数）。
        BLOCK_M (tl.constexpr): 每个程序处理的行数块大小。
        BLOCK_N (tl.constexpr): 每个程序处理的列数块大小。

    返回:
        无显式返回值；结果通过out_ptr写入输出张量。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rows = row_ids < P
    mask_cols = col_ids < D
    mask = mask_rows[:, None] & mask_cols[None, :]

    # 提示：确保列偏移是64个元素的倍数（对于fp32为256字节），以获得更好的内存对齐。
    tl.multiple_of(col_ids, 64)

    # 每个程序加载一次偏置块，并在BLOCK_M行之间复用。
    bias_tile = tl.load(bias_ptr + col_ids, mask=mask_cols, other=0.0).to(tl.float32)

    # 计算展平布局(P, D)中每行的基础偏移。
    out_offsets = row_ids[:, None] * stride_o_row + col_ids[None, :]

    # 将偏置沿BLOCK_M行广播。
    val_tile = tl.broadcast_to(bias_tile[None, :], (BLOCK_M, BLOCK_N))

    # 写回输出；Triton会在需要时将其转换为目标指针的数据类型。
    tl.store(out_ptr + out_offsets, val_tile, mask=mask)


# 在定义同名Python包装器之前保存Triton内核句柄。
_broadcast_v1_kernel = broadcast_v1


def broadcast_v1(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Python包装器函数，调用Triton内核执行广播操作。

    参数:
        x (torch.Tensor): 高维输入张量，必须位于CUDA设备上。
        bias (torch.Tensor): 低维偏置张量，必须位于CUDA设备上。

    返回:
        torch.Tensor: 广播后的输出张量，形状与输入张量相同。
    """
    # 确保输入为CUDA张量并具有连续性以支持ND格式。
    assert x.is_npu and bias.is_npu, "x and bias must be CUDA tensors"
    x_c = x.contiguous()
    bias_c = bias.contiguous()
    assert x_c.dim() >= 1, "x must have at least 1 dimension"
    D = bias_c.numel()
    assert x_c.shape[-1] == D, "bias last dimension must match x's last dimension"
    P = x_c.numel() // D

    # 分配与输入张量数据类型和形状匹配的输出张量。
    out = torch.empty_like(x_c)

    # 选择块大小。BLOCK_N是64的倍数（fp32为256字节对齐）。
    BLOCK_N = 256
    # 复用已加载的偏置以减少全局内存访问流量。
    BLOCK_M = 16

    # 对于连续的ND格式，展平行步长等于最后一维的大小。
    stride_row = D

    grid = lambda meta: (triton.cdiv(P, meta['BLOCK_M']), triton.cdiv(D, meta['BLOCK_N']))

    _broadcast_v1_kernel[grid](
        x_c, bias_c, out,
        P, D,
        stride_row, stride_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out
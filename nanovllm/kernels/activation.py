import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    X_ptr,          # 输入 [M, 2D]，前 D 列是 gate，后 D 列是 up
    Y_ptr,          # 输出 [M, D]
    stride_xm,
    stride_ym,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # gate 和 up 在同一行内，一次 kernel 同时读取，避免 chunk 产生的两次独立遍历
    gate = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X_ptr + row * stride_xm + D + cols, mask=mask, other=0.0).to(tl.float32)

    # silu(gate) * up 全程在寄存器完成，中间张量 silu(gate) 从不落 HBM
    silu = gate * tl.sigmoid(gate)
    y = silu * up

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x = x.contiguous().view(-1, orig_shape[-1])
    M, two_d = x.shape
    assert two_d % 2 == 0
    D = two_d // 2
    y = torch.empty((M, D), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 1024
    grid = (M, triton.cdiv(D, BLOCK_SIZE))
    _silu_and_mul_kernel[grid](
        x, y, x.stride(0), y.stride(0),
        D=D, BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
    )
    return y.view(*orig_shape[:-1], D)
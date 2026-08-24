import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    X_ptr,  # 输入 [M, N]
    W_ptr,  # 权重 [N]
    Y_ptr,  # 输出 [M, N]
    stride_m,   # 行步长 (以元素为单位)
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """无 residual 版本：y = x * rsqrt(mean(x^2) + eps) * w"""
    row = tl.program_id(0)
    cols = tl.arange(0,BLOCK_SIZE)
    mask = cols < N

    # 一次性把整行读进SRAM，后面所有计算不再读HBM
    x = tl.load(X_ptr + row * stride_m + cols,mask=mask,other=0.0).to(tl.float32)

    var = tl.sum(x*x,axis=0)/N
    rstd = 1.0 / tl.rsqrt(var + EPS)

    w = tl.load(X_ptr+row, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y_ptr + row * stride_m + cols, y ,mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    X_ptr,          # 输入 [M, N]，同时也是输出（原地）
    R_ptr,          # residual [M, N]，原地更新为 x + residual
    W_ptr,          # 权重 [N]
    stride_m,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """融合版本：
        residual = x + residual          (原地写回 R)
        x        = rmsnorm(residual) * w (原地写回 X)
    整个过程 HBM 只读 2 次、写 2 次。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)

    # --- 融合点 1：加法结果留在寄存器，直接参与后续归约 ---
    x = x + r
    # residual 需要给下一层用，这里必须落盘一次（无法避免）
    tl.store(R_ptr + row * stride_m + cols, x, mask=mask)

    # --- 融合点 2：平方和归约不再需要额外的 pass ---
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(X_ptr + row * stride_m + cols, y, mask=mask)
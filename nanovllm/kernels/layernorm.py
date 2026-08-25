import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    X_ptr,          # 输入，M 行 × N 列，行的起始偏移由两级 stride 算出
    W_ptr,          # 权重 [N]
    Y_ptr,          # 输出 [M, N]，连续
    stride_xo,      # 外层步长（以元素为单位）
    stride_xi,      # 行内步长；H == 1 时用不到
    H: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """无 residual 版本：y = x * rsqrt(mean(x^2) + eps) * w"""
    row = tl.program_id(0)
    # H == 1 时下面两步会被编译期折掉；H > 1 对应 q_norm / k_norm 那种
    # [num_tokens, num_heads, head_dim] 的非连续视图，行首偏移要分两级算
    x_off = (row // H) * stride_xo + (row % H) * stride_xi

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 一次性把整行读进 SRAM/寄存器，后面所有计算都不再碰 HBM
    x = tl.load(X_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y_ptr + row * N + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    X_ptr,          # 输入，同时也是输出（原地）
    R_ptr,          # residual，原地更新为 x + residual
    W_ptr,          # 权重 [N]
    stride_xo,
    stride_xi,
    stride_ro,
    stride_ri,
    H: tl.constexpr,
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
    outer = row // H
    inner = row % H
    x_off = outer * stride_xo + inner * stride_xi
    r_off = outer * stride_ro + inner * stride_ri

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + r_off + cols, mask=mask, other=0.0).to(tl.float32)

    # --- 融合点 1：加法结果留在寄存器，直接参与后续归约 ---
    x = x + r
    # residual 需要给下一层用，这里必须落盘一次（无法避免）
    tl.store(R_ptr + r_off + cols, x.to(R_ptr.dtype.element_ty), mask=mask)

    # --- 融合点 2：平方和归约不再需要额外的 pass ---
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(X_ptr + x_off + cols, y.to(X_ptr.dtype.element_ty), mask=mask)


def _launch_config(N: int):
    BLOCK_SIZE = triton.next_power_of_2(N)
    # 经验规则：行越长，用越多 warp 来并行归约。
    # 短行（head_dim=128）只给 1 个 warp——32 线程各处理 4 个元素刚好能向量化，
    # 给到 4 个 warp 反而是每线程 1 个元素 + 多一层跨 warp 归约。
    num_warps = min(max(BLOCK_SIZE // 256, 1), 16)
    return BLOCK_SIZE, num_warps


def _as_rows(x: torch.Tensor):
    """把输入看成 M 行 × N 列，返回 (M, N, H, stride_outer, stride_inner)。

    行首偏移 = (row // H) * stride_outer + (row % H) * stride_inner。

    这里刻意不做 contiguous：q_norm / k_norm 的输入是 qkv.split() 切出来的非连续
    视图（[T, H, D]，行内 stride=D 但跨 token 要跳过 kv 的部分），强行 contiguous
    会为每层的 q 和 k 各多 materialize 一份拷贝。
    """
    assert x.stride(-1) == 1, "最后一维必须连续"
    N = x.shape[-1]
    if x.dim() == 2:
        return x.shape[0], N, 1, x.stride(0), 0
    if x.dim() == 3:
        T, H, _ = x.shape
        if x.stride(0) == H * N and x.stride(1) == N:   # 本来就连续，退化成一级 stride
            return T * H, N, 1, N, 0
        return T * H, N, H, x.stride(0), x.stride(1)
    raise AssertionError(f"只支持 2-D / 3-D 输入，收到 {x.dim()}-D")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """按最后一维归一化，支持 qkv.split() 出来的非连续视图。"""
    orig_shape = x.shape
    M, N, H, stride_o, stride_i = _as_rows(x)
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BLOCK_SIZE, num_warps = _launch_config(N)
    _rmsnorm_kernel[(M,)](
        x, weight, y, stride_o, stride_i,
        H=H, N=N, EPS=eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return y.view(orig_shape)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """原地版本：x 和 residual 都被就地修改，不分配任何新显存。"""
    M, N, H, x_stride_o, x_stride_i = _as_rows(x)
    M_r, N_r, H_r, r_stride_o, r_stride_i = _as_rows(residual)
    assert (M, N, H) == (M_r, N_r, H_r), "x 和 residual 的形状必须一致"
    BLOCK_SIZE, num_warps = _launch_config(N)
    _fused_add_rmsnorm_kernel[(M,)](
        x, residual, weight,
        x_stride_o, x_stride_i, r_stride_o, r_stride_i,
        H=H, N=N, EPS=eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return x, residual

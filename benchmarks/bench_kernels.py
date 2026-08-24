import torch
import torch.nn.functional as F

from nanovllm.kernels.layernorm import fused_add_rmsnorm, rmsnorm
from nanovllm.kernels.activation import silu_and_mul
from bench_peak_bw import bench_rotating

PEAK_BW = 222.0  # ← 改成 4.1 实测出来的值

HIDDEN = 1024
INTER = 3072
EPS = 1e-6
DTYPE = torch.bfloat16


# ---------------- 基线 ----------------
def eager_add_rmsnorm(x, residual, w):
    xf = x.float().add_(residual.float())
    residual = xf.to(DTYPE)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf.mul_(torch.rsqrt(var + EPS))
    return xf.to(DTYPE).mul_(w), residual


compiled_add_rmsnorm = torch.compile(eager_add_rmsnorm)


def eager_silu_mul(x):
    a, b = x.chunk(2, -1)
    return F.silu(a) * b


compiled_silu_mul = torch.compile(eager_silu_mul)


def report(name, ms, useful_bytes, baseline_ms=None):
    gbps = useful_bytes / (ms * 1e-3) / 1e9
    util = gbps / PEAK_BW * 100
    speedup = f"{baseline_ms / ms:6.2f}x" if baseline_ms else "   —   "
    print(f"{name:<32} {ms*1000:8.1f} us  {gbps:7.1f} GB/s  {util:5.1f}%  {speedup}")


# ---------------- 输入池 ----------------
# fused_add_rmsnorm 是原地算子，如果在计时区里 clone 输入，clone 本身的
# 读+写会和 kernel 的访存量同一个量级，测出来的就不是 kernel 的延迟了。
# 改成预先分配一批输入轮换使用：clone 的开销挪到计时区外，且顺带避免
# 小 shape 反复命中 L2。
L2_BYTES = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 32 << 20)


def pool_len(bytes_per_iter, lo=2, hi=64):
    """池子总字节数取到 4×L2 以上（保证每次迭代都得回 HBM 取数），
    同时不超过空闲显存的 1/4，最后夹在 [lo, hi] 之间。"""
    free, _ = torch.cuda.mem_get_info()
    cap = min(4 * L2_BYTES, free // 4)
    return max(lo, min(hi, cap // max(bytes_per_iter, 1)))


def make_norm_pool(M, w):
    n = pool_len(2 * M * HIDDEN * 2)  # 每次迭代摸 x + residual
    return [
        (
            torch.randn(M, HIDDEN, dtype=DTYPE, device="cuda"),
            torch.randn(M, HIDDEN, dtype=DTYPE, device="cuda"),
            w,
        )
        for _ in range(n)
    ]


def refill_norm_pool_(pool):
    """原地算子会把池子里的 x / residual 写坏，跑下一个变体前重新填一遍，
    让三个实现都从同样干净的输入出发（w 是权重，不动）。"""
    for x, r, _ in pool:
        x.normal_()
        r.normal_()


def make_silu_pool(M):
    n = pool_len(M * 2 * INTER * 2)
    return [(torch.randn(M, 2 * INTER, dtype=DTYPE, device="cuda"),) for _ in range(n)]


if __name__ == "__main__":
    print(f"{'kernel':<32} {'latency':>11}  {'eff. BW':>11}  {'util':>6}  {'speedup':>8}")
    w = torch.randn(HIDDEN, dtype=DTYPE, device="cuda")

    for M in [512, 2048, 8192, 32768]:
        print(f"\n--- num_tokens = {M} ---")

        # === add_rmsnorm ===
        # useful bytes: 读 x + 读 residual + 写 x + 写 residual = 4 * M * N * 2
        ub = 4 * M * HIDDEN * 2
        pool = make_norm_pool(M, w)

        t_eager = bench_rotating(eager_add_rmsnorm, pool)
        report("add_rmsnorm / eager", t_eager, ub)

        refill_norm_pool_(pool)
        t_comp = bench_rotating(compiled_add_rmsnorm, pool)
        report("add_rmsnorm / torch.compile", t_comp, ub, t_eager)

        refill_norm_pool_(pool)
        t_triton = bench_rotating(lambda x, r, w: fused_add_rmsnorm(x, r, w, EPS), pool)
        report("add_rmsnorm / triton", t_triton, ub, t_eager)

        del pool  # 让下一个池子复用这块显存

        # === silu_and_mul ===
        # useful bytes: 读 2D + 写 D = 3 * M * D * 2
        ub2 = 3 * M * INTER * 2
        pool2 = make_silu_pool(M)

        t_eager2 = bench_rotating(eager_silu_mul, pool2)
        report("silu_and_mul / eager", t_eager2, ub2)

        t_comp2 = bench_rotating(compiled_silu_mul, pool2)
        report("silu_and_mul / torch.compile", t_comp2, ub2, t_eager2)

        t_triton2 = bench_rotating(silu_and_mul, pool2)
        report("silu_and_mul / triton", t_triton2, ub2, t_eager2)

        del pool2
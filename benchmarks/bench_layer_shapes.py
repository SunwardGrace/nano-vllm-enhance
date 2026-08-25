"""按 Qwen3-0.6B 模型里真实的调用点和张量形状测量，用来定位端到端收益来自哪一处。

单算子 benchmark（bench_kernels.py）测的是规整的 2-D 大矩阵，
但模型里 RMSNorm 有三个形态完全不同的调用点：

  input_layernorm / post_attention_layernorm : fused_add_rmsnorm, [bs, 1024]
  q_norm                                     : rmsnorm, [bs, 16, 128]，且输入是
                                               qkv.split() 出来的非连续视图
  k_norm                                     : rmsnorm, [bs,  8, 128]，同上
  mlp.act_fn                                 : silu_and_mul, [bs, 6144]

用法： python bench_layer_shapes.py
"""
import torch

from nanovllm.kernels.layernorm import fused_add_rmsnorm, rmsnorm
from nanovllm.kernels.activation import silu_and_mul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.activation import SiluAndMul
from bench_peak_bw import bench_rotating

# Qwen3-0.6B
HIDDEN = 1024
INTER = 3072
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
N_LAYERS = 28
Q_SIZE = N_HEADS * HEAD_DIM
KV_SIZE = N_KV_HEADS * HEAD_DIM
EPS = 1e-6
DTYPE = torch.bfloat16

POOL = 8


def new(*shape):
    return torch.randn(*shape, dtype=DTYPE, device="cuda")


def main():
    torch.set_default_dtype(DTYPE)
    torch.set_default_device("cuda")
    norm_h = RMSNorm(HIDDEN, EPS)
    norm_d = RMSNorm(HEAD_DIM, EPS)
    act = SiluAndMul()

    # 先按调用点建好全部输入池
    pools = {}          # name -> (shape_str, pool, calls_per_step)
    for bs, tag in [(64, "decode bs=64"), (8192, "prefill 8192tok")]:
        pools[f"fused_add_rmsnorm ({tag})"] = (
            f"[{bs}, {HIDDEN}]",
            [(new(bs, HIDDEN), new(bs, HIDDEN)) for _ in range(POOL)],
            2 * N_LAYERS,
        )
        # q_norm / k_norm 的输入是 qkv.split() 出来的非连续视图，复现这一点
        pool_q, pool_k = [], []
        for _ in range(POOL):
            qkv = new(bs, Q_SIZE + 2 * KV_SIZE)
            q, k, _ = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
            pool_q.append((q.view(-1, N_HEADS, HEAD_DIM),))
            pool_k.append((k.view(-1, N_KV_HEADS, HEAD_DIM),))
        pools[f"q_norm ({tag})"] = (f"[{bs}, {N_HEADS}, {HEAD_DIM}]", pool_q, N_LAYERS)
        pools[f"k_norm ({tag})"] = (f"[{bs}, {N_KV_HEADS}, {HEAD_DIM}]", pool_k, N_LAYERS)
        pools[f"silu_and_mul ({tag})"] = (
            f"[{bs}, {2 * INTER}]",
            [(new(bs, 2 * INTER),) for _ in range(POOL)],
            N_LAYERS,
        )

    def impls(name):
        if name.startswith("fused_add_rmsnorm"):
            return (lambda x, r: norm_h.add_rms_forward_torch(x, r),
                    lambda x, r: fused_add_rmsnorm(x, r, norm_h.weight, EPS))
        if name.startswith(("q_norm", "k_norm")):
            return (lambda q: norm_d.rms_forward_torch(q),
                    lambda q: rmsnorm(q, norm_d.weight, EPS))
        return (lambda g: act.forward_torch(g), silu_and_mul)

    # 关键：先把所有形状都跑一遍。同一个 @torch.compile 方法会被多种形状调用
    # （q_norm/k_norm、decode/prefill），dynamo 遇到第二个形状会重编译成动态形状版本。
    # 不预热完就计时的话，先测的那个形状会白占静态形状的便宜。
    for name, (_, pool, _) in pools.items():
        f_torch, f_triton = impls(name)
        for _ in range(3):
            f_torch(*pool[0])
            f_triton(*pool[0])
    torch.cuda.synchronize()

    print(f"{'调用点':<34} {'shape':>18} {'torch.compile':>14} {'triton':>10} {'speedup':>9}")
    rows = []
    for name, (shape, pool, calls) in pools.items():
        f_torch, f_triton = impls(name)
        t_torch = bench_rotating(f_torch, pool)
        if name.startswith("fused_add_rmsnorm"):     # 原地算子会写坏输入
            for x, r in pool:
                x.normal_(), r.normal_()
        t_triton = bench_rotating(f_triton, pool)
        rows.append((name, shape, t_torch, t_triton, calls))

    for name, shape, tt, tr, _ in rows:
        print(f"{name:<34} {shape:>18} {tt*1e3:11.1f} us {tr*1e3:7.1f} us {tt/tr:8.2f}x")

    # 把每个调用点的收益按"每个 step 调用次数"加权，估算单个 step 能省下多少
    for tag in ["decode bs=64", "prefill 8192tok"]:
        total = sum(tt * n for name, _, tt, _, n in rows if tag in name)
        saved = sum((tt - tr) * n for name, _, tt, tr, n in rows if tag in name)
        print(f"\n{tag}：这些调用点合计 torch.compile {total*1e3:.0f} us → "
              f"triton {(total-saved)*1e3:.0f} us，省下 {saved*1e3:.0f} us")


if __name__ == "__main__":
    main()

import torch

def bench(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep  # ms


def bench_rotating(fn, args_pool, warmup=25, rep=100):
    """轮换输入版本：每次迭代喂给 fn 一组不同的、预先分配好的参数。

    两个作用：
      1. 原地算子不需要在计时区里 clone 输入（clone 本身就是一次读+一次写，
         对 memory-bound kernel 来说等于把测出来的延迟翻倍）；
      2. 池子够大时每次迭代摸的是不同的显存，避免小 shape 整个驻留 L2
         而测出虚高的带宽。
    """
    n = len(args_pool)
    for i in range(warmup):
        fn(*args_pool[i % n])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(rep):
        fn(*args_pool[i % n])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep  # ms


if __name__ == "__main__":
    # 纯拷贝：读 1 次 + 写 1 次，是访存性能的天花板
    n = 256 * 1024 * 1024 // 2   # 256 MB 的 bf16
    src = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    ms = bench(lambda: dst.copy_(src))
    gbps = 2 * src.numel() * 2 / (ms * 1e-3) / 1e9
    print(f"Achievable peak bandwidth: {gbps:.1f} GB/s  ({ms:.3f} ms)")
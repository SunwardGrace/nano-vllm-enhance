"""端到端 benchmark：Triton 融合算子 vs torch.compile 基线。

用法（USE_TRITON 在 import 时读取，所以必须用两个独立进程对比）：
    NANOVLLM_TRITON=0 python bench_e2e.py --eager     # 基线 / eager
    NANOVLLM_TRITON=1 python bench_e2e.py --eager     # 优化后 / eager
    NANOVLLM_TRITON=0 python bench_e2e.py            # 基线 / CUDA Graph
    NANOVLLM_TRITON=1 python bench_e2e.py            # 优化后 / CUDA Graph
"""
import argparse
import json
import os
import time
from random import randint, seed
from statistics import median

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.kernels import USE_TRITON

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
NUM_SEQS = 64
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 512
MAX_MODEL_LEN = 2048


def percentile(sorted_vals, q):
    """线性插值分位数（避免为了 numpy 再引一个依赖）。"""
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def make_workload():
    seed(0)  # 两次运行的输入长度 / 输出长度完全一致
    prompts = [[randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(NUM_SEQS)]
    params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN))
              for _ in range(NUM_SEQS)]
    return prompts, params


def instrument(llm):
    """包住 step()，逐步记录 (num_tokens, latency)。

    num_tokens > 0 是 prefill 步，< 0 是 decode 步（见 LLMEngine.step）。
    step() 内部本来就有 .tolist() 造成的同步，这里补的 synchronize 几乎不额外收费。
    """
    records = []
    orig_step = llm.step

    def step():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, num_tokens = orig_step()
        torch.cuda.synchronize()
        records.append((num_tokens, time.perf_counter() - t0))
        return out, num_tokens

    llm.step = step
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", action="store_true", help="关掉 CUDA Graph")
    ap.add_argument("--profile", type=int, default=0, help="额外用 profiler 统计 N 个 decode step 的 kernel 数")
    args = ap.parse_args()

    backend = "triton" if USE_TRITON else "torch.compile"
    mode = "eager" if args.eager else "cudagraph"

    llm = LLM(MODEL, enforce_eager=args.eager, max_model_len=MAX_MODEL_LEN)
    num_kvcache_blocks = llm.model_runner.config.num_kvcache_blocks
    kv_bytes = llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()

    # 预热（触发 torch.compile / Triton JIT，这部分不该算进吞吐）
    llm.generate(["Benchmark: "], SamplingParams(max_tokens=8), use_tqdm=False)

    torch.cuda.synchronize()
    mem_after_init = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    prompts, params = make_workload()
    records = instrument(llm)

    t = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - t

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    out_tokens = sum(sp.max_tokens for sp in params)
    prefill = sorted(lat for n, lat in records if n > 0)
    decode = sorted(lat for n, lat in records if n < 0)
    prefill_tokens = sum(n for n, _ in records if n > 0)

    result = dict(
        backend=backend,
        mode=mode,
        num_seqs=NUM_SEQS,
        # ---- 吞吐 ----
        output_tokens=out_tokens,
        elapsed_s=round(elapsed, 3),
        throughput_tok_s=round(out_tokens / elapsed, 2),
        # ---- 步延迟（ms）----
        num_prefill_steps=len(prefill),
        num_decode_steps=len(decode),
        prefill_tokens=prefill_tokens,
        prefill_mean_ms=round(sum(prefill) / len(prefill) * 1e3, 3) if prefill else None,
        decode_mean_ms=round(sum(decode) / len(decode) * 1e3, 3),
        decode_p50_ms=round(median(decode) * 1e3, 3),
        decode_p99_ms=round(percentile(decode, 0.99) * 1e3, 3),
        decode_p999_ms=round(percentile(decode, 0.999) * 1e3, 3),
        decode_max_ms=round(decode[-1] * 1e3, 3),
        # ---- 显存（GB）----
        kv_cache_blocks=num_kvcache_blocks,
        kv_cache_gb=round(kv_bytes / 1e9, 3),
        mem_after_init_gb=round(mem_after_init / 1e9, 3),
        peak_gb=round(peak / 1e9, 3),
        transient_activation_gb=round((peak - mem_after_init) / 1e9, 4),
    )

    if args.profile:
        result.update(profile_kernels(llm, args.profile))

    print("RESULT " + json.dumps(result))


def profile_kernels(llm, num_steps):
    """统计单个 decode step 平均发射多少次 kernel launch。

    WSL2 下 CUPTI 采不到 device 端 kernel 活动（trace 里没有 cat="kernel"），
    所以退一步统计 CPU 侧的 launch API 调用次数：
    torch 算子走 cudaLaunchKernel，Triton 走 driver API 的 cuLaunchKernel，两者都要算。
    只在 eager 模式下有意义（CUDA Graph 整图 replay 只有一次 launch）。
    """
    import collections
    import json as _json
    import tempfile
    from torch.profiler import profile, ProfilerActivity

    prompts, params = make_workload()
    for p, sp in zip(prompts, params):
        sp.max_tokens = num_steps + 4
        llm.add_request(p, sp)
    # prefill 可能占多个 step（chunked prefill），一直走到第一个 decode step 为止
    while True:
        _, n = llm.step()
        if n < 0:
            break

    steps = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        while not llm.is_finished() and steps < num_steps:
            _, n = llm.step()
            assert n < 0, "profile 区间内混进了 prefill step"
            steps += 1
        torch.cuda.synchronize()
    while not llm.is_finished():
        llm.step()

    with tempfile.NamedTemporaryFile("r+", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        trace = _json.load(open(f.name))
    launches = collections.Counter(
        e["name"] for e in trace["traceEvents"]
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "LaunchKernel" in e["name"]
    )
    total = sum(launches.values())
    return dict(
        profiled_decode_steps=steps,
        launches_per_decode_step=round(total / max(steps, 1), 1),
        launch_api_breakdown={k: v for k, v in launches.most_common()},
    )


if __name__ == "__main__":
    main()

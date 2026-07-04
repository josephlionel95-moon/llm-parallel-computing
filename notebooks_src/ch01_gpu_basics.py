# %% [markdown]
# # Chapter 01 — GPU Architecture: Warps, Memory Hierarchy & the Roofline
#
# **Hardware:** best with 1+ GPU (Colab/Kaggle T4). Every cell is CPU-safe: GPU
# cells detect the environment and print a skip message instead of crashing, so
# you can read/run the whole notebook anywhere and re-run on a T4 later.
#
# Central question: *for any op, is the GPU limited by arithmetic or by memory
# traffic?* The tool is the roofline model:
#
# > attainable FLOP/s = min(peak_FLOPs, intensity × bandwidth),
# > intensity I = FLOPs / bytes moved to/from HBM.

# %%
import sys

sys.path.insert(0, "../src")

import torch

from llmdist.profiling.timers import bandwidth_gbps, timeit
from llmdist.utils import env_check

env = env_check.detect()
print(env.banner())
device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")

# %% [markdown]
# ## 1. Ridge points: the one number to know about your GPU
#
# Ridge = peak FLOP/s ÷ HBM bandwidth = "FLOPs you must do per byte fetched to be
# compute-bound". These are *vendor spec sheet* numbers (not measurements):

# %%
SPECS = {
    # name: (peak fp32 TFLOPS, peak fp16/bf16 tensor-core TFLOPS, HBM GB/s)
    "T4":   (8.1, 65.0, 320.0),
    "A100": (19.5, 312.0, 2039.0),
}
print(f"{'GPU':<6} {'fp32 ridge':>12} {'fp16-TC ridge':>15}   (FLOP/byte)")
for name, (p32, p16, bw) in SPECS.items():
    print(f"{name:<6} {p32 * 1e12 / (bw * 1e9):>12.0f} {p16 * 1e12 / (bw * 1e9):>15.0f}")
print("\nAny op doing fewer FLOPs per HBM byte than the ridge is memory-bound.")

# %% [markdown]
# ## 2. Matmul intensity is Θ(n); elementwise is Θ(1)
#
# Square fp32 matmul: FLOPs = 2n³, compulsory bytes = 4·3n² ⇒ **I = n/6**.
# Elementwise add: 1 FLOP per 12 bytes (2 reads + 1 write) ⇒ **I = 1/12**, forever.
# Let's tabulate where sizes land relative to the T4 ridges:

# %%
print(f"{'n':>6} {'I=n/6 (fp32)':>14} {'vs T4 fp32 ridge (25)':>22}")
for n in [64, 256, 1024, 4096]:
    intensity = n / 6
    verdict = "compute-bound" if intensity > 25 else "MEMORY-bound"
    print(f"{n:>6} {intensity:>14.1f} {verdict:>22}")
print(f"{'any':>6} {'0.08':>14} {'MEMORY-bound (x+y)':>22}")

# %% [markdown]
# ## 3. Measure it: memory-bound vs compute-bound
#
# We time a big matmul (high I) and an equal-*size* elementwise add (I=1/12) and
# report both **achieved GFLOP/s** and **achieved GB/s**. Prediction: the add
# achieves a tiny fraction of peak FLOPs but a large fraction of peak bandwidth;
# the matmul is the reverse. On CPU the same code runs (smaller sizes) — the
# qualitative contrast survives because CPUs have rooflines too.

# %%
n = 2048 if env.num_gpus >= 1 else 512
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)

mm = timeit(lambda: a @ b, iters=10, device=device)
mm_flops = 2 * n**3
mm_bytes = 4 * 3 * n * n  # compulsory: read A, B, write C

ew = timeit(lambda: a + b, iters=10, device=device)
ew_flops = n * n
ew_bytes = 4 * 3 * n * n  # read a, b, write out — same bytes as the matmul!

print(f"matmul {n}x{n}: {mm['mean_ms']:8.3f} ms  "
      f"{mm_flops / (mm['mean_ms'] * 1e-3) / 1e9:10.1f} GFLOP/s  "
      f"{bandwidth_gbps(mm_bytes, mm['mean_ms']):8.1f} GB/s")
print(f"add    {n}x{n}: {ew['mean_ms']:8.3f} ms  "
      f"{ew_flops / (ew['mean_ms'] * 1e-3) / 1e9:10.1f} GFLOP/s  "
      f"{bandwidth_gbps(ew_bytes, ew['mean_ms']):8.1f} GB/s")
print("\nSame bytes, ~1000x fewer FLOPs -> the add's time is set by bytes alone.")

# %% [markdown]
# ## 4. Tensor cores: dtype and alignment
#
# Tensor cores execute small matrix multiply-accumulates per clock; cuBLAS uses
# them for fp16/bf16 when M, N, K are multiples of 8 (best: 16+). Two probes:
# fp32 vs fp16 at the same size, and n=2048 vs n=2047 in fp16 (alignment cliff).

# %%
if env.num_gpus >= 1:
    for dtype in [torch.float32, torch.float16]:
        x = torch.randn(2048, 2048, device=device, dtype=dtype)
        s = timeit(lambda: x @ x, iters=10, device=device)
        tf = 2 * 2048**3 / (s["mean_ms"] * 1e-3) / 1e12
        print(f"matmul 2048^3 {str(dtype):<15}: {s['mean_ms']:7.3f} ms = {tf:6.2f} TFLOPS")
    for n_ in [2048, 2047]:
        x = torch.randn(n_, n_, device=device, dtype=torch.float16)
        s = timeit(lambda: x @ x, iters=10, device=device)
        tf = 2 * n_**3 / (s["mean_ms"] * 1e-3) / 1e12
        print(f"matmul fp16 n={n_:<5}          : {s['mean_ms']:7.3f} ms = {tf:6.2f} TFLOPS")
    print("Expect fp16 >> fp32, and n=2047 below n=2048 (no clean tensor-core tiling).")
else:
    print("[skip] no GPU: tensor cores don't exist on CPU. On a T4 expect fp16 to "
          "approach several x the ~8 TFLOPS fp32 peak (HW peak 65 TFLOPS fp16).")

# %% [markdown]
# ## 5. CUDA is asynchronous: the `time.time()` lie
#
# `a @ b` *enqueues* a kernel and returns in microseconds. Timing without a
# synchronize measures the launch, not the math. `llmdist.profiling.timers.timeit`
# uses CUDA events; here we expose the raw discrepancy.

# %%
import time

if env.num_gpus >= 1:
    big = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    c = big @ big                      # enqueue only!
    t_lie = (time.perf_counter() - t0) * 1e3

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = big @ big
    torch.cuda.synchronize()           # wait for the kernel to actually finish
    t_truth = (time.perf_counter() - t0) * 1e3

    print(f"unsynchronized 'time'  : {t_lie:8.3f} ms   <- launch overhead only")
    print(f"synchronized time      : {t_truth:8.3f} ms   <- the real kernel")
    print(f"the lie was {t_truth / max(t_lie, 1e-6):.0f}x too optimistic")
else:
    print("[skip] no GPU: CPU ops are synchronous, so time.time() is honest here.")
    print("       On GPU the unsynchronized number is ~launch cost (microseconds).")

# %% [markdown]
# ## 6. Kernel launch overhead
#
# Each launch costs ~3–10 µs of CPU/driver work. A step with thousands of small
# kernels pays that floor before any math. Measure by launching N trivial kernels:

# %%
if env.num_gpus >= 1:
    tiny = torch.randn(8, device=device)
    N = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        tiny.add_(1.0)
    torch.cuda.synchronize()
    per_launch_us = (time.perf_counter() - t0) / N * 1e6
    print(f"{N} tiny kernels -> {per_launch_us:.1f} us per launch")
    print("A model launching 2000 kernels/step pays ~"
          f"{2000 * per_launch_us / 1e3:.1f} ms/step of pure overhead.")
else:
    print("[skip] no GPU. Typical measured values are ~3-10 us per launch; "
          "fusion (torch.compile) and CUDA Graphs exist to amortize this.")

# %% [markdown]
# ## 7. PCIe: pageable vs pinned host memory
#
# Host↔device copies ride PCIe (~16 GB/s theoretical on the T4's PCIe 3.0 x16).
# Pageable memory needs a driver-side staging copy; pinned (page-locked) memory
# can be DMA'd directly *and asynchronously*. On Kaggle's 2x T4 the GPUs also talk
# to each other over this same bus — remember these numbers in ch07/ch08.

# %%
if env.num_gpus >= 1:
    mb = 256
    n_el = mb * 2**20 // 4
    host_pageable = torch.randn(n_el)
    host_pinned = torch.randn(n_el).pin_memory()

    for name, src in [("pageable", host_pageable), ("pinned  ", host_pinned)]:
        s = timeit(lambda: src.to(device, non_blocking=False), iters=10, device=device)
        print(f"H2D {name} {mb} MiB: {s['mean_ms']:7.2f} ms = "
              f"{bandwidth_gbps(src.numel() * 4, s['mean_ms']):6.2f} GB/s")
    print("Expect pinned ~2-4x pageable; both far below the 320 GB/s of HBM itself.")
else:
    print("[skip] no GPU: nothing to copy across PCIe.")
    print("       Expected on T4: pageable ~3-6 GB/s, pinned ~10-13 GB/s.")

# %% [markdown]
# ## Exercises
#
# 1. **(Easy)** Compute the intensity of fp16 `x * 2` and of an fp16 n=8192 matmul.
#    Which side of the T4 fp16 ridge (~200) does each fall on?
# 2. **(Medium)** In cell 3, the add and the matmul move the *same bytes*. Predict
#    the ratio of their runtimes on a T4 from the rooflines, then check with
#    `experiments/ch01/roofline_ops.py`.
# 3. **(Medium)** Why must `timeit` warm up before measuring? List three distinct
#    one-time costs the first call pays.
# 4. **(Hard)** Extend cell 7 to overlap an H2D copy (pinned, `non_blocking=True`,
#    its own `torch.cuda.Stream`) with a matmul on the default stream. Show total
#    ≈ max(copy, compute), not the sum.
#
# <details><summary>Solutions</summary>
#
# 1. `x*2` in fp16: 1 FLOP / 4 bytes = 0.25 — memory-bound (200/0.25 = 800x under
#    the ridge). Matmul: I = n/(3·2) ≈ 1365 — compute-bound.
# 2. Matmul time ≈ 2n³/P; add time ≈ 12n²/B. Ratio = (2n³/P)/(12n²/B) = nB/(6P).
#    For n=2048 fp32 on T4: 2048·320e9/(6·8.1e12) ≈ 13x — the matmul is ~13x
#    slower despite equal bytes, because it does ~1000x the FLOPs.
# 3. CUDA context creation, cuBLAS handle/workspace init + heuristic/autotune
#    kernel selection, caching-allocator pool growth (first allocations call
#    cudaMalloc). Any of these can dominate a first call by 10-1000x.
# 4. Key pieces: `s = torch.cuda.Stream(); with torch.cuda.stream(s): dst.copy_(pinned_src, non_blocking=True)`
#    while the matmul runs on the default stream; synchronize both; compare
#    against sequential execution. Pageable memory breaks the overlap because the
#    copy is no longer async. This is DDP's overlap trick in miniature (ch09).
#
# </details>
#
# ## Summary
#
# - Ridge point P/B decides everything: T4 fp32 ≈ 25, T4 fp16-TC ≈ 200 FLOP/byte.
# - Matmul: I = n/6 → compute-bound when big. Elementwise: I ≤ ~0.25 → always
#   memory-bound; its time is bytes ÷ bandwidth, FLOPs are irrelevant.
# - Tensor cores need fp16/bf16 + aligned shapes; ragged dims fall off a cliff.
# - CUDA is async: only CUDA events / synchronized clocks tell the truth.
# - Launches cost microseconds; PCIe is ~20-100x slower than HBM — the memory
#   hierarchy continues *between* GPUs, which is where this course is headed.

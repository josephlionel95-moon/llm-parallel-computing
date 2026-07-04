# Chapter 01 — GPU Architecture: SMs, Warps, Memory Hierarchy & the Roofline

> **Difficulty:** 🟡 Medium · **Study time:** 4–6 h · **Requires:** ch00
> **Notebook:** notebooks/ch01_gpu_basics.ipynb · **Experiments:** experiments/ch01/

Distributed training exists because a single GPU has finite FLOP/s and finite bytes.
To reason about *when* a second GPU helps, you must first be able to answer, for any
single operation: is it limited by arithmetic or by memory traffic, and how far from
the hardware's peak is it? This chapter builds that skill — the roofline model — from
the CUDA execution model up.

## Learning objectives

After this chapter you will be able to:

- Describe how a CUDA kernel launch maps onto grids, blocks, warps, and SMs, and define occupancy.
- Draw the GPU memory hierarchy (registers → shared/L1 → L2 → HBM) with approximate sizes, bandwidths, and latencies for T4 and A100.
- Explain what tensor cores compute and why fp16/bf16 matmuls with dimensions that are multiples of 8/16 hit the fast path.
- Derive arithmetic intensity for matmul ($\Theta(n)$) and elementwise ops ($\Theta(1)$), and place both on a roofline plot.
- Predict whether an op is memory-bound or compute-bound *before* running it, then verify by measurement.
- Explain why `time.time()` around a kernel launch lies, and time correctly with CUDA events.
- Estimate kernel-launch overhead and PCIe transfer bandwidth (pageable vs pinned), and explain why both matter for distributed training.

## Intuition

A CPU is a few brilliant workers with private offices; a GPU is a stadium of tens of
thousands of identical workers who can only act in synchronized rows. The stadium
wins whenever the job is "do the same arithmetic to millions of numbers" — which is
what a neural network is.

But the stadium has a supply problem. Its arithmetic capacity is enormous (T4:
~65 × 10¹² fp16 multiply-adds/s) while its connection to memory is merely large
(T4: 320 × 10⁹ bytes/s). Divide the two: the machine can perform roughly **200
floating-point operations in the time it takes to fetch one 1 float** from HBM at peak
bandwidth. Any computation that does *less* than ~200 ops per fetched value leaves
arithmetic units idle — it is **memory-bound**. This single ratio explains:

- why matmuls are fast (they reuse each fetched value ~n times),
- why everything else in a transformer (LayerNorm, GELU, softmax, residual adds) is memory-bound,
- why "kernel fusion" is the main single-GPU optimization (fewer trips to HBM),
- and, later, why *communication* — an even slower memory — dominates distributed training.

Chapter 03 applies this lens to a transformer; chapters 07+ apply it to networks
between GPUs. Same model, different rooflines.

## Theory

### The execution model: kernels → grids → blocks → warps

A **kernel** is a function executed by many threads at once. You launch it over a
**grid** of **thread blocks**; each block holds up to 1024 threads. The hardware
executes threads in groups of 32 called **warps** — one instruction pointer per
warp, all 32 lanes executing the same instruction (SIMT). Divergent branches
within a warp serialize both paths.

```
 kernel launch:  grid of blocks             one SM (streaming multiprocessor)
 ┌────┬────┬────┬────┐                     ┌────────────────────────────────┐
 │ B0 │ B1 │ B2 │ B3 │   blocks are       │ warp schedulers (4)            │
 ├────┼────┼────┼────┤   assigned to      │ ┌──────────────────────────┐   │
 │ B4 │ B5 │ B6 │ B7 │ ───► SMs, many     │ │ resident warps (up to 32/│   │
 └────┴────┴────┴────┘   blocks per SM    │ │ 64): W0 W1 W2 ... ready? │   │
                                          │ └──────────────────────────┘   │
 block = up to 1024 threads               │ CUDA cores | tensor cores      │
 warp  = 32 threads, one PC               │ registers  | shared mem / L1   │
                                          └────────────────────────────────┘
```

An **SM** (streaming multiprocessor) is the unit of replication: a T4 has 40, an
A100 has 108. Each SM keeps *many* warps resident simultaneously and its schedulers
issue from whichever warps are ready. This is how GPUs hide memory latency: when
warp W0 stalls on a 400-cycle HBM load, the scheduler issues W1, W2, ... — latency
is not eliminated, it is *covered* by parallelism.

**Occupancy** = resident warps / maximum resident warps per SM. Occupancy is limited
by whichever per-SM resource a block exhausts first: registers (64 K × 32-bit per
SM), shared memory, or the warp/block slots themselves. Low occupancy means too few
warps to hide latency; but 100 % occupancy is not the goal — a kernel with high data
reuse in registers may run best at 25 % occupancy (see Volkov's classic analysis).

**Kernel launch overhead.** Launching a kernel costs roughly 3–10 µs of CPU-side
driver work plus queueing (measure it: `experiments/ch01/async_and_pcie.py`). A
training step that launches 2,000 tiny kernels pays ~10–20 ms *before any math
happens* — this is why CUDA Graphs and `torch.compile` fusion exist, and why
small-batch inference is launch-bound.

### The memory hierarchy

Numbers below are vendor specifications or architecture-documented values for the
two GPUs this course targets (Turing TU104-based T4; GA100-based A100-SXM 80 GB).
Latencies are approximate, from published microbenchmarking studies (Jia et al.
2018/2019 for Turing/Volta; Luo et al. 2024 for Ampere/Hopper) — treat them as
order-of-magnitude, not contractual.

| level | scope | T4 size | A100 size | bandwidth (approx) | latency (approx) |
|---|---|---|---|---|---|
| registers | per thread | 256 KB/SM (64 K×32-bit) | 256 KB/SM | ~TB/s-class, per-SM | ~1 cycle (0.7 ns) |
| shared mem / L1 | per block / SM | 96 KB/SM combined | 192 KB/SM (up to 164 KB shared) | ~12 TB/s aggregate (T4-class) / ~19 TB/s (A100) | ~20–30 cycles |
| L2 cache | whole GPU | 4 MB | 40 MB | ~1–2 TB/s (T4) / ~4–7 TB/s (A100) | ~190–260 cycles |
| HBM / GDDR (global) | whole GPU | 16 GB GDDR6 | 80 GB HBM2e | 320 GB/s | 2039 GB/s | ~400–800 cycles |
| host RAM over PCIe | system | — | — | ~16 GB/s (PCIe 3.0 ×16, T4) / ~32 GB/s (PCIe 4.0, A100) | ~µs |
| peer GPU (NVLink) | system | n/a on T4 | 600 GB/s | — | ~µs |

Two readings of this table matter for the whole course:

1. **Each level down costs ~an order of magnitude in bandwidth.** Fast kernels
   stage data upward (HBM → shared → registers) and maximize reuse at the top.
   That is literally what a tiled matmul kernel is.
2. **The hierarchy continues past the GPU.** PCIe is ~20× slower than HBM on a T4;
   the network between nodes is slower still. Distributed training (ch06+) is
   "programming the memory hierarchy" at the next level out. The α–β cost model we
   use for collectives in ch07 is this table's latency/bandwidth columns, renamed.

### Tensor cores

CUDA cores execute one fused multiply-add (FMA) per clock per core, on scalars.
**Tensor cores** execute a small *matrix* FMA per clock — on Turing, effectively
D = A·B + C for 4×4×4 fp16 tiles (exposed to software as 16×16×16 `wmma`
fragments); Ampere widens the tile and adds bf16/tf32. That is why the fp16
peak (T4: ~65 TFLOPS) is ~8× the fp32 CUDA-core peak (~8.1 TFLOPS): same silicon
budget, but the operands' structure is baked into the datapath.

Two practical consequences:

- **dtype:** fp16/bf16 matmuls can use tensor cores; classic fp32 matmuls cannot
  (A100 adds tf32, which tensor-cores fp32 *inputs* at reduced mantissa —
  19 → 156 TFLOPS). This is half the reason mixed precision (ch20) exists; the
  other half is bandwidth (half the bytes).
- **shapes:** cuBLAS dispatches tensor-core kernels when M, N, K are multiples of 8
  (fp16, and ideally 16 for peak; 16 for int8), because the tile machinery loads
  operands in fixed-size fragments. Ragged dimensions fall back to slower kernels
  or pay padding. Hence the folk rule "pad vocab size to a multiple of 64" — check
  it yourself with `experiments/ch01/matmul_tflops_sweep.py` by comparing n=4096
  with n=4095/4097.

### Streams and asynchronous execution

Every CUDA operation is enqueued onto a **stream** (a FIFO work queue) and the CPU
call returns *immediately*. Operations on one stream execute in order; different
streams may overlap. PyTorch uses one default compute stream per device; DDP (ch09)
runs NCCL communication on *separate* streams precisely so gradient AllReduce can
overlap with backward compute.

```
 CPU:    launch K1 ─ launch K2 ─ launch memcpy ─ python continues ...
              │           │            │
 stream0: [   K1   ][    K2    ]       │
 stream1:                        [  memcpy (pinned, async)  ]
                                   time ──►
```

The corollary is the most common profiling bug in existence:

```python
t0 = time.time(); y = a @ b; t1 = time.time()   # measures ~launch time (µs), not the kernel!
```

`a @ b` merely *enqueues* the matmul. Correct timing uses CUDA events recorded on
the stream (`llmdist.profiling.timers.timeit` does this), or a
`torch.cuda.synchronize()` before each `time.perf_counter()` reading.
`experiments/ch01/async_and_pcie.py` demonstrates the lie and the fix side by side.

### PCIe, pinned memory, and why H2D bandwidth matters

Host↔device copies traverse PCIe. Pageable host memory cannot be DMA'd directly
(the OS may move it), so the driver first copies into an internal pinned staging
buffer — you pay an extra memcpy and get well below the bus peak. `pin_memory=True`
allocates page-locked memory the DMA engine can read directly, and *only pinned
memory can be copied asynchronously* (`tensor.to(device, non_blocking=True)`).
Expect on a T4 (PCIe 3.0 ×16, ~15.75 GB/s theoretical): pageable ≈ 3–6 GB/s,
pinned ≈ 10–13 GB/s — measure with `experiments/ch01/async_and_pcie.py`.

This is a preview of a theme: on Kaggle's 2× T4 there is no NVLink, so *inter-GPU
communication also rides PCIe* at these same few GB/s — 100× slower than HBM. Keep
that number in mind when ch08 asks why AllReduce hurts.

## Mathematics

### Arithmetic intensity and the roofline

Define **arithmetic intensity** $I = \dfrac{\text{FLOPs performed}}{\text{bytes moved to/from HBM}}$ (units: FLOP/byte).

A kernel's attainable throughput on a machine with peak compute $P$ (FLOP/s) and
peak memory bandwidth $B$ (byte/s) is bounded by both:

$$\text{FLOP/s} \;\le\; \min\!\big(P,\; I \cdot B\big)$$

Plotted on log–log axes (throughput vs $I$), this is a slanted "roof" ($I\cdot B$)
meeting a flat roof ($P$) at the **ridge point** $I^\* = P/B$:

```
  FLOP/s (log)
    P ┤            ___________________  compute-bound (I > P/B)
      │           /
      │          /   slope = B
      │         /
      │        /  memory-bound (I < P/B)
      │       /
      └──────┴──────────────────────────► I = FLOP/byte (log)
            I* = P/B
```

Ridge points for our hardware (fp32 sizes, HBM bandwidth):

| machine | P | B | ridge $I^\*$ |
|---|---|---|---|
| T4, fp32 CUDA cores | 8.1 TFLOPS | 320 GB/s | ~25 FLOP/B |
| T4, fp16 tensor cores | 65 TFLOPS | 320 GB/s | ~203 FLOP/B |
| A100, fp32 | 19.5 TFLOPS | 2039 GB/s | ~10 FLOP/B |
| A100, bf16 tensor cores | 312 TFLOPS | 2039 GB/s | ~153 FLOP/B |

### Matmul: $I = \Theta(n)$ — derivation

Take $C = A B$ with $A \in \mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$.

**FLOPs.** Each of the $mn$ output elements is a dot product of length $k$: $k$
multiplies + $k{-}1$ adds $\approx 2k$ FLOPs. Total:

$$F = 2\,m\,n\,k.$$

(Memorize $2mnk$; it powers every FLOP estimate in ch03.)

**Bytes (compulsory traffic).** Each input must be read at least once and the
output written once. With $s$ bytes/element:

$$M \ge s\,(mk + kn + mn).$$

**Intensity.** For square $m=n=k$, fp32 ($s=4$):

$$I = \frac{2n^3}{4\cdot 3n^2} = \frac{n}{6} \quad\Longrightarrow\quad I = \Theta(n).$$

So a 64×64 fp32 matmul has $I \approx 10.7$ — *below* the T4 fp32 ridge (25):
memory-bound. A 1024×1024 has $I \approx 171$: compute-bound with a wide margin
(if the kernel actually achieves near-compulsory traffic, which requires tiling —
each fetched element must be reused ~$n$ times via shared memory/registers).
This derivation assumes a perfect cache; real kernels approach it, never reach it.

**Batched/rectangular caution:** the transformer's "matmuls" during *inference
decoding* are matrix–vector products ($m{=}1$): $I = 2nk/(s(nk+k+n)) \approx 2/s$
— hopelessly memory-bound. Training is compute-bound; decoding is bandwidth-bound.
Same layer, opposite regime. (This becomes the KV-cache story in ch03.)

### Elementwise ops: $I = \Theta(1)$ — derivation

$y = f(x)$ over $n$ elements, one FLOP per element: reads $sn$ bytes, writes $sn$
bytes:

$$I = \frac{n}{2sn} = \frac{1}{2s} = 0.125 \text{ FLOP/B (fp32)}.$$

Constant — no $n$ makes it better. Even a fancier op like GELU (~10 FLOPs/element)
reaches only ~1.25 FLOP/B, still 20× below the T4 fp32 ridge. **Every elementwise
op is memory-bound on every modern GPU**; its runtime is simply
$\text{bytes}/B$. Adding two 1 GiB fp32 tensors on a T4:
$3 \cdot 2^{30} \cdot ... $ — precisely, moves $3 \times 1\,\text{GiB}$ ≈ 3.2 GB at
320 GB/s ⇒ ~10 ms, regardless of the "+" costing only 0.27 G FLOPs (which the fp32
units could do in 33 µs). The measurement script confirms the ratio.

**Fusion follows immediately:** computing `gelu(x + b)` as two kernels moves
$\sim 5sn$ bytes; fused, $\sim 3sn$ — a 1.67× speedup from arithmetic that costs
nothing. This is the entire economic case for `torch.compile`, FlashAttention
(ch03/ch13), and fused optimizers (ch04).

### Little's law for latency hiding (why occupancy exists)

To sustain bandwidth $B$ with memory latency $L$, the SM must keep
$B \times L$ bytes in flight (Little's law). T4: $320\,\text{GB/s} \times
{\sim}500\,\text{cycles}/(1.59\,\text{GHz}) \approx 100\,\text{KB}$ in flight
GPU-wide — thousands of outstanding loads. A warp can issue a handful; hence you
need *hundreds of resident warps* machine-wide, i.e., grids of thousands of
threads, i.e., **big tensors**. This is the mathematical reason batch size exists
and why tiny models can't saturate big GPUs (ch05's "the third wall").

## Implementation

The notebook mirrors this chapter with CPU-safe guards; the heavy measurements live
in three scripts.

**`experiments/ch01/matmul_tflops_sweep.py`** — sweeps $n \in \{256 \dots 8192\}$
and dtype ∈ {fp32, fp16, bf16}, computing

```python
flops = 2 * n**3
tflops = flops / (stats["mean_ms"] * 1e-3) / 1e12   # stats from llmdist timeit (CUDA events)
```

and printing achieved TFLOPS vs the hardware peak, plus an odd-size probe
(n vs n±1) to expose the tensor-core alignment cliff. On CPU it runs a reduced
sweep so CI still exercises the code path.

**`experiments/ch01/roofline_ops.py`** — times one compute-bound op (large matmul)
and several memory-bound ops (`add`, `mul`, GELU, LayerNorm, softmax) on equal-size
tensors, reports achieved GB/s (from known bytes moved) and GFLOP/s, and places
each op on a roofline using the measured — not nominal — peaks.

**`experiments/ch01/async_and_pcie.py`** — three demos:

1. *The `time.time()` lie*: wall-clock around an un-synchronized launch (µs) vs
   CUDA-event time of the same kernel (ms), printed side by side.
2. *Launch overhead*: mean time of `n` back-to-back trivial kernels ⇒ µs/launch.
3. *PCIe bandwidth*: H2D and D2H for pageable vs pinned buffers across sizes,
   with `non_blocking=True` overlap demonstrated on pinned memory.

All three scripts detect the environment via `llmdist.utils.env_check.detect()` and
print an informative skip on CPU-only machines, per the course contract. Timing
always uses `llmdist.profiling.timers.timeit` — warmup included, CUDA events on GPU
(read that module now; it is 50 lines and you will use it in every chapter).

## Profiling & measurement

What to expect (derived from the specs above — verify, don't trust):

- **Matmul sweep, T4:** fp32 should plateau around 6–7.5 TFLOPS (75–95 % of the
  8.1 peak) for $n \gtrsim 2048$; fp16 should reach several× that (tensor cores;
  achieving >50 % of the 65 TFLOPS peak needs large, aligned shapes). Small $n$
  (≤ 512) will sit far below both peaks — launch overhead plus too little
  parallelism to hide latency (Little's law above).
- **Alignment probe:** n=4096 vs 4097 in fp16 should show a marked drop for the
  ragged shape; fp32 shows little difference (no tensor-core path to lose).
- **Roofline ops, T4:** every elementwise op should land near the same *GB/s*
  (~250–300, i.e., a large fraction of 320), wildly different GFLOP/s. If two
  elementwise ops differ a lot in GB/s, suspect kernel overhead at small sizes —
  rerun with bigger tensors.
- **Launch overhead:** ~3–10 µs per launch. Multiply by your model's kernel count
  per step for a sobering floor on step time.
- **PCIe:** pageable a few GB/s, pinned ~10–13 GB/s on PCIe 3.0 ×16. If pinned is
  *not* faster, you are measuring allocation, not transfer — allocate once, copy
  many times (the script does).

How to read results: always compute *percent of peak*, not raw numbers. 90 % of
peak bandwidth for an elementwise op means "optimal; only fusion can help".
20 % of peak FLOPS for a big matmul means "wrong kernel — check dtype, alignment,
and whether you timed the first (uncached/autotuning) call".

## Common mistakes

1. **Symptom:** "my GPU op takes 0.05 ms" for a huge matmul. **Cause:** timed an async launch with `time.time()`, no synchronize. **Fix:** CUDA events or `torch.cuda.synchronize()` before reading the clock; use `llmdist.profiling.timers.timeit`.
2. **Symptom:** first iteration is 100× slower than the rest. **Cause:** CUDA context creation, cuBLAS handle init, kernel autotuning, memory-pool warmup. **Fix:** warm up before timing (our `timeit` does 5 warmup iters by default).
3. **Symptom:** fp16 matmul no faster than fp32. **Cause:** shapes not multiples of 8/16 (no tensor-core dispatch), or the op is so small it is launch/memory-bound where dtype barely matters. **Fix:** pad dims (e.g., vocab to a multiple of 64), test at large n.
4. **Symptom:** "adding more FLOPs made it no slower" (e.g., `x.mul(2).add(3)` ≈ cost of `x.add(3)` when fused). **Cause:** memory-bound regime — time is bytes/bandwidth; arithmetic is free. **Fix:** none needed — but *recognize* the regime; optimize bytes, not FLOPs.
5. **Symptom:** `pin_memory=True` shows no speedup in your benchmark. **Cause:** timing includes the (slow) one-time pinned allocation, or the tensor is tiny so latency dominates, or you never used `non_blocking=True` so there was no overlap to win. **Fix:** allocate once outside the loop; measure ≥ 64 MB transfers; overlap explicitly.
6. **Symptom:** GPU "utilization" is 100 % in `nvidia-smi` but training is slow. **Cause:** that metric means "a kernel was resident during the sample", not "the SMs were busy" — a memory-bound or launch-bound stream shows 100 % too. **Fix:** profile with torch.profiler / Nsight (ch24); compute percent-of-peak as above.
7. **Symptom:** OOM at a batch size the parameter math says should fit. **Cause:** activations and the caching allocator's reserved-but-unallocated pool (ch02) aren't in your math. **Fix:** the ch03 activation formula plus `torch.cuda.memory_summary()`.

## Limitations & outlook

The roofline is a bound, not a prediction: real kernels lose to wave quantization
(a 40-SM T4 running 41 blocks takes two "waves"), shared-memory bank conflicts, and
non-compulsory traffic. We also ignored L2 — medium tensors that fit in A100's
40 MB L2 can beat the HBM roofline. And we treated the GPU as alone: chs 06–09
extend the same $\min(P, I\cdot B)$ logic to a cluster, where $B$ becomes NVLink/
PCIe/network bandwidth and $I$ becomes FLOPs per *communicated* byte — arguably the
single most useful idea in this course. Finally, everything here was inference-of-
one-op; ch04 assembles ops into a training step and asks where the milliseconds go.

## Exercises

1. **(Easy)** Compute arithmetic intensity for: (a) fp32 dot product of two length-$n$ vectors; (b) fp16 square matmul at $n=8192$; (c) fp32 softmax over $n$ elements (~5 FLOPs/element, and note the second pass!). Which are memory-bound on an A100?
2. **(Easy)** Your model launches 1,800 kernels per step and steps take 25 ms on a fast GPU. Using 5 µs/launch, what fraction of the step is launch overhead alone? What tools attack it?
3. **(Medium)** From first principles, how long *must* `torch.nn.functional.gelu(x)` take on a T4 for `x` of 512 MiB fp32 (out-of-place)? Measure with `roofline_ops.py` and report percent of your predicted bound.
4. **(Medium)** Modify `matmul_tflops_sweep.py` to sweep rectangular shapes with fixed FLOPs ($2mnk$ constant): e.g., (4096,4096,4096) vs (16384,4096,1024) vs (65536,4096,256). Predict the ranking from intensity $2mnk / s(mk{+}kn{+}mn)$ before running.
5. **(Hard)** Estimate matmul time at $n = 4096$ fp16 on a T4 two ways: compute-roof ($2n^3/65\text{T}$) and memory-roof ($3n^2\cdot 2 / 320\text{G}$). Which binds? Now include the *tiled* traffic model: with 128×128 output tiles resident in shared memory, each A/B element is loaded $n/128$ times → effective bytes $= 2n^2\cdot(n/128)\cdot 2 + ...$ — redo the bound. What tile size makes traffic ≤ compute time?
6. **(Hard)** Write a two-stream experiment: stream A computes a big matmul while stream B runs an H2D copy from pinned memory. Show total time ≈ max(compute, copy), not the sum. Then break it by making the copy pageable. Explain both results.
7. **(Research)** FlashAttention claims attention is memory-bound because the $T{\times}T$ score matrix round-trips HBM. Using this chapter's tools, derive the intensity of naive attention ($O(T^2 d)$ FLOPs vs $O(T^2)$ bytes for scores) and of the tiled version that never materializes scores. At what $T$ does naive attention leave the compute roof on an A100 at $d_{head}=64$?

## Solutions

<details>
<summary>Solutions 1–4</summary>

**1.** (a) $2n$ FLOPs, $\ge 8n$ bytes read ⇒ $I = 0.25$: memory-bound everywhere. (b) $I = n/(3s) = 8192/6 \approx 1365$ FLOP/B ≫ 153: compute-bound. (c) ~5 FLOPs/element but two passes over the data (max, then exp/sum/div) ⇒ ≥ 3–4 memory touches: $I \lesssim 0.4$: memory-bound. Only (b) is compute-bound.

**2.** $1800 \times 5\,\mu s = 9$ ms of a 25 ms step = 36 %. Attacks: kernel fusion (`torch.compile`), CUDA Graphs (replay whole step with one launch), bigger batch (same launches, more work each).

**3.** Bytes = read + write = $2 \times 512\,\text{MiB} = 1\,\text{GiB} \approx 1.074$ GB ⇒ $t \ge 1.074/320 \approx 3.4$ ms. GELU's ~10 FLOPs/element (1.3 GFLOP total) would take ~0.17 ms of fp32 compute — irrelevant. Achieving ≥ 80 % of the bound is normal.

**4.** Intensity falls as shapes get skinnier (the $mn$ or $mk$ term balloons relative to $2mnk$): (4096³) $I \approx 683$ (fp16) > (16384,4096,1024) $I\approx 372$ > (65536,4096,256) $I \approx 114$. Expect achieved TFLOPS to fall in that order despite equal FLOPs; the skinniest may also lose tensor-core efficiency on the short-K dimension.

</details>

<details>
<summary>Notes toward 5–7</summary>

**5.** Compute roof: $2\cdot 4096^3 / 65\text{e}12 \approx 2.1$ ms. Naive traffic (every element re-read per output): absurd. Compulsory: $3\cdot4096^2\cdot 2\,B = 100$ MB ⇒ 0.31 ms. With 128×128 tiles: A read $n/128 = 32$ times ⇒ $A{+}B$ traffic $= 2\cdot 4096^2 \cdot 32 \cdot 2\,\text{B} \approx 2.1$ GB ⇒ 6.7 ms > compute — 128 tiles are *not* enough on paper; real kernels also tile K and reuse in registers, cutting traffic further. The exercise's point: shared-memory blocking is what buys the compute-bound regime, and the required reuse factor is $\ge B_{\text{needed}} = 2n^3\cdot s / (P^{-1} \cdot ...)$ — work it through.

**6.** With pinned + `non_blocking=True` + separate streams, the copy engine (DMA) and SMs are independent units ⇒ overlap; total ≈ max. Pageable forces a staging memcpy by a driver thread and serializes with the stream ⇒ total ≈ sum. This experiment is the seed of ch09's compute/communication overlap.

**7.** Naive: FLOPs $\approx 4T^2 d$ per head-layer (two matmuls), score-matrix traffic $\approx 2\cdot 2T^2$ bytes (write + read, fp16) ignoring Q,K,V ⇒ $I \approx d$ — with $d_{head}=64$, $I\approx 64 < 153$: below the A100 bf16 ridge ⇒ memory-bound *at any T* (T cancels!). Tiled (FlashAttention) keeps scores in SRAM: traffic reverts to $O(Td)$ ⇒ $I = O(T)$, compute-bound for large T. The punchline: the fix is a memory-hierarchy fix, not a FLOP fix.

</details>

## Interview questions

1. Walk me through what happens between `y = a @ b` in Python and electrons moving in the GPU: enqueue, launch, scheduling onto SMs, warps, retirement.
2. Define occupancy. Give a case where *increasing* occupancy makes a kernel slower.
3. Why do GPUs need thousands of threads in flight when a CPU core needs one or two? (Answer with Little's law.)
4. Derive the arithmetic intensity of an fp32 square matmul and of `x + y`. Where is each relative to an A100's ridge point?
5. What exactly does a tensor core compute, and why do shape multiples of 8/16 matter for hitting them?
6. Your colleague benchmarks a kernel at 0.02 ms with `time.time()` and it "gets slower" when they add a print of the output. What happened?
7. Explain the difference between pageable and pinned host memory, and why only pinned copies can be asynchronous.
8. `nvidia-smi` shows 100 % GPU utilization but you suspect the GPU is mostly idle. How is that possible, and what would you measure instead?
9. Why is LLM *training* typically compute-bound but LLM *decoding* memory-bound, when they run the same weights?
10. A training step launches thousands of small kernels. Name three techniques that reduce launch-bound overhead and the mechanism of each.

## Summary

- GPUs hide latency with parallelism: warps stall, schedulers switch — so throughput needs *lots* of resident work.
- The memory hierarchy loses ~an order of magnitude of bandwidth per level: registers → shared/L1 → L2 → HBM → PCIe → network. Distributed training is the outermost level of the same hierarchy.
- Roofline: attainable FLOP/s $= \min(P, I\cdot B)$; ridge $I^\* = P/B$ ≈ 25 (T4 fp32), ~200 (T4 fp16-TC), ~150 (A100 bf16-TC) FLOP/byte.
- Matmul intensity grows as $n/6$ (fp32) — compute-bound when large; every elementwise op is stuck at $I < 1$ — always memory-bound; optimize bytes, not FLOPs.
- Tensor cores multiply small matrix tiles per clock: use fp16/bf16 and aligned shapes or lose ~an order of magnitude of peak.
- Matrix–vector work (inference decoding) is memory-bound no matter what; batch it or pay bandwidth.
- Kernel launches cost ~3–10 µs; thousands per step add up — fuse (torch.compile) or replay (CUDA Graphs).
- CUDA is asynchronous: never trust a wall clock without a synchronize; use CUDA events (our `timeit`).
- Pinned memory ≈ 2–4× PCIe throughput and unlocks async H2D/D2H — the seed of all compute/comm overlap.
- Percent-of-peak is the only honest performance metric; compute it for every measurement in this course.

## References

See `references/ch01.md` for the annotated list. Highlights:

- Williams, Waterman & Patterson (2009), *Roofline: An Insightful Visual Performance Model* — the original; ten pages that organize this entire course.
- NVIDIA, *CUDA C++ Programming Guide* & *Best Practices Guide* — execution model, streams, pinned memory, from the source.
- Jia et al. (2018), *Dissecting the NVIDIA Volta/Turing GPU Architecture via Microbenchmarking* — where honest latency numbers come from.
- Volkov (2010), *Better Performance at Lower Occupancy* — kills the "maximize occupancy" myth.
- He et al./NVIDIA docs on Tensor Core requirements — the alignment rules, from the vendor.

# Chapter 04 — Single-GPU Training Anatomy

> **Difficulty:** 🟢 Easy · **Study time:** 3–4 h · **Requires:** chapters 02, 03
> **Notebook:** notebooks/ch04_single_gpu_training.ipynb · **Experiments:** experiments/ch04/

## Learning objectives

You will be able to...

- Name the four phases of one training step and say exactly which tensors exist during each.
- Explain why GPU memory *rises* during forward, *falls* during backward, and *jumps* at the first `optimizer.step()`.
- Derive the Adam memory ledger from first principles: 16 bytes per parameter for FP32 training, and — surprisingly — 16 bytes per parameter for mixed precision too.
- Predict a model's steady-state training memory before running it, using `llmdist.profiling.memory.model_memory_ledger`, and verify the prediction against `torch.cuda.max_memory_allocated`.
- Explain why FP16 needs dynamic loss scaling and BF16 does not, from the bit layouts of the two formats.
- Use `zero_grad(set_to_none=True)` and explain what it saves relative to `set_to_none=False`.
- Capture and read a `torch.profiler` chrome trace of one training step.
- Diagnose the classic "trains one step, OOMs on step one" failure.

## Intuition

A training step looks atomic from the outside: `loss.backward(); opt.step()`. From the
GPU's point of view it is a *choreography of allocations* — thousands of tensors born
and destroyed in a precise order. Distributed training (everything from chapter 06
onward) is about splitting this choreography across machines, so before we split it we
must be able to narrate it byte by byte on one device.

The analogy that will carry through the whole course: **training memory is a ledger**.
Every byte on the GPU is attributable to one of five accounts:

1. **Parameters** — the model weights. Constant after construction.
2. **Gradients** — one tensor per parameter, same shape. Exist between `backward()` and
   `zero_grad()`.
3. **Optimizer state** — Adam's momentum (`m`) and variance (`v`) buffers. **Created
   lazily on the first `step()`** — the single most common source of "it OOMed after
   the first step" surprise.
4. **Activations** — everything autograd saves during forward so backward can use it.
   The only account that scales with batch size and sequence length.
5. **Workspace/cache** — cuBLAS scratch, the CUDA caching allocator's slack
   (`reserved − allocated`), fragmentation.

If you can predict accounts 1–3 with arithmetic (this chapter) and account 4 with the
formulas from chapter 03, then every OOM stops being a mystery and becomes a subtraction
problem. That predictive power is what makes ZeRO (ch15), FSDP (ch16), and activation
checkpointing (ch21) *derivable* rather than magical: each one is just a decision to
move one account somewhere else.

## Theory

### The four phases of one step

```
 time ──────────────────────────────────────────────────────────────────▶

 phase        FORWARD              BACKWARD            STEP        ZERO_GRAD
            ┌───────────┐        ┌───────────┐      ┌────────┐    ┌────────┐
 params     ████████████████████████████████████████████████████████████████  (constant)
 activations       ▲▲▲▲▲▲▲▲▲     ▼▼▼▼▼▼▼▼▼                                     (rise, then freed as consumed)
 gradients                       ▲▲▲▲▲▲▲▲▲▲  ████████████████    (freed)       (appear per-tensor during backward)
 adam m,v                                        ▲▲▲▲▲██████████████████████  (born on FIRST step, then constant)
                                                     │
 memory                                              └── the step-1 jump
   ▲                          ●  ← peak: max activations + all grads so far
   │                     ●        ●
   │                ●                  ●●●●●●─────────  ← new, higher floor after step 1
   │      ●●●●●
   │ ●────                                   (params + grads + m + v)
   └──────────────────────────────────────────────────────────────▶ time
```

**Phase 1 — forward.** Each layer's output that backward will need is stashed by
autograd (`ctx.save_for_backward`, chapter 02). Memory climbs roughly linearly with
depth. For MiniGPT the dominant saved tensors per block are the attention matrix
`(B, H, T, T)` and the MLP's `4·d_model`-wide hidden activation.

**Phase 2 — backward.** Autograd walks the graph in reverse topological order. Two
opposing flows: gradient tensors are *allocated* (one per parameter, as its grad is
computed), and saved activations are *freed* (as soon as the last node needing them has
run — their refcount drops to zero and the caching allocator reclaims the block).
Net effect: memory descends a staircase but with a bump at the start — **the peak of
the whole step usually occurs early in backward**, when nearly all activations are
still alive *and* the first gradients have already been allocated.

**Phase 3 — `optimizer.step()`.** Adam updates every parameter:

```
m ← β₁·m + (1−β₁)·g          # first moment  (needs a buffer the size of params)
v ← β₂·v + (1−β₂)·g²         # second moment (another buffer the size of params)
p ← p − lr · m̂ / (√v̂ + ε)
```

PyTorch creates `m` and `v` *lazily*, inside the first `step()` call — not in the
`Adam(...)` constructor. Why lazily? Because until the first step the optimizer cannot
know which parameters actually receive gradients (think frozen layers), and eager
allocation would waste memory for them. The consequence: **step 0 fits, step 1 may
not** — the process runs forward, backward, and then dies inside `step()` while
allocating `2 × 4 bytes × N` of state. If you have ever "successfully" run one step
and OOMed on the next, this is why.

**Phase 4 — `zero_grad(set_to_none=True)`.** With `set_to_none=True` (the default since
PyTorch 2.0), gradient tensors are *deallocated*, not zero-filled. Three wins:
(a) no memset kernel; (b) the memory is returned to the caching allocator and can be
reused for next step's activations; (c) the next backward writes grads with `=` instead
of `+=`, saving a read. The cost: between `zero_grad` and the next `backward`, `p.grad`
is `None`, so code that pokes at gradients must handle that.

### Mixed precision changes *what* is stored, not *how much* (for the static accounts)

With `torch.autocast` + `GradScaler` (FP16) or autocast-BF16, the forward/backward run
in 16-bit, but the *master copy* of the weights and the Adam moments stay in FP32 —
otherwise tiny updates (`lr·m̂/√v̂` is often ~1e-6 of the weight) would be rounded to
nothing when added to a 16-bit weight. So mixed precision's win is in **activations**
(halved) and **speed** (tensor cores), not in the static ledger. The Mathematics
section makes this exact.

### Why FP16 needs loss scaling

FP16 has 5 exponent bits: the smallest normal number is 2⁻¹⁴ ≈ 6.1e-5, and subnormals
bottom out at 2⁻²⁴ ≈ 5.96e-8. Gradients of well-trained networks are routinely smaller
than that — they **underflow to zero silently**, and training stalls with no error.

The fix is a change of units: multiply the loss by a scale S (say 2¹⁶) before
`backward()`. By linearity of differentiation, every gradient in the graph is scaled by
S, lifting the small ones out of the underflow zone. Before `step()`, the optimizer
divides the grads by S ("unscale"). Two failure-handling rules complete the scheme:

- If any unscaled grad is `inf`/`nan` (the scale pushed something *over* the top,
  2¹⁶ · large ≥ 65504 = FP16 max), **skip the step** and halve S.
- If many consecutive steps succeed, double S (use the widest safe unit).

This is `torch.cuda.amp.GradScaler`, and it is why an AMP training loop occasionally
logs a skipped step — that's the scaler recalibrating, not a bug.

**BF16 doesn't need any of this.** BF16 keeps FP32's 8 exponent bits and sacrifices
mantissa (7 bits instead of FP16's 10). Same dynamic range as FP32 → no underflow → no
scaler → simpler code. The price is precision, which the FP32 master weights and FP32
accumulation in matmuls absorb. This is why every post-2022 LLM you have heard of
trains in BF16 (A100 and newer; T4 has no BF16 tensor cores, which matters on Kaggle).

## Mathematics

### The Adam ledger, byte by byte

Let N be the number of parameters. Full FP32 training:

| account            | dtype | bytes/param |
|--------------------|-------|-------------|
| parameters         | fp32  | 4           |
| gradients          | fp32  | 4           |
| Adam m             | fp32  | 4           |
| Adam v             | fp32  | 4           |
| **total static**   |       | **16**      |

**FP32 training: 16 N bytes**, before a single activation.

Mixed precision (the standard AMP recipe with FP16 compute):

| account              | dtype | bytes/param |
|----------------------|-------|-------------|
| parameters (compute) | fp16  | 2           |
| gradients            | fp16  | 2           |
| master parameters    | fp32  | 4           |
| Adam m               | fp32  | 4           |
| Adam v               | fp32  | 4           |
| **total static**     |       | **16**      |

**Mixed precision: also 16 N bytes.** The 16-bit savings on params and grads (−4)
are exactly cancelled by the FP32 master copy (+4). This "4 + 12" split — 4 bytes of
16-bit param+grad, 12 bytes of FP32 optimizer-side state — is *the* number to memorize:
it is the K = 12 in the ZeRO paper, and chapters 15–16 are entirely about whose GPU
holds those 12 bytes. (Note: `torch.autocast` alone doesn't make fp16 grads — grads
land in the params' dtype. The 2+2+12 layout is what DeepSpeed/Megatron-style fp16
trainers implement explicitly; the point of the table is the ledger, not one API.)

Sanity check against real models:

- MiniGPT default (cfg d=256, L=4, V=8192): N = 5,322,240 ≈ 5.3 M → static ≈ 5.3M × 16 B ≈ 85 MB.
- 7 B parameters: 7e9 × 16 B = **112 GB** static — the number that opens chapter 05.

### Activations: the dynamic account

From chapter 03, per transformer block with batch B, sequence T, width d, heads H, the
big saved tensors are ≈ `T·B·d·(16 + …) + B·H·T²·(…)` elements; the exact coefficient
depends on implementation, and our explicit (non-fused, non-flash) attention saves the
full `(B,H,T,T)` attention matrix twice (pre- and post-softmax), so the T² term
dominates early. In this chapter we do not re-derive the formula — we *measure* the
activation account as a residual:

```
activations_at_peak ≈ peak_measured − params − grads_alive_at_peak
```

and check it scales linearly with B (double the batch → double the residual), which the
notebook demonstrates.

### The peak, precisely

Steady-state floor (between steps, after zero_grad(set_to_none=True)):

```
floor = params + m + v            = 12 N bytes   (fp32)
```

Peak (early in backward, step ≥ 1):

```
peak ≈ params + m + v + activations_max + grads_allocated_so_far
     ≤ 16 N + activations_max
```

Step 0's peak is lower by 8 N (no m, v yet) — which is exactly why a model can complete
step 0 and OOM in step 1's `optimizer.step()`. The step-1 jump you will measure in
`experiments/ch04/memory_timeline.py` is ΔN×8 bytes, to within allocator noise.

### Loss scaling arithmetic

If the true gradient element is g and FP16's smallest representable magnitude is
2⁻²⁴, then g is representable after scaling iff `S·|g| ≥ 2⁻²⁴`, i.e.
`|g| ≥ 2⁻²⁴/S`. With S = 2¹⁶ the floor drops from 6e-8 to 2⁻⁴⁰ ≈ 9e-13 — comfortably
below any gradient that matters. The ceiling constraint is `S·|g| ≤ 65504`, i.e.
`|g| ≤ 2⁻¹⁶·65504 ≈ 1.0` — and gradients above 1 do occur (spikes), hence the
skip-and-halve rule rather than a fixed S.

## Implementation

The instrumented training step, from the notebook (`notebooks_src/ch04_single_gpu_training.py`):

```python
from llmdist.models.mini_gpt import MiniGPT, GPTConfig
from llmdist.profiling.memory import track, snapshot, model_memory_ledger

model = MiniGPT(GPTConfig()).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

with track("forward"):
    logits, loss = model(x, y)      # activations accumulate
with track("backward"):
    loss.backward()                 # grads born, activations freed
with track("optimizer.step"):
    opt.step()                      # m, v born HERE on the first call
with track("zero_grad"):
    opt.zero_grad(set_to_none=True) # grads deallocated
```

Run this for three consecutive steps and read the `Δalloc` column: forward is a large
`+`, backward a smaller `+` or a `−` (grads in, activations out), **step is a large `+`
on step 0 and ~0 afterwards**, zero_grad is a `−`.

The ledger prediction lives in `src/llmdist/profiling/memory.py`:

```python
def model_memory_ledger(model, optimizer_bytes_per_param=8, dtype_bytes=4):
    n = sum(p.numel() for p in model.parameters())
    return {"params_mb": mb(n * dtype_bytes),
            "grads_mb": mb(n * dtype_bytes),
            "optimizer_mb": mb(n * optimizer_bytes_per_param),
            "total_static_mb": mb(n * (2 * dtype_bytes + optimizer_bytes_per_param)), ...}
```

`optimizer_bytes_per_param=8` is Adam's m+v; pass `4` for momentum-SGD, `0` for plain
SGD — the ledger is optimizer-agnostic arithmetic. `experiments/ch04/ledger_check.py`
compares `total_static_mb` against the measured post-step-1 floor for several model
sizes; agreement within a few percent (the residual is buffers like the causal mask,
and allocator rounding to 512-byte blocks).

### torch.profiler in five lines

```python
from torch.profiler import profile, schedule, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=2),   # skip step 0's lazy-init noise
    profile_memory=True, record_shapes=True,
) as prof:
    for step in range(4):
        train_step(); prof.step()
prof.export_chrome_trace("profiling/ch04_train_step.json")
```

- `schedule` matters: step 0 contains cuDNN autotuning, lazy Adam init, and allocator
  warmup — profiling it tells you about startup, not steady state. `wait=1, warmup=1`
  discards those.
- `activities` — CPU shows the Python/dispatch side; CUDA shows kernels. You need both
  to see *gaps* (GPU idle while Python dawdles — the launch-bound regime from ch01).
- Open the trace at `chrome://tracing` or https://ui.perfetto.dev. Read it in this order:
  1. Find one `optimizer.step` on the CPU row — that brackets a step.
  2. Look at the CUDA stream row *under* forward: densely packed kernels = compute-bound;
     kernels separated by gaps = launch/CPU-bound (small models, like our MiniGPT on CPU-heavy configs).
  3. `aten::mm` / `aten::addmm` should dominate GPU time; if `aten::copy_` or elementwise
     ops dominate, you are memory-bound (ch01's roofline).

`experiments/ch04/profiler_trace.py` produces the trace; the doc cannot include the
numbers because they are hardware-specific — measure yours.

## Profiling & measurement

What to measure (all four scripts run on CPU with graceful degradation; on CPU we
account tensor bytes by walking `p.grad` and `opt.state` instead of asking the CUDA
allocator):

1. **`memory_timeline.py`** — allocated memory sampled after every phase for 3 steps.
   Expected on any CUDA GPU: the post-`step` floor rises by ≈ 8 bytes/param between
   step 0 and step 1, then stays flat. That prediction is arithmetic, not benchmark:
   ΔMB = N × 8 / 2²⁰.
2. **`ledger_check.py`** — predicted vs measured static memory for d_model ∈
   {128, 256, 512, 768}. Expect measured/predicted ≈ 1.00–1.05; the excess is
   non-parameter buffers (causal masks: `block_size²` bools per layer) and rounding.
3. **`throughput_vs_batch.py`** — tokens/s vs batch size, `Throughput` meter from
   `llmdist.profiling.timers`. Expect tokens/s to *rise* with batch (better SM
   occupancy, fixed per-launch overhead amortized) then *plateau* once the GPU is
   saturated — the plateau is your compute roof. On a T4 the knee for MiniGPT-default
   arrives at small batch sizes; find it with the script rather than trusting a number.
4. **`profiler_trace.py`** — exports `profiling/ch04_train_step.json` (chrome trace)
   plus a `key_averages()` table. On T4, expect matmuls on top; on CPU, expect
   the softmax/attention pointwise ops to be relatively more expensive.

Reading rule from the style guide: numbers in this doc are derivations; anything
measured comes from your run of these scripts.

## Common mistakes

1. **OOM at step 1 after a clean step 0.**
   *Symptom:* first iteration completes; `CUDA out of memory` inside `optimizer.step()`.
   *Cause:* Adam's m and v (8 bytes/param) are allocated lazily on the first `step()`.
   *Fix:* budget 16 bytes/param from the start (`model_memory_ledger`), or use a
   smaller model/8-bit optimizer/ZeRO (ch15).
2. **Measuring memory with `nvidia-smi` and panicking.**
   *Symptom:* nvidia-smi shows GBs more than your tensors could possibly use.
   *Cause:* nvidia-smi reports reserved cache + CUDA context (~300–600 MB), not live tensors.
   *Fix:* use `torch.cuda.memory_allocated/reserved`; the gap is reusable cache, not a leak.
3. **Timing the step with `time.time()` and no synchronize.**
   *Symptom:* forward "takes" 2 ms, backward "takes" 80 ms — backward looks 40× slower.
   *Cause:* kernel launches are async; the first synchronizing call absorbs all queued work.
   *Fix:* `llmdist.profiling.timers.timeit` (CUDA events + warmup), as in ch01.
4. **`zero_grad()` before PyTorch 2.0 semantics assumed (`set_to_none=False`).**
   *Symptom:* memory floor is 4 N bytes higher than the ledger predicts.
   *Cause:* zero-filled grads persist between steps instead of being freed.
   *Fix:* `opt.zero_grad(set_to_none=True)` and handle `p.grad is None` in any inspection code.
5. **FP16 training silently stalls (loss plateaus early, small-gradient layers frozen).**
   *Symptom:* no error, but embeddings/early layers stop moving.
   *Cause:* gradient underflow below 2⁻²⁴ without loss scaling.
   *Fix:* `GradScaler`, or switch to BF16 on Ampere+ hardware.
6. **GradScaler used but `scaler.unscale_` forgotten before gradient clipping.**
   *Symptom:* clipping threshold behaves absurdly (clips everything or nothing).
   *Cause:* you clipped the *scaled* grads — S times too large.
   *Fix:* `scaler.unscale_(opt)` before `clip_grad_norm_`, then `scaler.step(opt)`.
7. **Profiling step 0 and optimizing the wrong thing.**
   *Symptom:* trace shows huge one-off allocations and weird slow kernels.
   *Cause:* lazy init + autotuning pollute the first iterations.
   *Fix:* `schedule(wait=1, warmup=1, active=2)`; profile steady state.
8. **Keeping a Python reference to `loss` (or logging `loss` not `loss.item()`) across steps.**
   *Symptom:* memory climbs a little every step until OOM.
   *Cause:* the loss tensor retains the whole autograd graph, including saved activations.
   *Fix:* log `loss.item()` / `loss.detach()`; let the graph die each step.

## Limitations & outlook

- The ledger says a 7 B model needs ~112 GB *static* — no single affordable GPU holds
  that. Chapter 05 quantifies this "memory wall" and adds the compute wall (GPU-years)
  and the communication wall.
- Activations are the batch-scaling account; chapter 21 (activation checkpointing)
  trades one extra forward for freeing them, and chapter 03's formulas tell you when
  that trade wins.
- The 16 B/param split (4 hot + 12 optimizer-side) is precisely what ZeRO shards:
  stage 1 shards the 8 (m,v), stage 2 adds the grad 4, stage 3 adds the params.
  When you reach ch15, re-derive its memory claims from this chapter's table.
- `torch.profiler` on one GPU reads cleanly; with 8 ranks you get 8 traces and the
  interesting story is in their *alignment* (who waits for whom) — ch24.

## Exercises

1. **(Easy)** Without running anything: MiniGPT with `d_model=512, n_layer=8,
   vocab_size=8192, block_size=256, bias=True`. Use `param_formula()`'s expression to
   compute N, then the FP32 static ledger in MB. Check with `ledger_check.py`.
2. **(Easy)** Predict the step-0 → step-1 floor jump in MB for the default config
   (N ≈ 5.3 M). Verify with `memory_timeline.py` on a GPU (or reason it through the
   CPU accounting output).
3. **(Medium)** Modify the notebook's training loop to use SGD with momentum, and
   predict the new ledger (params 4 + grads 4 + momentum 4 = 12 B/param). Measure the
   smaller step-1 jump.
4. **(Medium)** Register a full-backward hook on each `Block` that prints
   `torch.cuda.memory_allocated()` (or CPU byte accounting), and confirm the
   staircase: memory decreases as backward proceeds through blocks in reverse.
5. **(Hard)** Instrument `GradScaler`: train MiniGPT in FP16 with an artificially high
   initial scale (`init_scale=2**24`) and count skipped steps (`scaler.get_scale()`
   shrinks after each skip). Explain the trajectory of the scale value.
6. **(Hard)** The peak occurs early in backward. Using `torch.cuda.memory._record_memory_history()`
   (or the profiler's memory view), identify the exact tensor allocation at the peak
   for B=8, T=256. Which block, which tensor?
7. **(Research)** 8-bit optimizers (bitsandbytes) store m and v in int8 with
   block-wise scaling, cutting the 8 B to ~2 B/param. Read Dettmers et al. 2022 and
   write the modified ledger; what is the new bottleneck account for a 7 B model on
   an 80 GB GPU?

## Solutions

<details>
<summary>Solutions to exercises</summary>

1. Per block: 12d² + 9d + 4d = 12·512² + 13·512 = 3,145,728 + 6,656 = 3,152,384.
   ×8 layers = 25,219,072. Embeddings: (8192+256)·512 = 4,325,376. Final LN: 1,024.
   N = 29,545,472 ≈ 29.5 M. Static = 16 × 29.5 M B ≈ 473 MB ≈ 451 MiB.
2. ΔMB = 5.32e6 × 8 / 2²⁰ ≈ 40.6 MiB (m+v in FP32). Measured jump should match within
   allocator rounding.
3. SGD+momentum stores one buffer → jump ≈ 4 B/param ≈ 20.3 MiB for the default
   config; plain SGD (no momentum): no jump at all — a nice falsifiable prediction.
4. Hooks fire in reverse block order; each print should be lower than the previous
   once that block's activations are consumed, minus the grads just allocated.
   The *first* hook (last block) often shows the global peak.
5. At init_scale=2²⁴, any gradient element > 65504/2²⁴ ≈ 3.9e-3 overflows → nearly
   every early step is skipped and the scale halves each time; it decays geometrically
   until S·max|g| < 65504, then training proceeds and the scale slowly doubles back
   (every `growth_interval` successful steps). Plot log₂(scale) vs step: a falling
   staircase then a slow rising one.
6. For explicit attention the largest single allocation is the attention weights
   `(B, H, T, T)`: 8×4×256×256 floats = 2,097,152 elem = 8 MiB each — and there are
   two alive per block (pre-softmax scores kept for softmax's backward, plus the
   softmax output kept for `att @ v`'s backward). Expect the peak allocation site in
   the last block's attention during early backward.
7. Ledger becomes 4 (p) + 4 (g) + ~2 (int8 m,v + scaling factors) ≈ 10 B/param FP32,
   or 2+2+4+~2 ≈ 10 B/param mixed with fp16 master-less variants — for 7 B: ~70 GB,
   which *fits* in 80 GB but leaves little for activations → the bottleneck moves to
   the activation account, and checkpointing (ch21) becomes mandatory rather than
   optional.

</details>

## Interview questions

1. Walk me through GPU memory during one training step of a transformer. Where is the peak and why?
2. Your job trains one step fine and OOMs on the second. What happened?
3. Derive bytes-per-parameter for Adam in FP32 and in mixed precision. Why are they equal?
4. Why does mixed precision keep an FP32 master copy of the weights?
5. What problem does loss scaling solve, mechanically? Why does BF16 not need it?
6. What does `zero_grad(set_to_none=True)` change about memory and about the next backward?
7. Why is `nvidia-smi` memory always higher than `torch.cuda.memory_allocated()`? Which one do you budget against?
8. When is profiling step 0 misleading? What does a correct profiler schedule look like?
9. In a chrome trace, how do you tell a launch-bound step from a compute-bound one?
10. If you switch Adam → SGD-with-momentum, how does the memory timeline change, quantitatively?

## Summary

- One step = forward (activations ↑), backward (grads ↑, activations ↓), step
  (optimizer state; born lazily at step 1), zero_grad (grads freed with `set_to_none=True`).
- FP32 Adam training: 4+4+4+4 = **16 bytes/param** static.
- Mixed precision: 2+2 hot + 4+4+4 FP32-side = **16 bytes/param** — same total,
  different layout; the 12-byte FP32 block is what ZeRO shards.
- The memory floor jumps by 8 B/param between step 0 and step 1 — Adam's lazy m, v.
- The per-step peak is early in backward: max activations + first grads.
- Activations are the only account scaling with batch × sequence; measure them as
  peak − static residual.
- FP16 underflows below 2⁻²⁴ → loss scaling (scale, unscale, skip on inf/nan);
  BF16 has FP32's exponent range and needs none of it.
- Predict with `model_memory_ledger`, measure with `track`/`max_memory_allocated`;
  they should agree to a few percent.
- Profile steady state (`schedule(wait=1, warmup=1, ...)`), read traces for gaps, and
  let matmuls dominate.
- Every OOM is a subtraction problem once the ledger is second nature.

## References

See `references/ch04.md` for the annotated list. Key items: Kingma & Ba 2015 (Adam —
where m and v come from); Micikevicius et al. 2018 (mixed-precision training — the
loss-scaling recipe verbatim); Rajbhandari et al. 2020 (ZeRO — names our 16 B/param
as 2+2+K, K=12); PyTorch AMP and profiler docs.

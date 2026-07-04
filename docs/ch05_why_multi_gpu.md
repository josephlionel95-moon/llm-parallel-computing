# Chapter 05 — Why Multi-GPU? The Three Walls

> **Difficulty:** 🟢 Easy · **Study time:** 2–3 h · **Requires:** chapters 03, 04
> **Notebook:** notebooks/ch05_why_multi_gpu.ipynb · **Experiments:** experiments/ch05/

## Learning objectives

You will be able to...

- State the three walls — memory, compute, communication — and identify which one a
  given training job is hitting.
- Compute, on paper, whether model X trains on GPU Y (the memory wall), using
  ch04's 16 B/param ledger plus ch03's activation formulas.
- Derive total training FLOPs from `C = 6·N·D` and Chinchilla's `D ≈ 20·N`, and convert
  them into GPU-years at a stated MFU.
- Show that GPT-3-scale training on a single GPU is a matter of *centuries*, from
  arithmetic alone.
- Derive Amdahl's law and explain why a 10% serial fraction caps speedup at 10×, no
  matter how many GPUs you buy.
- State Gustafson's rebuttal and explain why LLM training is closer to Gustafson's
  regime than Amdahl's.
- Explain why escaping walls 1 and 2 necessarily raises wall 3 (communication), and
  write the α + N/β cost preview that chapter 07 develops fully.
- Use `experiments/ch05/training_calculator.py` as a planning tool for any (model, GPU,
  cluster) combination.

## Intuition

Chapters 00–04 built a complete account of a single device: where every byte lives and
where every FLOP goes. This chapter asks the only question that justifies the rest of
the course: **why is one GPU not enough?**

There are exactly three independent reasons, and it pays to keep them separate because
they have different cures:

1. **The memory wall** — the model *does not fit*. Not "runs slowly": literally cannot
   be instantiated with its optimizer state. No amount of patience helps. Cure:
   put different bytes on different devices (ZeRO/FSDP, tensor/pipeline parallelism).
2. **The compute wall** — the model fits, but training would take years or centuries.
   Patience technically helps, but you will not live long enough, and the hardware
   won't either. Cure: put different FLOPs on different devices (data parallelism,
   mostly).
3. **The communication wall** — the meta-wall. Every cure for walls 1–2 requires
   devices to exchange bytes, and the exchange itself costs time. If you parallelize
   naively, communication eats your speedup. Cure: algorithms and overlap
   (chapters 07, 09, 23) — and this is why Part II of the course is the heart of it.

An analogy: a library that must be copied by hand. One scribe with a small desk hits
the *desk* wall (the book doesn't fit open on the desk) and the *time* wall (one scribe,
one lifetime). Hiring 100 scribes fixes both — but now they must pass pages between
desks, agree on who copies what, and wait for the slowest scribe at every chapter
boundary. The passing, agreeing, and waiting is the third wall, and managing it is a
skill entirely separate from copying.

## Theory

### Wall 1 — Memory

From chapter 04: fp32 Adam training needs **16 bytes/param static**, mixed precision
also 16 (with the 4+12 layout), plus activations, plus workspace. A practical planning
number is **16–20 B/param** (the upper end covering activations at modest batch size
with checkpointing, buffers, fragmentation).

Apply that to real models against real GPUs:

| Model | N | Static (16 B/param) | Plan for (18 B/param) | T4 16 GB | A100 40 GB | A100 80 GB | H100 80 GB |
|-------|-----|--------------------|----------------------|----------|-----------|-----------|-----------|
| MiniGPT default | 5.3 M | 0.09 GB | 0.10 GB | ✅ | ✅ | ✅ | ✅ |
| GPT-2 124M | 124 M | 2.0 GB | 2.2 GB | ✅ | ✅ | ✅ | ✅ |
| GPT-2 1.5B | 1.5 B | 24 GB | 27 GB | ❌ | ❌ (tight even before activations) | ✅ | ✅ |
| LLaMA-7B | 7 B | **112 GB** | 126 GB | ❌ | ❌ | ❌ | ❌ |
| LLaMA-70B | 70 B | 1.12 TB | 1.26 TB | ❌ | ❌ | ❌ | ❌ |
| GPT-3 175B | 175 B | 2.8 TB | 3.15 TB | ❌ | ❌ | ❌ | ❌ |

The line to internalize: **a 7 B model — small by modern standards, runnable for
*inference* on a laptop — cannot be *trained* naively on any single GPU ever
manufactured.** Inference needs 2 N bytes (fp16 weights); training needs 8× that before
the first activation. Training is a different sport.

(Generate this table yourself with `experiments/ch05/training_calculator.py`; it also
handles 8-bit optimizers, no-momentum SGD, and activation estimates.)

### Wall 2 — Compute

Chapter 03 derived FLOPs per token ≈ 6 N (2 N forward + 4 N backward, ignoring the
attention T-term). Total training compute:

```
C = 6 · N · D        FLOPs,   D = number of training tokens
```

How big is D? Chinchilla (Hoffmann et al. 2022) found compute-optimal training wants
**D ≈ 20 N** — 20 tokens per parameter. (Modern practice often "overtrains" far past
this for inference economy — LLaMA-3 used ~1900 tokens/param — which makes the wall
*worse*, so 20 N is a friendly lower bound.)

Divide by what one GPU delivers: peak × MFU, where **MFU (Model FLOPs Utilization)** is
the fraction of peak actually spent on the model's FLOPs. Well-tuned large-scale
training achieves 30–50%; we use 40% throughout.

The Mathematics section runs the numbers; the headline results:

| Model | D = 20N | C = 6ND | One A100 (312 TF peak, 40% MFU) | One T4 (65 TF fp16 peak, 40%) |
|-------|---------|---------|-------------------------------|------------------------------|
| 7 B | 140 B tok | 5.9e21 | **≈ 1.5 GPU-years** | ≈ 7 GPU-years |
| 70 B | 1.4 T tok | 5.9e23 | **≈ 150 GPU-years** | ≈ 715 GPU-years |
| 175 B (GPT-3, actual D=300 B) | 300 B tok | 3.2e23 | **≈ 80 GPU-years** | **≈ 384 GPU-years — centuries** |
| 175 B (Chinchilla-optimal) | 3.5 T tok | 3.7e24 | ≈ 930 GPU-years | ≈ 4,500 GPU-years |

These are derivations, not measurements — every entry is `6ND / (peak × 0.4)`,
reproducible by `training_calculator.py`. The conclusion is robust to any reasonable
MFU: **GPT-3 on one GPU is a multi-century project.** The only variable that rescues
you is the *number of GPUs*, which is why GPT-3 was trained on ~10,000 V100s and why
frontier runs use tens of thousands of H100s.

### Wall 3 — Communication (the toll gate)

Both cures create a new cost:

- Data parallelism (cure for wall 2): every step, all replicas must agree on the
  gradient — an AllReduce of N·(bytes/grad) bytes.
- Sharding/model parallelism (cure for wall 1): every forward/backward must move
  activations or gathered parameters between devices.

The universal first-order cost model for moving an N-byte message (developed properly
in chapter 07):

```
T(N) = α + N/β
        │     └── bandwidth term: β = bytes/second of the link
        └──────── latency term: α = fixed per-message cost (μs), paid even for 1 byte
```

Two regimes: small messages pay α (latency-bound — many tiny messages are poison),
large messages pay N/β (bandwidth-bound — the link's plumbing is the limit). The link
matters enormously: NVLink ~600 GB/s, PCIe 3.0 x16 ~16 GB/s theoretical
(Kaggle's 2× T4 are PCIe — we *measure* their real β in chapter 07), Ethernet between
nodes can be 100× slower than NVLink. A gradient AllReduce that is free on NVLink can
dominate the step on PCIe. **Parallelism does not remove work; it converts memory and
compute problems into communication problems** — and the rest of this course is about
doing that conversion at a favorable exchange rate.

### Amdahl vs Gustafson: what speedup is even possible?

Before buying 1000 GPUs, ask what fraction of the work parallelizes. Amdahl's law
(derived below) says a serial fraction s caps speedup at 1/s **for a fixed problem
size**. Gustafson's rebuttal: nobody holds the problem fixed — with more GPUs you train
a bigger model or more tokens, and the serial fraction shrinks with scale. LLM training
sits in Gustafson's regime (the workload is embarrassingly *growable*), but Amdahl
still rules *within one step*: the non-overlapped AllReduce time, the optimizer step,
data loading stalls — each is a serial term that caps per-step scaling. Chapter 09's
compute/communication overlap is precisely an attack on the serial fraction.

## Mathematics

### Memory wall arithmetic (worked example: 7 B)

```
static  = 16 B/param × 7e9            = 112 GB
activations (fp16, B·T = 4096 tokens per device, LLaMA-7B shape d=4096, L=32):
        ≈ L · T·B · d · c  bytes; with the ch03 coefficient c≈34 halves for
          non-flash attention this is O(10s of GB) more — checkpointing (ch21)
          compresses it to O(2·L·T·B·d) = ~2 GB at these settings
total   ≥ 114 GB  ≫  80 GB (largest single GPU)
```

Even with every activation trick, the static 112 GB alone exceeds an H100. To train
7 B on 80 GB you must *remove static bytes from the device*: shard the optimizer's
12 B/param across ranks (ZeRO-1/2 → 4+12/k B/param) or everything (ZeRO-3/FSDP →
16/k B/param). That derivation is chapter 15's opening move, and it is just this
paragraph divided by k.

### Compute wall derivation

Step 1 — FLOPs per token (ch03 recap): a forward matmul pass touches each parameter
in one multiply-accumulate = 2 FLOPs/param/token; backward computes two matmul
gradients (w.r.t. weights and w.r.t. inputs) = 4 FLOPs/param/token. Total ≈ 6 per
parameter per token:

```
C = 6 · N · D
```

Step 2 — Chinchilla: minimizing loss L(N, D) subject to C = 6ND gives N* ∝ C^0.5,
D* ∝ C^0.5, with the fitted constant working out to D ≈ 20 N at the optimum.

Step 3 — wall-clock:

```
time = C / (peak_FLOPs × MFU)
```

Worked, 7 B on one A100 (312 TFLOP/s bf16 dense peak):

```
C    = 6 × 7e9 × 1.4e11 = 5.88e21 FLOPs
rate = 3.12e14 × 0.40   = 1.25e14 FLOP/s
time = 5.88e21 / 1.25e14 = 4.7e7 s ≈ 545 days ≈ 1.5 years
```

GPT-3 (175 B, 300 B tokens as actually trained) on one T4 (65 TFLOP/s fp16 peak):

```
C    = 6 × 1.75e11 × 3e11 = 3.15e23 FLOPs
rate = 6.5e13 × 0.40      = 2.6e13 FLOP/s
time = 3.15e23 / 2.6e13   = 1.21e10 s ≈ 384 years
```

Four centuries, subject only to the assumption that a 175 B model fits on a T4, which
(wall 1) it does not — the walls compound.

Scaling both N and D (Chinchilla) makes compute grow **quadratically** in N:
C = 6·N·(20 N) = 120 N². Double the model → 4× the compute. This is why compute, not
data or ideas, is the currency of the field.

### Amdahl's law, derived

Let one unit of work take time 1 on one processor, with serial fraction s (cannot be
parallelized) and parallel fraction 1−s (perfectly divisible). On p processors:

```
T(p)     = s + (1 − s)/p
Speedup  S(p) = T(1)/T(p) = 1 / (s + (1 − s)/p)
```

Take p → ∞:

```
lim S(p) = 1/s
```

With s = 0.10: S(∞) = 10. **Ten percent serial work caps you at 10× forever** —
1,000 GPUs deliver S(1000) = 1/(0.1 + 0.9/1000) ≈ 9.91, i.e. 99% of the gain came from
the first ~100 GPUs and the other 900 bought almost nothing. The law is merciless
because the serial term does not shrink as p grows.

In distributed training, s hides in: the non-overlapped tail of gradient AllReduce,
optimizer step on unsharded state, Python-side data loading, checkpoint writes,
pipeline bubbles (ch11 derives the bubble as an explicit serial fraction).

### Gustafson's rebuttal, derived

Amdahl fixes the problem and shrinks the time. Gustafson fixes the *time* and grows the
problem: run for time 1 on p processors doing s serial + (1−s) parallel work; a single
processor would have needed

```
T(1) = s + p·(1 − s)
S_scaled(p) = s + p·(1 − s)      — linear in p, no ceiling
```

With s = 0.1, p = 1000: S_scaled ≈ 900. Same hardware, opposite conclusion — because
the question changed from "how much faster is *this* job" to "how much more work can I
do in the same time". LLM training answers Gustafson's question (more tokens, bigger
N, bigger global batch), which is why 10,000-GPU clusters make sense despite Amdahl.
But note the fine print: growing the global batch has diminishing returns beyond the
*critical batch size* (McCandlish et al. 2018), so even Gustafson's escape has a wall
behind it. There is always a wall.

## Implementation

There is deliberately little code in this chapter — the point is arithmetic you can do
on a napkin. The two scripts turn the napkin into a tool:

**`experiments/ch05/training_calculator.py`** — the whole chapter as a function:

```python
def training_memory_gb(n_params, bytes_per_param=16.0, activation_gb=0.0): ...
def training_time(n_params, tokens=None, peak_tflops=312.0, mfu=0.40):
    tokens = tokens or 20 * n_params           # Chinchilla default
    flops = 6.0 * n_params * tokens            # C = 6ND
    seconds = flops / (peak_tflops * 1e12 * mfu)
    ...
```

Run `python experiments/ch05/training_calculator.py --model 7e9 --gpu A100-80GB
--n-gpus 1` and it prints: fits/doesn't fit (and by how much), GPU-years, and the
minimum GPU count for a target wall-clock. No GPU required — it is a calculator.

**`experiments/ch05/oom_point_finder.py`** — the memory wall, experienced. Grows
MiniGPT's `d_model` and trains one step per size until CUDA OOM, then reports the last
size that fit and compares against the ledger's prediction of where the wall should be
given `torch.cuda.get_device_properties(0).total_memory`. On CPU it degrades to
prediction-only mode (estimates the OOM point for common GPUs from the ledger — no
fake measurements, clearly labeled as estimates).

Predicted OOM point, T4 16 GB, MiniGPT shape (L=4, V=8192, T=256, growing d): static
16 B/param plus measured activation slope; the script prints its prediction *before*
probing so you can watch theory meet the allocator.

## Profiling & measurement

1. **`training_calculator.py`** (no hardware needed) — reproduce the two tables in
   this chapter, then your own scenarios. Sanity anchors: 7 B → 112 GB static;
   7 B Chinchilla on one A100 at 40% MFU → ≈ 1.5 years; GPT-3/300B tokens on one
   T4 → ≈ 385 years. If your run disagrees, one of us divided wrong — the script
   shows every intermediate.
2. **`oom_point_finder.py`** on Kaggle/Colab T4 — expect the measured OOM `d_model` to
   land within ~10–20% of the ledger prediction (activations and cache slack account
   for the gap; the script prints predicted-vs-hit). On CPU, prediction mode only.
3. Cross-check MFU realism in later chapters: ch26 measures MiniGPT's actual MFU on
   T4; do not expect 40% from an unfused fp32 model — the 40% figure describes tuned
   large-model training, and the calculator takes `--mfu` so you can plug in what you
   measure.

## Common mistakes

1. **Sizing GPUs for training using inference math.**
   *Symptom:* "7 B in fp16 is 14 GB, my 16 GB card is fine" → OOM at `Adam(...)`... or
   rather, at the first `step()`.
   *Cause:* forgot grads + optimizer state + activations: training ≈ 8× inference bytes.
   *Fix:* 16–20 B/param for AdamW training; run the calculator.
2. **Using parameter count as a proxy for training cost without tokens.**
   *Symptom:* "70 B is 10× the cost of 7 B."
   *Cause:* C = 6ND and D also grows (Chinchilla D = 20N) → cost is 100×, quadratic.
   *Fix:* always compute 6ND, never 6N.
3. **Assuming MFU ≈ 100% when planning.**
   *Symptom:* project plan misses by 2.5×.
   *Cause:* peak FLOPs are marketing numbers for dense matmul; real training achieves
   30–50% after attention, memory-bound ops, and communication.
   *Fix:* plan at 40%, measure, revise.
4. **Expecting linear speedup from GPU count alone.**
   *Symptom:* 8 GPUs give 5×; buying 16 gives 6×.
   *Cause:* Amdahl — the serial fraction (non-overlapped comm, data loading) caps you.
   *Fix:* measure the serial fraction (profiler, ch24); attack it with overlap (ch09),
   not with more GPUs.
5. **Quoting Amdahl to argue big clusters are pointless.**
   *Symptom:* "10% serial → 10 GPUs max, ever."
   *Cause:* Amdahl holds the problem fixed; training workloads scale with the cluster.
   *Fix:* Gustafson's framing — but respect the critical-batch-size limit.
6. **Comparing GPUs by memory only (or FLOPs only).**
   *Symptom:* "A100-80GB and H100-80GB are the same for us — same memory."
   *Cause:* walls are independent; H100 has ~3× the BF16 FLOPs and ~1.7× the HBM
   bandwidth — it moves wall 2, not wall 1.
   *Fix:* check all three walls against your job; the binding constraint differs by job.

## Limitations & outlook

- The 16–20 B/param planning number assumes Adam-family optimizers; 8-bit optimizers,
  Adafactor, and offloading change the constant (recompute it — the *method* survives).
- C = 6ND ignores the attention T² term; at very long context (ch13) it no longer can
  be ignored, and the calculator under-counts.
- MFU = 40% is an assumption we flag everywhere it is used; your MiniGPT-on-T4 MFU
  (measured in ch26) will be far lower, and frontier labs squeeze above 50%.
- The α + N/β preview treated communication as one link; real clusters are hierarchies
  (NVLink inside a node, InfiniBand between) — chapters 07 and 28.
- We established *that* you need many GPUs. *How* they cooperate — processes, ranks,
  rendezvous, NCCL — begins in chapter 06, and the exchange rate of the communication
  toll is measured in chapter 07.

## Exercises

1. **(Easy)** GPT-2 1.5 B, fp32 Adam. Static memory in GB? Does it train (statically)
   on an A100-40GB? On an A100-80GB? Now add ~10 GB of activations — what changes?
2. **(Easy)** You have one H100 (1979 TFLOP/s bf16-with-sparsity marketing number;
   use 989 TFLOP/s dense). Chinchilla-train a 1 B model at 40% MFU. Days?
3. **(Medium)** Derive the GPU count needed to Chinchilla-train 70 B in 30 days on
   A100s at 40% MFU, ignoring communication. Then re-do at 30% MFU to see what a 10-pt
   MFU drop costs in GPUs.
4. **(Medium)** A step is 250 ms: 200 ms perfectly parallel compute, 50 ms
   non-overlappable AllReduce that stays 50 ms at any scale (latency-bound). Plot/derive
   speedup vs p by Amdahl. What's the ceiling? At what p have you reached 90% of it?
5. **(Medium)** Rewrite exercise 4 in Gustafson's frame: at each p you also grow the
   per-step token count so compute stays 200 ms per device. What is the scaled speedup
   at p = 64, and what assumption about batch size does it hide?
6. **(Hard)** Data parallelism sends 2·(p−1)/p·G bytes per rank per step (ring
   AllReduce, G = gradient bytes). For MiniGPT-default (5.3 M params, fp32 grads) on a
   16 GB/s effective PCIe link, α = 30 μs per hop: at what per-step compute time does
   communication reach 10% overhead at p = 2? Show that the answer flips between
   "irrelevant" and "dominant" as the model grows to 124 M params.
7. **(Research)** Read the Chinchilla paper's §3. The D ≈ 20N rule minimizes loss for
   fixed *training* compute. Derive qualitatively how the optimum shifts when the
   objective includes inference cost over the model's lifetime (the "LLaMA regime"),
   and what that does to the compute wall.

## Solutions

<details>
<summary>Solutions to exercises</summary>

1. 1.5e9 × 16 B = 24 GB static. A100-40GB: 24 GB static fits, but +10 GB activations
   = 34 GB — dangerously tight after workspace/fragmentation; expect OOM at real batch
   sizes without checkpointing. A100-80GB: comfortable. Illustrates: the wall is not a
   line, it's a zone whose width is the activation account.
2. C = 6 × 1e9 × 2e10 = 1.2e20. Rate = 9.89e14 × 0.4 ≈ 3.96e14 FLOP/s.
   t = 1.2e20/3.96e14 ≈ 3.0e5 s ≈ 3.5 days.
3. C = 6 × 7e10 × 1.4e12 = 5.88e23. Per-GPU rate at 40%: 1.25e14. 30 days = 2.59e6 s.
   p = 5.88e23 / (1.25e14 × 2.59e6) ≈ 1,817 GPUs. At 30% MFU: rate 9.36e13 →
   p ≈ 2,423. The 10-point MFU drop costs ~600 A100s — why MFU engineering is a job title.
4. T(p) = 0.05 + 0.2/p (in seconds). S(p) = 0.25/(0.05 + 0.2/p). Ceiling = 0.25/0.05 = 5×
   (serial fraction s = 0.2). 90% of ceiling = 4.5× → 0.25/(0.05+0.2/p) = 4.5 →
   0.2/p = 0.00556 → p = 36. Thirty-six GPUs to get 4.5×; the next thousand get you 0.5×.
5. Scaled: each device still computes 200 ms of *new* tokens; total useful work per
   250 ms wall-clock = p × 200 ms vs serial 200 ms + 50 ms overhead once.
   S_scaled(64) = 0.05/0.25 + 64 × 0.2/0.25 ≈ 0.2 + 51.2 ≈ 51.4×. Hidden assumption:
   a 64× larger global batch trains as usefully per token — false beyond the critical
   batch size.
6. G = 5.32e6 × 4 ≈ 21.3 MB. p = 2: bytes/rank = 2×(1/2)×21.3 MB = 21.3 MB. Time ≈
   α-terms (2 hops × 30 μs ≈ 0.06 ms) + 21.3e6/16e9 ≈ 1.33 ms → ≈ 1.4 ms. 10% overhead
   → compute ≥ 14 ms/step: MiniGPT steps this small are easily faster than 14 ms on a
   T4 at small batch — comm can dominate. At 124 M params: G = 496 MB → ≈ 31 ms comm
   → needs ≥ 310 ms compute per step to stay under 10% — now the *model* is bigger so
   steps are slower too; the race between G/β and compute/step is exactly what ch08
   measures. (Numbers here are derivations from the stated α, β.)
7. Inference cost ∝ 2N per token served; lifetime tokens served S adds 2NS to the
   objective. Minimizing 6ND_train + 2NS at fixed loss pushes toward smaller N,
   larger D (overtraining) — LLaMA-style. Compute wall worsens: D/N of 10³–10⁴ makes
   C = 6ND grow far beyond 120N², i.e. more total FLOPs spent to buy a cheaper-to-serve
   model.

</details>

## Interview questions

1. Name the three walls of single-GPU training and the cure for each.
2. Why does a 7 B model that runs inference on a laptop need >100 GB to train?
3. Derive C = 6ND. Where do the 2 and the 4 come from?
4. What is Chinchilla-optimal token count, and what does it do to the scaling of
   training cost with model size?
5. What is MFU, what values are realistic, and why is peak FLOPs unreachable?
6. Estimate GPU-years for a 70 B Chinchilla run on A100s. (Interviewers love watching
   you do 6·7e10·1.4e12 under pressure.)
7. State and derive Amdahl's law. Your training step has a fixed 20 ms serial tail —
   what's your maximum speedup if the parallel part is 180 ms on one GPU?
8. How does Gustafson's law answer Amdahl, and which regime does LLM pre-training
   occupy? What limits Gustafson's escape?
9. Why does escaping the memory wall necessarily increase communication? Give the
   α + N/β decomposition and one example of each regime.
10. You have 16 GB GPUs and must train a 13 B model. Which walls do you hit, in what
    order, and which chapters of this course fix them?

## Summary

- Three walls: memory (doesn't fit), compute (doesn't finish), communication (the toll
  every escape pays).
- Training memory ≈ 16–20 B/param (ch04's ledger + activations): 7 B → ≥ 112 GB —
  no single GPU trains it naively.
- Inference is 2 N bytes; training is ~8× that. Never size training with inference math.
- C = 6·N·D; Chinchilla D ≈ 20 N ⇒ C = 120 N² — training cost is quadratic in model size.
- time = C / (peak × MFU), MFU ≈ 40% when tuned; 7 B ≈ 1.5 A100-years, GPT-3 ≈
  centuries on a T4 — all derivable on a napkin.
- Communication cost preview: T = α + N/β; latency-bound small messages,
  bandwidth-bound large ones; the link (NVLink vs PCIe) sets β.
- Amdahl: S(∞) = 1/s; 10% serial caps you at 10× regardless of budget.
- Gustafson: grow the problem with the cluster → linear scaled speedup; LLM training
  lives here, bounded by critical batch size.
- Parallelism converts memory/compute problems into communication problems; the course
  is about the exchange rate.
- Tools: `training_calculator.py` (napkin, automated) and `oom_point_finder.py`
  (the wall, experienced).

## References

See `references/ch05.md`. Key items: Kaplan et al. 2020 and Hoffmann et al. 2022
(scaling laws / Chinchilla); Amdahl 1967; Gustafson 1988; McCandlish et al. 2018
(critical batch size); Brown et al. 2020 (GPT-3, the cluster that the arithmetic
demands
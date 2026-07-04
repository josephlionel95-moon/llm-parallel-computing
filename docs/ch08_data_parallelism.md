# Chapter 08 — Data Parallelism, Built by Hand

> **Difficulty:** 🟡 Medium · **Study time:** 4–6 h · **Requires:** chapters 04, 06, 07
> **Notebook:** notebooks/ch08_data_parallelism.ipynb · **Experiments:** experiments/ch08/

## Learning objectives

You will be able to...

- state data parallelism precisely: what is replicated, what is sharded, what is synchronized, and when;
- *prove* that gradient averaging makes a p-worker DP step mathematically identical to a single-worker step on the concatenated batch — and state the exact caveats to "identical";
- hand-implement the three DP primitives: parameter broadcast, epoch-consistent index sharding (a from-scratch `DistributedSampler`), and post-backward gradient averaging;
- explain why `set_epoch` exists and what silently breaks without it;
- apply the linear LR scaling rule + warmup and say where it comes from and when it fails;
- account for DP's memory (p full replicas; optimizer states dominate) and communication (∝ model size, not data size);
- measure the communication fraction of a DP step and articulate why naive DP wastes the backward window;
- name DP's three ceilings and which later chapter lifts each.

## Intuition

You have one model and too much data per unit of patience. The most obvious
parallelism in machine learning: give p workers *identical copies* of the
model, feed each a *different* slice of every batch, and make them agree
afterwards. Each worker does exactly what a single GPU would do; the only new
ingredient is the agreement step.

The analogy: p graders share a stack of exams, each computes the average score
of their pile, then they average their averages (piles are equal-sized, so
this is the true mean) and all write the *same* number in the gradebook. No
grader needs to see another's exams — they only exchange one number per
question. That's DP: gradients are exchanged, data never is (which is also why
DP is the parallelism that survives privacy constraints, and why federated
learning is DP's weird cousin).

What makes DP the *first* parallelism everyone deploys: it requires zero
knowledge of the model's internals. The model is a black box that produces
gradients; DP averages them. Tensor parallel (ch10) has to slice matmuls;
pipeline parallel (ch11) has to cut the network into stages; DP just needs
`loss.backward()` to finish. Its price: every worker carries the *entire*
model, gradients, and optimizer state — a memory bill that eventually
motivates most of Part IV.

## Theory

### The mechanism

```
                 rank 0                rank 1                rank 2
              ┌──────────┐         ┌──────────┐         ┌──────────┐
   params     │ θ (copy) │         │ θ (copy) │         │ θ (copy) │   identical
              └────┬─────┘         └────┬─────┘         └────┬─────┘   (broadcast once)
   batch B     B₀ = B[0::3]         B₁ = B[1::3]         B₂ = B[2::3]  disjoint shards
                   │                    │                    │
   fwd/bwd     g₀ = ∇L(B₀;θ)        g₁ = ∇L(B₁;θ)        g₂ = ∇L(B₂;θ)  independent
                   └────────────┬───────┴──────────┬─────────┘
                                │   all_reduce / p │                    (ch07 ring)
                   ḡ = (g₀+g₁+g₂)/3   on every rank
                   │                    │                    │
   update      θ ← θ − η·ḡ         θ ← θ − η·ḡ         θ ← θ − η·ḡ     identical again
```

Invariant: **replicas are byte-identical before every forward**. Everything in
this chapter is bookkeeping to establish (broadcast), preserve (identical
updates from identical ḡ), and not accidentally violate (sampler, RNG,
buffers) that invariant.

Three pieces, all in `src/llmdist/parallel/manual_dp.py`:

1. `broadcast_parameters(model)` — once, at start. Kills the "I seeded all
   ranks identically" assumption, which dies the first time one rank makes an
   extra RNG call. Broadcasts buffers too (our causal mask is harmless; a
   BatchNorm running mean would not be).
2. `shard_indices(n, rank, world, epoch)` — the hand-rolled
   `DistributedSampler` (below).
3. `average_gradients(model, flat=True)` — after backward, before step:
   all_reduce(sum) then divide by p. `flat=True` concatenates all grads into
   one buffer first: one α instead of ~n_tensors·α (the ch07 lesson, cashed).

### The sampler is load-bearing

`shard_indices` must satisfy three invariants, each preventing a specific
silent failure:

- **Shared permutation, disjoint slices.** All ranks seed a generator with
  `seed + epoch` (never the rank!) and take `perm[rank::world]`. If ranks
  shuffled independently, shards would overlap: duplicated samples get
  double weight in ḡ, missed samples get zero — a bias nobody ever sees in
  the loss curve. The notebook demonstrates the collision count.
- **Equal sizes via padding.** `ceil(n/p)` per rank, recycling indices from
  the head. Unequal shards ⇒ unequal step counts ⇒ one rank exits its epoch
  loop while the others sit in all_reduce waiting for it — a hang with no
  error message (see Common mistakes #1).
- **Epoch in the seed.** torch's `sampler.set_epoch(e)`. Forget it and every
  epoch replays epoch 0's order: each rank sees the *same* shard in the *same*
  order forever. Training still converges, just worse — the nastiest kind of
  bug.

### Naive DP wastes the backward window

`average_gradients` cannot start until `loss.backward()` returns, so the step
timeline is strictly serial:

```
naive DP:   [── forward ──][──── backward ────][── all_reduce ──][opt]
                                                 network idle ←──┘
                            └──→ network idle during all of backward
```

But the gradient of the last layer exists near the *start* of backward. Why is
the network silent while it waits? No reason — that dead air is exactly what
DDP (ch09) reclaims by communicating buckets of gradients *during* backward.
This chapter deliberately keeps the naive schedule so you can measure the gap.

## Mathematics

### The averaging theorem

Let the global batch B be partitioned into equal shards B₁,…,B_p, |B_k| = b,
|B| = pb. Training losses are means over examples:

$$L_B(\theta) \;=\; \frac{1}{pb}\sum_{x\in B}\ell(x;\theta)
\;=\; \frac{1}{p}\sum_{k=1}^{p}\Big(\frac{1}{b}\sum_{x\in B_k}\ell(x;\theta)\Big)
\;=\; \frac{1}{p}\sum_{k=1}^{p} L_{B_k}(\theta)$$

Differentiation is **linear**, so

$$\nabla_\theta L_B \;=\; \frac{1}{p}\sum_{k=1}^{p}\nabla_\theta L_{B_k}.$$

The right-hand side is exactly what `all_reduce(SUM)/p` computes from the
per-rank gradients. Therefore a DP step

$$\theta \leftarrow \theta - \eta\cdot\frac{1}{p}\sum_k g_k$$

equals the single-process step on batch B with the same η. **Not an
approximation.** Every rank applies the identical ḡ to identical θ, so
replicas remain identical — induction from the broadcast at step 0.

Three fine-print clauses:

1. **Equal shards.** With unequal sizes m_k the true gradient is the weighted
   mean Σ(m_k/M)g_k; plain averaging over-weights small shards (exercise 5).
   Equal-size padding in the sampler is what licenses plain averaging.
2. **"Bitwise-ish", not bitwise.** The two paths sum the same numbers in
   different orders (all_reduce ring order vs one big batch's kernel-internal
   reduction order), and fp addition is not associative. Measured residual in
   `experiments/ch08/verify_grad_sync.py`: ~1e-7 for fp32 MiniGPT — bounded,
   benign, permanent. Assert with `allclose`, never `==`. Cross-run
   nondeterminism (atomics in some kernels, ch02) adds the same flavor of
   noise.
3. **Loss must be a mean.** If your loss is a *sum* over tokens, or a mean
   over tokens with per-rank differing token counts (padding!), the constants
   shift. Token-weighted averaging is the fix (exercise 5 again). This bites
   real LLM pipelines during packing/padding changes.

### Effective batch and the linear scaling rule

DP at per-rank batch b has **effective batch** B_eff = pb: the optimizer takes
one step per pb examples. Per epoch you take p× fewer steps. Goyal et al.
(2017) proposed and validated: **scale η linearly with B_eff, and warm up**.

Where linearity comes from: compare one step at batch pb and LR pη against p
steps at batch b and LR η. If the gradient is approximately constant across
those p small steps (∇L(θ_t) ≈ ∇L(θ_{t+j})), the p small updates sum to
≈ pη·ḡ — the same displacement as the single large step. The assumption fails
exactly when gradients change fast: early training (hence *warmup* — ramp η
over the first epochs) and very large B (see ceilings). SGD analysis also
gives the noise-scale view: gradient noise variance ∝ 1/B, and η ∝ B keeps
the noise *temperature* per unit of data constant — two derivations, one rule.

### Communication cost: it scales with the model

Per step, per rank (ring all_reduce, ch07): bytes = 2(p−1)/p · 4·N_params
(fp32 grads). **Independent of batch and data size.** Doubling the model
doubles comm; doubling the data changes nothing per step. Consequences:

- comm fraction = T_comm / T_step; T_compute ∝ b (bigger per-rank batches
  amortize comm — one reason large B_eff is popular);
- as models grow at fixed hardware, DP's comm fraction grows until it
  dominates — measure with `experiments/ch08/manual_dp_train.py` and predict
  with ch07's α, β fit.

### Memory: the p-fold replica bill

Each rank (ch04 ledger, Adam, fp32): params 4N + grads 4N + optimizer states
8N (m, v) = **16N bytes**, replicated p times, plus activations (∝ b, the only
term DP actually shards). For a 7B model that is ~112 GB of state *per
replica* — on every GPU. The observation "m and v are identical everywhere,
why store them p times?" is, verbatim, ZeRO stage 1 (ch15).

## Implementation

`src/llmdist/parallel/manual_dp.py` is deliberately three functions — the
whole training loop diff from single-GPU is:

```python
broadcast_parameters(model)                       # once
for step in range(steps):
    shard = shard_indices(n, rank, world, epoch=step)
    x, y = make_batch(stream, shard[:bsz], block)
    loss = model(x, y)[1]
    loss.backward()
    average_gradients(model, flat=True)           # the DP line
    opt.step(); opt.zero_grad()
```

`average_gradients` — note the flat-buffer trick and the α accounting in its
docstring:

```python
buf = torch.cat([g.reshape(-1) for g in grads])   # 1 message, not ~50
dist.all_reduce(buf, op=dist.ReduceOp.SUM)
buf /= world
# copy slices back into each p.grad
```

`shard_indices`, the sampler in four lines (invariants in the docstring):

```python
g = torch.Generator().manual_seed(seed + epoch)   # SHARED across ranks
perm = torch.randperm(n, generator=g).tolist()
perm += perm[: per_rank * world_size - n]         # pad → equal sizes
return perm[rank::world_size]                     # disjoint slices
```

Training data is `llmdist.workers.ch08.synthetic_tokens` — a fixed random
bigram transition table sampled into a 65k-token stream. Deterministic,
downloads nothing, and *learnable*: a char-level model can drive loss well
below ln(V) by learning the table, so loss curves mean something (uniform
noise would pin every curve at ln V ≈ 5.55 and teach nothing). The notebook
trains MiniGPT (2L/128d) on 2 CPU processes with this loop and plots the
loss and the per-phase time breakdown; the torchrun version is
`experiments/ch08/manual_dp_train.py`.

## Profiling & measurement

1. **Correctness first**: `experiments/ch08/verify_grad_sync.py` (and
   `tests/test_manual_ddp.py::test_dp_matches_single_process_large_batch`) —
   DP-averaged grads vs single-process concatenated-batch grads, per
   parameter. Expect agreement to ~1e-5 with fp32 and a printed worst
   residual ~1e-7-ish; the residual is reduction order, not error.
2. **Comm fraction**: `manual_dp_train.py` times forward / backward / comm /
   optimizer separately. On CPU/Gloo with the 800k-param notebook model,
   comm is a visible minority share; the point is the *methodology*. On
   Kaggle 2×T4 (PCIe), predict T_comm from ch07's fitted α, β and
   bytes = 2(p−1)/p·4N before you run — then run.
3. **Scaling**: run with `--nproc 1` and `--nproc 2` and compare tokens/s.
   Speedup = 2·T₁/(T₁ + overhead); the shortfall from 2.0× *is* your comm +
   contention bill. On shared-core CPU expect well under 2× (the two workers
   also fight for memory bandwidth — be honest about this in writeups).
4. **Flat vs per-tensor**: `--per-tensor` switches `average_gradients` to one
   all_reduce per gradient. With ~30–50 tensors mostly below N\*, the α–β
   model predicts a multiple-× larger comm phase. Verify.

## Common mistakes

1. **Unequal steps per rank → silent hang.**
   *Symptom:* training freezes at an epoch boundary; no exception; NCCL/Gloo timeout minutes later.
   *Cause:* one rank had fewer batches and left the loop; others wait in all_reduce forever.
   *Fix:* pad shards to equal size (our sampler), or `join()` contexts; audit any `if rank == 0` branch containing a collective.
2. **Per-rank RNG in the sampler.**
   *Symptom:* accuracy slightly worse than single-GPU baseline at same B_eff; no error.
   *Cause:* independent shuffles → overlapping shards → double-counted and missing samples.
   *Fix:* seed with `seed + epoch` identically on all ranks; slice disjointly.
3. **Forgetting `set_epoch` (epoch not in the seed).**
   *Symptom:* val loss plateaus early vs baseline.
   *Cause:* same shard order every epoch.
   *Fix:* `sampler.set_epoch(epoch)` / our `epoch` argument.
4. **Forgetting to average (summing only).**
   *Symptom:* immediate loss spikes or divergence when scaling from 1→p workers.
   *Cause:* all_reduce(SUM) makes the gradient p× larger; effective LR silently pη.
   *Fix:* divide by world size (or ReduceOp.AVG under NCCL); test grads against single-process (our tests do).
5. **Dropout/init RNG divergence across ranks.**
   *Symptom:* grads differ across ranks *before* averaging by more than float noise; replicas drift when someone skips a sync.
   *Cause:* different RNG streams for model-side randomness.
   *Fix:* broadcast params (init), seed dropout identically... actually per-rank dropout is *fine* (it's part of the per-shard stochastic loss) — but per-rank *init* is fatal without broadcast. Know which randomness is data-like (OK to differ) vs model-like (must match).
6. **Believing bitwise equality claims.**
   *Symptom:* "DP changed my results!" — comparisons fail at 1e-8.
   *Cause:* reduction order; nondeterministic kernels.
   *Fix:* tolerance-based comparisons; `torch.use_deterministic_algorithms(True)` if you need reproducibility, and accept the slowdown.
7. **Scaling LR without warmup.**
   *Symptom:* divergence in the first hundreds of steps at large B_eff, fine later.
   *Cause:* linear-scaling assumption (locally constant gradient) is worst at init.
   *Fix:* linear warmup of η over the first ~epochs' worth of steps (Goyal et al.).

## Limitations & outlook

- **Batch ceiling.** Past a task-dependent critical batch size (gradient noise
  scale — McCandlish et al.), larger B_eff stops buying steps: you spend more
  compute for the same trajectory; push further and generalization can degrade
  (Keskar et al.'s sharp-minima line of work). DP's width is bounded by
  optimization, not hardware.
- **Memory wall.** p complete replicas; optimizer states alone are 2× params.
  DP cannot train a model that doesn't fit on ONE device — it multiplies
  throughput, never capacity. Lifted by ZeRO (ch15) and FSDP (ch16), which
  shard state across the same data-parallel group, and by tensor/pipeline
  parallelism (ch10/11), which shard the model itself.
- **Comm ∝ model size.** The all_reduce bill grows with N while the useful
  work per step grows with b — big models on weak interconnects strangle DP.
  Mitigations: overlap (ch09), gradient accumulation (ch22, fewer syncs per
  sample), compression (ch23).
- **The serialized comm window.** Naive DP's dead air is the very next
  chapter: DDP overlaps the all_reduce with backward using hooks and buckets.

## Exercises

1. **(Easy)** Run the notebook's training cell at world 1 (bsz 16) and world 2
   (bsz 8). Overlay the loss curves. They nearly coincide — give both reasons
   they don't exactly.
2. **(Easy)** In `shard_indices`, why pad by recycling the head of the
   permutation rather than truncating the tail? Name both failure modes
   avoided.
3. **(Medium)** Measure `average_gradients(flat=False)` vs `flat=True` comm
   time on the notebook model, then *predict* the ratio from ch07's fitted α,
   β and the model's ~34 gradient tensors. How close were you?
4. **(Medium)** Derive the weighted-averaging rule for unequal shards m_k, and
   demonstrate the bias of plain averaging numerically: give rank 0 a
   batch of 2 and rank 1 a batch of 14 identical-distribution samples, compare
   against the true 16-sample gradient.
5. **(Medium)** Token-count weighting: with padded sequences and a
   mean-over-tokens loss, show that per-rank token counts must weight the
   average, and construct a 2-rank example where ignoring this changes the
   gradient by >10%.
6. **(Hard)** Implement gradient accumulation *inside* manual DP: k backward
   passes before one `average_gradients`. Show (algebra + experiment) it
   equals DP at effective batch k·p·b, and quantify the comm savings (this
   pattern becomes `no_sync` in ch09 and all of ch22).
7. **(Hard)** Using ch07's α, β for your machine and this chapter's byte
   formula, derive the model size N at which comm = 50% of the step for your
   measured backward time. Test your prediction by growing `d_model`.
8. **(Research)** Estimate the gradient noise scale (McCandlish et al., eq. 2)
   for MiniGPT on the synthetic bigram corpus using per-rank gradients you
   already have (they are i.i.d. samples of the gradient). Does the implied
   critical batch match where your loss-vs-B_eff curves start to bend?

## Solutions

<details>
<summary>Solutions to exercises</summary>

1. (a) Float reduction order differs between the all_reduce path and the
   single big-batch kernels; (b) any per-rank stochastic ops (dropout) consume
   different RNG streams. Both produce slowly diverging — not wrong —
   trajectories (chaotic sensitivity, not bias).
2. Truncation loses samples every epoch (systematic under-training on the
   tail of each permutation — a bias), and can still leave shard sizes equal
   only if p | n. Unequal shards desynchronize step counts → the all_reduce
   hang of Common mistake #1. Padding costs ≤ p−1 duplicate samples per
   epoch — negligible and unbiased on average as the permutation changes.
3. Prediction: per-tensor pays ~34 messages × 2(p−1)α vs 1 × for flat, plus
   equal bytes; with Gloo's α ~ tens of µs the comm phase should be several ×
   slower per-tensor. Measured ratio typically lands within 2× of predicted —
   the discrepancy is Python overhead per call, which is *also* a per-message
   cost and fairly folds into α.
4. True gradient = (2·g_A + 14·g_B)/16 = Σ(m_k/M)g_k. Plain average =
   (g_A+g_B)/2 weights rank 0's samples 4× too heavily
   ((1/2)/(2/16) = 4). Numerically: any asymmetric data shows it directly.
5. With mean-over-valid-tokens per rank and t_k valid tokens, the global
   per-token loss is Σt_k L_k / Σt_k ⇒ weights t_k/Σt_j. Example: rank 0 has
   10 tokens, rank 1 has 1000; plain averaging inflates rank 0's influence
   ~50×; make rank 0's tokens systematically different (e.g. all the same
   class) and the gradient direction shifts >10% — construct and measure.
6. Locally, grads over k micro-batches accumulate in `p.grad` (autograd sums);
   averaging the accumulated sum over p ranks and dividing by k gives the
   mean over k·p·b samples — linearity again. Comm: 1 all_reduce per k
   backwards ⇒ bytes/k per sample; measure the comm fraction drop with k=4.
7. T_comm ≈ 2(p−1)α_msgs + 2(p−1)/p·4N·β. Set equal to measured T_bwd(N)
   (roughly linear in N too, so solve the linear equation; on CPU where
   backward is slow, N comes out large — state your numbers). The point is
   the *method*: both sides are functions of N you can now write down.
8. B_crit ≈ tr(Σ)/|g|² where Σ is per-example gradient covariance; estimate
   tr(Σ)/b from variance across the two ranks' gradients over several steps.
   Expect the bigram task to have a *small* noise scale (easy task, strong
   signal) — bends early. Compare against measured curves; write up
   discrepancies honestly (small-b estimates of tr(Σ) are noisy).

</details>

## Interview questions

1. Prove, in three lines, that DP gradient averaging equals large-batch
   training. Then list every assumption your proof used.
2. Your DP run and your single-GPU run at the same effective batch produce
   losses differing in the 7th decimal. Bug or not? What if the 2nd decimal?
3. What exactly does `DistributedSampler.set_epoch` do, and what is the
   training-quality symptom of forgetting it?
4. Why does DP communication cost not depend on batch size? What does it
   depend on, and what does that imply for scaling models vs scaling data?
5. Two ranks, one finishes its dataloader early, training hangs with no error.
   Walk through the mechanics of the hang and two fixes.
6. State the linear LR scaling rule, derive it from the "locally constant
   gradient" assumption, and explain why warmup patches its weakest moment.
7. Where does the 16 bytes/param figure for Adam-fp32 DP training come from,
   and which of those bytes are identical across all p ranks? (Follow-up: so
   why store them p times? — anticipate ZeRO.)
8. Your loss is mean-over-tokens and sequences are padded. Why might vanilla
   gradient averaging be subtly wrong, and how do production LLM stacks fix it?
9. When would you *sum* rather than average gradients across ranks, and what
   must you change elsewhere to keep training identical?
10. Why is per-rank dropout randomness acceptable in DP but per-rank
    initialization randomness fatal?

## Summary

- DP: replicate θ, shard the batch, average gradients, step identically —
  the invariant is byte-identical replicas before every forward.
- Averaging is exact by linearity of ∇ over a mean; caveats: equal (padded)
  shards, mean-form loss, float order ⇒ allclose-not-equal (~1e-7 measured).
- Three primitives, hand-built in `manual_dp.py`: broadcast_parameters,
  shard_indices (shared epoch-seeded permutation, disjoint strided slices,
  padding), average_gradients (flat buffer = one α).
- Sampler invariants prevent three silent failures: overlap bias, epoch
  replay, and the unequal-steps hang.
- B_eff = p·b; scale η linearly with warmup (Goyal et al.); ceiling set by the
  gradient noise scale, not by hardware.
- Comm bytes/step = 2(p−1)/p·4N — proportional to *model* size; comm fraction
  measured and predictable from ch07's α, β.
- Memory: 16N bytes of state per rank, ×p replicas; optimizer states dominate
  → ZeRO (ch15).
- Naive DP serializes all_reduce after backward — the dead air DDP reclaims
  next chapter.
- Verify everything: `verify_grad_sync.py`, `manual_dp_train.py --nproc 1/2`,
  tests in `tests/test_manual_ddp.py`.

## References

- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*
  (2017) — the linear scaling rule + warmup, with the honest experiments;
  §2 is this chapter's Mathematics section in the wild.
- McCandlish et al., *An Empirical Model of Large-Batch Training* (2018) — the
  gradient noise scale: when big batches stop helping; read before buying GPUs.
- Keskar et al., *On Large-Batch Training for Deep Learning: Generalization
  Gap and Sharp Minima* (2017) — the generalization-side caveat to batch
  scaling.
- Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel
  Training* (VLDB 2020) — how the framework you use actually does DP; the
  bridge to ch09.
- Dean et al., *Large Scale Distributed Deep Networks* (2012) — DistBelief:
  asynchronous parameter-server DP, the road not taken; understand why
  synchronous all_reduce won.
- Baidu SVAIL, *Bringing HPC techniques to deep learning* (2017) — ring
  all_reduce meets gradient averaging; the historical hinge between ch07 and
  this chapter.
- You et al., *Large Batch Optimization for Deep Learning: Training BERT in 76
  minutes* (LAMB, 2019) — what replaces plain linear scaling when B_eff gets
  extreme.

# Chapter 09 — DDP Internals: Buckets, Hooks, Overlap

> **Difficulty:** 🔴 Hard · **Study time:** 6–8 h · **Requires:** chapters 02, 07, 08
> **Notebook:** notebooks/ch09_ddp_internals.ipynb · **Experiments:** experiments/ch09/

## Learning objectives

You will be able to...

- explain the single autograd fact that makes DDP possible: gradients become final one by one, roughly last-layer-first, *during* backward;
- build a working DDP from scratch (our `ManualDDP`): per-parameter hooks, reverse-order buckets, async all-reduce, finalize-before-step, `no_sync`;
- justify each design decision quantitatively: hooks (communicate early), buckets (amortize α), reverse order (fill earliest), async (overlap);
- predict and measure the bucket-size U-curve, and say why 25 MB is PyTorch's default;
- read the real `DistributedDataParallel` fluently: Reducer, first-iteration bucket rebuilding, `gradient_as_bucket_view`, `find_unused_parameters`, `static_graph`, `broadcast_buffers`, `no_sync`;
- construct an overlap timeline from measured per-bucket timestamps and interpret it;
- state precisely what DDP does *not* do (shard anything) and why that boundary defines ch15/16;
- diagnose the classic DDP failure modes (unused-parameter hang, buffer drift, double-sync waste under accumulation).

## Intuition

Chapter 08 ended with an indictment: naive DP computes for hundreds of
milliseconds while the network does nothing, then communicates while the
compute does nothing. Two expensive resources, taking turns idling.

The unlock is realizing that **backward is a conveyor belt, not a monolith**.
Autograd computes ∂L/∂θ layer by layer, from the loss backwards; the output
head's gradient is finished while the embedding's is still queued. Standing at
the end of the belt waiting for *all* items before shipping *any* is a
logistics blunder — you ship boxes as they come off the belt.

But not one item at a time: each shipment has fixed paperwork cost (ch07's α),
and gradients-per-tensor are mostly small. So you accumulate items into
**cartons** (buckets, ~25 MB), and — because the belt delivers roughly in
reverse network order — you pack cartons in reverse order so the first carton
fills while the belt is still running. Each full carton ships **asynchronously**;
you only reconcile the paperwork (wait + average + write back) at the very
end, right before the optimizer needs the result.

That's the whole design. `DistributedDataParallel` is these four ideas plus a
decade of production armor. We build the four ideas in ~180 lines
(`ManualDDP`), verify gradient-for-gradient equality with the real one, then
read the armor.

## Theory

### The enabling fact

Register `register_post_accumulate_grad_hook` on every parameter and print
during backward (the notebook does): for MiniGPT the ready order is
approximately `head/ln_f → block 3 → block 2 → block 1 → block 0 → embeddings`
— the reverse of registration order, with instructive exceptions (tied
weights finalize at their *last* use in the backward graph; see exercise 4).
The gradient of the last layer can be **in flight over PCIe while autograd is
still differentiating the first layers.** Everything else is arrangement.

### The timeline we are trying to build

```
naive DP (ch08):
  compute: [fwd────][bwd───────────────────────]                [opt]
  network:                                      [allreduce █████]
                                                 total = bwd + comm

DDP (this chapter):
  compute: [fwd────][bwd───────────────────────][w][opt]
  network:              [b0██][b1██][b2██][b3██]
                        buckets launch as their grads complete;
                        [w] = wait for the last bucket only
                        total ≈ bwd + leftover comm of the final bucket
```

Perfect overlap bound: `T_step ≥ fwd + max(bwd, comm) + opt`. DDP approaches
it when buckets are small enough to start early and large enough not to drown
in α — the U-curve, below.

### Design decision 1 — hooks: know the moment a gradient is final

`p.register_post_accumulate_grad_hook(fn)` (ch02) fires when autograd has
finished *accumulating* into `p.grad` — after all graph paths into that
parameter are summed (crucial for tied weights, and it composes with gradient
accumulation). Firing *inside* `loss.backward()` is the property we exploit:
the hook is our chance to communicate mid-backward. (The real DDP attaches to
the AccumulateGrad autograd nodes; same trigger point, C++ side.)

### Design decision 2 — buckets: amortize α

MiniGPT-small has ~34 gradient tensors; a 7B LLM has ~300. Per-tensor
all_reduce pays `n_tensors × 2(p−1)α` of pure latency — ch07's fitted Gloo α
of tens of µs makes that milliseconds per step; NCCL kernel-launch α at scale
is no kinder. Pack gradients into flat buffers of `bucket_size_mb` and pay α
per *bucket*: 34 messages → ~a handful. Bytes are unchanged (β-term identical);
this is purely an α play. PyTorch's default: 25 MB (first bucket 1 MB, to
launch something as early as possible — a subtlety our Manual version skips).

### Design decision 3 — reverse order: fill the first carton first

Buckets are built over `reversed(model.parameters())`. Why: backward produces
gradients roughly in reverse registration order, so reverse-order buckets are
*contiguous runs of the ready order* — bucket 0 (head-side params) completes
while most of backward remains, and launches immediately. Build buckets in
forward order instead and bucket 0 (embeddings) completes only at the very end
of backward: every launch is late, the timeline degenerates toward naive DP.
(Exercise 3 makes you do this and measure the damage.)

Registration order is only a *heuristic* for ready order — hence the real
DDP's first-iteration rebuild, in the Implementation section.

### Design decision 4 — async + one wait at the end

`dist.all_reduce(bucket.buffer, async_op=True)` enqueues the communication and
returns a `Work` handle immediately; autograd resumes computing earlier
layers. `finalize()` — called after backward, before `optimizer.step()` —
waits on every handle, divides by p, and copies bucket slices back into
`p.grad`. By finalize time, all buckets except possibly the last have long
finished: the waits are nearly free. (The real DDP hides finalize inside an
autograd end-of-backward callback so users never see it; we keep it explicit
and therefore *timeable*.)

### Design decision 5 — `no_sync`: don't pay k times under accumulation

Gradient accumulation runs k micro-batch backwards per optimizer step. Since
all_reduce is linear, `allreduce(Σ gᵢ) = Σ allreduce(gᵢ)` — syncing every
micro-batch buys nothing and costs k× the bytes. Inside `with model.no_sync():`
our hooks simply return (grads accumulate locally in `p.grad`); the final,
non-wrapped backward triggers the single sync, whose bucket copies naturally
carry the accumulated sums. ch22 builds on this contract.

## Mathematics

### The bucket-size U-curve, derived

Model: N bytes of gradients, B buckets of N/B each, backward duration T_bwd
produced at a uniform rate, per-bucket all_reduce cost `c(n) = 2(p−1)α +
2(p−1)/p·nβ` (ch07). Two regimes:

**Left wall (B large, buckets tiny).** Total α cost `B·2(p−1)α` grows linearly
in B; once buckets are small enough that each transfer finishes before the
next bucket fills, comm is fully hidden *except* the α floor, which is not
hideable below the per-message cost on the critical path... more simply: step
overhead ≥ B·(launch cost visible to the hot path). Empirically the left wall
is steep and unmistakable.

**Right wall (B=1).** The single bucket completes only when the *last*
gradient lands ⇒ launch at t = T_bwd ⇒ zero overlap:
`T_step = T_bwd + c(N)` — naive DP with extra copies.

**Interior.** The last bucket (embedding side) fills at ≈ T_bwd regardless, so
the un-hideable tail is ≈ `c(N/B)`; total visible comm ≈
`B·ε + c(N/B)` where ε is per-bucket launch overhead. Minimizing
`B·ε + 2(p−1)/p·(N/B)β` over B gives `B* = √(2(p−1)Nβ/(pε))` — a square-root
law: the optimal bucket *count* grows with model size and link slowness. With
typical NCCL/GPU numbers this lands buckets in the tens-of-MB — hence 25 MB.
Don't trust the constant; trust the shape, and measure
(`experiments/ch09/bucket_size_sweep.py`).

### What overlap is worth

Let comm fraction under naive DP be φ = T_comm/(T_bwd + T_comm) (ignoring
fwd/opt). Perfect overlap gives speedup `1/(1−φ)` if T_comm ≤ T_bwd (comm
fully hidden), else bounded by T_comm becoming the critical path — at which
point *no scheduling trick helps*; you need fewer bytes (ch20 bf16 grads,
ch23 compression) or more bandwidth. Overlap is a one-time ~φ-sized win;
everything past it is bytes and links.

## Implementation

### ManualDDP (`src/llmdist/parallel/manual_ddp.py`)

The five decisions, in code. Construction: broadcast params+buffers (ch08's
invariant), then bucket in reverse:

```python
for p in reversed(params):
    group.append(p); group_bytes += p.numel() * p.element_size()
    if group_bytes >= cap:
        self._seal_bucket(group); group, group_bytes = [], 0
```

The hot path is the hook — copy into the bucket slot, count down, launch when
full:

```python
def _hook(self, param):
    if not self.require_backward_grad_sync:      # no_sync()
        return
    bucket, i = self._param_slot[id(param)]      # id(): tensors break dict-by-value
    off = bucket.offsets[i]
    bucket.buffer[off:off + param.numel()].copy_(param.grad.reshape(-1))
    bucket.pending -= 1
    if bucket.pending == 0:
        bucket.t_launch = time.perf_counter()    # timeline evidence
        bucket.work = dist.all_reduce(bucket.buffer, ReduceOp.SUM, async_op=True)
```

And the reconciliation:

```python
def finalize(self):                              # after backward, before step
    for b in self.buckets:
        b.work.wait()                            # usually already done
        b.buffer /= self.world_size
        for p, off in zip(b.params, b.offsets):
            p.grad.copy_(b.buffer[off:off + p.numel()].view_as(p))
```

If a bucket never launched, finalize raises with a message pointing at unused
parameters — the exact failure `find_unused_parameters` exists for.
`bucket_events` records (index, t_launch, t_done) per step;
`experiments/ch09/overlap_timeline.py` turns them into the Gantt chart via
`llmdist.viz.plots.timeline`. Correctness is enforced in
`tests/test_manual_ddp.py`: grads and 3-step parameter trajectories must match
`torch.nn.parallel.DistributedDataParallel` to 1e-6.

### The real DDP, feature by feature

With ManualDDP internalized, the production one reads as a checklist:

- **Reducer (C++).** Our hook/bucket/launch logic, off the Python hot path,
  with per-bucket `GradBucket` objects exposed to comm hooks
  (`register_comm_hook` lets you replace the all_reduce — fp16 compression,
  PowerSGD — ch23).
- **First-iteration bucket order discovery.** DDP buckets by reverse
  registration order initially, *records the actual ready order* during the
  first backward, then rebuilds buckets to match observed reality (tied
  weights, branches, and exotic graphs break the heuristic). Our version
  skips this; exercise 4 measures what it's worth.
- **`gradient_as_bucket_view=True`.** Makes `p.grad` a *view* into the bucket
  buffer: deletes the copy-in/copy-out and one full gradient copy of memory
  (params_mb worth). Off by default only for backward-compat edge cases.
- **`find_unused_parameters=True`.** After forward, DDP traverses the autograd
  graph from the outputs and marks parameters not participating as "ready"
  with their existing grad — so buckets still fill and nobody hangs. Cost: a
  graph traversal *every iteration* plus lost fusion; the documented advice is
  to leave it off unless your model genuinely has data-dependent unused
  params. Our ManualDDP's informative RuntimeError is the same problem's
  honest version.
- **`static_graph=True`.** The opposite promise: the graph never changes
  across iterations. DDP can then cache its bookkeeping, support multiple
  backward-through-graph patterns, and combine better with activation
  checkpointing (ch21).
- **`broadcast_buffers=True` (default).** Re-broadcasts buffers from rank 0
  every forward, because buffers *mutate during forward* (BatchNorm running
  stats) and replicas would silently diverge — each rank computing a
  different function. Transformers with LayerNorm have no mutating buffers;
  our MiniGPT's causal mask is static, so this is a no-op cost here.
- **`no_sync()`.** Same contract as ours: suppress hooks' reduction, let
  grads accumulate, sync on the first non-wrapped backward.
- **Join / uneven inputs (`model.join()`)** — the principled fix for ch08's
  unequal-shard hang, shadowing missing collectives for early-finishing ranks.

What DDP does **not** do: shard. Parameters, gradients, optimizer states —
all still fully replicated on every rank (ch08's 16N bytes). DDP is a
*scheduling* optimization on DP, not a memory one. Delete-the-redundancy is
ZeRO (ch15) and FSDP (ch16), which — equipped with this chapter — you can now
predict: replace the bucket all_reduce with reduce_scatter, add an all_gather
of parameters before each layer, overlap both with compute using exactly the
hook machinery you just built.

## Profiling & measurement

1. **`ddp_comparison.py`** — manual DP vs ManualDDP vs torch DDP, per-phase
   (fwd / bwd(+overlapped comm) / sync-wait) at 2 processes. Read it as: naive
   DP's `sync` column ≈ DDP rows' *extra* time inside `bwd` + their small
   `sync` — the same bytes, relocated under compute. On CPU/Gloo the win is
   partial (comm burns the same cores that compute); on Kaggle 2×T4 + NCCL the
   separation is clean. Also asserts grad equality (schedule must never change
   math).
2. **`bucket_size_sweep.py`** — step time across bucket sizes
   (0.1→100 MB; 100 MB ⇒ single bucket for the default ~40 MB-grad model).
   Expect the U: steep left wall (α), gentle right rise (lost overlap; flatter
   on CPU). Report the measured bottom, compare with 25 MB, explain the gap
   with the √-law.
3. **`overlap_timeline.py`** — the Gantt chart from real timestamps: bucket
   launch times (recorded in the hook, mid-backward — exact) and wait-return
   times (upper bound on completion; the transport's interior is invisible
   from Python — say so in any writeup). The evidentiary core: launch marks
   *inside* the backward bar.
4. On NCCL, cross-check with `torch.profiler` (ch24 does this properly):
   `ncclAllReduce` kernels on the comm stream overlapping backward kernels on
   the compute stream — the same picture, from the driver's viewpoint.

## Common mistakes

1. **A parameter gets no gradient → hang (or our RuntimeError).**
   *Symptom:* first iteration blocks in backward/finalize; NCCL timeout after minutes.
   *Cause:* conditional branch skipped a head; its bucket never fills; other ranks' all_reduce waits forever.
   *Fix:* `find_unused_parameters=True` (accept the per-step traversal cost) or make the graph static; with ManualDDP, read the exception — it names the stragglers.
2. **Calling the optimizer before finalize (ManualDDP) / before backward returns (any DDP).**
   *Symptom:* loss noisier than single-GPU; replicas drift; eventual divergence.
   *Cause:* stepping on un-synchronized (local) gradients.
   *Fix:* the contract is backward → finalize → step; with torch DDP the wait is inside backward's callback — never bypass it with manual grad surgery between backward and step.
3. **Syncing every micro-batch under gradient accumulation.**
   *Symptom:* accumulation barely faster than stepping every micro-batch.
   *Cause:* k all_reduces where 1 suffices (linearity!).
   *Fix:* `no_sync()` on the first k−1 micro-batches; verify comm counts (our tests do).
4. **Buffer drift with BatchNorm-style models.**
   *Symptom:* eval metrics differ per rank; training fine.
   *Cause:* `broadcast_buffers=False` (or a manual DDP without buffer sync) while buffers mutate in forward.
   *Fix:* keep `broadcast_buffers=True`, or use SyncBatchNorm, or use LayerNorm and stop having mutating buffers.
5. **Mutating `p.grad` between backward and step with `gradient_as_bucket_view=True`.**
   *Symptom:* "gradient clipping changed my other parameters' grads!?"
   *Cause:* grads are views into shared bucket storage; in-place ops on a flat slice can alias neighbours' memory (and DDP's bookkeeping).
   *Fix:* use the sanctioned APIs (`clip_grad_norm_` is fine — it's per-tensor), and understand what is a view of what before in-place surgery.
6. **Wrapping, then loading a checkpoint into the inner module on rank 0 only.**
   *Symptom:* loss jumps at resume; replicas disagree.
   *Cause:* DDP broadcast params at *construction*; your later load desynchronized them.
   *Fix:* load before wrapping, or load identically on all ranks, or re-broadcast.
7. **Benchmarking DDP's first iterations.**
   *Symptom:* "DDP is slower than naive DP" on a 3-step benchmark.
   *Cause:* iteration 1 includes bucket rebuilding, allocator warmup, NCCL communicator setup.
   *Fix:* warm up ≥2 steps before timing (all our experiments do).

## Limitations & outlook

- **Overlap only hides comm ≤ backward time.** When 2(p−1)/p·4N·β outgrows
  T_bwd (huge model, thin pipe, small per-rank batch), the network is the
  critical path and DDP has nothing left to schedule. Next levers: send fewer
  bytes (bf16 gradient reduction — ch20; PowerSGD/quantized comm hooks —
  ch23) or restructure the parallelism (ch10+).
- **Still p full replicas.** DDP optimizes time, not memory: 16N bytes of
  state per rank stands. ZeRO-1/2/3 (ch15) shard optimizer/grads/params over
  the DP group; FSDP (ch16) rebuilds DDP's overlap machinery around
  reduce_scatter + all_gather. You have now built every conceptual part they
  reuse.
- **Bucket heuristics are static.** Fixed-size buckets misfit heterogeneous
  layers (one giant embedding vs many small LNs); observed-order rebuild helps
  order, not sizing. Compiled autograd / CUDA-graph-era stacks (ch27) revisit
  this with whole-graph knowledge.
- **Python-visible timestamps bound what we can claim.** Our timeline's
  launch times are exact; completion times are wait-returns. ch24's profiler
  traces close that gap with stream-level truth.

## Exercises

1. **(Easy)** Print the grad-ready order for MiniGPT (the notebook's hook
   cell) and identify: which parameter is ready first? last? Where does the
   tied embedding/head weight appear, and why there?
2. **(Easy)** In ManualDDP, why is the division by world_size done on the
   bucket buffer instead of dividing each `p.grad` before the all_reduce?
   Give the math answer and the kernel-count answer.
3. **(Medium)** Add a `reverse: bool = True` flag to ManualDDP's bucket
   builder; measure the timeline and step time with forward-order buckets.
   Predict the qualitative change first; quantify the damage.
4. **(Medium)** Instrument ManualDDP to record the *observed* ready order and
   rebuild buckets from it after iteration 1 (the real DDP's trick). Does
   MiniGPT's step time change? Construct a model where it clearly does
   (hint: register parameters in scrambled order, or add heavy weight tying).
5. **(Hard)** Implement `gradient_as_bucket_view` in ManualDDP: pre-assign
   `p.grad` as a view into the bucket buffer before backward so autograd
   accumulates in place; delete both copies. Verify grads still match torch
   DDP, and measure the memory saving with `llmdist.profiling.memory.track`.
6. **(Hard)** Write a comm-hook-style compression for ManualDDP: cast bucket
   buffers to bf16 before all_reduce, upcast after, keeping an fp32 error
   accumulator per bucket (error feedback). Measure bytes saved vs loss-curve
   impact over 200 steps on the synthetic corpus.
7. **(Hard)** Derive the ideal step time `fwd + max(bwd, comm) + opt` from
   your measured naive-DP breakdown, then compute ManualDDP's *overlap
   efficiency* = (naive − ddp)/(naive − ideal). What eats the remainder?
   (Instrument: last-bucket wait, hook copy time.)
8. **(Research)** Read the PowerSGD paper (Vogels et al., 2019) and the DDP
   comm-hook docs. For a 1 GB-gradient model on 8×PCIe-GPUs, estimate the
   rank-r needed for PowerSGD to move less than 10% of the bytes while keeping
   the accuracy claims of the paper. Where does the orthogonalization cost
   land in your α–β budget?

## Solutions

<details>
<summary>Solutions to exercises</summary>

1. First: the deepest-in-backward parameters — `ln_f`/last block's mlp.proj
   region (the head's *matmul* contributes to the tied weight but that
   weight's grad only finalizes after its embedding-side contribution too).
   Last: `pos_emb` / `tok_emb`-tied weight — the tied Parameter has edges from
   both the head matmul (early in backward) and the embedding lookup (last in
   backward); AccumulateGrad fires once, at the later one. That is also the
   answer to "where": with weight tying the big vocab matrix is ready *last*,
   which is why it should live in the last-filled bucket — and why
   registration-order heuristics need the observed-order rebuild.
2. Math: identical by linearity — divide before or after, same ḡ. Kernels:
   dividing each p.grad = one kernel per parameter (~34); dividing buckets =
   one kernel per bucket (~4); also summing p fp32 values then dividing once
   rounds slightly better than dividing first. (Under NCCL, ReduceOp.AVG can
   fold it into the reduction itself.)
3. Prediction: bucket 0 now holds embeddings = last grads ⇒ every launch
   shifts to the end of backward ⇒ timeline collapses toward naive DP; step
   time rises by ≈ the previously-hidden comm. Measured on CPU the effect is
   visible but compressed (shared cores); under NCCL it is dramatic.
4. For plain MiniGPT the observed order ≈ reverse registration except the
   tied weight (see 1), so rebuilding changes little. Scrambled registration
   (e.g. `self.blocks = nn.ModuleList(reversed([...]))` tricks, or params
   created in random order) makes reverse-registration buckets interleave
   ready times, delaying launches; rebuild restores monotone fills and
   recovers the loss. This is precisely why the real DDP rebuilds.
5. Key steps: allocate buffers at init; `p.grad = buffer[off:off+n].view_as(p)`
   before each backward (autograd accumulates into views in place); hook now
   only decrements pending; finalize skips copy-back (grads *are* the
   buffer). Memory: −1× params_mb (no separate grad storage beyond buffers) —
   verify with the tracker; grads still match torch DDP within tolerance.
6. Per bucket: `e += fp32(buf); buf16 = bf16(e); e -= fp32(buf16);
   allreduce(buf16); buf = fp32(buf16)/p`. Bytes halve. On this easy task the
   loss curve is typically indistinguishable; the point is the machinery and
   the error-feedback accumulator, which is what makes aggressive compression
   safe in the literature.
7. Overlap efficiency on CPU/Gloo commonly lands well below 1: remainder =
   (a) last-bucket tail (structurally unhideable — embeddings are both the
   biggest tensor and the last ready: weight tying strikes again), (b) hook
   copy_ time, (c) CPU core contention between Gloo threads and autograd.
   Listing those three, with measurements, is the exercise's point.
8. Sketch: PowerSGD rank r moves ~2·r·(m+n) floats per m×n matrix vs m·n;
   for transformer matrices r≈4–32 hits <10% easily. Orthogonalization is a
   few small GEMMs — negligible β, but it adds kernel launches (α) per matrix
   per step; batching them is the implementation art. Error feedback is
   mandatory for the accuracy claims.

</details>

## Interview questions

1. Explain how DDP overlaps communication with computation. What property of
   autograd makes this possible, and what API surfaces it?
2. Why does DDP bucket gradients instead of all-reducing per-parameter or
   all-reducing one flat tensor? Derive both failure modes.
3. Why are buckets formed over parameters in *reverse* registration order?
   What does the real DDP do when that heuristic is wrong, and when is it?
4. You enable `find_unused_parameters=True` "to be safe". What does it cost,
   mechanically, per iteration? When is it actually required?
5. A DDP job hangs on iteration 1 with no error and times out after 30
   minutes. Give the two most likely causes and how you'd confirm each in
   under five minutes.
6. What does `no_sync()` do, exactly — and why is gradient accumulation
   without it a factor-k waste? Why is it mathematically safe?
7. `gradient_as_bucket_view=True`: what memory does it save, what does it
   change about `p.grad`'s identity, and what user code can it break?
8. Sketch the step-time lower bound for DP with overlap. Your comm takes
   1.5× your backward. What are your options, in order of engineering cost?
9. torch DDP and your from-scratch DDP produce gradients differing by 2e-7.
   Acceptable? Why is exact equality unreachable, and what tolerance policy
   would you set in CI?
10. DDP is sometimes described as "ZeRO stage 0". Unpack that: what is
    replicated, what would each ZeRO stage shard, and which parts of DDP's
    machinery survive into FSDP?
11. Why must `broadcast_buffers` default to True for BatchNorm models but is
    irrelevant for a LayerNorm-only transformer?
12. Your profiler shows all NCCL kernels executing *after* the last backward
    kernel despite using DDP. List three configuration/code causes.

## Summary

- Autograd delivers gradients progressively, roughly last-layer-first, during
  backward — DDP is the systematic exploitation of that fact.
- ManualDDP = hooks (know when) + reverse-order ~25 MB buckets (amortize α,
  fill earliest) + `async_op=True` (overlap) + finalize (wait/average/write
  back before step) + `no_sync` (one sync per accumulation cycle).
- Every decision is quantitative: buckets fight α, reverse order buys launch
  time, async buys max(bwd, comm) instead of bwd + comm.
- Bucket size is a U-curve — α wall left, lost-overlap wall right; optimum
  follows a √-law; PyTorch's 25 MB is a sane default, not a constant of
  nature. Measure with `bucket_size_sweep.py`.
- Verified: ManualDDP's gradients equal torch DDP's (tests, 1e-6); the
  overlap is real and visible in measured launch timestamps
  (`overlap_timeline.py`).
- The real DDP adds: C++ Reducer + comm hooks, first-iteration bucket
  rebuild from observed order, gradient_as_bucket_view,
  find_unused_parameters (costly, off by default), static_graph,
  broadcast_buffers, join().
- Overlap hides comm only up to T_bwd; past that, reduce bytes or change
  parallelism.
- DDP shards nothing — 16N bytes of replicated state per rank remains; ZeRO
  (ch15) and FSDP (ch16) take this exact machinery and shard through it.

## References

- Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel
  Training* (VLDB 2020) — the DDP design paper: buckets, overlap, and the
  measurements behind the defaults; this chapter's Rosetta stone.
- PyTorch docs, *DistributedDataParallel* + *DDP Communication Hooks* +
  *DDP notes* (pytorch.org/docs) — the authoritative flag-by-flag semantics
  (find_unused_parameters, static_graph, gradient_as_bucket_view, join).
- PyTorch source, `torch/nn/parallel/distributed.py` and
  `torch/csrc/distributed/c10d/reducer.cpp` — read Reducer::autograd_hook
  after building ManualDDP; it will feel familiar.
- Vogels et al., *PowerSGD: Practical Low-Rank Gradient Compression for
  Distributed Optimization* (NeurIPS 2019) — what to do when overlap isn't
  enough; ships as a DDP comm hook.
- Sergeev & Del Balso, *Horovod* (2018) — the independent invention of
  bucketed, overlapped ring all-reduce ("tensor fusion"); convergent design
  is evidence the design is right.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models* (2020) — read §3 now, before ch15: with this chapter done,
  ZeRO's schedule reads as "DDP with the all_reduce split in half".
- Goyal et al. (2017), §5 — the original public account of overlapping
  gradient aggregation with backward at scale; history worth knowing.

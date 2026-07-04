# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter 09 — DDP Internals: Buckets, Hooks, Overlap
#
# **Hardware**: runs ANYWHERE (CPU + Gloo multi-process). Overlap gains are
# larger on 2x T4 with NCCL (`experiments/ch09/` via torchrun) because there
# compute and communication use different silicon; on CPU they share cores,
# so expect the SHAPE of the effects, not their full size.
#
# ch08 ended with an accusation: naive DP leaves the network idle during
# backward and the compute idle during the all_reduce. This notebook builds
# the fix (ManualDDP: hooks -> buckets -> async), measures the overlap with
# real timestamps, sweeps bucket sizes to find the U-curve, and then reads
# the real torch DDP with informed eyes.

# %%
import sys
sys.path.insert(0, "../src")

from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())

# %% [markdown]
# ## 1. The key fact: gradients arrive DURING backward
#
# Backward is not a monolith. Autograd computes dL/dparam roughly from the
# last layer to the first; the head's gradient exists hundreds of
# milliseconds before the embedding's. Watch, on a single process:

# %%
import torch
from llmdist.models.mini_gpt import GPTConfig, MiniGPT

cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=4, n_head=4, d_model=256)
torch.manual_seed(0)
model = MiniGPT(cfg)

order: list[str] = []
name_of = {id(p): n for n, p in model.named_parameters()}
for p in model.parameters():
    p.register_post_accumulate_grad_hook(
        lambda param: order.append(name_of[id(param)]))

x = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
model(x, x)[1].backward()

print("first 6 grads ready :", order[:6])
print("last 6 grads ready  :", order[-6:])
first_param = next(iter(name_of.values()))
print(f"\nregistration order starts at '{first_param}' — backward delivers "
      f"roughly the REVERSE. Every DDP design choice follows from this.")

# %% [markdown]
# ## 2. ManualDDP: five decisions
#
# `src/llmdist/parallel/manual_ddp.py`, decision by decision:
#
# 1. **Hooks** (`register_post_accumulate_grad_hook`): the moment a grad is
#    final, we can ship it — inside `loss.backward()`, not after.
# 2. **Buckets**: shipping ~50 tensors individually pays ~50 x 2(p-1) alpha
#    (ch07: most grad tensors sit far below N* = alpha/beta). Group into
#    ~25 MB flat buffers; ship a bucket when its LAST grad lands.
# 3. **Reverse order**: buckets are built over `reversed(parameters())`.
#    Since backward produces grads in roughly that order, bucket 0 fills
#    while most of backward still remains -> earliest possible launch.
#    (Build them in FORWARD order and bucket 0 completes only at the very
#    end of backward — everything serializes again.)
# 4. **async_op=True**: `dist.all_reduce(bucket, async_op=True)` returns a
#    handle; autograd keeps crunching earlier layers while bytes move.
# 5. **finalize()**: before `optimizer.step()`, wait on all handles, divide
#    by p, copy back into `p.grad`. Plus `no_sync()` for gradient
#    accumulation: k micro-batches, 1 all_reduce instead of k.
#
# The hot path, abridged:
#
# ```python
# def _hook(self, param):
#     if not self.require_backward_grad_sync: return       # no_sync
#     bucket, i = self._param_slot[id(param)]
#     bucket.buffer[off:off+n].copy_(param.grad.reshape(-1))
#     bucket.pending -= 1
#     if bucket.pending == 0:                              # last grad landed
#         bucket.work = dist.all_reduce(bucket.buffer, async_op=True)
# ```
#
# Correctness first: the spawned test below asserts ManualDDP's grads match
# torch DDP's exactly (same batches, same buckets philosophy), and times
# naive DP vs ManualDDP vs torch DDP.

# %%
import os
from llmdist.utils.dist import run_distributed
from llmdist.workers.ch09 import (w_compare_dp_ddp, w_bucket_sweep,
                                  w_overlap_timeline)

run_distributed(w_compare_dp_ddp, world_size=2, port=29631,
                args=(8, 1.0, "ch09_compare.pt"))

# %%
import matplotlib.pyplot as plt

comp = torch.load("ch09_compare.pt"); os.remove("ch09_compare.pt")
fig, ax = plt.subplots(figsize=(7, 3.5))
names = list(comp["step_ms"])
ax.barh(names, [comp["step_ms"][n] for n in names])
ax.set_xlabel("ms/step (2 CPU procs, Gloo)")
ax.set_title("Same math, three schedules")
fig.tight_layout(); plt.show()
print(f"grad agreement ManualDDP vs torch DDP: max diff "
      f"{comp['grad_max_diff']:.2e}")
print("On CPU, overlap gains are partly masked (comm burns CPU too);")
print("the torchrun NCCL version of this table is experiments/ch09/ddp_comparison.py.")

# %% [markdown]
# ## 3. See the overlap: measured timeline
#
# ManualDDP records `perf_counter` at every bucket launch (inside the hook,
# i.e. inside backward) and after each wait in finalize(). Plotting those
# timestamps against the backward window is the whole DDP story in one
# picture: naive DP's comm block starts where backward ends; DDP's bucket
# launches are INSIDE the backward bar.

# %%
run_distributed(w_overlap_timeline, world_size=2, port=29632,
                args=("ch09_timeline.pt",))

# %%
from llmdist.viz.plots import timeline

tl = torch.load("ch09_timeline.pt"); os.remove("ch09_timeline.pt")
timeline(tl["naive"] + tl["ddp"],
         title=f"naive DP {tl['naive_total_ms']:.0f} ms vs ManualDDP "
               f"{tl['ddp_total_ms']:.0f} ms — bucket bars start inside backward")

# %% [markdown]
# Caveat printed by the worker, repeated because it matters: a bucket bar's
# END is when `wait()` returned during finalize — an upper bound on when the
# transfer finished (the transport's interior is invisible from Python).
# The bar's START is exact: it is the hook's timestamp, mid-backward. The
# starts are the evidence of overlap.
#
# ## 4. The bucket-size U-curve
#
# Two failure modes bracket the optimum (ch07 math):
#
# * bucket -> tiny: B buckets cost B * 2(p-1) * alpha of latency; with
#   B = hundreds, milliseconds of pure overhead per step.
# * bucket -> whole model: one all_reduce that can only launch when the LAST
#   grad exists = zero overlap = naive DP with extra steps.
#
# Between them: a U. torch's default 25 MB is a good general-purpose bottom.

# %%
SWEEP = (0.05, 0.25, 1.0, 4.0, 100.0)   # MB; 100 => single bucket for this model
run_distributed(w_bucket_sweep, world_size=2, port=29633,
                args=(SWEEP, 6, "ch09_sweep.pt"))

# %%
sweep = torch.load("ch09_sweep.pt"); os.remove("ch09_sweep.pt")
xs = sorted(sweep["step_ms"])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, [sweep["step_ms"][x] for x in xs], "o-")
ax.set_xscale("log")
ax.set_xlabel("bucket size (MB)"); ax.set_ylabel("ms/step")
ax.set_title("ManualDDP bucket-size sweep (2 CPU procs) — the U-curve")
ax.grid(alpha=0.3)
plt.show()
print("left wall = latency (alpha per bucket); right wall = lost overlap.")
print("CPU/Gloo flattens the right wall somewhat; NCCL on 2 GPUs shows it clearly.")

# %% [markdown]
# ## 5. Now read the real DDP
#
# With ManualDDP in your head, `torch.nn.parallel.DistributedDataParallel`
# is recognizable — plus production armor:
#
# * **Reducer** (C++): our hook+bucket logic, off the Python hot path.
# * **First-iteration bucket order discovery**: DDP initially buckets by
#   reverse registration order (like us), OBSERVES the true grad-ready order
#   in iteration 1, then REBUILDS buckets to match it. Registration order is
#   a heuristic; the observed order is ground truth.
# * **gradient_as_bucket_view=True**: p.grad becomes a VIEW into the bucket
#   buffer — deletes our copy-in/copy-out and halves gradient memory.
# * **find_unused_parameters=True**: if some params get no grad (conditional
#   branches), their buckets never fill and everyone hangs — our ManualDDP
#   raises exactly that error. DDP's fix traverses the autograd graph from
#   the outputs EVERY step to mark absentees ready — measurable overhead;
#   only enable it if you truly have unused params.
# * **static_graph=True**: promise the graph never changes; DDP then skips
#   the bookkeeping and can optimize further (e.g. with activation ckpt).
# * **broadcast_buffers=True** (default): re-broadcasts buffers (e.g.
#   BatchNorm running stats) each forward, because buffers MUTATE during
#   forward and would drift across ranks. LayerNorm-only transformers have
#   no such buffers — one reason LN is friendlier to DP than BN.
# * **no_sync()**: same contract as ours; ch22 builds grad accumulation on it.
#
# What DDP does NOT do: shard anything. Params, grads, optimizer states are
# still fully replicated — ch15 (ZeRO) starts from this exact picture and
# starts deleting copies.

# %%
# no_sync() in action: gradient accumulation with 1 sync per 3 micro-batches.
# (Semantics test lives in tests/test_manual_ddp.py::test_no_sync_...)
import inspect
from llmdist.parallel.manual_ddp import ManualDDP

print(inspect.getsource(ManualDDP.no_sync))

# %% [markdown]
# ## Exercises
#
# 1. **(Easy)** In the timeline plot, which bucket index launches FIRST, and
#    which parameters (by name) does it contain? Confirm with
#    `model.buckets[0].params`' shapes.
# 2. **(Easy)** Why must finalize() divide by world_size AFTER wait() rather
#    than dividing each local grad before the all_reduce? (Two answers: one
#    about math, one about kernels.)
# 3. **(Medium)** Build ManualDDP with buckets in FORWARD parameter order and
#    re-measure the timeline. Predict the change before running.
# 4. **(Medium)** Add a `bucket_ready_order` recording to ManualDDP and check:
#    does the real ready order match reverse registration order exactly for
#    MiniGPT? Where does it deviate (hint: weight tying, ln_f)?
# 5. **(Hard)** Implement gradient_as_bucket_view in ManualDDP: make each
#    p.grad a view into the bucket buffer (set p.grad BEFORE backward to a
#    view slice, and rely on autograd's in-place accumulation). Measure
#    memory with llmdist.profiling.memory.
# 6. **(Hard)** Estimate the ideal step time with perfect overlap:
#    max(backward_ms, comm_ms) + fwd + opt, from your measured naive
#    breakdown. How close does ManualDDP get? What eats the difference?
# 7. **(Research)** PyTorch DDP overlaps all_reduce with backward, but NOT
#    with forward or the optimizer. Read about ZeRO's and FSDP's
#    prefetch/overlap schedules (ch16) and sketch what a "fully overlapped"
#    step would look like on a timeline.
#
# ## Solutions
#
# <details><summary>Click to expand</summary>
#
# 1. Bucket 0 — built from `reversed(parameters())`, so it holds the LAST
#    registered params: `head`/`ln_f` region (with weight tying, tok_emb's
#    tensor is shared with head, which shifts it — see ex. 4).
# 2. Math: dividing before is also correct (linearity) — no difference.
#    Kernels/precision: pre-dividing adds one extra elementwise kernel per
#    param (or per bucket anyway), and summing p small numbers then dividing
#    once has marginally better rounding than dividing then summing.
#    The real reason to divide the BUCKET: one fused op on a flat buffer.
# 3. Forward-order bucket 0 = embeddings = LAST grads to arrive; every
#    bucket's launch shifts toward the end of backward, the timeline
#    collapses toward naive DP. Step time rises measurably even on CPU.
# 4. Mostly reverse, with exceptions: tied weights (tok_emb/head share one
#    Parameter whose grad finalizes only after BOTH uses' contributions
#    accumulate — i.e., at embedding-time, late) — precisely why real DDP
#    rebuilds buckets from the OBSERVED order.
# 5. Sketch: at init, allocate buffers; for each p do
#    `p.grad = bucket.buffer[off:off+n].view_as(p)`; hooks then skip the
#    copy (grad already accumulated in place); finalize skips copy-back.
#    Memory saving: one full gradient copy (= params_mb).
# 6. ideal = fwd + max(bwd, comm) + opt. ManualDDP misses it by (a) the
#    finalize wait on the LAST bucket (can't overlap: no compute left),
#    (b) hook-side copies, (c) on CPU, comm stealing compute cores.
# 7. Key ideas to find: FSDP pre-fetches the NEXT layer's all_gather during
#    the current layer's compute in both passes; optimizer can start on
#    buckets that finished reducing (per-bucket optim); at the limit the
#    step is a pipeline where the network is never idle.
#
# </details>
#
# ## Summary
#
# * Backward yields grads roughly last-layer-first, DURING the pass — the
#   fact all of DDP exploits.
# * ManualDDP = hooks (know when) + reverse-order buckets (amortize alpha,
#   fill early) + async all_reduce (overlap) + finalize (wait, average,
#   write back) + no_sync (accumulate).
# * Verified: ManualDDP grads == torch DDP grads to ~1e-7.
# * Measured: bucket launches inside the backward window; naive DP's comm
#   strictly after it.
# * Bucket size is a U-curve: alpha wall on the left, overlap loss on the
#   right; 25 MB is torch's default bottom.
# * Real DDP adds: C++ Reducer, observed-order bucket rebuild,
#   gradient_as_bucket_view, find_unused_parameters (costly — off by
#   default), static_graph, broadcast_buffers.
# * DDP shards NOTHING — every rank still holds full params/grads/optimizer.
#   Deleting that redundancy is ZeRO/FSDP (ch15/16).

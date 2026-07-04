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
# # Chapter 08 — Data Parallelism, Built by Hand
#
# **Hardware**: runs ANYWHERE (multi-process CPU + Gloo inside this notebook).
# For real 2-GPU numbers: `torchrun --nproc_per_node=2
# experiments/ch08/manual_dp_train.py` on Kaggle.
#
# The plan: state the DP idea, PROVE the gradient-averaging rule is exact
# (not a heuristic), then train MiniGPT across processes with three functions
# we write ourselves — and measure what it costs.

# %%
import sys
sys.path.insert(0, "../src")

from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())

# %% [markdown]
# ## 1. The idea and the theorem
#
# Replicate the model on p workers. Give each worker a distinct slice of the
# batch. Average the gradients. Step identically. That's data parallelism.
#
# Why is AVERAGING exactly right? Let the global batch B split into equal
# shards B_1..B_p. Losses are means over examples, so
#
# $$L_B(\theta) = \frac{1}{|B|}\sum_{x \in B} \ell(x;\theta)
#   = \frac{1}{p}\sum_{k=1}^{p} \underbrace{\frac{1}{|B_k|}\sum_{x\in B_k}
#     \ell(x;\theta)}_{L_{B_k}(\theta)}$$
#
# Differentiation is linear, so the gradient of the mean is the mean of the
# gradients:
#
# $$\nabla L_B = \frac{1}{p}\sum_k \nabla L_{B_k}$$
#
# The right side is precisely all_reduce(local grad)/p. So a DP step with
# learning rate eta equals a single-GPU step on the whole batch B with the
# same eta. No approximation — with two caveats we will MEASURE below:
# float addition order differs (bitwise-ish, not bitwise), and the shards
# must be equal-sized (that's the sampler's job).

# %%
from llmdist.utils.dist import run_distributed
from llmdist.workers.ch08 import (w_grad_equivalence, w_dp_train,
                                  w_sampler_overlap_bug, NB_CFG)

# The theorem, numerically: 2 ranks compute shard grads + all_reduce/p,
# and compare against ONE process running the concatenated batch.
run_distributed(w_grad_equivalence, world_size=2, port=29621,
                args=("ch08_equiv.pt",))

# %%
import os
import torch

equiv = torch.load("ch08_equiv.pt"); os.remove("ch08_equiv.pt")
print(f"max |DP grad - large-batch grad| = {equiv['max_abs_diff']:.2e}")
print("~1e-7-ish, not 0.0: the two paths sum the same numbers in different")
print("orders. fp32 addition is not associative. This is why distributed")
print("'correctness' is always allclose-with-tolerance, never ==.")

# %% [markdown]
# ## 2. The data side: a sampler you can trust
#
# Each rank must see a DISJOINT, EQUAL-SIZED shard, and shards must reshuffle
# each epoch. `torch.utils.data.DistributedSampler` does this; we hand-roll
# it in `llmdist.parallel.manual_dp.shard_indices` to see every decision:
#
# ```python
# g = torch.Generator().manual_seed(seed + epoch)   # SHARED seed: all ranks
# perm = torch.randperm(n, generator=g).tolist()    # agree on the permutation
# perm += perm[:pad]                                # equal sizes: equal #steps
# return perm[rank::world_size]                     # disjoint strided slices
# ```
#
# Three invariants: (1) same permutation on all ranks — the seed depends on
# epoch, NOT rank; (2) padding, because a rank with fewer batches would exit
# the loop while others sit in all_reduce forever; (3) epoch in the seed —
# torch's `sampler.set_epoch(e)` — or every epoch replays the same order.

# %%
from llmdist.parallel.manual_dp import shard_indices

n = 10
for epoch in (0, 1):
    shards = [shard_indices(n, r, 3, epoch=epoch) for r in range(3)]
    cover = sorted(i for s in shards for i in s)
    print(f"epoch {epoch}: shards={shards} covers {len(set(cover))}/{n}, "
          f"padded to {len(cover)}")

# What goes wrong with per-rank seeds (the classic bug), demonstrated:
run_distributed(w_sampler_overlap_bug, world_size=2, port=29622)

# %% [markdown]
# ## 3. Train MiniGPT with manual DP
#
# The full loop is in `llmdist.workers.ch08.w_dp_train` — an ordinary
# training loop plus exactly three distributed lines:
#
# ```python
# broadcast_parameters(model)          # once: replicas start identical
# shard = shard_indices(n, rank, world, epoch=step)
# average_gradients(model, flat=True)  # after backward, before step
# ```
#
# Data is a synthetic bigram corpus (`synthetic_tokens`) — deterministic,
# nothing downloaded, and LEARNABLE (fixed random transition table), so the
# loss actually falls, unlike uniform noise where ln(V) is a wall.

# %%
STEPS = 40
run_distributed(w_dp_train, world_size=2, port=29623,
                args=(STEPS, 8, "ch08_train.pt"))

# %%
import matplotlib.pyplot as plt

train = torch.load("ch08_train.pt"); os.remove("ch08_train.pt")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train["losses"])
ax1.set_xlabel("step"); ax1.set_ylabel("global mean loss")
ax1.set_title("MiniGPT, manual DP, world=2 (global batch 16)")
ax1.grid(alpha=0.3)
phases = {k: v / train["steps"] for k, v in train["phase_ms"].items()}
ax2.bar(phases.keys(), phases.values())
ax2.set_ylabel("ms/step")
comm_pct = 100 * phases["comm"] / sum(phases.values())
ax2.set_title(f"per-phase time — comm = {comm_pct:.0f}% of the step")
fig.tight_layout(); plt.show()
print(f"throughput: {train['tokens_per_s']:,.0f} tokens/s (2 CPU procs)")

# %% [markdown]
# ## 4. One flat all_reduce vs one per tensor: alpha strikes
#
# `average_gradients(model, flat=True)` concatenates every gradient into ONE
# buffer before the all_reduce; `flat=False` launches one all_reduce per
# gradient tensor. Same bytes either way — the difference is pure ch07 alpha:
# this model has ~34 gradient tensors, most far smaller than the crossover
# N*, so per-tensor mode pays ~34x the per-message latency.

# %%
run_distributed(w_dp_train, world_size=2, port=29624,
                args=(12, 8, "ch08_train_pt.pt", False))   # flat=False

# %%
pt = torch.load("ch08_train_pt.pt"); os.remove("ch08_train_pt.pt")
comm_flat = train["phase_ms"]["comm"] / train["steps"]
comm_per_tensor = pt["phase_ms"]["comm"] / pt["steps"]
print(f"comm, flat (1 message/step)      : {comm_flat:8.2f} ms/step")
print(f"comm, per-tensor (~34 msgs/step) : {comm_per_tensor:8.2f} ms/step")
print(f"ratio: {comm_per_tensor / max(comm_flat, 1e-9):.1f}x — the alpha-beta "
      f"model in action; DDP's buckets (ch09) are this same idea, made")
print("compatible with overlapping communication during backward.")

# %% [markdown]
# ## 5. What that comm slice really costs
#
# Look at the bar chart: the `comm` bar is pure overhead versus single-GPU
# training, and in naive DP it is SERIALIZED — it begins only after the last
# gradient of backward exists, while the compute units idle.
#
# Two independent inefficiencies:
# 1. **Dead air**: gradients of the last layers were ready long before
#    backward finished; we could have been sending them. Fixing this —
#    overlapping communication with the tail of backward — is DDP's entire
#    reason to exist, and chapter 09 builds it.
# 2. **Volume**: the all_reduce moves 2(p-1)/p * N_grad bytes per rank
#    (ch07), where N_grad = model size — NOT batch size. Doubling data or
#    batch doesn't change it; doubling the model does. DP communication
#    scales with the MODEL.
#
# ## 6. Effective batch size, LR scaling, and DP's ceilings
#
# DP with p workers at per-rank batch b trains at effective batch B = p*b.
# The classical recipe (Goyal et al., 2017, ImageNet-in-1h): scale LR
# linearly with B and warm up for a few epochs. Intuition: over one epoch,
# k-times-larger batches mean k-times fewer steps; keeping the total update
# displacement comparable requires k-times-larger steps — valid while the
# gradient is roughly constant across those k merged steps (hence warmup,
# when it isn't).
#
# The ceilings:
# * **Batch ceiling**: past a critical batch size, gradient noise stops
#   limiting progress and bigger B buys nothing per sample (and can hurt
#   generalization — the "generalization gap" literature).
# * **Memory**: every rank stores params + grads + Adam states + activations.
#   p GPUs hold p COPIES of everything but activations — for a 7B model,
#   ~112 GB of optimizer+param state per replica set, duplicated p times.
#   ZeRO (ch15) exists to delete this duplication.
# * **Comm ~ model size**: bigger models pay more per step, same data.

# %%
# The memory duplication, concretely, with the accounting from ch04:
from llmdist.models.mini_gpt import MiniGPT
from llmdist.profiling.memory import model_memory_ledger

model = MiniGPT(NB_CFG)
ledger = model_memory_ledger(model)              # Adam: 8 B/param states
print(f"MiniGPT({NB_CFG.d_model}d x {NB_CFG.n_layer}L): "
      f"{ledger['num_params']:,} params")
for k in ("params_mb", "grads_mb", "optimizer_mb", "total_static_mb"):
    print(f"  {k:<16} {ledger[k]:9.1f} MB   x p replicas under DP")
print("Optimizer states alone are 2x params - and DP replicates them p times.")
print("That observation, taken seriously, is ZeRO (ch15).")

# %% [markdown]
# ## Exercises
#
# 1. **(Easy)** Rerun the training cell with `world_size=1` (adjust bsz to 16
#    to keep the global batch fixed) and compare loss curves. They should
#    nearly coincide — why not exactly?
# 2. **(Easy)** In `shard_indices`, why does padding recycle indices from the
#    head instead of dropping the tail? What breaks if shards are unequal?
# 3. **(Medium)** Change `w_dp_train` to use `flat=False` (one all_reduce per
#    tensor) and measure the comm phase. Explain the difference with ch07's
#    alpha-beta model and this model's ~50 small tensors.
# 4. **(Medium)** Simulate the linear scaling rule: train world=2 with
#    lr=3e-4 vs world=2 with per-rank batch halved and lr halved. Compare
#    curves per SAMPLE seen (not per step).
# 5. **(Hard)** Prove that if shards have UNEQUAL sizes m_k, the correct
#    combination is a weighted average with weights m_k / sum(m_j), and
#    plain averaging biases toward small shards' examples.
# 6. **(Hard)** Estimate, with ch07's fitted alpha/beta for your machine, the
#    largest MiniGPT (params N) for which comm < 10% of a 100 ms step at
#    p=2. Verify by resizing the model.
# 7. **(Research)** Read McCandlish et al., "An Empirical Model of Large-Batch
#    Training". Estimate the critical batch size for our synthetic-bigram
#    task from gradient noise scale, and test it.
#
# ## Solutions
#
# <details><summary>Click to expand</summary>
#
# 1. Same math, different float summation order (batch-16 kernels reduce in
#    one order; all_reduce in another), plus dropout/init RNG if seeds
#    differ. Curves diverge slowly after many steps — chaos, not error.
# 2. Dropping the tail loses samples every epoch (biased coverage). Unequal
#    shards mean unequal step counts: the small-shard rank finishes its epoch
#    and stops calling all_reduce while others block in it — deadlock (or,
#    with uneven last batches, a shape mismatch/wrong weighting).
# 3. Per-tensor: ~n_tensors * 2(p-1) alpha of pure latency; MiniGPT here has
#    ~30-50 tensors, most far below N* (ch07), so latency dominates and comm
#    time grows several-fold. Flat: one alpha, same bytes. This IS why DDP
#    buckets (ch09).
# 4. Per-sample curves should roughly overlay: (B, eta) and (B/2, eta/2) take
#    2x more steps of half the size — the linear-scaling equivalence, valid
#    in the small-noise regime; early training may differ (hence warmup).
# 5. The global loss is (1/M) sum_k m_k L_k with M = sum m_j, so
#    grad = sum_k (m_k/M) grad_k. Equal weights over-weight examples in
#    small shards by 
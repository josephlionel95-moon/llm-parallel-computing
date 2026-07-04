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
# # Chapter 07 — Collective Communication, From Scratch
#
# **Hardware**: runs ANYWHERE. All multi-process cells use CPU + Gloo via
# `torch.multiprocessing.spawn`, so semantics are identical on Colab (1 GPU),
# Kaggle (2 GPU) or your laptop. NCCL-speed measurements belong to
# `experiments/ch07/` run with `torchrun` on Kaggle 2x T4.
#
# In this notebook you will:
# 1. run every collective and SEE per-rank before/after state,
# 2. verify our p2p-only implementations against `torch.distributed`,
# 3. measure time vs message size and fit the alpha-beta cost model,
# 4. watch ring all-reduce move chunks, step by step,
# 5. race manual naive vs manual ring vs the built-in all_reduce.

# %%
import sys
sys.path.insert(0, "../src")

from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())

# %% [markdown]
# ## 1. The cast of collectives
#
# Point-to-point (`send`/`recv`) is all a network gives you. Everything else
# is choreography. The seven moves, and where LLM training uses each:
#
# | collective     | semantics                                   | used by |
# |----------------|----------------------------------------------|---------|
# | broadcast      | one rank's tensor -> all ranks               | DDP param init |
# | reduce         | sum of all ranks' tensors -> one rank        | (building block) |
# | all_reduce     | sum -> ALL ranks                             | DDP gradient sync |
# | scatter        | one rank's list -> one chunk per rank        | data/checkpoint sharding |
# | gather         | one chunk per rank -> list on one rank       | logging, checkpoint save |
# | all_gather     | every chunk -> everyone                      | ZeRO-3/FSDP params |
# | reduce_scatter | sum, then each rank keeps 1/p of it          | ZeRO-2/FSDP grads |
# | all_to_all     | rank i's j-th chunk -> rank j's i-th slot    | MoE token routing (ch14) |
#
# The two most important identities of this whole course:
#
# ```
# all_reduce = reduce_scatter + all_gather        (ring all-reduce, this chapter)
# all_reduce = reduce + broadcast                 (naive; root bottleneck)
# ```
#
# ZeRO/FSDP (ch15/16) is literally "take DDP's all_reduce and split it into
# its two halves, keeping only your shard between them".

# %%
# Every multi-process demo: run_distributed spawns fresh processes and joins.
# Spawn targets must live in an importable module (a notebook cell is
# __main__, which children can't re-import) -> they live in llmdist.workers.
from llmdist.utils.dist import run_distributed
from llmdist.workers.ch07 import (w_collectives_tour, w_verify_manual,
                                  w_collective_family_bench,
                                  w_manual_vs_torch_bench, fit_alpha_beta)

run_distributed(w_collectives_tour, world_size=2, port=29601)

# %% [markdown]
# Read the output rank by rank. Note `reduce` leaves non-destination ranks
# untouched, `scatter` gives each rank a DIFFERENT chunk (vs broadcast: same),
# and `reduce_scatter` combines both ideas: everyone contributed to every
# chunk, everyone kept one.
#
# ## 2. Built from send/recv only
#
# Open `src/llmdist/comm/manual_collectives.py`. Three things to notice:
#
# **Deadlock discipline.** A ring step where every rank does
# `send(right); recv(left)` with blocking calls can deadlock: all p ranks
# block in `send`, no `recv` is ever posted. We post `irecv` FIRST, then
# `isend`, then wait both:
#
# ```python
# def _exchange(send_buf, dst, recv_buf, src):
#     recv_req = dist.irecv(recv_buf, src)
#     send_req = dist.isend(send_buf, dst)
#     recv_req.wait(); send_req.wait()
# ```
#
# Small messages often survive `send`-first because the transport buffers
# them — so the bug appears only at large sizes, in production, at 3am.
#
# **Deterministic reduction order.** Our `reduce` accumulates in increasing
# rank order, always. Floating-point addition is not associative; fixing the
# order makes reruns reproducible (a theme that returns in ch08).
#
# **Cost annotations.** Every docstring states message count and
# bytes-on-wire — you should be able to reproduce each from the code shape.
#
# Below: correctness of both manual all_reduce variants against
# `dist.all_reduce`, on 100k floats. The assert helpers use torch's own
# collectives as the referee, so our bugs can't referee themselves.

# %%
run_distributed(w_verify_manual, world_size=2, port=29602)

# %% [markdown]
# ## 3. The alpha-beta cost model
#
# Model a message of N bytes between two ranks as
#
# $$T(N) = \alpha + \beta N$$
#
# * $\alpha$: per-MESSAGE cost — software stack, protocol handshake, NIC
#   doorbell, (on GPU) kernel launch. Microseconds. Independent of N.
# * $\beta$: per-BYTE cost, the inverse bandwidth. E.g. PCIe 3.0 x16 at
#   ~12 GB/s effective -> beta ~ 0.083 ns/B.
#
# Two regimes with a sharp crossover at $N^* = \alpha/\beta$:
# below it you pay for MESSAGES (batch them!), above it for BYTES.
# With alpha = 30 us and beta for 12 GB/s, N* ~ 360 KB — right in the range
# of real gradient tensors, which is exactly why DDP buckets exist (ch09).
#
# Let's measure T(N) for the built-in collectives over 1 KB -> 4 MB on
# Gloo/CPU (loopback, so alpha is honest, beta is memory-bandwidth-ish),
# then fit alpha and beta by least squares.

# %%
import os
import torch

SIZES = tuple(2**10 * 4**i for i in range(7))     # 1 KB ... 4 MB
BENCH_PATH = "ch07_family_bench.pt"
run_distributed(w_collective_family_bench, world_size=2, port=29603,
                args=(SIZES, 8, BENCH_PATH))
family = torch.load(BENCH_PATH); os.remove(BENCH_PATH)
print("\nloaded:", {k: f"{len(v)} sizes" for k, v in family["results"].items()})

# %%
import matplotlib.pyplot as plt

results = family["results"]
fig, ax = plt.subplots(figsize=(8, 5))
for name, series in results.items():
    ax.plot(list(series), list(series.values()), "o-", label=name)

alpha_us, beta_ns = fit_alpha_beta(list(results["all_reduce"]),
                                   list(results["all_reduce"].values()))
fit = [alpha_us * 1e-3 + beta_ns * 1e-6 * s for s in SIZES]
ax.plot(SIZES, fit, "k--", alpha=0.7, label=f"fit: {alpha_us:.0f}us + N/{1/beta_ns:.1f}GB/s")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("message size (bytes)"); ax.set_ylabel("time (ms)")
ax.set_title("Collectives vs size (2 proc, Gloo/CPU): flat = alpha regime, "
             "slope 1 = beta regime")
ax.legend(); ax.grid(alpha=0.3)
plt.show()
print(f"alpha ~ {alpha_us:.1f} us | beta ~ {beta_ns:.3f} ns/B "
      f"(~{1/max(beta_ns,1e-9):.2f} GB/s) | crossover N* ~ "
      f"{alpha_us*1e3/max(beta_ns,1e-9)/2**10:.0f} KB")

# %% [markdown]
# Read the left side of the plot: all curves are FLAT — a 1 KB and a 16 KB
# all_reduce cost the same, you're paying alpha. The right side has slope ~1
# on log-log: doubling bytes doubles time, you're paying beta. Every
# communication decision in this course is about which side of $N^*$ you're on.
#
# ## 4. Ring all-reduce, derived
#
# Naive all_reduce (reduce + broadcast through a root) moves $2(p-1)N$ bytes
# through ONE node's link: cost $2(p-1)(\alpha + N\beta)$ — gets worse per
# byte as p grows, and p-1 links sit idle.
#
# The ring: arrange p ranks in a circle, split the tensor into p chunks.
#
# **Phase 1 — reduce-scatter, p-1 steps.** Each step: send one chunk right,
# receive one from the left, ADD it to your copy. Each chunk travels the
# ring once, gathering contributions; after p-1 steps, rank r holds the
# fully-reduced chunk r (schedule: rank r sends chunk (r-1-s) at step s).
#
# **Phase 2 — all-gather, p-1 steps.** Same motion, but the receiver
# OVERWRITES instead of adds, circulating the finished chunks.
#
# Per rank: $2(p-1)$ messages of $N/p$ bytes each:
#
# $$T_{ring} = 2(p-1)\alpha + \frac{2(p-1)}{p} N \beta$$
#
# The beta term is bounded by $2N$ **no matter how many ranks** — that's why
# it is called bandwidth-optimal: any all_reduce must move $\ge \frac{p-1}{p}N$
# unique bytes out of and into every rank, and the ring meets that bound
# (times 2 for the two phases). The alpha term, though, grows LINEARLY in p:
# for small tensors a tree ($O(\log p)$ messages) wins. NCCL measures your
# topology and switches algorithm by message size — you saw the two regimes
# in the plot above.

# %%
# Watch it happen. The animation module simulates the exact schedule of our
# implementation and renders which contributions each rank's chunks contain.
sys.path.insert(0, "../animations")
from ch07_ring_allreduce import build_states, draw_state

states, captions = build_states(p=4)
fig, axes = plt.subplots(2, 4, figsize=(17, 9))
for i, ax in enumerate(axes.flat):
    if i < len(states):
        phase = ("start" if i == 0 else f"RS {i}/3" if i <= 3 else f"AG {i-3}/3")
        draw_state(ax, states[i], f"step {i} ({phase})", 4)
    else:
        ax.axis("off")
fig.suptitle("Ring all-reduce, 4 ranks: cells show which ranks' data each "
             "copy contains; 'ALL' = fully reduced")
fig.tight_layout(); plt.show()

# %% [markdown]
# Follow one column (one chunk) across panels: it accumulates contributions
# hop by hop in phase 1 (darker = more), then the finished 'ALL' version
# spreads to every row in phase 2. Six steps total = 2(p-1) with p=4.
#
# To generate the GIF version (embedded in the docs):
# `python ../animations/ch07_ring_allreduce.py`
#
# ## 5. Manual ring vs manual naive vs the real thing

# %%
RACE_PATH = "ch07_race.pt"
run_distributed(w_manual_vs_torch_bench, world_size=2, port=29604,
                args=(tuple(2**10 * 4**i for i in range(6)), 6, RACE_PATH))
race = torch.load(RACE_PATH); os.remove(RACE_PATH)

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, series in race["results"].items():
    ax.plot(list(series), list(series.values()), "o-", label=name)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("size (bytes)"); ax.set_ylabel("time (ms)")
ax.set_title("all_reduce: manual naive vs manual ring vs torch (2 proc, Gloo)")
ax.legend(); ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# What to expect (and why not to over-read it): at p=2 the ring's bandwidth
# advantage over naive is small — $2(p-1)N/p = N$ vs $2(p-1)N = 2N$ at the
# root — and our Python-level loop pays extra alpha per step, so torch wins
# overall. The lesson is in the SLOPES at large N and in the formulas; the
# constant factors belong to the tuned library. Rerun on Kaggle 2x T4 with
# `experiments/ch07/manual_vs_builtin_allreduce.py` under NCCL.
#
# ## 6. Bus bandwidth (how nccl-tests reports numbers)
#
# Naively dividing N by all_reduce time understates the hardware: the
# algorithm moved $2(p-1)/p \cdot N$ bytes per rank. nccl-tests therefore
# reports **busbw = algbw x 2(p-1)/p**, a size-independent number comparable
# to link specs. Our `llmdist.profiling.timers.bandwidth_gbps(nbytes, ms,
# factor=2*(p-1)/p)` computes exactly this — used in `experiments/ch07/`.

# %%
p = 2
n_big = max(race["results"]["torch dist.all_reduce"])
ms_big = race["results"]["torch dist.all_reduce"][n_big]
from llmdist.profiling.timers import bandwidth_gbps
print(f"largest tested all_reduce: {n_big/2**20:.0f} MB in {ms_big:.2f} ms")
print(f"  algbw = {bandwidth_gbps(n_big, ms_big):.2f} GB/s")
print(f"  busbw = {bandwidth_gbps(n_big, ms_big, factor=2*(p-1)/p):.2f} GB/s "
      f"(x{2*(p-1)/p:.1f} algorithm factor)")

# %% [markdown]
# ## Exercises
#
# 1. **(Easy)** Modify `w_collectives_tour`'s tensors so that after
#    `all_reduce` every rank holds the AVERAGE instead of the sum, without
#    changing `manual_collectives.py`.
# 2. **(Easy)** From the fitted alpha and beta, predict the time of a 64 KB
#    all_reduce; compare with the measured point. Inside 2x?
# 3. **(Medium)** Implement `all_to_all` from send/recv (rank i sends its
#    j-th chunk to rank j). What are its message count and bytes-on-wire?
#    Where does MoE (ch14) use it?
# 4. **(Medium)** Our `broadcast` is linear. Time `broadcast` vs
#    `broadcast_tree` at 4 and 8 processes for 1 KB and 16 MB messages.
#    Which wins where, and why?
# 5. **(Hard)** Derive the ring all-reduce time for p=2 and show it equals
#    the naive version's bytes at the root divided by 2. Why is p=2 the
#    worst case for showing off rings?
# 6. **(Hard)** Implement a PIPELINED ring broadcast (split N into k chunks,
#    stream them around the ring). Show its cost is
#    (p-2+k)(alpha + N/k * beta) and find the optimal k.
# 7. **(Research)** NCCL also has tree all-reduce and (on NVSwitch) direct
#    algorithms. Read the NCCL 2.4 tree announcement and explain when a
#    double binary tree beats a ring at LLM scale (hint: alpha term at
#    p = thousands).
#
# ## Solutions
#
# <details><summary>Click to expand</summary>
#
# 1. Divide by `world_size` after the call: `t /= world` on every rank
#    (all ranks hold the same sum, so all divide identically).
# 2. T(64 KB) = alpha + beta * 65536; plug your fitted numbers. It should be
#    within ~2x — the fit is dominated by the large points, and 64 KB sits
#    near the crossover where neither term is negligible.
# 3. Every rank sends p-1 chunks of N/p bytes to distinct peers:
#    p-1 messages/rank, (p-1)N/p bytes on wire per rank. MoE routes each
#    token's hidden state to the rank hosting its chosen expert: that
#    permutation IS an all_to_all, twice per MoE layer (there and back).
# 4. 1 KB: tree wins at both world sizes (ceil(log2 p) alphas vs p-1 alphas).
#    16 MB: at small p they are close; as p grows the tree's N*beta per hop
#    (log p serial hops of the FULL message) loses to pipelined transfers.
#    That is why libraries also pipeline within trees.
# 5. p=2 ring: 2 messages of N/2 = N bytes moved per rank,
#    T = 2 alpha + N beta. Naive: root receives N and sends N: also ~2 alpha
#    + 2 N beta at the root but the non-root link idles half the time. The
#    ring advantage 2(p-1)/p vs 2(p-1) is a factor p — at p=2, just 2.
# 6. Chunk c starts moving at step c and the last of k chunks arrives after
#    p-2+k steps: T = (p-2+k)(alpha + (N/k) beta). dT/dk = 0 gives
#    k* = sqrt(N beta (p-2) / alpha); with k* chunks,
#    T ~ N beta + (p-2) alpha + 2 sqrt(N beta (p-2) alpha) -> N beta for
#    large N: the same bytes-optimality trick the ring uses.
# 7. Ring alpha term ~ 2(p-1) alpha: at p = 4096 and alpha = 5 us that is
#    ~41 ms *per collective* regardless of size. A double binary tree pays
#    ~2 log2(p) alpha ~ 120 us while keeping near-full bandwidth (two
#    complementary trees each carrying half the bytes) — NCCL >= 2.4 uses it
#    for exactly this regime.
#
# </details>
#
# ## Summary
#
# * Collectives are choreography over send/recv — we built all seven, and the
#   asserts prove them equivalent to `torch.distributed`'s.
# * Cost model: T = alpha + beta*N per message; regimes split at N* = alpha/beta.
#   Measured and fitted it live.
# * all_reduce = reduce_scatter + all_gather; that decomposition IS ZeRO/FSDP.
# * Ring all_reduce: 2(p-1) steps, 2(p-1)N/p bytes/rank — bandwidth-optimal,
#   but alpha grows with p; trees win for small N / large p; NCCL switches.
# * Naive (root-based) all_reduce concentrates 2(p-1)N bytes on one link.
# * busbw = algbw * 2(p-1)/p makes all_reduce numbers comparable to link specs.
# * Gloo has no reduce_scatter; our p2p version runs anywhere regardless.
# * LLM mapping: all_reduce->DDP, reduce_scatter/all_gather->ZeRO+FSDP,
#   all_to_all->MoE, broadcast->replica init.

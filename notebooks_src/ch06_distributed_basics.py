# %% [markdown]
# # Chapter 06 — Distributed Computing Basics
#
# **Hardware:** none required. Every multi-process cell uses
# `torch.multiprocessing.spawn` with the **Gloo** backend on CPU, so the full
# programming model — ranks, rendezvous, collectives, even deadlocks — runs on
# any machine. On Kaggle 2× T4 the final section switches to real NCCL.
#
# The one mental model to install: distributed PyTorch is **N identical OS
# processes** (SPMD). Nothing is shared; identity comes from env vars; the
# rendezvous at MASTER_ADDR:MASTER_PORT turns N strangers into a numbered group.

# %%
import sys
sys.path.insert(0, "../src")

import os
import torch
from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())

# %% [markdown]
# ## 1. Processes, ranks, world size
#
# - **WORLD_SIZE** — how many processes in the job.
# - **RANK** — global id, 0..world_size-1: algorithm logic ("rank 0 checkpoints").
# - **LOCAL_RANK** — id *within one machine*: exactly one job — picking the GPU,
#   `cuda:{LOCAL_RANK}`, because CUDA indices are per-machine.
#
# `RANK = node_rank * nproc_per_node + LOCAL_RANK`. On one node they coincide,
# which is how the `cuda:{RANK}` bug survives until your first two-node job.

# %%
def rank_arithmetic(rank: int, nproc_per_node: int) -> tuple[int, int]:
    return rank // nproc_per_node, rank % nproc_per_node   # node_rank, local_rank


print("3 nodes x 4 GPUs:")
print(f"{'RANK':>6}{'node_rank':>11}{'LOCAL_RANK':>12}{'device':>10}")
for r in range(12):
    node, local = rank_arithmetic(r, 4)
    print(f"{r:>6}{node:>11}{local:>12}{'cuda:' + str(local):>10}")

# %% [markdown]
# ## 2. The rendezvous, run for real (in this notebook)
#
# `torchrun` can't launch a Jupyter cell — it forks fresh processes running a
# *script*. Inside notebooks we use `torch.multiprocessing.spawn` via our
# `llmdist.utils.dist.run_distributed`, which does by hand what torchrun does
# with agents: set MASTER_ADDR/PORT + RANK/WORLD_SIZE in each child's env, then
# every child calls `dist.init_process_group`.
#
# What init does, step by step: rank 0 **binds a TCPStore server** at
# MASTER_ADDR:MASTER_PORT (a tiny key-value server — the control plane); other
# ranks connect as clients; everyone checks in; the backend then builds its
# communicators (NCCL would exchange a 128-byte unique id through the store and
# probe GPU topology). Gradients never pass through the store.
#
# spawn constraint: the worker fn must be **picklable** — define it at module /
# notebook top level, no closures.

# %%
import torch.distributed as dist
from llmdist.utils.dist import run_distributed


def hello(rank: int, world_size: int) -> None:
    # Each line below runs in a SEPARATE PROCESS with its own memory.
    t = torch.ones(1) * (rank + 1)
    dist.all_reduce(t)                     # sum over ranks: 1+2+...+p = p(p+1)/2
    print(f"[rank {rank}/{world_size}] pid={os.getpid()} "
          f"my value was {rank + 1}, all_reduce sum = {t.item():.0f}")


run_distributed(hello, world_size=2, backend="gloo")
print("\nexpected sum for p=2: 2*3/2 = 3 — and BOTH ranks printed it: that")
print("'everyone gets the answer' property is what makes it an ALL-reduce.")

# %%
# Same program, world_size=4 — the script didn't change, only the launch. SPMD.
run_distributed(hello, world_size=4, backend="gloo", port=29501)
print("\nexpected: 4*5/2 = 10")

# %% [markdown]
# Note the `port=29501`: each `run_distributed` call creates a fresh TCPStore,
# and a lingering socket from the previous world can collide on the port —
# failure mode 'address already in use'. Distinct ports sidestep it.
#
# ## 3. Nothing is shared — prove it

# %%
GLOBAL = {"counter": 0}


def mutate_global(rank: int, world_size: int) -> None:
    GLOBAL["counter"] += 100 + rank        # each process mutates ITS OWN copy
    print(f"[rank {rank}] my GLOBAL is {GLOBAL}")


run_distributed(mutate_global, world_size=2, backend="gloo", port=29502)
print(f"\nparent process GLOBAL after both workers ran: {GLOBAL}")
print("untouched — spawn gave each worker a fresh interpreter and address space.")
print("ALL sharing must go through explicit communication (ch07's collectives).")

# %% [markdown]
# ## 4. Backends: NCCL vs Gloo vs MPI
#
# | | NCCL | Gloo | MPI |
# |---|---|---|---|
# | tensors | CUDA only | CPU (+limited CUDA) | build-dependent |
# | speed on GPU | fastest | slow (via host) | varies |
# | why it exists | training | portability/CPU | legacy HPC |
#
# Why NCCL wins on GPUs: (1) **topology-aware** — at init it maps NVLink/PCIe/
# NIC wiring and builds rings & trees that saturate every link; (2) **in-kernel
# communication** — collectives are CUDA kernels moving data P2P without a host
# round-trip, reduction arithmetic on the SMs; (3) **stream semantics** — a
# collective is a kernel on a stream, so it can overlap with compute (the
# foundation of DDP's overlap, ch09).
#
# The iron rule: **NCCL moves GPU tensors only; Gloo moves CPU tensors.**
# Mismatch = immediate error (demonstrated safely in
# `experiments/ch06/failure_modes.py --mode nccl_cpu`).

# %%
print(f"nccl available : {dist.is_nccl_available()}  (needs CUDA build + GPUs)")
print(f"gloo available : {dist.is_gloo_available()}  (always, on any CPU)")
print(f"mpi  available : {dist.is_mpi_available()}   (only if torch was built with MPI)")
print(f"\nthis machine's preferred backend per llmdist: {ENV.backend}")

# %% [markdown]
# ## 5. Device binding — the 'everyone on GPU 0' bug
#
# CUDA's per-process "current device" defaults to **0**. If four ranks on a
# node skip `torch.cuda.set_device`, all four allocate on GPU 0: it OOMs while
# GPUs 1–3 sit idle, or NCCL aborts with "Duplicate GPU detected". The fix is
# one line, before any tensor exists:
#
# ```python
# torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
# ```
#
# Our `setup_from_env()` (used by every torchrun script in this course) does
# exactly this — and falls back to CPU/Gloo when there's no GPU, which is why
# the same scripts run on Kaggle, Colab, and CI. Walk through it:

# %%
import inspect
from llmdist.utils import dist as llmdist_dist

print(inspect.getsource(llmdist_dist.setup_from_env))

# %% [markdown]
# Line by line: (1) identity from env — torchrun wrote it (KeyError here means
# you ran `python` instead of `torchrun`); (2) LOCAL_RANK falls back to rank —
# fine on one node; (3–4) device chosen by LOCAL rank and bound with
# `set_device` BEFORE the process group; (5) no GPU → gloo, gracefully;
# (6) 120 s timeout instead of NCCL's default 30 *minutes*, so your mistakes
# fail fast while learning.
#
# ## 6. torchrun anatomy
#
# ```
# torchrun --nproc_per_node=2 train.py --lr 3e-4
#   ├─ parses its own flags, passes the rest to your script
#   ├─ rendezvous (trivial single-node; c10d store for multi-node)
#   ├─ forks 2 children, each with env:
#   │    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, ...
#   ├─ each child runs: python train.py --lr 3e-4
#   └─ watches children: one dies -> tear down (or restart with --max-restarts)
# ```
#
# `--standalone` = single-node shortcut: torchrun hosts an ephemeral rendezvous
# on a free port, so you set nothing. Multi-node: same command on every node
# with its own `--node_rank`, all pointing at one `--rdzv_endpoint`.
#
# We can't run torchrun in a cell, but we can watch it set the env — each rank
# below just dumps the variables the launcher (our spawn shim) provided:

# %%
def show_env(rank: int, world_size: int) -> None:
    keys = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    vals = {k: os.environ.get(k, "<unset>") for k in keys}
    print(f"[rank {rank}] {vals}")


run_distributed(show_env, world_size=2, backend="gloo", port=29503)
print("\n(torchrun would also set LOCAL_RANK and TORCHELASTIC_* bookkeeping;")
print(" our notebook shim sets the minimum needed for env:// rendezvous.)")

# %% [markdown]
# From a terminal (or a Kaggle `!` cell), the real thing:
#
# ```bash
# torchrun --standalone --nproc_per_node=2 experiments/ch06/hello_distributed.py
# ```
#
# ## 7. Hangs, and how to read them
#
# A distributed hang = an appointment someone missed: **every rank must execute
# every collective, in the same order, with compatible shapes.** A collective
# inside `if rank == 0:` strands everyone else, silently, until timeout.
#
# Debug kit (env vars set BEFORE launch):
# - `TORCH_DISTRIBUTED_DEBUG=DETAIL` — log each collective per rank; divergence visible.
# - `NCCL_DEBUG=INFO` — topology detected, rings/trees built, transport chosen
#   (this is how you verify Kaggle's T4s have NO NVLink: no NVL lines).
# - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` / `TORCH_NCCL_BLOCKING_WAIT=1` — turn
#   silent NCCL hangs into loud errors at the timeout.
# - `py-spy dump --pid <pid>` — where each rank is stuck, the ground truth.
#
# We demonstrate a real mismatch hang — safely, with a 15 s timeout — below.
# (The full collection: `experiments/ch06/failure_modes.py`.)

# %%
import datetime
import torch.multiprocessing as mp


def mismatched_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=15))
    t = torch.ones(1)
    try:
        if rank == 0:
            print("[rank 0] skipping the all_reduce (the bug) and exiting")
        else:
            print(f"[rank {rank}] entering all_reduce and waiting for rank 0 ...")
            dist.all_reduce(t)     # rank 0 never shows up -> blocks -> timeout
            print(f"[rank {rank}] finished (should NOT happen)")
    except Exception as e:
        print(f"[rank {rank}] TIMEOUT/ERROR as predicted: {type(e).__name__}: "
              f"{str(e)[:120]}")
    finally:
        dist.destroy_process_group()


print("demonstrating failure mode: mismatched collective (15 s timeout)...")
try:
    mp.spawn(mismatched_worker, args=(2, 29504), nprocs=2, join=True)
except Exception as e:
    print(f"spawn surfaced the failure: {type(e).__name__}")
print("\nlesson: hoist collectives OUT of rank-conditional code.")

# %% [markdown]
# ## 8. Rendezvous cost: seconds, paid once
#
# Spawn + store check-in + first collective costs seconds and grows with world
# size. It's amortized over a whole job — but the FIRST collective per group is
# also where lazy communicator setup happens, so benchmarks must warm up.

# %%
import time


def one_collective(rank: int, world_size: int) -> None:
    t = torch.ones(1)
    dist.all_reduce(t)


sizes = [2, 4]
timings = []
for i, ws in enumerate(sizes):
    t0 = time.perf_counter()
    run_distributed(one_collective, world_size=ws, backend="gloo", port=29510 + i)
    dt = time.perf_counter() - t0
    timings.append(dt)
    print(f"world_size={ws}: spawn + rendezvous + all_reduce + teardown = {dt:.2f} s")

print("\nseconds per JOB, not per step — but never time the first collective as")
print("if it were steady state. Full sweep: experiments/ch06/rendezvous_timing.py")

# %% [markdown]
# ## 9. Kaggle vs Colab
#
# - **Kaggle 2× T4**: real NCCL. The T4s hang off the PCIe tree with **no
#   NVLink** — verify with `!nvidia-smi topo -m` (expect PHB/PIX, no NV#) and
#   `NCCL_DEBUG=INFO`. Bandwidth consequences measured in ch07.
# - **Colab 1× T4**: one GPU → no real multi-GPU. Everything in this notebook
#   still works via CPU+Gloo, with identical *semantics* (ranks, collectives,
#   deadlocks). Only the speeds are fiction — never quote Gloo timings.
#
# The cell below runs real NCCL if (and only if) this machine has ≥2 GPUs.

# %%
def nccl_hello(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)                    # single node: local == global
    t = torch.ones(1, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(t)
    print(f"[rank {rank}] device={torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name(rank)}) sum={t.item():.0f}")


if ENV.multi_gpu:
    run_distributed(nccl_hello, world_size=2, backend="nccl", port=29520)
else:
    print(f"only {ENV.num_gpus} GPU(s) here — on Kaggle 2x T4 this cell runs real")
    print("NCCL: two processes, two devices, one in-kernel all_reduce over PCIe.")

# %% [markdown]
# ## Exercises
#
# 1. (Easy) 4 nodes × 8 GPUs. Which env vars does the process on node 2, GPU 5
#    see? Give RANK, LOCAL_RANK, WORLD_SIZE and its device string.
# 2. (Easy) Predict `hello`'s printed sum for world_size=8, then run it (pick a
#    fresh port).
# 3. (Medium) Write a worker where each rank holds `torch.arange(4) + 4*rank`
#    (world 4) and all_reduce-sums. Predict the resulting vector, verify, and
#    check its total equals sum(range(16)).
# 4. (Medium) Start two `run_distributed` calls on the SAME port from two
#    notebooks (or threads). Explain what happens using the TCPStore model.
# 5. (Hard) Extend section 8 to world sizes 2–8, separating spawn time from
#    init time (time inside the worker vs outside). Which grows faster, and why
#    does the second collective in a group cost ~nothing?
# 6. (Hard, Kaggle) Run hello_distributed.py under NCCL_DEBUG=INFO. Find the
#    topology, transport, and ring lines. Is there NVLink? What does that
#    predict for ch07?
#
# <details><summary>Solution sketches</summary>
#
# 1. RANK = 2*8+5 = 21, LOCAL_RANK = 5, WORLD_SIZE = 32, device cuda:5.
# 2. 8*9/2 = 36, printed by all 8 ranks.
# 3. Elementwise: arange(4)*4 + (0+4+8+12) = [24, 28, 32, 36]; total 120 =
#    sum(range(16)).
# 4. Second rank-0 fails to bind (EADDRINUSE) or, with unlucky timing, its
#    clients check into the FIRST job's store and corrupt both rendezvous —
#    why shared machines need per-job ports (torchrun --standalone solves it).
# 5. Spawn grows linearly (one fresh interpreter per rank); init grows mildly;
#    the first collective builds connections/communicators per group and is
#    cached afterwards — warmup discipline follows.
# 6. Expect P2P-over-PCIe or SHM transport, rings through the host bridge, no
#    NVL entries → ch07 bus bandwidth in single-digit GB/s, far below NVLink's
#    hundreds.
# </details>
#
# ## Summary
#
# - Distributed = N identical processes (SPMD); nothing shared; identity via
#   RANK / LOCAL_RANK / WORLD_SIZE; LOCAL_RANK's one job is picking `cuda:{i}`.
# - init_process_group = TCPStore rendezvous (rank 0 hosts the phone number at
#   MASTER_ADDR:MASTER_PORT) + backend communicator build. Control plane only.
# - NCCL: CUDA tensors, topology-aware rings/trees, in-kernel, overlappable.
#   Gloo: CPU, universal — our simulation backend. Match backend to device.
# - torchrun: sets env vars, forks N children, babysits; --standalone for one
#   node; mp.spawn (run_distributed) is its in-notebook stand-in.
# - Every rank must hit every collective, same order — or timeout. Debug with
#   TORCH_DISTRIBUTED_DEBUG, NCCL_DEBUG, py-spy, short timeouts.
# - Kaggle 2× T4 = real NCCL over PCIe (no NVLink); Colab = Gloo simulation.

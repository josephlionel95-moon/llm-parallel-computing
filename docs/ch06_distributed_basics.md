# Chapter 06 — Distributed Computing Basics: Processes, Ranks, torchrun, NCCL

> **Difficulty:** 🟡 Medium · **Study time:** 4–6 h · **Requires:** chapters 00, 05
> **Notebook:** notebooks/ch06_distributed_basics.ipynb · **Experiments:** experiments/ch06/

## Learning objectives

You will be able to...

- Define process, rank, world size, and local rank — and say precisely when
  `RANK != LOCAL_RANK`.
- Narrate what `dist.init_process_group` actually does, step by step: TCPStore
  creation, rendezvous, backend communicator setup.
- Choose between NCCL, Gloo, and MPI, and explain *why* NCCL is fastest (topology-aware
  rings/trees, communication executed inside CUDA kernels).
- Bind each process to its own GPU and diagnose the "everyone on GPU 0" bug on sight.
- Explain what `torchrun` forks, which environment variables it sets, and what
  `--standalone` and the rdzv backends do.
- Launch multi-process code *inside a notebook* with `torch.multiprocessing.spawn`.
- Debug hangs with `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL`, and
  recognize the four classic failure modes before wasting an afternoon.
- Read `src/llmdist/utils/dist.py` line by line — the plumbing every later chapter uses.

## Intuition

Chapter 05 proved you need many GPUs. This chapter answers the question that sounds
trivial and is not: **how do N GPUs even agree to work together?**

The mental model to install first: **distributed PyTorch is not threads, not magic —
it is N *identical copies of your script*, running as N ordinary operating-system
processes** (chapter 00's process model, now load-bearing). Nobody is "the program";
each copy discovers its identity from environment variables and behaves accordingly.
This is the SPMD model — Single Program, Multiple Data. The same `train.py` runs
everywhere; `if rank == 0:` is how one copy volunteers to write checkpoints.

Because they are separate processes (possibly on separate machines), they share
nothing: no variables, no tensors, no random seeds. Everything shared must be
*communicated*, and before they can communicate they must *find each other* — the
rendezvous problem. The entire apparatus of this chapter — MASTER_ADDR, TCPStore,
`init_process_group`, torchrun — is the answer to "N strangers, possibly on different
continents, must form a group with numbered seats and a shared address book, using
nothing but one well-known phone number."

The phone-number analogy is exact: MASTER_ADDR:MASTER_PORT is the one number everyone
knows in advance. Rank 0 answers that phone. Everyone else calls in, registers, and
receives the connection details of all other members. After that handshake the phone
is barely used — the real traffic (gradients, activations) flows over the fast links
(NVLink/PCIe/InfiniBand) that the backend negotiated during setup.

## Theory

### Vocabulary: rank, world, local rank

```
                     WORLD (world_size = 8)
   ┌───────────── node 0 ──────────────┐  ┌───────────── node 1 ──────────────┐
   │ proc 0    proc 1   proc 2  proc 3 │  │ proc 4    proc 5   proc 6  proc 7 │
   │ RANK=0    RANK=1   RANK=2  RANK=3 │  │ RANK=4    RANK=5   RANK=6  RANK=7 │
   │ LOCAL=0   LOCAL=1  LOCAL=2 LOCAL=3│  │ LOCAL=0   LOCAL=1  LOCAL=2 LOCAL=3│
   │ GPU 0     GPU 1    GPU 2   GPU 3  │  │ GPU 0     GPU 1    GPU 2   GPU 3  │
   └────────────────────────────────────┘  └───────────────────────────────────┘
```

- **world size** — total number of processes in the job (8 here).
- **rank** — globally unique id, 0 … world_size−1. Used for algorithm logic
  ("rank 0 saves the checkpoint", "send to rank (r+1) % p").
- **local rank** — id *within one machine*, 0 … nproc_per_node−1. Used for exactly one
  thing: **picking the GPU** (`cuda:{local_rank}`), because CUDA device indices are
  per-machine. On a single node RANK == LOCAL_RANK, which is why the distinction only
  bites when you first touch two nodes — and rank 5 tries to open `cuda:5` on a 4-GPU
  box, or worse, every rank on node 1 computes `cuda:rank` and crashes.

On Kaggle (one node, 2× T4): world_size=2, ranks {0,1}, local ranks {0,1} — identical.
Keep writing `cuda:{LOCAL_RANK}` anyway; code that survives the single-node phase
unchanged is the goal.

### The rendezvous: what `init_process_group` actually does

Step by step, for the default env:// initialization:

```
 rank 0                                    ranks 1..p-1
 ──────                                    ────────────
 1. read RANK, WORLD_SIZE,                 1. read RANK, WORLD_SIZE,
    MASTER_ADDR, MASTER_PORT                  MASTER_ADDR, MASTER_PORT
 2. create TCPStore SERVER:                2. create TCPStore CLIENT:
    bind MASTER_ADDR:MASTER_PORT              connect to MASTER_ADDR:MASTER_PORT
    (a tiny key-value server)                 (retry until timeout)
 3. wait until all p processes             3. announce self in the store
    have checked in ────────────────────────► (key: "worker{rank}")
 4. barrier via store: everyone sees "p processes present"
 5. backend init:
    - NCCL: rank 0 creates a ncclUniqueId, PUTs it in the store;
      others GET it; all call ncclCommInitRank(world, id, rank)
      → NCCL probes topology (NVLink? PCIe? NIC?) and builds
        ring/tree channels between the GPUs
    - Gloo: pairwise TCP connections established via store-published addresses
 6. init_process_group RETURNS on every rank — the group exists
```

Three facts worth engraving:

1. **The TCPStore is a real server socket owned by rank 0's process.** "Port already
   in use" errors, firewall hangs, and `MASTER_ADDR` typos are all failures of step 2 —
   before any GPU is touched.
2. **The store is control plane, not data plane.** It carries tiny strings (the NCCL
   unique id is 128 bytes). Gradients never pass through it. Slow store ≠ slow training;
   dead store = no training.
3. **Everything blocks until everyone arrives.** If you launch 3 processes with
   WORLD_SIZE=4, all three sit inside `init_process_group` until the timeout
   (default 30 min for NCCL jobs — set it lower while learning, as our
   `dist.py` does with 120 s).

`init_method="env://"` (the default) means "read MASTER_ADDR etc. from the
environment". The alternatives — `tcp://host:port`, `file:///shared/path` — are the
same rendezvous over a different bulletin board.

### Process groups and subgroups

`init_process_group` creates the **default group** (WORLD). Collectives take a
`group=` argument; later chapters carve subgroups out of the world:

```python
tp_group = dist.new_group([0, 1, 2, 3])   # tensor-parallel islands
dp_group = dist.new_group([0, 4])         # data-parallel pairs across islands
```

Rule that saves hours: **`new_group` must be called by *every* rank in the world with
the same arguments, in the same order** — even ranks that are not members of the new
group — because group creation is itself a collective agreement. Call it from inside
`if rank in members:` and you get chapter 06's favorite deadlock. 3D parallelism
(ch18) is little more than disciplined bookkeeping over such subgroups.

### Backends: NCCL vs Gloo vs MPI

| | NCCL | Gloo | MPI |
|---|---|---|---|
| tensors | CUDA (GPU) only | CPU (+ limited CUDA) | build-dependent |
| speed on GPUs | fastest — by design | slow (staging via CPU) | varies |
| runs without GPU | ❌ | ✅ | ✅ |
| typical use | all real GPU training | CPU tests, notebooks, rendezvous-ish control traffic | legacy HPC clusters |

*Why NCCL is fastest*, and it is not one reason but three:

1. **Topology awareness.** At init, NCCL discovers how GPUs are physically wired
   (NVLink mesh? PCIe switch? NUMA node? which NIC is closest?) and constructs
   communication **rings and trees** that traverse each physical link exactly as often
   as needed. A ring AllReduce on 8 NVLink-connected GPUs uses all links
   simultaneously in both directions; a naive implementation would bottleneck on one.
2. **In-kernel communication.** NCCL collectives *are CUDA kernels*. Data moves
   GPU-to-GPU via P2P (NVLink/PCIe peer access) or GPUDirect RDMA without a round trip
   through host memory, and the copy/reduce arithmetic (the `+` in AllReduce-sum) runs
   on the GPU's own SMs, pipelined chunk by chunk.
3. **Stream semantics.** Because a collective is just a kernel on a stream, it can be
   overlapped with compute kernels on another stream — the foundation of DDP's
   overlap (ch09). Gloo, running on CPU threads over TCP, can't participate in this.

Gloo's virtue is universality: CPU tensors, no GPU required — hence every notebook
simulation in this course. MPI matters when your cluster's job launcher is `mpirun`
and the fabric has tuned MPI libraries; PyTorch supports it but you must compile
PyTorch against it, so you will know if you need it.

**The rule that generates half of all beginner crashes:** NCCL moves *GPU tensors
only*. `dist.all_reduce(cpu_tensor)` under NCCL → error (see failure modes). Gloo
moves CPU tensors. Match backend to tensor device, always.

### Device binding: the "everyone on GPU 0" bug

CUDA has a per-process notion of "current device", defaulting to 0. If four processes
on one node all skip `torch.cuda.set_device`, then all four allocate on GPU 0:

```
 symptom                                   cause
 ───────                                   ─────
 GPU 0: 4 × memory, 4 × contexts, OOM      every rank left current device = 0
 GPU 1–3: idle in nvidia-smi
 NCCL error: "Duplicate GPU detected"      two ranks told NCCL to use cuda:0
```

The fix is one line, early, before any tensor is created and before (or as) the
process group initializes:

```python
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

Our `setup_from_env()` does this for you. An equivalent alternative is passing
`device_id` to `init_process_group` (newer PyTorch), but the explicit `set_device` is
universal and greppable in every codebase you will read.

### torchrun anatomy

```
$ torchrun --nproc_per_node=2 train.py --lr 3e-4
     │
     ├── parses ITS OWN flags, keeps the rest for your script
     ├── runs a rendezvous among AGENTS (trivial for one node;
     │   etcd/c10d store for multi-node) to assign rank ranges
     ├── forks nproc_per_node child processes, each with env:
     │       RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE,
     │       MASTER_ADDR, MASTER_PORT, plus TORCHELASTIC_* bookkeeping
     ├── each child executes:  python train.py --lr 3e-4
     └── the agent WATCHES the children: if one dies, it tears down
         the rest (and, with --max-restarts, relaunches the group)
```

torchrun is the successor of `python -m torch.distributed.launch`; it adds
*elasticity* (restart on failure, dynamic world membership) via TorchElastic, but for
this course its job is simply: set the env vars, fork N copies, babysit them.

Flags you will actually use:

- `--nproc_per_node=2` — processes on this machine (usually = GPUs). `gpu` and `auto`
  are accepted values.
- `--standalone` — single-node shortcut: torchrun spins up its own ephemeral
  rendezvous (a c10d TCPStore on a free port) so you don't set MASTER_ADDR/PORT at
  all. This is the form in our README quick start.
- `--nnodes=2 --node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=host:29400` —
  multi-node: every node runs the same command with its own `--node_rank`, all
  pointing at one rendezvous endpoint. `RANK = node_rank * nproc_per_node + local_rank`
  is then assigned for you.
- `--max-restarts=3` — elastic recovery; on failure the *whole group* restarts (state
  recovery is your job — ch19's checkpointing).

Notebook constraint: torchrun launches *fresh processes running a script file* — it
cannot run a Jupyter cell. Hence:

### `torch.multiprocessing.spawn` — distributed inside a notebook

`mp.spawn(fn, args=(...), nprocs=2, join=True)` forks N processes each running
`fn(rank, *args)`. We set the env vars ourselves and call `init_process_group`
inside the worker — see `run_distributed` below. Two subtleties:

1. spawn uses the **spawn** start method (fresh interpreter), not fork — required for
   CUDA (a forked child inheriting an initialized CUDA context is undefined behavior;
   chapter 00 explains why). Consequence: `fn` and its args must be picklable and
   importable — defined in a module or the notebook's top level, no closures over
   un-picklable state.
2. Each worker must be told its rank *by argument* (spawn passes it) and told the
   rendezvous address *by env var* — which the parent sets before spawning.

### Debugging hangs and crashes

A distributed hang is silent by design — everyone is politely waiting for someone who
will never come. Instruments, in the order to reach for them:

1. `TORCH_DISTRIBUTED_DEBUG=DETAIL` — PyTorch logs every collective call with shapes;
   mismatched-collective bugs (rank 0 calls all_reduce, rank 1 calls broadcast) become
   visible.
2. `NCCL_DEBUG=INFO` (and `NCCL_DEBUG_SUBSYS=INIT,NET` to filter) — NCCL prints the
   topology it detected, the rings/trees it built, and which transport each channel
   uses (P2P/SHM/NET). The line `via P2P/... over PCIe` vs `NVL` is how you *verify*
   Kaggle's T4s have no NVLink (ch07 measures the consequence).
3. `TORCH_NCCL_BLOCKING_WAIT=1` / `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` — convert
   infinite NCCL hangs into loud errors after the timeout. (Older spellings without
   the `TORCH_` prefix are deprecated since PyTorch 2.2 but still seen everywhere.)
4. `py-spy dump --pid <pid>` on each process — the ground truth of where each rank is
   stuck; two ranks in different collectives = your bug, found.
5. Lower the `timeout` in `init_process_group` while developing (ours: 120 s), so
   mistakes fail in minutes, not the NCCL default of 30.

### Failure modes table

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | All ranks freeze in `init_process_group`, then timeout | WORLD_SIZE says p but fewer processes launched (or one crashed pre-init) | launch counts must match; check the dead rank's traceback first |
| 2 | `EADDRINUSE: address already in use` at startup | MASTER_PORT still held by a previous (often crashed/zombie) run, or two jobs share a port | pick another port; kill zombies (`pkill -f train.py`); `--standalone` picks free ports for you |
| 3 | `Tensors must be CUDA and dense` / NCCL error on a CPU tensor | NCCL backend given a CPU tensor | move tensor to the rank's GPU, or use Gloo for CPU tensors |
| 4 | Multi-node: node 1 stuck connecting, node 0 fine | firewall blocks MASTER_PORT (or NCCL's ephemeral ports); MASTER_ADDR resolves to 127.0.0.1 on the wrong box | open the port range / set NCCL_SOCKET_IFNAME to the right NIC; use a reachable MASTER_ADDR |
| 5 | GPU 0 OOMs, GPUs 1..k idle | missing `torch.cuda.set_device(local_rank)` | bind device from LOCAL_RANK before creating tensors |
| 6 | Hang mid-training at a random step | ranks diverged: one rank skipped a collective (uneven data, `if rank == 0: all_reduce`) | every rank must execute every collective, same order, same shapes; TORCH_DISTRIBUTED_DEBUG=DETAIL |
| 7 | `Duplicate GPU detected: rank 0 and 1 both on CUDA device 0` | two ranks bound to the same GPU (bad set_device arithmetic) | use LOCAL_RANK, not RANK, on multi-node; one process per GPU |
| 8 | Works single-node, dies multi-node with `cuda:4` invalid | `cuda:{RANK}` instead of `cuda:{LOCAL_RANK}` | the vocabulary lesson, applied |

`experiments/ch06/failure_modes.py` reproduces modes 1, 2, 3, 6 (and 5 conceptually)
behind a `--mode` flag, each with a commented explanation and a bounded timeout so the
demonstration hurts for seconds, not minutes.

## Mathematics

This chapter is systems, not calculus — but two quantities deserve formulas.

**Rank arithmetic (multi-node).** With `nnodes` machines × `nproc` processes:

```
RANK        = node_rank · nproc + LOCAL_RANK
node_rank   = RANK div nproc
LOCAL_RANK  = RANK mod nproc
```

Every "works on one node, explodes on two" bug is a violation of one of these three
lines.

**Rendezvous cost.** The TCPStore handshake is p clients × O(1) small messages against
one server, so expect init time ≈ `c₀ + c₁·p` for process spawn + store check-in —
linear in p with a large constant c₀ (interpreter start, CUDA context ~0.5–2 s when
GPUs are involved). NCCL communicator setup adds topology probing that grows with
GPUs per node and channels built. The practical lesson:
`init_process_group` costs *seconds* — measured by `rendezvous_timing.py` — which is
(a) why real clusters amortize it over days of training, and (b) why elasticity
(restart-on-failure) is a feature worth torchrun's complexity, not an afterthought.
It also bounds the world size at which per-step latency α (chapter 07) matters far
more than setup ever will.

## Implementation

### `src/llmdist/utils/dist.py`, line by line

This 89-line file is the plumbing every later chapter imports. Read it with the theory
above in mind:

```python
def setup_from_env() -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])                       # (1)
    world_size = int(os.environ["WORLD_SIZE"])           # (1)
    local_rank = int(os.environ.get("LOCAL_RANK", rank)) # (2)

    if torch.cuda.is_available() and torch.cuda.device_count() > local_rank:
        device = torch.device(f"cuda:{local_rank}")      # (3)
        torch.cuda.set_device(device)                    # (4)  THE line
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"                                 # (5)

    dist.init_process_group(backend=backend,
                            timeout=datetime.timedelta(seconds=120))  # (6)
    return rank, world_size, device
```

1. Identity comes from the environment — torchrun wrote it. Crashing with a KeyError
   here means you ran `python script.py` instead of `torchrun ... script.py` (or
   didn't set the vars yourself).
2. `LOCAL_RANK` falls back to `rank`: correct on a single node, and lets hand-launched
   single-node runs omit it.
3. Device chosen by **local** rank — multi-node safe.
4. `set_device` before the process group and before any tensor: kills the
   "everyone on GPU 0" bug and lets NCCL bind the right device per rank.
5. Graceful degradation: no (or not enough) GPUs → CPU + Gloo. This one branch is why
   every experiment in the course runs on CI and on your laptop.
6. 120 s timeout instead of NCCL's default 30 *minutes*: while learning, you want
   mistakes to fail fast and loudly. `init_method` defaults to `env://` — the
   MASTER_ADDR/PORT that torchrun set complete the rendezvous.

```python
def _worker(rank, world_size, backend, port, fn, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"   # notebook = one machine
    os.environ["MASTER_PORT"] = str(port)     # all workers agree on the phone number
    os.environ["RANK"] = str(rank)            # spawn tells us who we are...
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=120))
    if backend == "nccl":
        torch.cuda.set_device(rank)           # one node: local_rank == rank
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()          # ALWAYS clean up: leaked groups
                                              # hold the port -> failure mode #2

def run_distributed(fn, world_size=2, backend="gloo", port=29500, args=()):
    mp.spawn(_worker, args=(world_size, backend, port, fn, args),
             nprocs=world_size, join=True)
```

`_worker` is torchrun-in-miniature: it does by hand (env vars, rank assignment) what
torchrun does with agents, then meets the same `init_process_group`. Passing
`rank=`/`world_size=` explicitly (instead of env-only) is the other documented
initialization style; we use it here so the code shows both. `rank0_print` closes the
file: the SPMD idiom for not printing everything p times.

### hello_distributed.py — the canonical first program

`experiments/ch06/hello_distributed.py` (the README quick-start target) does the four
things every distributed program must prove it can do: discover identity, bind a
device, pass a collective sanity check, and shut down cleanly:

```python
rank, world, device = setup_from_env()
t = torch.ones(1, device=device) * (rank + 1)
dist.all_reduce(t)          # sum over ranks: expect 1+2+...+p = p(p+1)/2
assert t.item() == world * (world + 1) / 2
```

The warmup AllReduce is not decoration: the *first* collective triggers lazy NCCL
communicator/channel construction and is often 100× slower than steady state —
a measurement lesson (never time the first call) and a sanity check (if anything about
rendezvous, binding, or the fabric is wrong, it dies here, one line into the job,
instead of 40 minutes into a training run).

### Kaggle vs Colab

| | Kaggle (2× T4) | Colab free (1× T4) |
|---|---|---|
| real NCCL multi-GPU | ✅ `torchrun --nproc_per_node=2` in a `!` cell, or spawn with backend="nccl" | ❌ only 1 GPU |
| interconnect | **PCIe, no NVLink** — `nvidia-smi topo -m` shows PHB/PIX, not NV#; β measured in ch07 | — |
| course fallback | — | `run_distributed(fn, 2, backend="gloo")`: 2 CPU processes, identical *semantics*, no NCCL speed |
| gotcha | both T4s share host PCIe bandwidth; don't extrapolate β×2 | Gloo timings teach nothing about GPUs — never quote them as performance |

The Gloo simulation preserves every rank/collective/deadlock behavior — you can learn
(and break) the entire programming model on a laptop; only the *speeds* are fiction.

## Profiling & measurement

1. **`hello_distributed.py`** — run it three ways and confirm all behave identically:
   `torchrun --standalone --nproc_per_node=2`, plain `python` (falls back to a
   self-initialized world of 1), and on Kaggle with 2 GPUs (banner should say NCCL and
   two distinct devices — if both lines say `cuda:0`, you have failure mode 5).
2. **`rendezvous_timing.py`** — measures wall-clock of spawn + `init_process_group` +
   first collective for world sizes 2/4/8 (CPU/Gloo processes). Expect linear-ish
   growth with a ~1 s-scale constant; the printout separates spawn cost from
   rendezvous cost. The number to remember: init is *seconds* — amortized once per
   job, never per step.
3. **`failure_modes.py --mode hang|port|nccl_cpu|mismatch`** — deliberately broken.
   Run each, read the error, read the comment, close the loop between symptom and
   cause while the stakes are zero. All modes bound their own damage with short
   timeouts (~10–20 s).
4. On Kaggle, run `nvidia-smi topo -m` and `NCCL_DEBUG=INFO torchrun --standalone
   --nproc_per_node=2 experiments/ch06/hello_distributed.py`; find the transport line
   in the log and confirm PCIe (no NVLink). Keep that log — ch07 quantifies what it
   costs.

## Common mistakes

1. **Running a torchrun script with plain `python`.**
   *Symptom:* `KeyError: 'RANK'`.
   *Cause:* no launcher set the env vars.
   *Fix:* `torchrun --standalone --nproc_per_node=N script.py`, or make the script
   fall back to single-process (as `hello_distributed.py` does).
2. **Forgetting `torch.cuda.set_device(local_rank)`.**
   *Symptom:* GPU 0 OOM, other GPUs idle, or NCCL "Duplicate GPU detected".
   *Cause:* per-process current device defaults to 0.
   *Fix:* bind by LOCAL_RANK first thing; use `setup_from_env()`.
3. **Using RANK to pick the CUDA device.**
   *Symptom:* fine on one node; `invalid device ordinal` on two.
   *Cause:* device indices are per-machine; RANK is global.
   *Fix:* `cuda:{LOCAL_RANK}`, always.
4. **Collectives inside `if rank == 0:`.**
   *Symptom:* hang at a random step, all GPUs at 100% util doing nothing (spinning in
   the collective kernel).
   *Cause:* collectives are appointments — every rank must show up, same order, same
   shapes.
   *Fix:* hoist collectives out of rank-conditional code; TORCH_DISTRIBUTED_DEBUG=DETAIL
   to find the divergence.
5. **NCCL backend, CPU tensor** (or Gloo with a GPU-only op).
   *Symptom:* immediate error naming the tensor device.
   *Cause:* backend/device mismatch.
   *Fix:* NCCL ⇔ CUDA tensors, Gloo ⇔ CPU tensors; in device-flexible code, choose
   the backend from the device as `setup_from_env` does.
6. **Reusing MASTER_PORT while a zombie holds it.**
   *Symptom:* `address already in use` seconds after a crashed run.
   *Cause:* the previous rank-0 process (or its TIME_WAIT socket) still owns the port.
   *Fix:* kill leftovers, change ports, or use `--standalone`; in notebooks, always
   `destroy_process_group()` in a `finally:` (our `_worker` does).
7. **Testing multi-node with MASTER_ADDR=localhost.**
   *Symptom:* node 0 proceeds, node 1 times out connecting.
   *Cause:* every node dialed itself.
   *Fix:* MASTER_ADDR must be node 0's address as reachable *from the other nodes*;
   verify with `nc -zv host port` before blaming NCCL.
8. **Leaving the 30-minute default timeout during development.**
   *Symptom:* a typo costs you half an hour of silence per attempt.
   *Cause:* defaults are tuned for production patience.
   *Fix:* `timeout=timedelta(seconds=120)` while learning; raise it for real jobs with
   slow first-batch compilation.

## Limitations & outlook

- We built the *group* but sent almost nothing through it. Chapter 07 opens the
  toolbox — broadcast, reduce, all_reduce, all_gather, reduce_scatter — and measures
  α and β for each on Kaggle's PCIe pair.
- The rendezvous story here is single-job, static-world (with a nod to elasticity).
  Production elastics (torchrun `--max-restarts`, Kubernetes operators) and
  fault-tolerant training are ch28 territory.
- We asserted NCCL's ring/tree superiority; ch07 *derives* ring AllReduce's
  2(p−1)/p·N bytes and verifies the bus-bandwidth math against nccl-tests conventions.
- `run_distributed`'s fixed port 29500 is fine for a course, not for shared clusters —
  two students on one machine collide (failure mode 2 — now you know why).
- Process groups will multiply: DDP uses one, tensor+data+pipeline parallelism uses
  several overlapping ones (ch10–18). The `new_group`-must-be-called-by-all rule from
  this chapter is the invariant that keeps 3D parallelism sane.

## Exercises

1. **(Easy)** Job: 3 nodes × 8 GPUs. For the process with RANK=17, compute node_rank,
   LOCAL_RANK, and its device string. Which env var did torchrun set to 17?
2. **(Easy)** Predict what `hello_distributed.py` prints for the all_reduce result at
   world sizes 2, 4, and 8, before running it. Then verify with torchrun (or spawn).
3. **(Medium)** Modify `hello_distributed.py` so rank 0 *skips* the all_reduce
   (`if rank != 0:`). Predict the behavior under Gloo with our 120 s timeout, then
   run `failure_modes.py --mode mismatch` and compare with your prediction.
4. **(Medium)** Using `run_distributed`, write a notebook cell where each of 4 CPU
   ranks creates `torch.arange(4) + 4*rank` and the group computes the global sum via
   `all_reduce`. Verify against `sum(range(16))` computed serially.
5. **(Medium)** Launch two *concurrent* `run_distributed` calls with the same port
   (e.g., from two notebooks). Explain the failure using the TCPStore model, then fix
   it without changing `dist.py`.
6. **(Hard)** Instrument `rendezvous_timing.py` to separately time (a) `mp.spawn`
   process startup, (b) `init_process_group`, (c) first collective, at world sizes
   2–8. Which component grows fastest with p on your machine, and why does (c) shrink
   to microseconds on the *second* call?
7. **(Hard)** On Kaggle 2× T4: capture `NCCL_DEBUG=INFO` output from
   `hello_distributed.py`. Identify the lines showing (a) detected topology,
   (b) chosen transport, (c) ring construction. What transport did NCCL pick, and
   what does that predict for ch07's bandwidth measurement?
8. **(Research)** Read the TorchElastic rendezvous design (pytorch.org docs). The
   c10d rendezvous must solve membership *changes* mid-job. Sketch the state machine:
   what happens to ranks, and to your training state, when one of 32 nodes dies and
   `--max-restarts=3` triggers? What does this imply for how often you checkpoint (ch19)?

## Solutions

<details>
<summary>Solutions to exercises</summary>

1. nproc=8: node_rank = 17 div 8 = 2, LOCAL_RANK = 17 mod 8 = 1, device `cuda:1`.
   torchrun set `RANK=17` (and LOCAL_RANK=1, WORLD_SIZE=24).
2. Sum of (rank+1) over ranks = p(p+1)/2: world 2 → 3.0, world 4 → 10.0,
   world 8 → 36.0 — every rank prints the same value; that sameness *is* AllReduce.
3. Ranks 1..p−1 enter all_reduce and block; rank 0 exits (or reaches
   destroy_process_group and blocks there). After 120 s the waiting ranks raise a
   timeout error naming the collective. Under NCCL without blocking-wait flags this
   can hang until the 30-min default — the argument for short dev timeouts.
4. Each rank's tensor sums to 6+16·rank... simpler: global elementwise result is
   `arange(4)*4 + (0+4+8+12)` = `[24,28,32,36]`; its sum 120 = sum(range(16)). The
   check passing on all ranks demonstrates all_reduce's "everyone gets the answer".
5. Both rank-0 workers try to bind 127.0.0.1:29500; the second gets EADDRINUSE (or
   worse, *joins the first job's store* if timing interleaves, corrupting both
   worlds). Fix: `run_distributed(fn, port=29501)` for the second call — the
   parameter exists precisely for this.
6. Typical shape: (a) grows linearly (one interpreter per proc, ~0.5–1 s each,
   sometimes parallelized by the OS), (b) grows mildly (store check-ins are cheap),
   (c) first collective is expensive once per *group* (Gloo connection setup / NCCL
   channel build), then cached — steady-state collectives skip setup entirely, which
   is why benchmarks always warm up. Exact numbers: report yours; they're
   machine-dependent.
7. Expect topology lines showing both GPUs under a PCIe host bridge (PHB/PIX in
   `nvidia-smi topo -m`), transport `P2P` over PCIe or SHM (shared host memory) —
   and **no NVL entries**. Prediction: ch07's measured bus bandwidth lands near
   PCIe-3-x16-shared reality (single-digit GB/s), an order of magnitude under NVLink.
8. Sketch: agents detect the failure → surviving agents re-enter rendezvous → a new
   world (possibly smaller, if elastic min/max allows) is formed with **new ranks** →
   torchrun restarts your script from scratch on every member. All in-memory state is
   gone; recovery = load latest checkpoint. Implication: checkpoint interval bounds
   lost work; expected loss ≈ MTBF-driven — ch19 does this arithmetic.

</details>

## Interview questions

1. What exactly does `torchrun --nproc_per_node=4 train.py` do before your first line
   of Python runs?
2. RANK vs LOCAL_RANK: definitions, and a concrete bug caused by confusing them.
3. Walk through `init_process_group` step by step. What is the TCPStore, who owns it,
   and what data passes through it?
4. Why is NCCL faster than Gloo for GPU collectives? Give at least two architectural
   reasons.
5. Your job hangs with all GPUs at 100% utilization. Top three hypotheses, and the
   exact tools/env vars you'd use to distinguish them.
6. Why must `dist.new_group` be called by every rank, including non-members?
7. What happens if you launch 3 processes with WORLD_SIZE=4? When do you find out?
8. Why can't torchrun launch code in a Jupyter cell, and what do you use instead?
   What constraint does the spawn start method put on your function?
9. What is the first NCCL collective in a process's life doing that later ones don't,
   and how does that change how you benchmark?
10. Design the launch commands for 2 nodes × 8 GPUs with torchrun, including the
    rendezvous flags. Which values differ between the two nodes?

## Summary

- Distributed PyTorch = N identical OS processes (SPMD); identity via env vars:
  RANK (global), LOCAL_RANK (per-node, picks the GPU), WORLD_SIZE.
- `RANK = node_rank × nproc + LOCAL_RANK`; use LOCAL_RANK for `cuda:{...}`, always.
- `init_process_group` = TCPStore rendezvous at MASTER_ADDR:MASTER_PORT (rank 0 hosts)
  + backend communicator setup (NCCL: topology probe, rings/trees, 128-byte unique id
  via the store).
- The store is control plane; gradients never touch it.
- NCCL: GPU tensors, topology-aware, in-kernel, overlappable — the training backend.
  Gloo: CPU, universal — the simulation backend. Match backend to tensor device.
- `torch.cuda.set_device(local_rank)` before anything else, or everyone lands on GPU 0.
- torchrun: sets env vars, forks N children, watches them; `--standalone` for
  single-node, rdzv flags for multi-node, restarts for elasticity.
- In notebooks: `mp.spawn` via our `run_distributed` — same rendezvous, hand-set env.
- Hangs are appointments missed: every rank, every collective, same order; debug with
  TORCH_DISTRIBUTED_DEBUG, NCCL_DEBUG, py-spy, short timeouts.
- Init costs seconds and is paid once; the per-message costs that dominate training
  are chapter 07's subject.

## References

See `references/ch06.md`. Key items: PyTorch distributed docs (API + debugging
flags), torchrun/TorchElastic docs (launcher anatomy and rendezvous), NCCL
documentation (topology, transports, NCCL_DEBUG), Jeon et al. on GPU cluster failure
analysis (why elasticity exists).

# %% [markdown]
# # Chapter 00 — Python Memory, Processes & the GIL
#
# **Hardware requirements: none.** Everything in this notebook runs on CPU; it runs
# identically on Colab, Kaggle, or your laptop. (That is the point: distributed
# training is built from OS processes and byte streams, which exist everywhere.)
#
# We answer four questions with running code:
#
# 1. What is a Python variable, and when do two tensors share memory?
# 2. What *actually* crosses a process boundary? (pickle bytes / shared-memory handles)
# 3. Why don't threads speed up Python compute (the GIL), and why does that make
#    PyTorch a process-based system?
# 4. Why does CUDA force the `spawn` start method?

# %%
import sys

sys.path.insert(0, "../src")

from llmdist.utils import env_check

env = env_check.detect()
print(env.banner())

# %% [markdown]
# ## 1. Names alias; storages decide
#
# A variable is a label tied to an object. For tensors, the object is a small
# header pointing at a **storage** buffer; `tensor.data_ptr()` exposes the buffer's
# address. Two headers over one storage = a *view*: cheap, and mutation travels.

# %%
import copy

import torch

a = torch.arange(6)
b = a.view(2, 3)      # view: same storage
c = a.clone()         # copy: new storage
d = a                 # alias: same *object*, not just same storage

print(f"a is d              : {a is d}")
print(f"a.data_ptr()        : {a.data_ptr():#x}")
print(f"b.data_ptr() (view) : {b.data_ptr():#x}  same={b.data_ptr() == a.data_ptr()}")
print(f"c.data_ptr() (clone): {c.data_ptr():#x}  same={c.data_ptr() == a.data_ptr()}")

a[0] = 999
print(f"after a[0]=999 -> b[0,0]={b[0, 0].item()}, c[0]={c[0].item()}")

# %% [markdown]
# Shallow vs deep copy is the container-level version of the same story. Watch a
# shallow-copied list of tensors share *inner* storages:

# %%
params = [torch.zeros(2), torch.zeros(2)]
shallow = copy.copy(params)
deep = copy.deepcopy(params)

params[0] += 1.0  # in-place
print(f"shallow[0] = {shallow[0].tolist()}   (mutated: inner objects are shared)")
print(f"deep[0]    = {deep[0].tolist()}   (safe: recursively copied)")

# %% [markdown]
# **Which ops view, which ops copy?** A 30-second empirical audit — this table is
# worth memorizing, because DDP gradient buckets and FSDP shards are all views.

# %%
x = torch.randn(4, 6)
ops = {
    "x.view(6, 4)": x.view(6, 4),
    "x.transpose(0, 1)": x.transpose(0, 1),
    "x[1:3]": x[1:3],
    "x.reshape(24)": x.reshape(24),
    "x.transpose(0,1).reshape(24)": x.transpose(0, 1).reshape(24),
    "x.clone()": x.clone(),
    "x.to(torch.float64)": x.to(torch.float64),
    "x.detach()": x.detach(),
}
print(f"{'operation':<32} {'shares storage?':>15}")
for name, t in ops.items():
    print(f"{name:<32} {str(t.data_ptr() == x.data_ptr()):>15}")

# %% [markdown]
# ## 2. Pickling — the bytes that cross process boundaries
#
# `multiprocessing`, DataLoader workers, and `torch.distributed`'s object
# collectives all move data as **pickle byte streams**. Two facts to internalize:
#
# - a tensor pickles as *metadata + one raw buffer dump* (fast, compact);
# - a Python list pickles *per element* (slow, interpreter-bound).
#
# The measured version is `experiments/ch00/pickle_bench.py`; here we check sizes
# and see what survives a round trip.

# %%
import pickle

n = 100_000
t32 = torch.randn(n)                    # fp32 tensor
lst = t32.tolist()                      # same numbers as Python floats

blob_t = pickle.dumps(t32)
blob_l = pickle.dumps(lst)
print(f"raw payload      : {4 * n:>10,} bytes (n=1e5 fp32)")
print(f"pickled tensor   : {len(blob_t):>10,} bytes  (payload + ~small header)")
print(f"pickled list     : {len(blob_l):>10,} bytes  (~9 B per float: 8 B double + opcode)")

t2 = pickle.loads(blob_t)
print(f"round trip equal : {torch.equal(t2, t32)}")
print(f"new storage      : {t2.data_ptr() != t32.data_ptr()}  (a true copy crossed)")

# %% [markdown]
# What *cannot* be pickled is just as instructive: lambdas (no importable name),
# and live OS state (files, sockets, CUDA contexts). This is exactly why
# `mp.spawn` demands top-level worker functions.

# %%
try:
    pickle.dumps(lambda x: x + 1)
except Exception as e:  # noqa: BLE001 - we want to show the error type
    print(f"pickling a lambda -> {type(e).__name__}: {e}")

def top_level_fn(x: int) -> int:
    return x + 1

blob = pickle.dumps(top_level_fn)
print(f"pickling a top-level function works: {len(blob)} bytes "
      f"(it stores the *import path*, not the code): {blob[:40]!r}...")

# %% [markdown]
# ## 3. The GIL: threads vs processes
#
# CPython allows one thread at a time to execute *bytecode*. C kernels (NumPy,
# torch CPU ops) release the GIL, so the rule is:
#
# | workload | threads | processes |
# |---|---|---|
# | pure-Python compute | no speedup | speedup |
# | C-extension compute / I/O | speedup | speedup |
#
# We time a deliberately Python-heavy function with 1 thread, 4 threads, and (in
# the experiment script) 4 processes. Notebook caveat: spawning processes from a
# notebook requires importable workers, so the full 4-quadrant benchmark lives in
# `experiments/ch00/gil_bench.py` — run it after this cell.

# %%
import time
from concurrent.futures import ThreadPoolExecutor

def py_work(n: int) -> int:
    """Pure-Python busy loop: every iteration is interpreter bytecode -> GIL-bound."""
    s = 0
    for i in range(n):
        s += i * i
    return s

N = 2_000_000

t0 = time.perf_counter()
py_work(N)
t_serial = time.perf_counter() - t0

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(py_work, [N // 4] * 4))
t_threads = time.perf_counter() - t0

print(f"pure Python, 1 thread : {t_serial * 1e3:8.1f} ms")
print(f"pure Python, 4 threads: {t_threads * 1e3:8.1f} ms   "
      f"speedup {t_serial / t_threads:.2f}x  (GIL: expect ~1x or worse)")

# %%
# Same experiment with a GIL-releasing C kernel: torch matmul chunks.
def torch_work(a: torch.Tensor) -> torch.Tensor:
    return a @ a

torch.set_num_threads(1)  # 1 intra-op thread so *our* threads are the parallelism
mats = [torch.randn(256, 256) for _ in range(64)]

t0 = time.perf_counter()
for m in mats:
    torch_work(m)
t_serial_k = time.perf_counter() - t0

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(torch_work, mats))
t_threads_k = time.perf_counter() - t0

print(f"torch kernel, 1 thread : {t_serial_k * 1e3:8.1f} ms")
print(f"torch kernel, 4 threads: {t_threads_k * 1e3:8.1f} ms   "
      f"speedup {t_serial_k / t_threads_k:.2f}x  (GIL released inside matmul)")
torch.set_num_threads(torch.get_num_threads())  # leave defaults alone afterwards

# %% [markdown]
# The asymmetry above is the entire justification for PyTorch's architecture:
#
# - **DataLoader workers are processes** because `__getitem__`/transforms contain
#   arbitrary Python.
# - **One training process per GPU** (torchrun) because processes also isolate
#   CUDA contexts and failures — that part survives even a GIL-free CPython.

# %% [markdown]
# ## 4. fork vs spawn, and shared-memory tensors
#
# `fork` clones the parent's address space; `spawn` starts a fresh interpreter and
# *pickles* the target function + args to it. CUDA state cannot survive a fork
# (driver threads and kernel-side bookkeeping are not cloned), so **CUDA requires
# spawn** — which is why `llmdist.utils.dist.run_distributed` uses
# `torch.multiprocessing.spawn`.
#
# Below we demonstrate the third data-movement mechanism: **shared memory**.
# `share_memory_()` moves a CPU tensor's storage into an OS shared-memory segment;
# a child process then mutates it *in place* — no message is ever sent back.

# %%
import torch.multiprocessing as mp

def child_writes(rank: int, t: torch.Tensor) -> None:
    """Runs in a separate process; writes into the parent's storage."""
    t[rank] = 100 + rank

if __name__ == "__main__":
    shared = torch.zeros(4)
    shared.share_memory_()
    print(f"is_shared={shared.is_shared()}   before: {shared.tolist()}")

    try:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=child_writes, args=(r, shared)) for r in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        ok = all(p.exitcode == 0 for p in procs)
    except Exception as e:  # noqa: BLE001 - some Jupyter kernels can't re-import __main__
        print(f"spawn raised {type(e).__name__}: {e}")
        ok = False
    if ok:
        print(f"after children ran     : {shared.tolist()}   (mutated across processes!)")
    else:
        print("spawn failed in this notebook environment (spawn must pickle the")
        print("worker by import path - ch00's lesson in action). Run the script")
        print("version instead: experiments/ch00/shared_memory_tensor.py")

# %% [markdown]
# If the spawn cell failed in your notebook environment (some Jupyter setups can't
# re-import `__main__`), the script version always works:
#
# ```bash
# python experiments/ch00/shared_memory_tensor.py
# python experiments/ch00/gil_bench.py
# python experiments/ch00/pickle_bench.py
# ```
#
# ## Exercises
#
# 1. **(Easy)** Predict, then verify: `a = torch.zeros(3); b = a[:2]; b += 1` —
#    what is `a`? Prove your explanation with `data_ptr()`.
# 2. **(Medium)** Extend the pickle cell to fp16 tensors and `list(range(100_000))`
#    and explain every size to within 10 %.
# 3. **(Medium)** In the GIL cell, why did we call `torch.set_num_threads(1)`?
#    Re-run without it and explain the change.
# 4. **(Hard)** Add an `mp.Barrier` to the shared-memory demo so all children write
#    "simultaneously", then have each child *read* its neighbor's slot after the
#    barrier. What guarantees do you now have, and which are still missing?
#
# <details><summary>Solutions</summary>
#
# 1. `a == [1, 1, 0]`: slicing views, `+=` is in-place, writes land in `a`'s storage
#    (`b.data_ptr() == a.data_ptr()`).
# 2. fp16 tensor ≈ 200 kB (+header); list of small ints ≈ 300–500 kB (pickle stores
#    ints ≤ 4 bytes with 1-byte opcodes, so ~3–5 B each) — containers pay per
#    element, buffers per byte.
# 3. Without it, each matmul already uses all cores via OpenMP; adding Python
#    threads oversubscribes cores, so the "4 threads" case shows little or no gain
#    — you'd wrongly conclude the GIL wasn't released.
# 4. The barrier orders *phases* (all writes precede all reads), so each child
#    reliably sees its neighbor's write. Still missing: atomicity for concurrent
#    writes to the same element, and any cross-*machine* story — that's what ch06's
#    process groups and ch07's collectives supply.
#
# </details>
#
# ## Summary
#
# - Assignment aliases; `data_ptr()` is the ground truth for "do these tensors share memory".
# - Process boundaries admit exactly three moves: pickle bytes, shared-memory handles, reconst
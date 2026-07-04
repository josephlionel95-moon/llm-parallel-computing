# Chapter 00 — Python Memory, Processes & the GIL

> **Difficulty:** 🟢 Easy · **Study time:** 3–4 h · **Requires:** nothing
> **Notebook:** notebooks/ch00_python_review.ipynb · **Experiments:** experiments/ch00/

Distributed training is, at bottom, *multiple operating-system processes exchanging bytes*.
Before we touch a GPU, we must be precise about what a Python "variable" is, what actually
gets copied when data crosses a process boundary, and why PyTorch is built on processes
rather than threads. Every confusion in later chapters ("why did my model get copied?",
"why does the DataLoader hang?", "why must I use spawn with CUDA?") traces back to the
material here.

## Learning objectives

After this chapter you will be able to:

- Explain what a Python name binds to, and predict when two names alias the same object using `id()` and `is`.
- Distinguish shallow from deep copies, and map both onto tensors via `tensor.data_ptr()`.
- Describe what pickling does, what is and is not picklable, and why *every* byte that crosses a process boundary in `torch.distributed` and `DataLoader` workers is either pickled or moved through shared memory / sockets.
- Explain the GIL: what it locks, what it does not, and why threads accelerate I/O but not pure-Python compute.
- Justify why `DataLoader(num_workers>0)` and `torchrun` use **processes**, not threads.
- Contrast `fork` and `spawn` start methods and state precisely why CUDA forbids `fork`.
- Use `torch.multiprocessing` shared-memory tensors to pass data between processes without serialization.

## Intuition

Think of a Python program as a warehouse (the process's heap) full of boxes (objects).
A variable is not a box — it is a *label tied to a box with string*. Assignment
(`b = a`) ties a second label to the *same* box. This is cheap and instantaneous,
which is exactly why it is dangerous: mutate the box through one label and every
label sees the change.

Now add a second warehouse (a second process). Labels cannot stretch between
warehouses: the operating system gives every process a private virtual address
space. The pointer `0x7f3a...` in process A means nothing in process B. So to move
an object between processes you have only three options:

1. **Serialize it** — flatten the object into a stream of bytes, ship the bytes
   (pipe, socket, file), and rebuild an equivalent object on the other side.
   This is *pickling*, and it is what `multiprocessing`, `DataLoader` workers,
   and `torch.distributed`'s object collectives (`broadcast_object_list`, etc.) do.
2. **Shared memory** — ask the OS for a region mapped into *both* address spaces,
   and put the raw data (not the Python object) there. This is what
   `torch.multiprocessing` does for CPU tensors, and what CUDA IPC does for GPU
   tensors on the same machine.
3. **Recreate it** — don't ship it at all; have each process construct its own copy
   from the same recipe (e.g., every rank builds the same model from the same seed,
   or loads the same checkpoint). Distributed training uses this constantly.

Everything in chapters 06–09 is a refinement of these three moves. NCCL AllReduce is
option 2 executed over NVLink/PCIe/network with clever scheduling; checkpoint loading
is option 3; sending a config dict to all ranks is option 1.

## Theory

### Names, objects, and `id()`

CPython objects live on a private heap. Every object has an identity (its address,
reported by `id()`), a type, and a value. Names live in namespaces (module dict,
function locals) and *refer* to objects.

```
   namespace                     heap
   ┌───────────┐            ┌─────────────────────────┐
   │ a ────────┼──────────► │ list [1, 2, 3]          │
   │ b ────────┼──────────► │  (refcount = 2)         │
   └───────────┘            └─────────────────────────┘

   a = [1, 2, 3]
   b = a            # no copy: b is a  →  True
   b.append(4)      # a sees [1, 2, 3, 4]
```

Reference counting is also why CPython needs the GIL (below): incrementing
`ob_refcnt` from two threads simultaneously without a lock corrupts it.

### Shallow vs deep copy — and the tensor version

- `copy.copy(x)` (shallow): new outer container, *same inner objects*.
- `copy.deepcopy(x)`: recursively new everything.

Tensors add a third layer, and this is the one that matters for the whole course:
a `torch.Tensor` is a small Python/C++ header (shape, strides, dtype, device,
`grad_fn`, ...) plus a pointer into a separately-managed **storage** buffer.
`tensor.data_ptr()` returns the address of the first element of that buffer.

```
  Tensor header A           storage (contiguous bytes)
  ┌──────────────┐          ┌───────────────────────────────┐
  │ shape (2,3)  │───┐      │ f0 f1 f2 f3 f4 f5             │
  │ stride (3,1) │   └────► │                               │
  └──────────────┘   ┌────► └───────────────────────────────┘
  Tensor header B    │
  ┌──────────────┐   │
  │ shape (3,2)  │───┘      B = A.view(3, 2)  → same data_ptr()
  │ stride (2,1) │
  └──────────────┘
```

Operations split cleanly into two families:

| creates a **view** (same `data_ptr`) | creates a **copy** (new `data_ptr`) |
|---|---|
| `view`, `reshape`\*, `transpose`, `permute`, `[a:b]` slicing, `squeeze`, `expand`, `detach` | `clone`, `contiguous`\*, `.to(device)`, `.to(dtype)`, `torch.cat`, advanced (tensor) indexing |

\* `reshape` and `contiguous` copy only when the layout forces them to.

Why you should care, concretely:

- **`detach()` shares storage.** Mutating a detached tensor mutates the original —
  and silently corrupts autograd's saved activations.
- **DDP flattens gradients into buckets** (ch09): each parameter's `.grad` becomes a
  *view* into one big flat buffer, verified by exactly the `data_ptr()` arithmetic
  you learn here.
- **ZeRO/FSDP shard parameters** by slicing flat storages. Reasoning about "who owns
  which bytes" *is* reasoning about `data_ptr()` and offsets.

### Pickling: what actually crosses a process boundary

`pickle.dumps(obj)` walks the object graph and emits a byte stream containing
(a) import paths of classes/functions and (b) constructor state. `pickle.loads`
re-imports and rebuilds. Consequences:

- **Functions and classes are pickled by reference** (module + qualified name),
  which is why `multiprocessing` targets must be importable, top-level functions —
  a lambda has no importable name, so `mp.spawn(lambda ...)` fails.
- **Tensors pickle as (metadata + raw storage bytes)** — compact and fast, close to
  a `memcpy` plus a small header. A Python `list[float]` of the same numbers pickles
  every float as a separate object: measure the difference with
  `experiments/ch00/pickle_bench.py` (expect the list to be several times larger
  and one to two orders of magnitude slower to serialize — it is a per-element
  Python-object walk versus one buffer dump).
- **`torch.multiprocessing` registers a custom tensor pickler** that, for CPU
  tensors in shared memory and for CUDA tensors, sends only a *handle* (a shared
  memory file descriptor / CUDA IPC handle) instead of the bytes. Zero-copy.
- Open file handles, sockets, CUDA contexts, and initialized process groups are
  **not picklable** — they are OS-level state, meaningless in another process.

This is the honest definition of "distributed": *the only things that move between
ranks are byte streams (pickles, tensor buffers) and shared-memory handles.*

### Threads, processes, and the GIL

A **thread** shares its process's address space; a **process** has its own.
CPython's Global Interpreter Lock is a mutex that a thread must hold *to execute
Python bytecode*. At any instant, at most one thread runs Python code, regardless
of core count.

What the GIL does **not** lock: C code that explicitly releases it. Blocking I/O
releases it. So do NumPy's and PyTorch's heavy kernels — a `torch.matmul` on CPU
runs multithreaded C++ (OpenMP) while the GIL is free. Hence the rule of thumb:

```
                     pure-Python compute      C-extension compute / I/O
  threads            no speedup (GIL)         speedup (GIL released)
  processes          speedup                  speedup (but IPC costs)
```

`experiments/ch00/gil_bench.py` demonstrates all four quadrants.

Why does `DataLoader` use processes, then, if decoding JPEGs is C code? Because
real data pipelines are a mix of Python (dataset `__getitem__`, transforms,
sampling logic) and C, and the Python part would serialize under threads. Processes
sidestep the GIL entirely; the price is that every sample must cross a process
boundary — which PyTorch pays for using shared memory, not pickling of the raw
pixels (the tensor pickler above).

> **2025 note.** CPython 3.13+ ships an experimental free-threaded ("no-GIL")
> build, and PyTorch is gradually adding support. The architecture of distributed
> training will not change: one process per GPU remains the model, because the
> isolation is about *CUDA contexts and failure domains*, not just the GIL.

### fork vs spawn — and why CUDA requires spawn

`multiprocessing` can start a child process two ways:

- **fork** (Unix default up to Python 3.13): the OS clones the parent's entire
  address space copy-on-write. Fast; the child inherits every object *as it was at
  fork time* — including locks possibly held by other threads (deadlock risk) and
  any initialized CUDA context.
- **spawn**: start a fresh interpreter, import the main module, and send the target
  function + args to it *by pickling*. Slow start, clean state. (Windows and macOS
  default; Python 3.14 makes it the default everywhere.)

CUDA forbids fork after initialization because a CUDA context is a fat bundle of
driver state: memory maps, device handles, host threads spawned by the driver.
A forked child inherits the *memory image* of that state but not the kernel-side
threads and driver bookkeeping that make it valid — using it is undefined behavior,
and PyTorch raises:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

This is why `llmdist.utils.dist.run_distributed` uses `torch.multiprocessing.spawn`,
and why `torchrun` starts workers as fresh processes: **every rank must build its
own CUDA context from scratch.**

The spawn method's pickling requirement also explains a classic notebook failure:
functions defined *in a notebook cell* live in `__main__`, which a spawned child
re-imports — usually fine in scripts, but in Jupyter `__main__` is not importable
the same way. That is why our multi-process notebook demos define worker functions
in importable modules or keep them at top level.

### torch.multiprocessing shared memory

`tensor.share_memory_()` moves a CPU tensor's storage into a POSIX/System-V shared
memory segment (a file under `/dev/shm` on Linux). Afterwards:

- `data_ptr()` in each process differs (each maps the segment wherever it likes),
  but the *bytes* are the same physical pages.
- Sending the tensor to a child (as a `spawn` argument, or through a queue) ships
  only the handle. Writes in the child are immediately visible in the parent —
  there is no message, no copy, no synchronization. You must bring your own locking.

`DataLoader` workers return batches this way; `queue.put(tensor)` in
`torch.multiprocessing` does it automatically. `experiments/ch00/shared_memory_tensor.py`
demonstrates a child mutating a parent's tensor in place.

## Mathematics

This chapter's math is bookkeeping — but it is the bookkeeping we will do for 175 B
parameters later, so let us be exact for one tensor.

**Size of a tensor's payload.** A tensor with shape $(d_1, \dots, d_k)$ and dtype of
$s$ bytes/element occupies

$$\text{bytes} = s \cdot \prod_{i=1}^{k} d_i$$

E.g. `torch.randn(1024, 1024)` (fp32, $s=4$): $4 \cdot 2^{20} = 4\,\text{MiB}$.

**Pickle size of a tensor** ≈ payload + ~200 bytes of header/metadata. Relative
overhead: $200 / (s \cdot n) \to 0$. Serialization *time* is one buffer copy:
$t \approx \text{bytes} / B_{\text{mem}}$, with $B_{\text{mem}}$ ≈ several GB/s of
host memcpy bandwidth.

**Pickle size of a Python `list[float]`.** Every float is boxed: pickle protocol 4
stores each as an 8-byte double plus ~1 byte of opcode, and *building* the stream
touches $n$ Python objects. So size ≈ $9n$ bytes (vs $8n$ raw — acceptable), but
time is $\Theta(n)$ *interpreter* operations, not $\Theta(n)$ bytes of memcpy — two
orders of magnitude slower per byte. In memory (before pickling) it is far worse: a
`PyFloatObject` is 24 bytes plus an 8-byte pointer in the list — $32n$ bytes, a 4×
overhead, scattered across the heap (cache-hostile).

**Amdahl preview.** If a data pipeline spends fraction $f$ of its time in
GIL-serialized Python, $k$ threads give speedup at most

$$S(k) = \frac{1}{f + (1-f)/k} \;\xrightarrow{k\to\infty}\; \frac{1}{f}.$$

With $f = 0.5$, infinite threads yield 2×. Processes make $f \to$ (IPC overhead)
instead, which shared memory drives toward 0. We will reuse this exact formula for
communication/computation overlap in ch09 and scaling laws in ch25.

## Implementation

The notebook (`notebooks_src/ch00_python_review.py`) walks through five stations;
key excerpts:

**1. Aliasing and `data_ptr`:**

```python
a = torch.arange(6)
b = a.view(2, 3)          # same storage
c = a.clone()             # new storage
a[0] = 999
assert b[0, 0] == 999 and c[0] == 0
assert a.data_ptr() == b.data_ptr() != c.data_ptr()
```

**2. What pickling preserves (and destroys):**

```python
blob = pickle.dumps(model.state_dict())
sd2  = pickle.loads(blob)
# Equal values, different storages — a true copy crossed the "boundary":
assert torch.equal(sd2["head.weight"], model.state_dict()["head.weight"])
assert sd2["head.weight"].data_ptr() != model.head.weight.data_ptr()
```

Note `state_dict()` itself returns *references* (detached views) to the live
parameters — it is the pickling that copies. This distinction bites people who
"save" a state_dict to a variable, keep training, and wonder why their "backup"
changed.

**3. GIL demonstration** — a pure-Python counting loop run (a) single thread,
(b) 4 threads, (c) 4 processes, plus the same wall-clock comparison for
`torch.matmul` in 4 threads (which *does* scale, since the kernel releases the GIL).

**4. fork vs spawn** — we print `mp.get_start_method()`, then show the pickle-path
that spawn takes for the target function.

**5. Shared memory** — parent creates `t = torch.zeros(4).share_memory_()`,
spawns a child that writes `t[rank] = rank`, parent observes the writes with no
explicit receive. This is one small step from ch07's collectives.

The experiments directory contains the measured versions:

- `experiments/ch00/pickle_bench.py` — bytes and ms to pickle a tensor / a NumPy
  array / a Python list holding the same 1 M floats, plus `torch.save` for
  comparison. Prints a table.
- `experiments/ch00/gil_bench.py` — the four-quadrant thread/process ×
  Python/C-kernel benchmark, with speedup columns.
- `experiments/ch00/shared_memory_tensor.py` — cross-process in-place mutation, with
  `data_ptr()` printed on both sides, and a timing comparison of sending a large
  tensor via queue (shared-memory handle) vs `pickle.dumps` bytes.

## Profiling & measurement

What to measure and what the theory above predicts (state your hardware when you
record numbers; these scripts are CPU-only, so laptop/Colab/Kaggle CPUs all work):

1. **`pickle_bench.py`.** Expect: tensor and NumPy pickles ≈ 8 MB for 1 M float64 /
   4 MB for float32, at memcpy-like speed; the list pickle ≈ 9 MB but dramatically
   slower (interpreter-bound). The exact ratio is machine-dependent — the *shape* of
   the result (orders of magnitude, not percent) is the lesson.
2. **`gil_bench.py`.** Expect: threads ≈ 1× on pure Python (slightly *below* 1× due
   to lock contention), ≈ `min(k, cores)`× with processes; and near-linear thread
   scaling for the GIL-releasing kernel until memory bandwidth interferes.
3. **`shared_memory_tensor.py`.** Expect queue-send of a shared tensor to be
   near-constant in tensor size (handle only), while `pickle.dumps` scales linearly
   with bytes.

Read wall-clock numbers with suspicion: the OS scheduler, turbo boost, and other
notebook cells all add noise. Report medians of several runs (the scripts do).

## Common mistakes

1. **Symptom:** "I copied my config with `cfg2 = cfg` and changing `cfg2` changed the original." **Cause:** assignment aliases; no copy occurred. **Fix:** `copy.deepcopy(cfg)`, or use immutable/frozen dataclasses.
2. **Symptom:** mutating a `detach()`ed tensor corrupts training loss. **Cause:** `detach` shares storage with the original, including autograd-saved activations. **Fix:** `detach().clone()` when you intend to mutate.
3. **Symptom:** `mp.spawn` raises `Can't pickle <lambda>` or `AttributeError: Can't get attribute 'worker' on <module '__main__'>`. **Cause:** spawn pickles the target by module path; lambdas and (some) notebook-cell functions aren't importable. **Fix:** top-level function in an importable module (this is why `llmdist.utils.dist._worker` is module-level).
4. **Symptom:** `Cannot re-initialize CUDA in forked subprocess`. **Cause:** CUDA was touched before `fork` (often just `torch.cuda.is_available()` counts a device query). **Fix:** `mp.set_start_method("spawn")` / `torch.multiprocessing.spawn`, and initialize CUDA only inside workers.
5. **Symptom:** adding `num_workers=8` to a DataLoader gives no speedup on a pure-Python synthetic dataset. **Cause:** the bottleneck was the *main* process's collate/H2D copy, or per-worker startup dominates a tiny epoch — not the GIL story you expected. **Fix:** profile first (ch24); use `pin_memory=True`, `persistent_workers=True`.
6. **Symptom:** "backed up" state_dict changed after further training. **Cause:** `state_dict()` returns tensors that share storage with the live parameters. **Fix:** `copy.deepcopy(model.state_dict())` or save to disk.
7. **Symptom:** child process sees stale data in a shared tensor, intermittently. **Cause:** shared memory has no built-in synchronization; you raced. **Fix:** use an `mp.Event`/`mp.Barrier` around the handoff (collectives in ch07 include this synchronization by construction).

## Limitations & outlook

Everything here is single-machine. Shared memory does not stretch across nodes; for
that you need sockets and, for GPUs, RDMA — chapter 06 introduces the process-group
abstraction that hides this, and chapter 28 shows the real fabric. The GIL story
also shifts under free-threaded CPython, but the *process-per-GPU* design survives
for isolation reasons. Finally, we treated pickling as the only serialization; real
checkpointing (ch19) uses `torch.save` zip archives and safetensors precisely to
escape pickle's security and compatibility problems.

## Exercises

1. **(Easy)** Predict the output, then run it: `a = torch.zeros(3); b = a[:2]; b += 1; print(a)`. Explain via `data_ptr()`.
2. **(Easy)** Show that `t.reshape(...)` sometimes copies and sometimes doesn't. Construct one case of each and prove it with `data_ptr()`.
3. **(Medium)** Measure pickle size for the *same* 1 M numbers as: fp32 tensor, fp16 tensor, list of Python floats, list of Python ints in `range(1_000_000)`. Explain each size to within 10 %.
4. **(Medium)** Write a threaded and a multiprocess version of "count primes below N" (pure Python). Find N where processes beat threads by ≥ 3× on your machine. Then replace the primality test with a vectorized NumPy sieve and show threads become competitive.
5. **(Hard)** Implement a toy "parameter server": a parent holds `w = torch.zeros(k).share_memory_()`, spawns 2 workers that each add noise to disjoint halves 100 times, synchronized with an `mp.Barrier` per round. Verify no updates are lost. What happens without the barrier — and why is the answer "usually nothing visible" (hint: disjoint cache lines vs the same cache line)?
6. **(Hard)** `torch.multiprocessing` queues send CUDA tensors via IPC handles, but the *receiving* process must keep the tensor alive until the sender is done with it (see the docs' "CUDA in multiprocessing" caveats). Design an experiment that provokes a use-after-free warning/error from violating this, single-GPU is fine.
7. **(Research)** Read PEP 703 (free-threaded CPython). If the GIL disappears, which parts of the PyTorch data-loading stack would you redesign around threads, and which must stay processes? Consider CUDA contexts, failure isolation, and memory-allocator contention.

## Solutions

<details>
<summary>Solutions to exercises 1–4</summary>

**1.** Prints `tensor([1., 1., 0.])`. Basic slicing creates a view (same `data_ptr` offset by 0), and `+=` is in-place (`add_`), so the writes land in `a`'s storage.

**2.** No copy: `torch.arange(6).reshape(2,3)` (contiguous — a view suffices; `data_ptr` equal). Copy: `t = torch.arange(6).view(2,3).t()` then `t.reshape(6)` — `t` is non-contiguous, so no single (offset, strides) can describe the flat order; `data_ptr` differs. `t.view(6)` would raise instead.

**3.** fp32 tensor ≈ 4 MB + ~200 B header; fp16 ≈ 2 MB; list of floats ≈ 9 MB (8-byte double + ~1 byte opcode each, protocol ≥ 2 uses `BINFLOAT`); list of *small distinct* ints ≈ 5–6 MB (pickle stores ints ≤ 4 bytes as `BININT` variants with 1-byte opcode — most values below 2²⁴ take 4+1 or 2+1 bytes). Key point: containers pay per element; buffers pay per byte.

**4.** Pure-Python primality is bytecode-bound → threads serialize on the GIL (expect ≈ 1×, processes ≈ cores×). N ≈ 200 k–500 k usually suffices for a ≥ 3× gap on a 4-core machine. The NumPy sieve spends its time in C with the GIL released, so 4 threads on 4 sieve-chunks scale until memory bandwidth saturates — typically 2–3.5× on 4 cores.

</details>

<details>
<summary>Notes toward exercises 5–7</summary>

**5.** Disjoint halves mean no logical race on *values*, so "without barrier" usually still ends correct; the barrier is protecting the *round structure* (worker A must not start round r+1 reading half-updated state if your update rule ever reads the other half). If both halves share a cache line at the boundary, you get false sharing: correctness preserved (cache coherence), throughput reduced. This foreshadows why collectives define synchronization points, not just data movement.

**6.** Sketch: producer creates a CUDA tensor, `queue.put(t)`, immediately `del t` and `torch.cuda.empty_cache()`, then allocates/overwrites new tensors while the consumer sleeps before reading. The IPC-shared memory can be reused by the producer's allocator once the producer-side refcount drops, so the consumer reads garbage or triggers `CUDA error: an illegal memory access`. The fix in real code: keep the sent tensor alive until the consumer confirms receipt (this is what DataLoader's pin-memory thread choreography guarantees).

**7.** Defensible answer: transforms/collate could become threads (share the decoded buffers, no IPC), but the *GPU-owning* training workers stay one process per GPU — a crashed rank must not take down others (elastic training, ch28), and per-process CUDA contexts/allocators avoid cross-tenant fragmentation and lock contention. Cite: PEP 703's discussion of single-threaded performance tax; NVIDIA's one-context-per-process guidance.

</details>

## Interview questions

1. What does `id(x)` return in CPython, and why can two objects have the same `id` at different times?
2. `b = a.transpose(0, 1)` — does `b` share memory with `a`? How do you verify it in one line?
3. Explain precisely what happens, step by step, when `multiprocessing` with the spawn method calls your worker function in a child process.
4. Why can't you pickle a lambda? Why does this matter for `mp.spawn` and DataLoader workers?
5. The GIL is released during `torch.matmul` on CPU. Why does PyTorch's DataLoader still use processes instead of threads?
6. Why exactly does CUDA prohibit `fork` after initialization? What error do you see and how do you fix it in a torch program?
7. When `torch.multiprocessing` sends a CPU tensor through a queue, what bytes actually travel through the pipe?
8. Your teammate stores "a backup" with `backup = model.state_dict()` and later finds it changed. Explain, and give two correct alternatives.
9. A list of 1 M Python floats and a 1 M-element fp64 tensor hold the same numbers. Compare their RAM footprint and their pickled size, with estimates.
10. In DDP, each rank builds the model independently and only gradients are communicated. Which of the three "cross-process data movement" strategies is that, and why is it chosen over shipping the model?

## Summary

- A Python variable is a label; assignment aliases, it never copies.
- A tensor = header + storage; views share `data_ptr()`, copies don't — this one check explains detach bugs, DDP buckets, and FSDP shards alike.
- Crossing a process boundary means pickling (bytes), shared memory (handles), or reconstruction (recipes). Distributed training uses all three, deliberately.
- Tensors pickle as one buffer dump; Python containers pickle per element — orders of magnitude apart in time.
- The GIL serializes Python *bytecode*, not C kernels or I/O; threads help the latter, never the former.
- DataLoader and torchrun use processes to escape the GIL and to isolate CUDA contexts and failures.
- `fork` clones state that CUDA cannot survive; `spawn` re-imports and pickles — hence top-level worker functions.
- `share_memory_()` gives zero-copy cross-process tensors with **no** synchronization; you supply barriers.
- Amdahl's law bounds thread speedup by 1/f, where f is the GIL-serialized fraction — the same law returns for communication overhead.
- Every "magic" of later chapters is one of these mundane mechanisms, scheduled well.

## References

See `references/ch00.md` for the annotated list. Highlights:

- CPython docs, *Data model* & `copy` module — the authoritative statement of names/objects and shallow/deep copy semantics.
- `pickle` module docs — read "What can be pickled" before ever debugging multiprocessing.
- PyTorch docs, *Multiprocessing best practices* — fork/spawn, CUDA IPC caveats, shared-memory strategy; the single most load-bearing page for this chapter.
- PEP 703 — the case for free-threaded CPython; explains what the GIL protects better than most textbooks.
- Beazley, *Understanding the Python GIL* (PyCon 2010) — the classic demonstration of GIL thrashing with threads.

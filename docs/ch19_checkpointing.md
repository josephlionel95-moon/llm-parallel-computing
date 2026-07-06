# Chapter 19 — Checkpointing: Exact Resume, Atomic Saves, Sharded State

> **Difficulty:** 🟡 Medium · **Study time:** 4–5 h · **Requires:** chapters 04, 06, 08
> **Notebook:** notebooks/ch19_checkpointing.ipynb · **Experiments:** experiments/ch19/

## Learning objectives

You will be able to...

- Enumerate the **seven pieces of training state** and predict exactly what goes wrong when each one is omitted from a checkpoint (weights-only resume, LR reset, RNG drift, data replay...).
- Demonstrate a **bitwise-exact resume**: train N steps straight vs. train N/2, kill the process, resume, train N/2 — and show the two loss sequences are identical to the last bit.
- Derive the **checkpoint size ledger** (12 bytes/param for Adam+FP32, 14–18 for mixed precision) and verify it against `os.path.getsize`.
- Write an **atomic save** (tmp + fsync + `os.replace`) and explain which failure mode each of the three ingredients defends against.
- Derive the **Young–Daly optimal checkpoint interval** τ\* = √(2δM) from a cost model, and apply it to Colab/Kaggle preemption in practice.
- Explain why `torch.load` is code execution, and what `weights_only=True` actually restricts.
- Checkpoint correctly under **DDP** (rank-0 save, barriers, `map_location`) and explain why ZeRO/FSDP make rank-0 saving impossible, motivating **sharded checkpoints** (`torch.distributed.checkpoint`) and **resharding**.
- Build a **preemption-proof training loop** for free-tier hardware: rotation, auto-resume, off-machine sync.

## Intuition

Training is a very long computation on very unreliable hardware. A Colab session dies
after at most 12 hours — often sooner, without warning. Kaggle gives you 30 GPU-hours a
week in slices. Even on a real cluster, at 10,000 GPUs *something* fails every few
hours; at GPT-4 scale, hardware failure is not an "if" but a scheduled visitor. The
question is never "will training be interrupted?" but "how much work do I lose when it
is?"

A checkpoint is the answer: a serialization of the training process's *complete* state,
such that a fresh process can load it and continue **as if the interruption never
happened**. The words doing the work are *complete* and *as if*.

The analogy that carries this chapter: **a training run is a save file in a video
game**. Saving only the model weights is like saving your character's level but not
your position, inventory, or quest progress — you reload and you're level 47 but back
at the starting village with an empty bag. Concretely, the "inventory" of a training
run has seven slots:

```
 ┌──────────────────────────────── one training run's state ─────────────────────────────┐
 │ 1. model parameters+buffers   the thing you actually want                             │
 │ 2. optimizer state            Adam's m, v — 2× larger than the model itself!          │
 │ 3. LR scheduler state         where you are in warmup/decay                           │
 │ 4. AMP GradScaler state       current loss scale + growth counter (FP16 only)         │
 │ 5. RNG streams                python, numpy, torch-CPU, torch-CUDA                    │
 │ 6. data position              which samples have been consumed this epoch             │
 │ 7. progress counters          step, epoch, best-val-loss, tokens seen                 │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

Slot 2 is the one that surprises people twice. First in size: for Adam in FP32 the
optimizer state is **two full copies of the model** (chapter 04's ledger). Second in
consequence: resume without it and Adam's moment estimates restart from zero — the
effective per-parameter learning rates are wrong for thousands of steps, and the loss
curve shows a visible bump exactly at the resume point. That bump is the signature of an
incomplete checkpoint, and you will learn to read it like an X-ray.

There is also a second, quieter problem this chapter solves: **the save itself can be
interrupted.** A checkpoint written with a bare `torch.save(state, path)` that gets
killed halfway leaves a truncated zip file *at the path of your only good checkpoint*.
You lost the run not despite checkpointing but *because* of it. Databases solved this
decades ago; we steal their answer (write-ahead + atomic rename).

Finally, scale: when the state is sharded across GPUs (ZeRO/FSDP, chapters 15–16), no
single rank *has* the full state, so "gather it on rank 0 and save" either OOMs or
takes so long that (by Young–Daly) you can afford almost no checkpoints. The fix is to
let every rank save its own shard and make the *loader* smart enough to reassemble —
even onto a different number of GPUs. That is `torch.distributed.checkpoint`, and it is
the state of the art in every serious training stack today.

## Theory

### What `state_dict()` actually is

Every `nn.Module` maintains an ordered mapping from *qualified names* to tensors:

```
MiniGPT.state_dict()
  "tok_emb.weight"            → FloatTensor (8192, 256)
  "blocks.0.attn.qkv.weight"  → FloatTensor (768, 256)
  "blocks.0.ln1.weight"       → FloatTensor (256,)
  ...
  "ln_f.weight"               → FloatTensor (256,)
```

Three properties matter:

1. **It contains parameters AND buffers.** Buffers are tensors that are part of the
   model's state but not trained by gradient descent — BatchNorm's `running_mean`, a
   causal mask, RoPE frequency tables. `model.parameters()` misses them; `state_dict()`
   does not. This is why you serialize state dicts, never parameter lists.
2. **The values are references, not copies.** `sd = model.state_dict()` gives you views
   of the live training tensors. If the optimizer steps while you are still serializing
   (async writers, chapter's Limitations section), the file gets a *torn* mixture of
   pre-step and post-step weights. Our `training_state_dict()` clones to CPU for
   exactly this reason.
3. **Loading is name-matching.** `load_state_dict` copies values into existing tensors
   by key. Wrappers change keys: DDP prepends `module.`, `torch.compile` prepends
   `_orig_mod.`. A checkpoint saved through a wrapper loads *only* into an identically
   wrapped model — hence our `unwrap()` discipline: **always save the bare module**.

The optimizer's `state_dict()` is shaped differently — parameters are identified by
*integer index* into the param-groups list, not by name:

```
opt.state_dict()
  "state":        {0: {"step": t, "exp_avg": m₀, "exp_avg_sq": v₀}, 1: {...}, ...}
  "param_groups": [{"lr": 3e-4, "betas": (0.9, 0.999), "params": [0, 1, ...]}]
```

Consequence: optimizer state is only loadable into an optimizer constructed over
parameters **in the same order**. Reorder your layers, or filter parameters differently,
and the moments silently attach to the wrong tensors — no error, wrong training. (This
is also why ZeRO's sharded optimizer needs a real resharding story, not key-matching.)

### The seven pieces, and the failure mode of forgetting each

| # | Piece | Forgotten ⇒ symptom at resume |
|---|-------|-------------------------------|
| 1 | model params+buffers | (you wouldn't) — but *buffers* get lost via `parameters()`-based saving: BN stats reset |
| 2 | optimizer state | loss bump; Adam moments re-warm over ~1/(1−β₂) ≈ 1000 steps; bias correction restarts as if t=0 |
| 3 | scheduler state | LR snaps back to warmup — often *increases* 10×, sometimes diverges |
| 4 | GradScaler state | scale re-calibrates from 2¹⁶: a burst of skipped steps (FP16 only) |
| 5 | RNG streams | dropout masks & augmentations differ ⇒ resumed curve ≠ original curve (close, not equal — nondeterminism you can't distinguish from a bug) |
| 6 | data position | first epochs replay the same leading batches ⇒ mild overfitting to them; with shuffling, sample-without-replacement guarantees break |
| 7 | step counter | scheduler/logging/eval cadence desynchronize; "best" model tracking corrupts |

The notebook demonstrates #2, #3 and #5 empirically — each has a recognizable loss-curve
signature.

### Atomicity: the save must survive its own death

A `torch.save` that is killed mid-write (preemption, OOM-killer, quota) leaves a
truncated file. The recipe that makes a save all-or-nothing:

```
                write               fsync              os.replace
   state ────────────▶ ckpt.pt.tmp ───────▶ (on disk) ────────────▶ ckpt.pt
                            │                   │                      │
   crash here: ckpt.pt      │  crash here:      │   crash here or     │
   untouched ✓              │  ckpt.pt          │   later: new file   │
                            │  untouched ✓      │   complete ✓        │
```

- **tmp in the same directory** — `os.replace` is atomic only within one filesystem;
  `/tmp` → Drive crosses filesystems and degrades to copy+delete (not atomic).
- **fsync before rename** — without it, the *rename* can hit disk before the *data*
  does; after a power loss you get a complete-looking, correctly-named, empty file.
  This ordering bug bit real filesystems (ext4's infamous zero-length-file era).
- **`os.replace`, not `shutil.copy`** — rename is a single metadata operation; readers
  see the old file or the new file, never a hybrid.

Then never write two consecutive checkpoints to *the same* final path anyway: keep the
last k (rotation). Disk is bounded, paranoia should be too.

### `torch.save`/`torch.load` under the hood

`torch.save` writes a **zip archive** (open one with `unzip -l`!): one entry
`data.pkl` — a pickle of the object graph where each tensor is a *stub* pointing to —
plus one entry per storage (`data/0`, `data/1`, ...) holding raw bytes. Two practical
consequences:

- **Loading a checkpoint is running a program.** Pickle can encode "call this
  function with these arguments" (`__reduce__`), so `torch.load` of an untrusted file is
  arbitrary code execution. `weights_only=True` (default since torch 2.6) swaps in a
  restricted unpickler that only builds tensors, containers, and primitives. Design your
  checkpoint schema to pass this loader — ours does (that is why `rng_state_dict`
  launders numpy state through tensors).
- **`map_location` matters.** Each storage remembers its device; a checkpoint saved
  from `cuda:0` deserializes onto `cuda:0` — on an 8-rank node, all eight processes
  loading one file would put eight copies on GPU 0. Rule: **save from CPU clones, load
  with `map_location="cpu"`**, and let `load_state_dict` copy into tensors that already
  live on the right device. Bonus: `torch.load(..., mmap=True)` maps the file instead of
  reading it — with CPU-resident storages, load time becomes O(what you touch).

### Distributed checkpointing

**DDP (chapter 08–09):** all ranks hold identical replicas, so the checkpoint problem
is only a coordination problem:

```
rank 0:  state = training_state_dict(...)  ──save──▶ ckpt.pt
rank 1:  (skip)                                          │
all:     dist.barrier()   ◀──────── everyone waits ──────┘
all:     load_checkpoint(..., map_location="cpu")        (identical file, own devices)
```

Why the barrier: without it, rank 1 can finish the job (or start reading the file)
while rank 0 is still writing. Why everyone loads the same file: cheaper than
broadcasting, and exercises the same code path as single-GPU.

**ZeRO/FSDP (chapters 15–16):** now rank i holds only shard i of the optimizer state
(ZeRO-1/2) or even of the parameters (ZeRO-3/FSDP). Two bad options and a good one:

- *Gather-to-rank-0:* materializes the full state on one host — the exact memory wall
  sharding existed to avoid, plus one node's disk bandwidth serializes the whole write.
- *Naive per-rank files* (`ckpt_rank{i}.pt`): fast, but welds the checkpoint to one
  world size and sharding layout. Your 8-GPU run's checkpoint cannot resume on the 2×T4
  Kaggle box.
- *Sharded checkpoint with a global index* — `torch.distributed.checkpoint` (DCP):
  every rank writes the shards it owns, in parallel, plus one metadata file recording
  which slices of which logical tensor live where:

```
checkpoint_dir/
  .metadata                "blocks.0.attn.qkv.weight" → [rank0: rows 0–383 @ __0_0.distcp,
                                                         rank1: rows 384–767 @ __1_0.distcp]
  __0_0.distcp             rank 0's shards (raw bytes)
  __1_0.distcp             rank 1's shards
```

On load, each rank declares what it needs (its *local* state dict — possibly under a
different world size), DCP consults the metadata, and every rank reads exactly the
byte-ranges it needs. **Resharding** (save on 2 ranks, load on 1, or 8→512) falls out
of this design for free: the checkpoint stores *logical tensors + layout metadata*, not
"whatever rank i happened to hold". Save-time scales with the slowest shard, not the
sum; this is also the substrate for async checkpointing (snapshot to CPU, write in a
background thread while training continues — see Limitations).

The experiment `sharded_dcp.py` runs exactly this on CPU/Gloo: save with world=2,
resume with world=1, verify every tensor.

## Mathematics

### 1. The checkpoint size ledger

Let N = number of parameters. FP32 training with Adam:

```
model weights            4N bytes
Adam exp_avg (m)         4N
Adam exp_avg_sq (v)      4N
───────────────────────────────
checkpoint              12N bytes  (+ ~KB of scheduler/RNG/counters — noise)
```

Derivation is just bookkeeping — the point is what it *implies*. Mixed precision with
FP32 master weights (chapter 04): the checkpoint stores master weights (4N) + m (4N) +
v (4N) = 12N; add the FP16 working copy if you save it too (+2N = 14N). GPT-3 scale,
N = 175·10⁹: **2.1 TB per checkpoint**. At a very generous 1 GB/s to stable storage
that is a 35-minute pause — per checkpoint. Numbers like these are why sharded and
asynchronous checkpointing exist; they are not conveniences, they are what makes
checkpointing affordable at all.

For this course's MiniGPT (vocab 8192, block 128, 4 layers, d_model 256):
N ≈ 5.3M ⇒ predicted checkpoint ≈ 12 × 5.3M ≈ 64 MB. `checkpoint_cost.py` measures the
actual file and the prediction lands within ~2% (zip framing + non-tensor state).

### 2. Young's optimal checkpoint interval, from scratch

Question: checkpoint too often and you pay the save cost δ constantly; too rarely and
each failure destroys hours. Where is the optimum?

**Setup.** Checkpoint cost δ seconds; failures arrive randomly with mean time between
failures M (memoryless/Poisson — reasonable for surprise preemptions); we checkpoint
every τ seconds of useful work. We want the overhead rate H(τ): wasted seconds per
useful second.

**Term 1 — checkpoint tax.** Every τ seconds of work we spend δ saving:

```
H_ckpt(τ) = δ / τ
```

**Term 2 — expected rework.** When a failure strikes, we lose the work since the last
checkpoint. For a memoryless failure process, the strike point is uniform within the
current interval, so the expected loss is τ/2 (add the restart/reload cost R if you
like; it does not depend on τ, so it will not affect the optimum). Failures occur at
rate 1/M, so per unit time:

```
H_fail(τ) = (τ/2 + R) / M
```

**Total and optimum.**

```
H(τ) = δ/τ + τ/(2M) + R/M

dH/dτ = −δ/τ² + 1/(2M) = 0
   ⇒ τ*² = 2δM
   ⇒ τ* = √(2δM)          (Young 1974; Daly 2006 refines for δ, τ ≈ M)
```

Second derivative 2δ/τ³ > 0 — a genuine minimum. Sanity-check the shape: τ\* grows with
the *square root* of both quantities. Checkpoints 4× cheaper ⇒ checkpoint 2× more often.
Hardware 4× flakier (M/4) ⇒ checkpoint 2× more often — not 4×. The square root is doing
real work: it says the response to changing conditions is always gentler than
proportional.

**Plug in your actual life.** Colab free tier: δ ≈ 30 s (60 MB save + Drive sync),
M ≈ 4 h = 14,400 s (empirically, sessions rarely survive a working day):

```
τ* = √(2 · 30 · 14400) ≈ 930 s ≈ 15 minutes
H(τ*) = √(2δ/M) ≈ 6.5%  overhead — and you never lose more than ~15 min of work
```

Checkpoint every step instead (τ = 2 s, δ = 30 s): H = 1500% — training 15× slower than
useful. Checkpoint hourly: expected loss per failure 30 min, overhead ~13%. The formula
beats both instincts. `optimal_interval.py` verifies this with 20,000 simulated
preempted runs — the empirical minimum lands on √(2δM) within simulation noise.

**Validity limits.** The derivation assumed δ ≪ τ ≪ M. If M approaches δ (machine dies
every few checkpoint-times), no interval saves you — Daly's higher-order version says
"get better hardware". And *fixed-horizon* preemption (Colab's hard 12 h wall) is not
memoryless: the right response to a known wall is a checkpoint *just before* it, on top
of the Poisson schedule for surprises. Exercise H2 makes you quantify this with real
trace shapes.

## Implementation

All library code: `src/llmdist/checkpoint.py`. Walkthrough of the load-bearing parts —
run the notebook alongside.

**Capturing state — `training_state_dict(model, opt, scheduler, scaler, step, extra)`**
assembles the seven pieces. Key lines:

```python
"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
```

`detach` (no autograd history in the file), `cpu` (loadable anywhere, frees the save
from `map_location` traps), `clone` (a *snapshot* — state_dict values are references,
and we refuse to serialize tensors that training might mutate mid-write). RNG capture
launders numpy's ndarray through a torch tensor so the whole file passes
`weights_only=True`.

**Atomic write — `save_checkpoint(path, state)`** is ten lines implementing the
tmp→fsync→replace diagram from Theory. Read it and map each line to the failure it
kills. `latest_checkpoint()` deliberately ignores `*.tmp` files — a leftover tmp *is*
the crash artifact, never a resume candidate.

**Restore — `load_checkpoint(path, model, opt, ...)`** loads with
`weights_only=True, map_location="cpu"` and restores exactly the components you pass.
It returns the raw dict so the trainer reads `state["step"]` and its own extras.

**Rotation — `RotatingCheckpointer`** keeps `keep_last` newest `step_XXXXXXXX.pt`
files, fires on steps and/or wall-clock seconds, and `protect()`s files (best-val) from
pruning. Note `maybe_save(state_fn, step)` takes a *callable*: assembling the state
clones every tensor — pay that only when a save is due.

**The resumable loop** (`experiments/ch19/resume_exactness.py`) is the pattern to
internalize — auto-resume is just "look before you loop":

```python
ckpt = latest_checkpoint(ckpt_dir)                      # None on first launch
start = 0
if ckpt:
    state = load_checkpoint(ckpt, model, opt, scheduler)
    start = state["step"]                               # RNG already restored

for step in range(start, total_steps):
    loss = train_step(model, opt, batch(step))
    scheduler.step()
    rot.maybe_save(lambda: training_state_dict(model, opt, scheduler,
                                               step=step + 1), step + 1)
```

The experiment then *proves* the loop correct: run A trains 100 steps straight; run B
trains 50, is killed, relaunches, trains 50 more. Losses 51–100 must satisfy
`a == b` — **bitwise**, not `allclose`. On CPU with fixed seeds there is no excuse for
a single differing bit; `allclose` would hide exactly the small-but-real divergences
(missed RNG stream, optimizer moment drift) this chapter exists to eliminate. When you
change anything about a training loop, this equality test is the cheapest correctness
oracle you own.

**Off-machine sync** (the last Colab piece): a checkpoint on preempted-VM disk dies
with the VM. Rotation dir on Drive (`/content/drive/MyDrive/ckpts/run1`) or push to HF
Hub (`huggingface_hub.upload_file`) after each save. Atomicity caveat from Theory
applies: Drive is a different filesystem, so write locally, then *copy* — the copy can
tear, but `latest_checkpoint` on Drive still sees the previous complete file, and the
local original is intact.

**Sharded** (`experiments/ch19/sharded_dcp.py`): the DCP API in one breath —

```python
import torch.distributed.checkpoint as dcp
dcp.save({"model": model.state_dict()}, checkpoint_id=ckpt_dir)   # every rank, parallel
dcp.load(state,                          checkpoint_id=ckpt_dir)  # in-place into `state`
```

Two differences from `torch.save` worth staring at: `save` is **collective** (every
rank participates and writes its own shards — with plain DDP replicas DCP dedups so
rank 0 writes), and `load` is **in-place** — you build the state dict with
correctly-shaped destination tensors first, DCP fills them from whatever shard layout
the metadata describes. That inversion (declare what you need, let the reader route
bytes) is what makes world-size changes between save and load a non-event.

## Profiling & measurement

What to measure, and the numbers we get (CPU-only CI box; your T4 numbers in brackets
where they differ meaningfully):

| Quantity | How | Expected |
|---|---|---|
| Checkpoint size | `os.path.getsize` vs 12N ledger | within ~2% of 12N (zip framing) |
| Save time, cold | wrap `save_checkpoint` in `perf_counter` | ~0.4 s for 64 MB local disk; dominated by fsync |
| Save time, GPU→CPU clone | time `training_state_dict` alone | [T4: ~50 ms for 5M params — PCIe copy, not the bottleneck] |
| Load time | time `load_checkpoint` | ≈ save time; `mmap=True` ⇒ near-zero until tensors touched |
| Resume exactness | `resume_exactness.py` | max |Δloss| = 0.0 exactly, 50/50 steps bitwise equal |
| Rotation | ls after 20 saves, keep_last=3 | 3 files + protected ones |
| Simulated overhead vs τ | `optimal_interval.py --plot` | U-curve, minimum at √(2δM) ± noise |

Reading the U-curve is the skill: left wall = checkpoint tax δ/τ (hyperbola), right
wall = rework τ/2M (line), the floor sits at overhead √(2δ/M) — for the Colab numbers
above, ~6.5%. If your measured floor is much higher, your δ estimate is wrong (Drive
sync time counts!).

## Common mistakes

1. **Weights-only resume.** *Symptom:* loss bump at resume, recovers over ~1k steps.
   *Cause:* fresh Adam — m, v = 0 and bias-correction t = 0 make early updates
   mis-scaled per-parameter. *Fix:* save/load optimizer state; run the bitwise test.
2. **Scheduler not saved.** *Symptom:* LR jumps (often back into warmup) at resume;
   possible divergence. *Cause:* scheduler recreated at step 0. *Fix:* piece #3; or
   compute LR as a pure function of the global step (then the step *is* the state).
3. **`module.`/`_orig_mod.` key mismatch.** *Symptom:* `Missing keys / Unexpected keys`
   — or, with `strict=False` sprinkled in to "fix" it, a silently untrained model.
   *Cause:* saved through DDP/compile wrapper. *Fix:* `unwrap()` before saving; treat
   `strict=False` as a code smell requiring a comment justifying it.
4. **All ranks load onto GPU 0.** *Symptom:* rank 0 OOMs at resume (fine at save).
   *Cause:* storages remember their device; default load re-materializes on `cuda:0`
   for every process. *Fix:* `map_location="cpu"` (or per-rank device map).
5. **Truncated checkpoint after preemption.** *Symptom:* `PytorchStreamReader failed:
   central directory not found`. *Cause:* killed mid-`torch.save`, no atomicity.
   *Fix:* tmp+fsync+`os.replace`; `latest_checkpoint` skips `.tmp`.
6. **Torn state from async/interleaved saving.** *Symptom:* resumed model is neither
   step-t nor step-t+1; sometimes NaNs. *Cause:* serializing live references while
   `opt.step()` runs. *Fix:* clone-to-CPU snapshot first (our `training_state_dict`),
   then write.
7. **Non-`weights_only` checkpoints / pickle trust.** *Symptom:* none — until you load
   a malicious file and it runs code. *Cause:* pickle's `__reduce__`. *Fix:* schema of
   tensors+primitives only; load with `weights_only=True`; treat downloaded checkpoints
   as untrusted input.
8. **RNG restored on a different device topology.** *Symptom:* exact on CPU, drift on
   GPU resume (or crash restoring 2 CUDA streams on 1 GPU). *Cause:* CUDA RNG state is
   per-device. *Fix:* restore the streams that exist (our loader truncates gracefully);
   accept documented nondeterminism across topology changes.
9. **Checkpointing too often.** *Symptom:* it/s craters; Drive quota exhausted.
   *Cause:* ignoring the δ/τ term. *Fix:* Young–Daly; measure δ honestly (include
   sync), solve √(2δM), round to something you can remember.
10. **Same-path overwrite.** *Symptom:* one bad save (disk full mid-write) destroys the
    only checkpoint. *Cause:* `torch.save(state, "ckpt.pt")` every time. *Fix:*
    step-numbered files + rotation; the newest checkpoint is never the only one.

## Limitations & outlook

- **Synchronous saves stall training.** Even atomic saves block the loop for δ. The
  production fix is a two-phase async checkpoint: snapshot to CPU (fast, must be
  synchronous — mistake #6), then serialize/upload in a background thread while
  training continues. `torch.distributed.checkpoint.async_save` implements this;
  exercise H1 has you build the core mechanism yourself.
- **Our checkpoints are whole-tensor.** True ZeRO-3/FSDP training needs the sharded
  path everywhere; this chapter shows DCP on replicated state (the mechanism, minus the
  sharding), and chapters 15–16 supply sharded state to feed it. Resharding across
  *tensor-parallel layout changes* (ch10) is harder still — the metadata must describe
  slicing along arbitrary dims; DCP's `DTensor` integration is the current answer.
- **Data-position checkpointing is shallow here.** We restore RNG, which fixes the
  shuffle, and skip consumed batches by fast-forwarding the sampler — fine for one
  epoch on synthetic data. Real streaming pipelines (multi-worker, prefetching,
  mid-shard) need stateful loaders: `torchdata`'s `StatefulDataLoader`, Meta/HF
  ecosystems all grew one. The interface (`state_dict()`/`load_state_dict` on the
  *loader*) mirrors everything you built here.
- **Frequency floor at frontier scale.** When δ is minutes even with parallel writes,
  τ\* is hours — unacceptable loss windows. Frontier systems checkpoint *hierarchically*:
  to neighboring GPU/host RAM every few minutes (Gemini-style peer redundancy), to
  parallel FS hourly, to object store daily. Same Young–Daly math, three nested (δ, M)
  pairs — a nice research-flavored exercise in multi-level optimization.
- **A checkpoint is not provenance.** Bitwise resume needs identical code, torch
  version, CUDA kernels, and world size. Serious runs record git SHA + config + env
  inside `extra` and refuse to resume on mismatch. We store `torch_version`; extend it.

## Exercises

**E1 (easy).** `state_dict()` values are references. Predict, then verify: save a
state dict *without* cloning, take 10 optimizer steps, compare the dict's tensors to
the freshly-saved file. Now explain why `torch.save(model.state_dict(), p)` *happens*
to be safe in a stopped loop but `training_state_dict` still clones.

**E2 (easy).** Extend `training_state_dict`/`load_checkpoint` to round-trip a
`torch.amp.GradScaler` and prove (bitwise test on a CUDA box, or by inspecting scale
values on CPU) that the loss scale and growth counter survive.

**E3 (medium).** Add `save_best(state, metric)` to `RotatingCheckpointer`: keep the
best-validation checkpoint outside rotation via `protect()`. Property to preserve:
`keep_last` *rotating* files regardless of how many are protected. Add a test.

**E4 (medium).** Break the resume-exactness test on purpose, three ways: (a) drop
optimizer state, (b) drop RNG state, (c) drop scheduler state. For each, plot the
resumed curve vs the straight-through curve and write one sentence describing the
signature. You are building your future debugging pattern-library.

**E5 (medium).** Measure δ honestly on your setup: `training_state_dict` time,
`save_checkpoint` time, Drive/HF sync time, for MiniGPT at d_model ∈ {256, 512, 1024}.
Fit δ(N) and give the Young–Daly τ\* for each size at M = 4 h.

**H1 (hard).** Implement `async_save(state_fn, path)`: snapshot synchronously
(clone-to-CPU), serialize+fsync+rename in a `threading.Thread`. Guarantee at most one
in-flight save (drop or queue?) and prove via the bitwise test that a checkpoint taken
*during* training equals one taken with training paused at the same step. Measure the
stall: sync vs async, steps/s around the save.

**H2 (hard).** Colab's failure process is not Poisson: hard wall at 12 h plus random
mid-session preemptions. Simulate a mixture (uniform-random death with M = 4 h, plus
certain death at 12 h) and find the empirically optimal *schedule* (hint: it is not a
constant τ — what should happen near the wall?). Compare total useful work vs plain
Young–Daly over 100 simulated days.

**R (research-level).** Hierarchical checkpointing: levels (RAM, local SSD, cloud)
with costs (δ₁ < δ₂ < δ₃) and coverage (RAM survives process death but not VM death,
...). Formalize as nested Young–Daly, derive or numerically find (τ₁, τ₂, τ₃) for
Colab-realistic parameters, and simulate. When does the RAM level stop paying for
itself? Compare your conclusions to the Gemini (2023) peer-redundancy design.

## Solutions

<details>
<summary>E1 — references vs clones</summary>

The un-cloned dict's tensors *are* the live weights — after 10 steps they differ from
the file written earlier (the file serialized the values at write time; the in-memory
dict kept drifting). `torch.save(model.state_dict(), p)` inside a stopped loop is safe
because serialization finishes before any further mutation. `training_state_dict`
clones anyway because it cannot know the caller has stopped — H1's async writer is
exactly the caller that hasn't.
</details>

<details>
<summary>E2 — GradScaler round-trip</summary>

Already plumbed: pass `scaler=` to both functions. The state is
`{"scale": float, "growth_tracker": int, ...}`. Verify by forcing a nonstandard scale
(`scaler.update(2.0**10)`), saving, loading into a fresh scaler, and asserting
`scaler.get_scale() == 1024.0`. On CUDA, the bitwise test with AMP enabled is the full
proof — without scaler restore it fails within a few steps (different skipped-step
pattern).
</details>

<details>
<summary>E3 — sketch</summary>

`save_best` saves to `best_step_{step}.pt` (a name `CKPT_PATTERN` does not match ⇒
naturally outside rotation) *and* calls `self.protect(path)` for belt-and-braces; keep
a `self.best_metric` float; overwrite semantics = save-then-delete-previous-best, both
atomic ops. Test: 20 saves with random metrics ⇒ exactly `keep_last` rotating files +
1 best file, and the best file's metric equals `min(metrics)`.
</details>

<details>
<summary>E4 — expected signatures</summary>

(a) optimizer dropped: bump at resume, decaying over ~1/(1−β₂) steps — moments
re-warming. (b) RNG dropped: no bump, but curves *diverge point-wise* immediately
(different dropout masks) while tracking the same trend — the two runs are now merely
statistically equivalent. (c) scheduler dropped: discontinuity in LR ⇒ kink in the
loss slope; with warmup restart, a brief plateau or spike. Note (b) is invisible in
loss *quality* and only caught by the equality test — that is the point of the test.
</details>

<details>
<summary>E5 — what you should find</summary>

δ is dominated by bytes: 12N × write bandwidth, plus a near-constant fsync/sync floor.
Drive sync usually multiplies local δ by 3–10×. Typical: d256 ≈ 0.5 s local / 4 s
synced; d1024 ≈ 16× more bytes ⇒ τ\*(synced, M=4h) goes from ~15 min to ~60 min. The
qualitative lesson: **δ scales with N, so τ\* scales with √N** — bigger models
checkpoint less often, which is why frontier systems attack δ itself (async, sharded,
hierarchical).
</details>

<details>
<summary>H1 — the two invariants</summary>

(1) Snapshot before returning control: the clone-to-CPU must complete synchronously —
this bounds the stall at PCIe-copy time (~50 ms for 60 MB), the file I/O overlaps
training. (2) One writer: a `threading.Lock` (or single-worker executor) + "latest
snapshot wins" queue-of-one; two concurrent writers to step-ordered files are safe for
atomicity but can invert `latest_checkpoint` ordering vs wall-clock. Bitwise proof
passes because the snapshot pins values at step t regardless of what training does
during serialization. Measured: sync save stalls one full δ; async stalls only the
snapshot (~10–20× less).
</details>

<details>
<summary>H2 — the shape of the answer</summary>

Near a known wall, expected-loss-per-second rises as remaining time shrinks; the
optimal schedule shortens intervals as t → 12 h and places one final checkpoint just
before the wall (cost δ, saves τ/2 on average — worth it whenever δ < remaining/2).
A simple "Young–Daly + forced save at 11:55" captures ≥95% of the optimal schedule's
useful work in simulation; fully optimal variable schedules buy little more. This
mirrors the real-world advice: handle *known* deadlines explicitly, model only the
*surprises* stochastically.
</details>

## Interview questions

1. A colleague's loss curve shows a bump exactly at every resume, recovering over
   ~1000 steps. Diagnose it, and explain where the ~1000 comes from.
2. Why must the tmp file live in the same directory as the final checkpoint, and what
   silently changes if it does not?
3. What does `fsync` buy that `flush` does not, and which historical filesystem
   behavior makes rename-before-writeback dangerous?
4. `torch.load` of an untrusted file is code execution — explain the mechanism, and
   what `weights_only=True` restricts to prevent it.
5. Your 8-GPU job resumes and rank 0 OOMs while all other ranks are fine. What
   happened, and what one-argument fix prevents it?
6. Derive τ\* = √(2δM) from the two competing overhead terms. Then: your checkpoints
   get 4× cheaper — how much more often should you checkpoint, and why not 4×?
7. Why can't ZeRO-3 checkpoints be produced by "gather on rank 0 and torch.save"?
   What two things does a DCP-style metadata file record that make resharding possible?
8. Why does `training_state_dict` clone tensors to CPU instead of saving
   `model.state_dict()` directly? Name the two distinct failure modes cloning prevents.
9. The bitwise resume test passes on CPU but not on a GPU box. List three legitimate
   (non-bug) sources of the difference and how you'd confirm each.
10. You checkpoint to Google Drive. Walk through what happens if the VM is preempted
    (a) during the local save, (b) during the Drive copy, (c) between the two — and
    what your resume code must check in each case.

## Summary

- A checkpoint = **seven pieces** (model, optimizer, scheduler, scaler, RNG, data
  position, counters); each omission has a recognizable loss-curve signature.
- The correctness oracle is the **bitwise resume test** — `==`, never `allclose`.
- Optimizer state is 2/3 of the bytes: **12 bytes/param** for Adam FP32 — verify
  against the ledger.
- Saves must be **atomic**: same-dir tmp → fsync → `os.replace`; never resume a `.tmp`.
- **Save unwrapped, from CPU clones; load with `weights_only=True, map_location="cpu"`.**
- Checkpoint every **τ\* = √(2δM)** seconds (Young); for Colab-like numbers ≈ 15 min at
  ~6.5% overhead; known walls get an explicit save, only surprises get modeled.
- Rotation bounds disk (`keep_last`), protection exempts best-val; off-machine sync or
  the checkpoint dies with the VM.
- DDP: rank 0 saves + barrier; ZeRO/FSDP: no rank has the whole state ⇒ **sharded
  checkpoints with metadata** (DCP), which also buys resharding and parallel writes.
- `torch.save` = zip of pickle + raw storages; pickle is why untrusted checkpoints are
  programs.
- Async saving = synchronous CPU snapshot + background write — the snapshot is the
  correctness-critical half.

## References

- Young, J. W. — *A first order approximation to the optimum checkpoint interval*
  (CACM 1974). Three pages; the τ\* = √(2δM) derivation you just did, in its original form.
- Daly, J. T. — *A higher order estimate of the optimum checkpoint interval for
  restart dumps* (FGCS 2006). What changes when δ and τ are not ≪ M; the standard
  reference in HPC fault tolerance.
- PyTorch docs — *torch.distributed.checkpoint* and *Serialization semantics*. The
  DCP API surface (save/load/async_save, planners, DTensor) and the zip-format/
  `weights_only`/mmap details this chapter summarizes.
- Rajbhandari et al. — *ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models* (SC 2020). Why optimizer state dominates and how sharding it forces the
  sharded-checkpoint problem.
- Eisenman et al. — *Check-N-Run: Checkpointing for Training Deep Recommendation
  Models* (NSDI 2022). Production checkpointing under tight time budgets: incremental
  + quantized checkpoints; good for calibrating "what does industry actually do".
- Wang et al. — *Gemini: Fast Failure Recovery in Distributed Training with In-Memory
  Checkpoints* (SOSP 2023). The hierarchical/peer-RAM level from Limitations, fully
  developed.
- Ansel et al. — *PyTorch 2* (ASPLOS 2024), §serialization discussion — background on
  `_orig_mod` and compile-time state, for mistake #3's second half.

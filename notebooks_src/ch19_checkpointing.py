# %% [markdown]
# # Chapter 19 — Checkpointing: Exact Resume, Atomic Saves, Sharded State
#
# **Hardware:** runs anywhere, CPU included — checkpointing is about *state and
# files*, not FLOPs. A GPU only changes the numbers (PCIe copy time shows up in
# the snapshot cost); every demonstration below is exact on CPU, which is
# precisely why we can demand **bitwise** results.
#
# Goal: treat a training run as a state machine with SEVEN pieces of state
# (model, optimizer, scheduler, scaler, RNG, data position, counters), learn to
# serialize ALL of them atomically, prove the resume is bitwise-exact, break it
# on purpose to learn each failure signature, derive how OFTEN to save
# (Young–Daly), and reshard a distributed checkpoint from 2 workers to 1.

# %%
import sys
sys.path.insert(0, "../src")

import os
import tempfile

import matplotlib.pyplot as plt
import torch

from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())
WORK = tempfile.mkdtemp(prefix="ch19_")
print(f"\nscratch dir for this notebook: {WORK}")

# %% [markdown]
# ## 1. What is actually in a `state_dict`?
#
# An ordered mapping *qualified name → tensor*, containing **parameters AND
# buffers**. Buffers (BatchNorm statistics, causal masks, RoPE tables) are state
# that gradient descent never touches — `model.parameters()` misses them, which
# is why checkpoints serialize state dicts, never parameter lists.
#
# The optimizer's state dict is shaped differently: parameters are identified by
# **integer position**, not name — remember that when you see it print below.
# It also does not exist yet at construction: Adam allocates `exp_avg`/
# `exp_avg_sq` lazily, inside the first `step()` (the "step-1 memory jump" of
# chapter 04).

# %%
from llmdist.models.mini_gpt import GPTConfig, MiniGPT

torch.manual_seed(0)
cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2,
                d_model=64, dropout=0.1)   # dropout ON — RNG will matter later
model = MiniGPT(cfg)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

sd = model.state_dict()
print(f"{len(sd)} entries; first/last few:")
for k in [*list(sd)[:3], "...", *list(sd)[-2:]]:
    print(f"  {k:40s} {tuple(sd[k].shape) if k != '...' else ''}")

n_params = sum(1 for _ in model.named_parameters())
n_buffers = sum(1 for _ in model.named_buffers())
print(f"\nparameters: {n_params}   buffers: {n_buffers}   "
      f"(state_dict carries both)")
print(f"optimizer state BEFORE first step: {len(opt.state_dict()['state'])} entries")


def train_step() -> float:
    """One LM step on synthetic data drawn from the GLOBAL torch RNG —
    deliberate: it makes the data sequence part of the RNG state, so
    restoring RNG also restores the data position in this toy setting."""
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size + 1))
    _, loss = model(idx[:, :-1].contiguous(), targets=idx[:, 1:].contiguous())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


train_step()
inner = opt.state_dict()["state"]
print(f"optimizer state AFTER one step : {len(inner)} entries, e.g. "
      f"{ {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in inner[0].items()} }")

# %% [markdown]
# ## 2. The trap: `state_dict()` values are *references*
#
# `sd = model.state_dict()` hands you views of the **live** training tensors.
# Serialize that dict while training mutates it (async writers!) and the file is
# a *torn* mix of step-t and step-t+1. Watch the "saved" dict drift:

# %%
sd_ref = model.state_dict()                       # references
sd_snap = {k: v.detach().cpu().clone() for k, v in sd_ref.items()}  # snapshot

w = "tok_emb.weight"
before = sd_ref[w][0, :4].clone()
for _ in range(3):
    train_step()

print(f"reference dict drifted : {not torch.equal(sd_ref[w][0, :4], before)}")
print(f"snapshot stayed put    : {torch.equal(sd_snap[w][0, :4], before)}")
print("\n=> training_state_dict() clones every tensor to CPU. Always snapshot,")
print("   then serialize the snapshot.")

# %% [markdown]
# ## 3. The seven pieces, assembled — and the size ledger
#
# `llmdist.checkpoint.training_state_dict` collects: **model, optimizer,
# scheduler, scaler, RNG streams, counters** (+ your extras, e.g. data
# position). Ledger prediction for Adam+FP32: weights 4N + exp_avg 4N +
# exp_avg_sq 4N = **12 bytes/param**. Let's check it against the actual file.

# %%
from llmdist.checkpoint import (RotatingCheckpointer, latest_checkpoint,
                                load_checkpoint, save_checkpoint,
                                training_state_dict, young_interval)

sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 10, 1.0))
state = training_state_dict(model, opt, sched, step=4,
                            extra={"tokens_seen": 4 * 4 * 32})
print("checkpoint schema:", list(state))

path = os.path.join(WORK, "demo", "step_00000004.pt")
nbytes = save_checkpoint(path, state)

N = model.num_params()
print(f"\nparams N          : {N:,}")
print(f"12 B/param rule   : {12 * N / 2**20:6.2f} MB")
print(f"actual file       : {nbytes / 2**20:6.2f} MB")
print("\ngap = weight tying (head.weight IS tok_emb.weight: counted once in N,")
print("but present twice in state_dict once cloned) + buffers + RNG smallprint.")
print("experiments/ch19/checkpoint_cost.py tables this across model sizes.")

# %% [markdown]
# ## 4. Atomicity: a save that survives its own death
#
# `save_checkpoint` = write to same-dir `.tmp` → `fsync` → `os.replace`.
# Each ingredient kills one failure mode (same-dir: rename atomic only within a
# filesystem; fsync: data on disk before the rename is; replace: readers see old
# XOR new, never partial). Simulate the crash that bare `torch.save` does not
# survive:

# %%
# a "crash" left a truncated tmp behind — exactly what preemption produces
with open(os.path.join(WORK, "demo", "step_00000005.pt.tmp"), "wb") as f:
    f.write(b"PK\x03\x04 half a checkpoint")

found = latest_checkpoint(os.path.join(WORK, "demo"))
print(f"latest_checkpoint() -> {os.path.basename(found)}   (.tmp ignored ✓)")

# and if garbage ever lands at a FINAL path, loading fails loudly, not quietly:
bad = os.path.join(WORK, "demo", "step_00000006.pt")
with open(bad, "wb") as f:
    f.write(b"PK\x03\x04 not a zip either")
try:
    load_checkpoint(bad)
except Exception as e:
    print(f"truncated final file: {type(e).__name__} raised (good!)")
os.remove(bad)

# %% [markdown]
# ## 5. The centerpiece: bitwise-exact resume
#
# Run A: 40 steps straight. Run B: 20 steps → checkpoint → **everything
# destroyed and rebuilt with a WRONG seed** → resume → 20 more. If the
# checkpoint really captured the whole machine, B's losses 21–40 equal A's
# **bitwise** — `==`, not `allclose`. On CPU with fixed seeds there is no
# excuse for a single differing bit; `allclose` would hide exactly the small
# divergences (a missed RNG stream, drifting moments) we exist to eliminate.

# %%
def build(seed: int):
    global model, opt, sched, cfg
    torch.manual_seed(seed)
    model = MiniGPT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 10, 1.0))


def run(n: int) -> list[float]:
    out = []
    for _ in range(n):
        loss = train_step()
        sched.step()
        out.append(loss)
    return out


build(seed=123)
losses_a = run(40)                                   # run A: uninterrupted

build(seed=123)
losses_b = run(20)                                   # run B: first half
ck = os.path.join(WORK, "resume", "step_00000020.pt")
save_checkpoint(ck, training_state_dict(model, opt, sched, step=20))

build(seed=666)                                      # "new process", wrong seed
loaded = load_checkpoint(ck, model, opt, sched)      # ...must not matter
losses_b += run(40 - loaded["step"])                 # run B: second half

exact = sum(a == b for a, b in zip(losses_a, losses_b))
print(f"bitwise-equal steps: {exact}/40   "
      f"max |Δ| = {max(abs(a - b) for a, b in zip(losses_a, losses_b)):.1e}")
assert exact == 40, "resume leaked state!"
print("The checkpoint IS the training process. ✓")

# %% [markdown]
# ## 6. Break it on purpose: the signature of each missing piece
#
# Delete one piece of state before saving and watch WHERE the curves split.
# You are building a debugging pattern-library: a **bump** that decays = fresh
# optimizer moments; an immediate **point-wise divergence** with the same trend
# = lost RNG (statistically fine, bitwise wrong — invisible in loss quality,
# which is why the equality test exists); an **LR discontinuity kink** = lost
# scheduler.

# %%
def resumed_run(sabotage: str | None) -> list[float]:
    build(seed=123)
    losses = run(20)
    st = training_state_dict(model, opt, sched, step=20)
    if sabotage:
        del st[sabotage]
    p = os.path.join(WORK, f"sab_{sabotage}", "step_00000020.pt")
    save_checkpoint(p, st)
    build(seed=666)
    load_checkpoint(p, model, opt, sched)
    return losses + run(20)


fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(losses_a, "k-", lw=2, label="uninterrupted (truth)")
for sab, style in [(None, "g."), ("optimizer", "r--"), ("rng", "b--")]:
    ls = resumed_run(sab)
    err = sum(a != b for a, b in zip(losses_a, ls))
    ax.plot(ls, style, alpha=0.8,
            label=f"resume minus {sab or 'nothing'} ({err} steps differ)")
ax.axvline(20, color="gray", ls=":", label="resume point")
ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.set_title("What an incomplete checkpoint looks like")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Rotation: bounded disk, unbounded training
#
# Keep the newest k step-numbered files; `protect()` exempts special ones
# (best-val). `maybe_save` takes a **callable** so the expensive snapshot is
# only assembled when a save is actually due.

# %%
rot = RotatingCheckpointer(os.path.join(WORK, "rot"), keep_last=3, every_steps=5)
build(seed=0)
for step in range(1, 21):
    train_step()
    saved = rot.maybe_save(
        lambda: training_state_dict(model, opt, step=step), step)
    if saved:
        print(f"step {step:>2}: saved {os.path.basename(saved)}")
print("on disk:", sorted(os.listdir(rot.directory)), " <= keep_last=3")

# %% [markdown]
# ## 8. How often? Young (1974): τ\* = √(2δM)
#
# Overhead rate H(τ) = δ/τ (checkpoint tax) + τ/2M (expected rework per
# failure). Minimize: τ\* = √(2δM) — full derivation in the chapter doc.
# Colab-free-tier numbers: δ ≈ 30 s (save + Drive sync), M ≈ 4 h:

# %%
delta, M = 30.0, 4 * 3600
tau = young_interval(delta, M)
print(f"τ* = √(2·{delta:.0f}·{M:.0f}) = {tau:.0f} s ≈ {tau / 60:.0f} min")
print(f"overhead at τ* ≈ √(2δ/M) = {(2 * delta / M) ** 0.5:.1%}")

taus = [t * 60 for t in (1, 2, 5, 10, 15, 30, 60, 120, 240)]
H = [delta / t + t / (2 * M) for t in taus]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([t / 60 for t in taus], H, "o-")
ax.axvline(tau / 60, ls="--", c="tab:red", label=f"τ* = {tau / 60:.0f} min")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("checkpoint interval (min)"); ax.set_ylabel("overhead H(τ)")
ax.set_title("δ/τ vs τ/2M: the U-curve (analytic)")
ax.legend(); plt.tight_layout(); plt.show()
print("Monte-Carlo verification with real preemptions: "
      "experiments/ch19/optimal_interval.py --save-plots")

# %% [markdown]
# ## 9. Distributed: DDP coordination, then sharded state
#
# **DDP** (replicas): rank 0 assembles + saves, `dist.barrier()`, everyone
# loads the same file with `map_location="cpu"` (default load would put every
# rank's copy on GPU 0 — mistake #4 in the chapter).
#
# **ZeRO/FSDP** (shards): no rank holds the full state, so rank-0 saving would
# re-create the memory wall sharding removed. `torch.distributed.checkpoint`
# (DCP) instead lets every rank write the shards it owns + one metadata file
# mapping *logical tensor → (shard, byte-range)*. Loading is **in-place** and
# layout-agnostic: declare the tensors you need, DCP routes bytes — so saving
# on 2 workers and resuming on 1 (or 8 → 512) is a non-event. We run exactly
# that, on CPU/Gloo, right here:

# %%
# The workers live in a script (spawn targets must be importable, not defined
# in a notebook cell). Save with world=2, then load with world=1:
sys.path.insert(0, "../experiments/ch19")
from sharded_dcp import load_worker, save_worker  # noqa: E402

from llmdist.utils.dist import run_distributed  # noqa: E402

dcp_dir = os.path.join(WORK, "dcp_ckpt")
ref_path = os.path.join(WORK, "dcp_reference.pt")
try:
    run_distributed(save_worker, world_size=2, backend="gloo", port=29612,
                    args=(dcp_dir, ref_path))
    run_distributed(load_worker, world_size=1, backend="gloo", port=29613,
                    args=(dcp_dir, ref_path))
except Exception as e:  # some notebook hosts forbid spawn
    print(f"spawn unavailable here ({type(e).__name__}); run instead:\n"
          "  python experiments/ch19/sharded_dcp.py")

# %% [markdown]
# ## 10. The preemption-proof loop (Colab/Kaggle pattern)
#
# Everything assembled — this is the loop to copy into real projects:
#
# ```python
# rot = RotatingCheckpointer(DRIVE_DIR, keep_last=3,
#                            every_seconds=young_interval(delta_measured, 4*3600))
# ckpt = latest_checkpoint(DRIVE_DIR)                  # None on first launch
# start = 0
# if ckpt:
#     start = load_checkpoint(ckpt, model, opt, sched)["step"]   # RNG restored too
#
# for step in range(start, TOTAL):
#     loss = train_step()
#     sched.step()
#     rot.maybe_save(lambda: training_state_dict(model, opt, sched,
#                                                step=step + 1), step + 1)
# ```
#
# Notes that earn their keep: (1) checkpoint dir on Drive/HF Hub — VM disk dies
# with the VM; but write locally + copy if you can: `os.replace` is not atomic
# across filesystems. (2) Measure δ *including* the sync before choosing τ.
# (3) Add a forced save just before Colab's known 12 h wall — Young–Daly only
# models the surprises (chapter, Mathematics §2).

# %% [markdown]
# ## Exercises
#
# **E1.** Predict-then-run: set `dropout=0.0` in §6's model. Which sabotage
# curves change, and why? (Answer below.)
#
# **E2.** Extend §7 with `save_best`: track a fake val metric, protect the best
# file, prove rotation still keeps exactly `keep_last` rotating files.
#
# **E3.** Time `training_state_dict` vs `save_checkpoint` separately for
# d_model ∈ {128, 256, 512} (see `checkpoint_cost.py`). Which grows faster, and
# which half can async checkpointing overlap?
#
# **H1.** Implement `async_save` (snapshot sync, write in a `threading.Thread`,
# max one in-flight). Prove correctness with §5's bitwise test; measure the
# stall reduction.
#
# **H2.** Colab's real failure law = random preemption (M≈4h) + hard wall at
# 12 h. Modify `optimal_interval.py` to simulate it and find the best schedule.
# Hint: what should happen at 11:55?

# %% [markdown]
# <details><summary><b>Solutions (spoilers)</b></summary>
#
# **E1.** With dropout off, the *rng* sabotage curve becomes bitwise-identical
# to truth **in this toy** — until you notice data is drawn from the same RNG
# stream, so it still diverges here. Decouple data into its own generator
# (`torch.Generator`) and then rng-sabotage truly stops mattering: no dropout,
# no RNG-dependent compute. The lesson: the pieces you must checkpoint = the
# pieces your step actually consumes.
#
# **E2.** Save best to `best_step_X.pt` (name outside `CKPT_PATTERN` ⇒ outside
# rotation) *and* `protect()` it; assert `len(rotating files) == keep_last`.
#
# **E3.** Both scale ~linearly with bytes (12N), but write includes fsync's
# constant floor; async overlaps the WRITE only — the snapshot must stay
# synchronous or you tear state (§2).
#
# **H1.** Invariants: clone-to-CPU completes before returning; one writer
# (lock); step-numbered files keep atomicity safe even if two writes race.
#
# **H2.** Intervals shrink as the wall approaches + one forced save just before
# it; vs plain Young–Daly, ~an extra τ*/2 of work retained per 12 h session.
# </details>

# %% [markdown]
# ## Summary
#
# - A checkpoint = **7 pieces**; forget one and the resume silently diverges —
#   each omission has a loss-curve signature you can now read.
# - The correctness oracle is **bitwise equality** (`==`), never `allclose`.
# - `state_dict()` returns **references** — snapshot (clone→CPU) before writing.
# - **12 bytes/param** (Adam FP32); optimizer state is 2/3 of your checkpoint.
# - Atomic save = same-dir tmp → **fsync** → `os.replace`; never resume a `.tmp`.
# - Load with `weights_only=True, map_location="cpu"` — pickle is a program,
#   and storages remember their device.
# - Save every **τ\* = √(2δM)** (≈15 min on free Colab, ~6.5% overhead); handle
#   known walls explicitly.
# - Rotation bounds disk; sync off-machine or the checkpoint dies with the VM.
# - DDP: rank-0 + barrier. Sharded (ZeRO/FSDP): DCP — everyone writes their
#   shards, metadata makes **resharding** a non-event.

# %%
import shutil
shutil.rmtree(WORK, ignore_errors=True)
print("scratch cleaned. ch19 done — ch20 (mixed precision) changes what the")
print("bytes IN the checkpoint mean; ch15–16 make its sharded path mandatory.")

"""Checkpointing: capture, persist, and restore *complete* training state.

Why this exists: a training run is a deterministic state machine. If you can
serialize its full state, you can stop it anywhere and continue as if nothing
happened — which is the difference between "Colab disconnected, I lost a day"
and "Colab disconnected, I lost four minutes". The subtlety is the word
*complete*: the model weights are only one of SEVEN pieces of state
(model, optimizer, LR scheduler, AMP scaler, RNG streams, data position,
step counter). Forget any one and the resumed run silently diverges from the
uninterrupted run. Everything in this module exists to make "resume" mean
*bitwise* resume, and to make a save survive being killed halfway through.

Design decisions, each explained where it is implemented:
  - saves are ATOMIC (tmp file + os.replace) so a crash never corrupts
    the previous good checkpoint;
  - RNG state is stored in a torch.load(weights_only=True)-friendly form
    (tensors and plain ints only, no pickled numpy internals);
  - the DDP/compile wrappers are unwrapped before saving so checkpoints are
    loadable regardless of how the model will be wrapped on resume;
  - rotation keeps the newest k checkpoints so free-tier disk quotas survive.
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

try:  # numpy is optional at import time; RNG capture degrades gracefully
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

CKPT_PATTERN = re.compile(r"step_(\d+)\.pt$")


# --------------------------------------------------------------------------
# RNG state — the piece everyone forgets
# --------------------------------------------------------------------------

def rng_state_dict() -> dict[str, Any]:
    """Capture every RNG stream that can influence training.

    Four independent generators can affect a training step: Python's `random`
    (shuffling in some datasets), numpy (augmentations), torch CPU (dropout on
    CPU, weight init, DataLoader workers), and torch CUDA (dropout on GPU).
    Restoring only some of them is the classic source of "resumed loss curve
    is close but not identical".

    Storage format note: numpy's `get_state()` returns an ndarray which
    `torch.load(weights_only=True)` refuses to unpickle. We convert it to a
    torch tensor here (and back in `load_rng_state`) so checkpoints stay
    loadable under the safe loader.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),                # nested tuples of ints — safe
        "torch_cpu": torch.get_rng_state(),         # a uint8 tensor — safe
    }
    if np is not None:
        name, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = {
            "name": name,
            # MT19937 keys are uint32, which torch tensors can't hold —
            # widen to int64 here, narrow back in load_rng_state.
            "keys": torch.from_numpy(keys.astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()  # list of tensors
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    """Restore every RNG stream captured by `rng_state_dict`.

    Tolerates environment changes (e.g. checkpoint saved on GPU, resumed on
    CPU): streams that do not exist here are skipped rather than crashing,
    because a *slightly* nondeterministic resume beats no resume.
    """
    random.setstate(_to_tuples(state["python"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if np is not None and "numpy" in state:
        s = state["numpy"]
        np.random.set_state((s["name"], s["keys"].numpy().astype(np.uint32),
                             s["pos"], s["has_gauss"], s["cached_gaussian"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        saved = state["torch_cuda"][: torch.cuda.device_count()]
        torch.cuda.set_rng_state_all([t.cpu() for t in saved])


def _to_tuples(x: Any) -> Any:
    """Recursively lists -> tuples (JSON-ish loaders turn tuples into lists)."""
    if isinstance(x, (list, tuple)):
        return tuple(_to_tuples(i) for i in x)
    return x


# --------------------------------------------------------------------------
# Unwrapping — checkpoints must outlive wrappers
# --------------------------------------------------------------------------

def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / torch.compile wrappers to reach the plain module.

    Why: DDP prefixes every key with "module." and torch.compile with
    "_orig_mod.". A checkpoint saved through the wrapper only loads into an
    identically-wrapped model. Saving the unwrapped module makes the
    checkpoint wrapper-agnostic — load it bare, DDP-wrapped, or compiled.
    """
    if hasattr(model, "_orig_mod"):            # torch.compile
        model = model._orig_mod
    if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        model = model.module                   # DDP / DataParallel
    return model


# --------------------------------------------------------------------------
# The complete training state
# --------------------------------------------------------------------------

def training_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the seven pieces of training state into one dict.

    `state_dict()` returns *references* to the live tensors, not copies — we
    clone them onto CPU here. Two reasons: (a) without a copy, training that
    continues while an async writer serializes would tear the tensors;
    (b) CPU tensors make the checkpoint loadable on any machine without
    `map_location` gymnastics.
    """
    m = unwrap(model)
    state: dict[str, Any] = {
        "model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
        "step": int(step),
        "rng": rng_state_dict(),
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        state["optimizer"] = _optim_state_to_cpu(optimizer.state_dict())
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        state["scaler"] = scaler.state_dict()
    if extra:
        state["extra"] = extra
    return state


def _optim_state_to_cpu(sd: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy an optimizer state_dict with all tensors moved to CPU."""
    def move(x: Any) -> Any:
        if torch.is_tensor(x):
            return x.detach().cpu().clone()
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, list):
            return [move(v) for v in x]
        return x
    return move(sd)


# --------------------------------------------------------------------------
# Atomic save / safe load
# --------------------------------------------------------------------------

def save_checkpoint(path: str, state: dict[str, Any]) -> int:
    """Write `state` to `path` atomically. Returns bytes written.

    The atomicity recipe (write-ahead pattern used by every database):
      1. serialize to `path + ".tmp"` in the SAME directory (os.replace is
         only atomic within one filesystem);
      2. flush + fsync so the bytes are on disk, not in the page cache —
         a power loss after replace() but before writeback would otherwise
         leave a complete-looking but empty file;
      3. os.replace(tmp, path) — atomic on POSIX and Windows: readers see
         either the old complete file or the new complete file, never a
         partial one.
    If we crash anywhere before step 3, `path` still holds the previous good
    checkpoint and only a stale .tmp is left behind.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return os.path.getsize(path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore every component that was passed in.

    Returns the raw state dict (so callers can read `state["step"]`, extras).

    `weights_only=True`: torch.save uses pickle, and pickle executes
    arbitrary code on load — a checkpoint downloaded from the internet is a
    program. The safe loader only reconstructs tensors and containers of
    primitives; this module deliberately stores nothing that needs more.

    `map_location="cpu"`: saved CUDA tensors otherwise deserialize onto the
    GPU index they were saved from — on a multi-GPU node every rank would
    load onto GPU 0 (rank0's device) and OOM it. Load to CPU, then let
    `load_state_dict` copy into the (already correctly placed) live tensors.
    """
    # Version gate, not a try/except: falling back to the unsafe loader
    # *on error* would let a malicious file trigger its own unsafe load.
    # torch < 2.0's restricted unpickler lacks basic opcodes (even floats);
    # this repo requires torch >= 2.2, so the gate is False only if you are
    # deliberately running an ancient torch.
    safe = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 0)
    state = torch.load(path, map_location=map_location, weights_only=safe)
    if model is not None:
        unwrap(model).load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng and "rng" in state:
        load_rng_state(state["rng"])
    return state


# --------------------------------------------------------------------------
# Rotation — bounded disk, unbounded training
# --------------------------------------------------------------------------

@dataclass
class RotatingCheckpointer:
    """Periodic checkpointing with a bounded disk footprint.

    Saves at most every `every_steps` steps and/or `every_seconds` seconds
    (whichever fires), keeps the newest `keep_last` files, never deletes
    `protected` ones (e.g. best-validation). Layout: `dir/step_00001234.pt`.

    Why time-based AND step-based: on preemptible hardware what you lose is
    measured in wall-clock minutes (Young–Daly, see the chapter), but what
    you reason about scientifically is steps. Supporting both lets you say
    "every 10 min, but also right after each eval".
    """
    directory: str
    keep_last: int = 3
    every_steps: int | None = None
    every_seconds: float | None = None
    protected: set[str] = field(default_factory=set)
    _last_save_time: float = field(default_factory=time.monotonic, repr=False)
    _last_save_step: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    # -- policy ------------------------------------------------------------
    def should_save(self, step: int) -> bool:
        due_steps = (self.every_steps is not None
                     and step - max(self._last_save_step, 0) >= self.every_steps)
        due_time = (self.every_seconds is not None
                    and time.monotonic() - self._last_save_time >= self.every_seconds)
        return due_steps or due_time

    # -- actions -----------------------------------------------------------
    def save(self, state: dict[str, Any], step: int) -> str:
        path = os.path.join(self.directory, f"step_{step:08d}.pt")
        save_checkpoint(path, state)
        self._last_save_step = step
        self._last_save_time = time.monotonic()
        self._prune()
        return path

    def maybe_save(self, state_fn: Any, step: int) -> str | None:
        """`state_fn` is a zero-arg callable returning the state dict.

        Why a callable and not the dict: assembling the state clones every
        tensor to CPU, which costs time and RAM — we only want to pay that
        when a save is actually due.
        """
        if not self.should_save(step):
            return None
        return self.save(state_fn(), step)

    def latest(self) -> str | None:
        found = self._list()
        return found[-1][1] if found else None

    def protect(self, path: str) -> None:
        """Exempt a checkpoint (e.g. best-val) from rotation."""
        self.protected.add(os.path.basename(path))

    # -- internals -----------------------------------------------------------
    def _list(self) -> list[tuple[int, str]]:
        out = []
        for name in os.listdir(self.directory):
            m = CKPT_PATTERN.search(name)
            if m:
                out.append((int(m.group(1)), os.path.join(self.directory, name)))
        return sorted(out)

    def _prune(self) -> None:
        candidates = [(s, p) for s, p in self._list()
                      if os.path.basename(p) not in self.protected]
        for _, path in candidates[: max(0, len(candidates) - self.keep_last)]:
            os.remove(path)


def latest_checkpoint(directory: str) -> str | None:
    """Find the newest `step_*.pt` in a directory, or None. Ignores .tmp files
    (a leftover .tmp is exactly the crash artifact atomic saving protects
    against — it must never be resumed from)."""
    if not os.path.isdir(directory):
        return None
    best: tuple[int, str] | None = None
    for name in os.listdir(directory):
        m = CKPT_PATTERN.search(name)
        if m and not name.endswith(".tmp"):
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, os.path.join(directory, name))
    return best[1] if best else None


# --------------------------------------------------------------------------
# Young–Daly optimal checkpoint interval
# --------------------------------------------------------------------------

def young_interval(checkpoint_seconds: float, mtbf_seconds: float) -> float:
    """Young's (1974) optimal checkpoint interval:  τ* = sqrt(2 · δ · M).

    δ = time one checkpoint costs, M = mean time between failures.
    Derivation in the chapter's Mathematics section: overhead per unit of
    useful work is H(τ) = δ/τ (checkpoint tax) + τ/(2M) (expected rework
    after a failure); dH/dτ = 0 gives τ*. Valid when δ << τ << M.
    """
    if checkpoint_seconds <= 0 or mtbf_seconds <= 0:
        raise ValueError("checkpoint cost and MTBF must be positive")
    return float((2.0 * checkpoint_seconds * mtbf_seconds) ** 0.5)

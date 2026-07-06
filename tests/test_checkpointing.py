"""Tests for src/llmdist/checkpoint.py — each test guards one design decision."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest
import torch
import torch.nn as nn

from llmdist.checkpoint import (
    RotatingCheckpointer, latest_checkpoint, load_checkpoint, load_rng_state,
    rng_state_dict, save_checkpoint, training_state_dict, unwrap,
    young_interval)


def tiny_setup(seed: int = 0) -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(16, 4))
    return model, torch.optim.Adam(model.parameters(), lr=1e-2)


def one_step(model: nn.Module, opt: torch.optim.Optimizer) -> float:
    x = torch.randn(4, 8)
    loss = model(x).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


# ---------------------------------------------------------------- round trip

def test_bitwise_resume(tmp_path):
    """The chapter's central claim: full state ⇒ bitwise-identical continuation."""
    model, opt = tiny_setup(seed=1)
    ref_losses = [one_step(model, opt) for _ in range(20)]

    model, opt = tiny_setup(seed=1)
    losses = [one_step(model, opt) for _ in range(10)]
    p = str(tmp_path / "step_00000010.pt")
    save_checkpoint(p, training_state_dict(model, opt, step=10))

    model, opt = tiny_setup(seed=99)              # wrong seed must not matter
    state = load_checkpoint(p, model, opt)
    assert state["step"] == 10
    losses += [one_step(model, opt) for _ in range(10)]
    assert losses == ref_losses                    # == : bitwise, not allclose


def test_dropping_optimizer_breaks_exactness(tmp_path):
    """Negative control: the test above must have teeth."""
    model, opt = tiny_setup(seed=1)
    ref = [one_step(model, opt) for _ in range(20)]

    model, opt = tiny_setup(seed=1)
    losses = [one_step(model, opt) for _ in range(10)]
    state = training_state_dict(model, opt, step=10)
    del state["optimizer"]
    p = str(tmp_path / "step_00000010.pt")
    save_checkpoint(p, state)

    model, opt = tiny_setup(seed=99)
    load_checkpoint(p, model, opt)
    losses += [one_step(model, opt) for _ in range(10)]
    assert losses[:10] == ref[:10] and losses != ref


def test_state_is_snapshot_not_reference(tmp_path):
    """training_state_dict clones: later steps must not mutate captured state."""
    model, opt = tiny_setup()
    one_step(model, opt)
    state = training_state_dict(model, opt)
    before = {k: v.clone() for k, v in state["model"].items()}
    for _ in range(3):
        one_step(model, opt)
    assert all(torch.equal(state["model"][k], before[k]) for k in before)


def test_rng_round_trip():
    saved = rng_state_dict()
    a = (torch.randn(5), torch.randint(0, 100, (5,)))
    load_rng_state(saved)
    b = (torch.randn(5), torch.randint(0, 100, (5,)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.skipif(
    tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 0),
    reason="torch<2 restricted unpickler lacks basic opcodes")
def test_weights_only_loadable(tmp_path):
    """The schema must pass the safe (restricted-unpickler) loader."""
    model, opt = tiny_setup()
    one_step(model, opt)
    p = str(tmp_path / "step_00000001.pt")
    save_checkpoint(p, training_state_dict(model, opt, step=1,
                                           extra={"tokens": 123}))
    state = torch.load(p, map_location="cpu", weights_only=True)  # no error
    assert state["extra"]["tokens"] == 123


# ---------------------------------------------------------------- atomicity

def test_no_tmp_left_and_truncation_survivable(tmp_path):
    model, opt = tiny_setup()
    p = str(tmp_path / "step_00000001.pt")
    save_checkpoint(p, training_state_dict(model, opt, step=1))
    assert os.listdir(tmp_path) == ["step_00000001.pt"]  # tmp cleaned up

    # simulate a crash mid-save of the NEXT checkpoint: a stale .tmp appears
    with open(str(tmp_path / "step_00000002.pt.tmp"), "wb") as f:
        f.write(b"truncated garbage")
    # resume logic must pick the intact file, never the .tmp
    assert latest_checkpoint(str(tmp_path)).endswith("step_00000001.pt")


def test_truncated_final_file_fails_loudly(tmp_path):
    """If corruption DOES reach a final path, loading must raise, not
    half-load: torch's zip reader notices the missing central directory."""
    p = str(tmp_path / "step_00000003.pt")
    with open(p, "wb") as f:
        f.write(b"PK\x03\x04 not really a zip")
    with pytest.raises(Exception):
        load_checkpoint(p)


# ---------------------------------------------------------------- unwrap

def test_unwrap_and_wrapper_free_keys():
    model, _ = tiny_setup()

    class FakeDDP(nn.Module):  # same attribute contract as DDP
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    wrapped = FakeDDP(model)
    assert unwrap(wrapped) is model
    state = training_state_dict(wrapped)
    assert not any(k.startswith("module.") for k in state["model"])


# ---------------------------------------------------------------- rotation

def test_rotation_keeps_last_k_and_protected(tmp_path):
    model, opt = tiny_setup()
    rot = RotatingCheckpointer(str(tmp_path), keep_last=3, every_steps=1)
    for s in range(1, 9):
        p = rot.save(training_state_dict(model, opt, step=s), s)
        if s == 1:
            rot.protect(p)   # protect BEFORE rotation can prune it (step 1 = "best val")
    rot.save(training_state_dict(model, opt, step=9), 9)
    kept = sorted(os.listdir(tmp_path))
    assert "step_00000001.pt" in kept               # protected survived
    assert len(kept) == 4                           # 3 rotating + 1 protected
    assert rot.latest().endswith("step_00000009.pt")


def test_maybe_save_respects_cadence(tmp_path):
    model, opt = tiny_setup()
    rot = RotatingCheckpointer(str(tmp_path), keep_last=10, every_steps=5)
    saved = [rot.maybe_save(lambda: training_state_dict(model, opt, step=s), s)
             for s in range(1, 16)]
    assert [s is not None for s in saved].count(True) == 3   # steps 5, 10, 15


# ---------------------------------------------------------------- young

def test_young_interval():
    assert young_interval(30, 4 * 3600) == pytest.approx((2 * 30 * 14400) ** 0.5)
    with pytest.raises(ValueError):
        young_interval(0, 100)

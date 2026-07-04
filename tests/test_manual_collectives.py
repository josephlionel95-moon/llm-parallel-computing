"""Correctness tests for src/llmdist/comm/manual_collectives.py.

Every manual collective is compared against torch.distributed's built-in on
2 CPU/Gloo processes, so these run anywhere (CI included). Worker functions
are module-level because mp.spawn pickles them by module path.

An AssertionError raised inside a spawned worker propagates to the parent as
torch.multiprocessing.ProcessRaisedException, which fails the pytest test.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmdist.comm import manual_collectives as mc          # noqa: E402
from llmdist.utils.dist import run_distributed             # noqa: E402

WORLD = 2


def _seeded(rank: int, n: int = 1000, seed: int = 0) -> torch.Tensor:
    """A tensor whose value depends on the rank — collectives that 'work'
    only because every rank already agrees would pass with equal inputs."""
    g = torch.Generator().manual_seed(seed * 100 + rank)
    return torch.randn(n, generator=g)


# --------------------------------------------------------------------- workers

def _worker_p2p_collectives(rank: int, world: int) -> None:
    """broadcast / broadcast_tree / reduce / gather / scatter / barrier."""
    # broadcast: everyone must end with rank 0's tensor.
    t = _seeded(rank, seed=1)
    ref = _seeded(0, seed=1)
    mc.broadcast(t, src=0)
    assert torch.equal(t, ref), "broadcast: mismatch with src data"
    mc.assert_all_ranks_equal(t, msg="broadcast:")

    # broadcast_tree, from a non-zero root for good measure.
    t = _seeded(rank, seed=2)
    ref = _seeded(world - 1, seed=2)
    mc.broadcast_tree(t, src=world - 1)
    assert torch.equal(t, ref), "broadcast_tree: mismatch with src data"

    # reduce to dst=0: compare against torch's all_reduce result.
    t = _seeded(rank, seed=3)
    expect = t.clone()
    dist.all_reduce(expect)                     # torch is the referee
    mc.reduce(t, dst=0)
    if rank == 0:
        assert torch.allclose(t, expect, atol=1e-6), "reduce: wrong sum on dst"

    # gather to dst=0.
    t = _seeded(rank, seed=4)
    got = mc.gather(t, dst=0)
    if rank == 0:
        assert got is not None and len(got) == world
        for r in range(world):
            assert torch.equal(got[r], _seeded(r, seed=4)), f"gather: rank {r} chunk wrong"
    else:
        assert got is None

    # scatter from src=0.
    chunks = [_seeded(i, seed=5) for i in range(world)] if rank == 0 else None
    out = torch.empty(1000)
    mc.scatter(out, chunks, src=0)
    assert torch.equal(out, _seeded(rank, seed=5)), "scatter: wrong chunk delivered"

    mc.barrier()                                # just must not hang or crash


def _worker_ring_collectives(rank: int, world: int) -> None:
    """all_gather / reduce_scatter / all_reduce_naive / all_reduce_ring."""
    # all_gather.
    t = _seeded(rank, seed=6, n=500)
    out = torch.empty(world, 500)
    mc.all_gather(out, t)
    for r in range(world):
        assert torch.equal(out[r], _seeded(r, seed=6, n=500)), \
            f"all_gather: slot {r} wrong on rank {rank}"

    # reduce_scatter vs torch reference (sum inp[r] over ranks).
    inp = torch.stack([_seeded(rank, seed=7 + i, n=300) for i in range(world)])
    ref = inp.clone()
    dist.all_reduce(ref)                        # full sum on every rank
    out = torch.empty(300)
    mc.reduce_scatter(out, inp)
    assert torch.allclose(out, ref[rank], atol=1e-5), "reduce_scatter: wrong chunk"

    # all_reduce, naive and ring, incl. a size NOT divisible by world (padding path).
    for n in (1000, 1001):
        t = _seeded(rank, seed=8, n=n)
        orig = t.clone()
        mc.all_reduce_naive(t)
        mc.assert_matches_torch_allreduce(t, orig)
        mc.assert_all_ranks_equal(t, atol=1e-5, msg="all_reduce_naive:")

        t = _seeded(rank, seed=9, n=n)
        orig = t.clone()
        mc.all_reduce_ring(t)
        mc.assert_matches_torch_allreduce(t, orig)
        mc.assert_all_ranks_equal(t, atol=1e-5, msg="all_reduce_ring:")


def _worker_ring_multidim(rank: int, world: int) -> None:
    """all_reduce_ring must handle non-flat tensors (it flattens internally)."""
    t = _seeded(rank, seed=10, n=6 * 7 * 5).reshape(6, 7, 5)
    orig = t.clone()
    mc.all_reduce_ring(t)
    mc.assert_matches_torch_allreduce(t, orig)
    assert t.shape == (6, 7, 5)


# ----------------------------------------------------------------------- tests

def test_p2p_collectives():
    run_distributed(_worker_p2p_collectives, world_size=WORLD, port=29611)


def test_ring_collectives():
    run_distributed(_worker_ring_collectives, world_size=WORLD, port=29612)


def test_ring_multidim():
    run_distributed(_worker_ring_multidim, world_size=WORLD, port=29613)

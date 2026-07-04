"""Correctness tests for manual DP (ch08) and ManualDDP (ch09).

The gold standard: after one identical training step, ManualDDP's gradients
must match torch.nn.parallel.DistributedDataParallel's within float
tolerance, and manual DP must match both. 2 CPU/Gloo processes.
"""
from __future__ import annotations

import copy
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmdist.models.mini_gpt import GPTConfig, MiniGPT      # noqa: E402
from llmdist.parallel.manual_ddp import ManualDDP           # noqa: E402
from llmdist.parallel.manual_dp import (                    # noqa: E402
    average_gradients, broadcast_parameters, shard_indices)
from llmdist.utils.dist import run_distributed              # noqa: E402

WORLD = 2
CFG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)


def _batch(rank: int, seed: int = 0, bsz: int = 4):
    g = torch.Generator().manual_seed(1000 * seed + rank)
    x = torch.randint(0, CFG.vocab_size, (bsz, CFG.block_size), generator=g)
    y = torch.randint(0, CFG.vocab_size, (bsz, CFG.block_size), generator=g)
    return x, y


def _grads(model: nn.Module) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


# --------------------------------------------------------------------- workers

def _worker_broadcast_params(rank: int, world: int) -> None:
    """Ranks start with DIFFERENT inits; broadcast must equalize them."""
    torch.manual_seed(rank)                    # deliberately divergent
    model = MiniGPT(CFG)
    broadcast_parameters(model)
    for p in model.parameters():
        gathered = [torch.empty_like(p) for _ in range(world)]
        dist.all_gather(gathered, p.detach())
        assert torch.equal(gathered[0], gathered[1]), "params differ after broadcast"


def _worker_manual_ddp_vs_torch_ddp(rank: int, world: int) -> None:
    """Same init, same per-rank batches => identical grads and identical
    params after 3 optimizer steps."""
    torch.manual_seed(0)                       # same init on all ranks
    base = MiniGPT(CFG)
    m_manual = ManualDDP(copy.deepcopy(base), bucket_size_mb=0.05)  # tiny -> many buckets
    m_torch = nn.parallel.DistributedDataParallel(copy.deepcopy(base))
    opt_m = torch.optim.SGD(m_manual.module.parameters(), lr=0.1)
    opt_t = torch.optim.SGD(m_torch.module.parameters(), lr=0.1)

    for step in range(3):
        x, y = _batch(rank, seed=step)
        for model, opt in ((m_manual, opt_m), (m_torch, opt_t)):
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            if isinstance(model, ManualDDP):
                model.finalize()
            opt.step()
        for gm, gt in zip(_grads(m_manual.module), _grads(m_torch.module)):
            assert torch.allclose(gm, gt, atol=1e-6, rtol=1e-5), \
                f"step {step}: manual DDP grad != torch DDP grad " \
                f"(max diff {(gm - gt).abs().max():.2e})"

    for pm, pt in zip(m_manual.module.parameters(), m_torch.module.parameters()):
        assert torch.allclose(pm, pt, atol=1e-6, rtol=1e-5), "params diverged"


def _worker_manual_dp_vs_manual_ddp(rank: int, world: int) -> None:
    """Naive DP (post-hoc flat all-reduce) and ManualDDP (bucketed, async)
    must produce the same averaged gradients."""
    torch.manual_seed(0)
    base = MiniGPT(CFG)
    m_dp = copy.deepcopy(base)
    m_ddp = ManualDDP(copy.deepcopy(base), bucket_size_mb=0.02)

    x, y = _batch(rank, seed=7)
    _, loss = m_dp(x, y)
    loss.backward()
    average_gradients(m_dp)

    _, loss = m_ddp(x, y)
    loss.backward()
    m_ddp.finalize()

    for g1, g2 in zip(_grads(m_dp), _grads(m_ddp.module)):
        assert torch.allclose(g1, g2, atol=1e-6, rtol=1e-5), "DP grad != DDP grad"


def _worker_no_sync(rank: int, world: int) -> None:
    """2 micro-batches with no_sync + 1 synced == averaging locally
    accumulated grads: hooks must not fire inside no_sync, and the final
    sync must carry the accumulated total."""
    torch.manual_seed(0)
    base = MiniGPT(CFG)
    m_ddp = ManualDDP(copy.deepcopy(base), bucket_size_mb=0.02)
    m_ref = copy.deepcopy(base)

    batches = [_batch(rank, seed=s) for s in (11, 12, 13)]

    with m_ddp.no_sync():
        for x, y in batches[:-1]:
            m_ddp(x, y)[1].backward()          # accumulate locally
    m_ddp(*batches[-1])[1].backward()          # last micro-batch: hooks fire
    m_ddp.finalize()

    for x, y in batches:
        m_ref(x, y)[1].backward()              # plain local accumulation
    average_gradients(m_ref)                   # one averaged all-reduce

    for g1, g2 in zip(_grads(m_ddp.module), _grads(m_ref)):
        assert torch.allclose(g1, g2, atol=1e-6, rtol=1e-5), "no_sync accumulation wrong"


def _worker_grad_vs_single_process(rank: int, world: int) -> None:
    """DP over shards == single process on the concatenated batch (ch08 math).
    Loose tolerance: reduction order differs (documented caveat)."""
    torch.manual_seed(0)
    model = MiniGPT(CFG)
    single = copy.deepcopy(model)

    x, y = _batch(rank, seed=21)
    model(x, y)[1].backward()
    average_gradients(model)

    xs, ys = zip(*[_batch(r, seed=21) for r in range(world)])
    single(torch.cat(xs), torch.cat(ys))[1].backward()

    for gd, gs in zip(_grads(model), _grads(single)):
        assert torch.allclose(gd, gs, atol=1e-5, rtol=1e-4), \
            f"DP grad != large-batch grad (max diff {(gd - gs).abs().max():.2e})"


# ----------------------------------------------------------------------- tests

def test_shard_indices_partition():
    """Pure function — no processes needed. Shards must partition the epoch."""
    n, world = 103, 4
    shards = [shard_indices(n, r, world, epoch=3) for r in range(world)]
    sizes = {len(s) for s in shards}
    assert len(sizes) == 1, "unequal shard sizes would desynchronize step counts"
    seen = [i for s in shards for i in s]
    assert set(seen) == set(range(n)), "some sample never trains"
    assert len(seen) == (n + world - 1) // world * world  # only pad duplicates
    assert shard_indices(n, 0, world, epoch=0) != shard_indices(n, 0, world, epoch=1), \
        "set_epoch must reshuffle"


def test_broadcast_parameters():
    run_distributed(_worker_broadcast_params, world_size=WORLD, port=29621)


def test_manual_ddp_matches_torch_ddp():
    run_distributed(_worker_manual_ddp_vs_torch_ddp, world_size=WORLD, port=29622)


def test_manual_dp_matches_manual_ddp():
    run_distributed(_worker_manual_dp_vs_manual_ddp, world_size=WORLD, port=29623)


def test_no_sync_gradient_accumulation():
    run_distributed(_worker_no_sync, world_size=WORLD, port=29624)


def test_dp_matches_single_process_large_batch():
    run_distributed(_worker_grad_vs_single_process, world_size=WORLD, port=29625)

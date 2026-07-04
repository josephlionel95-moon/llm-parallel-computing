"""Spawn workers + synthetic data for the ch08 notebook (data parallelism)."""
from __future__ import annotations

import time

import torch
import torch.distributed as dist

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.parallel.manual_dp import (average_gradients, broadcast_parameters,
                                        shard_indices)

# Small enough to train on CPU in a notebook cell, big enough that the
# gradient all-reduce is measurable.
NB_CFG = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=4, d_model=128)


def synthetic_tokens(n_tokens: int = 65_536, vocab: int = 256,
                     seed: int = 1234) -> torch.Tensor:
    """A learnable synthetic corpus — NO downloads, fully deterministic.

    Tokens are drawn from a random-but-fixed bigram model: each token id has
    a preferred small set of successors. A char-LM can push loss well below
    ln(vocab) by learning the transition table, so training curves are
    meaningful, unlike uniform-random tokens where the best possible loss is
    exactly ln(vocab) and every curve is flat.
    """
    g = torch.Generator().manual_seed(seed)
    # Each row: logits concentrated on ~8 successors -> low-entropy bigrams.
    logits = torch.full((vocab, vocab), -8.0)
    for i in range(vocab):
        succ = torch.randint(0, vocab, (8,), generator=g)
        logits[i, succ] = torch.rand(8, generator=g) * 4.0
    probs = torch.softmax(logits, dim=-1)
    out = torch.empty(n_tokens, dtype=torch.long)
    out[0] = torch.randint(0, vocab, (1,), generator=g)
    for t in range(1, n_tokens):
        out[t] = torch.multinomial(probs[out[t - 1]], 1, generator=g)
    return out


def make_batch(stream: torch.Tensor, idx: list[int], block: int,
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice windows [i, i+block] out of the token stream; y is x shifted."""
    xs = torch.stack([stream[i:i + block] for i in idx])
    ys = torch.stack([stream[i + 1:i + 1 + block] for i in idx])
    return xs, ys


def w_dp_train(rank: int, world: int, steps: int, bsz: int,
               out_path: str, flat: bool = True) -> None:
    """Manual-DP training loop with a per-phase timing breakdown.

    Every rank: forward/backward on ITS shard -> average_gradients -> step.
    Rank 0 saves {losses, phase_ms, tokens_per_s} for the notebook to plot.
    The comm fraction it reports is the punchline: naive DP pays it fully,
    serialized after backward.
    """
    torch.manual_seed(0)                          # same init everywhere (belt)
    model = MiniGPT(NB_CFG)
    broadcast_parameters(model)                   # ...and suspenders
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    stream = synthetic_tokens()                   # identical on all ranks
    n_windows = len(stream) - NB_CFG.block_size - 1
    phase_ms = {"forward": 0.0, "backward": 0.0, "comm": 0.0, "optimizer": 0.0}
    losses: list[float] = []
    t_start = time.perf_counter()

    for step in range(steps):
        # Hand-rolled DistributedSampler: same permutation, disjoint slices.
        shard = shard_indices(n_windows, rank, world, epoch=step)
        idx = shard[:bsz]                         # one batch per "epoch" here
        x, y = make_batch(stream, idx, NB_CFG.block_size)

        t0 = time.perf_counter()
        _, loss = model(x, y)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        average_gradients(model, flat=flat)       # THE data-parallel line
        t3 = time.perf_counter()
        opt.step(); opt.zero_grad()
        t4 = time.perf_counter()

        phase_ms["forward"] += (t1 - t0) * 1e3
        phase_ms["backward"] += (t2 - t1) * 1e3
        phase_ms["comm"] += (t3 - t2) * 1e3
        phase_ms["optimizer"] += (t4 - t3) * 1e3

        # Log the GLOBAL mean loss (an all-reduce of a scalar — cheap).
        loss_t = loss.detach().clone()
        dist.all_reduce(loss_t)
        losses.append(loss_t.item() / world)
        if rank == 0 and (step % max(1, steps // 8) == 0 or step == steps - 1):
            print(f"step {step:>3}  global loss {losses[-1]:.4f}", flush=True)

    elapsed = time.perf_counter() - t_start
    tok_s = steps * bsz * NB_CFG.block_size * world / elapsed
    if rank == 0:
        total = sum(phase_ms.values())
        print(f"\nper-phase time ({steps} steps, world={world}):")
        for k, v in phase_ms.items():
            print(f"  {k:<10} {v / steps:8.1f} ms/step  ({100 * v / total:4.1f}%)")
        print(f"throughput: {tok_s:,.0f} tokens/s (global)")
        torch.save({"losses": losses, "phase_ms": phase_ms, "steps": steps,
                    "world": world, "tokens_per_s": tok_s}, out_path)


def w_grad_equivalence(rank: int, world: int, out_path: str) -> None:
    """The ch08 theorem, tested: averaged per-shard grads == grad of the
    full concatenated batch, up to float reduction-order noise.

    Every rank has the same seeded model and can reconstruct every rank's
    shard, so each rank verifies the claim locally after the all-reduce.
    """
    torch.manual_seed(0)
    model = MiniGPT(NB_CFG)
    reference = MiniGPT(NB_CFG)
    reference.load_state_dict(model.state_dict())

    stream = synthetic_tokens(n_tokens=8192)
    per_rank_idx = [list(range(r * 4, r * 4 + 4)) for r in range(world)]

    # Distributed path: my shard, then average.
    x, y = make_batch(stream, per_rank_idx[rank], NB_CFG.block_size)
    model(x, y)[1].backward()
    average_gradients(model)

    # Single-process path: everyone's samples in one big batch.
    all_idx = [i for shard in per_rank_idx for i in shard]
    xg, yg = make_batch(stream, all_idx, NB_CFG.block_size)
    reference(xg, yg)[1].backward()

    worst = 0.0
    for (n, p), q in zip(model.named_parameters(), reference.parameters()):
        d = (p.grad - q.grad).abs().max().item()
        worst = max(worst, d)
        assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4), \
            f"{n}: max diff {d:.2e}"
    if rank == 0:
        print("DP-averaged grads == single-process large-batch grads")
        print(f"max abs difference over all params: {worst:.2e}")
        print("(not 0.0: float addition order differs -> ch08 'bitwise-ish' caveat)")
        torch.save({"max_abs_diff": worst}, out_path)


def w_sampler_overlap_bug(rank: int, world: int) -> None:
    """Anti-demo: shuffle with a RANK-DEPENDENT seed and count collisions —
    the bug shard_indices' shared seed exists to prevent."""
    n = 64
    g = torch.Generator().manual_seed(rank)       # WRONG: per-rank seed
    perm = torch.randperm(n, generator=g).tolist()
    mine = set(perm[rank::world])

    shards: list = [None] * world
    dist.all_gather_object(shards, sorted(mine))
    if rank == 0:
        seen: set[int] = set()
        duplicated = 0
        for s in shards:
            duplicated += len(seen & set(s))
            seen |= set(s)
        print(f"independent shuffles: {len(seen)}/{n} samples covered, "
              f"{duplicated} duplicated -> duplicated samples are silently "
              f"over-weighted in the averaged gradient; missing ones never train",
              flush=True)

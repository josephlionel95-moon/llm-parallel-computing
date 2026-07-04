"""Render the naive-DP vs DDP overlap timeline from MEASURED timestamps.

    torchrun --nproc_per_node=2 experiments/ch09/overlap_timeline.py --save-plots
    python experiments/ch09/overlap_timeline.py --nproc 2 --save-plots

Naive DP:  |---- backward ----||-- all_reduce --|      comm serialized
ManualDDP: |---- backward ----||wait|                  comm rides inside:
              |b_k|...|b_1|b_0|                        buckets launch as
                                                       their grads appear
Timestamps come from ManualDDP.bucket_events (t_launch recorded in the hook
that fires mid-backward; t_done when finalize()'s wait returned). Figure is
built with llmdist.viz.plots.timeline and saved to visualizations/.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402
from llmdist.parallel.manual_ddp import ManualDDP  # noqa: E402
from llmdist.parallel.manual_dp import (  # noqa: E402
    average_gradients, broadcast_parameters)
from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--bucket-mb", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def worker(rank: int, world: int) -> None:
    args = parse_args()
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=args.n_layer,
                    n_head=4, d_model=args.d_model)
    torch.manual_seed(0)
    base = MiniGPT(cfg)
    g = torch.Generator().manual_seed(rank)
    x = torch.randint(0, cfg.vocab_size, (args.bsz, cfg.block_size), generator=g)
    y = x.roll(-1, dims=1)

    # --- naive DP: backward, then comm --------------------------------------
    m_dp = copy.deepcopy(base); broadcast_parameters(m_dp)
    m_dp(x, y)[1].backward()                     # warmup
    for p in m_dp.parameters():
        p.grad = None
    dist.barrier()
    t0 = time.perf_counter()
    m_dp(x, y)[1].backward()
    t_bwd = time.perf_counter()
    average_gradients(m_dp)
    t_comm = time.perf_counter()

    # --- ManualDDP: buckets launch during backward ---------------------------
    m_ddp = ManualDDP(copy.deepcopy(base), bucket_size_mb=args.bucket_mb)
    m_ddp(x, y)[1].backward(); m_ddp.finalize()  # warmup
    for p in m_ddp.parameters():
        p.grad = None
    dist.barrier()
    d0 = time.perf_counter()
    m_ddp(x, y)[1].backward()
    d_bwd = time.perf_counter()
    m_ddp.finalize()
    d_fin = time.perf_counter()

    if rank != 0:
        return
    ms = lambda a, b: (b - a) * 1e3               # noqa: E731
    naive_total, ddp_total = ms(t0, t_comm), ms(d0, d_fin)
    print(f"naive DP : bwd {ms(t0, t_bwd):7.1f} + comm {ms(t_bwd, t_comm):7.1f} "
          f"= {naive_total:7.1f} ms")
    print(f"ManualDDP: bwd {ms(d0, d_bwd):7.1f} + wait {ms(d_bwd, d_fin):7.1f} "
          f"= {ddp_total:7.1f} ms   "
          f"({len(m_ddp.bucket_events)} buckets, {args.bucket_mb} MB)")
    for i, tl, td in m_ddp.bucket_events:
        launched = "DURING backward" if tl < d_bwd else "after backward"
        print(f"  bucket {i:>2}: launched at {ms(d0, tl):7.1f} ms ({launched}), "
              f"wait returned {ms(d0, td):7.1f} ms")
    print("\nbucket launches inside the backward window ARE the overlap. "
          "'wait returned' is an upper bound on transfer completion (we cannot "
          "observe the transport's interior).")

    if args.save_plots:
        from llmdist.viz.plots import timeline
        events = [("naive DP", "compute", 0.0, ms(t0, t_bwd)),
                  ("naive DP", "comm", ms(t0, t_bwd), naive_total),
                  ("ManualDDP", "compute", 0.0, ms(d0, d_bwd)),
                  ("ManualDDP", "sync", ms(d0, d_bwd), ddp_total)]
        events += [(f"DDP bucket {i}", "comm", ms(d0, tl), ms(d0, td))
                   for i, tl, td in m_ddp.bucket_events]
        out = os.path.join(ROOT, "visualizations", "ch09_overlap_timeline.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        timeline(events, title=f"Naive DP vs ManualDDP, one step, rank 0 "
                               f"(world={world}, measured)", save=out)
        print(f"saved {out}")


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        rank, world, _ = setup_from_env()
        try:
            worker(rank, world)
        finally:
            cleanup()
    else:
        run_distributed(worker, world_size=args.nproc, port=29557)


if __name__ == "__main__":
    main()

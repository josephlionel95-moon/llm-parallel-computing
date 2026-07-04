"""Manual DP vs ManualDDP vs torch DDP: step time, breakdown, correctness.

    torchrun --nproc_per_node=2 experiments/ch09/ddp_comparison.py
    python experiments/ch09/ddp_comparison.py --nproc 2

Three implementations of the same mathematics (average gradients across
replicas), three different schedules:
  manual DP : backward, THEN one flat all_reduce           (serialized)
  ManualDDP : per-bucket async all_reduce DURING backward  (overlapped)
  torch DDP : same idea, C++ Reducer, tuned                (overlapped++)
Also asserts ManualDDP's grads equal torch DDP's — schedule must never
change the math.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402
from llmdist.parallel.manual_ddp import ManualDDP  # noqa: E402
from llmdist.parallel.manual_dp import (  # noqa: E402
    average_gradients, broadcast_parameters)
from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--bucket-mb", type=float, default=1.0,
                    help="bucket size for ManualDDP and torch DDP")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    return ap.parse_args()


def worker(rank: int, world: int) -> None:
    args = parse_args()
    device = (torch.device(f"cuda:{rank}") if dist.get_backend() == "nccl"
              else torch.device("cpu"))
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=args.n_layer,
                    n_head=4, d_model=args.d_model)
    torch.manual_seed(0)
    base = MiniGPT(cfg).to(device)

    def batch(step: int):
        g = torch.Generator().manual_seed(step * 1000 + rank)
        x = torch.randint(0, cfg.vocab_size, (args.bsz, cfg.block_size), generator=g)
        return x.to(device), x.roll(-1, dims=1).to(device)

    m_dp = copy.deepcopy(base); broadcast_parameters(m_dp)
    m_man = ManualDDP(copy.deepcopy(base), bucket_size_mb=args.bucket_mb)
    m_tor = nn.parallel.DistributedDataParallel(
        copy.deepcopy(base), bucket_cap_mb=args.bucket_mb,
        device_ids=[rank] if device.type == "cuda" else None)

    def zero(m: nn.Module) -> None:
        for p in m.parameters():
            p.grad = None

    # Per-phase breakdown for each implementation (fwd / bwd / sync).
    def run(name: str, model, sync_fn) -> dict[str, float]:
        t = {"forward": 0.0, "backward": 0.0, "sync": 0.0}
        for s in range(2):                        # warmup, incl. bucket building
            model(*batch(s))[1].backward(); sync_fn(); zero(model)
        dist.barrier()
        wall0 = time.perf_counter()
        for s in range(args.steps):
            x, y = batch(s + 10)
            t0 = time.perf_counter()
            _, loss = model(x, y)
            t1 = time.perf_counter()
            loss.backward()
            t2 = time.perf_counter()
            sync_fn()
            t3 = time.perf_counter()
            zero(model)
            t["forward"] += t1 - t0; t["backward"] += t2 - t1; t["sync"] += t3 - t2
        dist.barrier()
        t["total_wall"] = time.perf_counter() - wall0
        if rank == 0:
            ms = 1e3 / args.steps
            print(f"{name:<18} fwd {t['forward'] * ms:7.1f}  "
                  f"bwd(+comm) {t['backward'] * ms:7.1f}  "
                  f"sync/wait {t['sync'] * ms:7.1f}  "
                  f"= {t['total_wall'] * 1e3 / args.steps:7.1f} ms/step", flush=True)
        return t

    if rank == 0:
        n = base.num_params()
        print(f"model {n:,} params ({n * 4 / 2**20:.1f} MB grads), world={world}, "
              f"backend={dist.get_backend()}, bucket={args.bucket_mb} MB, "
              f"{len(m_man.buckets)} ManualDDP buckets\n(times in ms/step)")

    run("manual DP (naive)", m_dp, lambda: average_gradients(m_dp))
    run("ManualDDP", m_man, m_man.finalize)
    run("torch DDP", m_tor, lambda: None)         # sync hidden inside backward

    # ---- correctness: schedule must not change the math -------------------
    x, y = batch(999)
    m_man(x, y)[1].backward(); m_man.finalize()
    m_tor(x, y)[1].backward()
    worst = 0.0
    for pm, pt in zip(m_man.module.parameters(), m_tor.module.parameters()):
        worst = max(worst, (pm.grad - pt.grad).abs().max().item())
        assert torch.allclose(pm.grad, pt.grad, atol=1e-6, rtol=1e-5), \
            f"grad mismatch: {worst:.2e}"
    if rank == 0:
        print(f"\nManualDDP grads == torch DDP grads, max diff {worst:.2e}  [OK]")
        print("interpretation: 'bwd(+comm)' for DDP rows contains the overlapped "
              "communication; naive DP pays it separately in 'sync'. On CPU/Gloo "
              "overlap gains are modest (compute and comm share cores) — on 2 GPUs "
              "with NCCL the DDP rows pull clearly ahead.")


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        rank, world, _ = setup_from_env()
        try:
            worker(rank, world)
        finally:
            cleanup()
    else:
        run_distributed(worker, world_size=args.nproc, port=29555)


if __name__ == "__main__":
    main()

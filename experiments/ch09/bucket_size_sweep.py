"""Bucket-size sweep: the latency-vs-overlap U-curve, measured.

    torchrun --nproc_per_node=2 experiments/ch09/bucket_size_sweep.py
    python experiments/ch09/bucket_size_sweep.py --nproc 2 --sizes 0.25,1,5,25,100

Theory (ch09): with B buckets over N bytes of gradients,
  latency cost   ~ B * 2(p-1) * alpha          (each bucket pays messages)
  overlap loss   ~ last bucket of N/B bytes cannot start before its grads
                   exist; too few buckets => comm serializes after backward.
Step time is therefore U-shaped in bucket size. torch defaults to 25 MB.
On CPU/Gloo the curve is flatter (compute shares cores with comm), but the
small-bucket latency wall is unmistakable.
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
from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--sizes", type=str, default="0.1,0.5,1,5,25,100",
                    help="bucket sizes in MB, comma separated")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def worker(rank: int, world: int) -> None:
    args = parse_args()
    sizes_mb = [float(s) for s in args.sizes.split(",")]
    device = (torch.device(f"cuda:{rank}") if dist.get_backend() == "nccl"
              else torch.device("cpu"))
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=args.n_layer,
                    n_head=4, d_model=args.d_model)
    torch.manual_seed(0)
    base = MiniGPT(cfg).to(device)
    grad_mb = base.num_params() * 4 / 2**20

    def batch(step: int):
        g = torch.Generator().manual_seed(step * 1000 + rank)
        x = torch.randint(0, cfg.vocab_size, (args.bsz, cfg.block_size), generator=g)
        return x.to(device), x.roll(-1, dims=1).to(device)

    if rank == 0:
        print(f"model {base.num_params():,} params -> {grad_mb:.1f} MB of grads, "
              f"world={world}, backend={dist.get_backend()}")
        print(f"{'bucket MB':>10} {'#buckets':>9} {'ms/step':>9}")

    results: dict[float, float] = {}
    nbuckets: dict[float, int] = {}
    for mb in sizes_mb:
        model = ManualDDP(copy.deepcopy(base), bucket_size_mb=mb)
        nbuckets[mb] = len(model.buckets)

        def step_fn(s: int) -> None:
            model(*batch(s))[1].backward()
            model.finalize()
            for p in model.parameters():
                p.grad = None

        step_fn(0)                                # warmup
        dist.barrier()
        t0 = time.perf_counter()
        for s in range(args.steps):
            step_fn(s + 1)
        dist.barrier()
        results[mb] = (time.perf_counter() - t0) * 1e3 / args.steps
        if rank == 0:
            print(f"{mb:>10.2f} {nbuckets[mb]:>9} {results[mb]:>9.1f}", flush=True)

    if rank == 0:
        best = min(results, key=results.get)
        print(f"\nfastest at {best} MB buckets here. Expect a U: tiny buckets pay "
              f"2(p-1)*alpha per bucket; one giant bucket ({grad_mb:.0f}+ MB) can't "
              f"launch until backward is nearly done, killing overlap.")
        if args.save_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = sorted(results)
            ax.plot(xs, [results[x] for x in xs], "o-")
            for x in xs:
                ax.annotate(f"{nbuckets[x]} bkts", (x, results[x]), fontsize=8,
                            textcoords="offset points", xytext=(4, 6))
            ax.set_xscale("log")
            ax.set_xlabel("bucket size (MB)"); ax.set_ylabel("ms/step")
            ax.set_title(f"ManualDDP bucket-size sweep (world={world}, "
                         f"{grad_mb:.0f} MB grads)")
            ax.grid(alpha=0.3)
            out = os.path.join(ROOT, "visualizations", "ch09_bucket_sweep.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.tight_layout(); fig.savefig(out, dpi=120)
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
        run_distributed(worker, world_size=args.nproc, port=29556)


if __name__ == "__main__":
    main()

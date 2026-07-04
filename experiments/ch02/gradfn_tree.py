"""Walk MiniGPT's grad_fn graph and print it as a tree.

The graph autograd builds during forward is a real, inspectable data structure:
loss.grad_fn is the root; node.next_functions point toward the leaves
(AccumulateGrad nodes wrapping parameters). This script prints it, counts node
types, and shows the diamonds created by residual connections.

Run:  python experiments/ch02/gradfn_tree.py [--layers 1] [--max-depth 14]
CPU-only friendly (tiny model by default).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check


def print_tree(fn, depth: int, seen: set[int], max_depth: int, counter: Counter) -> None:
    if fn is None:
        return
    name = type(fn).__name__
    # Annotate AccumulateGrad with the parameter it feeds, when identifiable.
    if name == "AccumulateGrad" and hasattr(fn, "variable"):
        name += f"  ->  param{tuple(fn.variable.shape)}"
    revisit = id(fn) in seen
    if depth <= max_depth:
        print("  " * depth + ("~ " if revisit else "- ") + name
              + ("  (already shown: DAG diamond)" if revisit else ""))
    if revisit:
        return
    seen.add(id(fn))
    counter[type(fn).__name__] += 1
    for child, _ in fn.next_functions:
        print_tree(child, depth + 1, seen, max_depth, counter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=14)
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    torch.manual_seed(0)

    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=args.layers,
                    n_head=2, d_model=16)
    model = MiniGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(idx, targets=idx)

    print(f"MiniGPT({args.layers} layer) loss graph, depth <= {args.max_depth}:")
    print("('-' = first visit, '~' = revisited node: residuals make the graph a DAG)\n")
    sys.setrecursionlimit(10_000)
    counter: Counter = Counter()
    print_tree(loss.grad_fn, 0, set(), args.max_depth, counter)

    print(f"\nUnique nodes in the whole graph: {sum(counter.values())}")
    print("Node census (top 12):")
    for name, cnt in counter.most_common(12):
        print(f"  {name:<28} x{cnt}")
    n_acc = counter.get("AccumulateGrad", 0)
    n_params = sum(1 for _ in model.parameters())
    print(f"\nAccumulateGrad nodes = {n_acc}, model parameters = {n_params} "
          f"({'match' if n_acc == n_params else 'MISMATCH?'})")
    print("Each AccumulateGrad is where '.grad +=' happens - and where DDP-style")
    print("hooks announce 'this gradient is ready' (see grad_arrival_order.py).")


if __name__ == "__main__":
    main()

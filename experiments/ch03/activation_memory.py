"""Measure activation memory vs the ch03 formula, sweeping batch and sequence.

Two torchinfo-free instruments:
  1. forward hooks on every module record OUTPUT bytes (the classic estimator);
  2. torch.autograd.graph.saved_tensors_hooks weighs exactly what autograd
     saves for backward, deduplicated by storage (ground truth; CPU-safe).

Formula under test (fp32 MiniGPT, dropout=0, derived in docs/ch03):
  per layer:  s*b*h*(64 + 4*a*s/h) bytes
  + logits tail: ~8*s*b*V bytes (logits + log-softmax) - dominant when V >> d!

Run:  python experiments/ch03/activation_memory.py [--d-model 256] [--layers 4]
CPU-only friendly; on GPU also reports max_memory_allocated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check


def formula_bytes(b: int, s: int, cfg: GPTConfig) -> tuple[float, float]:
    h, a = cfg.d_model, cfg.n_head
    per_layer = s * b * h * (64 + 4 * a * s / h)
    tail = 8 * s * b * cfg.vocab_size
    return cfg.n_layer * per_layer, tail


def measure(model: MiniGPT, idx: torch.Tensor, device: torch.device) -> dict[str, float]:
    saved: dict[int, int] = {}
    hook_out_bytes = 0

    def fwd_hook(_m, _inp, out) -> None:
        nonlocal hook_out_bytes
        if isinstance(out, torch.Tensor):
            hook_out_bytes += out.numel() * out.element_size()

    def pack(t: torch.Tensor) -> torch.Tensor:
        base = t if t._base is None else t._base
        saved[id(base)] = base.numel() * base.element_size()
        return t

    handles = [m.register_forward_hook(fwd_hook) for m in model.modules()]
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = model(idx, targets=idx)
    loss.backward()
    model.zero_grad(set_to_none=True)
    for h in handles:
        h.remove()
    out = {"saved_mb": sum(saved.values()) / 2**20, "hook_mb": hook_out_bytes / 2**20}
    if device.type == "cuda":
        torch.cuda.synchronize()
        out["peak_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    torch.manual_seed(0)

    cfg = GPTConfig(n_layer=args.layers, d_model=args.d_model,
                    n_head=max(2, args.d_model // 64))
    model = MiniGPT(cfg).to(device)
    print(f"MiniGPT: L={cfg.n_layer}, d={cfg.d_model}, a={cfg.n_head}, V={cfg.vocab_size}; "
          f"params {model.num_params() * 4 / 2**20:.1f} MiB fp32\n")

    extra = "  " + f"{'CUDA peak':>10}" if device.type == "cuda" else ""
    hdr = (f"{'b':>3} {'s':>5} {'formula MiB':>12} {'saved MiB':>10} "
           f"{'ratio':>6} {'fwd-hook MiB':>13}{extra}")
    print(hdr)
    print("-" * len(hdr))
    for b, s in [(1, 128), (2, 128), (4, 128), (8, 128), (4, 64), (4, 256)]:
        idx = torch.randint(0, cfg.vocab_size, (b, s), device=device)
        r = measure(model, idx, device)
        blocks, tail = formula_bytes(b, s, cfg)
        pred = (blocks + tail) / 2**20
        row = (f"{b:>3} {s:>5} {pred:>12.1f} {r['saved_mb']:>10.1f} "
               f"{r['saved_mb'] / pred:>6.2f} {r['hook_mb']:>13.1f}")
        if "peak_mb" in r:
            row += f"  {r['peak_mb']:>10.1f}"
        print(row)

    b, s = 4, 128
    blocks, tail = formula_bytes(b, s, cfg)
    print(
        f"\nAt b={b}, s={s}: blocks account for {blocks / 2**20:.1f} MiB, the logits"
        f"\ntail (8sbV) for {tail / 2**20:.1f} MiB - with V = {cfg.vocab_size // cfg.d_model}x d, the"
        "\nloss tail rivals or beats all transformer blocks combined. Real systems"
        "\nfuse the loss (Megatron's vocab-parallel CE, ch10) for exactly this reason."
        "\n\nReading the sweep: memory is linear in b, super-linear in s (the a*s^2*b"
        "\nsoftmax term). Params never moved. Activations are a DATA-SHAPE bill."
        "\nThe fwd-hook column over-counts (it sums ALL module outputs, kept or not)"
        "\n- a good estimator's upper bound, and why saved_tensors_hooks is the scale."
    )


if __name__ == "__main__":
    main()

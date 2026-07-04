"""Measure pickle size and time: tensor vs numpy vs Python list.

Why: every object that crosses a process boundary (DataLoader workers,
mp.spawn args, dist object collectives) is serialized. Buffers (tensors,
ndarrays) serialize as one memcpy-like dump; Python containers serialize
per element through the interpreter. This script quantifies the gap.

Run:  python experiments/ch00/pickle_bench.py [--n 1000000]
CPU-only; no GPU required.
"""
from __future__ import annotations

import argparse
import io
import pickle
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch


def time_ms(fn, repeats: int = 5) -> float:
    """Median wall time in ms (CPU work only, so perf_counter is honest here)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def torch_save_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000, help="number of float32 elements")
    args = ap.parse_args()
    n = args.n

    tensor32 = torch.randn(n)
    tensor16 = tensor32.to(torch.float16)
    array32 = tensor32.numpy().copy()
    pylist = tensor32.tolist()

    cases = [
        ("torch fp32 tensor (pickle)", lambda: pickle.dumps(tensor32)),
        ("torch fp16 tensor (pickle)", lambda: pickle.dumps(tensor16)),
        ("numpy fp32 array  (pickle)", lambda: pickle.dumps(array32)),
        ("python list[float](pickle)", lambda: pickle.dumps(pylist)),
        ("torch fp32 tensor (torch.save)", lambda: torch_save_bytes(tensor32)),
    ]

    raw = 4 * n
    print(f"n = {n:,} elements, raw fp32 payload = {raw / 2**20:.1f} MiB\n")
    header = f"{'object':<32} {'bytes':>14} {'x raw':>7} {'ms':>10} {'MB/s':>10}"
    print(header)
    print("-" * len(header))
    for name, fn in cases:
        blob = fn()
        ms = time_ms(fn)
        mbps = (len(blob) / 2**20) / (ms / 1e3) if ms > 0 else float("inf")
        print(f"{name:<32} {len(blob):>14,} {len(blob) / raw:>7.2f} {ms:>10.2f} {mbps:>10.0f}")

    print(
        "\nReading the table:\n"
        "  - tensor/ndarray pickles ~= payload + small header, at memcpy-like MB/s.\n"
        "  - the Python list is ~9 B/element (8 B double + opcode) but 10-100x slower\n"
        "    per byte: pickling walks n PyObjects through the interpreter.\n"
        "  - fp16 halves the bytes: dtype choice is a *communication* choice too\n"
        "    (this is why gradient compression and bf16 comms exist, ch20/ch23)."
    )


if __name__ == "__main__":
    main()

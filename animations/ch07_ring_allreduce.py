"""Animated ring all-reduce on 4 ranks: watch every chunk travel and reduce.

Produces:
  animations/ch07_ring_allreduce.gif           (FuncAnimation + PillowWriter)
  visualizations/ch07_ring_allreduce_steps.png (static multi-panel, for docs)

The visualization is a 4x4 grid per time step: row = rank, column = chunk.
Each cell lists WHICH ranks' contributions its copy of that chunk currently
contains ("0+1+2" = partial sum of ranks 0,1,2); shading encodes the count.
The schedule matches src/llmdist/comm/manual_collectives.all_reduce_ring
exactly: p-1 reduce-scatter steps, then p-1 all-gather steps.

Run:  python animations/ch07_ring_allreduce.py [--ranks 4] [--no-gif]
The notebook (ch07) imports build_states()/draw_state() to embed the same
animation inline.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

State = list[list[frozenset[int]]]  # state[rank][chunk] = contributing ranks


def build_states(p: int = 4) -> tuple[list[State], list[str]]:
    """Simulate the ring schedule and return the state after every step.

    Returns (states, captions): 1 initial state + (p-1) reduce-scatter states
    + (p-1) all-gather states = 2p-1 total, each with a caption describing
    the sends that produced it.
    """
    state: State = [[frozenset({r}) for _ in range(p)] for r in range(p)]
    states: list[State] = [[row[:] for row in state]]
    captions = ["initial: rank r owns only its own contribution to every chunk"]

    # Phase 1 — reduce-scatter: step s, rank r sends chunk (r-1-s) mod p to
    # r+1, which ADDS it into its copy. Chunk k ends fully reduced at rank k.
    for s in range(p - 1):
        new = [row[:] for row in state]
        for r in range(p):
            sender = (r - 1) % p
            idx = (sender - 1 - s) % p           # chunk the sender passes on
            new[r][idx] = state[r][idx] | state[sender][idx]
        state = new
        states.append([row[:] for row in state])
        captions.append(
            f"reduce-scatter step {s + 1}/{p - 1}: rank r sends chunk (r-1-{s}) mod {p} "
            f"to r+1, receiver adds it (reduction happens en route)")

    # Phase 2 — all-gather: step s, rank r sends chunk (r-s) mod p to r+1,
    # which OVERWRITES its copy with the fully-reduced version.
    for s in range(p - 1):
        new = [row[:] for row in state]
        for r in range(p):
            sender = (r - 1) % p
            idx = (sender - s) % p
            new[r][idx] = state[sender][idx]
        state = new
        states.append([row[:] for row in state])
        captions.append(
            f"all-gather step {s + 1}/{p - 1}: rank r sends chunk (r-{s}) mod {p} "
            f"to r+1, receiver copies (no adds left to do)")
    return states, captions


def draw_state(ax: plt.Axes, state: State, caption: str, p: int) -> None:
    """Render one time step as a shaded rank x chunk grid."""
    counts = np.array([[len(state[r][c]) for c in range(p)] for r in range(p)],
                      dtype=float)
    ax.imshow(counts, cmap="Greens", vmin=0.5, vmax=p, aspect="equal")
    for r in range(p):
        for c in range(p):
            members = sorted(state[r][c])
            label = "+".join(str(m) for m in members) if len(members) < p else "ALL"
            full = len(members) == p
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=9 if not full else 10,
                    fontweight="bold" if full else "normal",
                    color="white" if len(members) >= p - 1 else "black")
    ax.set_xticks(range(p), [f"chunk {c}" for c in range(p)], fontsize=8)
    ax.set_yticks(range(p), [f"rank {r}" for r in range(p)], fontsize=8)
    ax.set_title(caption, fontsize=9, wrap=True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_gif(p: int = 4, path: str | None = None, interval_ms: int = 1400) -> str:
    """Save the animation as a GIF via PillowWriter; returns the path."""
    states, captions = build_states(p)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    def frame(i: int):
        ax.clear()
        draw_state(ax, states[i], f"step {i}/{len(states) - 1}\n{captions[i]}", p)
        return []

    anim = animation.FuncAnimation(fig, frame, frames=len(states),
                                   interval=interval_ms, blit=False)
    path = path or os.path.join(HERE, "ch07_ring_allreduce.gif")
    anim.save(path, writer=animation.PillowWriter(fps=1000 / interval_ms))
    plt.close(fig)
    return path


def save_static_panels(p: int = 4, path: str | None = None) -> str:
    """Static multi-panel version of every step, embeddable in the docs."""
    states, captions = build_states(p)
    n = len(states)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            phase = ("start" if i == 0 else
                     f"RS {i}/{p - 1}" if i <= p - 1 else f"AG {i - (p - 1)}/{p - 1}")
            draw_state(ax, states[i], f"step {i} ({phase})", p)
        else:
            ax.axis("off")
    fig.suptitle(f"Ring all-reduce, {p} ranks: {p - 1} reduce-scatter steps "
                 f"+ {p - 1} all-gather steps. Cell = which ranks' data this "
                 f"copy contains.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = path or os.path.join(REPO, "visualizations", "ch07_ring_allreduce_steps.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--no-gif", action="store_true", help="skip the (slow) GIF")
    args = ap.parse_args()
    print("static panels ->", save_static_panels(args.ranks))
    if not args.no_gif:
        print("gif           ->", save_gif(args.ranks))

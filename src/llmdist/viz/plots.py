"""Shared visualization helpers: memory maps and communication timelines."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def memory_map(breakdowns: dict[str, dict[str, float]], title: str = "GPU memory breakdown",
               save: str | None = None) -> None:
    """Stacked bar chart of memory by category, one bar per configuration.

    breakdowns: {"1 GPU baseline": {"params": 480, "grads": 480, "optimizer": 960,
                 "activations": 1200}, "ZeRO-2": {...}}  (values in MB)
    """
    cats: list[str] = []
    for b in breakdowns.values():
        for k in b:
            if k not in cats:
                cats.append(k)
    labels = list(breakdowns)
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 4.5))
    bottom = np.zeros(len(labels))
    for c in cats:
        vals = np.array([breakdowns[l].get(c, 0.0) for l in labels])
        ax.bar(labels, vals, bottom=bottom, label=c)
        bottom += vals
    ax.set_ylabel("MB")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    plt.show()


def timeline(events: list[tuple[str, str, float, float]], title: str = "Timeline",
             save: str | None = None) -> None:
    """Gantt-style compute/communication timeline.

    events: list of (lane, kind, start_ms, end_ms); kind in
    {"compute", "comm", "idle", "sync"} controls color.
    Lanes are typically "GPU 0 compute", "GPU 0 comm", "GPU 1 compute", ...
    """
    colors = {"compute": "#4c72b0", "comm": "#dd8452", "idle": "#cccccc", "sync": "#c44e52"}
    lanes: list[str] = []
    for lane, *_ in events:
        if lane not in lanes:
            lanes.append(lane)
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(lanes) + 1.5))
    for lane, kind, s, e in events:
        y = lanes.index(lane)
        ax.barh(y, e - s, left=s, height=0.6, color=colors.get(kind, "#999999"),
                edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(lanes)), lanes)
    ax.invert_yaxis()
    ax.set_xlabel("time (ms)")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=8, ncols=4)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    plt.show()

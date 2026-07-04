# Chapter Style Guide

Every chapter consists of three artifacts. Authors (human or AI) must follow this template
so the course feels like one book, not thirty blog posts.

## 1. `docs/chXX_<slug>.md` — the deep-dive

Required sections, in order:

```markdown
# Chapter XX — <Title>

> **Difficulty:** 🟢/🟡/🔴/🎓 · **Study time:** N h · **Requires:** chapters A, B
> **Notebook:** notebooks/chXX_<slug>.ipynb · **Experiments:** experiments/chXX/

## Learning objectives        (5–8 bullet "you will be able to..." items)
## Intuition                  (why does this exist; analogy; the problem it solves)
## Theory                     (the mechanism, with ASCII diagrams)
## Mathematics                (derivations step by step — never jump to a formula)
## Implementation             (walk through the src/ + notebook code, key excerpts inline)
## Profiling & measurement    (what to measure, expected numbers on T4s, how to read them)
## Common mistakes            (numbered, each with symptom → cause → fix)
## Limitations & outlook      (what breaks at scale; what later chapters fix)
## Exercises                  (Easy / Medium / Hard / Research-level, ≥6 total)
## Solutions                  (collapsed under <details> tags)
## Interview questions        (≥8, no answers — answers are earned by studying)
## Summary                    (10 bullets max)
## References                 (papers with one-line "why read this" annotations)
```

## 2. `notebooks_src/chXX_<slug>.py` — the notebook (jupytext py:percent)

- First cell: title + hardware requirements banner.
- Second cell: environment detection via `llmdist.utils.env_check.detect()` — the notebook
  must adapt to 0/1/2 GPUs and print what mode it is running in.
- Alternate `# %% [markdown]` explanation cells with `# %%` code cells. Explanation cells
  teach; they never merely narrate the code below them.
- Every experiment cell prints measured numbers (memory MB, ms, GB/s) and, where useful,
  renders a matplotlib figure.
- Multi-process cells use `torch.multiprocessing.spawn` with `gloo` on CPU so they run
  *inside* a notebook on any machine; equivalent `torchrun` scripts live in `experiments/`.
- End with: Exercises, Solutions (in comments or `<details>`), Summary.

## 3. `experiments/chXX/*.py` — measurement scripts

- Launchable via `torchrun --nproc_per_node=N` (or plain `python` for single-process).
- Must run on: 2× T4 (Kaggle), 1× T4 (Colab), CPU-only (CI). Detect and degrade gracefully.
- Print a results table (use `rich` if available, plain text fallback).
- Save figures to `visualizations/chXX_*.png` when `--save-plots` is passed.

## Code style

- PyTorch + `torch.distributed`; type hints on public functions; docstrings explain *why*.
- No hidden magic: important algorithms implemented in the chapter, not imported.
- Shared plumbing (env detection, timers, memory tracking, mini-GPT) lives in `src/llmdist/`.
- Keep modules < ~300 lines; prefer clarity over cleverness.

## Numbers discipline

Every performance claim must be reproducible by a script in `experiments/`. State the
hardware in the text ("on Kaggle 2× T4, PCIe"). Never invent benchmark numbers in docs —
write "measure with `experiments/chXX/foo.py`; on 2× T4 expect roughly X" only when the
expectation follows from bandwidth math shown in the chapter.

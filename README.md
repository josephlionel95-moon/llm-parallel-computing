# LLM Distributed Training — From First Principles

**Understand exactly what every GPU is doing at every step of training a transformer.**

This repository is a complete, hands-on course on distributed LLM training. The goal is not
to train a state-of-the-art model — it is to *understand* multi-GPU training deeply enough
to reason about trillion-parameter systems, by building every mechanism yourself first:
manual AllReduce, manual DDP, manual tensor/pipeline parallelism, manual ZeRO and FSDP —
then comparing against the official implementations.

> Designed to run on **free hardware**: Kaggle (2× T4) for real multi-GPU NCCL experiments,
> Colab (1× T4) with CPU/Gloo multi-process simulation fallbacks. Every notebook detects
> your hardware and adapts.

---

## Who is this for?

You know Python, basic PyTorch, and transformers. You have trained models on one GPU.
You have **never truly understood** what happens when a second GPU joins.

## What you will understand by the end

- How GPUs communicate (NCCL, collectives, ring/tree algorithms, NVLink vs PCIe vs InfiniBand)
- Why communication is expensive (bandwidth vs latency, α–β cost models)
- Where memory goes (parameters, gradients, optimizer states, activations, buffers, CUDA cache)
- Every parallelism axis: **data, tensor, pipeline, sequence, context, expert**
- ZeRO stages 1–3, FSDP, DeepSpeed, Megatron-LM's 3D parallelism
- How trillion-parameter models are actually trained
- How to profile, benchmark, and overlap compute with communication

## Course roadmap

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  PART I — FOUNDATIONS (single GPU)                      │
                 │  00 Python memory/process model    01 GPU architecture  │
                 │  02 PyTorch execution & autograd   03 Transformer math  │
                 │  04 Single-GPU training anatomy    05 Why multi-GPU?    │
                 └───────────────────────────┬─────────────────────────────┘
                                             │
                 ┌───────────────────────────▼─────────────────────────────┐
                 │  PART II — COMMUNICATION (the heart of the course)      │
                 │  06 Processes, ranks, torchrun, NCCL/Gloo               │
                 │  07 Collectives: broadcast/reduce/all-reduce/...        │
                 │  08 Data parallelism, built by hand                     │
                 │  09 DDP internals: buckets, hooks, overlap              │
                 └───────────────────────────┬─────────────────────────────┘
                                             │
                 ┌───────────────────────────▼─────────────────────────────┐
                 │  PART III — MODEL PARALLELISM                           │
                 │  10 Tensor parallel (Megatron-style)                    │
                 │  11 Pipeline parallel (GPipe, 1F1B, interleaved)        │
                 │  12 Sequence parallel   13 Context parallel             │
                 │  14 Expert parallel (MoE)                               │
                 └───────────────────────────┬─────────────────────────────┘
                                             │
                 ┌───────────────────────────▼─────────────────────────────┐
                 │  PART IV — MEMORY & SHARDING                            │
                 │  15 ZeRO 1/2/3        16 FSDP (manual → official)       │
                 │  17 DeepSpeed         18 Megatron 3D parallelism        │
                 │  19 Distributed checkpointing                           │
                 │  20 Mixed precision   21 Activation checkpointing       │
                 │  22 Gradient accumulation                               │
                 └───────────────────────────┬─────────────────────────────┘
                                             │
                 ┌───────────────────────────▼─────────────────────────────┐
                 │  PART V — SYSTEMS & SCALE                               │
                 │  23 Communication optimization  24 Profiling            │
                 │  25 Scaling experiments & laws  26 End-to-end mini-GPT  │
                 │  27 Modern training stack       28 Production clusters  │
                 │  29 Paper deep-dives                                    │
                 └─────────────────────────────────────────────────────────┘
```

## Chapter index

| # | Chapter | Difficulty | Study time | Docs | Notebook | Status |
|---|---------|-----------|------------|------|----------|--------|
| 00 | Python memory, processes & the GIL | 🟢 Easy | 3–4 h | [docs](docs/ch00_python_review.md) | [nb](notebooks/ch00_python_review.ipynb) | ✅ |
| 01 | GPU architecture: SMs, warps, memory hierarchy | 🟡 Medium | 4–6 h | [docs](docs/ch01_gpu_basics.md) | [nb](notebooks/ch01_gpu_basics.ipynb) | ✅ |
| 02 | PyTorch execution: autograd, hooks, checkpointing | 🟡 Medium | 4–6 h | [docs](docs/ch02_pytorch_execution.md) | [nb](notebooks/ch02_pytorch_execution.ipynb) | ✅ |
| 03 | Transformer internals: params, activations, FLOPs | 🟡 Medium | 5–7 h | [docs](docs/ch03_transformer_internals.md) | [nb](notebooks/ch03_transformer_internals.ipynb) | ✅ |
| 04 | Single-GPU training anatomy | 🟢 Easy | 3–4 h | [docs](docs/ch04_single_gpu_training.md) | [nb](notebooks/ch04_single_gpu_training.ipynb) | ✅ |
| 05 | Why multi-GPU? The three walls | 🟢 Easy | 2–3 h | [docs](docs/ch05_why_multi_gpu.md) | [nb](notebooks/ch05_why_multi_gpu.ipynb) | ✅ |
| 06 | Distributed computing basics: ranks, torchrun, NCCL | 🟡 Medium | 4–6 h | [docs](docs/ch06_distributed_basics.md) | [nb](notebooks/ch06_distributed_basics.ipynb) | ✅ |
| 07 | Collective communication, from scratch | 🔴 Hard | 6–8 h | [docs](docs/ch07_collective_communication.md) | [nb](notebooks/ch07_collective_communication.ipynb) | ✅ |
| 08 | Data parallelism, built by hand | 🟡 Medium | 4–6 h | [docs](docs/ch08_data_parallelism.md) | [nb](notebooks/ch08_data_parallelism.ipynb) | ✅ |
| 09 | DDP internals: buckets, hooks, overlap | 🔴 Hard | 6–8 h | [docs](docs/ch09_ddp_internals.md) | [nb](notebooks/ch09_ddp_internals.ipynb) | ✅ |
| 10 | Tensor parallelism (Megatron-style) | 🔴 Hard | 6–8 h | — | — | 🔜 Phase 2 |
| 11 | Pipeline parallelism | 🔴 Hard | 6–8 h | — | — | 🔜 Phase 2 |
| 12 | Sequence parallelism | 🟡 Medium | 3–4 h | — | — | 🔜 Phase 2 |
| 13 | Context parallelism | 🔴 Hard | 4–6 h | — | — | 🔜 Phase 2 |
| 14 | Expert parallelism (MoE) | 🔴 Hard | 6–8 h | — | — | 🔜 Phase 2 |
| 15 | ZeRO optimizer, stages 1–3 | 🔴 Hard | 6–8 h | — | — | 🔜 Phase 2 |
| 16 | FSDP: manual → official | 🔴 Hard | 6–8 h | — | — | 🔜 Phase 2 |
| 17 | DeepSpeed | 🟡 Medium | 4–5 h | — | — | 🔜 Phase 3 |
| 18 | Megatron concepts & 3D parallelism | 🔴 Hard | 5–7 h | — | — | 🔜 Phase 3 |
| 19 | Checkpointing: exact resume, atomic saves, sharded state | 🟡 Medium | 4–5 h | [docs](docs/ch19_checkpointing.md) | [nb](notebooks/ch19_checkpointing.ipynb) | ✅ |
| 20 | Mixed precision: FP32/FP16/BF16/FP8 | 🟡 Medium | 4–5 h | — | — | 🔜 Phase 3 |
| 21 | Activation checkpointing | 🟡 Medium | 3–4 h | — | — | 🔜 Phase 3 |
| 22 | Gradient accumulation | 🟢 Easy | 2–3 h | — | — | 🔜 Phase 3 |
| 23 | Communication optimization | 🔴 Hard | 5–7 h | — | — | 🔜 Phase 3 |
| 24 | Profiling: torch.profiler, Nsight, memory | 🟡 Medium | 4–6 h | — | — | 🔜 Phase 3 |
| 25 | Scaling experiments: Amdahl & Gustafson | 🟡 Medium | 4–6 h | — | — | 🔜 Phase 4 |
| 26 | End-to-end mini-GPT: DDP vs FSDP vs DeepSpeed | 🔴 Hard | 8–12 h | — | — | 🔜 Phase 4 |
| 27 | The modern LLM training stack | 🟢 Easy | 3–4 h | — | — | 🔜 Phase 4 |
| 28 | Production systems: DGX, NVSwitch, InfiniBand | 🟡 Medium | 4–5 h | — | — | 🔜 Phase 4 |
| 29 | Research paper deep-dives | 🎓 Research | ongoing | — | — | 🔜 Phase 4 |

**Total estimated study time (full course): ~130–170 hours.**

## How to study

1. **Read the chapter doc** (`docs/chXX_*.md`) — theory, math derivations, diagrams.
2. **Run the notebook** (`notebooks/chXX_*.ipynb`) — every concept re-derived in code.
3. **Run the experiments** (`experiments/chXX/`) — measure memory, throughput, communication.
4. **Do the exercises** — each chapter has Easy → Research-level exercises with solutions.
5. **Answer the interview questions** at the end of each chapter without looking.

Recommended order is numeric. Chapters 00–02 can be skimmed if you're confident, but
**do not skip 03 (transformer memory/FLOPs math) or 07 (collectives)** — everything
downstream depends on them.

## Quick start

```bash
git clone https://github.com/josephlionel95-moon/llm-parallel-computing.git
cd llm-parallel-computing
pip install -r requirements.txt

# Verify your environment (detects Colab / Kaggle / local, counts GPUs)
python -m llmdist.utils.env_check

# Run a 2-process distributed hello world (works even on CPU)
torchrun --nproc_per_node=2 experiments/ch06/hello_distributed.py

# Single-GPU baseline vs data parallelism, on the same model & data:
python experiments/baseline/train_single_gpu.py --global-batch 16
torchrun --nproc_per_node=2 experiments/ch08/manual_dp_train.py --bsz 8 --metrics
python experiments/baseline/compare_runs.py --save-plots
```

Every trainer writes the same [RunRecord](src/llmdist/utils/metrics.py) JSON, so any
strategy can be compared against the single-GPU baseline — throughput, speedup, scaling
efficiency, communication share, peak memory, and loss curves. See
[experiments/baseline/](experiments/baseline/README.md).

### On Kaggle (2× T4 — recommended for real multi-GPU)

Create a notebook with accelerator **GPU T4 ×2**, then:

```python
!git clone https://github.com/josephlionel95-moon/llm-parallel-computing.git
%cd llm-parallel-computing
!pip install -q -r requirements.txt
!torchrun --nproc_per_node=2 experiments/ch08/manual_dp_train.py
```

### On Colab (1× T4)

All multi-GPU code falls back to **multi-process CPU + Gloo**, which preserves the
*semantics* of every collective (you learn the same thing, just without NCCL speed).
Notebooks flag which cells need real 2-GPU hardware.

## Repository layout

```
docs/            One markdown deep-dive per chapter (theory, math, diagrams)
notebooks/       One runnable notebook per chapter (.ipynb, hardware-adaptive)
notebooks_src/   Jupytext py:percent sources of the notebooks (easier to diff/review)
src/llmdist/     Shared library: models, distributed utils, profilers, viz
experiments/     Per-chapter measurement scripts (torchrun-launchable)
visualizations/  Memory maps, communication timelines, topology diagrams
animations/      Animated collective-communication diagrams (matplotlib/HTML)
profiling/       Profiler traces and how to read them
tests/           Unit tests (CPU/Gloo — run anywhere)
cheatsheets/     One-page summaries: collectives, memory math, parallelism choices
references/      Annotated bibliography per chapter
scripts/         torchrun launchers for Kaggle / local
```

## Prerequisites

- Python ≥ 3.10, PyTorch ≥ 2.2
- Comfort with `nn.Module`, backprop, and a basic transformer
- No CUDA C++ required (we read a little PTX/SASS conceptually in ch01, that's all)

## Guiding principles

1. **Build it before you import it.** Manual AllReduce before `dist.all_reduce`; manual
   DDP before `nn.parallel.DistributedDataParallel`; manual ZeRO before DeepSpeed.
2. **Measure everything.** Every chapter profiles memory and time. Claims come with numbers.
3. **Every byte accounted for.** If memory "explodes", we show exactly which tensor did it.
4. **Free-tier first.** Everything runs on Kaggle 2× T4 or degrades gracefully to CPU.

## License

MIT — see [LICENSE](LICENSE).

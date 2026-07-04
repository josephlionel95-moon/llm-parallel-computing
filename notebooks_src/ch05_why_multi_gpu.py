# %% [markdown]
# # Chapter 05 — Why Multi-GPU? The Three Walls
#
# **Hardware:** none required — this chapter is arithmetic. A GPU makes the
# final OOM-probe section experiential instead of predictive, but every wall is
# derivable on a napkin, and this notebook is that napkin, automated.
#
# The three walls: **memory** (model doesn't fit), **compute** (training doesn't
# finish in a lifetime), **communication** (every escape from walls 1–2 pays a
# toll in bytes moved). We derive each from ch03/ch04 formulas.

# %%
import sys
sys.path.insert(0, "../src")

import torch
from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())
DEVICE = torch.device("cuda:0" if ENV.num_gpus >= 1 else "cpu")

# %% [markdown]
# ## 1. Wall 1 — Memory
#
# From ch04's ledger: Adam training costs **16 bytes/param static** (fp32:
# 4+4+8; mixed precision: 2+2+12 — same total), plus activations. A practical
# planning number is 16–20 B/param. Compare that against every GPU you can buy.

# %%
GPUS = {                       # name: (memory GB, peak dense TFLOP/s for 16-bit training)
    "T4 16GB":      (16,   65.0),    # fp16 tensor cores (no bf16 on Turing)
    "A100 40GB":    (40,  312.0),    # bf16
    "A100 80GB":    (80,  312.0),
    "H100 80GB":    (80,  989.0),    # bf16, dense (the 1979 figure is 2:4 sparsity)
}

MODELS = {                     # name: parameter count
    "MiniGPT default": 5.3e6,
    "GPT-2 124M":      124e6,
    "GPT-2 1.5B":      1.5e9,
    "LLaMA-7B":        7e9,
    "LLaMA-70B":       70e9,
    "GPT-3 175B":      175e9,
}

BYTES_PER_PARAM = 16          # Adam static; use 18-20 to budget activations too

print(f"{'model':<18}{'params':>10}{'train GB':>10}  fits on...")
for name, n in MODELS.items():
    gb = n * BYTES_PER_PARAM / 1e9
    fits = [g for g, (mem, _) in GPUS.items() if gb <= mem] or ["NOTHING"]
    print(f"{name:<18}{n/1e9:>9.3f}B{gb:>10.1f}  {', '.join(fits)}")

# %% [markdown]
# **The line to internalize:** a 7 B model — trivially served on a laptop in
# 4-bit — needs ≥112 GB just for params+grads+optimizer. *No single GPU ever
# manufactured trains it naively.* Inference needs 2N bytes; training ~8× that.
#
# ## 2. Wall 2 — Compute
#
# ch03: training FLOPs/token ≈ 6N (2N forward, 4N backward).
# Total compute **C = 6·N·D**. Chinchilla: compute-optimal **D ≈ 20·N**.
# Wall-clock = C / (peak × MFU); well-tuned big runs hit MFU ≈ 30–50%, we plan
# at 40%. Everything below is division, not benchmarking.

# %%
MFU = 0.40
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def training_time_years(n_params: float, tokens: float | None,
                        peak_tflops: float, mfu: float = MFU) -> float:
    tokens = tokens if tokens is not None else 20 * n_params   # Chinchilla
    flops = 6.0 * n_params * tokens
    return flops / (peak_tflops * 1e12 * mfu) / SECONDS_PER_YEAR


print(f"MFU = {MFU:.0%}, D = 20N (Chinchilla) unless stated\n")
print(f"{'model':<26}{'D tokens':>12}{'1x A100 (312TF)':>17}{'1x T4 (65TF)':>15}")
for name, n, d in [("LLaMA-7B", 7e9, None),
                   ("LLaMA-70B", 70e9, None),
                   ("GPT-3 175B (D=300B real)", 175e9, 300e9),
                   ("GPT-3 175B (Chinchilla)", 175e9, None)]:
    dd = d if d is not None else 20 * n
    a100 = training_time_years(n, d, 312.0)
    t4 = training_time_years(n, d, 65.0)
    print(f"{name:<26}{dd/1e9:>10.0f}B{a100:>14.1f} yr{t4:>12.0f} yr")

# %% [markdown]
# GPT-3 on the T4 you get for free: **~4 centuries** (and it doesn't fit anyway
# — the walls compound). Note also C = 6N·20N = 120N²: Chinchilla-optimal cost
# is *quadratic* in model size. 10× the model = 100× the compute.
#
# The only free variable is GPU count:

# %%
def gpus_needed(n_params: float, days: float, peak_tflops: float = 312.0,
                tokens: float | None = None, mfu: float = MFU) -> float:
    tokens = tokens if tokens is not None else 20 * n_params
    flops = 6.0 * n_params * tokens
    return flops / (peak_tflops * 1e12 * mfu * days * 86400)


for n, label in [(7e9, "7B"), (70e9, "70B"), (175e9, "175B (Chinchilla)")]:
    print(f"{label:>18}: {gpus_needed(n, 30):>8,.0f} A100s for a 30-day run "
          f"(communication ignored!)")

# %% [markdown]
# "Communication ignored" is doing heavy lifting there — which is Wall 3.
#
# ## 3. Wall 3 — Communication: the toll gate
#
# Data parallelism must AllReduce N gradient bytes every step; sharding must
# move params/activations every layer. First-order cost of moving a message of
# n bytes over one link:
#
# **T(n) = α + n/β**  — a fixed latency α (paid even for 1 byte) plus a
# bandwidth term. Small messages are latency-bound; large ones bandwidth-bound.
# Chapter 07 measures α and β on real links; here we just see what the model
# *implies* for gradient sync on Kaggle's PCIe vs a DGX's NVLink.

# %%
def comm_time_ms(nbytes: float, alpha_us: float, beta_gbps: float) -> float:
    return alpha_us / 1e3 + nbytes / (beta_gbps * 1e9) * 1e3


links = {"PCIe 3.0 x16 (~13 GB/s eff.)": (30, 13.0),
         "NVLink 3 (~250 GB/s eff.)": (10, 250.0)}
grads = {"MiniGPT 5.3M fp32": 5.3e6 * 4, "GPT-2 124M fp32": 124e6 * 4,
         "7B fp16": 7e9 * 2}

print(f"{'gradient payload':<22}" + "".join(f"{l:>30}" for l in links))
for gname, nbytes in grads.items():
    row = f"{gname:<22}"
    for alpha, beta in links.values():
        row += f"{comm_time_ms(nbytes, alpha, beta):>27.1f} ms"
    print(row)
print("\n(α, β here are ILLUSTRATIVE link parameters for the formula; measure")
print(" your real ones with experiments/ch07 — never trust a spec sheet.)")

# %% [markdown]
# A 7B fp16 gradient sync: ~1 second on PCIe vs ~56 ms on NVLink — for a step
# that might take 1 s of compute, that is 100% vs 5% overhead. **Parallelism
# converts memory/compute problems into communication problems; the exchange
# rate is set by your interconnect.**
#
# ## 4. Amdahl's law: the serial fraction is a ceiling
#
# Fix the problem size. Serial fraction s, parallel fraction 1−s spread over p
# processors: T(p) = s + (1−s)/p, so **S(p) = 1/(s + (1−s)/p) → 1/s**.
# 10% serial ⇒ never more than 10×, no matter the budget.

# %%
import matplotlib.pyplot as plt
import numpy as np

p = np.logspace(0, 4, 100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for s in (0.01, 0.05, 0.10, 0.25):
    ax1.plot(p, 1 / (s + (1 - s) / p), label=f"s = {s:.0%} (cap {1/s:.0f}x)")
    ax2.plot(p, s + p * (1 - s), label=f"s = {s:.0%}")
ax1.set_xscale("log"); ax1.set_xlabel("processors p"); ax1.set_ylabel("speedup")
ax1.set_title("Amdahl: fixed problem — hard ceilings"); ax1.legend()
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("processors p")
ax2.set_title("Gustafson: scaled problem — linear"); ax2.legend()
fig.tight_layout(); plt.show()

s = 0.10
for pp in (10, 100, 1000, 10000):
    print(f"p={pp:>6}: Amdahl speedup {1/(s+(1-s)/pp):6.2f}x   "
          f"Gustafson scaled speedup {s + pp*(1-s):9.1f}x")

# %% [markdown]
# Same s = 10%, opposite verdicts. Amdahl answers "how much faster is THIS
# job"; Gustafson answers "how much MORE work in the same time". LLM training
# is Gustafson's regime — with more GPUs you train bigger models on more tokens
# — bounded by the critical batch size (McCandlish et al. 2018). But *within*
# one step Amdahl rules: the non-overlapped AllReduce tail, the optimizer step,
# data-loader stalls are serial terms. Ch09's overlap engineering attacks s.
#
# ## 5. Scaling arithmetic drills
#
# The muscle this chapter builds is napkin arithmetic. Automated napkin:

# %%
def can_i_train(n_params: float, gpu_mem_gb: float, n_gpus: int = 1,
                bytes_per_param: float = 16.0) -> None:
    need = n_params * bytes_per_param / 1e9
    have = gpu_mem_gb * n_gpus
    verdict = "FITS" if need <= have else "DOES NOT FIT"
    print(f"  {n_params/1e9:.1f}B params x {bytes_per_param:.0f} B = {need:,.0f} GB "
          f"vs {n_gpus} x {gpu_mem_gb:.0f} GB = {have:,.0f} GB -> {verdict}"
          + ("" if need <= have else f" (short {need-have:,.0f} GB — this gap is "
             f"what ZeRO/FSDP/TP must shard away, ch15+)"))


print("Can I train (statically) ...")
can_i_train(7e9, 16, 1)          # 7B on one T4
can_i_train(7e9, 80, 1)          # 7B on one H100
can_i_train(7e9, 80, 2)          # 7B on two H100s — fits IF bytes are sharded
can_i_train(70e9, 80, 16)        # 70B on a 16xH100 pod
can_i_train(175e9, 80, 64)       # GPT-3 on 64 H100s

# %% [markdown]
# Note the fine print on the multi-GPU rows: 2×80 GB only helps if the 16
# B/param are actually *partitioned* across the GPUs. Plain data parallelism
# (ch08–09) REPLICATES them — every rank pays the full 112 GB. Making the
# bytes-per-GPU denominator real is the entire subject of Part IV.
#
# ## 6. Meet the wall in person (GPU) / predict it (CPU)
#
# Grow MiniGPT until the ledger says your device must OOM. On CUDA the probe
# actually allocates (small sizes only here — the full sweep is
# `experiments/ch05/oom_point_finder.py`); on CPU we compute predictions only.

# %%
from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.profiling.memory import model_memory_ledger

widths = [256, 512, 1024, 2048, 4096]
cap_gb = ENV.gpu_mem_gb[0] if ENV.num_gpus else 16.0   # predict for a T4 if CPU
print(f"capacity assumed: {cap_gb:.1f} GB "
      f"({'measured' if ENV.num_gpus else 'hypothetical T4 — CPU mode'})\n")
print(f"{'d_model':>8}{'params':>14}{'static GB':>11}   verdict (static only)")
for d in widths:
    cfg = GPTConfig(vocab_size=8192, block_size=256, n_layer=8,
                    n_head=max(4, d // 64), d_model=d)
    m = MiniGPT(cfg)
    led = model_memory_ledger(m)
    gb = led["total_static_mb"] / 1024
    print(f"{d:>8}{led['num_params']:>14,}{gb:>11.2f}   "
          f"{'fits' if gb < cap_gb * 0.9 else 'expect OOM (before activations!)'}")
    del m

# %% [markdown]
# ## Exercises
#
# 1. (Easy) Redo the section-1 table at 20 B/param and with an 8-bit optimizer
#    (~10 B/param). Which verdicts flip?
# 2. (Easy) How many days to Chinchilla-train a 1B model on one H100 (989
#    TFLOP/s dense) at 40% MFU? At the 20% MFU you might actually get?
# 3. (Medium) Your step: 180 ms parallel compute + 20 ms fixed serial tail.
#    Maximum speedup? How many GPUs to reach 90% of it?
# 4. (Medium) LLaMA-3-8B trained on 15T tokens (~1900 tokens/param). Compute C
#    and A100-years at 40% MFU. Compare with the Chinchilla-optimal 8B run.
# 5. (Hard) Using T(n)=α+n/β with the illustrative PCIe numbers above, find the
#    gradient size where latency and bandwidth terms are equal. What does that
#    say about syncing many small tensors vs one flat bucket (foreshadowing
#    ch09's gradient buckets)?
# 6. (Research) McCandlish et al.'s critical batch size bounds Gustafson's
#    escape. Sketch why gradient noise scale sets it, and what it implies for
#    "just add more data-parallel ranks".
#
# <details><summary>Solution sketches</summary>
#
# 1. At 20 B/param GPT-2 1.5B needs 30 GB — A100-40 now marginal. At 10 B/param
#    7B needs 70 GB: fits on one 80 GB card (statically) — why 8-bit optimizers
#    matter. 2. C=1.2e20; 989e12×0.4=3.96e14 → 3.0e5 s ≈ 3.5 days; at 20% → 7 days.
# 3. s=0.1 → cap 10×; 90% of cap = 9× → 1/(0.1+0.9/p)=9 → p=81.
# 4. C = 6×8e9×1.5e13 = 7.2e23 → 7.2e23/1.248e14 = 5.77e9 s ≈ 183 A100-years
#    (vs Chinchilla D=160B: 7.7e21 ≈ 2 years). Overtraining ~93× the tokens costs
#    ~93× the compute — paid once to serve cheaply forever.
# 5. α = n/β → n = αβ = 30e-6 × 13e9 ≈ 390 KB. Tensors smaller than ~0.4 MB are
#    latency-dominated → fuse them into buckets before reducing (exactly what DDP does).
# 6. Gradient noise scale ≈ batch size at which averaging more samples stops
#    reducing gradient variance usefully; past it, extra ranks buy almost no
#    optimization progress per token — scaled speedup in tokens/s no longer
#    means scaled speedup in loss/s.
# </details>
#
# ## Summary
#
# - Wall 1 memory: 16–20 B/param to train; 7B ⇒ ≥112 GB ⇒ no single GPU.
# - Wall 2 compute: C = 6ND, D≈20N ⇒ C = 120N² — quadratic; GPT-3 on one T4 ≈
#   four centuries, by division.
# - Wall 3 communication: T = α + n/β; the interconnect sets the exchange rate
#   for escaping walls 1–2.
# - Amdahl: serial fraction s caps speedup at 1/s forever (10% ⇒ 10×).
# - Gustafson: grow the problem with the cluster ⇒ linear scaled speedup;
#   LLM training lives here, bounded by critical batch size.
# - Data parallelism replicates bytes; only sharding (Part IV) divides the
#   memory wall by p.

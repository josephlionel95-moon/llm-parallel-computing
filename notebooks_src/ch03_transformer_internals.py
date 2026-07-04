# %% [markdown]
# # Chapter 03 — Transformer Internals: Parameters, Activations & FLOPs
#
# **Hardware:** none required. All accounting runs on CPU; GPU cells add CUDA
# memory counters when available.
#
# The accountant's view: we derive and then *measure* three bills for
# `src/llmdist/models/mini_gpt.py`:
#
# 1. **Parameters**  N = (V+T)d + L(12d² + 13d) + 2d
# 2. **Activations** per layer ≈ s·b·h·(64 + 4·a·s/h) bytes (fp32, no dropout)
# 3. **FLOPs**       ≈ (6N + 12·L·s·d) per token, fwd:bwd = 1:2
#
# and end with the scaling table whose last column is the reason this course
# exists: GPT-3 cannot fit on any single GPU.

# %%
import sys

sys.path.insert(0, "../src")

import torch
import torch.nn.functional as F

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check

env = env_check.detect()
print(env.banner())
device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
torch.manual_seed(0)

# %% [markdown]
# ## 1. Parameters: derive, then diff against the model
#
# Per block: attention 4d² (+4d bias), MLP 8d² (+5d bias), two LayerNorms 4d
# → 12d² + 13d. Outside: (V+T)d embeddings, 2d final LN, head **tied** to
# tok_emb (one storage — ch00's `data_ptr()` would agree). The model carries this
# as `param_formula()`; `num_params()` counts real tensors. They must match.

# %%
cfg = GPTConfig()  # V=8192, T=256, L=4, a=4, d=256
model = MiniGPT(cfg)

d, V, T, L = cfg.d_model, cfg.vocab_size, cfg.block_size, cfg.n_layer
emb = (V + T) * d
blocks = L * (12 * d * d + 9 * d + 4 * d)
final_ln = 2 * d
print(f"embeddings (V+T)d      : {emb:>12,}   ({100 * emb / model.num_params():.1f}% of model!)")
print(f"blocks L(12d^2+13d)    : {blocks:>12,}")
print(f"final LN 2d            : {final_ln:>12,}")
print(f"hand total             : {emb + blocks + final_ln:>12,}")
print(f"model.param_formula()  : {model.param_formula():>12,}")
print(f"model.num_params()     : {model.num_params():>12,}")
assert model.param_formula() == model.num_params() == emb + blocks + final_ln
print("\nFolk rule 12Ld^2 =", f"{12 * L * d * d:,}",
      "- small models are embedding-heavy; at GPT-3 scale embeddings are noise.")

# %%
# The tied head, verified with ch00's tool of truth:
print(f"head.weight is tok_emb.weight   : {model.head.weight is model.tok_emb.weight}")
print(f"same data_ptr                   : "
      f"{model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()}")
print("-> counted ONCE in params, but its matmul appears TWICE in FLOPs (emb lookup")
print("   is free; the d->V head matmul is real work).")

# %% [markdown]
# ## 2. Activations: what backward keeps, weighed exactly
#
# Ch02 showed *why* tensors are saved (dW = Xᵀ G needs X). Here we weigh them two
# ways, no torchinfo:
#
# - **estimator**: forward hooks record every module's output bytes;
# - **ground truth**: `saved_tensors_hooks` sees exactly what autograd stores
#   (dedup by storage — views share bytes).
#
# Formula to beat, fp32 / dropout=0, per layer: **s·b·h·(64 + 4·a·s/h)** bytes,
# plus the tail everyone forgets: logits + log-softmax ≈ 8·s·b·V bytes — which
# for MiniGPT (V = 32h) outweighs all blocks combined.

# %%
def measure_saved_bytes(model: MiniGPT, idx: torch.Tensor) -> float:
    """Total bytes autograd saves for backward, deduped by storage."""
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        base = t if t._base is None else t._base
        seen[id(base)] = base.numel() * base.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = model(idx, targets=idx)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return sum(seen.values())

def formula_bytes(b: int, s: int, cfg: GPTConfig) -> float:
    h, a, L, V = cfg.d_model, cfg.n_head, cfg.n_layer, cfg.vocab_size
    per_layer = s * b * h * (64 + 4 * a * s / h)
    logits_tail = 8 * s * b * V
    return L * per_layer + logits_tail

b, s = 8, 256
idx = torch.randint(0, cfg.vocab_size, (b, s))
meas = measure_saved_bytes(model, idx)
pred = formula_bytes(b, s, cfg)
print(f"batch={b}, seq={s}")
print(f"formula  : {pred / 2**20:8.1f} MiB  (blocks {cfg.n_layer * s * b * cfg.d_model * (64 + 4 * cfg.n_head * s / cfg.d_model) / 2**20:.1f} + logits tail {8 * s * b * cfg.vocab_size / 2**20:.1f})")
print(f"measured : {meas / 2**20:8.1f} MiB")
print(f"ratio    : {meas / pred:8.2f}   (formula ignores LN stats, emb output, etc.)")
print(f"\nparams are only {model.num_params() * 4 / 2**20:.1f} MiB -> activations are "
      f"~{meas / (model.num_params() * 4):.0f}x params at this batch/seq!")

# %%
# Sweep batch and sequence: activations scale with DATA SHAPE, params don't.
print(f"{'b':>3} {'s':>5} {'formula MiB':>12} {'measured MiB':>13} {'ratio':>7}")
for b_ in [1, 2, 4, 8]:
    idx_ = torch.randint(0, cfg.vocab_size, (b_, 128))
    m_ = measure_saved_bytes(model, idx_)
    p_ = formula_bytes(b_, 128, cfg)
    print(f"{b_:>3} {128:>5} {p_ / 2**20:>12.1f} {m_ / 2**20:>13.1f} {m_ / p_:>7.2f}")
for s_ in [64, 128, 256]:
    idx_ = torch.randint(0, cfg.vocab_size, (4, s_))
    m_ = measure_saved_bytes(model, idx_)
    p_ = formula_bytes(4, s_, cfg)
    print(f"{4:>3} {s_:>5} {p_ / 2**20:>12.1f} {m_ / 2**20:>13.1f} {m_ / p_:>7.2f}")
print("\nLinear in b; SUPER-linear in s (the a·s²·b softmax term) -> long context")
print("OOMs are quadratic, and FlashAttention/sequence-parallel exist to kill that term.")

# %% [markdown]
# ## 3. FLOPs: 6N per token, from matmul dims
#
# One token through W (d_in × d_out) = 2·d_in·d_out FLOPs = 2 × params(W).
# Sum over the model → forward ≈ 2N. Backward does two matmuls per forward one
# (dX = G Wᵀ, dW = Xᵀ G) → 4N. Total **6N**, plus attention's weightless matmuls
# q@kᵀ and att@v: 12·s·d per layer-token with backward.

# %%
N_ne = model.num_params(non_embedding=True)
attn_term = cfg.n_layer * 12 * cfg.block_size * cfg.d_model
print(f"6N (N non-embedding = {N_ne:,})        : {6 * N_ne / 1e6:8.1f} MFLOPs/token")
print(f"attention 12*L*T*d                    : {attn_term / 1e6:8.1f} MFLOPs/token "
      f"({100 * attn_term / (6 * N_ne):.0f}% extra; equals 6N only at s = 6d = {6 * d})")
print(f"model.flops_per_token()               : {model.flops_per_token() / 1e6:8.1f} MFLOPs/token")

# %%
# Measure achieved FLOP/s and the fwd:(fwd+bwd) time ratio (theory: 1:3).
from llmdist.profiling.timers import timeit

model_d = model.to(device)
idx_d = idx.to(device)

def fwd():
    with torch.no_grad():
        model_d(idx_d, targets=idx_d)

def fwd_bwd():
    _, loss = model_d(idx_d, targets=idx_d)
    loss.backward()
    model_d.zero_grad(set_to_none=True)

t_f = timeit(fwd, iters=5, warmup=2, device=device)
t_fb = timeit(fwd_bwd, iters=5, warmup=2, device=device)
tokens = idx_d.numel()
step_flops = model_d.flops_per_token() * tokens
print(f"forward        : {t_f['mean_ms']:8.1f} ms")
print(f"forward+backward: {t_fb['mean_ms']:8.1f} ms   ratio {t_fb['mean_ms'] / t_f['mean_ms']:.2f} "
      f"(theory 3.0 when compute-bound)")
print(f"achieved       : {step_flops / (t_fb['mean_ms'] * 1e-3) / 1e9:8.1f} GFLOP/s")
if env.num_gpus >= 1:
    print("On a T4, compare against 8.1 TFLOPS fp32 peak -> MFU. MiniGPT is small,")
    print("so expect single-digit % (ch01: launch/memory-bound); try d_model=1024.")
else:
    print("(CPU: absolute rate is modest; the lesson is the ratio and 6N scaling.)")

# %% [markdown]
# ## 4. The scaling table — and the punchline
#
# Now apply the same four formulas to real configs. Every number below is
# *derived*, not benchmarked — that is the accountant's power.

# %%
def account(name, L, d_, a_, V_, T_, n_known=None, bytes_w=2, akv=None):
    N = n_known if n_known else 12 * L * d_ * d_ + (V_ + T_) * d_
    akv = akv or a_
    kv = 2 * L * T_ * (akv * d_ // a_) * 2  # fp16 KV cache, b=1
    return {
        "model": name, "N": N,
        "weights_gb": N * bytes_w / 2**30,
        "train_gb": N * 16 / 2**30,          # mixed-precision Adam rule (ch04)
        "flops_tok": 6 * N,
        "kv_gb": kv / 2**30,
    }

rows = [
    account("MiniGPT", 4, 256, 4, 8192, 256, n_known=model.num_params(), bytes_w=4),
    account("GPT-2 small", 12, 768, 12, 50257, 1024),
    account("GPT-3 175B", 96, 12288, 96, 50257, 2048, n_known=175_000_000_000),
    account("Llama-2 70B", 80, 8192, 64, 32000, 4096, n_known=70_000_000_000, akv=8),
]
print(f"{'model':<13} {'N':>8} {'weights':>9} {'train state':>12} "
      f"{'FLOPs/tok':>10} {'KV@T,b=1':>9}")
for r in rows:
    n_str = f"{r['N'] / 1e9:.1f}B" if r["N"] > 1e9 else f"{r['N'] / 1e6:.1f}M"
    print(f"{r['model']:<13} {n_str:>8} {r['weights_gb']:>8.1f}G {r['train_gb']:>11.1f}G "
          f"{r['flops_tok'] / 1e9:>9.2f}G {r['kv_gb']:>8.2f}G")

print("\nPUNCHLINE: the biggest single GPU holds ~0.19 TB. GPT-3's fp16 weights")
print("alone are 0.35 TB; its training state ~2.8 TB (>= 35x A100-80GB just to")
print("EXIST); its 300B-token run costs 3.15e23 FLOPs (~64 years on one A100 at")
print("50% MFU). Multi-GPU training is not an optimization - it is the only way.")
print("Parts II-IV of this course are about paying these three bills across GPUs.")

# %% [markdown]
# ## Exercises
#
# 1. **(Easy)** Set `bias=False` in the config. Predict the exact change in
#    `num_params()` before running (answer: L·9d), then verify.
# 2. **(Medium)** Add a "your dream model" row to the table: L=32, d=4096, a=32,
#    V=128k, T=8192. Can you train it (Adam, mixed precision) on 8× A100-80GB
#    *before* activations?
# 3. **(Medium)** In the sweep of cell 2, compute the predicted s-doubling factor
#    (2s(64+8as/h))/(s(64+4as/h)) for s: 128→256 and compare with the measured one.
# 4. **(Hard)** Modify `measure_saved_bytes` to *attribute* bytes per module
#    (hint: record which module is running via forward pre/post hooks and tag
#    packs). Rank MiniGPT's modules by saved bytes; verify the GELU input (16sbh)
#    and logits dominate.
#
# <details><summary>Solutions</summary>
#
# 1. Biases removed: qkv 3d + attn.proj d + fc 4d + mlp.proj d = 9d per block,
#    ×L = 4·9·256 = 9,216 fewer; LayerNorms keep theirs (elementwise_affine).
# 2. N ≈ 12·32·4096² + (128k+8k)·4096 ≈ 6.44B + 0.56B ≈ 7.0B → 16 B/param ≈ 112 GB
#    train state: fits in 8×80 = 640 GB with room — activations and the 8sbV
#    logits tail (V=128k!) become the real question; hence ZeRO + fused losses.
# 3. With a=4, h=256: as/h at s=128 is 2 → constant 64+8=72; at s=256 → 64+16=80.
#    Factor = 2·80/72 ≈ 2.22×; the measured blocks term should match closely, but
#    the total includes the logits tail (linear in s), pulling the ratio toward 2.
# 4. Sketch: a module-stack maintained by `register_forward_pre_hook`/
#    `register_forward_hook`; `pack` tags bytes with the stack top. Expect order:
#    loss/logits ≫ mlp (fc out + gelu out = 32sbh) > attention linears > LNs.
#
# </details>
#
# ## Summary
#
# - N = (V+T)d + L(12d²+13d) + 2d — exact; verified against the live model.
# - Activations ≈ L·sbh(64+4as/h) + 8sbV (fp32 MiniGPT): scale with data shape,
#   linear in b, quadratic in s; they, not params, set your max batch.
# - FLOPs ≈ (6N + 12Lsd)/token; fwd:bwd = 1:2 because every matmul spawns two.
# - KV = 2·L·s·d_kv·bytes — GQA divides d_kv; decoding is memory-bound.
# - GPT-3 fails to fit by 3 independent arguments (weights, state, time):
#   distributed training is mandatory, and now you can prove it with arithmetic.

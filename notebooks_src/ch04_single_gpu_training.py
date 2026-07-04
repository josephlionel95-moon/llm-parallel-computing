# %% [markdown]
# # Chapter 04 — Single-GPU Training Anatomy
#
# **Hardware:** runs anywhere. A CUDA GPU (Colab/Kaggle T4) gives real allocator
# measurements; on CPU we fall back to *byte accounting* — walking the actual
# tensors (`p.grad`, `opt.state`) and summing `numel() * element_size()` — which
# teaches the same ledger with exact (if less dramatic) numbers.
#
# Goal: narrate one training step byte by byte — forward, backward,
# `optimizer.step()` (Adam state born lazily!), `zero_grad(set_to_none=True)` —
# then verify the 16 bytes/param ledger against measurement.

# %%
import sys
sys.path.insert(0, "../src")

import torch
from llmdist.utils.env_check import detect

ENV = detect()
print(ENV.banner())
DEVICE = torch.device("cuda:0" if ENV.num_gpus >= 1 else "cpu")
print(f"\nrunning this notebook on: {DEVICE}")

# %% [markdown]
# ## 1. The five memory accounts
#
# Every byte on the device belongs to one of: **params**, **grads**,
# **optimizer state**, **activations**, **workspace/cache**. The first three are
# pure arithmetic (this notebook), activations scale with batch×sequence (ch03),
# and workspace is the allocator's business (ch02).
#
# We need one instrument the CUDA allocator gives us for free but CPU does not:
# "how many bytes do my gradient / optimizer tensors hold right now?" So we
# write the accountant explicitly — it works on both devices and, on CPU, it IS
# our memory measurement.

# %%
from llmdist.models.mini_gpt import MiniGPT, GPTConfig
from llmdist.profiling.memory import mb, snapshot, track, model_memory_ledger


def account_bytes(model: torch.nn.Module, opt: torch.optim.Optimizer) -> dict[str, float]:
    """Walk live tensors and sum bytes per ledger account (works on CPU and GPU)."""
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    grads = sum(p.grad.numel() * p.grad.element_size()
                for p in model.parameters() if p.grad is not None)
    opt_state = sum(t.numel() * t.element_size()
                    for st in opt.state.values()
                    for t in st.values() if torch.is_tensor(t))
    return {"params_mb": mb(params), "grads_mb": mb(grads),
            "opt_state_mb": mb(opt_state),
            "static_total_mb": mb(params + grads + opt_state)}


cfg = GPTConfig(vocab_size=8192, block_size=128, n_layer=4, n_head=4, d_model=256)
model = MiniGPT(cfg).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
N = model.num_params()
print(f"MiniGPT: {N:,} params ({mb(N * 4):.1f} MB in fp32)")
print("formula check:", model.param_formula() == N)

# %% [markdown]
# ## 2. One step under the microscope
#
# Watch the accountant across the four phases. Predictions before we run:
#
# - after **forward**: grads = 0, opt state = 0 (activations exist but are owned
#   by autograd — the accountant can't see them, the CUDA allocator can)
# - after **backward**: grads = params bytes (one fp32 grad per param)
# - after **step 0**: opt state = 2 × params bytes — **Adam's m and v are
#   created lazily inside the first `step()` call**, not in the constructor
# - after **zero_grad(set_to_none=True)**: grads = 0 again — freed, not zeroed

# %%
B, T = 8, cfg.block_size
x = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
y = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)


def show(phase: str) -> None:
    a = account_bytes(model, opt)
    line = (f"{phase:<28} params {a['params_mb']:7.1f} MB | grads {a['grads_mb']:7.1f} MB"
            f" | opt {a['opt_state_mb']:7.1f} MB")
    if DEVICE.type == "cuda":
        line += f" | cuda alloc {snapshot(DEVICE).allocated_mb:8.1f} MB"
    print(line)


for step in range(3):
    print(f"--- step {step} ---")
    logits, loss = model(x, y)
    show("after forward")
    loss.backward()
    show("after backward")
    opt.step()
    show("after optimizer.step()")      # watch opt jump 0 -> 2*params at step 0
    opt.zero_grad(set_to_none=True)
    show("after zero_grad")

# %% [markdown]
# Read the `opt` column: 0 MB during all of step 0's forward/backward, then it
# jumps to exactly 2× the params bytes inside the first `step()` and never moves
# again. This is the classic "trained one step, OOMed on the next" trap: your
# *budget* must include m and v from the start even though the *allocation*
# happens late.
#
# On CUDA, the `cuda alloc` column also shows the forward bump (activations,
# which the tensor-walking accountant cannot see) deflating during backward.

# %%
# The ledger PREDICTION (arithmetic) vs the accountant's MEASUREMENT (tensors).
# We are between steps (after zero_grad), so grads are None: the measured
# floor should equal params + m + v = 12 B/param; the full 16 B/param appears
# only while grads are alive (between backward and zero_grad).
pred = model_memory_ledger(model, optimizer_bytes_per_param=8, dtype_bytes=4)
meas = account_bytes(model, opt)
print(f"predicted static (16 B/param, grads alive): {pred['total_static_mb']:8.1f} MB "
      f"(= {pred['num_params']:,} params x 16 B)")
print(f"measured now: params {meas['params_mb']:.1f} + grads {meas['grads_mb']:.1f} "
      f"(freed by zero_grad!) + opt {meas['opt_state_mb']:.1f} "
      f"= {meas['static_total_mb']:.1f} MB")
print(f"\nfloor prediction (12 B/param between steps): "
      f"{12 * pred['num_params'] / 2**20:8.1f} MB")
print(f"floor measured  (params + m + v):           "
      f"{meas['params_mb'] + meas['opt_state_mb']:8.1f} MB")

# %% [markdown]
# ## 3. Where the peak lives: early in backward
#
# During backward, autograd frees each saved activation as its consumer runs,
# while allocating gradient tensors. The global peak is therefore usually just
# after backward *starts*: nearly all activations alive + first grads allocated.
#
# On CUDA we see this with `llmdist.profiling.memory.track`; on CPU the cell
# still runs but the CUDA columns read 0 — the narrative is the point.

# %%
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with track("forward", DEVICE):
        logits, loss = model(x, y)
    with track("backward", DEVICE):
        loss.backward()
    with track("optimizer.step", DEVICE):
        opt.step()
    with track("zero_grad", DEVICE):
        opt.zero_grad(set_to_none=True)
else:
    print("CPU mode: CUDA allocator stats unavailable; the byte-accounting table")
    print("above already showed the grads/opt dynamics. On a T4, run this cell to")
    print("see forward's +Δalloc (activations) turn into backward's staircase.")

# %% [markdown]
# ## 4. Activations scale with batch size — measure them as a residual
#
# activations_at_peak ≈ peak − (params + grads + opt state). Double the batch,
# and the residual should roughly double while the static part stays fixed.
# On CPU we can't measure the peak, so we instead *predict* the dominant saved
# tensors per block: two (B, H, T, T) attention matrices + the 4d-wide MLP hidden.

# %%
def predicted_activation_hotspots_mb(cfg: GPTConfig, B: int, T: int) -> float:
    """Rough lower bound: the few biggest saved tensors per block, fp32."""
    att = 2 * B * cfg.n_head * T * T           # pre-softmax scores + softmax out
    mlp = B * T * 4 * cfg.d_model              # gelu input
    resid = 2 * B * T * cfg.d_model            # block outputs kept for residual adds
    return mb((att + mlp + resid) * 4 * cfg.n_layer)


for B_try in (2, 4, 8, 16):
    xb = torch.randint(0, cfg.vocab_size, (B_try, T), device=DEVICE)
    yb = torch.randint(0, cfg.vocab_size, (B_try, T), device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(DEVICE)
        base = snapshot(DEVICE).allocated_mb
        _, loss = model(xb, yb)
        loss.backward()
        peak = snapshot(DEVICE).max_allocated_mb
        opt.zero_grad(set_to_none=True)
        print(f"B={B_try:3d}: measured residual ≈ {peak - base:8.1f} MB | "
              f"predicted hotspots ≥ {predicted_activation_hotspots_mb(cfg, B_try, T):8.1f} MB")
    else:
        print(f"B={B_try:3d}: predicted activation hotspots ≥ "
              f"{predicted_activation_hotspots_mb(cfg, B_try, T):8.1f} MB (linear in B)")

# %% [markdown]
# ## 5. Mixed precision and loss scaling
#
# The ledger says mixed precision does NOT shrink the static accounts:
# 2 (fp16 p) + 2 (fp16 g) + 4 (fp32 master) + 4 (m) + 4 (v) = 16 B/param — the
# savings live in activations and speed. What fp16 *adds* is a correctness
# problem: gradients below 2^-24 ≈ 6e-8 underflow to zero silently.
#
# Demonstrate the underflow with pure tensor arithmetic (no GPU needed):

# %%
g = torch.tensor([1e-3, 1e-5, 1e-7, 1e-8, 1e-9])
print("fp32 gradient:        ", g.tolist())
print("cast to fp16:         ", g.half().tolist(), "  <- small ones vanish")
S = 2.0 ** 16
print(f"scaled by 2^16, fp16: ", (g * S).half().tolist())
print("unscaled back in fp32:", ((g * S).half().float() / S).tolist())
print("\ncast to bf16:         ", g.bfloat16().tolist(), " <- range survives")
print("bf16 keeps fp32's 8 exponent bits -> no underflow -> no GradScaler needed;")
print("it pays with mantissa: bf16(1.001) =", torch.tensor(1.001).bfloat16().item())

# %% [markdown]
# The `GradScaler` loop (runs on CUDA; shown for reference on CPU):
#
# ```python
# scaler = torch.cuda.amp.GradScaler()
# with torch.autocast("cuda", dtype=torch.float16):
#     _, loss = model(x, y)
# scaler.scale(loss).backward()   # grads are S times too big
# scaler.unscale_(opt)            # divide by S (do this BEFORE grad clipping!)
# scaler.step(opt)                # skips the step if any grad is inf/nan
# scaler.update()                 # halve S on skip, slowly grow it on success
# ```

# %%
if DEVICE.type == "cuda":
    scaler = torch.cuda.amp.GradScaler()
    amp_model = MiniGPT(cfg).to(DEVICE)
    amp_opt = torch.optim.Adam(amp_model.parameters(), lr=3e-4)
    for step in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = amp_model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(amp_opt)
        scaler.step(amp_opt)
        scaler.update()
        amp_opt.zero_grad(set_to_none=True)
        print(f"step {step}: loss {loss.item():.3f}  scale {scaler.get_scale():.0f}")
    del amp_model, amp_opt
else:
    print("CPU mode: GradScaler is CUDA-only; the underflow demo above is the physics.")

# %% [markdown]
# ## 6. Throughput vs batch size
#
# Bigger batches amortize per-launch overhead and fill more SMs — until the
# device saturates and tokens/s plateaus. Find the knee; don't assume it.

# %%
from llmdist.profiling.timers import timeit

model_t = MiniGPT(cfg).to(DEVICE)
opt_t = torch.optim.Adam(model_t.parameters(), lr=3e-4)
batches = (1, 2, 4, 8) if DEVICE.type == "cpu" else (1, 2, 4, 8, 16, 32)
results = []
for B_try in batches:
    xb = torch.randint(0, cfg.vocab_size, (B_try, T), device=DEVICE)
    yb = torch.randint(0, cfg.vocab_size, (B_try, T), device=DEVICE)

    def one_step() -> None:
        _, loss = model_t(xb, yb)
        loss.backward()
        opt_t.step()
        opt_t.zero_grad(set_to_none=True)

    t = timeit(one_step, iters=5, warmup=2, device=DEVICE)
    tok_s = B_try * T / (t["mean_ms"] / 1e3)
    results.append((B_try, t["mean_ms"], tok_s))
    print(f"B={B_try:3d}: {t['mean_ms']:8.1f} ms/step  {tok_s:12,.0f} tokens/s")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot([r[0] for r in results], [r[2] for r in results], "o-")
ax.set_xlabel("batch size"); ax.set_ylabel("tokens/s"); ax.set_xscale("log", base=2)
ax.set_title(f"MiniGPT throughput vs batch ({DEVICE})")
fig.tight_layout(); plt.show()

# %% [markdown]
# ## 7. torch.profiler: the step on film
#
# `schedule(wait=1, warmup=1, active=2)` skips step 0 (lazy Adam init, autotune
# noise) and records two steady-state steps. Open the exported JSON at
# `chrome://tracing` or https://ui.perfetto.dev.

# %%
from torch.profiler import ProfilerActivity, profile, schedule

activities = [ProfilerActivity.CPU]
if DEVICE.type == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities,
             schedule=schedule(wait=1, warmup=1, active=2),
             profile_memory=True, record_shapes=True) as prof:
    for _ in range(4):
        _, loss = model_t(x, y)
        loss.backward()
        opt_t.step()
        opt_t.zero_grad(set_to_none=True)
        prof.step()

sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
try:
    prof.export_chrome_trace("../profiling/ch04_train_step.json")
    print("trace written to profiling/ch04_train_step.json")
except Exception as e:  # directory may not exist in some environments
    print(f"trace export skipped: {e}")

# %% [markdown]
# Reading the table: matmuls (`aten::addmm`, `aten::mm`, `aten::bmm`) should top
# the list. If copies or pointwise ops dominate, you are memory- or launch-bound
# (ch01's roofline). In the chrome trace, look for GAPS in the GPU stream —
# GPU idle while Python dispatches is the launch-bound signature of small models.
#
# ## Exercises
#
# 1. (Easy) Change the optimizer to `torch.optim.SGD(momentum=0.9)` in section 2
#    and predict the `opt` column before running. Then `momentum=0` — predict again.
# 2. (Easy) Set `set_to_none=False` in `zero_grad` and re-read the grads column.
#    What changed, and what stayed allocated between steps?
# 3. (Medium) Use `account_bytes` inside a full-backward hook on each block to
#    print the grads account mid-backward. Confirm grads grow monotonically
#    while (on CUDA) total allocated falls.
# 4. (Medium) In section 5, feed the scaler `init_scale=2**24` and log
#    `scaler.get_scale()` for 20 steps. Explain the falling staircase.
# 5. (Hard) Predict the measured residual in section 4 more tightly by adding
#    the remaining saved tensors (layernorm inputs, qkv splits). How close can
#    you get to the CUDA measurement?
#
# <details><summary>Solution sketches</summary>
#
# 1. momentum=0.9: opt column = 1× params bytes (one buffer). momentum=0: stays
#    exactly 0.0 MB forever — SGD without momentum keeps no state.
# 2. grads column stays at params-bytes even "between" steps: zeros are stored,
#    not freed. Floor rises by 4 B/param vs set_to_none=True.
# 3. Hook order is reverse block order; grads_mb increases with each hook call;
#    on CUDA, allocated falls because freed activations outweigh new grads.
# 4. Any |g| > 65504/2^24 ≈ 3.9e-3 overflows in fp16 → step skipped, S halved;
#    log2(S) falls by 1 per skip until S·max|g| < 65504, then grows by ×2 every
#    growth_interval (default 2000) successful steps.
# 5. Adding qkv (3·B·T·d), the two LN inputs (2·B·T·d), attention's y
#    (B·T·d) and softmax input typically closes most of the gap; the last few MB
#    are allocator block rounding (512 B granularity) and cuBLAS workspace.
# </details>
#
# ## Summary
#
# - Four phases; memory rises in forward, staircases down in backward, and the
#   floor JUMPS at step 1 when Adam lazily allocates m and v (8 B/param).
# - fp32 Adam static ledger: 4+4+4+4 = 16 B/param. Mixed precision: 2+2+12 = 16.
# - fp16 underflows below 2^-24 → GradScaler (scale, unscale, skip, adapt);
#   bf16 has fp32's exponent range and needs none of it.
# - zero_grad(set_to_none=True) frees grads instead of zeroing them.
# - Predict with model_memory_ledger, verify with track/max_memory_allocated
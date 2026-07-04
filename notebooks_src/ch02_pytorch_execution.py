# %% [markdown]
# # Chapter 02 — PyTorch Execution: Autograd, Hooks & Checkpointing
#
# **Hardware:** none required — autograd is identical on CPU and GPU. The one
# CUDA-allocator cell detects your environment and skips gracefully.
#
# Plan:
# 1. Walk the `grad_fn` graph that forward builds.
# 2. Leaves, `.grad` accumulation, and why accumulation is a *feature*.
# 3. Hooks — including the exact mechanism DDP uses to trigger communication.
# 4. Custom `autograd.Function`s: `PrintGrad` (identity fwd, side-effect bwd)
#    and a `ManualLinear` verified by gradcheck.
# 5. `torch.utils.checkpoint`: recompute-in-backward, measured.
# 6. The CUDA caching allocator: allocated vs reserved.

# %%
import sys

sys.path.insert(0, "../src")

import torch
import torch.nn as nn

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check

env = env_check.detect()
print(env.banner())
torch.manual_seed(0)

# %% [markdown]
# ## 1. The graph forward builds
#
# Every op on a `requires_grad` tensor records a node: the *backward closure* plus
# references to tensors it will need. `t.grad_fn` is the node that produced `t`;
# `node.next_functions` point toward the inputs. Let's walk a tiny graph, then a
# real MiniGPT loss.

# %%
a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = (a * b).sum()

def walk(fn, depth=0, seen=None, max_depth=12):
    if fn is None or depth > max_depth:
        return
    seen = seen if seen is not None else set()
    tag = " (revisited)" if id(fn) in seen else ""
    print("    " * depth + type(fn).__name__ + tag)
    if id(fn) in seen:
        return
    seen.add(id(fn))
    for child, _ in fn.next_functions:
        walk(child, depth + 1, seen, max_depth)

walk(z.grad_fn)
print("\nLeaves end in AccumulateGrad — the node whose whole job is `.grad += g`.")

# %%
# The same walker on a 2-layer MiniGPT loss (depth-limited: the full graph has
# hundreds of nodes). Note the diamond shapes from residual connections: nodes
# get revisited because the graph is a DAG, not a tree.
cfg = GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=2, d_model=32)
model = MiniGPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, 16))
logits, loss = model(idx, targets=idx)
walk(loss.grad_fn, max_depth=6)
print("\nFull-graph version: experiments/ch02/gradfn_tree.py")

# %% [markdown]
# ## 2. `.grad` accumulates — by design
#
# `AccumulateGrad` performs `p.grad += g`. Hence (a) you must zero between steps,
# and (b) *gradient accumulation over micro-batches needs zero extra machinery* —
# linearity of ∇ means the sum of micro-batch gradients IS the big-batch gradient.

# %%
lin = nn.Linear(4, 1, bias=False)
x = torch.ones(1, 4)

lin(x).sum().backward()
g1 = lin.weight.grad.clone()
lin(x).sum().backward()          # no zero_grad in between!
print(f"after 2 backwards, grad/g1 = {(lin.weight.grad / g1).mean().item():.1f}  (accumulated)")

lin.weight.grad = None           # what zero_grad(set_to_none=True) does
big = torch.randn(8, 4)
lin(big).mean().backward()
g_big = lin.weight.grad.clone()

lin.weight.grad = None
for k in range(4):               # 4 micro-batches of 2 = same 8 rows
    micro = big[2 * k: 2 * k + 2]
    (lin(micro).mean() / 4).backward()   # /4: cross-entropy-style 'mean' rescaling
print(f"micro-batched grad == big-batch grad: "
      f"{torch.allclose(lin.weight.grad, g_big, atol=1e-6)}")

# %% [markdown]
# ## 3. Hooks: the mechanism DDP is built on
#
# A hook is a callback on a graph event. The one that matters most for this
# course: **per-parameter "gradient is ready" hooks** — DDP attaches these to
# know when a bucket of gradients can start AllReducing *while backward is still
# running*. We record the ARRIVAL ORDER of gradients in MiniGPT: it is roughly
# the reverse of forward order, which is exactly why overlap works.

# %%
arrival: list[str] = []

def make_hook(name: str):
    def hook(*_) -> None:
        arrival.append(name)
    return hook

handles = []
for name, p in model.named_parameters():
    if hasattr(p, "register_post_accumulate_grad_hook"):   # torch >= 2.1
        handles.append(p.register_post_accumulate_grad_hook(make_hook(name)))
    else:
        handles.append(p.register_hook(make_hook(name)))

model.zero_grad()
_, loss = model(idx, targets=idx)
loss.backward()
for h in handles:
    h.remove()

print("first 6 gradients to arrive:", arrival[:6])
print("last 4 gradients to arrive :", arrival[-4:])
print("\n-> tail of the network first, embeddings last: DDP starts communicating")
print("   the tail while the head still computes. (Note tok_emb is tied to the")
print("   head, so it accumulates only after BOTH uses contribute - it lands late.)")

# %% [markdown]
# ## 4. Custom autograd.Function
#
# `PrintGrad`: identity in forward, side effect in backward. This shape — do
# nothing forward, do something (print / clip / **AllReduce**) backward — is the
# skeleton of every tensor-parallel layer in ch10.

# %%
class PrintGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, label: str) -> torch.Tensor:
        ctx.label = label
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        print(f"    backward passing '{ctx.label}': |g| = {grad_out.norm():.4f}")
        return grad_out, None  # one return per forward input; None for the str

x = torch.randn(2, 3, requires_grad=True)
y = PrintGrad.apply(x, "checkpoint-A")
(y * y).sum().backward()
print("PrintGrad changed nothing numerically; it merely observed the pass.")

# %%
# ManualLinear: we supply the backward formulas ourselves.
#   Y = X W^T + b;  dX = G W;  dW = G^T X;  db = G.sum(0)
# Backward = TWO matmuls per forward's one -> the microscopic origin of
# "backward costs 2x forward" (and of ch03's 6N FLOPs/token).
class ManualLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)   # <- THIS is activation memory
        return x @ weight.t() + bias

    @staticmethod
    def backward(ctx, g):
        x, weight = ctx.saved_tensors
        return g @ weight, g.t() @ x, g.sum(0)

xd = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
wd = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
bd = torch.randn(3, dtype=torch.double, requires_grad=True)
ok = torch.autograd.gradcheck(ManualLinear.apply, (xd, wd, bd))
print(f"gradcheck(ManualLinear) -> {ok}")
print("Note what backward NEEDED: the saved input x. Saving inputs for dW is why")
print("activations must be kept (or recomputed) - quantified in ch03.")

# %% [markdown]
# ## 5. Checkpointing: recompute instead of save
#
# `torch.utils.checkpoint.checkpoint(fn, x)` saves only `x` (+ RNG state), then
# re-runs `fn` during backward. Cost: one extra forward ≈ +33 % FLOPs (fwd:bwd =
# 2:4, recompute adds 2 → 8/6). Benefit: intermediate activations of `fn` are
# never stored. We *weigh* the saved tensors with `saved_tensors_hooks` — a
# CPU-safe scale (no CUDA counters needed).

# %%
import time

from torch.utils.checkpoint import checkpoint

cfg2 = GPTConfig(vocab_size=256, block_size=128, n_layer=4, n_head=4, d_model=128)
m2 = MiniGPT(cfg2)
idx2 = torch.randint(0, cfg2.vocab_size, (4, 128))

def forward_loss(use_ckpt: bool) -> torch.Tensor:
    B, T = idx2.shape
    pos = torch.arange(T)
    x = m2.drop(m2.tok_emb(idx2) + m2.pos_emb(pos))
    for blk in m2.blocks:
        x = checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)
    logits = m2.head(m2.ln_f(x))
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), idx2.view(-1))

def measure(use_ckpt: bool):
    saved = {}
    def pack(t):
        saved[id(t) if t._base is None else id(t._base)] = t.numel() * t.element_size()
        return t
    t0 = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = forward_loss(use_ckpt)
    loss.backward()
    m2.zero_grad(set_to_none=True)
    return sum(saved.values()) / 2**20, (time.perf_counter() - t0) * 1e3

base_mb, base_ms = measure(False)
ck_mb, ck_ms = measure(True)
print(f"{'':<16}{'saved-for-bwd MB':>18}{'fwd+bwd ms':>12}")
print(f"{'baseline':<16}{base_mb:>18.1f}{base_ms:>12.1f}")
print(f"{'checkpointed':<16}{ck_mb:>18.1f}{ck_ms:>12.1f}")
print(f"memory ratio {base_mb / ck_mb:.1f}x saved | time ratio {ck_ms / base_ms:.2f}x "
      f"(theory ~1.33x for the recomputed part)")

# %% [markdown]
# ## 6. The CUDA caching allocator
#
# `memory_allocated` = bytes in live tensors; `memory_reserved` = bytes PyTorch
# caches from CUDA (freeing a tensor returns its block to the *cache*, not to the
# GPU). nvidia-smi shows reserved + context — always scarier than your tensors.

# %%
if env.num_gpus >= 1:
    from llmdist.profiling.memory import snapshot

    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print(f"start   : {snapshot(dev)}")
    big_t = torch.randn(64 * 2**20 // 4, device=dev)      # 64 MiB
    print(f"alloc 64: {snapshot(dev)}")
    del big_t
    print(f"del     : {snapshot(dev)}   <- allocated drops; reserved does NOT (cache)")
    torch.cuda.empty_cache()
    print(f"empty   : {snapshot(dev)}   <- only empty_cache() lowers reserved")
else:
    print("[skip] no GPU: the caching allocator only exists for CUDA memory.")
    print("       Remember the invariant: reserved >= allocated; nvidia-smi shows")
    print("       reserved + CUDA context; fragmentation = free-but-unusable gaps.")

# %% [markdown]
# ## Exercises
#
# 1. **(Easy)** Add `c` to cell 1's graph: `z = (a*b + c).sum()`. Predict the tree
#    before running. Which node consumes `c`'s gradient?
# 2. **(Medium)** Re-run the arrival-order cell with `n_layer=6`. Compute the
#    Spearman correlation between arrival rank and (reversed) forward parameter
#    index. Which parameter is the biggest outlier and why?
# 3. **(Medium)** Write `ClampGrad(Function)` (identity fwd, clamp bwd to ±c) and
#    show a case where it differs from `t.register_hook(clamp)` — hint: give the
#    tensor two consumers.
# 4. **(Hard)** Checkpoint only blocks 0–1 vs only blocks 2–3 in cell 5. Are the
#    savings the same? Explain using what `checkpoint` saves (the block *input*)
#    and what remains un-recomputed (embeddings, head).
#
# <details><summary>Solutions</summary>
#
# 1. `SumBackward0 ← AddBackward0 ← {MulBackward0 ← {Acc(a), Acc(b)}, Acc(c)}` —
#    `c` feeds `AccumulateGrad` directly: add passes gradients through unchanged.
# 2. Correlation ≈ −0.9 or stronger. Outlier: `tok_emb.weight` — tied to the head,
#    so its accumulation completes only after gradients from BOTH the first-layer
#    use and the head use arrive; it lands last despite the head being "first".
# 3. With `y = t + t.relu()`, a hook on `t` clamps the *summed* fan-in gradient;
#    `ClampGrad` inserted on one branch clamps only that branch before summation.
#    Position in the graph is semantics.
# 4. Nearly the same for equal-size blocks: each checkpointed block contributes
#    (its input) instead of (its ~dozen intermediates), independent of depth. The
#    asymmetry you do see comes from what is NOT checkpointed either way: the
#    embedding/dropout output and the final LN/head activations dominate both.
#
# </details>
#
# ## Summary
#
# - Forward records backward-closures + saved tensors; backward is a reverse
#   topological sweep; the tape dies as it is consumed (hence retain_graph errors).
# - `.grad` accumulation makes K-micro-batch gradients exact by linearity — the
#   basis of ch22 and of DDP's `no_sync`.
# - Per-parameter "grad ready" hooks + reverse arrival order = DDP's overlap.
# - identity-forward/side-effect-backward Functions are how comm layers are built.
# - Checkpointing: keep inputs, replay insides; ≤ 4/3 time for big activation cuts.
# - allocated = tensors, reserved = cache; only `empty_cache()` returns memory.

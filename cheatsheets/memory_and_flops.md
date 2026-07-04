# Cheatsheet — Memory & FLOPs (one page)

Symbols: L layers · d model width (=h) · a heads · V vocab · T max context ·
b batch · s sequence · N params · p world size. Derivations: ch03 (ch04 for Adam).

## Parameters

```
N = (V + T)·d + L·(12d² + 13d) + 2d          (GPT w/ bias, tied head; MiniGPT exact)
N ≈ 12·L·d²                                   (folk rule; embeddings negligible at scale)
per block: attn 4d² (qkv 3d², proj d²) + MLP 8d² (fc 4d², proj 4d²) + 13d bias/LN
```

## Training memory (bytes) — the ledger

```
weights          : N · b_w        (fp32: 4, fp16/bf16: 2)
gradients        : N · b_g        (usually same dtype as weights)
Adam m, v        : N · 8          (two fp32 moments)          ┐ full detail
fp32 master copy : N · 4          (mixed precision only)      ┘ in ch04
─────────────────────────────────────────────────────────────
fp32 training    : ~16 N          (4+4+8)
mixed precision  : ~16 N          (2+2+8+4)  ← the famous "16 bytes per param"
+ activations, + logits tail, + CUDA cache/context (below)
```

## Activations saved for backward (per layer, bytes)

```
fp16, with dropout (Megatron paper):   s·b·h·(34 + 5·a·s/h)
fp32, no dropout (our MiniGPT):      ≈ s·b·h·(64 + 4·a·s/h)
logits + log-softmax tail:           ≈ 8·s·b·V   (fp32) — dominates when V >> d
full activation checkpointing keeps: ~L·s·b·h·b_w (block inputs) + one live block
scaling: linear in b, QUADRATIC in s (the a·s²·b softmax term)
```

## FLOPs

```
matmul (m×k)(k×n)      : 2·m·k·n
forward                : ≈ 2N per token         (2 FLOPs per matmul param)
backward               : ≈ 4N per token         (two matmuls per forward matmul)
training total         : ≈ 6N per token   [+ attention: 12·L·s·d per token]
attention term matters when s ≳ 6d
whole run              : C ≈ 6·N·D  (D = tokens); Chinchilla-optimal D ≈ 20·N
inference (no cache-miss): ≈ 2N per generated token — memory-bound (ch01)
```

## KV cache (inference, bytes)

```
KV = 2 · L · s · d_kv · b · bytes/elt        (2 = K and V)
MHA: d_kv = d          GQA/MQA: d_kv = a_kv · d/a
```

## Quick numbers

| model | N | fp16 weights | train state (~16N) | FLOPs/token (6N) | KV @T,b=1 (fp16) |
|---|---|---|---|---|---|
| MiniGPT (course) | 5.32 M | 10.6 MB (21 fp32) | 85 MB | 34.7 M (incl. attn) | 2.1 MB (fp32) |
| GPT-2 small | 124 M | 0.25 GB | 2.0 GB | 0.74 G | 0.15 GB |
| GPT-3 175B | 175 B | **350 GB** | **2.8 TB** | 1.05 T | 9.7 GB |
| Llama-2 70B | 70 B | 140 GB | 1.12 TB | 0.42 T | 1.3 GB (GQA) |

Largest single GPU ≈ 0.08–0.19 TB → 175B is unfittable by >2× on weights alone;
training state needs ≥ 35 × A100-80GB *before activations*. Hence Parts II–IV.

## Rules of thumb

- Percent-of-peak or it didn't happen: MFU = achieved FLOP/s ÷ peak; 40–60 % is
  excellent for LLM training.
- What nvidia-smi shows = reserved + context ≥ allocated = your tensors (ch02).
- Elementwise ops: time = bytes ÷ bandwidth; FLOPs are free (ch01 roofline).
- Doubling s can more than double activation memory; doubling b exactly doubles it.
- Every parallelism divides ONE bill: DP → activations/rank · TP,PP → weights ·
  ZeRO → optimizer states (+grads, +weights by stage) · seq/context-P → the s terms.

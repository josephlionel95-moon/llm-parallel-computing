# Chapter 03 — Transformer Internals: Parameters, Activations & FLOPs

> **Difficulty:** 🟡 Medium · **Study time:** 5–7 h · **Requires:** ch01, ch02
> **Notebook:** notebooks/ch03_transformer_internals.ipynb · **Experiments:** experiments/ch03/
> **Do not skim this chapter.** Every later chapter spends these formulas.

This is the accountant's view of a transformer: not what it *learns*, but what it
*costs* — every parameter counted, every saved activation weighed, every FLOP
priced. The reference implementation is `src/llmdist/models/mini_gpt.py`
(read it first; it is 157 deliberately plain lines), whose methods
`param_formula()`, `num_params()`, and `flops_per_token()` implement what we
derive here, so theory can be diffed against measurement. The chapter ends with
the punchline that motivates the rest of the course: **GPT-3 175B cannot fit on
any single GPU that exists — not even close — and you can now prove it on one
page.**

## Learning objectives

After this chapter you will be able to:

- Derive the exact parameter count of a GPT from its config, and verify it against `MiniGPT.param_formula()` and `num_params()`.
- Explain where the folk rule $N \approx 12 L d^2$ comes from and when the embedding term breaks it.
- Derive per-layer activation memory — the classic fp16 estimate $sbh\,(34 + 5as/h)$ bytes — step by step, and adapt it to our fp32, dropout-free MiniGPT ($\approx sbh\,(64 + 4as/h)$).
- Derive training FLOPs ≈ $6N$ per token (forward $2N$, backward $4N$) from matmul dimensions, plus attention's $12\,L\,T\,d$ correction.
- Derive the KV-cache size formula and compute it for real models.
- Scale all formulas from MiniGPT to GPT-2, GPT-3 175B, and Llama-2 70B, and show why 175B can never fit on one GPU.
- Predict, before measuring, how activation memory moves with batch and sequence length — then verify with forward hooks.

## Intuition

A transformer is bookkept by five numbers: layers $L$, model width $d$ (=`d_model`,
a.k.a. $h$), heads $a$ (=`n_head`), vocabulary $V$, and context $T$ (=`block_size`);
at runtime add batch $b$ and sequence $s \le T$.

Three different "weights" of the model answer three different questions:

- **Parameters** ($\propto 12Ld^2$): what you must *store forever* — and, in
  training, quadruple-count for gradients and Adam states (ch04).
- **Activations** ($\propto L\,s\,b\,d$, plus an $s^2$ attention term): what you
  must *remember between forward and backward*. Chapter 02 showed the mechanism
  (saved tensors: backward of $Y=XW$ needs $X$); here we total the bill. Unlike
  parameters, this bill scales with *your batch and sequence*, which is why OOMs
  appear when you change data shape, not model.
- **FLOPs** ($\approx 6N$ per token): what you must *compute*, hence time and
  electricity. It is proportional to parameters — a deep fact with a one-line
  proof below.

The accountant's discipline pays off because every parallelism strategy in Parts
II–IV is a scheme for splitting exactly one of these bills: data parallelism
splits the *activation* bill (per-rank batch shrinks), tensor/pipeline parallelism
split the *parameter* bill, ZeRO splits the *optimizer-state* bill. If you can
compute the bills, you can choose the strategy — that is literally what the
Megatron and DeepSpeed papers do, with these same formulas.

## Theory

### The shapes, from the source

From `src/llmdist/models/mini_gpt.py` (defaults: $V{=}8192$, $T{=}256$, $L{=}4$,
$a{=}4$, $d{=}256$, bias=True, dropout=0, weight-tied head):

```
                     x: (b, s, d)
      ┌─────────────────────────────────────────────────────┐
      │ Block (×L)                                          │
      │  ln1: LayerNorm(d)                 [2d params]      │
      │  attn.qkv:  Linear(d → 3d)         [3d²+3d]         │
      │      q,k,v: (b, a, s, d/a) each                     │
      │      att = q@kᵀ/√(d/a): (b, a, s, s)   ← the s² !   │
      │      y = att@v → (b, s, d)                          │
      │  attn.proj: Linear(d → d)          [d²+d]           │
      │  ln2: LayerNorm(d)                 [2d]             │
      │  mlp.fc:    Linear(d → 4d)         [4d²+4d]         │
      │  mlp.proj:  Linear(4d → d)         [4d²+d]          │
      └─────────────────────────────────────────────────────┘
   tok_emb: (V, d)   pos_emb: (T, d)   ln_f: [2d]   head: tied to tok_emb
```

### Parameter count — exact

Per block, weights: attention $3d^2 + d^2 = 4d^2$; MLP $4d^2 + 4d^2 = 8d^2$;
total $12d^2$. Biases (when `bias=True`): qkv $3d$, attn.proj $d$, fc $4d$,
mlp.proj $d$ ⇒ $9d$. LayerNorms: $2 \times 2d = 4d$. So

$$\text{per block} = 12d^2 + 13d \qquad (12d^2 + 4d \text{ if bias=False})$$

Outside the blocks: token embedding $Vd$, positional embedding $Td$, final
LayerNorm $2d$, head **zero** extra (tied: `self.head.weight = self.tok_emb.weight`
— one storage, one parameter; recall ch00: `data_ptr()` equal, and
`Module.parameters()` deduplicates by identity). Total:

$$\boxed{N = (V + T)\,d + L\,(12d^2 + 13d) + 2d}$$

which is exactly `MiniGPT.param_formula()`. For the defaults:
$(8192{+}256)\cdot 256 + 4\cdot(12\cdot 256^2 + 13\cdot 256) + 512
= 2{,}162{,}688 + 3{,}159{,}040 + 512 = \mathbf{5{,}322{,}240}$.
`experiments/ch03/param_count_check.py` asserts `param_formula() == num_params()`
across a grid of configs — run it; if you edit the model and the assertion breaks,
the formula (or your edit) is lying.

**The folk rule.** For large models $d \gg 13$ and $Ld^2 \gg (V{+}T)d$, so
$N \approx 12Ld^2$. Sanity checks: GPT-2 small ($L{=}12, d{=}768$):
$12\cdot12\cdot768^2 = 84.9$M, plus embeddings $(50257{+}1024)\cdot768 = 39.4$M
⇒ ≈ 124M ✓. GPT-3 ($L{=}96, d{=}12288$): $12\cdot96\cdot12288^2 = 174.0$B ✓ —
embeddings (0.6B) are a rounding error at that scale, while for MiniGPT they are
**41 %** of the model. Small models are embedding-heavy; this inversion matters
when you extrapolate course experiments upward.

### Activation memory — the derivation

"Activations" = tensors saved for backward (ch02's `ctx.save_for_backward`),
dominated by per-layer intermediates of shape $(b, s, d)$ — count them as $sbh$
elements each ($h \equiv d$) — plus attention's $(b, a, s, s)$ tensors of
$a s^2 b$ elements.

**First, the literature-standard estimate** (Korthikanti et al. 2022): fp16
(2 bytes/element), with dropout, per layer:

| # | saved tensor | why backward needs it | bytes |
|---|---|---|---|
| 1 | LN1 input | LayerNorm backward | $2sbh$ |
| 2 | qkv input (LN1 out) | $\partial W_{qkv}$ needs input | $2sbh$ |
| 3 | $q, k$ | backward of $qk^\top$ | $4sbh$ |
| 4 | $v$ | backward of $\text{att}\cdot v$ | $2sbh$ |
| 5 | softmax output | softmax backward | $2as^2b$ |
| 6 | softmax-dropout mask | dropout backward | $1as^2b$ (1 B) |
| 7 | dropout output (=att) | backward of $\text{att}\cdot v$ | $2as^2b$ |
| 8 | proj input $y$ | $\partial W_{proj}$ | $2sbh$ |
| 9 | attn dropout mask | | $1sbh$ |
| 10 | LN2 input | | $2sbh$ |
| 11 | fc input (LN2 out) | $\partial W_{fc}$ | $2sbh$ |
| 12 | GELU input (fc out) | GELU'(x) | $8sbh$ |
| 13 | mlp.proj input (GELU out) | $\partial W_{proj}$ | $8sbh$ |
| 14 | mlp dropout mask | | $1sbh$ |

Sum: $sbh(2{+}2{+}4{+}2{+}2{+}1{+}2{+}2{+}8{+}8{+}1) + as^2b(2{+}1{+}2)$

$$\boxed{A_{\text{layer}}^{\text{fp16}} = s\,b\,h\left(34 + 5\,\frac{a\,s}{h}\right) \text{ bytes}}$$

**Now adapt to our MiniGPT** — fp32 (4 bytes) and dropout $p{=}0$ (no masks, no
dropout-output copies; PyTorch's softmax backward saves only its *output*, and the
causal `masked_fill`'s bool mask is a persistent buffer, not an activation):

| group | tensors | bytes |
|---|---|---|
| LN inputs | rows 1, 10 | $8sbh$ |
| attention linears | rows 2, 3, 4, 8 | $4sbh{+}8sbh{+}4sbh{+}4sbh = 20sbh$ |
| softmax output | row 5 (single copy) | $4as^2b$ |
| MLP | rows 11, 12, 13 | $4sbh{+}16sbh{+}16sbh = 36sbh$ |
| LN mean/rstd | $2\times 2sb\cdot4$ | $16sb$ (negligible) |

$$\boxed{A_{\text{layer}}^{\text{MiniGPT,fp32}} \approx s\,b\,h\left(64 + 4\,\frac{a\,s}{h}\right) \text{ bytes}}$$

Outside the blocks, do not forget the tail, which *dominates small models*:
embedding-sum output $4sbh$; final-LN input & output $8sbh$; and the **logits**
$4sbV$ plus cross-entropy's saved log-softmax $\approx 4sbV$ — for MiniGPT
($V = 32h$) the logits pair costs $8sbV = 256\,sbh$, i.e. *more than all four
blocks combined*. (Real systems fuse the loss to avoid materializing both; see
Megatron's vocab-parallel cross-entropy, ch10.)

Numbers for the default config ($s{=}256, b{=}8, h{=}256, a{=}4$):
$sbh = 0.52$M elements; per layer $= 0.52\text{M} \times (64 + 4) \cdot 1\,\text{B} \approx 36$ MB
(here $as/h = 4$, so $4as/h = 16$: per layer $0.52\text{M}\times 80 = 42$ MB);
4 layers ≈ 168 MB; logits + log-softmax ≈ 134 MB. Compare parameters: 21 MB.
**Activations ≈ 14× parameters** at these settings — measure it with
`experiments/ch03/activation_memory.py` and watch the ratio move linearly in $b$
and superlinearly in $s$ (the $4as^2bh/h$ term).

Two scaling regimes hide in the boxed formula, echoing ch01's rooflines:

- $s \ll h/a \cdot 16$: linear term dominates; memory $\propto sb$ — double the
  batch, double the bill.
- $s$ large: the $s^2$ softmax term dominates — the reason long-context training
  blew up memory before FlashAttention (which recomputes instead of saving the
  $s^2$ tensor — checkpointing, ch02, applied surgically) and the reason
  sequence/context parallelism (ch12–13) exist.

### FLOPs — forward $2N$, backward $4N$, total $6N$ per token

From ch01: a matmul $(m{\times}k)(k{\times}n)$ costs $2mkn$ FLOPs. Push **one
token** (one row of width $d$) through a weight matrix $W \in \mathbb{R}^{d_{in}\times d_{out}}$:
$2\,d_{in} d_{out}$ FLOPs $= 2 \times (\text{\#params in } W)$. Summing over every
weight matrix in the model:

$$\text{forward} \approx 2N_{\text{matmul}} \text{ FLOPs/token},$$

where $N_{\text{matmul}}$ = parameters that live in matmuls — for MiniGPT
`num_params(non_embedding=True)` ≈ all of them minus `pos_emb` (a lookup, 0 FLOPs;
`tok_emb` *is counted* because its tied second life as the head is a genuine
$d\times V$ matmul). LayerNorms and biases add $O(d)$ per token — noise next to
$O(d^2)$.

Backward (derived in ch02, §Mathematics): each matmul does **two** matmuls in
backward ($\partial X = G W^\top$, $\partial W = X^\top G$), so backward
$\approx 4N$. Total:

$$\boxed{C \approx 6N \;\text{FLOPs/token}\qquad\text{(training)}}$$

**Attention's extra term.** The two batched matmuls $qk^\top$ and $\text{att}\cdot v$
have no weights; per token per layer they cost $2\,s\,d + 2\,s\,d = 4sd$ FLOPs
forward (each output row of $qk^\top$ is $s$ dot products of length $d/a$ across
$a$ heads $= sd$ multiply-adds), hence $12sd$ with backward. Over $L$ layers:

$$C \approx 6N + 12\,L\,s\,d \;\text{FLOPs/token}$$

— which is precisely `MiniGPT.flops_per_token()` (with $s{=}T$). The correction
matters when $12LTd \gtrsim 6N \approx 72Ld^2$, i.e. $T \gtrsim 6d$: negligible
for GPT-3 ($T{=}2048 \ll 6d{=}73728$), dominant for 128k-context training — the
FLOP-side reason for context parallelism (ch13). For MiniGPT:
$6 \cdot 5{,}256{,}704 + 4\cdot 12\cdot 256\cdot 256 = 31.5\text{M} + 3.1\text{M} = 34.7$ M FLOPs/token.
(A pedantic note: causality means the *average* token attends to $s/2$ keys, so
$12Lsd$ overstates attention ~2×; the convention above matches Chinchilla/PaLM
accounting.)

### KV cache (inference — but it shapes training systems too)

Generating autoregressively, each new token needs every previous token's $k, v$
per layer. Cache them:

$$\boxed{\text{KV bytes} = 2 \times L \times s \times d_{kv} \times \text{bytes/elt} \times b}$$

with $d_{kv} = d$ for standard MHA (MiniGPT: $2\cdot4\cdot256\cdot256\cdot4 = 2.1$ MB
per sequence) and $d_{kv} = a_{kv}\cdot d/a$ for grouped-query attention. This is
why decoding is the memory-bound regime of ch01: each generated token *reads* the
whole cache and all weights to do a mere $2N$ FLOPs — arithmetic intensity O(1).

### The scaling table — and the punchline

All formulas, evaluated. Training column assumes Adam with the stated precision
(full accounting of the 16 B/param mixed-precision rule is ch04; formulas in
`cheatsheets/memory_and_flops.md`).

| | MiniGPT | GPT-2 small | GPT-3 175B | Llama-2 70B |
|---|---|---|---|---|
| $L$ / $d$ / $a$ | 4 / 256 / 4 | 12 / 768 / 12 | 96 / 12288 / 96 | 80 / 8192 / 64 |
| $V$ / $T$ | 8192 / 256 | 50257 / 1024 | 50257 / 2048 | 32000 / 4096 |
| $N$ (formula) | 5.32 M | ≈ 124 M | ≈ 175 B | ≈ 70 B |
| weights, train dtype | 21 MB (fp32) | 0.5 GB (fp32) | 350 GB (fp16) | 140 GB (bf16) |
| + grads + Adam (≈16 B/param mixed) | 85 MB | 2.0 GB | **2.8 TB** | 1.12 TB |
| activations/layer @($b{=}1$, $s{=}T$) fp16-est | 0.08 GB† | 0.03 GB | 1.16 GB | 0.79 GB‡ |
| train FLOPs/token ($6N$) | 34.7 M (incl. attn) | 0.74 G | 1.05 T | 0.42 T |
| KV cache @$s{=}T, b{=}1$ | 2.1 MB | 0.15 GB (fp16) | 9.7 GB (fp16) | 1.3 GB (GQA, fp16) |

† MiniGPT at its actual fp32/b=8 settings: ~42 MB/layer as derived above; the row
uses b=1,s=T fp16-estimate for comparability. ‡ with GQA ($a_{kv}=8$) reducing rows 3–5.

**The punchline.** The largest single accelerator you can buy holds ~192 GB
(H200/B200-class; the A100 of the GPT-3 era: 40–80 GB). GPT-3's *weights alone*
in half precision are 350 GB — **no batch size, no checkpointing trick, no
inference-only mode makes 350 > 192**. Training state is 2.8 TB before a single
activation: ≥ 35 × 80 GB A100s just to *hold* it. And time: $6N \times 300\,\text{B tokens}
= 3.15\times10^{23}$ FLOPs — a T4 at its full 65 fp16 TFLOPS would need ~154
years; even an A100 at a realistic 50 % MFU (156 TFLOPS achieved) needs ~64 years.
GPT-3 was trained in weeks — therefore *thousands* of GPUs, therefore sharded
memory, therefore everything in Parts II–IV. This table is the course's thesis
stated in arithmetic.

## Mathematics

The heavy derivations are inline above (they *are* this chapter). Collected:

$$N = (V{+}T)d + L(12d^2 + 13d) + 2d \;\approx\; 12Ld^2$$
$$A_{\text{layer}} = sbh(34 + 5as/h)\ \text{B (fp16, dropout)};\quad sbh(64 + 4as/h)\ \text{B (fp32 MiniGPT)}$$
$$A_{\text{logits}} \approx 8sbV \text{ B (fp32 logits + log-softmax)}$$
$$C \approx (6N + 12Lsd)\ \text{FLOPs/token};\quad \text{fwd}:\text{bwd} = 1:2$$
$$\text{KV} = 2\,L\,s\,d_{kv}\,b \cdot \text{bytes/elt}$$

One more derivation worth doing once: **why is fwd:bwd exactly 1:2 and not
model-dependent?** Because every FLOP in forward lives in some matmul $Y = XW$,
and backward performs precisely two same-shaped matmuls per forward one,
*regardless of architecture*. Elementwise ops break the exact ratio, but they are
FLOP-negligible (ch01: they are bytes, not FLOPs). Empirically you will measure
fwd:step ≈ 1:3 wall-clock only when compute-bound; on small/CPU models, overhead
and memory-bound ops smear it — `experiments/ch03/flops_vs_time.py` shows both.

## Implementation

`src/llmdist/models/mini_gpt.py` encodes the accounting as executable claims:

```python
def param_formula(self) -> int:
    c = self.cfg
    d, V, T = c.d_model, c.vocab_size, c.block_size
    bias = 9 * d if c.bias else 0
    per_block = 12 * d * d + bias + 4 * d
    return (V + T) * d + c.n_layer * per_block + 2 * d

def flops_per_token(self) -> int:
    N = self.num_params(non_embedding=True)
    return 6 * N + c.n_layer * 12 * c.block_size * c.d_model
```

The three experiments diff these against reality:

1. **`experiments/ch03/param_count_check.py`** — asserts formula == measured over
   a config grid (varying $L, d, V$, bias, and checking the tied-weight dedup),
   and prints the per-component breakdown (embeddings vs blocks vs LN) so you see
   the embedding share shrink as $d, L$ grow.
2. **`experiments/ch03/activation_memory.py`** — no torchinfo: forward hooks on
   every module record output shapes/bytes (the estimator), and
   `saved_tensors_hooks` weighs what autograd actually kept (the ground truth,
   deduped by storage — ch00's `data_ptr` again). Sweeps $b \in \{1,2,4,8\}$ and
   $s \in \{64,128,256\}$, prints measured vs $sbh(64+4as/h)\cdot L + 8sbV$ and the
   ratio, on CPU or GPU identically (on GPU it also shows `max_memory_allocated`).
3. **`experiments/ch03/flops_vs_time.py`** — computes `flops_per_token()`, times
   forward and forward+backward with `llmdist.profiling.timers.timeit`, reports
   achieved FLOP/s and — on GPU — percent of peak (MFU), plus the fwd:(fwd+bwd)
   ratio against the theoretical 1:3.

The notebook reproduces all three interactively and builds the scaling table
programmatically from the formulas, so you can add your own dream model row.

## Profiling & measurement

- **Param check:** must match to the digit. If you flip `bias=False`, formula and
  measurement move together by $L\cdot 9d$ — a good self-test.
- **Activation sweep expectations:** measured saved-bytes should track the formula
  within ~10–20 % (the formula ignores LN statistics, the embedding/dropout
  output, and small autograd bookkeeping; the logits term must be included or you
  will be off by ~2× on MiniGPT). Doubling $b$ at fixed $s$: memory ×2.0. Doubling
  $s$ at $b$ fixed: more than ×2 — compute the predicted factor
  $\frac{2s(64 + 8as/h)}{s(64+4as/h)}$ first, then confirm.
- **FLOPs vs time (T4):** MiniGPT default is *small*; expect single-digit
  percent MFU (launch overhead + memory-bound smallness, ch01) — scale $d$ to
  1024 in the script's `--d-model` flag and watch MFU climb toward the tens of
  percent. On CPU, expect ~10–100 GFLOP/s-class numbers; the point on CPU is the
  fwd:bwd ratio and the *linear* scaling of time with $6N$, not absolute rates.
- Always report: config, dtype, device, and formula-vs-measured side by side.

## Common mistakes

1. **Symptom:** hand count ≠ `num_params()`. **Cause:** forgot weight tying (head counted twice), or `pos_emb`, or LayerNorm's 2 params/feature, or the 9d biases. **Fix:** diff per-component against `param_count_check.py`'s breakdown.
2. **Symptom:** "params fit, so it'll train on my 16 GB GPU." **Cause:** ignored grads + Adam (×4 on fp32 params) and activations (× batch-dependent). **Fix:** full ledger: $4N$(w) $+4N$(g) $+8N$(Adam) + activations + logits, cf. `llmdist.profiling.memory.model_memory_ledger`.
3. **Symptom:** OOM appears only on long sequences though "memory is linear in tokens". **Cause:** the $as^2b$ softmax term — quadratic in $s$. **Fix:** the formula; then FlashAttention/checkpointing/sequence parallel (ch12–13, 21).
4. **Symptom:** activation measurement ~2× the per-layer formula on a small model. **Cause:** logits + log-softmax ($8sbV$) dwarf blocks when $V \gg d$. **Fix:** count the tail; fuse the loss if it hurts.
5. **Symptom:** FLOPs "measured" with $6N$ overestimates throughput needs — or underestimates for long context. **Cause:** used $N$ *including* embeddings (pos_emb does no matmul), or dropped $12Lsd$ at large $s$. **Fix:** `num_params(non_embedding=True)` + the attention term, exactly as `flops_per_token()` does.
6. **Symptom:** measured step time ≫ FLOPs/peak on small models; someone concludes "the GPU is broken". **Cause:** ch01 regimes — launch-bound and memory-bound ops don't ride the compute roofline. **Fix:** report MFU with model size; grow $d$ until compute-bound before judging hardware.
7. **Symptom:** KV-cache estimate 8× too big for a modern model. **Cause:** assumed $d_{kv} = d$ where the model uses GQA/MQA. **Fix:** $d_{kv} = a_{kv}\, d/a$; check the config's `num_key_value_heads`.
8. **Symptom:** memory grows across iterations though shapes are constant. **Cause:** keeping the loss graph alive (`losses.append(loss)` instead of `loss.item()`) — an accounting leak, not an allocator bug (ch02). **Fix:** detach/`item()` anything stored.

## Limitations & outlook

Our formulas assume vanilla architecture: FlashAttention removes the $as^2b$
*memory* term (not the FLOPs); GQA shrinks KV; MoE (ch14) breaks $C \propto N$ by
activating a fraction of parameters per token; and the fp16 constant 34 becomes
"depends" under selective checkpointing (Korthikanti et al. table 2 lists the
variants). The accounting also stops at one device: ch04 assembles the single-GPU
training-step ledger (adding optimizer states and the allocator behaviors from
ch02), ch05 turns this chapter's punchline into the formal "three walls", and
each parallelism chapter thereafter opens by re-deriving *its* slice of these
bills — divided by the parallel degree, plus a communication term from ch07. Keep
`cheatsheets/memory_and_flops.md` printed next to your keyboard.

## Exercises

1. **(Easy)** Compute $N$ by hand for `GPTConfig(vocab_size=1000, block_size=128, n_layer=2, n_head=2, d_model=128, bias=False)`, then verify with `param_formula()` and `num_params()`. What fraction is embeddings?
2. **(Easy)** For GPT-2 small, compute weights/grads/Adam memory in fp32 training. Does it fit on a T4 (16 GB) *before* activations? With $b{=}8, s{=}1024$ activations (fp32 formula), does it still?
3. **(Medium)** Derive the activation formula variant for MiniGPT with dropout $p>0$ (masks are 1 byte/element in the fp16 accounting; PyTorch fp32 saves them as bool too). Which rows of the table change, and what is the new constant?
4. **(Medium)** At what sequence length does attention's $12Lsd$ equal $6N$ for (a) MiniGPT, (b) GPT-3? Interpret: for which models is "FLOPs ≈ 6N" a safe lie?
5. **(Medium)** Llama-2 70B uses GQA with 8 KV heads (head dim 128). Compute KV cache for $b{=}32$ concurrent 4k-context sequences in fp16, and compare with an MHA counterfactual. How many extra concurrent users does GQA buy per 80 GB GPU, all else equal?
6. **(Hard)** Prove that the fwd:bwd FLOP ratio for the *attention score* matmuls is also 1:2 (they have no weights — where do the two backward matmuls come from?). Then account for causal masking: what is the exact FLOP count of the masked $qk^\top$ if you skip masked entries, and why do real kernels usually not skip them?
7. **(Hard)** Using this chapter's formulas only, find the largest $d$ (at $L{=}48$, $V{=}50257$, $T{=}2048$, fp16 weights, mixed-precision Adam ≈ 16 B/param, $b{=}1$, full activation formula, no checkpointing) trainable on one 80 GB A100. Now allow full activation checkpointing (keep only $L\cdot sbh\cdot 2$ B of block inputs + one live block) — how much does $d$ grow? Which bill binds in each case?
8. **(Research)** Chinchilla says compute-optimal training uses $\approx 20$ tokens/parameter. Combine with $C = 6N\cdot\text{tokens}$ to express optimal $N$ as a function of a compute budget $C$ in FLOPs, then compute the optimal model for "1000 A100-days at 40 % MFU". Read §3 of the Chinchilla paper and reconcile your constant with theirs.

## Solutions

<details>
<summary>Solutions 1–5</summary>

**1.** Per block: $12\cdot128^2 + 4\cdot128 = 196{,}608 + 512 = 197{,}120$ (bias=False ⇒ no $9d$); ×2 = 394,240. Embeddings $(1000{+}128)\cdot128 = 144{,}384$; final LN 256. $N = 538{,}880$. Embedding share ≈ 26.8 %.

**2.** $N \approx 124$M: fp32 weights 496 MB + grads 496 MB + Adam m,v 992 MB ≈ 1.98 GB — fits easily. Activations, fp32 formula, $b{=}8, s{=}1024, h{=}768, a{=}12$: $as/h = 16$ ⇒ per layer $sbh(64{+}64) = 6.29\text{M}\times128$ B ≈ 0.81 GB ⇒ ×12 ≈ 9.7 GB, plus logits $8sbV \approx 3.3$ GB ⇒ ~15 GB activations: **no longer fits** (16 GB, minus context/cache). Exactly why ch21's checkpointing and ch20's fp16 exist.

**3.** Add: softmax-dropout mask $as^2b$ B, its output another $4as^2b$ B (fp32), attn-dropout mask $sbh$ and MLP dropout mask $sbh$ (bool = 1 B), plus the embedding dropout's mask outside the layers. New per-layer ≈ $sbh(66 + 9as/h)$ B fp32 (bool masks at 1 B/elt; if stored as fp32 multipliers, worse). The point: dropout's memory cost is masks + extra saved outputs, not "free".

**4.** $12Lsd = 6N \approx 72Ld^2 \Rightarrow s = 6d$. (a) MiniGPT: $s = 1536$ (6× its 256 context — so at $s{=}T{=}256$ attention adds only ~10 %, matching 3.1M vs 31.5M). (b) GPT-3: $s = 73{,}728 \gg 2048$: 6N is a fine lie. For any 128k-context model with $d < 21$k, it is not.

**5.** GQA: $2\cdot80\cdot4096\cdot(8\cdot128)\cdot2\,\text{B} = 1.34$ GB/seq ⇒ ×32 ≈ 43 GB. MHA counterfactual: $d_{kv} = 8192$ ⇒ 10.7 GB/seq ⇒ 344 GB (impossible). Per 80 GB GPU with, say, 20 GB spare for cache: GQA ~14 users vs MHA ~1. GQA is a memory-hierarchy decision, not a modeling one.

</details>

<details>
<summary>Notes toward 6–8</summary>

**6.** For $S = qk^\top$: $\partial q = G_S k$ and $\partial k = G_S^\top q$ — two matmuls of the same $2s^2 d/a$-per-head cost as forward's one; likewise for $\text{att}\cdot v$. The "weights" of these matmuls are just the *other activation*, so the 1:2 law is really "every matmul spawns two in backward", weightless or not. Causal skipping would halve to $s(s{+}1)/2$ rows, but tensor cores want dense tiles; kernels instead mask post-hoc (FlashAttention *does* skip fully-masked tiles — tile-granular causality — recovering most of the 2×).

**7.** Set up $M(d) = 16N(d) + A(d) \le 80$ GB with $N \approx 12\cdot48\,d^2 + 51{,}281\,d$ ... solve the quadratic; without checkpointing the *activation* term at $s{=}2048$ (fp16 constant 34, $a$ s.t. $d/a{=}128$) binds first for large $d$; with full checkpointing the $16N$ term binds and $d$ roughly doubles. The meta-lesson: checkpointing shifts which wall you hit, it does not remove walls — ZeRO (ch15) attacks the 16N wall itself.

**8.** $C = 6N \cdot 20N = 120N^2 \Rightarrow N = \sqrt{C/120}$. 1000 A100-days at 40 % MFU: $C = 1000 \cdot 86400 \cdot 0.4\cdot 312\times10^{12} \approx 1.08\times10^{22}$ ⇒ $N \approx 9.5$B, ~190B tokens. Chinchilla's fitted exponents are ≈ 0.5 as this crude model predicts; their constant differs (~20 tokens/param is itself the fit). Discrepancies to discuss: embedding FLOPs, attention term, MFU drift with scale.

</details>

## Interview questions

1. Derive the parameter count of a decoder-only transformer from $L, d, V, T$. Where does $12Ld^2$ come from, term by term?
2. Why does weight tying change the parameter count but not the FLOP count?
3. Walk through why training FLOPs ≈ $6N$ per token. Which architectural changes break the proportionality to $N$? (Name at least two.)
4. Derive the $sbh(34 + 5as/h)$ activation estimate for one fp16 transformer layer. Which term explains long-context OOMs?
5. Your model OOMs when you double sequence length but not when you double batch. What does that tell you quantitatively?
6. What exactly is saved for backward by `Linear`, `LayerNorm`, `softmax`, and `GELU` respectively, and why those tensors specifically?
7. Compute GPT-3's minimum GPU count on 80 GB A100s for (a) fp16 inference weights only, (b) mixed-precision Adam training state, ignoring activations. Show the arithmetic.
8. Why is autoregressive decoding memory-bound while training the same model is compute-bound? Use arithmetic intensity and the KV cache in your answer.
9. What does GQA change in the KV-cache formula, and why does it barely change model quality economics but transform serving economics?
10. You have the total FLOPs of a training run and a GPU's peak. What is MFU, what values are respectable for LLM training, and what eats the gap?
11. Estimate activations-per-parameter for MiniGPT at $b{=}8, s{=}256$ and for GPT-3 at $b{=}1, s{=}2048$. Why does the ratio invert with scale, and which parallelism strategy does each regime call for?
12. If a new optimizer needed only 4 bytes/param of state, which entries of the scaling table change, and by how much for the 175B row?

## Summary

- $N = (V{+}T)d + L(12d^2+13d) + 2d$ — exact for MiniGPT, `param_formula()` proves itself against `num_params()`; folk rule $12Ld^2$ once embeddings are negligible.
- Small models are embedding-heavy (MiniGPT: 41 %); big ones are block-heavy — remember when extrapolating.
- Activation memory per layer: fp16 $sbh(34{+}5as/h)$ B; our fp32/no-dropout MiniGPT $\approx sbh(64{+}4as/h)$ B; plus a logits tail $8sbV$ that dominates small-$d$/big-$V$ models.
- Activations scale with *data shape* ($\propto b$, worse than linear in $s$); parameters don't — the two bills OOM differently.
- Training FLOPs ≈ $6N$/token ($2N$ fwd + $4N$ bwd, two backward matmuls per forward one) + attention's $12Lsd$, which matters when $s \gtrsim 6d$.
- KV cache $= 2Lsd_{kv}b\cdot$bytes; GQA shrinks $d_{kv}$; decoding reads everything to compute almost nothing — memory-bound by construction.
- GPT-3: 350 GB fp16 weights, ~2.8 TB train state, $3.15\times10^{23}$ FLOPs — impossible on one GPU by three separate arguments; this is the course's motivation, now proven.
- fwd:bwd = 1:2 in FLOPs is architecture-independent (per-matmul law); wall-clock only matches when compute-bound (ch01 regimes).
- Every parallelism strategy is a division of exactly one bill: DP÷activations-per-rank, TP/PP÷parameters, ZeRO÷optimizer states.
- Verify everything: `experiments/ch03/` diffs each formula against a measurement; the cheatsheet holds the formulas.

## References

See `references/ch03.md` for the annotated list. Highlights:

- Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models* — the activation-memory formula's source (their §4), plus selective checkpointing.
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models* — appendix derives the $C \approx 6N$ accounting used industry-wide.
- Hoffmann et al. (2022), *Chinchilla* — turns the FLOP formula into the tokens-per-parameter rule.
- Brown et al. (2020), *GPT-3* — table 2.1 is the config data for our scaling table.
- Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters (Megatron-LM)* — where these formulas meet 3072 GPUs; the destination of this course.
- Pope et al. (2022), *Efficiently Scaling Transformer Inference* — the KV-cache/memory-bound-decoding analysis done properly.

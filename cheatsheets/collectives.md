# Collectives cheatsheet — one page

**α–β model**: one message of N bytes costs `T = α + βN`.
α = per-message latency (µs: launch/handshake/doorbell) · β = 1/bandwidth (s/B)
Crossover `N* = α/β` (≈ 100s of KB): below → batch messages; above → minimize bytes on busiest link.

| collective | before → after (4 ranks) | cost (good algo) | LLM use |
|---|---|---|---|
| **broadcast** | `A ∙ ∙ ∙` → `A A A A` | tree: `⌈log₂p⌉(α+Nβ)`; pipelined ring → `~Nβ` | replica init (DDP/ch08) |
| **reduce** | `A₀ A₁ A₂ A₃` → `Σ ∙ ∙ ∙` | tree: `⌈log₂p⌉(α+Nβ)` | building block |
| **all_reduce** | `A₀ A₁ A₂ A₃` → `Σ Σ Σ Σ` | ring: `2(p−1)α + 2(p−1)/p·Nβ` | **DDP grad sync** (ch08/09) |
| **scatter** | `[C₀C₁C₂C₃] ∙ ∙ ∙` → `C₀ C₁ C₂ C₃` | `(p−1)(α + N/p·β)` at root | data/ckpt sharding |
| **gather** | `C₀ C₁ C₂ C₃` → `[C₀C₁C₂C₃] ∙ ∙ ∙` | `(p−1)(α + N/p·β)` at root | metrics, ckpt save (ch19) |
| **all_gather** | `C₀ C₁ C₂ C₃` → all: `[C₀C₁C₂C₃]` | ring: `(p−1)α + (p−1)/p·Nβ` | **ZeRO-3/FSDP params** (ch15/16) |
| **reduce_scatter** | rows `[AᵢBᵢCᵢDᵢ]` → `ΣA ΣB ΣC ΣD` (one per rank) | ring: `(p−1)α + (p−1)/p·Nβ` | **ZeRO-2/FSDP grads** (ch15/16) |
| **all_to_all** | rank i chunk j → rank j slot i | `(p−1)(α + N/p·β)` | **MoE token routing** (ch14) |
| **barrier** | — | dissemination: `⌈log₂p⌉·α` | phase sync, timing |

N = full payload per rank. Costs are per-rank critical path.

## The two identities

```
all_reduce = reduce + broadcast              naive: 2(p−1)(α+Nβ) through ONE root link — anti-scales
all_reduce = reduce_scatter + all_gather     ring: balanced, bandwidth-optimal — and split in half it IS ZeRO/FSDP
```

## Ring all_reduce (the formula to memorize)

p−1 reduce-scatter steps (send chunk right, ADD incoming) + p−1 all-gather
steps (send right, OVERWRITE). Chunks of N/p; every link busy every step.

```
T_ring = 2(p−1)·α  +  (2(p−1)/p)·N·β          bytes/rank = 2(p−1)N/p < 2N  ∀p   (β-optimal)
```

α term linear in p ⇒ small N or huge p: use trees (`~2log₂p·α`) — NCCL
auto-switches by size. Deadlock rule for any ring step: post `irecv` BEFORE `isend`.

## Bus bandwidth (nccl-tests)

```
busbw = (N / t) × 2(p−1)/p        ← comparable to link specs, unlike algbw = N/t
```

## DP/DDP quick numbers (ch08/09)

- DP comm per step per rank = `2(p−1)/p · 4·N_params` bytes (fp32) — scales with MODEL, not data.
- Grad averaging is exact: ∇mean = mean(∇) (linearity); expect ~1e-7 float-order residue, use allclose.
- DDP = hooks (grads ready DURING backward, last-layer-first) + reverse-order ~25 MB buckets (amortize α)
  + async all_reduce (overlap) + finalize before step + no_sync for accumulation.
- Bucket size: U-curve — too small = α wall; too big = no overlap. Step floor: `fwd + max(bwd, comm) + opt`.
- Memory: DDP shards NOTHING — 16 bytes/param (fp32+Adam) on every rank → ZeRO (ch15).

Everything above is measured by `experiments/ch07..09/`; fit YOUR α, β with
`experiments/ch07/benchmark_collectives.py` before believing any prediction.

# Chapter 07 — Collective Communication, From Scratch

> **Difficulty:** 🔴 Hard · **Study time:** 6–8 h · **Requires:** chapters 05, 06
> **Notebook:** notebooks/ch07_collective_communication.ipynb · **Experiments:** experiments/ch07/

## Learning objectives

You will be able to...

- name every collective, draw its before/after state on 4 ranks from memory, and map each to the LLM-training subsystem that depends on it;
- implement broadcast, reduce, all-reduce, scatter, gather, all-gather, reduce-scatter, and a barrier using **nothing but point-to-point send/recv**;
- derive the α–β cost model from first principles and fit α and β to your own hardware from measurements;
- derive the ring all-reduce cost 2(p−1)α + 2(p−1)/p·Nβ step by step, and prove its bandwidth term is within a factor of the information-theoretic lower bound;
- explain why NCCL switches between ring and tree algorithms by message size;
- compute *bus bandwidth* the way `nccl-tests` does, and say why raw N/time understates the hardware;
- recognize and avoid the send/send deadlock in ring exchanges;
- predict which regime (latency vs bandwidth) any given gradient tensor lives in.

## Intuition

A network gives you exactly one ability: move bytes from one process to another
(`send`/`recv`). Everything else — "give every GPU the sum of all gradients",
"shard this checkpoint across 64 workers" — is *choreography* built on that
primitive. Libraries like NCCL hide the choreography behind one-line calls, which
is wonderful for shipping code and terrible for understanding cost. When
`dist.all_reduce` takes 90 ms and your step budget is 200 ms, the only way to
reason about it is to know who sends what to whom, in what order, and what each
message costs.

The analogy that survives contact with reality: a **relay of buckets around a
circle of people**. If everyone hands their full bucket to one person (naive
reduce), that person becomes the bottleneck and everyone else stands idle. If
instead everyone simultaneously passes a *cup* (1/p of a bucket) to their right
neighbour, pouring incoming cups into their own — after p−1 passes, each person
holds one perfectly mixed cup, and after p−1 more passes everyone has tasted
everything. Nobody ever idles; every hand is busy every round. That is ring
all-reduce, and it is why 8 GPUs can average a 10 GB gradient in roughly the
time it takes to push 2·10 GB through *one* link, regardless of p.

This chapter is the heart of the course. Data parallelism (ch08–09) is
"all_reduce after backward". ZeRO and FSDP (ch15–16) are "split the all_reduce
into its two halves and keep only your shard in between". MoE (ch14) is
"all_to_all the tokens to their experts". If you own this chapter, the rest of
the course is applications.

## Theory

### The cast, on four ranks

Each diagram shows the state **before → after** on ranks 0–3. `A₀` means "rank
0's tensor A"; `Σ` means elementwise sum over all ranks.

**broadcast(src=0)** — one rank's tensor to everyone. *LLM use: DDP initial
parameter sync — every replica must start from identical weights.*

```
before:   rank0: A        rank1: -        rank2: -        rank3: -
after:    rank0: A        rank1: A        rank2: A        rank3: A
```

**reduce(dst=0)** — elementwise sum lands on one rank; others unchanged.

```
before:   rank0: A₀       rank1: A₁       rank2: A₂       rank3: A₃
after:    rank0: ΣAᵢ      rank1: A₁       rank2: A₂       rank3: A₃
```

**all_reduce** — the sum lands on *every* rank. *LLM use: DDP gradient
averaging; also loss/metric aggregation.*

```
before:   rank0: A₀       rank1: A₁       rank2: A₂       rank3: A₃
after:    rank0: ΣAᵢ      rank1: ΣAᵢ      rank2: ΣAᵢ      rank3: ΣAᵢ
```

**scatter(src=0)** — rank 0 holds a list of p chunks; rank i receives chunk i.
*LLM use: rank 0 sharding an index or checkpoint across workers.*

```
before:   rank0: [C₀ C₁ C₂ C₃]   rank1: -     rank2: -     rank3: -
after:    rank0: C₀              rank1: C₁    rank2: C₂    rank3: C₃
```

**gather(dst=0)** — the inverse: chunks come home to one rank. *LLM use:
metric collection, single-writer checkpointing (ch19).*

```
before:   rank0: C₀       rank1: C₁       rank2: C₂       rank3: C₃
after:    rank0: [C₀ C₁ C₂ C₃]   others unchanged
```

**all_gather** — every rank ends with every chunk. *LLM use: ZeRO-3/FSDP
re-materializing full parameters from shards before each layer's
forward/backward.*

```
before:   rank0: C₀       rank1: C₁       rank2: C₂       rank3: C₃
after:    rank0..3: [C₀ C₁ C₂ C₃]   (identical everywhere)
```

**reduce_scatter** — sum p tensors, then each rank keeps only slice i of the
result. *LLM use: ZeRO-2/FSDP gradient reduction — each rank only needs the
gradient shard its optimizer partition owns.*

```
before:   rank0: [A₀ B₀ C₀ D₀]  rank1: [A₁ B₁ C₁ D₁]  rank2: [...]  rank3: [...]
after:    rank0: ΣAᵢ            rank1: ΣBᵢ            rank2: ΣCᵢ    rank3: ΣDᵢ
```

**all_to_all** — rank i's j-th chunk goes to rank j's i-th slot (a distributed
transpose). *LLM use: MoE token routing (ch14): each token's hidden state must
travel to the rank hosting its chosen expert, and back.*

```
before:   rank0: [x00 x01 x02 x03]     rank1: [x10 x11 x12 x13]  ...
after:    rank0: [x00 x10 x20 x30]     rank1: [x01 x11 x21 x31]  ...
```

**barrier** — no data; nobody passes until everybody arrives.

### The two identities that organize everything

```
all_reduce  =  reduce  +  broadcast            (naive: a root bottleneck)
all_reduce  =  reduce_scatter  +  all_gather   (ring: perfectly balanced)
```

The second identity is arguably the most consequential equation in distributed
training. DDP performs the whole `all_reduce`; ZeRO-2/FSDP performs
`reduce_scatter`, keeps only its gradient shard, updates only its optimizer
shard, and performs the `all_gather` on *parameters* later. Same bytes, moved
at different times, against radically different memory footprints.

### Deadlock: the one bug you must design against

A ring step is "send right, receive left" — *simultaneously on every rank*. If
everyone calls a blocking `send` first, all p ranks wait for a `recv` that
nobody has posted. Worse, small messages often *appear* to work because the
transport buffers them internally; the deadlock manifests only above some size
threshold. Our implementation's rule (see `_exchange` in
`src/llmdist/comm/manual_collectives.py`): post the non-blocking `irecv`
first, then `isend`, then wait on both. MPI programmers know this dance as
`sendrecv`.

## Mathematics

### The α–β model, from first principles

What does it cost to move one message of N bytes between two processes?
Empirically, the time is affine in N:

$$T(N) = \alpha + \beta N$$

Where does each term come from?

- **α (latency, seconds/message).** Costs that occur once per message and do
  not depend on N: syscalls and driver calls, protocol handshakes, NIC
  doorbells, on GPUs a kernel launch and stream synchronization, on Ethernet
  the packetization floor. Typical magnitudes: ~1–5 µs (InfiniBand RDMA),
  ~10–50 µs (TCP/Gloo), ~5–20 µs (NCCL kernel launch path).
- **β (inverse bandwidth, seconds/byte).** Every byte must traverse the
  narrowest link: β = 1/BW. PCIe 3.0 x16 ≈ 12 GB/s effective → β ≈ 0.083 ns/B;
  NVLink 3 ≈ 250 GB/s aggregate; T4s on Kaggle talk over PCIe.

Setting the two terms equal gives the **crossover size** N\* = α/β. For
α = 30 µs and 12 GB/s, N\* ≈ 360 KB:

- N ≪ N\*: **latency-dominated.** Ten 1 KB messages cost ~10α; one 10 KB
  message costs ~α. *Batch your messages* — this single observation motivates
  DDP's gradient buckets (ch09) and `average_gradients(flat=True)` (ch08).
- N ≫ N\*: **bandwidth-dominated.** Only total bytes matter; algorithm choice
  should minimize bytes over the busiest link.

`experiments/ch07/benchmark_collectives.py` measures T(N) from 1 KB to 64 MB
and least-squares fits α and β on your machine; the notebook does the same
inline on Gloo. On a log-log plot the curve is flat left of N\* and slope-1
right of it — you can *see* the model.

### Naive all-reduce: why the root drowns

Reduce-to-root then broadcast, both linear: the root receives p−1 messages of
N bytes, then sends p−1 messages of N bytes, all serialized through its single
link:

$$T_{naive} = 2(p-1)(\alpha + N\beta)$$

Bytes crossing the root's link: 2(p−1)N. Meanwhile p−1 other links are idle.
Adding workers makes it *worse* — the definition of an unscalable algorithm.

### Ring all-reduce, derived step by step

Arrange p ranks in a logical ring; cut the tensor into p equal chunks of N/p
bytes.

**Phase 1 — reduce-scatter (p−1 steps).** At step s, every rank sends one
chunk to its right neighbour and receives one from the left, *adding* the
incoming chunk to its own copy (schedule: rank r sends chunk (r−1−s) mod p —
chosen so chunk k finishes at rank k). Track chunk k: it starts at rank k+1,
visits k+2, k+3, …, collecting one contribution per hop, and after p−1 hops
rests at rank k containing all p contributions. Every rank sends exactly one
chunk and receives exactly one chunk per step — **all p links busy, every
step**.

**Phase 2 — all-gather (p−1 steps).** Same motion, but receivers *overwrite*
instead of add, circulating the p finished chunks to everyone.

Cost per rank: 2(p−1) messages, each α + (N/p)β:

$$\boxed{\;T_{ring} = 2(p-1)\alpha \;+\; \frac{2(p-1)}{p}\,N\beta\;}$$

Bytes sent per rank: 2(p−1)·N/p → **less than 2N no matter how large p
grows**. Compare the table:

| algorithm | messages (critical path) | bytes over busiest link | scales? |
|---|---|---|---|
| naive (root) | 2(p−1) | 2(p−1)·N | no — linear in p |
| ring | 2(p−1) | 2(p−1)/p·N < 2N | bandwidth: yes; latency: no |
| tree (binomial) | 2⌈log₂p⌉ | ~2N·(pipelining-dependent) | latency: yes |

**Why the ring is bandwidth-optimal.** Lower bound argument: in any
all-reduce, each rank owns N/p·(p−1)... more directly: each rank holds N bytes
of unique input that must influence every other rank's output, and must
receive the influence of (p−1)N bytes of others' data. With perfect in-network
reduction along the way, a rank still must send at least (p−1)N/p bytes (its
data's share of the reduced chunks it doesn't finalize) and receive at least
(p−1)N/p bytes (the reduced chunks it didn't compute) in each phase-equivalent;
totaling 2(p−1)N/p per rank. The ring achieves exactly this. No algorithm can
beat its β term; algorithms only differ in α and in constant-factor pipelining.

**Why NCCL switches algorithms by size.** The ring's α term, 2(p−1)α, is
linear in p: at p = 1024 and α = 5 µs, that's ~10 ms *before a single byte
counts*. A (double binary) tree pays ~2log₂(p)α ≈ 100 µs instead, at modest
bandwidth cost. So: small messages → tree (or even direct algorithms on
NVSwitch); large messages → ring. NCCL benchmarks the topology at init and
picks per call. You will reproduce the crossover on CPU in the notebook.

### Bus bandwidth (the nccl-tests convention)

Dividing N by the all-reduce time ("algorithm bandwidth", algbw) understates
the hardware, because the algorithm moved 2(p−1)/p·N bytes per rank, not N.
nccl-tests therefore reports

$$busbw = algbw \times \frac{2(p-1)}{p}$$

a number directly comparable to link specs across different p. Our
`llmdist.profiling.timers.bandwidth_gbps(nbytes, ms, factor=2*(p-1)/p)`
implements it. When someone says "we get 480 GB/s all-reduce on 8×A100", they
mean busbw.

## Implementation

Everything lives in `src/llmdist/comm/manual_collectives.py` — built **only**
from point-to-point primitives, each function annotated with message count and
bytes-on-wire. Highlights:

The deadlock-free exchange used by every ring step:

```python
def _exchange(send_buf, dst, recv_buf, src):
    recv_req = dist.irecv(recv_buf, src)   # post receive FIRST
    send_req = dist.isend(send_buf, dst)
    recv_req.wait(); send_req.wait()
```

Ring all-reduce is the two phases, verbatim from the math (note the chunk
index arithmetic matching the derivation):

```python
# Phase 1: reduce-scatter — chunk k ends fully reduced at rank k
for step in range(p - 1):
    send_idx = (rank - 1 - step) % p
    recv_idx = (rank - 2 - step) % p
    _exchange(chunks[send_idx], right, buf, left)
    chunks[recv_idx].add_(buf)             # reduce EN ROUTE
# Phase 2: all-gather — circulate the finished chunks
for step in range(p - 1):
    send_idx = (rank - step) % p
    recv_idx = (rank - 1 - step) % p
    _exchange(chunks[send_idx], right, chunks[recv_idx], left)
```

Arbitrary sizes are handled by padding the flat view to a multiple of p and
unpadding at the end. The `barrier()` is a dissemination barrier —
⌈log₂p⌉ rounds of token passing, versus 2(p−1) messages for a naive
root-based one.

Also included: `broadcast_tree` (binomial: the informed set doubles each
round, ⌈log₂p⌉ critical-path messages) to contrast with the linear
`broadcast`, and two referee helpers — `assert_all_ranks_equal` and
`assert_matches_torch_allreduce` — which deliberately use *torch's own*
collectives for checking, so a bug in our code cannot hide itself.

A fact worth savoring: **Gloo has no reduce_scatter**, yet our p2p ring
reduce-scatter runs on Gloo just fine — once you own the primitive, backend
capability tables stop being walls.

The notebook (`notebooks_src/ch07_collective_communication.py`) walks the tour
(per-rank before/after prints), verifies both all-reduce variants against
`dist.all_reduce`, and renders the ring animation
(`animations/ch07_ring_allreduce.py` — GIF via `FuncAnimation` + PillowWriter,
static multi-panel for this doc at
`visualizations/ch07_ring_allreduce_steps.png`).

## Profiling & measurement

What to measure, with `experiments/ch07/`:

1. **`benchmark_collectives.py`** — T(N) for broadcast/all_reduce/all_gather
   (+ reduce_scatter under NCCL) from 1 KB to 64 MB; α–β fit; busbw column.
   Run CPU/Gloo first, then Kaggle 2×T4 (`torchrun --nproc_per_node=2 ...`).
   How to read it: find the flat region (α), the slope-1 region (β), the
   crossover; compare fitted β⁻¹ against the link spec. On Kaggle's PCIe T4s
   expect all-reduce busbw of very roughly the single-direction PCIe effective
   bandwidth (order 5–10 GB/s) — the expectation follows from the busbw
   definition and PCIe math in ch01; measure, don't trust.
2. **`manual_vs_builtin_allreduce.py`** — correctness asserts, then a race:
   manual naive vs manual ring vs `dist.all_reduce`. Expect the library to win
   on constants (C++ vs our Python loop, one α per step); expect the *slopes*
   at large N to tell the algorithmic story. At p=2 naive-vs-ring differ by at
   most ~2× in bytes — the ring's advantage is a factor p/(something≤2) and
   grows with world size.
3. Timing collectives correctly: barrier-align all ranks before starting the
   clock (a collective's time is defined by the *slowest* participant), use
   `min` over repetitions to strip scheduler noise, and synchronize CUDA when
   under NCCL (async kernels — the ch01/ch04 lesson again).

## Common mistakes

1. **Send-first ring exchange.**
   *Symptom:* hangs, but only for tensors above ~64 KB, and not always.
   *Cause:* both neighbours block in `send`; small messages hid in transport buffers.
   *Fix:* post `irecv` before `isend` (`_exchange`), or use `dist.batch_isend_irecv`.
2. **Benchmarking without a barrier.**
   *Symptom:* rank 0 reports 2 ms, rank 1 reports 40 ms for the same collective.
   *Cause:* ranks entered at different times; you measured skew, not the collective.
   *Fix:* `dist.barrier()` immediately before starting the timer, every iteration.
3. **Comparing manual results to torch with `==`.**
   *Symptom:* "my all_reduce is wrong by 3e-7!"
   *Cause:* different accumulation order; float addition is not associative.
   *Fix:* `allclose` with sane tolerances; this is inherent, not a bug (returns in ch08).
4. **Calling collectives on a subset of ranks.**
   *Symptom:* deadlock with no error, eventually a timeout.
   *Cause:* collectives are *collective* — every rank of the group must call, same op, same shapes.
   *Fix:* audit control flow for rank-dependent branches around collectives (classic: `if rank == 0: dist.barrier()`).
5. **Reporting algbw and calling it bandwidth.**
   *Symptom:* "our network only does 6 GB/s" on 12 GB/s links.
   *Cause:* forgot the 2(p−1)/p algorithm factor.
   *Fix:* report busbw; compare like with like.
6. **Assuming Gloo == NCCL semantics-and-features.**
   *Symptom:* `reduce_scatter` raises on CPU; CUDA tensors crawl on Gloo.
   *Cause:* backend capability matrices differ; Gloo lacks reduce_scatter and is not CUDA-optimized.
   *Fix:* detect backend (our `env_check`), use CPU tensors on Gloo, NCCL for CUDA; or use the manual p2p versions, which run anywhere.
7. **Reusing one buffer as both send and recv in an exchange.**
   *Symptom:* silently corrupted reductions (values slightly wrong).
   *Cause:* incoming bytes overwrite data still being sent.
   *Fix:* distinct buffers (our ring uses a temp `buf` for the incoming chunk in phase 1).

## Limitations & outlook

- The ring is bandwidth-optimal but its latency term 2(p−1)α is *linear in p*:
  at thousands of ranks, small collectives are latency disasters. Trees,
  hierarchical rings-of-rings (intra-node NVLink ring + inter-node IB ring),
  and NVSwitch direct reductions address this — ch23 and ch28.
- We modeled one flat network. Real clusters are hierarchical (NVLink ≫ PCIe ≫
  InfiniBand ≫ Ethernet), and NCCL's channels/graphs exploit that; our α–β
  model becomes per-level.
- We reduce in fp32 with `+`. Production stacks reduce in bf16/fp32-accum
  (ch20) and sometimes compress or quantize gradients (ch23).
- Everything here is synchronous. Overlapping these collectives with compute —
  hiding them entirely — is the subject of ch09, and scheduling them across
  parallelism axes is ch18/ch23.
- ch08 now uses exactly one of these primitives (all_reduce) to build data
  parallelism; ch15/16 will split it in half.

## Exercises

1. **(Easy)** After `all_reduce_ring`, every rank holds the sum. Produce the
   *mean* without any extra communication. Why is post-division correct?
2. **(Easy)** Count messages and bytes-on-wire for `gather` then `broadcast`
   of the gathered result, vs one `all_gather`. Same bytes? Same critical path?
3. **(Medium)** Implement `all_to_all` from send/recv with the deadlock-free
   exchange pattern. State its per-rank message count and bytes. (MoE, ch14,
   will call this twice per layer.)
4. **(Medium)** Measure `broadcast` vs `broadcast_tree` at world sizes 4 and 8
   (spawn on CPU) for 1 KB and 16 MB. Explain the winner in each cell of the
   2×2 table with the α–β model.
5. **(Medium)** Derive the cost of *pipelined* ring broadcast (split N into k
   chunks): T = (p−2+k)(α + (N/k)β). Optimize over k. Show it approaches Nβ —
   independent of p — for large N.
6. **(Hard)** Prove the lower bound: any all-reduce where each rank starts
   with N bytes of unique data requires every rank to send at least (p−1)N/p
   bytes, even with in-network reduction. (Hint: how many bytes of information
   about *other* ranks' data must leave rank r's neighbourhood... consider what
   rank r's final N bytes depend on.)
7. **(Hard)** Our ring sends p−1 messages per phase regardless of chunk
   size, so tiny tensors pay 2(p−1)α. Design and implement a hybrid that
   switches to `all_reduce_naive` below a size threshold learned from your
   fitted α, β. Verify the crossover experimentally.
8. **(Research)** Read the NCCL double-binary-tree post (2.4) and the
   Bandwidth-optimal all-reduce literature (Patarasuk & Yuan). At what (p, N)
   frontier does tree beat ring for α = 5 µs, β = (1/150) ns/B (NVLink-class)?
   Sketch the phase diagram.

## Solutions

<details>
<summary>Solutions to exercises</summary>

1. `t /= world_size` on every rank after the collective. Correct because every
   rank holds the *identical* sum and division is elementwise and
   deterministic — replicas cannot diverge. (DP in ch08 does exactly this.)
2. gather: p−1 messages, (p−1)·N bytes into the root; broadcast of the
   concatenated pN bytes: (p−1)·pN bytes out of the root. Total bytes
   (p−1)N(1+p), all through one link, critical path 2(p−1) serialized sends.
   Ring all_gather: p−1 steps of N bytes per rank, (p−1)N per rank, all links
   parallel. Same asymptotic bytes per rank but a factor ~p better on the
   bottleneck link.
3. For s in 1..p−1: exchange with peer `(rank+s)%p` (send its chunk, recv
   yours) using irecv-then-isend; copy own chunk locally. p−1 messages of N/p
   bytes per rank → (p−1)N/p bytes per rank, T = (p−1)(α + N/p·β).
4. 1 KB (α-regime): tree wins at both sizes — ⌈log₂p⌉α vs (p−1)α, i.e. 2 vs 3
   at p=4, 3 vs 7 at p=8. 16 MB (β-regime): linear broadcast serializes
   (p−1)Nβ at the root; the tree still moves the full N over log p serial
   hops (Nβ·log₂p on the critical path) — better than linear, worse than a
   pipelined ring (~Nβ). Expect tree ≥ linear only in pathological small-p
   cases; measurement will show tree winning both 1 KB cells clearly and 16 MB
   less dramatically.
5. Chunk i leaves the root at step i; the last chunk arrives after (p−2+k)
   hops. Each hop costs α + (N/k)β. dT/dk = α − (p−2)Nβ/k² = 0 →
   k\* = √((p−2)Nβ/α), T\* = Nβ + (p−2)α + 2√((p−2)Nβα) → Nβ as N→∞.
6. Rank r's final tensor is a function of all pN input bytes. The (p−1)N/p
   bytes of the final reduced tensor that rank r does *not* compute locally
   during reduce-scatter must arrive from elsewhere (≥(p−1)N/p received), and
   symmetric counting on what r's data must influence elsewhere forces
   ≥(p−1)N/p sent per conceptual phase; summing both phases of any
   all-reduce-equivalent gives 2(p−1)N/p. (A fully rigorous proof: Patarasuk &
   Yuan 2009, Thm 1.)
7. Threshold at N where 2(p−1)α + 2(p−1)/p·Nβ = 2(p−1)(α+Nβ) ... note naive's
   root serialization makes it worse in α too at p>2; on 2 ranks the two match
   closely and the empirical crossover sits near N\* from your fit. Implement:
   `if t.numel()*t.element_size() < threshold: all_reduce_naive(t) else:
   all_reduce_ring(t)`.
8. Ring: 2(p−1)α + 2Nβ·(p−1)/p. Double binary tree: ≈ 2log₂(p)α + c·2Nβ with
   c ≈ 1 (two complementary trees each carry half). Setting equal:
   N_frontier ≈ αp/β·(const) — grows linearly in p: at p=512,
   frontier is in the multi-MB range; below it use the tree. The phase diagram
   is a single increasing line in (p, N) space; NCCL's tuner approximates it.

</details>

## Interview questions

1. Whiteboard ring all-reduce on 4 GPUs for a 4 MB tensor: exact messages,
   bytes per link, and total time given α = 10 µs, β for 12 GB/s.
2. Why is ring all-reduce called bandwidth-optimal, and what precisely is
   optimal about it? What is *not* optimal about it?
3. Your all-reduce of a 4 KB tensor takes the same time as a 400 KB one.
   Explain, and name the two hardware/protocol constants involved.
4. Write all_reduce as a composition of two other collectives, both ways.
   Which decomposition does FSDP exploit, and how?
5. A junior engineer's ring exchange deadlocks at 1 MB messages but passes
   unit tests at 1 KB. Diagnose the bug and give two fixes.
6. Define bus bandwidth as nccl-tests reports it. Why multiply by 2(p−1)/p,
   and what would using algbw instead do to cross-cluster comparisons?
7. Why does NCCL use trees for small messages and rings for large ones? What
   changes at p = 4096 that makes trees indispensable?
8. Where exactly does all_to_all appear in Mixture-of-Experts training, and
   what determines its message sizes?
9. Gloo doesn't implement reduce_scatter. You need one on CPU for a ZeRO-2
   prototype. What do you do (two options), and what does each cost?
10. Every rank must call every collective with matching signatures. Give three
    real code patterns that violate this and their observable symptoms.

## Summary

- Collectives are choreography over send/recv; we implemented all of them from
  p2p only, with per-function message/byte accounting, and proved them
  equivalent to `torch.distributed`'s.
- Cost model: T = α + βN per message; regimes split at N\* = α/β — measure α, β
  once per machine and you can predict any collective.
- Naive all-reduce funnels 2(p−1)N bytes through one root link; it anti-scales.
- Ring all-reduce = reduce-scatter + all-gather: 2(p−1) steps, 2(p−1)/p·N bytes
  per rank, β-term optimal; α-term linear in p — hence NCCL's size-based
  ring/tree switch.
- `all_reduce = reduce_scatter + all_gather` is the equation ZeRO/FSDP is built
  on; DDP uses the whole thing, MoE needs all_to_all, replica init needs
  broadcast.
- busbw = algbw × 2(p−1)/p; report it, compare it to link specs.
- Deadlock discipline: irecv before isend, always; bugs hide below the
  transport's buffer size.
- Time collectives with barriers, min-of-runs, and CUDA sync under NCCL.
- Gloo lacks features (reduce_scatter) — owning the p2p implementation makes
  backend tables advisory, not binding.
- Everything measured here: `experiments/ch07/` fits α, β and races manual vs
  library on your hardware.

## References

- Chan, Heimlich, Purkayastha, van de Geijn, *Collective communication: theory,
  practice, and experience* (2007) — the canonical α–β treatment of every
  collective; read §ring/recursive-halving to see this chapter with proofs.
- Patarasuk & Yuan, *Bandwidth optimal all-reduce algorithms for clusters of
  workstations* (2009) — the ring all-reduce optimality result; short and
  readable.
- Thakur, Rabenseifner, Gropp, *Optimization of collective communication
  operations in MPICH* (2005) — how a real library picks algorithms by size;
  the intellectual ancestor of NCCL's tuner.
- NVIDIA, *Massively scale deep learning with NCCL 2.4 double binary trees*
  (2019, developer blog) — why rings stop sufficing at large p; the log-p fix.
- NVIDIA, *nccl-tests* (github.com/NVIDIA/nccl-tests) — the busbw definition
  and the benchmark harness everyone quotes; run it after our experiments.
- Sergeev & Del Balso, *Horovod: fast and easy distributed deep learning in
  TensorFlow* (2018) — ring all-reduce's arrival in DL training at scale.
- Baidu SVAIL, *Bringing HPC techniques to deep learning* (2017) — the blog
  post that popularized ring all-reduce for gradients; historically where
  ch08's story begins.

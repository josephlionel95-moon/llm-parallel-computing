# Chapter 02 — PyTorch Execution: Autograd, Hooks & Checkpointing

> **Difficulty:** 🟡 Medium · **Study time:** 4–6 h · **Requires:** ch00, ch01
> **Notebook:** notebooks/ch02_pytorch_execution.ipynb · **Experiments:** experiments/ch02/

Every distributed-training system in this course is a *parasite on autograd*: DDP
injects communication into backward via hooks; ZeRO re-materializes parameters
inside backward; activation checkpointing replays forward inside backward. You
cannot understand the hosts of these parasites — let alone write one — without a
precise model of what `loss.backward()` actually executes. This chapter builds
that model, then uses it to implement the two mechanisms everything else rests on:
**gradient hooks** and **recompute-in-backward**.

## Learning objectives

After this chapter you will be able to:

- Draw the `grad_fn` graph PyTorch builds during forward, and predict its nodes for small programs.
- Define leaf vs non-leaf tensors, `requires_grad`, `retain_grad`, and state exactly which tensors get a populated `.grad`.
- Explain why `.grad` *accumulates* (+=) rather than assigns, why we call `zero_grad()`, and why the accumulation semantics is precisely what makes gradient accumulation (ch22) free.
- Use tensor hooks (`register_hook`) and module hooks (`register_full_backward_hook`, `register_post_accumulate_grad_hook`) — and identify the hook DDP uses to trigger AllReduce.
- Write a custom `torch.autograd.Function` with correct `forward`/`backward`, including one that is an identity in forward but performs a side effect in backward (the skeleton of every comm layer in ch08–10).
- Explain `torch.utils.checkpoint`: what is saved, what is recomputed, and derive its memory/time trade.
- Distinguish `memory_allocated` from `memory_reserved`, explain the CUDA caching allocator, and recognize fragmentation.

## Intuition

Autograd is a *tape recorder with a twist*. During forward, every operation on a
`requires_grad` tensor appends a node — not the data, but a closure that knows how
to compute the operation's vector–Jacobian product, plus references to whatever
tensors that closure will need (the "saved tensors" — this is where activation
memory comes from, quantified in ch03). `backward()` then plays the tape in
reverse topological order.

The twist: the tape is rebuilt from scratch *every iteration* ("define-by-run").
That is why Python control flow just works, and why the graph is a snapshot you
can walk, print, and — crucially — *attach callbacks to*. A hook is a doorbell on
a graph node: "when the gradient passes through here, call me." DDP's entire
overlap machinery (ch09) is doorbell-ringing: gradients for the *last* layers
arrive first (backward runs in reverse), so DDP starts AllReducing them while
earlier layers are still computing. The experiment `grad_arrival_order.py` lets
you watch exactly that order on MiniGPT.

Checkpointing is the complementary deal with the same tape: instead of saving
tensors for the closure, save nothing but the inputs and *re-run forward* when the
closure fires. Pay ~one extra forward, save almost all activation memory. Both
tricks are pure autograd; no CUDA magic is involved.

## Theory

### The graph autograd builds

```python
x = torch.randn(3, requires_grad=True)   # leaf
w = torch.randn(3, requires_grad=True)   # leaf
y = x * w        # y.grad_fn = <MulBackward0>,  saves x, w
z = y.sum()      # z.grad_fn = <SumBackward0>
```

```
            forward data flow ──►
   x ────┐
         ├──[ mul ]── y ──[ sum ]── z
   w ────┘

            what autograd stored (reverse edges) ◄──
   AccumulateGrad(x) ◄──┐
                        ├── MulBackward0 ◄── SumBackward0 ◄── z.grad_fn
   AccumulateGrad(w) ◄──┘        │
                            saved: x, w   (the "activation memory")
```

Facts to internalize:

- **Nodes are functions, edges are tensors.** `t.grad_fn` is the node that
  *produced* `t`; `node.next_functions` are the nodes that will consume the
  gradient flowing out of it. Walking `next_functions` recursively prints the
  whole graph — `experiments/ch02/gradfn_tree.py` does this for MiniGPT's loss.
- **Leaf tensors** are tensors not produced by a tracked operation (parameters,
  inputs you created). Their graph node is `AccumulateGrad`, whose entire job is
  `leaf.grad += incoming`. **Non-leaf** tensors get their gradient *passed
  through*, not stored, unless you call `t.retain_grad()`.
- `backward()` performs a reverse topological traversal from the root, calling
  each node once all gradients for its output have been summed ("dependency
  counting"). Multi-use tensors (e.g., a residual-stream tensor consumed by two
  branches) therefore receive the *sum* of both branch gradients — that is the
  chain rule for fan-out, implemented by the traversal itself.
- The graph is **freed during backward** (buffers released node by node) unless
  `retain_graph=True`. Calling `backward()` twice without it raises the famous
  `RuntimeError: Trying to backward through the graph a second time`. You need
  `retain_graph` almost never — its legitimate uses (multiple backward passes over
  a shared trunk, some GAN setups) all involve *two different roots over one tape*.

### `.grad` accumulation is a feature

`AccumulateGrad` does `p.grad = p.grad + g` (allocating on first touch). Design
consequences:

1. **You must `optimizer.zero_grad()`** (or `set_to_none=True`, the default since
   2.x, which frees instead of zero-fills — less memory traffic, and lets the
   allocator reuse the block).
2. **Gradient accumulation costs zero extra code**: run K micro-batch
   forward/backwards without zeroing, then step once. The `.grad` buffers hold
   $\sum_{k} \nabla L_k$ automatically. Chapter 22 builds on exactly this; DDP's
   `no_sync()` context exists to make the K−1 non-final backwards skip
   communication.
3. A **double-counted gradient bug** (e.g., calling backward twice on cached loss
   pieces) produces silently doubled `.grad`s — no error, just a wrong step. The
   fix is auditing *when AccumulateGrad fires*, which hooks make visible.

### Hooks: the doorbells

Three tiers, all used by real systems:

| hook | fires | signature | canonical user |
|---|---|---|---|
| `t.register_hook(fn)` (tensor) | when grad w.r.t. `t` is computed | `fn(grad) -> grad or None` | gradient clipping/inspection at one point; our arrival-order probe |
| `m.register_full_backward_hook(fn)` (module) | after grads w.r.t. *module inputs* computed | `fn(mod, gin, gout)` | debugging per-layer gradient norms |
| `p.register_post_accumulate_grad_hook(fn)` (param, ≥2.1) | after `p.grad` is finally accumulated | `fn(param)` | **DDP-style comm triggering**, fused optimizers |
| `torch.autograd.graph.Node.register_hook` | grad passing a graph node | | advanced surgery |

**The DDP connection, precisely:** `DistributedDataParallel` registers an autograd
hook per parameter (internally on the `AccumulateGrad` node — the public
equivalent is `register_post_accumulate_grad_hook`). Each hook marks the parameter
"ready" in its bucket; when a bucket's parameters are all ready, DDP launches an
async `all_reduce` on a side stream and moves on. Because backward produces
gradients roughly in *reverse forward order*, buckets for the tail of the network
fly while the head still computes. When you run `grad_arrival_order.py` you will
see MiniGPT's `head/ln_f` gradients arrive first and `tok_emb` last (with a wrinkle:
weight tying makes `tok_emb == head.weight`, so its accumulation happens only after
*both* uses have contributed — a real DDP subtlety: tied weights delay their bucket).

Also in the toolbox: `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` —
intercepts every tensor the graph *saves* for backward. We use it in ch03 to weigh
activation memory exactly, and it is how CPU-offloading of activations is
implemented (`save_on_cpu`).

### Custom `autograd.Function`: teaching backward new tricks

```python
class PrintGrad(torch.autograd.Function):
    """Identity in forward; announces (and could transform) the gradient in backward."""
    @staticmethod
    def forward(ctx, x, label):
        ctx.label = label
        return x                       # no compute, no copy

    @staticmethod
    def backward(ctx, grad_out):
        print(f"backward passing {ctx.label}: |g|={grad_out.norm():.4f}")
        return grad_out, None          # one grad per forward input
```

This "identity forward / side-effect backward" pattern is not a toy: Megatron's
tensor-parallel layers (ch10) are exactly this shape — `f`/`g` functions that are
identity (or AllReduce) in forward and AllReduce (or identity) in backward. Once
you can write `PrintGrad`, you can write `CopyToTensorParallelRegion`.

Rules that bite: `backward` must return one output per `forward` input (`None`
for non-tensor args); save tensors with `ctx.save_for_backward(...)` (not on `ctx`
directly, so autograd can track versions and support double-backward checks);
mark in-place mutations with `ctx.mark_dirty`. The notebook also implements a
full **ManualLinear** (returns `grad_x = g @ W`, `grad_W = gᵀ @ x`, `grad_b = g.sum(0)`)
and checks it against `nn.Linear` with `torch.autograd.gradcheck`.

### Activation checkpointing mechanics

`torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)`:

- **Forward:** run `fn(x)` under `no_grad`-like saving suppression: the graph
  *inside* fn saves nothing; only `x` (and RNG state, for dropout determinism) is
  kept. The output tensor's grad_fn is a special node holding `fn` and `x`.
- **Backward:** when the traversal reaches that node, it *re-runs* `fn(x)` with
  grad tracking on (restoring RNG state), building the sub-graph it skipped, then
  immediately backwards through it and discards it again.

Cost model, previewing ch03's numbers: a transformer block's forward is ~⅓ of its
total training FLOPs (forward 2N, backward 4N per token — ch03). Checkpointing
every block re-runs forward once more: total ≈ 8N vs 6N ⇒ **~33 % more FLOPs
for ~O(L)× less activation memory** (you keep only block inputs, `L·sbh·4` bytes,
instead of every intermediate). `experiments/ch02/checkpoint_memory_vs_time.py`
measures both sides of the trade on MiniGPT — on CPU it weighs saved-for-backward
bytes via `saved_tensors_hooks` (the honest CPU-visible proxy), on GPU it reads
`max_memory_allocated`.

Use `use_reentrant=False` (the modern implementation): the reentrant version runs
its recompute *outside* the normal engine in ways that break `retain_graph`
semantics, nesting, and — relevant to us — DDP's hook firing (DDP with reentrant
checkpointing requires `static_graph=True`; the non-reentrant one just works).

### The CUDA caching allocator

`cudaMalloc`/`cudaFree` are slow and synchronize the device, so PyTorch requests
big blocks from CUDA once and *caches* them:

- `torch.cuda.memory_allocated()` — bytes inside live tensors.
- `torch.cuda.memory_reserved()` — bytes PyTorch holds from CUDA (≥ allocated).
- `nvidia-smi` shows reserved + CUDA context (~hundreds of MB) — always scarier.

Freed tensors return blocks to the cache, not to CUDA. The allocator maintains
size-binned free lists; a request is served by splitting a cached block if
possible. **Fragmentation** = reserved ≫ allocated with an OOM anyway: no *single*
free block is big enough even though total free bytes are. Classic triggers:
varying sequence lengths (each length allocates odd sizes; blocks split into
unusable shards) and interleaving long-lived tiny tensors between short-lived big
ones. Mitigations: bucketed/padded shapes, `expandable_segments:True` (2.1+),
`torch.cuda.empty_cache()` (returns cache to CUDA — helps *other* processes, not
fragmentation within yours), and the profiler you already own:
`torch.cuda.memory_summary()`.

`llmdist.profiling.memory.snapshot()/track()` wrap these counters; every memory
number quoted in later chapters comes through that module.

## Mathematics

### Backward is two matmuls per one forward matmul

For $Y = XW$ with $X \in \mathbb{R}^{B\times d_{in}}$, $W \in \mathbb{R}^{d_{in}\times d_{out}}$,
given $G = \partial L/\partial Y$:

$$\frac{\partial L}{\partial X} = G W^{\top} \qquad (B{\times}d_{out})(d_{out}{\times}d_{in}): 2Bd_{in}d_{out}\ \text{FLOPs}$$
$$\frac{\partial L}{\partial W} = X^{\top} G \qquad (d_{in}{\times}B)(B{\times}d_{out}): 2Bd_{in}d_{out}\ \text{FLOPs}$$

Forward cost was $2Bd_{in}d_{out}$; backward is exactly **2×** forward for every
linear layer. This is the microscopic origin of ch03's "forward 2N, backward 4N,
total 6N FLOPs/token" — and of the checkpointing overhead ratio: recompute adds
one forward, $(2+4+2)/(2+4) = 4/3 \approx 1.33$.

Note also *what backward needs*: computing $\partial L/\partial W$ requires $X$ —
the layer's **input activation**. That single fact is why activations must be
saved (or recomputed), i.e., why activation memory exists at all.

### Gradient accumulation is exact, not approximate

With loss $L = \frac{1}{K}\sum_{k=1}^{K} L_k$ over $K$ micro-batches,
linearity of differentiation gives

$$\nabla_\theta L = \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta L_k.$$

`AccumulateGrad` computes the sum term by term across K backwards; dividing each
micro-loss by K (or the grads once at the end) yields *bit-for-bit the same
gradient* as one big batch — up to floating-point non-associativity and any
batch-statistics layers (BatchNorm breaks this; LayerNorm, used by transformers,
does not, because it normalizes per token, not per batch).

### Checkpointing's memory–time frontier

Let a model have $L$ blocks, activation bytes $a$ per block, block forward time
$t_f$, backward $\approx 2t_f$. Checkpoint every block:

$$\text{memory: } La \to L\,a_{\text{in}} + a \quad (\text{inputs of each block} + \text{one live block's insides}),$$
$$\text{time: } 3Lt_f \to 4Lt_f \;=\; 1.33\times.$$

Chen et al.'s $\sqrt{L}$ schedule (checkpoint every $\sqrt L$ blocks) achieves
$O(\sqrt{L}\,a)$ memory at the same 1.33× bound — derive it in exercise 6.

## Implementation

Read in this order:

1. **`experiments/ch02/gradfn_tree.py`** — builds MiniGPT (tiny config), computes
   a loss, and pretty-prints the reverse graph:

```python
def walk(fn, depth=0, seen=None):
    ...
    print("  " * depth + type(fn).__name__)
    for child, _ in fn.next_functions:
        if child is not None:
            walk(child, depth + 1, seen)

walk(loss.grad_fn)
```

   With `--max-depth` and a `seen` set (the graph is a DAG — residual connections
   make diamonds, and you will see them as revisited nodes).

2. **`experiments/ch02/grad_arrival_order.py`** — attaches
   `register_post_accumulate_grad_hook` (fallback: `Tensor.register_hook`) to every
   parameter, runs one backward, prints parameters in *arrival order* with a
   timestamp, and checks the correlation with reverse-forward order. This is DDP's
   bucketing assumption, verified empirically — including the tied-embedding
   anomaly discussed above.

3. **`experiments/ch02/checkpoint_memory_vs_time.py`** — wraps each MiniGPT block
   in `torch.utils.checkpoint.checkpoint` (non-reentrant), then measures step time
   (llmdist `timeit`) and activation bytes: `saved_tensors_hooks` byte-counter on
   CPU, `max_memory_allocated` deltas on GPU. Prints baseline vs checkpointed with
   the predicted 1.33× time ratio alongside.

The notebook additionally implements `PrintGrad` and `ManualLinear` (with
`gradcheck`), demonstrates `retain_graph` failure and `retain_grad`, shows a
double-backward-free `no_sync`-style accumulation loop, and inspects
`memory_allocated/reserved` around allocations (GPU cells guarded).

## Profiling & measurement

- **Arrival order:** expect a near-perfect reverse of forward order at module
  granularity: `head`(tied)…`blocks.3`…`blocks.0`…`pos_emb`. Quantify with
  Spearman rank correlation (the script prints it; expect ≈ −0.9 or stronger
  against forward index, with the tied embedding as the visible outlier).
- **Checkpointing on GPU (T4):** with MiniGPT default (4 layers, d=256, B=8,
  T=256), expect activation peak to drop by roughly the ratio of "all
  intermediates" to "block inputs only" — the ch03 formula predicts the numerator;
  time should rise by ~1.25–1.4× (the 4/3 bound plus non-recomputed parts).
  On CPU, expect saved-bytes to drop by ~10× or more while wall time rises ~1.3×.
- **Allocator:** after `del`eting a large tensor, `memory_allocated` falls,
  `memory_reserved` does not — that is the cache working as designed. Only
  `empty_cache()` lowers reserved.

## Common mistakes

1. **Symptom:** `Trying to backward through the graph a second time`. **Cause:** re-calling `backward()` on (part of) a freed tape — often a loss computed once outside a loop, or a hidden state carried across iterations. **Fix:** recompute the loss per step; `detach()` carried state; `retain_graph=True` only if you truly need two passes over one tape.
2. **Symptom:** `p.grad` is `None` after backward for some parameter. **Cause:** the parameter never participated (unused module/branch), or you inspected a *non-leaf* (e.g., a `.to(device)` result — the leaf stayed on CPU). **Fix:** check `p.is_leaf`; move the module *before* creating the optimizer; DDP has `find_unused_parameters` for the first cause (at a cost, ch09).
3. **Symptom:** loss diverges after "just adding logging" like `loss += aux.detach()... ` gone wrong, or gradients exactly 2× expected. **Cause:** double backward through a shared subgraph, or forgotten `zero_grad` making accumulation silently persist. **Fix:** audit with a `PrintGrad`/post-accumulate hook; adopt `set_to_none=True` discipline.
4. **Symptom:** in-place op error `one of the variables needed for gradient computation has been modified`. **Cause:** you mutated a tensor some `grad_fn` had saved (version counter mismatch) — e.g., `relu_` on a residual stream, or mutating through a `detach()` view (ch00!). **Fix:** drop the underscore; `clone()` before mutating shared storage.
5. **Symptom:** `register_full_backward_hook` never fires, or fires with surprising `grad_input`. **Cause:** module output not used in the loss; or in-place ops inside the module (which invalidate the hook's contract; PyTorch warns). **Fix:** use tensor/post-accumulate hooks for per-parameter logic — modules are the wrong granularity for it (DDP hooks *parameters*, not modules).
6. **Symptom:** checkpointing gives no memory savings. **Cause:** everything else (params/grads/optimizer, ch04) dominates activations at your scale; or inputs you checkpointed are the *big* tensors; or `use_reentrant=True` interacting badly with your wrapping. **Fix:** measure the activation share first (ch03 formula); checkpoint the activation-heavy region only.
7. **Symptom:** `nvidia-smi` says the GPU is "full" while `memory_allocated` is small. **Cause:** caching allocator's reserved pool + CUDA context — not a leak. **Fix:** compare `memory_reserved`; read `torch.cuda.memory_summary()`; only chase a leak if *allocated* grows step over step.
8. **Symptom:** dropout-using model gives different loss when checkpointed. **Cause:** recompute replayed different RNG. **Fix:** default `preserve_rng_state=True` handles it — you or a custom Function bypassed it; seed inside the checkpointed fn or keep the default.

## Limitations & outlook

Autograd's tape is per-process; nothing here synchronizes two GPUs — that is
exactly the gap DDP fills by *bolting communication onto the hooks you now know*
(ch08–09 build it by hand). Checkpointing trades FLOPs for memory but cannot touch
parameter/optimizer memory — ZeRO/FSDP (ch15–16) shard those, and they too are
implemented as autograd extensions (pre-forward gather hooks, post-backward
reduce-scatter hooks). The caching allocator story continues in ch24 with memory
snapshots and in ch16 where FSDP's alloc/free rhythm makes fragmentation a
first-class enemy. Finally, `torch.compile` captures graphs *ahead* of execution,
which changes the "rebuilt every iteration" premise — but its AOTAutograd
component still produces exactly the forward-2×/backward-4× structure derived here.

## Exercises

1. **(Easy)** Draw (on paper) the grad_fn graph of `z = (a*b + c).sum()` for leaf tensors a, b, c. Verify with a 5-line walker. Which node type appears for `c` and why?
2. **(Easy)** Demonstrate `.grad` accumulation: two backwards without zeroing, show `p.grad` doubles; then show `optimizer.zero_grad(set_to_none=True)` leaves `p.grad is None` — and explain why that is cheaper than zero-filling.
3. **(Medium)** Implement `ClampGrad(autograd.Function)`: identity forward, clamps gradient to [−c, c] in backward. Verify against `register_hook`-based clipping on the same model — where do the two mechanisms differ if the tensor feeds two consumers?
4. **(Medium)** Using `saved_tensors_hooks`, print every tensor (shape, dtype, bytes) MiniGPT's forward saves for one block. Match each against a line of `mini_gpt.py` — which op saved it, and which backward formula needs it?
5. **(Hard)** Write `manual_grad_accumulation(model, batches, K)` producing bit-identical `.grad` (allclose atol=0) to one concatenated big batch, on MiniGPT with dropout=0. Explain every place where floating-point order could have broken exactness, and why cross-entropy's internal mean forces you to weight micro-losses.
6. **(Hard)** Derive the $\sqrt{L}$ checkpointing result: with segments of size $k$, peak memory $\approx (L/k)\,a_{\text{in}} + k\,a$; minimize over $k$. Then implement segment checkpointing on MiniGPT's block list and verify the sweet spot for L=8 (use the CPU saved-bytes meter).
7. **(Research)** DDP fires AllReduce per *bucket*, not per parameter. Using arrival timestamps from `grad_arrival_order.py`, design and simulate a bucket-assignment policy that minimizes end-of-backward stall time given a bandwidth model from ch01's PCIe numbers. Compare against DDP's simple "25 MB in reverse order" default. What does weight tying do to your policy?

## Solutions

<details>
<summary>Solutions 1–4</summary>

**1.** `SumBackward0 ← AddBackward0 ← {MulBackward0 ← {AccumulateGrad(a), AccumulateGrad(b)}, AccumulateGrad(c)}`. `c` goes straight to `AccumulateGrad` because add's Jacobian is identity — no saved tensors, no intermediate node needed beyond Add itself.

**2.** After the second backward `p.grad` is exactly 2× (AccumulateGrad does +=). `set_to_none=True` deallocates: next backward's first accumulation *assigns* instead of adds — one less full-tensor write per param per step, and the allocator can recycle the block meanwhile.

**3.** Function version: `backward` returns `grad_out.clamp(-c, c)`; hook version: `t.register_hook(lambda g: g.clamp(-c, c))`. With two consumers, the tensor hook sees the *summed* fan-in gradient (it fires where grads w.r.t. `t` are complete), while a `ClampGrad` inserted on one branch clamps only that branch *before* summation — different results. Lesson: position in the graph is semantics.

**4.** You will see, per block: LN1 input (+ its mean/rstd), qkv input, q,k (for `q@kᵀ` backward), softmax output (for softmax and `att@v` backward), v, proj input, LN2 input (+stats), fc input, GELU input (fc output), proj(MLP) input. Each maps to "backward of op X needs operand Y": e.g., `grad_W_fc = xᵀ g` needs fc's input; softmax backward needs its *output* ($\mathrm{d}x = (g - (g\cdot y)\mathbf{1})y$ uses y only).

</details>

<details>
<summary>Notes toward 5–7</summary>

**5.** Key subtlety: `F.cross_entropy` averages over its *own* batch, so K micro-batches of size B/K each produce means over B/K — summing them ≠ mean over B. Scale each micro-loss by `(B/K)/B = 1/K`… which is only exact when micro-batches are equal-sized; unequal ones need per-batch weights $B_k/B$. Bit-exactness further requires identical reduction order inside each kernel — guaranteed here because each micro-batch's kernels see the same data in the same order; the only cross-micro-batch op is `+=` into `.grad`, which is associative-in-practice… but summing K terms in a different order than the fused big-batch reduction can still differ in the last ulp; on CPU float64 you can show exact equality, on fp32 use allclose with tiny atol and explain why.

**6.** $M(k) = (L/k)a_{\text{in}} + k\,a_{\text{peak}}$; $\mathrm{d}M/\mathrm{d}k = -(L/k^2)a_{\text{in}} + a_{\text{peak}} = 0 \Rightarrow k^* = \sqrt{L\,a_{\text{in}}/a_{\text{peak}}}$, giving $M(k^*) = 2\sqrt{L\,a_{\text{in}}\,a_{\text{peak}}} = O(\sqrt L)$. Time stays ≤ 4/3 since each segment recomputes once.

**7.** A good policy solves a scheduling problem: bucket i's comm can start at $t_{\text{ready}}(i)$ = max arrival in bucket, costs $s_i/B_{\text{pcie}} + \alpha$; total stall = $\max(0, \text{comm finish} - t_{\text{backward end}})$. Small tail buckets start comm earlier (less waiting for stragglers) but pay more α's; the α–β model of ch07 gives the trade. Tied weights arrive *late* (last accumulation wins), so put them in the final bucket regardless of their forward position — which is what DDP effectively experiences.

</details>

## Interview questions

1. What exactly does `loss.backward()` traverse, in what order, and how does it know when a node with multiple consumers is ready?
2. Leaf vs non-leaf: which tensors get `.grad` populated, and what do `retain_grad()` and `requires_grad_()` each change?
3. Why does PyTorch *accumulate* into `.grad` instead of overwriting? Name two features this enables and one bug class it creates.
4. Describe the hook mechanism DDP uses to overlap communication with backward. Why does bucketing exist on top of it?
5. In backward, why do gradients for the last layers of a network become available first, and what would break if a framework assumed *strictly* reverse order? (Hint: weight tying, shared modules.)
6. Write the backward of `Y = XW + b` on a whiteboard: shapes, formulas, FLOP count relative to forward, and which tensor must have been saved.
7. Explain what `torch.utils.checkpoint` saves, what it recomputes, when RNG state matters, and derive the ~33 % FLOP overhead.
8. `memory_allocated` says 2 GB, `memory_reserved` 10 GB, and you OOM allocating 3 GB. What happened and what are three mitigations?
9. Why must custom `autograd.Function.backward` return `None` for non-tensor forward arguments, and what does `ctx.save_for_backward` do that stashing on `ctx` does not?
10. Gradient accumulation over K micro-batches: prove it matches the big batch mathematically, and name the two practical caveats (loss scaling with reduction='mean', batch-statistics layers).

## Summary

- Forward builds a DAG of backward closures; `backward()` is a reverse-topological sweep with dependency counting; the tape is rebuilt every step.
- Leaves end in `AccumulateGrad` (`.grad += g`); non-leaves pass gradients through. Accumulation ⇒ `zero_grad`, and ⇒ free gradient accumulation.
- Backward of a linear layer = two matmuls per forward matmul (2× FLOPs) and *requires the saved input* — the origin of both "6N" and activation memory.
- Hooks are callbacks on graph events; `register_post_accumulate_grad_hook` is the public version of the mechanism DDP uses to launch AllReduce mid-backward.
- Gradient arrival order ≈ reverse forward order — the fact that makes comm/compute overlap possible; tied weights are the canonical exception.
- Custom `autograd.Function` = identity-forward/side-effect-backward is the skeleton of every distributed layer you will write (ch08/ch10).
- Checkpointing: save inputs, recompute insides; ≤ 4/3× time for O(L)→O(√L or L·inputs) activation memory; use non-reentrant.
- Allocator: allocated = live tensors, reserved = PyTorch's cache; fragmentation = free bytes you cannot use; nvidia-smi ≠ your tensors.
- `saved_tensors_hooks` weighs activation memory exactly — our scale for ch03.
- Everything above is per-process and per-GPU; distribution is these mechanisms plus communication, nothing more.

## References

See `references/ch02.md` for the annotated list. Highlights:

- PyTorch docs, *Autograd mechanics* — the official statement of leaves, accumulation, and in-place rules; read twice.
- Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library* (NeurIPS) — the design rationale for define-by-run autograd.
- Chen, Xu, Zhang & Guestrin (2016), *Training Deep Nets with Sublinear Memory Cost* — the √L checkpointing paper behind `torch.utils.checkpoint`.
- Li et al. (2020), *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (VLDB) — DDP's bucket/hook design straight from its authors; ch09's main text, but its §3 belongs to this chapter.
- PyTorch docs, *CUDA memory management* — allocated vs reserved, `expandable_segments`, fragmentation knobs.

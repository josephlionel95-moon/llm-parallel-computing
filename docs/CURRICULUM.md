# Full Curriculum — LLM Distributed Training

The definitive syllabus. Phase 1 (✅) ships chapters 00–09; later phases fill the rest.

## Part I — Foundations

**00 Python memory, processes & the GIL** ✅
Objects and references; shallow vs deep copies; serialization/pickling (what actually
crosses process boundaries); threads vs processes; the GIL and why PyTorch data loading
and `torch.multiprocessing` are built the way they are.

**01 GPU architecture** ✅
CUDA execution model: kernels, grids, blocks, warps; streaming multiprocessors and
occupancy; the memory hierarchy (registers → shared/L1 → L2 → HBM/global); tensor cores;
why matmul shape and dtype decide whether you get 8% or 80% of peak FLOPs; CUDA streams
and asynchronous execution.

**02 PyTorch execution** ✅
Dynamic autograd graph construction; `grad_fn` chains; what `.backward()` actually
traverses; tensor/module hooks (the mechanism DDP is built on); custom `autograd.Function`;
`torch.utils.checkpoint` mechanics; CUDA caching allocator (`memory_allocated` vs
`memory_reserved`).

**03 Transformer internals** ✅
Attention, MLP, LayerNorm, residuals from the memory/compute accountant's view. Exact
parameter count formula; exact activation memory per layer per token; FLOPs ≈ 6·N·D
derivation; KV-cache size; where the bytes and the FLOPs live — the numbers that motivate
every parallelism strategy.

**04 Single-GPU training anatomy** ✅
The training step under a microscope: forward, backward, optimizer. The 16–20 bytes/param
rule for Adam mixed-precision training, derived. Loss scaling. Reading
`torch.profiler` traces and `torch.cuda.memory._dump_snapshot` timelines.

**05 Why multi-GPU?** ✅
The three walls: memory wall (model + optimizer + activations vs 16–80 GB), compute wall
(GPU-years for one run), communication wall (the price of escaping the first two).
Scaling-law arithmetic: what training a 7B/70B/1T model actually requires.

## Part II — Communication

**06 Distributed computing basics** ✅
Processes, ranks, world size, `LOCAL_RANK`; rendezvous and the TCP store; `torchrun` vs
manual spawn; environment variables; NCCL vs Gloo vs MPI backends; process groups;
device binding; common initialization failures and their error messages.

**07 Collective communication** ✅
Broadcast, Reduce, AllReduce, ReduceScatter, AllGather, Scatter, Gather, All-to-All,
Barrier — each with a diagram, a manual point-to-point implementation, and a benchmark.
Ring AllReduce derived step by step (why 2(p−1)/p · N bytes); tree algorithms; the α–β
cost model; bandwidth vs latency regimes; measured NCCL curves vs message size.

**08 Data parallelism, by hand** ✅
Replicate the model, shard the batch, average the gradients — implemented with raw
collectives. Why you average (the linearity of gradients, derived). Gradient
synchronization correctness tests. Throughput and memory measurements. Why naive DP wastes
the backward pass; limitations that motivate DDP (ch09) and ZeRO (ch15).

**09 DDP internals** ✅
Build DDP from scratch: parameter broadcast, per-parameter grad hooks, bucketing,
async AllReduce overlapping backward compute. Then dissect the real
`DistributedDataParallel`: Reducer, bucket order, `find_unused_parameters`,
`no_sync()`, gradient-as-bucket-view. Overlap visualized on a timeline.

## Part III — Model parallelism (Phase 2)

**10 Tensor parallelism** — column/row-parallel linears, Megatron attention/MLP split,
f/g conjugate operators, communication per layer, manual 2-GPU implementation.
**11 Pipeline parallelism** — micro-batches, bubble fraction derivation, GPipe,
PipeDream, 1F1B, interleaved schedules; manual 2-stage pipeline with P2P ops.
**12 Sequence parallelism** — the LayerNorm/dropout activation redundancy in TP and how
scattering along sequence dim removes it.
**13 Context parallelism** — long-context training, ring attention, P2P KV exchange.
**14 Expert parallelism** — MoE routing, top-k gating, load balancing losses, All-to-All
dispatch/combine, capacity factors.

## Part IV — Memory & sharding (Phases 2–3)

**15 ZeRO** — stages 1/2/3 built manually; the 4+K bytes/param memory ledger per stage.
**16 FSDP** — manual flat-param sharding with AllGather/ReduceScatter; official FSDP;
wrapping policies; comparison with DeepSpeed ZeRO-3.
**17 DeepSpeed** — engine architecture, JSON config, launcher, ZeRO-Offload/Infinity.
**18 Megatron concepts** — combining TP×PP×DP×SP×EP; how to choose degrees; 3D parallelism
of GPT-3/PaLM/Llama-scale runs.
**19 Checkpointing** ✅
The full training state (seven pieces) and bitwise-exact resume; atomic saves
(tmp+fsync+replace); checkpoint size ledger (12 B/param); Young–Daly optimal interval
τ*=√(2δM), derived and Monte-Carlo verified; rotation and preemption-proof loops for
free-tier hardware; DDP rank-0 saving; sharded save/load + resharding with
`torch.distributed.checkpoint`. (Written ahead of 15–18 — it depends only on 04/06/08;
its sharded path is exercised again once FSDP arrives.)
**20 Mixed precision** — IEEE formats, FP16 vs BF16 dynamic range, loss scaling, master
weights, FP8 (Transformer Engine) concepts.
**21 Activation checkpointing** — recompute math (√L policy), selective checkpointing.
**22 Gradient accumulation** — effective batch size, `no_sync`, interaction with schedulers.

## Part V — Systems & scale (Phases 3–4)

**23 Communication optimization** — overlap, bucket sizing, compression, PowerSGD,
gradient sparsification.
**24 Profiling** — torch.profiler deep dive, Kineto traces, Nsight Systems concepts,
memory snapshots, NCCL debugging (`NCCL_DEBUG=INFO`).
**25 Scaling experiments** — 1→2→4 GPUs, strong vs weak scaling, Amdahl vs Gustafson,
efficiency curves measured on real hardware.
**26 End-to-end mini-GPT** — the capstone: same model trained single-GPU, DDP, FSDP,
DeepSpeed, TP, PP, hybrid; full comparison matrix (throughput, memory, utilization,
communication overhead, convergence).
**27 Modern stack survey** — NCCL/CUDA/Transformer Engine/FlashAttention/Triton/
Accelerate/Ray/SLURM/Kubernetes; who does what.
**28 Production systems** — DGX, SuperPOD, NVLink/NVSwitch, PCIe generations, InfiniBand,
RDMA, rail-optimized topologies; Colab/Kaggle vs DGX differences quantified.
**29 Paper deep-dives** — Megatron-LM, ZeRO, GPipe, PipeDream, GShard, Switch,
GPT-3, PaLM, Llama, BLOOM, Mixtral, DeepSeek-V3 — section-by-section readings.

"""Every collective, built ONLY from point-to-point send/recv.

Why this module exists
----------------------
`dist.all_reduce` is one line. It is also completely opaque: you cannot see
who sends what to whom, so you cannot reason about its cost. Here we rebuild
every collective from the two primitives a network actually gives you —
"send these bytes to that rank" and "receive bytes from that rank" — and we
annotate every function with its exact message count and bytes-on-wire.

Cost model (derived in ch07): sending a message of n bytes costs

    T(n) = alpha + n * beta

where alpha is per-message latency (microseconds: kernel launch, protocol
handshake, NIC doorbell) and beta is seconds-per-byte (1/bandwidth). A
collective's cost is the sum along its *critical path*, not the total work.

Deadlock discipline
-------------------
`dist.send` is (potentially) blocking: if every rank in a ring calls send
first, everyone waits for a receiver that never posts — the classic
head-to-head deadlock. Small messages may sneak through library buffering,
which makes the bug intermittent and therefore worse. Our rule: at every
exchange step, post the *receive* first as a non-blocking `irecv`, then
`isend`, then wait on both. isend/irecv are still point-to-point primitives —
we are not cheating, just refusing to gamble on buffer sizes.

Conventions
-----------
* All functions operate on CPU or CUDA tensors; the process group's backend
  (Gloo/NCCL/MPI) does the actual byte movement.
* "p" is the world size, "N" the payload size in bytes, "r" the caller's rank.
* Reductions are SUM unless noted; callers divide for a mean.
* Deterministic accumulation order (increasing rank) wherever we control it —
  floating-point addition is not associative, and ch08 cares.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def _exchange(send_buf: torch.Tensor, dst: int,
              recv_buf: torch.Tensor, src: int) -> None:
    """One deadlock-free bidirectional step: recv from `src`, send to `dst`.

    Posting irecv before isend guarantees a matching receive exists somewhere
    in the ring before anyone commits to a blocking send. Cost: one message
    each way, alpha + n*beta (the two directions overlap on full-duplex links).
    """
    recv_req = dist.irecv(recv_buf, src)
    send_req = dist.isend(send_buf, dst)
    recv_req.wait()
    send_req.wait()


# ---------------------------------------------------------------------------
# barrier
# ---------------------------------------------------------------------------

def barrier() -> None:
    """Dissemination barrier from p2p messages only.

    Round k (k = 0, 1, ...): rank r sends a token to (r + 2^k) mod p and
    receives one from (r - 2^k) mod p. After ceil(log2 p) rounds, every rank
    has (transitively) heard from every other rank, so nobody can be past the
    barrier while another rank is still before it.

    Cost: ceil(log2 p) messages of 1 byte per rank -> ~alpha * log2(p).
    Compare the naive gather-then-broadcast barrier: 2(p-1) messages through
    a single root, ~2*alpha*(p-1) — latency grows linearly with p.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    token_out = torch.zeros(1, dtype=torch.uint8)
    token_in = torch.zeros(1, dtype=torch.uint8)
    step = 1
    while step < p:
        _exchange(token_out, (rank + step) % p, token_in, (rank - step) % p)
        step *= 2


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------

def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Linear broadcast: src sends the whole tensor to each other rank in turn.

    After: every rank holds src's data. In-place; returns `tensor`.

    Cost: p-1 messages, all leaving one NIC serially ->
        T = (p-1) * (alpha + N*beta).
    The root is a serial bottleneck: bytes-on-wire is (p-1)*N and *all* of it
    crosses the root's single link. This is why real libraries use trees or
    pipelines (see broadcast_tree below, and ch07 "why NCCL switches
    algorithms by size").

    LLM use: DDP broadcasts the initial parameters from rank 0 exactly once,
    so every replica starts identical (ch08/ch09).
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    if rank == src:
        for dst in range(p):
            if dst != src:
                dist.send(tensor, dst)        # blocking is fine: receiver is posted below
    else:
        dist.recv(tensor, src)
    return tensor


def broadcast_tree(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Binomial-tree broadcast: the set of ranks that have the data doubles
    every round, so it finishes in ceil(log2 p) rounds.

        round 0:  src -> src+1                (1 sender)
        round 1:  {src, src+1} -> +2          (2 senders in parallel)
        round 2:  {src..src+3} -> +4          (4 senders in parallel)

    Cost along the critical path: ceil(log2 p) * (alpha + N*beta).
    Same total bytes as linear broadcast, but the sends happen *in parallel*
    on different links, so latency drops from O(p) to O(log p). For large N a
    pipelined ring beats both (each byte crosses each link once, ~N*beta
    regardless of p) — the small-vs-large message trade-off ch07 derives.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    vr = (rank - src) % p                     # virtual rank: src becomes 0
    mask = 1
    while mask < p:
        if vr < mask:                         # I already have the data
            peer = vr + mask
            if peer < p:
                dist.send(tensor, (peer + src) % p)
        elif vr < 2 * mask:                   # my turn to receive this round
            dist.recv(tensor, (vr - mask + src) % p)
        mask *= 2
    return tensor


# ---------------------------------------------------------------------------
# reduce / gather / scatter
# ---------------------------------------------------------------------------

def reduce(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    """Linear SUM-reduce to `dst`: every other rank sends; dst accumulates.

    After: dst holds the elementwise sum over all ranks; other ranks'
    tensors are unchanged. In-place on dst; returns `tensor`.

    We accumulate in increasing rank order on purpose: floating-point + is
    not associative, and a *deterministic* order makes runs reproducible
    (torch's reductions do not promise this across algorithms — ch08 caveat).

    Cost: p-1 messages into one NIC -> (p-1) * (alpha + N*beta), plus
    (p-1) local adds. Same root bottleneck as linear broadcast.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    if rank == dst:
        buf = torch.empty_like(tensor)
        for src_rank in range(p):             # fixed order => deterministic sum
            if src_rank != dst:
                dist.recv(buf, src_rank)
                tensor.add_(buf)
    else:
        dist.send(tensor, dst)
    return tensor


def gather(tensor: torch.Tensor, dst: int = 0) -> Optional[list[torch.Tensor]]:
    """Gather: dst ends with [tensor_from_rank_0, ..., tensor_from_rank_{p-1}].

    Returns the list on dst, None elsewhere.

    Cost: p-1 messages -> (p-1) * (alpha + N*beta); dst receives (p-1)*N bytes.
    LLM use: collecting metrics/losses to rank 0 for logging; checkpoint
    consolidation (gather shards to one writer, ch19).
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    if rank == dst:
        out: list[torch.Tensor] = []
        for src_rank in range(p):
            if src_rank == dst:
                out.append(tensor.clone())
            else:
                buf = torch.empty_like(tensor)
                dist.recv(buf, src_rank)
                out.append(buf)
        return out
    dist.send(tensor, dst)
    return None


def scatter(out: torch.Tensor, chunks: Optional[list[torch.Tensor]],
            src: int = 0) -> torch.Tensor:
    """Scatter: src holds p chunks; rank i ends with chunks[i] in `out`.

    The inverse of gather. `chunks` is required on src, ignored elsewhere.

    Cost: p-1 messages of n bytes each (n = one chunk) -> (p-1)*(alpha+n*beta)
    serialized at src's NIC. Total bytes (p-1)*n — note each rank receives
    only its 1/p share, unlike broadcast.
    LLM use: rank 0 shards a dataset index or a checkpoint across workers.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    if rank == src:
        assert chunks is not None and len(chunks) == p, "src must supply p chunks"
        for dst_rank in range(p):
            if dst_rank == src:
                out.copy_(chunks[dst_rank])
            else:
                dist.send(chunks[dst_rank].contiguous(), dst_rank)
    else:
        dist.recv(out, src)
    return out


# ---------------------------------------------------------------------------
# all_gather / reduce_scatter — the two halves of ring all-reduce
# ---------------------------------------------------------------------------

def all_gather(out: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Ring all-gather: every rank ends with every rank's tensor.

    `out` must have shape (p, *tensor.shape); after the call, out[i] holds
    rank i's tensor, on every rank.

    Ring algorithm: put your own tensor in out[rank], then for p-1 steps pass
    the chunk you most recently received to your right neighbour while
    receiving a new one from the left. Each chunk travels the full ring.

    Cost per rank: p-1 messages of n bytes (n = one tensor) ->
        T = (p-1) * (alpha + n*beta) = (p-1)*alpha + (p-1)/p * N * beta
    where N = p*n is the total output. Every link carries each byte exactly
    once and all p links work in parallel: bandwidth-optimal.

    LLM use: ZeRO-3/FSDP re-materializes full parameters from shards with an
    all-gather before each layer's forward and backward (ch15/ch16).
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    assert out.shape[0] == p and out.shape[1:] == tensor.shape
    right, left = (rank + 1) % p, (rank - 1) % p
    out[rank].copy_(tensor)
    for step in range(p - 1):
        send_idx = (rank - step) % p          # chunk I received last step (or my own)
        recv_idx = (rank - step - 1) % p      # chunk arriving from the left
        _exchange(out[send_idx], right, out[recv_idx], left)
    return out


def reduce_scatter(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Ring reduce-scatter: elementwise-sum p tensors, leave rank r chunk r.

    `inp` has shape (p, *out.shape): inp[i] is the chunk destined (after
    reduction) for rank i. After the call, out on rank r holds
    sum over all ranks of their inp[r]. Matches torch's reduce_scatter.

    Ring algorithm: for p-1 steps, send one chunk right, receive one from the
    left, and *add it into your copy*. Schedule chosen so chunk k starts at
    rank k+1 and ends, fully accumulated, at rank k:

        step s: send chunk (r-1-s) mod p, recv+add chunk (r-2-s) mod p.

    Cost per rank: p-1 messages of n bytes (n = one chunk) ->
        T = (p-1)*alpha + (p-1)/p * N * beta,   N = p*n total input,
    plus p-1 local adds. Also bandwidth-optimal: each byte of the reduction
    crosses each link once, and reductions happen *en route* — the work of
    summing is distributed around the ring instead of piling up at a root.

    LLM use: ZeRO-2/FSDP reduce-scatters gradients so each rank keeps only
    the shard its optimizer partition needs (ch15/ch16).
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    assert inp.shape[0] == p and inp.shape[1:] == out.shape
    right, left = (rank + 1) % p, (rank - 1) % p
    acc = inp.clone()                          # never mutate caller's input
    buf = torch.empty_like(out)
    for step in range(p - 1):
        send_idx = (rank - 1 - step) % p
        recv_idx = (rank - 2 - step) % p
        _exchange(acc[send_idx].contiguous(), right, buf, left)
        acc[recv_idx].add_(buf)               # reduce en route
    out.copy_(acc[rank])
    return out


# ---------------------------------------------------------------------------
# all_reduce — naive and ring
# ---------------------------------------------------------------------------

def all_reduce_naive(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce as reduce-to-root + broadcast. Correct, simple, slow.

    Cost: 2(p-1) sequential messages of N bytes through rank 0's NIC:
        T = 2*(p-1) * (alpha + N*beta).
    Bytes across the busiest link: 2*(p-1)*N — the root touches everything
    twice while p-1 NICs sit idle. Compare ring all-reduce: 2*(p-1)/p * N
    per rank, perfectly balanced. The ratio is exactly p: naive all-reduce
    gets *slower* per byte as you add workers.
    """
    reduce(tensor, dst=0)
    broadcast(tensor, src=0)
    return tensor


def all_reduce_ring(tensor: torch.Tensor) -> torch.Tensor:
    """Bandwidth-optimal ring all-reduce: reduce-scatter phase + all-gather
    phase, 2(p-1) steps total. In-place SUM; returns `tensor`.

    Phase 1 (p-1 steps): ring reduce-scatter. Afterwards rank r holds the
    fully reduced chunk r (1/p of the tensor).
    Phase 2 (p-1 steps): ring all-gather of those reduced chunks. Afterwards
    every rank holds the full reduced tensor.

    Cost per rank (derived step by step in ch07):
        messages : 2(p-1)
        bytes    : 2(p-1) * N/p  = 2N(p-1)/p   -> 2N as p -> inf
        T        = 2(p-1)*alpha + 2(p-1)/p * N*beta

    The beta term is within 2x of the lower bound for *any* all-reduce
    (every rank must send >= N(p-1)/p bytes of its unique data and receive as
    much) — this algorithm meets the bound, hence "bandwidth-optimal". The
    alpha term is 2(p-1) though: for tiny tensors, latency dominates and a
    tree (O(log p) messages) wins. NCCL picks per message size (ch07).

    Handles any tensor size: pads the flat view to a multiple of p, unpads
    at the end. Works on views of the caller's storage where possible.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    flat = tensor.reshape(-1)
    n = flat.numel()
    pad = (-n) % p
    work = torch.cat([flat, flat.new_zeros(pad)]) if pad else flat.clone()
    chunks = work.view(p, -1)                  # p equal chunks
    right, left = (rank + 1) % p, (rank - 1) % p
    buf = torch.empty_like(chunks[0])

    # --- Phase 1: reduce-scatter (p-1 steps). Chunk k ends reduced at rank k.
    for step in range(p - 1):
        send_idx = (rank - 1 - step) % p
        recv_idx = (rank - 2 - step) % p
        _exchange(chunks[send_idx], right, buf, left)
        chunks[recv_idx].add_(buf)

    # --- Phase 2: all-gather (p-1 steps). Pass reduced chunks around.
    for step in range(p - 1):
        send_idx = (rank - step) % p           # start with my reduced chunk r
        recv_idx = (rank - 1 - step) % p
        _exchange(chunks[send_idx], right, chunks[recv_idx], left)

    flat.copy_(work[:n])
    return tensor


# ---------------------------------------------------------------------------
# correctness helpers
# ---------------------------------------------------------------------------
# The checkers deliberately use torch.distributed's OWN collectives, so a bug
# in this module cannot hide itself by also being in the checker.

def assert_all_ranks_equal(tensor: torch.Tensor, atol: float = 0.0,
                           msg: str = "") -> None:
    """Assert every rank holds the same tensor (post-broadcast invariant)."""
    p = dist.get_world_size()
    gathered = [torch.empty_like(tensor) for _ in range(p)]
    dist.all_gather(gathered, tensor)
    for i, g in enumerate(gathered):
        if not torch.allclose(gathered[0], g, atol=atol, rtol=0):
            raise AssertionError(
                f"{msg} rank 0 vs rank {i} differ by "
                f"{(gathered[0] - g).abs().max().item():.3e}")


def assert_matches_torch_allreduce(ours: torch.Tensor,
                                   original: torch.Tensor,
                                   atol: float = 1e-5) -> None:
    """Assert `ours` equals dist.all_reduce(original) within atol.

    atol, not exact: our accumulation order (ring hops) differs from the
    backend's, and float addition is order-sensitive. Bound the error, don't
    demand bit equality — a lesson that returns in ch08.
    """
    ref = original.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    if not torch.allclose(ours, ref, atol=atol, rtol=1e-5):
        raise AssertionError(
            f"manual all_reduce differs from torch by "
            f"{(ours - ref).abs().max().item():.3e} (atol={atol})")

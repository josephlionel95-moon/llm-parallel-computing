"""Manual parallelism strategies, built before importing the official ones.

ch08: manual data parallelism (broadcast params, all-reduce grads).
ch09: manual DDP (hooks, buckets, async overlap) — a working miniature of
      torch.nn.parallel.DistributedDataParallel.
"""
from .manual_dp import (  # noqa: F401
    average_gradients,
    broadcast_parameters,
    shard_indices,
)
from .manual_ddp import ManualDDP  # noqa: F401

"""Manual collective communication built from point-to-point primitives.

Chapter 07 builds every collective (broadcast, reduce, all_reduce, scatter,
gather, all_gather, reduce_scatter) from nothing but send/recv, so that the
cost model of each is derived, not asserted.
"""
from .manual_collectives import (  # noqa: F401
    all_gather,
    all_reduce_naive,
    all_reduce_ring,
    assert_all_ranks_equal,
    assert_matches_torch_allreduce,
    barrier,
    broadcast,
    broadcast_tree,
    gather,
    reduce,
    reduce_scatter,
    scatter,
)

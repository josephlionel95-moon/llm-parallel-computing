"""Spawn-target worker functions for the notebooks.

torch.multiprocessing.spawn pickles workers by module path, so functions
defined inside a notebook kernel (module __main__) cannot be spawned — the
child process has no __main__ to re-import them from. Every multi-process
demo used by a notebook therefore lives here, in an importable module, and
the notebook just calls llmdist.utils.dist.run_distributed(worker, ...).

Workers cannot return values across processes; those that produce
measurements torch.save() a results dict from rank 0 to a caller-supplied
path, which the notebook then loads and plots.
"""

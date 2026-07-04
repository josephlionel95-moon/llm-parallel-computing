# Distributed Setup — One-Page Cheat Sheet

## torchrun

```bash
# single node, N processes (the 95% case)
torchrun --standalone --nproc_per_node=2 script.py [script args]

# single node, explicit rendezvous (when --standalone port-picking is unwanted)
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 script.py

# multi-node: SAME command on every node, only --node_rank differs
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
         --rdzv_backend=c10d --rdzv_endpoint=NODE0_IP:29400 script.py   # on node 0
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
         --rdzv_backend=c10d --rdzv_endpoint=NODE0_IP:29400 script.py   # on node 1

# elastic: restart the whole group up to 3 times on failure
torchrun --standalone --nproc_per_node=4 --max-restarts=3 script.py
```

In notebooks (torchrun can't run a cell):
```python
from llmdist.utils.dist import run_distributed
run_distributed(fn, world_size=2, backend="gloo")   # fn(rank, world_size, *args)
```

## Environment variables (set by torchrun, read by your script)

| Var | Meaning | Typical use |
|-----|---------|-------------|
| `RANK` | global process id, 0..W−1 | algorithm logic (`if rank == 0:`) |
| `LOCAL_RANK` | id within this node | **GPU binding**: `cuda:{LOCAL_RANK}` |
| `WORLD_SIZE` | total processes | loop bounds, averaging divisors |
| `LOCAL_WORLD_SIZE` | processes on this node | per-node resources |
| `MASTER_ADDR` | rank 0's host | rendezvous phone number |
| `MASTER_PORT` | rank 0's TCPStore port | rendezvous phone number |

`RANK = node_rank × nproc_per_node + LOCAL_RANK`

## Boilerplate that prevents the classic bugs

```python
rank       = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)              # BEFORE tensors, or all ranks hit GPU 0
dist.init_process_group("nccl",                # gloo if CPU
                        timeout=datetime.timedelta(seconds=120))  # not 30 min
...
dist.destroy_process_group()                   # in finally: — leaked groups hold the port
```
(or just: `from llmdist.utils.dist import setup_from_env, cleanup`)

## Backend chooser

| Situation | Backend |
|-----------|---------|
| GPU tensors, ≥1 CUDA device per rank | **nccl** (always, for training) |
| CPU tensors / no GPU / CI / notebook simulation | **gloo** |
| cluster launched by `mpirun`, torch built with MPI | mpi (you'd know) |
| mixed (GPU training + occasional CPU control tensors) | nccl + a second gloo group |

Iron rule: **NCCL ⇔ CUDA tensors, Gloo ⇔ CPU tensors.** Mismatch = instant error.

## Debugging flags

| Flag | Effect | Reach for it when |
|------|--------|-------------------|
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | log every collective per rank (shapes, order) | hang: find which rank diverged |
| `NCCL_DEBUG=INFO` | topology found, rings/trees built, transport per channel | verify NVLink vs PCIe; init failures |
| `NCCL_DEBUG_SUBSYS=INIT,NET` | filter the above | log too noisy |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | NCCL hang → error at timeout | jobs that freeze silently |
| `TORCH_NCCL_BLOCKING_WAIT=1` | synchronous waits, precise stacks (slow) | pinpointing the stuck collective |
| `NCCL_SOCKET_IFNAME=eth0` | pick the NIC | multi-node picks the wrong interface |
| `NCCL_P2P_DISABLE=1` | force staging via host | suspect broken P2P/IOMMU |
| `py-spy dump --pid P` | live stack of a stuck rank | the ground truth for any hang |

## Failure modes — 10-second diagnosis

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| freeze in init, then timeout | launched ≠ WORLD_SIZE processes | match counts; read the dead rank's traceback |
| `address already in use` | zombie run holds MASTER_PORT | kill it / new port / `--standalone` |
| `Tensors must be CUDA and dense` | CPU tensor on NCCL | `.to(device)` or gloo |
| GPU 0 OOM, others idle / "Duplicate GPU" | missing `set_device(LOCAL_RANK)` | add the line |
| fine 1-node, `invalid device ordinal` 2-node | `cuda:{RANK}` not `cuda:{LOCAL_RANK}` | use LOCAL_RANK |
| hang mid-training, GPUs 100% busy | a rank skipped a collective (rank-conditional code, uneven data) | every rank, every collective, same order |
| node 1 can't connect | firewall / wrong MASTER_ADDR / wrong NIC | `nc -zv addr port`; NCCL_SOCKET_IFNAME |

## Free-tier notes

- **Kaggle 2× T4**: real NCCL; PCIe only (no NVLink — check `nvidia-smi topo -m`).
- **Colab 1× T4**: simulate with `run_distributed(fn, 2, backend="gloo")` — identical
  semantics, meaningless speeds.
- First collective per group is slow (lazy setup): **always warm up before timing.**

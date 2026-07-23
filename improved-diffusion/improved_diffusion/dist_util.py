"""
Helpers for distributed training.

Uses torchrun-style environment variables (``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``) instead of MPI.

* Single-GPU: ``python scripts/train.py ...``
* Multi-GPU/multi-node: ``torchrun --nproc_per_node=N scripts/train.py ...``
"""

import io
import os

import torch as th
import torch.distributed as dist


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def setup_dist():
    """
    Initialize ``torch.distributed`` from torchrun env vars. For single-process
    runs the group is still initialized (``world_size=1``) so that callers can
    freely use ``dist.get_rank()`` / ``dist.get_world_size()``.
    """
    if dist.is_initialized():
        return

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if th.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for this process.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{_local_rank() % max(th.cuda.device_count(), 1)}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch checkpoint. Rank 0 reads from disk and broadcasts to the
    other ranks so that shared storage is not hammered by every process.
    """
    if _world_size() <= 1 or not dist.is_initialized():
        return th.load(path, **kwargs)

    if _rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
        payload = [data]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    return th.load(io.BytesIO(payload[0]), **kwargs)


def sync_params(params):
    """
    Broadcast a sequence of Tensors from rank 0 to every other rank.
    No-op in single-process runs.
    """
    if _world_size() <= 1 or not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

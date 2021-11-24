import os
import resource
import subprocess

import torch
from torch import distributed as dist


def reduce_tensor_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def configure_nccl():
    """Configure multi-machine environment variables.

    It is required for multi-machine training.
    """
    # os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
    os.environ["NCCL_IB_DISABLE"] = "1"

    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))


def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()


def all_gather(x, dim=0):
    """collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype) for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=dim)


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_distributed() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

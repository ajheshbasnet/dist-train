import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def init_process():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    global_rank = int(os.environ['RANK'])

    if torch.cuda.is_available():
        # === For GPU ===
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        return device, global_rank, world_size

    else:
        # === For CPU ===
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=global_rank, world_size=world_size)
        return device, global_rank, world_size


def get_parallel_dims(world_size: int):
    """
    TP and PP sizes come from env vars (defaults: TP_SIZE=2, PP_SIZE=2).
    DP size is whatever is left: world_size // (TP * PP).
      4 processes -> tp=2, pp=2, dp=1
      8 processes -> tp=2, pp=2, dp=2
    """
    tp_dim = int(os.environ.get('TP_SIZE', 2))
    pp_dim = int(os.environ.get('PP_SIZE', 2))
    assert world_size % (tp_dim * pp_dim) == 0, (
        f"world_size={world_size} must be divisible by TP_SIZE*PP_SIZE={tp_dim * pp_dim}"
    )
    dp_dim = world_size // (tp_dim * pp_dim)
    return tp_dim, pp_dim, dp_dim


def create_device_mesh(tp_dim: int, pp_dim: int, dp_dim: int, world_size):
    assert tp_dim * pp_dim * dp_dim == world_size, "Device mesh dimensions must match world size"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # Standard ordering: PP outermost, TP innermost (TP ranks are adjacent -> same node / NVLink).
    return init_device_mesh(device_type, (pp_dim, dp_dim, tp_dim), mesh_dim_names=("pp", "dp", "tp"))

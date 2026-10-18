import os
import torch
import torch.distributed as dist

def init_process():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    global_rank = int(os.environ['RANK'])

    if torch.cuda.is_available():
        # === For GPU ===
        device = torch.device(f'cuda:{local_rank}')
        torch.set_device(device)
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        return device, global_rank, world_size        

    else:
        # === For CPU ===
        device = torch.device(f'cpu')
        dist.init_process_group("gloo", rank=global_rank, world_size=world_size)
        return device, global_rank, world_size


def create_device_mesh(tp_dim: int, pp_dim: int, dp_dim: int, world_size):
    assert tp_dim * pp_dim * dp_dim == world_size, "Device mesh dimensions must match world size"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return dist.init_device_mesh(device_type, (tp_dim, pp_dim, dp_dim), mesh_dim_names=("tp", "pp", "dp"))
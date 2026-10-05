import torch
import torch.distributed as dist
import os

def init_process():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Use gloo backend for Windows compatibility
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        return rank, world_size, device

    if torch.cpu.is_available():
        device = torch.device('cpu')
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        return rank, world_size, device
    
    raise ValueError("No available device found")
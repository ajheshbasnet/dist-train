import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from colorama import Fore, Style

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

from transformer import Transformer
from dataloader import return_dataset
from configs import configs
from torch.utils.data import DataLoader

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
global_rank = int(os.environ['RANK'])

print(Fore.CYAN + f"Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}" + Style.RESET_ALL)

if torch.cuda.is_available():
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(
        "nccl",
        rank=global_rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)

else:
    device = torch.device('cpu')
    dist.init_process_group(
        "gloo",
        rank=global_rank,
        world_size=world_size
    )

model = Transformer(configs()).to(device)

is_first = global_rank == 0
is_last = global_rank == world_size-1

if not is_first:
    model.token_embedding = None

if not is_last:
    model.layernorm = None
    model.output_proj = None

layer_per_stage = configs().n_layers // world_size

model.layers = nn.ModuleList(model.layers[global_rank * layer_per_stage : (global_rank + 1) * layer_per_stage])

#  == setting up GPIPE ==

from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

stage = PipelineStage(
    model, stage_index = global_rank, num_stages = world_size, device = device
)

n_micro_batches = 2

def loss_function(logits, targets):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

schedule = ScheduleGPipe(stage, n_micro_batches, loss_fn = loss_function)

optimizers = torch.optim.Adam(model.parameters(), lr=configs.lr)

dataset, _ = return_dataset()
dataload = DataLoader(dataset, batch_size=configs.batch_size * n_micro_batches, shuffle=True, drop_last=True)

for batches in dataload:
    input_ids, targets, attn_mask = batches['inputs'], batches['targets'], batches['attn_mask']
    input_ids, targets, attn_mask = input_ids.to(device), targets.to(device), attn_mask.to(device)


    optimizers.zero_grad()

    if is_first:
        schedule.step(input_ids, attn_mask=attn_mask)

    elif is_last:
        all_losses = []
        schedule.step(target=targets, losses=all_losses, attn_mask=attn_mask)

    else:
        schedule.step(attn_mask=attn_mask)

    optimizers.step()    

    if is_last:
        loss = torch.stack(all_losses).mean().item()
        print(Fore.GREEN + f"Loss: {loss:.4f}" + Style.RESET_ALL)

import os
import sys
import torch
import torch.nn as nn
from colorama import Fore, Style
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

from transformer import Transformer
from configs import configs
from communication import init_process
from dataloader import return_dataset

rank, world_size, device = init_process()

print(f"{Fore.GREEN}Rank: {rank}, World Size: {world_size}, Device: {device}{Style.RESET_ALL}")

model = Transformer(configs())

layer_per_device = configs.n_layers // world_size

is_first = rank == 0
is_last = rank == world_size - 1

if not is_first:
    model.token_embedding = None

if not is_last:
    model.layernorm = None
    model.output_proj = None

model.layers = nn.ModuleList(
    list(model.layers)[rank * layer_per_device:(rank + 1) * layer_per_device]
)

model = model.to(device)

# == stage + schedule ==

from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

stage = PipelineStage(
    model, stage_index=rank, num_stages=world_size, device=device
)

n_micro_batches = 4

def loss_function(logits, target):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

schedule = ScheduleGPipe(
    stage, n_microbatches=n_micro_batches, loss_fn=loss_function
)

optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

dataset, _ = return_dataset()

g = torch.Generator().manual_seed(42)
dataload = DataLoader(dataset, batch_size=configs.batch_size * n_micro_batches,
                      shuffle=True, generator=g, drop_last=True)
for batches in dataload:
    input_ids, targets, attn_mask = batches['inputs'], batches['targets'], batches['attn_mask']
    input_ids, targets, attn_mask = input_ids.to(device), targets.to(device), attn_mask.to(device)

    optimizer.zero_grad()

    if is_first:
        schedule.step(input_ids)

    elif is_last:
        losses = []
        schedule.step(target=targets, losses=losses)

    else:
        schedule.step()

    optimizer.step()
    
    if is_last:
        loss = torch.stack(losses).mean()
        print(f"Loss: {loss}")
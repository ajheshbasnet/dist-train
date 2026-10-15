import os
import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

from transformer import Transformer
from dataloader import return_dataset
from configs import configs

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
global_rank = int(os.environ['RANK'])

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
    model.layer_norm = None
    model.output_proj = None

layer_per_stage = configs().n_layers // world_size

model.layers = nn.ModuleList(model.layers[global_rank * layer_per_stage : (global_rank + 1) * layer_per_stage])

#  == setting up GPIPE ==

from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

stage = PipelineStage(
    model, stage_rank = global_rank, num_stages = world_size, device = device
)

n_micro_batches = 2

def loss_function(logits, targets):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

schedule = ScheduleGPipe(stage, n_micro_batches, lossfn = loss_function)

optimizers = torch.optim.Adam(model.parameters(), lr=configs.lr)

dataset, _ = return_dataset()
dataload = DataLoader(dataset, batch_size=configs.batch_size * n_micro_batches, shuffle=True, drop_last=True)

for batches in dataload:
    input_ids, targets, attn_mask = batches['inputs'], batches['targets'], batches['attn_mask']
    input_ids, targets, attn_mask = input_ids.to(device), targets.to(device), attn_mask.to(device)


    optimizers.zero_grad()

    if is_first:
        schedule.step(input_ids)

    elif is_last:
        schedule.step(target=targets, losses=[])    

    else:
        schedule.step()

    optimizers.step()    

    if is_last:
        loss = schedule.loss()
        print(f"Loss: {loss}")

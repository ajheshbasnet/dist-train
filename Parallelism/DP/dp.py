import torch
import torch.nn as nn
from configs import configs
from transformer import Transformer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

import torch
import torch.distributed as dist
from transformer import Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import return_dataset
from configs import configs

from torch.utils.data import DataLoader, DistributedSampler

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
global_rank = int(os.environ['RANK'])

# == setting up backend ==

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

elif torch.cpu.is_available():
    device = torch.device(f'cpu')
    dist.init_process_group(backend="gloo", rank=global_rank, world_size=world_size)

else:
    raise ValueError("No available device found")

# == loading the dataset == 
dataset, _ = return_dataset()

# == setting up for DDP == 
'''
1. Define the model object and load to device
2. Wrap the model with DDP
3. Create the sampler
4. Create the dataloader
'''

model = Transformer(configs()).to(device)
if device.type=='cuda':
  model = DDP(model, device_ids=[local_rank]) # replicating weights the model across the GPUs and assiging the ranks
else:
  model = DDP(model)  

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
dataload = DataLoader(dataset, batch_size=configs.batch_size, sampler=sampler)


# == define optimizers for the model == 
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # will change this later.

# == training loop ==

for epoch in range(configs().epochs):
    for batch in dataload:
        input_ids = batch['inputs']
        attention_mask = batch['attn_mask']
        target_ids = batch['targets']
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        average_loss = loss.clone()
        dist.reduce(average_loss, op=dist.ReduceOp.SUM, dst=0) # now rank=0 will get the global average loss
        average_loss /= world_size

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if local_rank==0:
          print(f'Rank: {global_rank} | Epoch: {epoch} | Loss: {average_loss.item() :.2f}')
import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add parent directory for communication
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from communication import init_process

# Add Transformer directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))
from transformer import Transformer
from configs import configs
from dataloader import return_dataset

def main():

    rank, world_size, device = init_process()

    print(f'== Rank: {rank} | World Size: {world_size}\n')
    
    model = Transformer(configs())  
    model = model.to(device)
    # gloo backend doesn't use device_ids
    model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    dataset, _ = return_dataset()

    dataloader = DataLoader(dataset, batch_size=configs.batch_size, pin_memory = True,shuffle=False, sampler=DistributedSampler(dataset))

    for batch in dataloader:
        input_ids, target_ids, attention_mask = batch['inputs'], batch['targets'], batch['attn_mask']
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        loss_tensor = torch.tensor([loss.item()], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Rank: {rank} | Loss: {avg_loss :.2f}')
        

if __name__ == "__main__":
    main()
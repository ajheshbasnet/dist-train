import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import ( 
    ColwiseParallel, 
    RowwiseParallel, 
    SequenceParallel, 
    parallelize_module
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

# from communication import init_process
from transformer import Transformer
from configs import configs
from dataloader import return_dataset
import torch.distributed as dist


def build_device_mesh(tp_dim: int, dp_dim: int):  
    ''' it will automateically handle regarding init_process_group, no need to seperately define the function'''
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_type = device_type,
        mesh_shape = (dp_dim, tp_dim),
        mesh_dim_names = ("dp", "tp")
    )

def apply_tensor_sequence_parallel(model: Transformer, mesh_grid):
    
    # Use 1D TP mesh for tensor parallelism
    tp_mesh = mesh_grid["tp"]
    
    # == for token embedding and position embedding ==

    parallelize_module(model, tp_mesh, {
        "token_embedding.token_embedding": RowwiseParallel(
            input_layouts = Replicate(),
            output_layouts = Shard(1)  # [B, T, C] so shard across Sequence dimension because now it will go for Normalisation
        ),
        "token_embedding.position_embedding": RowwiseParallel(
            input_layouts = Replicate(),
            output_layouts = Shard(1)  # [B, T, C] so shard across Sequence dimension because now it will go for Normalisation
        ),
        "layernorm": SequenceParallel()
    })
    
    # == now going for individual layers == 

    for layer in model.layers:
        
        layer_plan = {
                "attn_norm": SequenceParallel(),
                "attention.q_proj": ColwiseParallel(input_layouts = Replicate(), output_layouts = Shard(-1)),
                "attention.k_proj": ColwiseParallel(input_layouts = Replicate(), output_layouts = Shard(-1)),
                "attention.v_proj": ColwiseParallel(input_layouts = Replicate(), output_layouts = Shard(-1)),
                "attention.out_proj": RowwiseParallel(input_layouts = Shard(-1), output_layouts = Shard(1)),

                "ffnn_norm": SequenceParallel(),
                "ffnn.fc1": ColwiseParallel(input_layouts = Replicate(), output_layouts = Shard(-1)),
                "ffnn.fc2": RowwiseParallel(input_layouts = Shard(-1), output_layouts = Shard(1)),
            }
        
        layer.attention.num_heads = layer.attention.num_heads // tp_mesh.size()
        
        # layer gives the model architct, 
        # mesh_grid says which gpus are allowed to talk to each other to communicate and use all_reduce, all_gather stuffs, 
        # plan is the plan we want to shard.
        
        parallelize_module(layer, tp_mesh, layer_plan)

    return model    

def main():
    tp_dim = 2
    dp_dim = 1
    
    device_mesh = build_device_mesh(tp_dim, dp_dim)
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    print(f"Rank {rank}: Using device {device}")
    torch.manual_seed(42)

    model = Transformer(configs()).to(device)

    model = apply_tensor_sequence_parallel(model, device_mesh)

    optimizer = torch.optim.AdamW(model.parameters(), lr = configs.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    dataset, _ = return_dataset()

    dataloader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    for batch in dataloader:
        input_ids, target_ids, attention_mask = batch['inputs'], batch['targets'], batch['attn_mask']
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)   #[ B, T, VOCABS/2]
        # Convert DTensor to regular tensor for loss computation
        logits = logits.to_local() if hasattr(logits, 'to_local') else logits

        # With tensor parallelism, logits are sharded across ranks
        # Gather full logits from all ranks for loss computation
        if logits.size(-1) < configs.vocab_size:
            # Logits are sharded - gather from all ranks
            gathered_logits = [torch.zeros_like(logits) for _ in range(world_size)]
            dist.all_gather(gathered_logits, logits)
            logits = torch.cat(gathered_logits, dim=-1)

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
    dist.destroy_process_group()
    
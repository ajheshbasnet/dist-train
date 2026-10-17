import torch
import os 
import sys
import torch.distributed as dist
from colorama import Fore, Style
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    RowwiseParallel, ColwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput
)
from torch.distributed.tensor.parallel import loss_parallel

from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

from transformer import Transformer
from dataloader import return_dataset
from configs import configs

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
global_rank = int(os.environ['RANK'])

torch.manual_seed(0)                            # same shuffle on every rank

print(Fore.GREEN + f'Rank: {global_rank} | Local Rank: {local_rank} | World Size: {world_size}' + Style.RESET_ALL)

if torch.cuda.is_available():
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", rank=global_rank, world_size=world_size
    )

else:
    device = torch.device('cpu')
    dist.init_process_group(
        'gloo', rank=global_rank, world_size=world_size
    )

def build_device_mesh(tp_dim: int):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(device_type, (tp_dim,), mesh_dim_names=('tp',))

def apply_tensor_shard(model: Transformer, tp_mesh):

    layout = {
        'token_embedding.token_embedding': RowwiseParallel(input_layouts=Replicate(), output_layouts = Shard(1)),
        'token_embedding.position_embedding': RowwiseParallel(input_layouts=Replicate(), output_layouts = Shard(1)),
        'layernorm': SequenceParallel(),
        'output_proj': ColwiseParallel(
          input_layouts = Shard(1), 
          output_layouts = Shard(-1), 
          use_local_output=False
          )
    }

    parallelize_module(model, tp_mesh, layout)

    for layer in model.layers:
        
        layout = {
            'attn_norm': SequenceParallel(),
            'ffnn_norm': SequenceParallel(),

            'attention': PrepareModuleInput(
                input_layouts=(Shard(1),), 
                desired_input_layouts=(Replicate(),),
                use_local_output = True
            ),

            'attention.q_proj': ColwiseParallel(),
            'attention.k_proj': ColwiseParallel(),
            'attention.v_proj': ColwiseParallel(),
            'attention.out_proj': RowwiseParallel(output_layouts = Shard(1)),   

            'ffnn': PrepareModuleInput(
              input_layouts = (Shard(1),),
              desired_input_layouts = (Replicate(),),
              use_local_output = True
            ),

            'ffnn.fc1': ColwiseParallel(),
            'ffnn.fc2': RowwiseParallel(output_layouts=Shard(1))

        }

        layer.attention.num_heads = configs.n_heads // tp_mesh.size()

        parallelize_module(layer, tp_mesh, layout)
    
    return model


tp_dim = 4
mesh_grid = build_device_mesh(tp_dim)

model = Transformer(configs()).to(device)

model = apply_tensor_shard(model, mesh_grid['tp'])
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

dataset, _ = return_dataset()
dataload = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

assert configs().vocab_size % tp_dim==0, "vocab size should be divisible by tp_dim to shard the logits_head to each rank"

for batch in dataload:
    input_ids, target_ids, attention_mask = batch['inputs'], batch['targets'], batch['attn_mask']
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    attention_mask = attention_mask.to(device)
    logits = model(input_ids, attention_mask)

    # logits = logits.to_local() if hasattr(logits, 'to_local') else logits

    optimizer.zero_grad()

    with loss_parallel():

      loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
      loss.backward()                             # backward must be inside the context

    optimizer.step()

    loss_val = loss.full_tensor().item() if hasattr(loss, 'full_tensor') else loss.item()

    if global_rank==0:

      print(Fore.RED+f"Loss: {loss_val.item():.4f}")
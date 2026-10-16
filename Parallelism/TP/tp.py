import torch
import torch.distributed as dist
from colorama import Fore, Style
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    RowwiseParallel, ColwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput
)

from torch.distributed.device_mesh import init_device_mesh

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Transformer'))

from transformer import Transformer
from dataloader import return_dataset
from configs import configs

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
global_rank = int(os.environ['RANK'])

print(Fore.GREEN + f'Rank: {global_rank} | Local Rank: {local_rank} | World Size: {world_size}' + Style.RESET_ALL)

if torch.cuda.is_available():
    device = torch.device(f'cuda:{local_rank}')
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
        'layers': SequenceParallel(),
        'output_proj': ColwiseParallel(input_layouts = Replicate(), output_layouts = Replicate())
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

        layer.num_heads = configs.n_heads // tp_mesh.size()

        parallelize_module(layer, tp_mesh, layout)
    
    return model


tp_dim = 4
mesh_grid = build_device_mesh(tp_dim)

model = apply_tensor_shard(model, mesh_grid['tp'])
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

dataset, _ = return_dataset(configs.batch_size, configs.seq_len)

for batch in dataset:
    input_ids, targets = batch
    input_ids, targets = input_ids.to(device), targets.to(device)
    logits = model(input_ids)

    if global_rank==0:

        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        optimizer.zero_grad()

        loss.backward()        

        optimizer.step()
        
        print(f"Loss: {loss.item()}")
        
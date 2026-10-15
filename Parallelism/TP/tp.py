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
        'output_proj': ColwiseParallel(input_layouts = Replicate(), output_layouts = Shard(1))
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
            'attention.q_proj': ColwiseParallel(output_layouts = Shard(2)),
            'attention.k_proj': ColwiseParallel(output_layouts = Shard(2)),
            'attention.v_proj': ColwiseParallel(output_layouts = Shard(2)),
            'attention.out_proj': RowwiseParallel( 
                input_layouts = 
                output_layouts = Shard(1)),
        
        }

        parallelize_module(layer, tp_mesh, layout)
        return model



    
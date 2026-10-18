import os, sys
import torch
import torch.nn as nn
import torch.distributed as dist  
from utils import init_process, create_device_mesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Transformer'))
from Transformer import Transformer
from dataloader import return_dataset
from configs import configs
from colorama import Fore, Style

# === For DP ===
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.data import DistributedSampler, DataLoader

# === For TP ===
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    RowwiseParallel, ColwiseParallel,   
    SequenceParallel, parallelize_module, 
    PrepareModuleInput, loss_parallel
)

# === For PP ===
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

# === Initializing Backend ===
device, global_rank, world_size = init_process()

# === Creating Device Mesh ===
mesh = create_device_mesh(tp_dim=2, pp_dim=2, dp_dim=2)
n_micro_batches = 4

# === Creating TP ===
def create_tp_model(model:Transformer, tp_mesh, is_first, is_last):

    layouts = {}

    if is_first:
        layouts['token_embedding.token_embedding'] = RowwiseParallel(output_layouts=Shard(1))
        layouts['token_embedding.position_embedding'] = RowwiseParallel(output_layouts=Shard(1))

    if is_last:
        layouts['layernorm'] = SequenceParallel()
        layouts['output_proj'] = ColwiseParallel(
            input_layouts = Shard(1),
            output_layouts=Shard(-1),  # because the [B, T, vocab_size] dimension is way too huge so shard across the rank in that dimension.
            use_local_output=False)
        
    if layouts:
        parallelize_module(model, tp_mesh, layouts)

    for layer in model.layers:
        format = {
            "attn_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts = (Shard(1), None),
                desired_input_layouts = (Replicate(), None),
                use_local_output = True
                ),
            "attention.q_proj": ColwiseParallel(),
            "attention.k_proj": ColwiseParallel(),
            "attention.v_proj": ColwiseParallel(),
            "attention.o_proj": RowwiseParallel(output_layouts = Shard(1)),

            "ffnn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts = (Shard(1),),
                desired_input_layouts = (Replicate(),),
                use_local_output = True
                ),
            "feed_forward.fc1": ColwiseParallel(),
            "feed_forward.fc2": RowwiseParallel(output_layouts=Shard(1)),
        }
        layer.attention.num_heads = layer.attention.num_heads // tp_mesh.size()
        parallelize_module(layer, tp_mesh, format)

    return model

def apply_fsdp(model: Transformer, dp_mesh):
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward = False)
    fully_shard(model, mesh=dp_mesh, reshard_after_forward = False)
    return model    

def loss_function(output, target):
    return nn.CrossEntropyLoss()(output.reshape(-1, output.size(-1)), target.reshape(-1))
    
def main():
    torch.manual_seed(1234)          
    model = Transformer(configs()).to(device)
    tp_mesh, pp_mesh, dp_mesh = mesh['tp'], mesh['pp'], mesh['dp']

    # === Creating PP ===
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    is_first = pp_rank==0
    is_last  = pp_rank==pp_size-1

    if not is_first:
        model.token_embedding = None
    if not is_last:
        model.layernorm = None
        model.output_proj = None

    layers_per_stage = len(model.layers) // pp_size
    model.layers = nn.ModuleList(model.layers[pp_rank * layers_per_stage : (pp_rank+1)*layers_per_stage])
    
    # === Creating TP ===
    model = create_tp_model(model, tp_mesh, is_first, is_last)
    
    # === Creating DP ===
    model = apply_fsdp(model, dp_mesh)

    dataset, _ = return_dataset()
    sampler = DistributedSampler(dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), drop_last=True)
    dataload = DataLoader(dataset, batch_size=configs().batch_size * n_micro_batches, sampler=sampler, drop_last=True)


    stage = PipelineStage(model, stage_index=pp_rank, num_stages=pp_size, device=device, group=pp_mesh.get_group())

    schedule = ScheduleGPipe(stage, n_micro_batches, loss_fn = loss_function)    

    optimizer = torch.optim.AdamW(model.parameters(), lr = configs().lr)

    for epoch in range(configs().epoch):

        sampler.set_epoch(epoch)

        for batches in dataload:
            input_ids, targets, attn_mask = batches['inputs'], batches['targets'], batches['attn_mask']
            input_ids, targets, attn_mask = input_ids.to(device), targets.to(device), attn_mask.to(device)
                        
            all_loss = []
            optimizer.zero_grad()

            with loss_parallel():
                if is_first:
                    schedule.step(input_ids, attn_mask=attn_mask)
                elif is_last:
                    schedule.step(target=targets, attn_mask=attn_mask, losses=all_loss)    
                else:
                    schedule.step(attn_mask=attn_mask)    

            optimizer.step()    

        if is_last:
            loss_val = torch.stack(all_loss).mean().item()
            print(Fore.LIGHTBLACK_EX + f"Epoch {epoch}: Loss = {loss_val :.4f}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()

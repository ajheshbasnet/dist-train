import os, sys
import torch
import torch.nn as nn
import torch.distributed as dist
from utils import init_process, create_device_mesh, get_parallel_dims
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Transformer'))
from transformer import Transformer
from dataloader import return_dataset
from configs import configs
from colorama import Fore, Style
from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing import cosine_annealing

# === For DP ===
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DistributedSampler, DataLoader

# === For TP ===
from torch.distributed.tensor import Shard, Replicate, DTensor
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
# Sizes come from env vars TP_SIZE / PP_SIZE (default 2 / 2); DP = world_size // (TP*PP).
TP_DIM, PP_DIM, DP_DIM = get_parallel_dims(world_size)
mesh = create_device_mesh(tp_dim=TP_DIM, pp_dim=PP_DIM, dp_dim=DP_DIM, world_size=world_size)
n_micro_batches = int(os.environ.get('N_MICRO_BATCHES', 4))

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
SAVE_EVERY_EPOCHS = 10

cfg = configs()

def validate_parallel_config(cfg, tp, pp):
    assert cfg.n_heads % tp == 0, f"n_heads={cfg.n_heads} must be divisible by TP={tp}"
    assert cfg.hidden_dim % tp == 0, f"hidden_dim={cfg.hidden_dim} must be divisible by TP={tp}"
    assert cfg.max_seq_len % tp == 0, f"max_seq_len={cfg.max_seq_len} must be divisible by TP={tp} (sequence parallel)"
    assert cfg.vocab_size % tp == 0, f"vocab_size={cfg.vocab_size} must be divisible by TP={tp}"
    assert cfg.n_layers % pp == 0, f"n_layers={cfg.n_layers} must be divisible by PP={pp}"
    assert n_micro_batches >= pp, f"N_MICRO_BATCHES={n_micro_batches} should be >= PP={pp}"

# === Creating TP ===
def create_tp_model(model:Transformer, tp_mesh, is_first, is_last):

    layouts = {}

    if is_first:
        layouts['token_embedding.token_embedding'] = RowwiseParallel(input_layouts = Replicate(), output_layouts=Shard(1))
        layouts['token_embedding.position_embedding'] = RowwiseParallel(input_layouts = Replicate(), output_layouts=Shard(1))

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
                input_layouts = (Shard(1), ),
                desired_input_layouts = (Replicate(),),
                use_local_output = True
                ),
            "attention.q_proj": ColwiseParallel(),
            "attention.k_proj": ColwiseParallel(),
            "attention.v_proj": ColwiseParallel(),
            "attention.out_proj": RowwiseParallel(output_layouts = Shard(1)),

            "ffnn_norm": SequenceParallel(),
            "ffnn": PrepareModuleInput(
                input_layouts = (Shard(1),),
                desired_input_layouts = (Replicate(),),
                use_local_output = True
                ),
            "ffnn.fc1": ColwiseParallel(),
            "ffnn.fc2": RowwiseParallel(output_layouts=Shard(1)),
        }
        layer.attention.num_heads = layer.attention.num_heads // tp_mesh.size()
        parallelize_module(layer, tp_mesh, format)

    return model

def apply_fsdp(model: Transformer, dp_mesh):
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward = False)
    fully_shard(model, mesh=dp_mesh, reshard_after_forward = False)
    return model

criterion = nn.CrossEntropyLoss(ignore_index=-100)
def loss_function(output, target):
    return criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

def clip_grad_norm(model, max_norm, pp_mesh):
    '''Global grad-norm clipping across DP/TP (DTensor handles it) and across PP stages (manual all-reduce).'''
    params = [p for p in model.parameters() if p.grad is not None]
    grads = [p.grad for p in params]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0)   # foreach auto: fast on GPU, works on CPU
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    if pp_mesh.size() > 1:
        total_norm = total_norm ** 2
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
        total_norm = total_norm ** 0.5
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm)
    return total_norm

def save_checkpoint(model, pp_rank, pp_size, layers_per_stage, is_stage_writer):
    '''
    Every rank of a stage takes part in gathering full tensors (collective over that stage's DP/TP groups).
    One rank per stage writes stage_{pp_rank}.pt; global rank 0 then merges all stages into weight.pth,
    which loads directly into a plain (non-distributed) Transformer.
    '''
    state = {}
    for k, v in model.state_dict().items():
        if isinstance(v, DTensor):
            v = v.full_tensor()
        if k.startswith('layers.'):
            parts = k.split('.')
            parts[1] = str(int(parts[1]) + pp_rank * layers_per_stage)   # local layer idx -> global layer idx
            k = '.'.join(parts)
        state[k] = v.detach().cpu()

    if is_stage_writer:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(state, os.path.join(CKPT_DIR, f'stage_{pp_rank}.pt'))
    dist.barrier()

    if global_rank == 0:
        merged = {}
        for s in range(pp_size):
            merged.update(torch.load(os.path.join(CKPT_DIR, f'stage_{s}.pt'), weights_only=True))
        torch.save(merged, os.path.join(CKPT_DIR, 'weight.pth'))
        print(Fore.GREEN + f'=== Checkpoint saved to {os.path.join(CKPT_DIR, "weight.pth")} ===' + Style.RESET_ALL)
    dist.barrier()

def main():
    torch.manual_seed(1234)          # same seed on every rank -> identical initial weights before sharding
    validate_parallel_config(cfg, TP_DIM, PP_DIM)

    model = Transformer(cfg).to(device)
    tp_mesh, pp_mesh, dp_mesh = mesh['tp'], mesh['pp'], mesh['dp']

    # === Creating PP ===
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    dp_rank = dp_mesh.get_local_rank()

    is_first = pp_rank==0
    is_last  = pp_rank==pp_size-1
    is_logger = is_last and tp_rank == 0 and dp_rank == 0       # exactly one rank prints the loss
    is_stage_writer = tp_rank == 0 and dp_rank == 0             # one rank per stage writes the checkpoint

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
    sampler = DistributedSampler(dataset, num_replicas=dp_mesh.size(), rank=dp_rank, drop_last=True)
    dataload = DataLoader(dataset, batch_size=cfg.batch_size * n_micro_batches, sampler=sampler, drop_last=True)

    if len(dataload) == 0:
        raise RuntimeError(
            f"Dataloader is empty: {len(dataset)} sequences is not enough for "
            f"DP={dp_mesh.size()} x batch {cfg.batch_size * n_micro_batches}. Lower batch_size or N_MICRO_BATCHES."
        )

    # === LR schedule needs total steps ===
    cfg.max_steps = len(dataload) * cfg.epochs

    if global_rank == 0:
        print(f'=== Mesh: PP={PP_DIM} DP={DP_DIM} TP={TP_DIM} | Steps/epoch: {len(dataload)} | Total steps: {cfg.max_steps} ===')

    stage = PipelineStage(model, stage_index=pp_rank, num_stages=pp_size, device=device, group=pp_mesh.get_group())

    schedule = ScheduleGPipe(stage, n_micro_batches, loss_fn = loss_function)

    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.max_lr)
    optimizer_scheduler = LambdaLR(optimizer, lambda step: cosine_annealing(step, cfg))

    print(f"=== Training Has Been Started at rank {global_rank} ===")

    for epoch in range(cfg.epochs):

        sampler.set_epoch(epoch)

        epoch_train_loss = 0
        epoch_train_step = 0
        epoch_train_perplexity = 0

        for batches in dataload:
            input_ids, targets, attn_mask = batches['inputs'], batches['targets'], batches['attn_mask']
            input_ids, targets, attn_mask = input_ids.to(device), targets.to(device), attn_mask.to(device)

            all_loss = []
            optimizer.zero_grad()

            with loss_parallel():
                if is_first and is_last:
                    schedule.step(input_ids, target=targets, losses=all_loss, attn_mask=attn_mask)
                elif is_first:
                    schedule.step(input_ids, attn_mask=attn_mask)
                elif is_last:
                    schedule.step(target=targets, losses=all_loss, attn_mask=attn_mask)
                else:
                    schedule.step(attn_mask=attn_mask)

            clip_grad_norm(model, cfg.max_grad_norm, pp_mesh)    # collective: every rank must call it
            optimizer.step()
            optimizer_scheduler.step()

            if is_logger:
                loss_val_per_batch = torch.stack(all_loss).mean()
                if isinstance(loss_val_per_batch, DTensor):
                    loss_val_per_batch = loss_val_per_batch.full_tensor()
                epoch_train_loss += loss_val_per_batch.item()
                epoch_train_perplexity += loss_val_per_batch.exp().item()
                epoch_train_step += 1
                if epoch_train_step % 10 == 0:
                    print(Fore.BLUE + f'Batch Loss@{epoch_train_step}: {(epoch_train_loss/epoch_train_step):.2f}'
                          + Style.RESET_ALL)

        if is_logger and epoch_train_step > 0:
            print(Fore.RED + f'Epoch {epoch+1}/{cfg.epochs} | Loss: {(epoch_train_loss/epoch_train_step):.2f} | '
                  f'Perplexity: {(epoch_train_perplexity / epoch_train_step) :.2f}' + Style.RESET_ALL)

        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0 or (epoch + 1) == cfg.epochs:
            save_checkpoint(model, pp_rank, pp_size, layers_per_stage, is_stage_writer)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

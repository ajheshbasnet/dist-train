from transformer import Transformer
from configs import configs
import torch
import torch.nn as nn
import random
import numpy as np
from colorama import Fore, Style
from dataloader import return_dataloader
from get_wandb_run import get_run

log_to_wandb = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if log_to_wandb:
        run = get_run()

    model = Transformer(configs()).to(device)
    dataloader, tokenizer = return_dataloader()
    optimizers = torch.optim.Adam(model.parameters(), lr=configs.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    total_params = sum(p.numel() for p in model.parameters())

    print(Fore.GREEN + f'== Trainable Parameters: {total_params/1e6 :.2f} Millions ==' + Style.RESET_ALL)

    global_step = 0
    
    for epoch in range(configs.epochs):

        epoch_train_loss = 0
        epoch_train_step = 0
        epoch_train_perplexity = 0

        print(Fore.BLUE + f'\n== Epoch: {epoch+1} / {configs.epochs} ==' + Style.RESET_ALL)

        for batches in dataloader:

            input_batch, target_batch, attn_mask = batches['inputs'], batches['targets'], batches['attn_mask']

            input_batch = input_batch.to(device)
            attn_mask = attn_mask.to(device)
            target_batch = target_batch.to(device)
            
            output = model(input_batch, attn_mask)
            
            loss = criterion(
                output.view(-1, output.size(-1)), 
                target_batch.view(-1),
                )

            optimizers.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), configs.max_grad_norm)
            optimizers.step()

            perplexity = torch.exp(loss)
            epoch_train_loss += loss.item()
            epoch_train_perplexity += perplexity.item()
            epoch_train_step += 1
            global_step += 1

            if log_to_wandb:
                run.log({'epoch': epoch, 'batch_train_loss': loss.item(), 'batch_perplexity': perplexity.item()}, step=global_step)
            
            
            print(Fore.RED + f'   Loss: {loss.item():.3f} | Perplexity: {perplexity:.3f}' + Style.RESET_ALL)
            

        if log_to_wandb:
            run.log({'epoch_train_loss': epoch_train_loss / epoch_train_step, 'epoch_train_perplexity': epoch_train_perplexity / epoch_train_step}, step=global_step)

if __name__ == "__main__":
    main()
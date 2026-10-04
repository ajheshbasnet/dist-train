import wandb
from configs import configs

def get_run():
    wandb.login(key=configs.wandb_api_key)
    run = wandb.init(
        project="transformer",
        config={
            'lr': configs.lr,
            'epochs': configs.epochs,
            'batch_size': configs.batch_size,
            'hidden_dim': configs.hidden_dim,
            'n_heads': configs.n_heads,
            'n_layers': configs.n_layers,
            'resid_dropout': configs.resid_dropout,
            'vocab_size': configs.vocab_size,
            'max_seq_len': configs.max_seq_len,
        },
    )
    return run

class configs:
    hidden_dim: int = 128
    max_seq_len: int = 32
    n_heads: int = 4
    n_layers: int = 8
    vocab_size: int = 4000
    attn_dropout: float = 0.05
    max_grad_norm: float = 1.0
    ffnn_dropout: float = 0.05
    resid_dropout: float = 0.01
    batch_size: int = 2
    epochs: int = 100
    
    max_lr:float = 1e-3
    min_lr: float = 5e-5
    warmup_steps: int = None

    max_new_tokens: int = 64

    tokenizer_path: str = "my_tokenizer"

    wandb_api_key: str = "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"

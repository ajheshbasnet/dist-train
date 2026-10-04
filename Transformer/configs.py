class configs:
    hidden_dim: int = 256
    max_seq_len: int = 756
    n_heads: int = 4
    n_layers: int = 8
    vocab_size: int = 12000
    attn_dropout: float = 0.2
    max_grad_norm: float = 1.0
    ffnn_dropout: float = 0.1
    resid_dropout: float = 0.1
    batch_size: int = 2
    epochs: int = 1
    lr: float = 1e-4

    wandb_api_key: str = "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"

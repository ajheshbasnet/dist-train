import torch
import torch.nn as nn
from configs import configs

class Token_Embedding(nn.Module):

    def __init__(self, configs: configs):
        super().__init__()
        self.token_embedding = nn.Embedding(configs.vocab_size, configs.hidden_dim)
        self.position_embedding = nn.Embedding(configs.max_seq_len, configs.hidden_dim)
    
    def forward(self, x):
        _, T = x.size()
        tok_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)   #[1, T]
        pos_emb = self.position_embedding(pos)                #[1, T, D]
        return tok_emb + pos_emb
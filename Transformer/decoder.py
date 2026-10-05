import torch
import torch.nn as nn
from configs import configs

from attention import Attention
from ffnn import FFNN

class SingleDecoderBlock(nn.Module):
    
    def __init__(self, configs: configs):
        super().__init__()
        self.attention = Attention(configs)
        self.ffnn = FFNN(configs)
        self.attn_norm = nn.LayerNorm(configs.hidden_dim)
        self.ffnn_norm = nn.LayerNorm(configs.hidden_dim)
        self.dropout = nn.Dropout(configs.resid_dropout)

    def forward(self, x, attn_mask = None):
        x = x + self.dropout(self.attention(self.attn_norm(x), attn_mask = attn_mask))
        x = x + self.dropout(self.ffnn(self.ffnn_norm(x)))
        return x

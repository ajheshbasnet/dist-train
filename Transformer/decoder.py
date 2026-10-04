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
        self.ln1 = nn.LayerNorm(configs.hidden_dim)
        self.ln2 = nn.LayerNorm(configs.hidden_dim)
        self.dropout = nn.Dropout(configs.resid_dropout)

    def forward(self, x, attn_mask = None):
        x = x + self.dropout(self.attention(self.ln1(x), attn_mask = attn_mask))
        x = x + self.dropout(self.ffnn(self.ln2(x)))
        return x

import torch
import math
import torch.nn as nn
from configs import configs

class Attention(nn.Module):
    
    def __init__(self, configs: configs):
        super().__init__()
        self.head_dim = configs.hidden_dim // configs.n_heads
        self.num_heads = configs.n_heads
        
        self.q_proj = nn.Linear(configs.hidden_dim, configs.hidden_dim)
        self.k_proj = nn.Linear(configs.hidden_dim, configs.hidden_dim)
        self.v_proj = nn.Linear(configs.hidden_dim, configs.hidden_dim)
        self.out_proj = nn.Linear(configs.hidden_dim, configs.hidden_dim)
        self.dropout = nn.Dropout(configs.attn_dropout)
        
    def forward(self, x, attn_mask = None):
        B, T, C = x.shape

        '''attn_mask will be of shape: [B, 1]'''

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  #[B, N_HEADS, T, D_HEAD]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_score = q@k.transpose(-2, -1)
        attn_score = attn_score / math.sqrt(self.head_dim)
        causual_mask = torch.tril(torch.ones(T, T, device = x.device, dtype = torch.bool))
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            mask = causual_mask & (attn_mask).bool()
            attn_score = attn_score.masked_fill(~mask, float('-inf'))
        else:
            attn_score = attn_score.masked_fill(~causual_mask, float('-inf'))

        norm_attn_score = attn_score.softmax(dim = -1)
        norm_attn_score = self.dropout(norm_attn_score)
        outputs = norm_attn_score @ v                                                       #[B, N_HEADS, T, D_HEAD]
        outputs = outputs.permute(0, 2, 1, 3).contiguous()                                  #[B, T, N_HEADS, D_HEAD]
        outputs = outputs.view(B, T, -1)                                                   #[B, T, C] - infer last dim for TP
        outputs = self.out_proj(outputs)
        return outputs
import torch
import torch.nn as nn
from token_embedding import Token_Embedding
from decoder import SingleDecoderBlock
from configs import configs


class Transformer(nn.Module):
    
    def __init__(self, configs: configs):
        super().__init__()
        self.token_embedding = Token_Embedding(configs)
        self.decoder = nn.ModuleList([SingleDecoderBlock(configs) for _ in range(configs.n_layers)])
        self.layernorm = nn.LayerNorm(configs.hidden_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, attn_mask = None):
        x = self.token_embedding(x)
        for decoder in self.decoder:
            x = decoder(x, attn_mask)
        x = self.layernorm(x)    
        logits = x @ self.token_embedding.token_embedding.weight.T    
        return logits

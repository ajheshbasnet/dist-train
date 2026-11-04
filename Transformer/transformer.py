import os
import torch
import torch.nn as nn
from token_embedding import Token_Embedding
from decoder import SingleDecoderBlock
from configs import configs
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_dir, configs.tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

class Transformer(nn.Module):
    
    def __init__(self, configs: configs):
        super().__init__()
        self.token_embedding = Token_Embedding(configs)
        self.layers = nn.ModuleList([SingleDecoderBlock(configs) for _ in range(configs.n_layers)])
        self.layernorm = nn.LayerNorm(configs.hidden_dim)
        self.output_proj = nn.Linear(configs.hidden_dim, configs.vocab_size)

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

        if self.token_embedding is not None:
            x = self.token_embedding(x)

        for decoder in self.layers:
            x = decoder(x, attn_mask)

        if self.layernorm is not None:        
            x = self.layernorm(x)
            logits = self.output_proj(x)
            return logits    
        return x

    def generate(self, texts, max_new_tokens: int = configs.max_new_tokens, temperature: float = 1.0, top_p: float = 1.0):

        input_ids = tokenizer(texts, return_tensors = 'pt')['input_ids']
        output_ids = list(input_ids.clone()[0])

        for _ in range(max_new_tokens):
            logits = self(input_ids)
            new_token_logits = logits[:, -1, :][0]
            norm_token_logits = new_token_logits.softmax(dim=-1)
            vals, indices = torch.sort(norm_token_logits, descending=True)   
            cumsum = vals.cumsum(dim=-1)
            top_p_true = cumsum<top_p
            indices = indices[top_p_true]
            top_p_prob_val = vals[indices].softmax(dim=-1)
            output_token_idx = torch.multinomial(top_p_prob_val, 1)
            output_ids.append(output_token_idx)
            input_ids = torch.cat([input_ids, output_token_idx.unsqueeze(-1)], dim = -1)

        output_ids = [items.item() for items in output_ids]
        return output_ids, tokenizer.decode(output_ids)

    
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

    @torch.inference_mode()
    def generate(
        self,
        texts,
        max_new_tokens: int = configs.max_new_tokens,
        temperature: float = 1.0,
        top_p: float = 1.0
    ):

      if temperature <= 0:
          raise ValueError("Temperature must be greater than zero.")

      if not 0 < top_p <= 1:
          raise ValueError("top_p must be in the interval (0, 1].")

      device = next(self.parameters()).device

      was_training = self.training
      self.eval()

      try:
          input_ids = tokenizer(
              texts,
              return_tensors="pt"
          )["input_ids"].to(device)

          if input_ids.size(0) != 1:
              raise ValueError("This implementation supports one prompt at a time.")

          output_ids = input_ids[0].tolist()

          for _ in range(max_new_tokens):

              # 1. Forward pass
              logits = self(input_ids)

              # 2. Get logits of the last token
              next_token_logits = logits[:, -1, :]

              # 3. Apply temperature
              next_token_logits = next_token_logits / temperature

              # 4. Convert logits into probabilities
              probs = torch.softmax(next_token_logits, dim=-1)[0]

              # 5. Top-p / nucleus sampling
              if top_p < 1.0:

                  sorted_probs, sorted_indices = torch.sort(
                      probs,
                      descending=True
                  )

                  cumulative_probs = sorted_probs.cumsum(dim=-1)

                  # Preserve the token that crosses the top-p threshold
                  keep_mask = (cumulative_probs - sorted_probs) < top_p

                  filtered_sorted_probs = sorted_probs * keep_mask

                  # Restore original vocabulary ordering
                  filtered_probs = torch.zeros_like(probs)

                  filtered_probs.scatter_(
                      dim=0,
                      index=sorted_indices,
                      src=filtered_sorted_probs
                  )

                  # Renormalize
                  probs = filtered_probs / filtered_probs.sum()

              # 6. Sample next token
              output_token_idx = torch.multinomial(probs, num_samples=1)

              # 7. Append generated token
              token_id = output_token_idx.item()

              output_ids.append(token_id)

              input_ids = torch.cat(
                  [input_ids, output_token_idx.unsqueeze(-1)],
                  dim=-1
              )

              # 8. Stop at EOS
              if tokenizer.eos_token_id is not None:
                  if token_id == tokenizer.eos_token_id:
                      break

          return output_ids, tokenizer.decode(output_ids)

      finally:
          self.train(was_training)
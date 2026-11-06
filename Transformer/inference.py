import os
import sys
from transformer import Transformer
from configs import configs
import torch

# Default: weights from single-device training (Transformer/main.py).
# For weights from distributed training pass: python inference.py ../TrainingParallelism/checkpoints/weight.pth
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(current_dir, "weight.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(configs).to(device)

model.load_state_dict(
    torch.load(
        weight_path,
        map_location=device,
        weights_only=True
    )
)

def eval(text):
      output_ids, output_text = model.generate(
          text,
          max_new_tokens = 100,
          temperature = 0.05,
          top_p = 0.93
      )

      return output_text

print(eval("Cristiano Ronaldo"))

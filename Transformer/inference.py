from transformer import Transformer
from configs import configs
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(configs).to(device)

model.load_state_dict(
    torch.load(
        "/content/dist-train/Transformer/weight.pth",
        map_location=device,
        weights_only=True
    )
)

def eval(text):
      output_ids, output_text = model.generate(
          text,
          max_new_tokens = 100,
          temperature = 0.8,
          top_p = 0.9
      )

      return output_text

print(eval("In his youth, Jobs's parents"))
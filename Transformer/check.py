from transformers import AutoTokenizer
from dataset import load_local_dataset

tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")

print(tokenizer)
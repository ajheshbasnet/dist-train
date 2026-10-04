from datasets import load_dataset, load_from_disk
import os
from configs import configs
from transformers import AutoTokenizer
import torch

def download_dataset():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Load and save dataset
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1"
    )

    dataset['train'] = dataset['train'].select(torch.arange(100))
    dataset['test'] = dataset['test'].select(torch.arange(50))
    dataset['validation'] = dataset['validation'].select(torch.arange(50))

    # Save to disk
    dataset.save_to_disk(data_dir)
    print(f"Dataset saved to: {data_dir}")
    return dataset

def load_local_dataset():
    # Load from local directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    dataset = load_from_disk(data_dir)
    print(f"Dataset loaded from: {data_dir}")
    return dataset

if __name__ == "__main__":
    # Download the dataset
    dataset = download_dataset()
    dataset = load_local_dataset()
    print(dataset)

    # Just use GPT-2's tokenizer setup as the starting point
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    new_tokenizer = base_tokenizer.train_new_from_iterator(
        dataset['train']['text'],
        vocab_size=configs.vocab_size,
    )

    new_tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'bos_token': '[BOS]',
    })
    new_tokenizer.save_pretrained("./my_tokenizer")

    print(new_tokenizer)

    print(f'Tokenizer is saved')

from prepare_dataset import load_local_dataset
from configs import configs
from transformers import AutoTokenizer
import math
import torch
from colorama import Fore, Style

def return_dataset():

    datasets = load_local_dataset()

    tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\hp\\OneDrive\\Desktop\\Distributed Training\\Transformer\\my_tokenizer")

    text_stream = ''.join(datasets['train']['text'])

    text_stream_ids = tokenizer(text_stream)['input_ids']  # since it's in shape [1, K] so taking only K dimension

    ids_len = len(text_stream_ids)

    print(Fore.GREEN + f'\nTotal Tokens Used For Training: {ids_len}\n' + Style.RESET_ALL)

    # == since inorder to form the stack of tensor, they should have the equal dimension so we need to fix the length by adding padding tokens ==

    desired_length_for_tensors = (math.ceil(ids_len/configs.max_seq_len)*configs.max_seq_len) - ids_len

    # == adding the padding token ==
    text_stream_ids.extend([tokenizer.pad_token_id] * (desired_length_for_tensors + 1))

    input_ids = []
    target_ids = []

    for i in range(0, len(text_stream_ids)-1 , configs.max_seq_len):
        inp_id = text_stream_ids[i : i + configs.max_seq_len]
        tgt_id = text_stream_ids[i+1 : i+1 + configs.max_seq_len]
        
        input_ids.append(inp_id)
        target_ids.append(tgt_id)

    class MyCustomDataset:
        def __init__(self, input_ids, target_ids):
            self.input_ids = input_ids
            self.target_ids = target_ids
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            inputs, targets = self.input_ids[idx], self.target_ids[idx]
            inputs = torch.tensor(inputs, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)

            attn_mask = (inputs != tokenizer.pad_token_id).long()
            return {
                'inputs': inputs, 
                'targets': targets, 
                'attn_mask': attn_mask
            }

    dataset = MyCustomDataset(input_ids, target_ids)
    return dataset, tokenizer

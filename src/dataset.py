# src/dataset.py
import torch
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    """
    token_ids: list[int] (tokenized full text)
    Creates (input_seq, target_seq) pairs using sliding window.
    """
    def __init__(self, token_ids, max_length=512, stride=128):
        self.token_ids = token_ids
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        N = len(token_ids)
        for i in range(0, N - max_length, stride):
            x = token_ids[i:i+max_length]
            y = token_ids[i+1:i+max_length+1]
            self.examples.append((torch.tensor(x, dtype=torch.long),
                                  torch.tensor(y, dtype=torch.long)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

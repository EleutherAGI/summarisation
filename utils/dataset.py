import json
import torch
from torch.utils.data import Dataset

class TLDRDataset(Dataset):
    """TLDR dataset"""

    def __init__(self, file):
        """
        Args: 
            file (string): Path to the csv file with annotations.
        """
        with open(file) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns: 
            prompt (string): The prompt portions of the reddit post.
            summary (string): The summary portions of the reddit post.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]

        return data['content'], data['summary']

        
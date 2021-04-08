import json
import torch
from torch.utils.data import Dataset

class TLDRDataset(Dataset):
    """TLDR dataset"""

    def __init__(self, file, tokenizer):
        """
        Args: 
            file (string): Path to the csv file with annotations.
            tokenizer (transformers.tokenizer): Tokenizer to use for training to get length of encoded data.
        """
        self.tokenizer = tokenizer
        with open(file) as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns: 
            combined (string): The entire body of the reddit post.
            summary length (int): The length of summary portion of the reddit post after tokenization.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]

        #Seems like a little bit of a hack to get length of summary, but alright
        return data['content'] + ' TLDR:' + data['summary'], self.tokenizer(data['summary'], return_length = True)['length']

        
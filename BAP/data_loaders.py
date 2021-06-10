import sys
import os
from argparse import ArgumentParser
import copy
import glob
import numpy as np
import pandas as pd
import json
from urllib.parse import urlparse, urljoin

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

def commonsense(data_save_path = '/content/'):
    url = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"
    tar_path = os.path.join(data_save_path, 'temp')
    try:
        os.mkdir(tar_path)
    except:
        pass
    
    urllib.request.urlretrieve(url, os.path.join(tar_path, 'ethics.tar'))
    my_tar = tarfile.open(os.path.join(tar_path, 'ethics.tar'))
    my_tar.extractall(tar_path)
    my_tar.close()

    file_names = [os.path.join(tar_path, 'ethics/commonsense/cm_test.csv'),
                  os.path.join(tar_path, 'ethics/commonsense/cm_train.csv')]

    X_total = pd.DataFrame()
    for file_name in file_names:
        X = pd.read_csv(file_name)
        X = X[X['is_short'] == True]
        X_total = pd.concat((X_total, X), axis = 0)

    X = X_total

    X = X[['input', 'label']]
    X = X.rename(columns = {'input':'sentences', 'label':'labels'})

    return list(X['sentences'])


class TextDataset(Dataset):
    def __init__(self, args, partition, data=data):
        self.sentences = []
        n_words = 6
        for date in data:
            words = date.split(' ')
            if len(words) > n_words:
                self.sentences.append(' '.join(words[:n_words]))
        # self.sentences = ["A man walks into a bar", "What's the deal with"]

        self.len = len(self.sentences)
        if partition == 'train':
            self.sentences = self.sentences[:int(0.9*self.len)]
        elif partition == 'val':
            self.sentences = self.sentences[int(0.9*self.len):int(0.95*self.len)]
        else:
            self.sentences = self.sentences[int(0.95*self.len):]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

class TextDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = TextDataset(self.args, partition = 'train')
            self.val_data = TextDataset(self.args, partition = 'val')
        if stage == 'test' or stage is None:
            self.test_data = TextDataset(self.args, partition = 'test')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
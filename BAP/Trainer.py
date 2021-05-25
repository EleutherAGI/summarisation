from collections import OrderedDict
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from adapters import add_adapters, add_adapter_skip, AdapterLayer

class BAPTrainer:
    """
    Babble and Prune trainer
    """
    def __init__(self, babbler, value_model, tokenizer, dataset, args):
        self.babbler = babbler
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.dataset = dataset

        if self.args.use_adapters:
            self.babbler.requires_grad = False
            self.babbler.adapter_grad(True)

        self.optimizer = Adam(babbler.parameters(), lr=args['lr'])


    def step(self, sentences):
        """
        Run a BAP optimisation step.
        returns:
            loss
        """
        input = self.tokenizer(sentences)
        babbles = self.babbler.babble(input)
        values = self.value_model(babbles)
        
        batch_size = values.shape[0]
        n_best = int(self.arg.top_rho*batch_size)
        best_values, _ = torch.topk(values, n_best)

        loss = -torch.sum(torch.log(best_values))/batch_size
        return loss

    def train(self):
        for epoch in tqdm(range(self.args.steps//self.args.batch_size)):
            # torch.cuda.empty_cache()
            logs = dict()
        
            sentence_batch = self.dataset.sample(self.args.batch_size)

            loss = self.step(sentence_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            #### Log everything
            logs.update({'game_log':wandb.Table(
                columns=['query', 'response', 'reward'],
                rows=table_rows)})
            logs['env/loss'] = torch.mean(loss).cpu().numpy()
            wandb.log(logs)

        # TODO: pytorch lightning, will take care of validation and .to(device)
# based on https://github.com/lvwerra/trl/blob/master/nbs/04-gpt2-sentiment-ppo-training.ipynb

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
from Trainer import BAPTrainer

def main(args):
    wandb.init(name='run-1', project='gpt-Babble-and-Prune', config = args)
    #wandb.config.update(args)

    # load data
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    datasets = load_dataset(extension, field='data', data_files=data_files)

    # load models
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    babbler = AutoModel.from_pretrained(args.lm_name, return_dict=True)
    if args.use_adapters:
        babbler = add_adapters(babbler)
        babbler = add_adapter_skip(babbler)
        babbler = add_adapter_grad(babbler)

    # TODO: add a .babble method to the babbler to generate text
    value_model = ValueFunction(args)
    wandb.watch(babbler, log='all')
    bap_trainer = BAPTrainer(babbler, value_model, tokenizer, dataset, args)
    bap_trainer.train()

    os.makedirs('gpt-babble-and-prune')
    #gpt2_model.save_pretrained('gpt2-imdb-pos')
    #gpt2_tokenizer.save_pretrained('gpt2-imdb-pos')

# https://github.com/EleutherAGI/summarisation/blob/main/run_ppo.py

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('lm_name', type = str, default = "EleutherAI/gpt-neo-1.3B", help='Name languge model')
    parser.add_argument('train_file', type = str)
    parser.add_argument('validation_file', type = str)
    parser.add_argument('use_adapters', type=bool, default = False, help = 'whether to use adapters')
    parser.add_argument('lr', type=float, default = 1e-4, help = ' learning rate')
    parser.add_argument('top_rho', type=float, default = 0.2, help = 'fraction of best babbles to train on, rho in (0, 1]')
    parser.add_argument('steps', type=int, default = 1e5, help = 'total number of training steps')
    parser.add_argument('batch_size', type=int, default = 128, help = 'batch size')
    args = parser.parse_args()

    main(args)
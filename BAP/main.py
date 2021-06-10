import sys
import os
from argparse import ArgumentParser
import copy
import glob
import numpy as np
import json
import os

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from urllib.parse import urlparse, urljoin

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

import tarfile
import urllib

import wandb
#wandb.Api()

data = commonsense()

from reward_model import Reward
from data_loaders import TextDataModule, TextDataset, commonsense
from adapters import AdapterLayer, add_adapters, add_adapter_skip
from trainer import create_prompt, BaP


from pytorch_lightning.loggers import WandbLogger
wandb.init()

def main(args):
    wandb_logger = WandbLogger()
    Text = TextDataModule(args)
    model = BaP(args)

    trainer = pl.Trainer(logger=wandb_logger, gpus=1, max_epochs=args.n_epochs,
                        progress_bar_refresh_rate=10, val_check_interval=args.val_check_interval,
                        precision=args.precision, gradient_clip_val=0.5, accumulate_grad_batches=4,
                        log_every_n_steps=1)

    trainer.fit(model, Text)

if __name__ == '__main__':
    parser = ArgumentParser()
    # model args
    parser.add_argument('--self_prune', type = bool, default = False, help='use self-pruning: language model as a reward model')
    parser.add_argument('--lm_name', type = str, default = 'gpt2-large', help='Name language model') # "EleutherAI/gpt-neo-1.3B" 'sshleifer/tiny-gpt2'
    parser.add_argument('--use_adapters', type=bool, default = True, help = 'Whether to use adapters')

    # data loader args
    parser.add_argument('--batch_size', type=int, default = 4, help = 'Batch size training')
    parser.add_argument('--val_batch_size', type=int, default = 4, help = 'Batch size validation and test')
    parser.add_argument('--num_workers', type=int, default = 0, help = 'Number of workers')

    # trainer args
    parser.add_argument('--val_check_interval', type=float, default = 1., help = 'Frequency validation set check')
    parser.add_argument('--precision', type=int, default = 32, help = 'Bit precision')

    # adapter args
    parser.add_argument('--reduction_factor', type = int, default = 12, help = 'Reduction factor inner dimension adapters')

    # babble args
    parser.add_argument('--num_beams', type = float, default = 16, help='Number of beams')
    parser.add_argument('--num_return_sequences', type = int, default = 8, help='Number of babbles')
    parser.add_argument('--max_babble_len', type = int, default = 15, help='Length generated text')

    # optimizer args
    parser.add_argument('--lr_init', type = float, default = 1e-3, help='Initial learning rate')
    parser.add_argument('--lr_min', type = float, default = 1e-4, help='Final learning rate')
    parser.add_argument('--n_epochs', type = int, default = 4, help='Number of training epochs')
    parser.add_argument('--scheduler_period', type = int, default = 20, help='Frequency at which the learning rate gets updated')

    # loss args
    parser.add_argument('--loss_fn', type = str, default = 'CE', help='Loss function type: in [CE, PPO]')
    parser.add_argument('--loss_fns_val', type = list, default = ['CE'], help='Validation loss functions')
    parser.add_argument('--CE_top_k', type = int, default = 1, help='Number of best babbles to train on')
    parser.add_argument('--beta', type = float, default = 0.1, help='Weight of the regularisation KL term')

    args = parser.parse_args()

    main(args)
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

class BaP(LightningModule):
    def __init__(self, args):
        super().__init__()

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # load language model
        self.lm = AutoModelForCausalLM.from_pretrained(args.lm_name, return_dict=True, pad_token_id=self.tokenizer.eos_token_id)
        if args.use_adapters:
            self.lm = add_adapters(self.lm)
            self.lm = add_adapter_skip(self.lm)
            self.lm = add_adapter_grad(self.lm)
        
        # load reward model
        self.reward = Reward(args)

        # babble args
        self.num_beams = args.num_beams
        self.num_return_sequences = self.num_return_sequences
        self.max_babble_len = self.max_babble_len
        assert self.num_return_sequences <= self.num_beams
        
        # optimizer args
        self.lr_init, self.lr_min = args.lr_init, args.lr_min
        self.n_epochs = args.n_epochs
        self.scheduler_period = args.scheduler_period
        
        # loss function
        if args.loss_fn == 'CE':
            self.loss_fn = self.CE_loss
        self.loss_fns_val = args.loss_fns_val
        self.CE_top_rho = args.CE_top_rho
        
        #self.hparams = args
        #self.save_hyperparameters()
        
    # loss functions:
    def CE_loss(self, v):
        batch_size = v.shape[0]
        n_best = int(self.CE_top_rho*batch_size)
        best_values, _ = torch.topk(values, n_best)

        loss = -torch.sum(torch.log(best_values))/batch_size
        return loss

    def PPO_loss(self, v):
        pass
    
    def babble(self, prompts):
        inputs = self.tokenizer(prompts, padding = True, return_tensors="pt")
        batch_size, max_len = inputs.input_ids.shape
        with torch.no_grad():
            # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/3
            outputs = model.generate(input_ids=inputs.input_ids.to(model.device),
                                     attention_mask=inputs.attention_mask.to(model.device),
                                     pad_token_id=self.tokenizer.eos_token_id,
                                     num_beams=self.num_beams, # num_return_sequences <= num_beams!
                                     num_return_sequences=self.num_return_sequences,
                                     max_length=max_len + self.max_babble_len)
            outputs = outputs.reshape(batch_size, self.num_beams, -1)
        return outputs

    def create_prompt(self, sentences):
        return ['To do:', 'Change this']

    def training_step(self, batch, batch_id):
        babbles = self(batch)
        values = self.reward(babbles)
        loss = self.loss_fn(values)
        return loss
    
    # Use for inference only
    def forward(self, batch, batch_id):
        prompt = self.create_prompt(batch)
        # torch.cuda.empty_cache()
        babbles = self.babble(prompt)
        return babbles

    def validation_step(self, batch, batch_id):
        babbles = self(batch)
        values = self.reward(babbles)
        loss = self.loss_fn(values)
        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        for loss_type in validation_step_outputs[0]:
            loss_tot = torch.stack([losses[loss_type] for losses in validation_step_outputs])
            loss_avg = loss_tot[loss_tot != -1].mean()
            self.log('val-'+loss_type, loss_avg, logger=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr_init)
        n_batches = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        # second arg = how many step's (don't understand its behaviour for 'epoch') it takes to reach lr_min
        scheduler = {'scheduler': CosineAnnealingLR(optimizer, n_batches*self.epochs//self.scheduler_period, self.lr_min),
                     'interval': 'step', 'frequency': self.scheduler_period}
        # retardedly enough, THE HIGHER THE FREQUENCY THE LESS OFTEN IT STEPS
        # every 'frequency' steps it takes one scheduler.step and it takes 'second arg of 'scheduler'' to reach lr_min
        return [optimizer], [scheduler]

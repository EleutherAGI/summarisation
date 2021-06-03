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

import torchmetrics

def create_prompt(sentences):
    return sentences

class BaP(LightningModule):
    def __init__(self, args):
        super().__init__()

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # needed for batch text generation

        # load language model
        self.lm = AutoModelForCausalLM.from_pretrained(args.lm_name, return_dict=True, pad_token_id=self.tokenizer.eos_token_id)
        self.lm = self.lm.eval()
        for _, param in self.lm.named_parameters():
            param.requires_grad = False
        
        if args.use_adapters:
            self.lm = add_adapters(self.lm, args.reduction_factor)
            self.lm = add_adapter_skip(self.lm)
            assert self.lm.transformer.h[0].mlp.Adapter.adapter[0].weight.requires_grad == True
        # assert self.lm.transformer.h[0].mlp.MLP.c_fc.weight.requires_grad == False # TODO make this work too for gptneo architecture

        # load reward model
        self.reward = Reward(args)

        # babble args
        self.num_beams = args.num_beams
        self.num_return_sequences = args.num_return_sequences
        self.max_babble_len = args.max_babble_len
        assert self.num_return_sequences <= self.num_beams
        
        # optimizer args
        self.lr_init, self.lr_min = args.lr_init, args.lr_min
        self.n_epochs = args.n_epochs
        self.scheduler_period = args.scheduler_period
        
        # loss function
        if args.loss_fn == 'CE':
            self.loss_fn = self.CE_loss
        
        self.loss_fns_val = {} # TODO use weights instead, calculate loss when weight neq 0
        if 'CE' in args.loss_fns_val:
            self.loss_fns_val['CE'] = self.CE_loss
        self.CE_top_k = args.CE_top_k
        assert self.CE_top_k < self.num_return_sequences
        self.beta = args.beta

        # logger
        #self.train_loss = pl.metrics.Accuracy()
        #self.val_loss = pl.metrics.Accuracy()
        #self.train_values = pl.metrics.Accuracy()
        #self.val_values = pl.metrics.Accuracy()

        self.save_hyperparameters()
        
    # loss functions:
    def CE_loss(self, values):
        
        loss = -torch.sum(torch.log(best_values))/values.numel()
        return loss

    def PPO_loss(self, v):
        pass
    
    def babble(self, prompts):
        inputs = self.tokenizer(prompts, padding = True, return_tensors="pt")
        batch_size, max_len = inputs.input_ids.shape

        # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/3
        # https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
        """
        # 
        with torch.no_grad():
            outputs = self.lm.generate(input_ids = inputs.input_ids.to(self.device),
                                       attention_mask = inputs.attention_mask.to(self.device),
                                       pad_token_id = self.tokenizer.eos_token_id,
                                       num_return_sequences = self.num_return_sequences,
                                       num_beams = self.num_beams,
                                       max_length = max_len + self.max_babble_len)#,
                                       #return_dict_in_generate=True, output_scores =True)
        """
        with torch.no_grad():
            outputs = self.lm.generate(input_ids = inputs.input_ids.to(self.device),
                                       attention_mask = inputs.attention_mask.to(self.device),
                                       pad_token_id = self.tokenizer.eos_token_id,
                                       num_return_sequences = self.num_return_sequences,
                                       max_length = max_len + self.max_babble_len,
                                       do_sample = True,
                                       top_p = 0.95, top_k = 40)
        
        # outputs = outputs.reshape(batch_size, self.num_return_sequences, -1)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False) #, outputs['scores']
    
    def babble_and_prune(self, batch, return_babbles = False):
        # 1. Generate babbles: multiple reponses per prompt (no grad)
        self.lm.adapter_skip(False)
        babbles = self(batch, None)
        # 2. Rank them (no grad)
        self.lm.adapter_skip(True)
        values = self.reward(babbles, self.tokenizer, self.lm, grad = False)
        # 3. Select best babbles/prune bad babbles
        values = values.reshape(len(batch), self.num_return_sequences)
        _, best_idx = torch.topk(values, self.CE_top_k, dim=1)
        best_babbles = [babbles[idx_babble  + i*self.num_return_sequences] for (i, idx_text) in enumerate(best_idx) for idx_babble in idx_text]
        # 4. Train model on best babbles
        self.lm.adapter_skip(False)
        inputs = self.tokenizer(best_babbles, padding = True, return_tensors="pt")
        outputs = self.lm(input_ids = inputs.input_ids.to(self.device), attention_mask = inputs.attention_mask.to(self.device))
        # 5. Compute loss
        loss = nn.CrossEntropyLoss()(outputs['logits'].reshape(-1, outputs['logits'].shape[-1]), inputs.input_ids.reshape(-1).to(self.device))
        # 6. Get regularisation
        # TODO
        if return_babbles:
            return loss, values, best_babbles
        else:
            return loss, values
        

    def training_step(self, batch, batch_id):
        loss, values, best_babbles = self.babble_and_prune(batch, True)
        #self.log('train_values', values.mean())
        #self.log('train_loss', loss)
        return {'loss': loss, 'values':values.mean(), 'babbles':best_babbles[0]}#loss
    
    def training_step_end(self, outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        #self.train_loss(outs["loss"], outs["values"])
        #self.log({"train/loss": outs['loss'], "train/values":outs['values'], "global_step": trainer.global_step})
        self.log("train/loss", outs['loss'], on_epoch=False, on_step=True)
        self.log("train/values", outs['values'], on_epoch=False, on_step=True)
        self.log("train/babbles", outs['babbles'], on_epoch=False, on_step=True)
    
    def forward(self, batch, batch_id):
        prompt = create_prompt(batch)
        # torch.cuda.empty_cache()
        babbles = self.babble(prompt)
        return babbles

    #def validation_step(self, batch, batch_id):
    #    loss, values, best_babbles = self.babble_and_prune(batch, return_babbles = True)
    #    #self.log('val_values', values.mean())
    #    #self.log('val_loss', loss)
    
    #def validation_step_end(self, outs):
    #    # log accuracy on each step_end, for compatibility with data-parallel
    #    self.val_loss(outs["loss"], outs["values"])
    #    self.log({"train/step": self.train_loss})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lm.parameters(), lr = self.lr_init)
        n_batches = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        # second arg = how many step's (don't understand its behaviour for 'epoch') it takes to reach lr_min
        scheduler = {'scheduler': CosineAnnealingLR(optimizer, n_batches*self.n_epochs//self.scheduler_period, self.lr_min),
                     'interval': 'step', 'frequency': self.scheduler_period}
        # retardedly enough, THE HIGHER THE FREQUENCY THE LESS OFTEN IT STEPS
        # every 'frequency' steps it takes one scheduler.step and it takes 'second arg of 'scheduler'' to reach lr_min
        return [optimizer], [scheduler]
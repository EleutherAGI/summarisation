import sys
import os
from argparse import ArgumentParser
import copy
import glob
import numpy as np
import glob
import json

import torch
import torch.nn as nn


class Reward(nn.Module):
    """
    Uses probabilities of the words ' bad' and ' negative' as reward
    """
    def __init__(self, args):
        super(Reward, self).__init__()
        self.self_prune = args.self_prune
    
    # TODO maybe move some of these methods to the trainer
    def construct_reward_prompt(self, text):
        #return "\"" + text + ".\" That joke was very"# funny.'
        return "\"" + text + ".\" This is"
    
    def forward(self, texts, tokenizer, lm, grad = True):
        reward_prompts = [self.construct_reward_prompt(text) for text in texts]
        reward_inputs = tokenizer(reward_prompts, padding = True, return_tensors="pt")
        if grad:
            reward_logits = lm(input_ids = reward_inputs.input_ids.to(lm.device), attention_mask = reward_inputs.attention_mask.to(lm.device))['logits']
        else:
            with torch.no_grad():
                reward_logits = lm(input_ids = reward_inputs.input_ids.to(lm.device), attention_mask = reward_inputs.attention_mask.to(lm.device))['logits']
        reward_probs = nn.Softmax(-1)(reward_logits)
        return (reward_probs[:, -1, 2089] + reward_probs[:, -1, 4633])/2. # reward_probs[:, -1, 8258]
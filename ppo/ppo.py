import numpy
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, kl_coef, target, horizon):
        self.value = kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        proportional_error = np.clip(current/self.target -1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

class PPOTrainer:
    def __init__(self, model, ref_model, **ppo_params):

        
        self.ref_model = ref_model
        self.model = model
     
        self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])

    def step(self, query, response, scores):

        # model input
        gen_len = response.shape[1]
        model_input = torch.cat((query, response), axis=1)

        logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len)

    def forward_pass(self, model_input, gen_len):
            logits = self.model(m_input)['logits']
            ref_logits = self.ref_model(m_input)['logits']
from models import BaseLinePPO
import torch
from tqdm import tqdm
import util
import ppo
from constants import LEARNING_RATE
import numpy as np


from transformers import AutoModelForSequenceClassification, GPT2Tokenizer
from gpt2withvaluehead import GPT2HeadWithValueModel


sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilgpt2").cuda()
gpt2_model = GPT2HeadWithValueModel.from_pretrained("distilgpt2").cuda()
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained("distilgpt2").cuda()
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

storage = util.RolloutStorage()

opt = torch.optim.Adam(gpt2_model.parameters(), lr = LEARNING_RATE)

BATCH_SIZE = 1
STEPS = 1000
INIT_KL_COEF = 0.2

for epoch in tqdm(range(int(np.ceil(STEPS/BATCH_SIZE)))):
    
    # Get responce from gpt2 to prompt, equivalent to an action
    inputs = tokenizer(["Hello, my dog is cute and", "Hello, my cat is fuzzy and"], return_tensors="pt") # TODO implement batch sampling
    generation_output = gpt2_model.generate(**inputs, return_dict_in_generate=True, output_hidden_states=True)
    values = torch.cat([torch.squeeze(v, dim=2) for v in generation_output['hidden_states']], dim=1) # Ugly way to extract val from gen
    generation_tokens = generation_output['sequences']

    # Caclulate rewards
    rewards = sentiment_model(generation_tokens)
    rewards =- INIT_KL_COEF * torch.log(
        gpt2_model(generation_tokens)['logits'] / gpt2_model_ref(generation_tokens)['logits']
    ).mean(-1).mean(-1)

    #TODO tie this up with the rest of the PPO algorithm
    #TODO move PPO algorith to DDP for multi-gpu support


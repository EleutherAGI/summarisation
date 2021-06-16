import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from classifier import GPT2ForSequenceClassification #needed as HF can't calculate sequence length from embeds
from datasets import load_dataset
from tqdm import tqdm

import wandb
wandb.init(project='transformer_estimator')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_model = GPT2ForSequenceClassification.from_pretrained("../models/checkpoint-15000").to(device)
gpt2_model = GPT2LMHeadModel.from_pretrained("../models/checkpoint-35000").to(device)
gpt2_model_ref = GPT2LMHeadModel.from_pretrained("../models/checkpoint-35000").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentiment_model.config.pad_token_id = tokenizer.eos_token_id
gpt2_model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    return torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)


datasets = load_dataset("json", field='data', data_files={
    "train": "../data/tldr-filtered-train.json",
    "validation": "../data/tldr-filtered-test.json"
})

# prep dataset
def tokenize_function(examples):
    output = tokenizer([txt + ' TLDR:' for txt in examples['content']], max_length=512, truncation=True, padding=True)
    return output

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns = datasets["train"].column_names
)

def collate_wrapper(batch):
    return tokenizer.pad(batch, return_tensors='pt')

loader = DataLoader(tokenized_datasets['train'], batch_size=1, pin_memory=False, collate_fn=collate_wrapper)

response_len = 64
optimizer = torch.optim.Adam(gpt2_model.parameters(), lr=4e-5)

sentiment_wte = sentiment_model.get_input_embeddings()
fake_embedding = nn.Linear(sentiment_wte.weight.size(0), sentiment_wte.weight.size(1), bias = False)
fake_embedding.weight = torch.nn.Parameter(sentiment_wte.weight.t())


optimizer.zero_grad()
for idx, batch in enumerate(tqdm(loader)):
    batch = batch.to(device)
    generation_output = gpt2_model.generate(input_ids=batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            max_length=batch['attention_mask'].size(1)+response_len)
                                            #do_sample=True, 
                                            #top_p = 1.0,)
    
    outputs = gpt2_model(input_ids=generation_output)
    logits = outputs['logits'][:,:-1,:][:, batch['input_ids'].shape[1]-1:]
    logits = F.gumbel_softmax(logits, tau=1, hard=True, dim=2)
    response_embeds = fake_embedding(logits)
    
    ref_logits = gpt2_model_ref(input_ids = generation_output)['logits']

    logprobs = logprobs_from_logits(outputs['logits'], 
                                    generation_output[:,1:])[:, -response_len:]

    ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], 
                                        generation_output[:,1:])[:, -response_len:]
    
    #Straight through estimator
    #response_embeds = l2e(logits)
    
    
    query_embeds = sentiment_wte.weight[batch['input_ids']]

    embeds = torch.cat([query_embeds,response_embeds],dim=1)

    lengths = torch.ne(generation_output, 50256).sum(-1)
    
    input_embeds = sentiment_wte.weight[50256].repeat(embeds.shape[0],embeds.shape[1],1)
    ne = torch.ne(generation_output, 50256)

    for i in range(ne.shape[0]):
        #rint(ne[i].sum(-1))
        input_embeds[i, :ne[i].sum(-1)] = embeds[i,ne[i]]

    #Need to do a final shift here
    score = sentiment_model(inputs_embeds = input_embeds, sequence_lengths = lengths-1)['logits']
    
    #optimizer.zero_grad()
    kl = logprobs-ref_logprobs
    loss = (-score + -0.3 * kl.mean()).mean()
    loss.backward()
    
    #optimizer.step()
    if (idx+1)%32 == 0: 
        optimizer.step()
        optimizer.zero_grad()

    wandb.log({
        "loss": loss.cpu().detach(),
        "scores": score.mean().cpu().detach(),
        "kls": kl.mean(),
    })
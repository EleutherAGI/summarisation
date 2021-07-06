import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    set_seed,
)
from datasets import load_dataset

set_seed(42)

MODEL = "distilgpt2_for_generation"

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

#load model
config_kwargs = {
    "cache_dir": None,
    "num_labels": 1,
    "pad_token_id": tokenizer.pad_token_id
}

config = AutoConfig.from_pretrained(
            MODEL, **config_kwargs)

# Slows down learning but allows gpt2-xl to fit into memory
# Credit too https://github.com/Xirider/finetune-gpt2xl
config.gradient_checkpointing = True
config.use_cache = False

model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=config)

# load dataset
datasets = load_dataset("json", field='data', data_files={
    "train": "../data/comparisons-train.json",
    "validation": "../data/comparisons-test.json"
})

# prep dataset
def tokenize_function(examples):
    output = {}
    for i in range(2):
        text = [f"SUBREDDIT: r/{content['subreddit']}\nTITLE: {content['title']}\nPOST: {content['post']}\nTL;DR: {summary[i]['text']}" for content, summary in zip(examples['info'], examples['summaries'])]
        tokenizer_output = tokenizer(text, max_length=512, truncation=True, padding='max_length')
        output[f"input_ids_{i}"] = tokenizer_output.pop("input_ids")
        output[f"attention_mask_{i}"] = tokenizer_output.pop("attention_mask")
    output['labels'] = examples['choice']
    return output


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns = datasets["train"].column_names
)


# collate data
class DualSampleDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.max_length = None
        self.pad_to_multiple_of = None

    def __call__(self, features):
        batch = {'labels': torch.tensor([feature.pop('labels') for feature in features])}
        for i in range(2):
            needing_padding = {f'input_ids': [feature.pop(f'input_ids_{i}') for feature in features],
                               f'attention_mask': [feature.pop(f'attention_mask_{i}') for feature in features]}

            output = self.tokenizer.pad(
                needing_padding,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch[f'input_ids_{i}'] = output[f'input_ids']
            batch[f'attention_mask_{i}'] = output[f'attention_mask']
        return batch

collator = DualSampleDataCollator(
    tokenizer=tokenizer
)

# On 8 gpus this will add up too an effective batch size of 64 (2*4*8)
# In openai paper they use a batch size of 64

args = {
    "remove_unused_columns" : False,
    "gradient_accumulation_steps" : 8,
    "num_train_epochs" : 2,
    "save_steps" : 1000,
    "logging_steps": 100,
    "eval_steps": 100, 
    "evaluation_strategy": "steps",
    "per_device_train_batch_size" : 4,
    "per_device_eval_batch_size" : 4,
    "output_dir" : f'{MODEL}_for_scoring',
    #"deepspeed":"ds_config.json",
    "lr_scheduler_type" : 'cosine',
    "learning_rate" : 1.5e-5,
    "do_train" : True,
    "do_eval" : True,
    "fp16" : True,
}
# define trainer
training_args = TrainingArguments(**args)


class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        logits = model(
            input_ids=torch.cat([
                inputs['input_ids_0'], 
                inputs['input_ids_1']]), 
            attention_mask=torch.cat([
                inputs['attention_mask_0'],
                inputs['attention_mask_1']])
        )['logits']
        # split of concat batch and and concat on dim=1
        # shape = [batch_size, 2]
        split_logits = torch.cat((logits[:logits.shape[0]//2], 
                        logits[logits.shape[0]//2:]), dim = 1)

        labels = torch.tensor(inputs["labels"]).to(logits.device)
        mask = torch.stack([labels, 1-labels], dim=1)
        indexed = torch.gather(split_logits, 1, mask)

        loss = -torch.log(torch.sigmoid(indexed[:,0]-indexed[:,1])).mean()

        # can't pass split logits to compute metrics, because hf trainer adjusts size
        # if it's not batch size
        return (loss, logits) if return_outputs else loss

def compute_metrics(eval_pred):
    #Not working atm
    logits, labels = eval_pred
    #eval_pred
    #accuracy = np.count_nonzero(np.argmax(logits, axis=1) == np.array(labels)) / logits.shape[0]
    return {'accuracy': accuracy}

trainer = ClassificationTrainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=collator,
    #compute_metrics = compute_metrics,
)

train_result = trainer.train()

trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate()

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

#TODO normalize average outputs to zero across dataset
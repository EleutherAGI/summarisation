import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    set_seed
)
from datasets import load_dataset

set_seed(42)


MODEL = "gpt2-medium"

#load model
config_kwargs = {
    "cache_dir": None,
}

config = AutoConfig.from_pretrained(
            MODEL, **config_kwargs)

# Slows down learning but allows gpt2-xl to fit into memory
# Credit too https://github.com/Xirider/finetune-gpt2xl
config.gradient_checkpointing = True
config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(MODEL, config=config)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# load dataset
datasets = load_dataset("json", field='data', data_files={
    "train": "../data/tldr-filtered/train.json",
    "validation": "../data/tldr-filtered/test.json"
})

# prep dataset
def tokenize_function(examples):
    text = [f'SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR: {summary}' for subreddit, title, post, summary in zip(
        examples['subreddit'], 
        examples['title'], 
        examples['post'], 
        examples['summary'],)]
    output = tokenizer(text, max_length=512, truncation=True, return_length = True)
    output["total_length"] = output.pop("length")
    output["summary_length"] = tokenizer(examples['summary'], return_length = True)['length']
    return output

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns = datasets["train"].column_names
)

# collate data
class DataCollatorWithPaddingAndMask:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.max_length = None
        self.pad_to_multiple_of = None

    def __call__(self, features):
        #TODO The following columns in the evaluation set  don't have a corresponding argument in `GPT2LMHeadModel.forward` and have been ignored: summary_length, length.
        total_length = [feature.pop('total_length') for feature in features]
        summary_length = [feature.pop('summary_length') for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        mask = torch.full_like(batch['attention_mask'], False)
        for idx, (t, s) in enumerate(zip(total_length, summary_length)):
            mask[idx][t-s:t] = True
        batch['mask'] = mask

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

collator = DataCollatorWithPaddingAndMask(
    tokenizer=tokenizer
)


args = {
    "remove_unused_columns" : False,
    "gradient_accumulation_steps" : 16,
    "num_train_epochs" : 1,
    "save_steps" : 500,
    "evaluation_strategy": "steps",
    "logging_steps": 100,
    "eval_steps": 100, 
    "per_device_train_batch_size" : 4,
    "per_device_eval_batch_size" : 4,
    "output_dir" : f'{MODEL}_for_generation',
    #"deepspeed":"ds_config.json",
    "lr_scheduler_type" : 'cosine',
    "learning_rate" : 6.35e-5,
    "do_train" : True,
    "do_eval" : True,
    "fp16" : True,
}

# define trainer
training_args = TrainingArguments(**args)

# On 8 gpus this will add up too an effective batch size of 128 (4*4*8)
# In openai paper they use a batch size of 128

# Little bit hacky but this is needed to eval uses
# custom compute_loss function
training_args.label_names = ['mask']

class MaskedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        mask = inputs.pop("mask").bool()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()

        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = inputs['input_ids'][..., 1:].contiguous().view(-1)
        shift_mask = mask[..., 1:].contiguous().view(-1)

        loss = loss_fct(shift_logits[shift_mask], shift_labels[shift_mask])

        return (loss, outputs) if return_outputs else loss


trainer = MaskedTrainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=collator
)

train_result = trainer.train()

trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate(ignore_keys=['mask'])

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


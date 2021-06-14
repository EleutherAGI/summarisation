import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

MODEL = "distilgpt2"

#load model
model = AutoModelForCausalLM.from_pretrained(MODEL)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# load dataset
datasets = load_dataset("json", field='data', data_files={
    "train": "../data/tldr-filtered-train.json",
    "validation": "../data/tldr-filtered-test.json"
})

# prep dataset
def tokenize_function(examples):
    text = [content + ' TLDR:' + summary for content, summary in zip(examples['content'], examples['summary'])]
    output = tokenizer(text, return_length = True)
    output["total_length"] = output.pop("length")
    output["summary_length"] = tokenizer(examples['summary'], return_length = True)['length']
    return output

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns = datasets["train"].column_names
)

tokenized_datasets['train'][0]

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

# define trainer
training_args = TrainingArguments(f"{MODEL}_for_generation")
training_args.remove_unused_columns = False
training_args.gradient_accumulation_steps = 4
training_args.save_steps = 5000

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

trainer.train()
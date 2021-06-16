import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

MODEL = "gpt2"


#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

#load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.config.pad_token_id = tokenizer.pad_token_id


# load dataset
datasets = load_dataset("json", field='data', data_files={
    "train": "../data/comparisons-train.json",
    "validation": "../data/comparisons-test.json"
})

# prep dataset
def tokenize_function(examples):
    output = {}
    for i in range(2):
        text = [content['post'] + ' TLDR:' + summary[i]['text'] for content, summary in zip(examples['info'], examples['summaries'])]
        tokenizer_output = tokenizer(text, max_length=512, truncation=True, padding=True)
        output[f"input_ids_{i}"] = tokenizer_output.pop("input_ids")
        output[f"attention_mask_{i}"] = tokenizer_output.pop("attention_mask")
    output['label'] = examples['choice']
    return output

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns = datasets["train"].column_names
)

tokenized_datasets['train'][0]

# collate data
class DualSampleDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.max_length = None
        self.pad_to_multiple_of = None

    def __call__(self, features):
        batch = {'label': [feature.pop('label') for feature in features]}
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

# define trainer
training_args = TrainingArguments(f"{MODEL}_for_scoring")
training_args.remove_unused_columns = False
training_args.gradient_accumulation_steps = 4
training_args.save_steps = 1000
training_args.eval_steps = 1000
training_args.evaluation_strategy = "steps"
training_args.num_train_epochs = 6

class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        loss_function = torch.nn.CrossEntropyLoss()
        
        outputs_0 = model(input_ids=inputs['input_ids_0'], attention_mask=inputs['attention_mask_0'])
        outputs_1 = model(input_ids=inputs['input_ids_1'], attention_mask=inputs['attention_mask_1'])

        logits = torch.cat((outputs_0['logits'], outputs_1['logits']), dim = 1)

        labels = torch.tensor(inputs.pop("label")).to(logits.device)

        loss = loss_function(logits, labels)

        return (loss, logits) if return_outputs else loss

trainer = ClassificationTrainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=collator
)

trainer.train()

#TODO normalize average outputs to zero across dataset
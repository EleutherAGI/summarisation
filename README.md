# summarisation
[Our nice Google Doc](https://docs.google.com/document/d/15HSqH0njR4nyINHRDDKOLONml2YUVY0A7Jlxzyuv1MM/edit)
[Our nice WANDB table](https://wandb.ai/kdog/summarisation)
[Our nice WANDB table 2](https://wandb.ai/kdog/huggingface)

## Mid tuning gpt2

scripts for midtuning / finetuning classification are a near copy of huggingfaces run_clm.py

midtune-mask trains on only the masked summary where as midtune trains on the whole text (prompt + summary)

The following runs a distilgpt2 on a single 2080ti and takes about 10 hours
```
python midtune_mask.py \
    --model_name_or_path distilgpt2 \
    --train_file ./data/tldr-filtered-train.json \
    --validation_file ./data/tldr-filtered-test.json \
    --do_train \
    --do_eval \
    --output_dir ./models/distilgpt2_masked \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8
```

The following runs a gpt2-xl on a 6x nvidia a100's and takes about 70 hours
```
deepspeed --num_gpus=6 midtune_mask.py \
--deepspeed ds_config.json \
--model_name_or_path gpt2-xl \
--train_file ./data/tldr-filtered-train.json \
--validation_file ./data/tldr-filtered-test.json \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir ./models/gpt2-xl \
--eval_steps 200 \
--save_steps 200 \
--logging_steps 25 \
--num_train_epochs 2 \
--gradient_accumulation_steps 8 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4
```

## Fine-tuning reward model




# Links

Download comparisons from OpenAI `azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/*" . --recursive`.

(not currently working)
`https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered` and `https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered_queries` also host OpenAI's filtered verson of the dataset [TL;DR dataset](https://zenodo.org/record/1168855) by Syed, Shahbaz, Voelske, Michael, Potthast, Martin, & Stein, Benno (2018). It is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). 

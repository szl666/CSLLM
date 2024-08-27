#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type math \
    --model_name_or_path /data1/home/hhu01/zlsong/llama3-8bf-hf\
    --lora_model_path output_models/finetune_with_lora_sym_method \
    --dataset_path data/test \
    --prompt_structure "input: {input}" \
    --deepspeed examples/ds_config.json \
    --metric accuracy

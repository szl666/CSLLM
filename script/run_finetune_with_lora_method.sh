#!/bin/bash
# Please run this script under ${project_id} in project directory of

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=finetune_with_lora_sym_method
project_dir=/data1/home/hhu01/zlsong/CSLLM
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/method/train

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path /data1/home/hhu01/zlsong/llama3-8bf-hf \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 128 \
    --per_device_train_batch_size 1 \
    --use_lora 1 \
    --lora_r 8 \
    --save_aggregated_lora 0\
    --deepspeed configs/ds_config_zero2.json \
    --run_name finetune_with_lora_sym \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
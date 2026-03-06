#!/usr/bin/env bash
set -euo pipefail

model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=2  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

batch_size=64
micro_batch_size=1
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

q_data_path=data/train/${task}/explore/vanilla.jsonl
q_model_name=${exp_name}-${model_name}-${task}-Q
sft_model_name=${exp_name}-${model_name}-${task}-sft_run1

export CUDA_VISIBLE_DEVICES=0,2

CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20001 qlass/train_q.py \
    --model_name_or_path ${save_dir}${sft_model_name} \
    --data_path ${q_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${q_model_name} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${micro_batch_size} \
    --gradient_accumulation_steps ${accumulation_step} \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --remove_unused_columns False 


if [ $? -ne 0 ]; then
    echo "Q model training failed"
    exit 1
fi
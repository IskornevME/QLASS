#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT=/home/m.iskornev/qlass/QLASS
OUT_DIR=/home/m.iskornev/qlass/models/qlass-Llama-2-7b-chat-hf-sciworld-sft_run1

export CUDA_VISIBLE_DEVICES=5

cd "$ROOT"
mkdir -p "$OUT_DIR"

model_name=base-Llama-2-7b-chat-hf
task=sciworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=1  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=1
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft

CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20001 qlass/train_sft.py \
    --model_name_or_path ${model_path}${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir "$OUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --gradient_accumulation_steps ${accumulation_step} \
    --save_strategy "no" \
    --save_total_limit 5 \
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
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi
#!/usr/bin/env bash
set -euo pipefail

# Train candidate-normalized parametric-action DuelingQNet for ALFWorld.
# This script intentionally gives every run its own output_dir and TensorBoard
# logging_dir so that models, checkpoints, configs and logs never overwrite
# previous experiments.

model_name=Llama-2-7b-chat-hf
task=alfworld
exp_name=qlass

# Number of visible GPUs used by torch.distributed.run.
node_num=${NODE_NUM:-2}

# Paths. MODEL_PATH should end with '/' in the original QLASS setup.
model_path=${MODEL_PATH}
save_dir=${MODEL_PATH}

# Effective batch = node_num * micro_batch_size * gradient_accumulation_steps.
batch_size=${BATCH_SIZE:-64}
micro_batch_size=${MICRO_BATCH_SIZE:-1}
accumulation_step=$((batch_size / node_num / micro_batch_size))

# Data paths. Use split_dueling_q_data.py before running this script.
data_dir=${DATA_DIR:-data/train/${task}/explore_7b_sft_d8_i0_s2_mpr3}
q_data_train_path=${Q_DATA_TRAIN_PATH:-${data_dir}/dueling_vanilla_raw_train.json}
q_data_val_path=${Q_DATA_VAL_PATH:-${data_dir}/dueling_vanilla_raw_val.json}

# Model names.
sft_model_name=${exp_name}-${model_name}-${task}-sft_run1
base_exp_name=${exp_name}-${model_name}-${task}-DuelingQ-raw

# Hyperparameters. Defaults match the original QNet launch where applicable.
learning_rate=${LEARNING_RATE:-1e-5}
weight_decay=${WEIGHT_DECAY:-0.}
warmup_ratio=${WARMUP_RATIO:-0.03}
lr_scheduler_type=${LR_SCHEDULER_TYPE:-cosine}
num_train_epochs=${NUM_TRAIN_EPOCHS:-2}
seed=${SEED:-42}

# Dueling-specific hyperparameters.
dueling_value_loss_coef=${DUELING_VALUE_LOSS_COEF:-0.1}
dueling_head_hidden_size=${DUELING_HEAD_HIDDEN_SIZE:-1024}
min_candidates=${MIN_CANDIDATES:-2}
drop_zero_variance=${DROP_ZERO_VARIANCE:-False}
drop_state_mismatch=${DROP_STATE_MISMATCH:-False}

# Logging/eval/save cadence. For load_best_model_at_end=True, save_steps must be
# aligned with eval_steps; by default we keep the original QNet save cadence.
logging_steps=${LOGGING_STEPS:-5}
eval_steps=${EVAL_STEPS:-200}
save_steps=${SAVE_STEPS:-1000}
save_total_limit=${SAVE_TOTAL_LIMIT:-2}

# Unique run id. You can override it, e.g. RUN_ID=raw_lr1e-5_vcoef0.1_seed42.
timestamp=$(date +%Y%m%d_%H%M%S)
run_id=${RUN_ID:-raw_lr${learning_rate}_bs${batch_size}_vcoef${dueling_value_loss_coef}_seed${seed}_${timestamp}}

# Directory layout:
#   <experiment_root>/runs/<run_id>/          final model + checkpoints + trainer state
#   <experiment_root>/tensorboard/<run_id>/   TensorBoard event files
experiment_root=${DUELING_Q_EXP_ROOT:-${save_dir}${base_exp_name}}
output_dir=${OUTPUT_DIR:-${experiment_root}/runs/${run_id}}
logging_dir=${LOGGING_DIR:-${experiment_root}/tensorboard/${run_id}}

mkdir -p "${output_dir}" "${logging_dir}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}

echo "[train_dueling_qnet_7b] run_id=${run_id}"
echo "[train_dueling_qnet_7b] output_dir=${output_dir}"
echo "[train_dueling_qnet_7b] logging_dir=${logging_dir}"
echo "[train_dueling_qnet_7b] train_data=${q_data_train_path}"
echo "[train_dueling_qnet_7b] val_data=${q_data_val_path}"

# Build eval args only if validation split exists. This keeps the same script
# usable for final training on all data.
eval_args=()
if [[ -f "${q_data_val_path}" ]]; then
    eval_args+=(
        --eval_data_path "${q_data_val_path}"
        --eval_strategy "steps"
        --eval_steps "${eval_steps}"
        --per_device_eval_batch_size 1
    )
else
    echo "[train_dueling_qnet_7b] WARNING: validation file not found; evaluation is disabled."
    eval_args+=(--eval_strategy "no")
fi

python -m torch.distributed.run \
    --nproc_per_node=${node_num} \
    --master_port=${MASTER_PORT:-20002} \
    qlass/train_dueling_q.py \
    --model_name_or_path "${save_dir}${sft_model_name}" \
    --data_path "${q_data_train_path}" \
    "${eval_args[@]}" \
    --bf16 True \
    --output_dir "${output_dir}" \
    --run_name "${run_id}" \
    --report_to "tensorboard" \
    --logging_dir "${logging_dir}" \
    --num_train_epochs "${num_train_epochs}" \
    --per_device_train_batch_size "${micro_batch_size}" \
    --gradient_accumulation_steps "${accumulation_step}" \
    --save_strategy "steps" \
    --save_steps "${save_steps}" \
    --save_total_limit "${save_total_limit}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --warmup_ratio "${warmup_ratio}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --logging_steps "${logging_steps}" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --remove_unused_columns False \
    --seed "${seed}" \
    --dueling_value_loss_coef "${dueling_value_loss_coef}" \
    --dueling_head_hidden_size "${dueling_head_hidden_size}" \
    --min_candidates "${min_candidates}" \
    --drop_zero_variance "${drop_zero_variance}" \
    --drop_state_mismatch "${drop_state_mismatch}" \
    --pad_to_max_length True

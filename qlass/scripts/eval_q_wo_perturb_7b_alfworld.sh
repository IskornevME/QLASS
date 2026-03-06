#!/usr/bin/env bash
set -euo pipefail

model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=2  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

QNET_STEP=14382
sft_model_name=${exp_name}-${model_name}-${task}-sft_run1
q_model_name=${exp_name}-${model_name}-${task}-Q_my
q_model_name_checkpoint=${q_model_name}/infer-checkpoint-${QNET_STEP}

# QNET_ROOT="${MODEL_PATH}${q_model_name}"
# QNET_CKPT="${QNET_ROOT}/checkpoint-${QNET_STEP}"


export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

explore_model_name="${sft_model_name}"

sg_worker_port=21001
sg_worker_port2=21003

mkdir -p logs_q_inf

SERVER_GPUS=(0 2)
WORKER_GPUS=(1 3)
num_server=2

CUDA_VISIBLE_DEVICES="${SERVER_GPUS[0]}" python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port} >> logs_q_inf/${exp_name}-sg_worker_explore.log 2>&1 &
CUDA_VISIBLE_DEVICES="${SERVER_GPUS[1]}" python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port2} >> logs_q_inf/${exp_name}-sg_worker_explore2.log 2>&1 &

sleep 30

BON=2
ICL=1
SPLT=dev
data_prefix=debug
RUN_ID=1

OUT_DIR="data/train/${task}/${explore_model_name}/q_without_perturb/${data_prefix}_bon${BON}_${SPLT}_run_${RUN_ID}/"
mkdir -p "${OUT_DIR}"


slice_num=2
for (( slice_id=0; slice_id<slice_num; slice_id++ )); do
  server_id=$(( slice_id % num_server ))
  worker_gpu="${WORKER_GPUS[$server_id]}"
  agent_cfg="sglang${server_id}"

  CUDA_VISIBLE_DEVICES="${worker_gpu}" python qlass/q_guided_inference.py \
    --agent_config "${agent_cfg}" \
    --agent_path qlass/configs/model/ \
    --qnet_path "${save_dir}${q_model_name_checkpoint}" \
    --exp_name "${exp_name}" \
    --exp_path qlass/configs/task/ \
    --exp_config "${task}" \
    --split "${SPLT}" \
    --slice_num "${slice_num}" \
    --slice_id "${slice_id}" \
    --model_name "${explore_model_name}" \
    --num_icl_examples "${ICL}" \
    --sample_mode bon \
    --best_of_N "${BON}" \
    --n_trajs 3 \
    --force_first \
    --disable_perturb \
    --output_dir "${OUT_DIR}" &

  sleep 10
done
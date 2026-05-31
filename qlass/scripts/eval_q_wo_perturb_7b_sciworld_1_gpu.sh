#!/usr/bin/env bash
set -euo pipefail

model_name=Llama-2-7b-chat-hf
task=sciworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=1  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

QNET_STEP=8870
sft_model_name=${exp_name}-${model_name}-${task}-sft_run1
# sft_model_name=${exp_name}-${model_name}-${task}-sft
q_model_name=${exp_name}-${model_name}-${task}-Q_my_v2
# q_model_name=${exp_name}-${model_name}-${task}-Q
q_model_name_checkpoint=${q_model_name}/infer-checkpoint-${QNET_STEP}
# q_model_name_checkpoint=${q_model_name}


export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

explore_model_name="${sft_model_name}"

sg_worker_port=21001

# mkdir -p logs_q_inf

SERVER_GPU=2
WORKER_GPU=3

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port} >> logs_sciworld/${exp_name}-sq_worker.log 2>&1 &

sleep 30

BON=2
ICL=1
SPLT=dev
data_prefix=debug
RUN_ID=6

slice_num=1
slice_id=0

OUT_DIR="data/train/${task}/${explore_model_name}/q_without_perturb/${data_prefix}_bon${BON}_${SPLT}_run_${RUN_ID}/"
mkdir -p "${OUT_DIR}"

worker_gpu="${WORKER_GPU}"
agent_cfg="sglang"
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
  --model_name "${save_dir}${explore_model_name}" \
  --num_icl_examples "${ICL}" \
  --sample_mode bon \
  --best_of_N "${BON}" \
  --n_trajs 3 \
  --force_first \
  --disable_perturb \
  --output_dir "${OUT_DIR}" \
  --tokenizer_path "/home/m.iskornev/qlass/models/qlass-Llama-2-7b-chat-hf-sciworld-sft_run1" \
  --qnet_max_prompt_tokens 3800 \
  --qnet_keep_first_n 3 \
  --qnet_min_tail_msgs 4

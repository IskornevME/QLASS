#!/usr/bin/env bash
set -euo pipefail

model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SERVER=${NUM_SERVER:-2}

SERVER_GPU_0=${SERVER_GPU_0:-5}
SERVER_GPU_1=${SERVER_GPU_1:-6}

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=4

sft_model_name=${exp_name}-${model_name}-${task}-sft_run1
sg_worker_port0=${SG_WORKER_PORT0:-21001}
sg_worker_port1=${SG_WORKER_PORT1:-21003}

MAX_DEPTH=${MAX_DEPTH:-8}
MIN_PRUNE_DEPTH=${MIN_PRUNE_DEPTH:-3}
NUM_ICL_EXAMPLES=${NUM_ICL_EXAMPLES:-0}
SAMPLES_PER_DEPTH=${SAMPLES_PER_DEPTH:-4}
MAX_CHILD_SAMPLE_ATTEMPTS=${MAX_CHILD_SAMPLE_ATTEMPTS:-8}
MAX_EXPANDED_NODES_PER_TASK=${MAX_EXPANDED_NODES_PER_TASK:-1000}
MAX_ENQUEUED_CHILDREN_PER_NODE=${MAX_ENQUEUED_CHILDREN_PER_NODE:-1}
MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-8}
PROGRESS_LOG_EVERY=${PROGRESS_LOG_EVERY:-50}

RUN_NAME=${RUN_NAME:-d${MAX_DEPTH}_i${NUM_ICL_EXAMPLES}_s${SAMPLES_PER_DEPTH}_mpr${MIN_PRUNE_DEPTH}_attempt${MAX_CHILD_SAMPLE_ATTEMPTS}_enqueue${MAX_ENQUEUED_CHILDREN_PER_NODE}_roll${MAX_ROLLOUT_STEPS}_cap${MAX_EXPANDED_NODES_PER_TASK}}
OUT_DIR=${OUT_DIR:-data/train/${task}/explore_7b_sft_${RUN_NAME}/}

LOG_DIR=${LOG_DIR:-logs/explore_7b_sft_${RUN_NAME}}
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"

echo "[INFO] run_name=${RUN_NAME}" | tee "${LOG_DIR}/launcher.log"
echo "[INFO] out_dir=${OUT_DIR}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] log_dir=${LOG_DIR}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] num_workers=${NUM_WORKERS} num_server=${NUM_SERVER}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] samples_per_depth=${SAMPLES_PER_DEPTH} max_child_sample_attempts=${MAX_CHILD_SAMPLE_ATTEMPTS}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] max_expanded_nodes_per_task=${MAX_EXPANDED_NODES_PER_TASK}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] max_enqueued_children_per_node=${MAX_ENQUEUED_CHILDREN_PER_NODE}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] max_rollout_steps=${MAX_ROLLOUT_STEPS}" | tee -a "${LOG_DIR}/launcher.log"
echo "[INFO] progress_log_every=${PROGRESS_LOG_EVERY}" | tee -a "${LOG_DIR}/launcher.log"


WORKER_PIDS=()
PGIDS=()

start_server() {
  local gpu="$1"
  local port="$2"
  local name="$3"
  local log_file="${LOG_DIR}/${name}.log"

  echo "[INFO] Starting SGLang ${name}: gpu=${gpu} port=${port} log=${log_file}" | tee -a "${LOG_DIR}/launcher.log"

  setsid bash -c "
    CUDA_VISIBLE_DEVICES='${gpu}' exec python3 -m sglang.launch_server \
      --model-path '${save_dir}${sft_model_name}' \
      --port '${port}'
  " > "${log_file}" 2>&1 &

  local pid=$!
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"

  PGIDS+=("${pgid}")

  echo "[INFO] ${name}: pid=${pid} pgid=${pgid}" | tee -a "${LOG_DIR}/launcher.log"
}

cleanup() {
  local exit_code=$?
  echo "[INFO] Cleanup exit_code=${exit_code}" | tee -a "${LOG_DIR}/launcher.log"

  for pgid in "${PGIDS[@]:-}"; do
    if [[ -n "${pgid}" ]] && kill -0 "-${pgid}" 2>/dev/null; then
      echo "[INFO] Stopping process group ${pgid}" | tee -a "${LOG_DIR}/launcher.log"
      kill -TERM "-${pgid}" 2>/dev/null || true
      sleep 5
      if kill -0 "-${pgid}" 2>/dev/null; then
        kill -KILL "-${pgid}" 2>/dev/null || true
      fi
    fi
  done

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

start_server "${SERVER_GPU_0}" "${sg_worker_port0}" "sg_worker_explore0"
start_server "${SERVER_GPU_1}" "${sg_worker_port1}" "sg_worker_explore1"

echo "[INFO] Sleeping 60s for SGLang servers..." | tee -a "${LOG_DIR}/launcher.log"
sleep 60

for ((i=0; i<${NUM_WORKERS}; i=i+1)); do
  server_id=$((i % NUM_SERVER))
  worker_log="${LOG_DIR}/worker_${i}_of_${NUM_WORKERS}.log"

  echo "[INFO] Starting worker ${i}/${NUM_WORKERS}, server=${server_id}, log=${worker_log}" | tee -a "${LOG_DIR}/launcher.log"

  CUDA_VISIBLE_DEVICES="" python qlass/explore_sft_agent.py \
    --agent_config "sglang_explore${server_id}" \
    --agent_path qlass/configs/model/ \
    --exp_name "${exp_name}" \
    --exp_path qlass/configs/task/ \
    --exp_config "${task}" \
    --split train \
    --slice_num "${NUM_WORKERS}" \
    --slice_id "${i}" \
    --model_name "${sft_model_name}" \
    --max_depth "${MAX_DEPTH}" \
    --min_prune_depth "${MIN_PRUNE_DEPTH}" \
    --num_icl_examples "${NUM_ICL_EXAMPLES}" \
    --samples_per_depth "${SAMPLES_PER_DEPTH}" \
    --max_child_sample_attempts "${MAX_CHILD_SAMPLE_ATTEMPTS}" \
    --max_expanded_nodes_per_task "${MAX_EXPANDED_NODES_PER_TASK}" \
    --max_enqueued_children_per_node "${MAX_ENQUEUED_CHILDREN_PER_NODE}" \
    --max_rollout_steps "${MAX_ROLLOUT_STEPS}" \
    --progress_log_every "${PROGRESS_LOG_EVERY}" \
    --output_dir "${OUT_DIR}" \
    > "${worker_log}" 2>&1 &

  WORKER_PIDS+=("$!")
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] Process pid=${pid} failed" | tee -a "${LOG_DIR}/launcher.log"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] At least one process failed. See ${LOG_DIR}" | tee -a "${LOG_DIR}/launcher.log"
  exit 1
fi

echo "[INFO] Explore finished successfully. Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/launcher.log"

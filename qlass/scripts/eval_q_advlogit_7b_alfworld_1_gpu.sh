#!/usr/bin/env bash
set -euo pipefail

model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

QNET_STEP=14382
# sft_model_name=${exp_name}-${model_name}-${task}-sft_run1
sft_model_name=${exp_name}-${model_name}-${task}-sft
# q_model_name=${exp_name}-${model_name}-${task}-Q_my
q_model_name=${exp_name}-${model_name}-${task}-Q
# q_model_name_checkpoint=${q_model_name}/infer-checkpoint-${QNET_STEP}
q_model_name_checkpoint=${q_model_name}


export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

explore_model_name="${sft_model_name}"

sg_worker_port=21001

mkdir -p logs_q_inf

SERVER_GPU=2
WORKER_GPU=3

setsid bash -c "
  CUDA_VISIBLE_DEVICES='${SERVER_GPU}' exec python3 -m sglang.launch_server \
    --model-path '${save_dir}${sft_model_name}' \
    --port '${sg_worker_port}'
" >> logs_q_inf/${exp_name}-sg_worker_advlogit.log 2>&1 &

SGLANG_PID=$!
SGLANG_PGID="$(ps -o pgid= -p "${SGLANG_PID}" | tr -d ' ')"

echo "[INFO] Started SGLang server:"
echo "       PID=${SGLANG_PID}"
echo "       PGID=${SGLANG_PGID}"
echo "       GPU=${SERVER_GPU}"
echo "       PORT=${sg_worker_port}"
cleanup() {
  exit_code=$?

  echo "[INFO] Cleanup triggered with exit_code=${exit_code}"

  if [[ -n "${SGLANG_PGID:-}" ]]; then
    echo "[INFO] Stopping SGLang process group PGID=${SGLANG_PGID}"

    if kill -0 "-${SGLANG_PGID}" 2>/dev/null; then
      echo "[INFO] Sending SIGTERM to process group ${SGLANG_PGID}"
      kill -TERM "-${SGLANG_PGID}" 2>/dev/null || true

      for _ in {1..10}; do
        if ! kill -0 "-${SGLANG_PGID}" 2>/dev/null; then
          echo "[INFO] SGLang process group ${SGLANG_PGID} stopped gracefully"
          break
        fi
        sleep 1
      done

      if kill -0 "-${SGLANG_PGID}" 2>/dev/null; then
        echo "[WARN] SGLang process group ${SGLANG_PGID} is still alive; sending SIGKILL"
        kill -KILL "-${SGLANG_PGID}" 2>/dev/null || true
      fi
    else
      echo "[INFO] SGLang process group ${SGLANG_PGID} is already stopped"
    fi
  fi

  # Extra safety: if the port is still occupied, kill the process listening on it.
  # This should normally be unnecessary if process-group cleanup worked.
  if command -v lsof >/dev/null 2>&1; then
    PORT_PIDS="$(lsof -tiTCP:"${sg_worker_port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${PORT_PIDS}" ]]; then
      echo "[WARN] Port ${sg_worker_port} is still occupied by PID(s): ${PORT_PIDS}"
      echo "[WARN] Killing remaining listener(s) on port ${sg_worker_port}"
      kill -TERM ${PORT_PIDS} 2>/dev/null || true
      sleep 2
      kill -KILL ${PORT_PIDS} 2>/dev/null || true
    fi
  else
    echo "[WARN] lsof is not installed; cannot check remaining listeners on port ${sg_worker_port}"
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM
sleep 30

BON=${BON:-4}
ICL=1
SPLT=test
data_prefix=debug
RUN_ID=${RUN_ID:-0}

NUM_WORKERS=${NUM_WORKERS:-4}
N_TRAJS=${N_TRAJS:-3}

# Actor + Q-Adv correction:
# corrected_score = ACTOR_LOGPROB_COEF * actor_logprob_mean + Q_ADV_BETA * zscore(QNet/env scores)
SELECTION_STRATEGY=q_adv_logit_argmax
ACTOR_LOGPROB_TYPE=mean
ACTOR_LOGPROB_COEF=${ACTOR_LOGPROB_COEF:-1.0}
Q_ADV_BETA=${Q_ADV_BETA:-1.0}
Q_ADV_EPS=${Q_ADV_EPS:-1e-6}
Q_ADV_CLIP=${Q_ADV_CLIP:-5.0}

# Candidate generation with one canonical prompt:
# raw samples per round = BON * CANDIDATE_OVERSAMPLE_FACTOR
CANDIDATE_OVERSAMPLE_FACTOR=${CANDIDATE_OVERSAMPLE_FACTOR:-3}
CANDIDATE_MAX_ROUNDS=${CANDIDATE_MAX_ROUNDS:-1}

slice_num="${NUM_WORKERS}"
slice_id=0

OUT_DIR="data/train/${task}/${explore_model_name}/q_advlogit/${data_prefix}_${SELECTION_STRATEGY}_bon${BON}_beta${Q_ADV_BETA}_actor${ACTOR_LOGPROB_COEF}_${ACTOR_LOGPROB_TYPE}_oversample${CANDIDATE_OVERSAMPLE_FACTOR}x${CANDIDATE_MAX_ROUNDS}_workers${NUM_WORKERS}_ntrajs${N_TRAJS}_${SPLT}_run_${RUN_ID}/"
mkdir -p "${OUT_DIR}"

RUN_LOG="${OUT_DIR}/q_guided_inference.log"
echo "  run_log=${RUN_LOG}"

worker_gpu="${WORKER_GPU}"
agent_cfg="sglang"

PIDS=()

for slice_id in $(seq 0 $((NUM_WORKERS - 1))); do
  RUN_LOG="${OUT_DIR}/q_guided_inference_slice${slice_id}of${slice_num}.log"
  echo "[INFO] Starting worker slice_id=${slice_id}/${slice_num}, log=${RUN_LOG}"

  (
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
      --selection_strategy "${SELECTION_STRATEGY}" \
      --best_of_N "${BON}" \
      --actor_logprob_type "${ACTOR_LOGPROB_TYPE}" \
      --actor_logprob_coef "${ACTOR_LOGPROB_COEF}" \
      --q_adv_beta "${Q_ADV_BETA}" \
      --q_adv_eps "${Q_ADV_EPS}" \
      --q_adv_clip "${Q_ADV_CLIP}" \
      --candidate_oversample_factor "${CANDIDATE_OVERSAMPLE_FACTOR}" \
      --candidate_max_rounds "${CANDIDATE_MAX_ROUNDS}" \
      --n_trajs "${N_TRAJS}" \
      --force_first \
      --disable_perturb \
      --verbose \
      --output_dir "${OUT_DIR}" \
      2>&1 | tee "${RUN_LOG}"
  ) &

  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] Worker PID=${pid} failed"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] At least one worker failed"
  exit 1
fi

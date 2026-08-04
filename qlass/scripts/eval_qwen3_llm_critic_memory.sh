#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-alfworld}"
TASK="${BENCHMARK}"
EXP_NAME="${EXP_NAME:-qlass}"

MODEL_PROFILE="${MODEL_PROFILE:-qwen3_4b_instruct_2507}"

source "qlass/configs/model_profiles/${MODEL_PROFILE}.sh"

SGLANG_PYTHON="${SGLANG_PYTHON:-/home/m.iskornev/miniforge3/envs/qlass_qwen_server/bin/python}"
QLASS_PYTHON="${QLASS_PYTHON:-/home/m.iskornev/miniforge3/envs/my_env}"

SERVER_GPU="${SERVER_GPU:-0}"
SGLANG_PORT="${SGLANG_PORT:-21003}"
SERVER_TP="${SERVER_TP:-1}"

POLICY_SERVER_ADDRESS="${POLICY_SERVER_ADDRESS:-http://127.0.0.1:${SGLANG_PORT}}"

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${SGLANG_PYTHON}" \
  -m sglang.launch_server \
  --model-path "${POLICY_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${SGLANG_PORT}" \
  --tp-size "${SERVER_TP}" \
  --context-length "${SGLANG_CONTEXT_LENGTH}" > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!






for attempt in $(seq 1 120); do
  if curl -fsS \
    "${POLICY_SERVER_ADDRESS}/v1/models" \
    >/dev/null; then
    echo "[INFO] SGLang server is ready."
    break
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] SGLang server stopped."
    tail -n 100 "${SERVER_LOG}" || true
    exit 1
  fi

  sleep 2
done




CUDA_VISIBLE_DEVICES="${WORKER_GPU}" "${QLASS_PYTHON}" qlass/q_guided_inference.py \
  --agent_config "${AGENT_CONFIG}" \
  --agent_path qlass/configs/model/ \
  
  --policy_server_address "${POLICY_SERVER_ADDRESS}" \
  --model_name "${POLICY_MODEL_NAME}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  
  --policy_max_prompt_tokens "${POLICY_MAX_PROMPT_TOKENS}" \
  --policy_max_new_tokens "${POLICY_MAX_NEW_TOKENS}" \
  --policy_keep_first_n "${POLICY_KEEP_FIRST_N}" \
  --policy_min_tail_msgs "${POLICY_MIN_TAIL_MSGS}" \
  --policy_temperature "${POLICY_TEMPERATURE}" \
  --policy_top_p "${POLICY_TOP_P}" \
  --policy_top_k "${POLICY_TOP_K}" \
  --policy_min_p "${POLICY_MIN_P}" \
  --policy_presence_penalty "${POLICY_PRESENCE_PENALTY}" \
  
  --critic_backend "${CRITIC_BACKEND}" \
  
  --critic_server_address "${POLICY_SERVER_ADDRESS}" \
  --critic_model_name "${CRITIC_MODEL_NAME}" \
  --critic_tokenizer_path "${CRITIC_TOKENIZER_PATH}" \
  --critic_max_prompt_tokens "${CRITIC_MAX_PROMPT_TOKENS}" \
  --critic_max_new_tokens "${CRITIC_MAX_NEW_TOKENS}" \
  --critic_keep_first_n "${CRITIC_KEEP_FIRST_N}" \
  --critic_min_tail_msgs "${CRITIC_MIN_TAIL_MSGS}" \
  --critic_temperature "${CRITIC_TEMPERATURE}" \
  --critic_top_p "${CRITIC_TOP_P}" \
  --critic_top_k "${CRITIC_TOP_K}" \
  --critic_failure_mode "${CRITIC_FAILURE_MODE}" \
  --critic_neutral_score "${CRITIC_NEUTRAL_SCORE}" \
  --critic_max_parse_attempts "${CRITIC_MAX_PARSE_ATTEMPTS}" \
  --critic_request_timeout "${CRITIC_REQUEST_TIMEOUT}" \
  --critic_retry_delay_seconds "${CRITIC_RETRY_DELAY_SECONDS}" \

  --exp_name "${EXP_NAME}" \
  --exp_path qlass/configs/task/ \
  --benchmark "${BENCHMARK}" \
  --max_steps "${MAX_STEPS}" \
  --split "${SPLIT}" \
  --slice_num "${SLICE_NUM}" \
  --slice_id "${SLICE_ID}" \
  --num_icl_examples "${ICL}" \
  --sample_mode bon \
  --best_of_N "${BON}" \
  --n_trajs "${N_TRAJS}" \
  "${MAX_TASKS_ARGS[@]}" \
  --force_first \
  --disable_perturb \
  --verbose \
  --output_dir "${OUT_DIR}" \
  "${MEMORY_ARGS[@]}" \
  "${SELECTION_ARGS[@]}" \
  2>&1 | tee "${INFERENCE_LOG}"
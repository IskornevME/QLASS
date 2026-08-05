#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-${TASK:-alfworld}}"
case "${BENCHMARK}" in
  alfworld|sciworld)
    ;;
  *)
    echo "[ERROR] Unsupported benchmark: ${BENCHMARK}" >&2
    exit 1
    ;;
esac

TASK="${BENCHMARK}"
EXP_NAME="${EXP_NAME:-qlass}"

# -----------------------------------------------------------------------------
# Model-specific defaults.
# Values exported by the caller take precedence over defaults in the profile.
# -----------------------------------------------------------------------------
MODEL_PROFILE="${MODEL_PROFILE:-qwen3_4b_instruct_2507}"
MODEL_PROFILE_PATH="/home/m.iskornev/qlass/QLASS/qlass/configs/model_profiles/${MODEL_PROFILE}.sh"

if [[ ! -f "${MODEL_PROFILE_PATH}" ]]; then
  echo "[ERROR] Model profile not found: ${MODEL_PROFILE_PATH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${MODEL_PROFILE_PATH}"

if [[ ! -d "${POLICY_MODEL_PATH}" ]]; then
  echo "[ERROR] Qwen checkpoint directory not found: ${POLICY_MODEL_PATH}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Python environments.
# SGLang server and QLASS client may live in separate environments.
# -----------------------------------------------------------------------------
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/m.iskornev/miniforge3/envs/qlass_qwen_server/bin/python}"
QLASS_PYTHON="${QLASS_PYTHON:-/home/m.iskornev/miniforge3/envs/my_env/bin/python}"


# -----------------------------------------------------------------------------
# GPU allocation.
#
# llm_judge:
#   GPU SERVER_GPU: Qwen actor + prompted Qwen critic via one SGLang server.
#   QLASS client does not load a neural model locally.
#
# none:
#   GPU SERVER_GPU: Qwen actor via SGLang.
#   QLASS client does not load a neural model locally.
#
# qnet:
#   GPU SERVER_GPU: Qwen actor via SGLang.
#   GPU WORKER_GPU: original trained Llama QNet loaded locally by q_guided_inference.py.
# -----------------------------------------------------------------------------
SERVER_GPU="${SERVER_GPU:-0}"
# Preserve the old behavior for actor-only and LLM-judge runs: no local GPU is exposed to q_guided_inference.py.
# For QNet, use a second physical GPU by default.
if [[ -z "${WORKER_GPU+x}" ]]; then
  if [[ "${CRITIC_BACKEND}" == "qnet" ]]; then
    WORKER_GPU="1"
  else
    WORKER_GPU=""
  fi
fi

SGLANG_PORT="${SGLANG_PORT:-21003}"
POLICY_SERVER_ADDRESS="${POLICY_SERVER_ADDRESS:-http://127.0.0.1:${SGLANG_PORT}}"
KEEP_SERVER_ALIVE="${KEEP_SERVER_ALIVE:-0}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-120}"
SERVER_READY_SLEEP_SECONDS="${SERVER_READY_SLEEP_SECONDS:-2}"

# -----------------------------------------------------------------------------
# QLASS evaluation settings.
# -----------------------------------------------------------------------------
BON="${BON:-2}"
ICL="${ICL:-1}"
N_TRAJS="${N_TRAJS:-3}"
MAX_STEPS="${MAX_STEPS:-40}"
MAX_TASKS="${MAX_TASKS:-}"

SPLIT="${SPLIT:-test}"
SLICE_NUM="${SLICE_NUM:-1}"
SLICE_ID="${SLICE_ID:-0}"
DATA_PREFIX="${DATA_PREFIX:-qwen}"
RUN_ID="${RUN_ID:-0}"

MAX_TASKS_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  if ! [[ "${MAX_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_TASKS must be a positive integer, got: ${MAX_TASKS}" >&2
    exit 1
  fi
  MAX_TASKS_ARGS=(--max_tasks "${MAX_TASKS}")
fi

# -----------------------------------------------------------------------------
# Episodic-memory and action-augmentation settings.
# ENABLE_MEMORY=0 gives the Qwen QLASS-style baseline.
# For augmentation-only use ENABLE_MEMORY=1,
# ENABLE_MEMORY_ACTION_AUGMENTATION=1, MEMORY_WEIGHT=0.0.
# -----------------------------------------------------------------------------
ENABLE_MEMORY="${ENABLE_MEMORY:-0}"
MEMORY_WEIGHT="${MEMORY_WEIGHT:-0.0}"
MEMORY_GAMMA="${MEMORY_GAMMA:-0.97}"
MEMORY_TOP_K="${MEMORY_TOP_K:-10}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-0.75}"
MEMORY_TERMINAL_STEP_PENALTY="${MEMORY_TERMINAL_STEP_PENALTY:-0.0}"
MEMORY_SCOPE="${MEMORY_SCOPE:-global}"
MEMORY_REWARD_MODE="${MEMORY_REWARD_MODE:-terminal_only}"
RESET_MEMORY="${RESET_MEMORY:-1}"
DISABLE_DYNAMIC_THRESHOLD="${DISABLE_DYNAMIC_THRESHOLD:-0}"
PREFER_TERMINAL_SUCCESS="${PREFER_TERMINAL_SUCCESS:-0}"

ENABLE_MEMORY_ACTION_AUGMENTATION="${ENABLE_MEMORY_ACTION_AUGMENTATION:-0}"
MEMORY_MAX_AUGMENTED_ACTIONS="${MEMORY_MAX_AUGMENTED_ACTIONS:-2}"
MEMORY_AUG_MAX_PER_CANONICAL="${MEMORY_AUG_MAX_PER_CANONICAL:-1}"
MEMORY_AUGMENTATION_MIN_MEAN_RETURN="${MEMORY_AUGMENTATION_MIN_MEAN_RETURN:-1e-12}"
MEMORY_AUGMENTED_ACTION_FORMAT="${MEMORY_AUGMENTED_ACTION_FORMAT:-retrieved_thought_exact_only}"

if [[ "${ENABLE_MEMORY_ACTION_AUGMENTATION}" == "1" && "${ENABLE_MEMORY}" != "1" ]]; then
  echo "[ERROR] ENABLE_MEMORY_ACTION_AUGMENTATION=1 requires ENABLE_MEMORY=1." >&2
  exit 1
fi

case "${CRITIC_BACKEND}" in
  llm_judge)
    ;;
  none)
    if [[ "${BON}" != "1" ]]; then
      echo "[ERROR] CRITIC_BACKEND=none requires BON=1." >&2
      exit 1
    fi
    if [[ "${ENABLE_MEMORY}" == "1" ]]; then
      echo "[ERROR] CRITIC_BACKEND=none is intended for the actor-only baseline; set ENABLE_MEMORY=0." >&2
      exit 1
    fi
    ;;
  qnet)
  if [[ -z "${WORKER_GPU}" ]]; then
    echo \
      "[ERROR] CRITIC_BACKEND=qnet requires a non-empty WORKER_GPU." \
      >&2
    exit 1
  fi

  if [[ "${SERVER_GPU}" == "${WORKER_GPU}" ]]; then
    echo \
      "[ERROR] Qwen actor and QNet critic must use different GPUs. " \
      "Got SERVER_GPU=${SERVER_GPU} and WORKER_GPU=${WORKER_GPU}." \
      >&2
    exit 1
  fi

  if [[ ! -d "${QNET_PATH}" ]]; then
    echo \
      "[ERROR] QNet checkpoint directory not found: ${QNET_PATH}" \
      >&2
    exit 1
  fi

  if [[ ! -d "${QNET_TOKENIZER_PATH}" ]]; then
    echo \
      "[ERROR] Llama QNet tokenizer directory not found: " \
      "${QNET_TOKENIZER_PATH}" \
      >&2
    exit 1
  fi

  if [[ -z "${QNET_MODEL_NAME}" ]]; then
    echo \
      "[ERROR] QNET_MODEL_NAME must not be empty." \
      >&2
    exit 1
  fi
  ;;
esac

if [[ "${PREFER_TERMINAL_SUCCESS}" == "1" ]]; then
  TERMINAL_TAG="terminal_override"
else
  TERMINAL_TAG="no_terminal_override"
fi

if [[ "${ENABLE_MEMORY_ACTION_AUGMENTATION}" == "1" ]]; then
  AUG_TAG="aug${MEMORY_MAX_AUGMENTED_ACTIONS}_percanon${MEMORY_AUG_MAX_PER_CANONICAL}_${MEMORY_AUGMENTED_ACTION_FORMAT}"
else
  AUG_TAG="no_aug"
fi

if [[ "${ENABLE_MEMORY}" == "1" ]]; then
  MEMORY_TAG="memory_reward${MEMORY_REWARD_MODE}_lambda${MEMORY_WEIGHT}_gamma${MEMORY_GAMMA}_k${MEMORY_TOP_K}_thr${MEMORY_THRESHOLD}_${MEMORY_SCOPE}"
else
  MEMORY_TAG="no_memory"
fi

RUN_TAG="${DATA_PREFIX}_${CRITIC_BACKEND}_bon${BON}_traj${N_TRAJS}_steps${MAX_STEPS}_${SPLIT}_run${RUN_ID}_${MEMORY_TAG}_${TERMINAL_TAG}_${AUG_TAG}"

if [[ "${CRITIC_BACKEND}" == "qnet" ]]; then
  RUN_FAMILY="qwen_llama_qnet"
else
  # Preserve the existing output location for llm_judge and actor-only runs.
  RUN_FAMILY="qwen_llm_critic"
fi

OUT_DIR="${OUT_DIR:-data/train/${TASK}/${POLICY_MODEL_NAME}/${RUN_FAMILY}/${RUN_TAG}/}"
case "${OUT_DIR}" in
  */) ;;
  *) OUT_DIR="${OUT_DIR}/" ;;
esac

MEMORY_DIR="${MEMORY_DIR:-${OUT_DIR}memory}"
MEMORY_LOG_FILE="${MEMORY_LOG_FILE:-${OUT_DIR}${SLICE_ID}of${SLICE_NUM}_slices_bon_memory_correction_decisions.jsonl}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}logs}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
if [[ "${ENABLE_MEMORY}" == "1" ]]; then
  mkdir -p "${MEMORY_DIR}"
fi

SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/${EXP_NAME}_qwen_sglang_server.log}"
INFERENCE_LOG="${INFERENCE_LOG:-${LOG_DIR}/${EXP_NAME}_qwen_inference.log}"

# -----------------------------------------------------------------------------
# Build optional CLI argument groups.
# -----------------------------------------------------------------------------
MEMORY_ARGS=()
if [[ "${ENABLE_MEMORY}" == "1" ]]; then
  MEMORY_ARGS=(
    --enable_memory_correction
    --memory_dir "${MEMORY_DIR}"
    --memory_log_file "${MEMORY_LOG_FILE}"
    --memory_weight "${MEMORY_WEIGHT}"
    --memory_gamma "${MEMORY_GAMMA}"
    --memory_top_k "${MEMORY_TOP_K}"
    --memory_threshold "${MEMORY_THRESHOLD}"
    --memory_reward_mode "${MEMORY_REWARD_MODE}"
    --no-memory_use_env_reward_fallback
    --memory_terminal_step_penalty "${MEMORY_TERMINAL_STEP_PENALTY}"
    --memory_scope "${MEMORY_SCOPE}"
  )

  if [[ "${RESET_MEMORY}" == "1" ]]; then
    MEMORY_ARGS+=(--reset_memory)
  fi

  if [[ "${DISABLE_DYNAMIC_THRESHOLD}" == "1" ]]; then
    MEMORY_ARGS+=(--memory_disable_dynamic_threshold)
  fi
fi

if [[ "${ENABLE_MEMORY_ACTION_AUGMENTATION}" == "1" ]]; then
  MEMORY_ARGS+=(
    --enable_memory_action_augmentation
    --memory_max_augmented_actions "${MEMORY_MAX_AUGMENTED_ACTIONS}"
    --memory_aug_max_per_canonical "${MEMORY_AUG_MAX_PER_CANONICAL}"
    --memory_augmentation_min_mean_return "${MEMORY_AUGMENTATION_MIN_MEAN_RETURN}"
    --memory_augmented_action_format "${MEMORY_AUGMENTED_ACTION_FORMAT}"
  )
fi

SELECTION_ARGS=()
if [[ "${PREFER_TERMINAL_SUCCESS}" == "1" ]]; then
  SELECTION_ARGS+=(--prefer_terminal_success)
fi

CRITIC_ARGS=(--critic_backend "${CRITIC_BACKEND}")
if [[ "${CRITIC_BACKEND}" == "llm_judge" ]]; then
  CRITIC_ARGS+=(
    --critic_server_address "${POLICY_SERVER_ADDRESS}"
    --critic_model_name "${CRITIC_MODEL_NAME}"
    --critic_tokenizer_path "${CRITIC_TOKENIZER_PATH}"
    --critic_max_prompt_tokens "${CRITIC_MAX_PROMPT_TOKENS}"
    --critic_max_new_tokens "${CRITIC_MAX_NEW_TOKENS}"
    --critic_keep_first_n "${CRITIC_KEEP_FIRST_N}"
    --critic_min_tail_msgs "${CRITIC_MIN_TAIL_MSGS}"
    --critic_temperature "${CRITIC_TEMPERATURE}"
    --critic_top_p "${CRITIC_TOP_P}"
    --critic_top_k "${CRITIC_TOP_K}"
    --critic_failure_mode "${CRITIC_FAILURE_MODE}"
    --critic_neutral_score "${CRITIC_NEUTRAL_SCORE}"
    --critic_max_parse_attempts "${CRITIC_MAX_PARSE_ATTEMPTS}"
    --critic_request_timeout "${CRITIC_REQUEST_TIMEOUT}"
    --critic_retry_delay_seconds "${CRITIC_RETRY_DELAY_SECONDS}"
  )

  if [[ -n "${CRITIC_API_MODEL_NAME:-}" ]]; then
    CRITIC_ARGS+=(--critic_api_model_name "${CRITIC_API_MODEL_NAME}")
  fi
elif [[ "${CRITIC_BACKEND}" == "qnet" ]]; then
  CRITIC_ARGS+=(
    --qnet_path "${QNET_PATH}"
    --qnet_tokenizer_path "${QNET_TOKENIZER_PATH}"
    --qnet_model_name "${QNET_MODEL_NAME}"
    --qnet_max_prompt_tokens "${QNET_MAX_PROMPT_TOKENS}"
    --qnet_keep_first_n "${QNET_KEEP_FIRST_N}"
    --qnet_min_tail_msgs "${QNET_MIN_TAIL_MSGS}"
  )
fi

# q_guided_inference.py currently reads OPENAI_API_KEY at import time even
# when perturbation is disabled. A dummy value is sufficient for this run.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# -----------------------------------------------------------------------------
# Configuration summary.
# -----------------------------------------------------------------------------
echo "[CONFIG] Benchmark:        ${BENCHMARK}"
echo "[CONFIG] Policy model:     ${POLICY_MODEL_PATH}"
echo "[CONFIG] Agent config:     ${AGENT_CONFIG}"
echo "[CONFIG] Critic backend:   ${CRITIC_BACKEND}"
if [[ "${CRITIC_BACKEND}" == "qnet" ]]; then
  echo "[CONFIG] QNet checkpoint: ${QNET_PATH}"
  echo "[CONFIG] QNet tokenizer:  ${QNET_TOKENIZER_PATH}"
  echo "[CONFIG] QNet model name: ${QNET_MODEL_NAME}"
  echo "[CONFIG] QNet GPU:        ${WORKER_GPU}"
fi
echo "[CONFIG] Server:           gpu=${SERVER_GPU}, tp=${SERVER_TP}, address=${POLICY_SERVER_ADDRESS}"
echo "[CONFIG] Client GPU:       ${WORKER_GPU:-hidden}"
echo "[CONFIG] BON / trajectories: ${BON} / ${N_TRAJS}"
echo "[CONFIG] Split / max steps: ${SPLIT} / ${MAX_STEPS}"
echo "[CONFIG] Output dir:       ${OUT_DIR}"
echo "[CONFIG] Memory enabled:   ${ENABLE_MEMORY}"
echo "[CONFIG] Memory weight:    ${MEMORY_WEIGHT}"
echo "[CONFIG] Augmentation:     ${ENABLE_MEMORY_ACTION_AUGMENTATION}"
echo "[CONFIG] Memory dir:       ${MEMORY_DIR}"
echo "[CONFIG] Max tasks:        ${MAX_TASKS:-all}"

# -----------------------------------------------------------------------------
# Start the SGLang server and guarantee cleanup on exit.
# -----------------------------------------------------------------------------
SERVER_PID=""
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${SERVER_PID}" && "${KEEP_SERVER_ALIVE}" != "1" ]] \
    && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping SGLang server PID=${SERVER_PID}."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi

  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

if curl -fsS --max-time 2 "${POLICY_SERVER_ADDRESS}/v1/models" >/dev/null 2>&1; then
  echo "[ERROR] Another server already responds at ${POLICY_SERVER_ADDRESS}." >&2
  echo "[ERROR] Stop it or select another SGLANG_PORT to avoid using the wrong model." >&2
  exit 1
fi

echo "[INFO] Starting SGLang server on ${POLICY_SERVER_ADDRESS}."
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${SGLANG_PYTHON}" \
  -m sglang.launch_server \
  --model-path "${POLICY_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${SGLANG_PORT}" \
  --tp-size "${SERVER_TP}" \
  --context-length "${SGLANG_CONTEXT_LENGTH}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

SERVER_READY=0
for attempt in $(seq 1 "${SERVER_READY_ATTEMPTS}"); do
  if curl -fsS "${POLICY_SERVER_ADDRESS}/v1/models" >/dev/null; then
    SERVER_READY=1
    echo "[INFO] SGLang server is ready after attempt ${attempt}."
    break
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] SGLang server terminated during startup." >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi

  sleep "${SERVER_READY_SLEEP_SECONDS}"
done

if [[ "${SERVER_READY}" != "1" ]]; then
  echo "[ERROR] SGLang server did not become ready in time." >&2
  tail -n 100 "${SERVER_LOG}" >&2 || true
  exit 1
fi

# -----------------------------------------------------------------------------
# Build and run q_guided_inference.py.
# Arrays avoid fragile backslash continuations and preserve empty groups.
# -----------------------------------------------------------------------------
INFERENCE_ARGS=(
  qlass/q_guided_inference.py
  --agent_config "${AGENT_CONFIG}"
  --agent_path qlass/configs/model/
  --policy_server_address "${POLICY_SERVER_ADDRESS}"
  --model_name "${POLICY_MODEL_NAME}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --policy_max_prompt_tokens "${POLICY_MAX_PROMPT_TOKENS}"
  --policy_max_new_tokens "${POLICY_MAX_NEW_TOKENS}"
  --policy_keep_first_n "${POLICY_KEEP_FIRST_N}"
  --policy_min_tail_msgs "${POLICY_MIN_TAIL_MSGS}"
  --policy_temperature "${POLICY_TEMPERATURE}"
  --policy_top_p "${POLICY_TOP_P}"
  --policy_top_k "${POLICY_TOP_K}"
  --policy_min_p "${POLICY_MIN_P}"
  --policy_presence_penalty "${POLICY_PRESENCE_PENALTY}"
  "${CRITIC_ARGS[@]}"
  --exp_name "${EXP_NAME}"
  --exp_path qlass/configs/task/
  --benchmark "${BENCHMARK}"
  --max_steps "${MAX_STEPS}"
  --split "${SPLIT}"
  --slice_num "${SLICE_NUM}"
  --slice_id "${SLICE_ID}"
  --num_icl_examples "${ICL}"
  --sample_mode bon
  --best_of_N "${BON}"
  --n_trajs "${N_TRAJS}"
  "${MAX_TASKS_ARGS[@]}"
  --force_first
  --disable_perturb
  --verbose
  --output_dir "${OUT_DIR}"
  "${MEMORY_ARGS[@]}"
  "${SELECTION_ARGS[@]}"
)

echo "[INFO] Running Qwen QLASS inference."
CUDA_VISIBLE_DEVICES="${WORKER_GPU}" \
  "${QLASS_PYTHON}" "${INFERENCE_ARGS[@]}" \
  2>&1 | tee "${INFERENCE_LOG}"

echo "[INFO] Inference completed."
echo "[INFO] Trajectories and tree: ${OUT_DIR}"
echo "[INFO] Runtime log:           ${INFERENCE_LOG}"
if [[ "${ENABLE_MEMORY}" == "1" ]]; then
  echo "[INFO] Memory records:        ${MEMORY_DIR}"
  echo "[INFO] Memory diagnostics:    ${MEMORY_LOG_FILE}"
fi
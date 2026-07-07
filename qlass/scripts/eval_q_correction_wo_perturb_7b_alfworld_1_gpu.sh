#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
TASK="${TASK:-alfworld}"
EXP_NAME="${EXP_NAME:-qlass}"

SFT_MODEL_NAME="${SFT_MODEL_NAME:-${EXP_NAME}-${MODEL_NAME}-${TASK}-sft}"
Q_MODEL_NAME="${Q_MODEL_NAME:-${EXP_NAME}-${MODEL_NAME}-${TASK}-Q}"
# To use an inference checkpoint instead, launch with e.g.:
#   Q_MODEL_NAME_CHECKPOINT="qlass-Llama-2-7b-chat-hf-alfworld-Q/infer-checkpoint-14382" bash ...
Q_MODEL_NAME_CHECKPOINT="${Q_MODEL_NAME_CHECKPOINT:-${Q_MODEL_NAME}}"

SFT_MODEL_PATH="${MODEL_PATH%/}/${SFT_MODEL_NAME}"
QNET_PATH="${MODEL_PATH%/}/${Q_MODEL_NAME_CHECKPOINT}"

# The uploaded baseline used two device ids (server=0 and QNet worker=1).
# Preserve that default. For an actual single-GPU launch, explicitly set
# SERVER_GPU=0 WORKER_GPU=0, provided both models fit into GPU memory.
SERVER_GPU="${SERVER_GPU:-0}"
WORKER_GPU="${WORKER_GPU:-1}"
SGLANG_PORT="${SGLANG_PORT:-21001}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-30}"
# If we don't want server to end in case of fail
KEEP_SERVER_ALIVE="${KEEP_SERVER_ALIVE:-0}"

BON="${BON:-2}"
ICL="${ICL:-1}"
N_TRAJS="${N_TRAJS:-3}"
SPLIT="${SPLIT:-test}"
SLICE_NUM="${SLICE_NUM:-1}"
SLICE_ID="${SLICE_ID:-0}"
DATA_PREFIX="${DATA_PREFIX:-debug}"
RUN_ID="${RUN_ID:-0}"
AGENT_CONFIG="${AGENT_CONFIG:-sglang}"

# ---------------------------------------------------------------------------
# Correction-memory experiment settings.
# A unique run tag and an isolated memory directory prevent contamination
# between different lambda/gamma/reward/override configurations.
# ---------------------------------------------------------------------------
MEMORY_WEIGHT="${MEMORY_WEIGHT:-0.0}"
MEMORY_GAMMA="${MEMORY_GAMMA:-0.97}"
MEMORY_TOP_K="${MEMORY_TOP_K:-10}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-0.75}"
MEMORY_TERMINAL_STEP_PENALTY="${MEMORY_TERMINAL_STEP_PENALTY:-0.0}"
MEMORY_SCOPE="${MEMORY_SCOPE:-global}"       # global | per_task
RESET_MEMORY="${RESET_MEMORY:-1}"            # 1: clean memory before the run
DISABLE_DYNAMIC_THRESHOLD="${DISABLE_DYNAMIC_THRESHOLD:-0}"
PREFER_TERMINAL_SUCCESS="${PREFER_TERMINAL_SUCCESS:-0}"

if [[ "${PREFER_TERMINAL_SUCCESS}" == "1" ]]; then
  TERMINAL_TAG="terminal_override"
else
  TERMINAL_TAG="no_terminal_override"
fi

RUN_TAG="${DATA_PREFIX}_bon${BON}_${SPLIT}_run_${RUN_ID}_lambda${MEMORY_WEIGHT}_gamma${MEMORY_GAMMA}_k${MEMORY_TOP_K}_thr${MEMORY_THRESHOLD}_${MEMORY_SCOPE}_${TERMINAL_TAG}"
OUT_DIR="${OUT_DIR:-data/train/${TASK}/${SFT_MODEL_NAME}/q_correction_memory_without_perturb/${RUN_TAG}/}"
MEMORY_DIR="${MEMORY_DIR:-${OUT_DIR}/memory}"
MEMORY_LOG_FILE="${MEMORY_LOG_FILE:-${OUT_DIR}/${SLICE_ID}of${SLICE_NUM}_slices_bon_memory_correction_decisions.jsonl}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

mkdir -p "${OUT_DIR}" "${MEMORY_DIR}" "${LOG_DIR}"

SERVER_LOG="${LOG_DIR}/${EXP_NAME}_sglang_server.log"
INFERENCE_LOG="${LOG_DIR}/${EXP_NAME}_q_correction_inference.log"

MEMORY_ARGS=(
  --enable_memory_correction
  --memory_dir "${MEMORY_DIR}"
  --memory_log_file "${MEMORY_LOG_FILE}"
  --memory_weight "${MEMORY_WEIGHT}"
  --memory_gamma "${MEMORY_GAMMA}"
  --memory_top_k "${MEMORY_TOP_K}"
  --memory_threshold "${MEMORY_THRESHOLD}"
  --memory_reward_mode terminal_only
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

SELECTION_ARGS=()
if [[ "${PREFER_TERMINAL_SUCCESS}" == "1" ]]; then
  SELECTION_ARGS+=(--prefer_terminal_success)
fi

echo "[CONFIG] Repository root: ${REPO_ROOT}"
echo "[CONFIG] SFT model:       ${SFT_MODEL_PATH}"
echo "[CONFIG] QNet model:      ${QNET_PATH}"
echo "[CONFIG] GPUs:            server=${SERVER_GPU}, qnet_worker=${WORKER_GPU}"
echo "[CONFIG] Output dir:      ${OUT_DIR}"
echo "[CONFIG] Memory dir:      ${MEMORY_DIR}"
echo "[CONFIG] Memory log:      ${MEMORY_LOG_FILE}"
echo "[CONFIG] Correction:      lambda=${MEMORY_WEIGHT}, gamma=${MEMORY_GAMMA}, top_k=${MEMORY_TOP_K}, threshold=${MEMORY_THRESHOLD}"
echo "[CONFIG] Memory scope:    ${MEMORY_SCOPE}; reset=${RESET_MEMORY}; dynamic_threshold_disabled=${DISABLE_DYNAMIC_THRESHOLD}"
echo "[CONFIG] Terminal success override: ${PREFER_TERMINAL_SUCCESS}"

# ---------------------------------------------------------------------------
# Start the SFT policy server; clean it up when inference ends.
# ---------------------------------------------------------------------------
SERVER_PID=""
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID}" && "${KEEP_SERVER_ALIVE}" != "1" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping SGLang server PID=${SERVER_PID}."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting SGLang policy server on port ${SGLANG_PORT}."
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" python3 -m sglang.launch_server \
  --model-path "${SFT_MODEL_PATH}" \
  --port "${SGLANG_PORT}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

sleep "${SERVER_STARTUP_SECONDS}"
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[ERROR] SGLang server terminated during startup. Tail of ${SERVER_LOG}:" >&2
  tail -n 100 "${SERVER_LOG}" >&2 || true
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

echo "[INFO] Running QLASS + terminal-only memory correction."
CUDA_VISIBLE_DEVICES="${WORKER_GPU}" python qlass/q_guided_inference.py \
  --agent_config "${AGENT_CONFIG}" \
  --agent_path qlass/configs/model/ \
  --qnet_path "${QNET_PATH}" \
  --exp_name "${EXP_NAME}" \
  --exp_path qlass/configs/task/ \
  --exp_config "${TASK}" \
  --split "${SPLT}" \
  --slice_num "${SLICE_NUM}" \
  --slice_id "${SLICE_ID}" \
  --model_name "${SFT_MODEL_NAME}" \
  --num_icl_examples "${ICL}" \
  --sample_mode bon \
  --best_of_N "${BON}" \
  --n_trajs "${N_TRAJS}" \
  --force_first \
  --disable_perturb \
  --verbose \
  --output_dir "${OUT_DIR}" \
  "${MEMORY_ARGS[@]}" \
  "${SELECTION_ARGS[@]}" \
  2>&1 | tee "${INFERENCE_LOG}"

echo "[INFO] Inference completed."
echo "[INFO] Trajectories and tree: ${OUT_DIR}"
echo "[INFO] Memory records:        ${MEMORY_DIR}"
echo "[INFO] Memory diagnostics:    ${MEMORY_LOG_FILE}"
echo "[INFO] Runtime log:           ${INFERENCE_LOG}"

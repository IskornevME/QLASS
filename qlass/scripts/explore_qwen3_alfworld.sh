#!/usr/bin/env bash
set -Eeuo pipefail


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

TASK="${TASK:-alfworld}"
EXP_NAME="${EXP_NAME:-qlass}"

if [[ "${TASK}" != "alfworld" ]]; then
    echo "[ERROR] This launcher supports only ALFWorld." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Qwen policy profile
# -----------------------------------------------------------------------------

MODEL_PROFILE="${MODEL_PROFILE:-qwen3_4b_instruct_2507}"
MODEL_PROFILE_PATH="/home/m.iskornev/qlass/QLASS/qlass/configs/model_profiles/${MODEL_PROFILE}.sh"

if [[ ! -f "${MODEL_PROFILE_PATH}" ]]; then
    echo "[ERROR] Model profile not found: ${MODEL_PROFILE_PATH}" >&2
    exit 1
fi

source "${MODEL_PROFILE_PATH}"

# Exact actor configuration used in the successful ReAct baseline.
POLICY_MAX_NEW_TOKENS=2048
POLICY_TEMPERATURE=0.7


# -----------------------------------------------------------------------------
# Python environments
# -----------------------------------------------------------------------------

SGLANG_PYTHON="${SGLANG_PYTHON:-/home/m.iskornev/miniforge3/envs/qlass_qwen_server/bin/python}"
QLASS_PYTHON="${QLASS_PYTHON:-/home/m.iskornev/miniforge3/envs/my_env/bin/python}"


# -----------------------------------------------------------------------------
# Exploration settings
# -----------------------------------------------------------------------------

NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SERVERS="${NUM_SERVERS:-2}"

MAX_STEPS="${MAX_STEPS:-30}"
MAX_DEPTH="${MAX_DEPTH:-5}"
MIN_PRUNE_DEPTH="${MIN_PRUNE_DEPTH:-3}"
SAMPLES_PER_DEPTH="${SAMPLES_PER_DEPTH:-2}"
HISTORY_LENGTH="${HISTORY_LENGTH:-50}"

MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-}"

SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_TASKS="${SMOKE_TASKS:-2}"

KEEP_SERVERS_ALIVE="${KEEP_SERVERS_ALIVE:-0}"


# -----------------------------------------------------------------------------
# Server configuration
# -----------------------------------------------------------------------------

SERVER_GPU_0="${SERVER_GPU_0:-0}"
SERVER_GPU_1="${SERVER_GPU_1:-2}"

SGLANG_PORT_0="${SGLANG_PORT_0:-21003}"
SGLANG_PORT_1="${SGLANG_PORT_1:-21005}"

SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-120}"
SERVER_READY_SLEEP_SECONDS="${SERVER_READY_SLEEP_SECONDS:-2}"

if (( NUM_SERVERS < 1 || NUM_SERVERS > 2 )); then
    echo "[ERROR] NUM_SERVERS must be 1 or 2." >&2
    exit 1
fi

if (( NUM_WORKERS < 1 )); then
    echo "[ERROR] NUM_WORKERS must be positive." >&2
    exit 1
fi

SERVER_GPUS=(
    "${SERVER_GPU_0}"
    "${SERVER_GPU_1}"
)

SERVER_PORTS=(
    "${SGLANG_PORT_0}"
    "${SGLANG_PORT_1}"
)

if [[ "${NUM_SERVERS}" == "2" ]] \
    && [[ "${SERVER_GPU_0}" == "${SERVER_GPU_1}" ]]; then
    echo "[ERROR] Two SGLang replicas must use different GPUs." >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Smoke-test mode
# -----------------------------------------------------------------------------

MAX_TASK_ARGS=()

if [[ "${SMOKE_TEST}" == "1" ]]; then
    echo "[INFO] Smoke-test mode enabled."

    NUM_SERVERS=1
    NUM_WORKERS=1

    if (( SMOKE_TASKS < 1 || SMOKE_TASKS > 2 )); then
        echo "[ERROR] SMOKE_TASKS must be 1 or 2." >&2
        exit 1
    fi

    MAX_TASK_ARGS=(
        --max_tasks "${SMOKE_TASKS}"
    )

    DEFAULT_OUTPUT_DIR="data/train/${TASK}/explore_qwen3_react_smoke/"
else
    if [[ -n "${MAX_TASKS_PER_WORKER}" ]]; then
        if ! [[ "${MAX_TASKS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]]; then
            echo \
                "[ERROR] MAX_TASKS_PER_WORKER must be a positive integer." \
                >&2
            exit 1
        fi

        MAX_TASK_ARGS=(
            --max_tasks "${MAX_TASKS_PER_WORKER}"
        )
    fi

    DEFAULT_OUTPUT_DIR="data/train/${TASK}/explore_qwen3_react_d${MAX_DEPTH}_s${SAMPLES_PER_DEPTH}_mpr${MIN_PRUNE_DEPTH}/"
fi


OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

case "${OUTPUT_DIR}" in
    */) ;;
    *) OUTPUT_DIR="${OUTPUT_DIR}/" ;;
esac

LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}logs}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"


# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

SERVER_PIDS=()
WORKER_PIDS=()

cleanup() {
    exit_code=$?

    trap - EXIT INT TERM

    # Stop remaining workers on failure/interruption.
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[INFO] Stopping worker PID=${pid}."
            kill "${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${WORKER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    if [[ "${KEEP_SERVERS_ALIVE}" != "1" ]]; then
        for pid in "${SERVER_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                echo "[INFO] Stopping SGLang server PID=${pid}."
                kill "${pid}" 2>/dev/null || true
            fi
        done

        for pid in "${SERVER_PIDS[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
    fi

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM


# -----------------------------------------------------------------------------
# Start SGLang replicas
# -----------------------------------------------------------------------------

start_server() {
    local server_idx="$1"
    local gpu="${SERVER_GPUS[server_idx]}"
    local port="${SERVER_PORTS[server_idx]}"
    local server_log="${LOG_DIR}/sglang_server_${server_idx}.log"

    local address="http://127.0.0.1:${port}"

    if curl -fsS --max-time 2 \
        "${address}/v1/models" >/dev/null 2>&1; then
        echo "[ERROR] A server already responds at ${address}." >&2
        exit 1
    fi

    echo "[INFO] Starting SGLang server ${server_idx}: GPU=${gpu}, port=${port}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
        "${SGLANG_PYTHON}" \
        -m sglang.launch_server \
        --model-path "${POLICY_MODEL_PATH}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --tp-size "${SERVER_TP}" \
        --context-length "${SGLANG_CONTEXT_LENGTH}" \
        > "${server_log}" 2>&1 &

    local pid=$!
    SERVER_PIDS+=("${pid}")

    local ready=0

    for attempt in $(seq 1 "${SERVER_READY_ATTEMPTS}"); do
        if curl -fsS \
            "${address}/v1/models" >/dev/null 2>&1; then
            ready=1
            echo "[INFO] Server ${server_idx} is ready."
            break
        fi

        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[ERROR] Server ${server_idx} crashed during startup." >&2
            tail -n 100 "${server_log}" >&2 || true
            exit 1
        fi

        sleep "${SERVER_READY_SLEEP_SECONDS}"
    done

    if [[ "${ready}" != "1" ]]; then
        echo "[ERROR] Server ${server_idx} did not become ready." >&2
        tail -n 100 "${server_log}" >&2 || true
        exit 1
    fi
}


for ((server_idx=0; server_idx<NUM_SERVERS; server_idx++)); do
    start_server "${server_idx}"
done


# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------

echo "[CONFIG] Model:              ${POLICY_MODEL_PATH}"
echo "[CONFIG] Servers:            ${NUM_SERVERS}"
echo "[CONFIG] Workers:            ${NUM_WORKERS}"
echo "[CONFIG] Max steps:          ${MAX_STEPS}"
echo "[CONFIG] Max depth:          ${MAX_DEPTH}"
echo "[CONFIG] Samples per depth:  ${SAMPLES_PER_DEPTH}"
echo "[CONFIG] History length:     ${HISTORY_LENGTH}"
echo "[CONFIG] Output dir:         ${OUTPUT_DIR}"
echo "[CONFIG] Smoke test:         ${SMOKE_TEST}"


# -----------------------------------------------------------------------------
# Start exploration workers
# -----------------------------------------------------------------------------

for ((i=0; i<NUM_WORKERS; i++)); do
    server_idx=$((i % NUM_SERVERS))
    current_port="${SERVER_PORTS[server_idx]}"

    worker_log="${LOG_DIR}/worker_${i}.log"

    echo \
        "[INFO] Starting worker ${i}: " \
        "server=${server_idx}, port=${current_port}, log=${worker_log}"

    CUDA_VISIBLE_DEVICES="" \
        "${QLASS_PYTHON}" \
        qlass/explore_sft_agent.py \
        --agent_config "${AGENT_CONFIG}" \
        --agent_path qlass/configs/model/ \
        --policy_server_address "http://127.0.0.1:${current_port}" \
        --model_name "${POLICY_MODEL_NAME}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --policy_max_prompt_tokens "${POLICY_MAX_PROMPT_TOKENS}" \
        --policy_max_new_tokens "${POLICY_MAX_NEW_TOKENS}" \
        --policy_temperature "${POLICY_TEMPERATURE}" \
        --policy_top_p "${POLICY_TOP_P}" \
        --policy_top_k "${POLICY_TOP_K}" \
        --policy_min_p "${POLICY_MIN_P}" \
        --policy_presence_penalty "${POLICY_PRESENCE_PENALTY}" \
        --alfworld_react_prompt \
        --alfworld_history_length "${HISTORY_LENGTH}" \
        --max_steps "${MAX_STEPS}" \
        --exp_name "${EXP_NAME}" \
        --exp_path qlass/configs/task/ \
        --exp_config "${TASK}" \
        --split train \
        --slice_num "${NUM_WORKERS}" \
        --slice_id "${i}" \
        --max_depth "${MAX_DEPTH}" \
        --min_prune_depth "${MIN_PRUNE_DEPTH}" \
        --samples_per_depth "${SAMPLES_PER_DEPTH}" \
        --num_icl_examples 0 \
        --output_dir "${OUTPUT_DIR}" \
        "${MAX_TASK_ARGS[@]}" \
        > "${worker_log}" 2>&1 &

    WORKER_PIDS+=("$!")
done


# -----------------------------------------------------------------------------
# Wait for workers.
# Any non-zero worker exit makes the launcher fail; EXIT trap then kills servers.
# -----------------------------------------------------------------------------

for idx in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[idx]}"

    if ! wait "${pid}"; then
        echo "[ERROR] Exploration worker ${idx} failed." >&2
        echo "[ERROR] See ${LOG_DIR}/worker_${idx}.log" >&2
        exit 1
    fi

    echo "[INFO] Worker ${idx} completed successfully."
done


echo "[INFO] All exploration workers completed successfully."
echo "[INFO] Data: ${OUTPUT_DIR}"

#!/usr/bin/env bash
set -Eeuo pipefail


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

TASK="${TASK:-alfworld}"
EXP_NAME="${EXP_NAME:-qlass}"

if [[ "${TASK}" != "alfworld" ]]; then
    echo "[ERROR] This launcher currently supports only ALFWorld." >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Qwen model
# -----------------------------------------------------------------------------

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[ERROR] MODEL_PATH must point to the directory containing model checkpoints." >&2
    exit 1
fi

MODEL_PROFILE="${MODEL_PROFILE:-qwen3_4b_instruct_2507}"

MODEL_PROFILE_PATH="/home/m.iskornev/qlass/QLASS/qlass/configs/model_profiles/${MODEL_PROFILE}.sh"

if [[ ! -f "${MODEL_PROFILE_PATH}" ]]; then
    echo "[ERROR] Model profile not found: ${MODEL_PROFILE_PATH}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${MODEL_PROFILE_PATH}"

# именно эта модель, а не какая-то sft версия
BASE_MODEL_PATH="${POLICY_MODEL_PATH}"

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "[ERROR] Qwen checkpoint not found: ${BASE_MODEL_PATH}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Data / output
# -----------------------------------------------------------------------------

Q_DATA_PATH="${Q_DATA_PATH:-data/train/${TASK}/qwen3_react_qnet/vanilla.jsonl}"

Q_MODEL_NAME="${Q_MODEL_NAME:-${EXP_NAME}-${POLICY_MODEL_NAME}-${TASK}-Q}"

OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH%/}/${Q_MODEL_NAME}}"

if [[ ! -f "${Q_DATA_PATH}" ]]; then
    echo "[ERROR] Q dataset not found: ${Q_DATA_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"


# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------

TRAIN_PYTHON="${TRAIN_PYTHON:-/home/m.iskornev/miniforge3/envs/qlass_qwen_qnet/bin/python}"


"${TRAIN_PYTHON}" - <<'PY'
from packaging.version import Version
import torch
import accelerate
import datasets
import transformers

minimum = Version("4.51.0")
current = Version(transformers.__version__)

print(f"[CHECK] transformers={current}")

if current < minimum:
    raise RuntimeError(
        f"Qwen3 requires transformers >= {minimum}; "
        f"found {current}"
    )

from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

print("[CHECK] Qwen3DecoderLayer import OK")
PY


# -----------------------------------------------------------------------------
# GPU / distributed setup
# -----------------------------------------------------------------------------

GPU_LIST="${GPU_LIST:-0,2}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_ARRAY[@]}"

if (( NUM_GPUS < 1 )); then
    echo "[ERROR] No GPUs configured." >&2
    exit 1
fi

MASTER_PORT="${MASTER_PORT:-20001}"


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

DENOM=$((NUM_GPUS * MICRO_BATCH_SIZE))

if (( GLOBAL_BATCH_SIZE % DENOM != 0 )); then
    echo "[ERROR] GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * MICRO_BATCH_SIZE." >&2
    exit 1
fi

GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / DENOM))

QNET_MODEL_MAX_LENGTH="${QNET_MODEL_MAX_LENGTH:-8192}"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"


echo "[CONFIG] Base model:       ${BASE_MODEL_PATH}"
echo "[CONFIG] Q data:           ${Q_DATA_PATH}"
echo "[CONFIG] Output:           ${OUTPUT_DIR}"
echo "[CONFIG] GPUs:             ${GPU_LIST}"
echo "[CONFIG] Num GPUs:         ${NUM_GPUS}"
echo "[CONFIG] Global batch:     ${GLOBAL_BATCH_SIZE}"
echo "[CONFIG] Micro batch:      ${MICRO_BATCH_SIZE}"
echo "[CONFIG] Grad accumulation:${GRAD_ACCUM}"
echo "[CONFIG] Max length:       ${QNET_MODEL_MAX_LENGTH}"
echo "[CONFIG] Epochs:           ${NUM_EPOCHS}"
echo "[CONFIG] Learning rate:    ${LEARNING_RATE}"
echo "[CONFIG] Seed:             ${SEED}"


CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${TRAIN_PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    qlass/train_q.py \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --data_path "${Q_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --qnet_mode final \
    --padding_side right \
    --truncation_side left \
    --bf16 True \
    --tf32 True \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${MICRO_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config qlass/configs/fsdp_qwen3.json \
    --model_max_length "${QNET_MODEL_MAX_LENGTH}" \
    --lazy_preprocess False \
    --remove_unused_columns False \
    --seed "${SEED}" \
    --report_to none


echo "[INFO] QNet training completed."
echo "[INFO] Model saved to: ${OUTPUT_DIR}"
#!/usr/bin/env bash
set -Eeuo pipefail


TASK="${TASK:-alfworld}"

DATA_PATH="${DATA_PATH:-data/train/${TASK}/explore_qwen3_react_d5_s2_mpr3}"

Q_TYPE="${Q_TYPE:-vanilla}"
UPPER_NUM="${UPPER_NUM:-300}"
SEED="${SEED:-42}"

OUTPUT_PATH="${OUTPUT_PATH:-data/train/${TASK}/qwen3_react_qnet/${Q_TYPE}.jsonl}"


if [[ ! -d "${DATA_PATH}" ]]; then
    echo "[ERROR] Exploration directory not found: ${DATA_PATH}" >&2
    exit 1
fi


mkdir -p "$(dirname "${OUTPUT_PATH}")"


echo "[CONFIG] task:        ${TASK}"
echo "[CONFIG] data path:   ${DATA_PATH}"
echo "[CONFIG] output path: ${OUTPUT_PATH}"
echo "[CONFIG] q type:      ${Q_TYPE}"
echo "[CONFIG] upper num:   ${UPPER_NUM}"
echo "[CONFIG] seed:        ${SEED}"


python qlass/construct_q_data.py \
    --task "${TASK}" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --q_type "${Q_TYPE}" \
    --upper_num "${UPPER_NUM}" \
    --seed "${SEED}" \
    --require_critic_state


echo "[INFO] Q dataset saved to: ${OUTPUT_PATH}"
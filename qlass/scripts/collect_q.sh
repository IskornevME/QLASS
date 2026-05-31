#!/usr/bin/env bash
set -euo pipefail

task=sciworld  # webshop, scienceworld, alfworld

# ВАЖНО: реальный output_dir от exploration
data_path="data/train/${task}/explore_7b_sft_d6_i0_s2_mpr3_v2"
exp_name=qlass

# Куда construct_q_data.py сохранит итог: data/train/<task>/explore/<q_type>.jsonl
mkdir -p "data/train/${task}/explore_v2"

# Чтобы при перезапуске не было дублей (скрипт подхватывает ВСЕ .jsonl/.pkl в data_path)
rm -f "${data_path}/combined_traj.jsonl" \
      "${data_path}/combined_tree.pkl" \
      "${data_path}/q_data.jsonl" \
      "${data_path}/r_data.jsonl"

python qlass/construct_q_data.py \
  --task "${task}" \
  --data_path "${data_path}" \
  --q_type vanilla

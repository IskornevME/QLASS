#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH=/home/m.iskornev/qlass/models/
cd /home/m.iskornev/qlass/QLASS

mkdir -p logs
exec bash qlass/scripts/train_qnet_7b.sh

#!/usr/bin/env bash
#
# Timed benchmark run: N epochs of joint training, no interventions.
#
# Usage:
#   ./run.sh                        # 5 epochs, default config
#   ./run.sh 10                     # 10 epochs
#   ./run.sh 5 gpu_preload=True     # 5 epochs + extra hydra overrides
#
set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniconda3/envs/scbm/bin/python}"
EPOCHS="${1:-5}"
shift || true

cd "$(dirname "$0")"

echo "=== SCBM benchmark: ${EPOCHS} epochs, train_only ==="
echo "python:   ${PYTHON}"
echo "overrides: $*"
echo

START=$(date +%s.%N)

"${PYTHON}" train.py \
    +model=SCBM \
    +data=CUB \
    data.data_path=datasets/ \
    train_only=True \
    model.j_epochs="${EPOCHS}" \
    "$@"

END=$(date +%s.%N)

ELAPSED=$(echo "${END} - ${START}" | bc)
PER_EPOCH=$(echo "scale=3; ${ELAPSED} / ${EPOCHS}" | bc)

echo
echo "======================================"
printf "Wall time:   %.2f s\n" "${ELAPSED}"
printf "Per epoch:   %.2f s  (%d epochs)\n" "${PER_EPOCH}" "${EPOCHS}"
echo "======================================"

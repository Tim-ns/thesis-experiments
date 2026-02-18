#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPS_DIR="${EXP_ROOT}/dependencies/ttt-lm-jax"

cd "${EXP_ROOT}"

export PYTHONPATH="${EXP_ROOT}:${DEPS_DIR}/ttt-lm-jax:${PYTHONPATH}"
DATASET_NAME="test_bookcorpus"

mkdir -p ttt-lm/logs/training
LOG_FILE="ttt-lm/logs/training/tokenize_$(date +%Y%m%d_%H%M%S).log"

pytest -q -s ttt/dataloader/tokenization.py -k "${DATASET_NAME}" 2>&1 | tee "${LOG_FILE}"

echo "Tokenization of ${SCRIPT_DIR} completed successfully."
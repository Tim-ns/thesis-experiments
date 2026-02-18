#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPS_DIR="${EXP_ROOT}/dependencies"

cd "${EXP_ROOT}"

export PYTHONPATH="${EXP_ROOT}:${DEPS_DIR}/ttt-lm-jax:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

TRANSFORMER_PATH="${EXP_ROOT}/ttt-experiments/bookcorpus_125m_15000_steps"
TTT_PATH="${EXP_ROOT}/ttt-experiments/bookcorpus_125m_ttt_50000_steps"
TTT_LR_MULT=1.5

mkdir -p ttt-lm/logs/inference
LOG_FILE="ttt-lm/logs/inference/inference_$(date +%Y%m%d_%H%M%S).log"

echo "Starting inference with:"
echo "Transformer Path: ${TRANSFORMER_PATH}"
echo "TTT Path: ${TTT_PATH}"

python3 -m run_inference \
        --transformer_pretrained_model_path="${TRANSFORMER_PATH}" \
        --ttt_pretrained_model_path="${TTT_PATH}" \
        --ttt_lr_mult=${TTT_LR_MULT} 2>&1 | tee "${LOG_FILE}"

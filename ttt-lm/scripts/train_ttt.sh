#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPS_DIR="${EXP_ROOT}/dependencies"

cd "${EXP_ROOT}"

DATA_PATH="${EXP_ROOT}/data/bookcorpus_release"
DATA_NAME="bookcorpus" 
VAL_RATIO=0.05

export PYTHONPATH="${EXP_ROOT}:${DEPS_DIR}/ttt-lm-jax:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_all_gather_combine_threshold_bytes=104857600 --xla_gpu_all_reduce_combine_threshold_bytes=104857600"

# Training Parameters
MODEL_SIZE=125m
SEQ_LEN=1024
BS=32
ACCUM=1
LR_START=1.5e-3
LR_END=1e-5
LR_WARMUP_STEPS=500
TOTAL_STEPS=50000

# Experiment details
EXP_NAME="bookcorpus_${MODEL_SIZE}_ttt_${TOTAL_STEPS}_steps"
EXP_DIR="${EXP_ROOT}/ttt-experiments"

mkdir -p "${EXP_DIR}/${EXP_NAME}"

mkdir -p ttt-lm/logs/training/ttt
LOG_FILE="ttt-lm/logs/training/ttt/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting TTT training with:"
echo "Data Path: ${DATA_PATH}"
echo "Exp Dir: ${EXP_DIR}"

python3 -m ${DEPS_DIR}.ttt-lm-jax.ttt.train \
        --mesh_dim='1,1,-1' \
        --dtype='bf16' \
        --update_model_config="dict(remat_block='nothing_saveable', scan_mlp=False, seq_modeling_block='ttt_linear', ttt_base_lr=1.0)" \
        --total_steps=${TOTAL_STEPS} \
        --save_checkpoint_freq=2500 \
        --save_milestone_freq=5000 \
        --load_model_config='125m-TTT' \
        --dataset_path="${DATA_PATH}" \
        --dataset_name="${DATA_NAME}" \
        --val_ratio=${VAL_RATIO} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --loader_workers=16 \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=${LR_START} \
        --optimizer.adamw_optimizer.end_lr=${LR_END} \
        --optimizer.adamw_optimizer.lr_warmup_steps=${LR_WARMUP_STEPS} \
        --optimizer.adamw_optimizer.lr_decay_steps=${TOTAL_STEPS} \
        --exp_dir="${EXP_DIR}" \
        --exp_name="${EXP_NAME}" 2>&1 | tee "${LOG_FILE}"
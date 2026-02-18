#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPS_DIR="${EXP_ROOT}/dependencies"
DATA_ROOT="${EXP_ROOT}/data/google_dreambooth"

# Training Parameters
NUM_STEPS=1000
NUM_CLASS_IMAGES=500
GRADIENT_ACCUMULATION_STEPS=2
LR=2e-5
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=0
PRIOR_LOSS_WEIGHT=0.4

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="${DATA_ROOT}/dataset/cat2"
export CLASS_DIR="${DATA_ROOT}/classes/class_cat"
export OUTPUT_DIR="${EXP_ROOT}/dreambooth/results/training/cat_lora_encoder"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_HOME="/usr/local/cuda-12.8"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CLASS_DIR}"

cd "${DEPS_DIR}/diffusers/examples/dreambooth"

mkdir -p logs/training/cat
LOG_FILE="logs/training/cat/train_$(date +%Y%m%d_%H%M%S).log"

python ${DEPS_DIR}/diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=${PRIOR_LOSS_WEIGHT} \
  --instance_prompt="a photo of sks cat" \
  --class_prompt="a photo of cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate=${LR} \
  --lr_scheduler="${LR_SCHEDULER}" \
  --lr_warmup_steps=${LR_WARMUP_STEPS} \
  --num_class_images=${NUM_CLASS_IMAGES} \
  --max_train_steps=${NUM_STEPS} \
  --seed="0" \
  --report_to="wandb" 2>&1 | tee "${LOG_FILE}"
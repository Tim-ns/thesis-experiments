#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="${ROOT_DIR}/dependencies"

TTT_REPO_URL="https://github.com/test-time-training/ttt-lm-jax.git"
DIFFUSERS_REPO_URL="https://github.com/huggingface/diffusers.git"

TTT_COMMIT="6f529b124c7fb5879b33c06926408b15add1d82f"
DIFFUSERS_COMMIT="bcbbded7c3fc873343a7c8f8a63d91d5c727a4a3"

echo "--------------------------------------------------------"
echo "Setting up the Experiments Environment"
echo "Root Directory: ${ROOT_DIR}"
echo "--------------------------------------------------------"

mkdir -p "${DEPS_DIR}"
cd "${DEPS_DIR}"

if [ ! -d "ttt-lm-jax" ]; then
    echo "Cloning TTT-LM-JAX..."
    git clone "${TTT_REPO_URL}" ttt-lm-jax
    cd ttt-lm-jax && git checkout ${TTT_COMMIT} && cd ..
else
    echo "TTT-LM-JAX already exists, skipping clone."
fi

if [ ! -d "diffusers" ]; then
    echo "Cloning Diffusers..."
    git clone "${DIFFUSERS_REPO_URL}" diffusers
    cd diffusers && git checkout ${DIFFUSERS_COMMIT} && cd ..
else
    echo "Diffusers already exists, skipping clone."
fi

echo "Installing Python dependencies..."

TTT_REQ="${DEPS_DIR}/ttt-lm-jax/requirements/gpu_requirements.txt"

if [ -f "${ROOT_DIR}/requirements.txt" ]; then
    echo "Installing from main requirements.txt..."
    pip install -r "${ROOT_DIR}/requirements.txt"
fi

if [ -f "${TTT_REQ}" ]; then
    echo "Installing TTT-LM-JAX requirements..."
    pip install -r "${TTT_REQ}"
else
    echo "Warning: TTT-LM-JAX requirements not found at ${TTT_REQ}"
fi

echo "Installing Diffusers in editable mode..."
pip install -e "${DEPS_DIR}/diffusers"

cd "${DEPS_DIR}/diffusers/examples/dreambooth"
pip install -r requirements.txt

export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

pip install pytest python-dotenv deepspeed
pip install -U transformers peft accelerate

echo "--------------------------------------------------------"
echo "Setup Complete"
echo "--------------------------------------------------------"

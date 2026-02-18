#!/bin/bash

# Copies dataloader files and ensures TTT-LM-JAX is compatible.

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${EXP_ROOT}/ttt-lm/dataloader"
DEST_DIR="${EXP_ROOT}/dependencies/ttt-lm-jax/ttt/dataloader"

FILES=("bookcorpus.py" "tokenization.py" "language_modeling_hf.py" "lm_dataset.py")

for FILE in "${FILES[@]}"; do
    if [ -f "${SRC_DIR}/${FILE}" ]; then
        cp "${SRC_DIR}/${FILE}" "${DEST_DIR}/"
        echo "Copied ${FILE}"
    else
        echo "Warning: ${SRC_DIR}/${FILE} not found"
    fi
done

echo "--------------------------------------------------------"
echo "Preparation Complete!"
echo "--------------------------------------------------------"

#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export DATA_DIR="${EXP_ROOT}/dreambooth/google_dreambooth"

echo "--------------------------------------------------------"
echo "Downloading DreamBooth Dataset"
echo "Destination: ${DATA_DIR}"
echo "--------------------------------------------------------"

mkdir -p "${DATA_DIR}"

python3 "${SCRIPT_DIR}/download_data.py"

echo "--------------------------------------------------------"
echo "Download Complete!"
echo "--------------------------------------------------------"

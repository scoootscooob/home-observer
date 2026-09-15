#!/usr/bin/env bash
set -euo pipefail
cd /workspace/home-observer
mkdir -p artifacts/run
export HF_HOME=/workspace/huggingface
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
python3 -m venv /workspace/home-observer-venv
source /workspace/home-observer-venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install -e '.[data]' -r requirements-gpu.txt
python -m pip freeze > artifacts/run/environment.txt
nvidia-smi -q > artifacts/run/nvidia-smi.txt
DATASET_ROOT=${DATASET_ROOT:-data}
ADAPTER_PATH=${ADAPTER_PATH:-data/adapter}
# No public HTTP port. Access from the local SSH tunnel printed in state.json.
python -m home_observer.serve --dataset-root "$DATASET_ROOT" --adapter "$ADAPTER_PATH" --host 127.0.0.1 --port 8000

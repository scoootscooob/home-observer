#!/usr/bin/env bash
# Reproducible bounded training job; cloud.py owns retrieval and Pod cleanup.
set -euo pipefail
cd /workspace/home-observer
mkdir -p artifacts/training
export HF_HOME=/workspace/huggingface
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
python3 -m venv /workspace/home-observer-venv
source /workspace/home-observer-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install -e '.[data]' -r requirements-gpu.txt
python -m pip freeze > artifacts/training/environment.txt
nvidia-smi -q > artifacts/training/nvidia-smi.txt
python -m home_observer.train \
  --train data/mixed-v2/train.jsonl --validation data/mixed-v2/validation-dev.jsonl \
  --dataset-root data --policy data/mixed-v2/policy.json \
  --output artifacts/training/adapter --quantization none \
  --max-steps "${TRAIN_STEPS:-400}" --gradient-accumulation-steps 2 --rank 16 \
  --learning-rate 0.0001 --decision-field-order actions-first \
  --generation-validation data/mixed-v2/generation-validation.jsonl \
  --generation-eval-every 100 --checkpoint-every 100
# Report the final candidate; this does not promote it or overwrite release-status.
python scripts/evaluate_release.py --adapter artifacts/training/adapter \
  --policy data/mixed-v2/policy.json --output artifacts/release-trained \
  --quantization none --batch-size 4 --max-new-tokens 768
python scripts/model_annotation_metrics.py --release-root artifacts/release-trained

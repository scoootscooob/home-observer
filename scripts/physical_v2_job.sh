#!/usr/bin/env bash
# Bounded research job (Runpod H100): train the v2 physical adapter on the fresh continuous
# training windows, then run the matched native evaluation of base / v1 / v2 on the frozen
# continuous test windows. Runs from the staged project at /workspace/home-observer.
# cloud.py owns artifact retrieval and Pod cleanup; nothing here touches provider credentials.
set -uo pipefail
cd /workspace/home-observer
mkdir -p artifacts/physical-v2
export HF_HOME=/workspace/huggingface
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset RUNPOD_API_KEY
phase() { printf '%s\n' "$1" > artifacts/physical-v2/job.phase; date -u +%Y-%m-%dT%H:%M:%SZ >> artifacts/physical-v2/job.timeline; printf '%s\n' "$1" >> artifacts/physical-v2/job.timeline; }
trap 'code=$?; printf "%s\n" "$code" > artifacts/physical-v2/job.exit' EXIT

setup_environment() {
  set -e
  python3 -m venv /workspace/home-observer-venv
  source /workspace/home-observer-venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  python -m pip install -e '.[data]' -r requirements-gpu.txt
  python -m pip freeze > artifacts/physical-v2/environment.txt
  nvidia-smi -q > artifacts/physical-v2/nvidia-smi.txt
  python - <<'PY'
from huggingface_hub import snapshot_download
from home_observer.model import DEFAULT_MODEL, DEFAULT_REVISION
snapshot_download(DEFAULT_MODEL, revision=DEFAULT_REVISION, ignore_patterns=['*.gguf','*.md'])
PY
  export HF_HUB_OFFLINE=1
  sha256sum data/continuous-v2/train-supervised.jsonl data/continuous-v2/validation-supervised.jsonl data/continuous-v2/test.jsonl data/continuous-v2/supervised-selection.json \
    data/adapter-v1/adapter_model.safetensors data/adapter-v1/adapter_config.json > artifacts/physical-v2/input-hashes.txt
}
phase setup
(set -e; setup_environment) > artifacts/physical-v2/setup.log 2>&1
setup_exit=$?
printf '%s\n' "$setup_exit" > artifacts/physical-v2/setup-exit
if [[ "$setup_exit" != 0 ]]; then exit "$setup_exit"; fi
source /workspace/home-observer-venv/bin/activate
export HF_HUB_OFFLINE=1
STEPS="${V2_TRAIN_STEPS:-300}"
FRAMES="${V2_MAX_FRAMES:-8}"

phase train-v2
python -m home_observer.physical_train \
  --train data/continuous-v2/train-supervised.jsonl --validation data/continuous-v2/validation-supervised.jsonl \
  --dataset-root data/continuous-v2 --output artifacts/physical-v2/adapter \
  --quantization none --max-steps "$STEPS" --rank 16 --learning-rate 1e-4 \
  --gradient-accumulation-steps 2 --checkpoint-every 100 --generation-limit 4 \
  --max-frames "$FRAMES" --max-input-tokens 8192 --max-new-tokens 768 \
  > artifacts/physical-v2/train.log 2>&1
train_exit=$?
printf '%s\n' "$train_exit" > artifacts/physical-v2/train-exit
if [[ "$train_exit" != 0 ]]; then exit "$train_exit"; fi

# Matched native evaluation: identical rows, frames, prompt, greedy decoding, token budgets and GPU.
evaluate() {
  local name="$1"; shift
  phase "eval-$name"
  python scripts/predict_physical.py --dataset data/continuous-v2/test.jsonl --dataset-root data/continuous-v2 \
    --output "artifacts/physical-v2/eval-$name" --max-frames "$FRAMES" --max-input-tokens 8192 \
    --max-new-tokens 1024 "$@" > "artifacts/physical-v2/eval-$name.log" 2>&1
  local code=$?
  printf '%s\n' "$code" > "artifacts/physical-v2/eval-$name-exit"
  return "$code"
}
evaluate v2 --adapter artifacts/physical-v2/adapter
evaluate v1 --adapter data/adapter-v1
evaluate base
# Validation-split predictions for threshold/format selection stay separate from the test rows.
phase eval-v2-validation
python scripts/predict_physical.py --dataset data/continuous-v2/validation.jsonl --dataset-root data/continuous-v2 \
  --output artifacts/physical-v2/eval-v2-validation --adapter artifacts/physical-v2/adapter \
  --max-frames "$FRAMES" --max-input-tokens 8192 --max-new-tokens 1024 > artifacts/physical-v2/eval-v2-validation.log 2>&1 || true
phase complete
date -u +%Y-%m-%dT%H:%M:%SZ > artifacts/physical-v2/job.finished
exit 0

#!/usr/bin/env bash
set -uo pipefail
setup_environment() {
  set -e
  cd /workspace/home-observer
  mkdir -p artifacts/inference
  export HF_HOME=/workspace/huggingface
  export PIP_CACHE_DIR=/workspace/pip-cache
  python3 -m venv /workspace/home-observer-venv
  source /workspace/home-observer-venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  python -m pip install -e '.[data]' -r requirements-gpu.txt
  python -m pip freeze > artifacts/inference/environment.txt
  nvidia-smi -q > artifacts/inference/nvidia-smi.txt
  python - <<'PY'
from huggingface_hub import snapshot_download
from home_observer.model import DEFAULT_MODEL, DEFAULT_REVISION
snapshot_download(DEFAULT_MODEL, revision=DEFAULT_REVISION, ignore_patterns=['*.gguf','*.md'])
PY
}
(set -e; setup_environment)
setup_exit=$?
printf '%s\n' "$setup_exit" > /workspace/home-observer/artifacts/setup-exit
touch /workspace/home-observer-inference-ready
while [ ! -f /workspace/home-observer-inference-finish ]; do sleep 2; done
if [ -f /workspace/home-observer/artifacts/final-exit ]; then
  exit "$(cat /workspace/home-observer/artifacts/final-exit)"
fi
exit "$setup_exit"

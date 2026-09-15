#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /workspace/home-observer
export UV_CACHE_DIR=/workspace/home-observer-native-cache
export HF_HOME=/workspace/huggingface
export PIP_DISABLE_PIP_VERSION_CHECK=1
native_env=/workspace/home-observer-venv
mkdir -p artifacts/native-setup
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'The selected image must provide uv for the isolated native setup.' >&2
  exit 1
fi
if [[ ! -x "$native_env/bin/python" ]]; then
  uv venv --python "$(command -v python3)" "$native_env"
fi
timeout --signal=TERM --kill-after=15s 300s uv pip install --python "$native_env/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
timeout --signal=TERM --kill-after=15s 300s uv pip install --python "$native_env/bin/python" \
  -r requirements-gpu.txt -e '.[data]'
uv pip freeze --python "$native_env/bin/python" > artifacts/native-setup/requirements-resolved.txt
"$native_env/bin/python" - <<'PY'
import importlib.metadata as m,json
from pathlib import Path
packages={p:m.version(p) for p in ['torch','torchvision','torchaudio','transformers','peft','accelerate']}
Path('artifacts/native-setup/versions.json').write_text(json.dumps(packages,indent=2)+'\n')
print(json.dumps(packages))
PY

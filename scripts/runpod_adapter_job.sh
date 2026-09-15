#!/usr/bin/env bash
# The Runpod supervisor enforces the outer deadline and collects artifacts.
set -euo pipefail
umask 077
cd /workspace/home-observer
backend=${1:-vllm}
case "$backend" in
  vllm)
    HOME_OBSERVER_SETUP_ONLY=1 bash scripts/vllm_session.sh
    runtime=/workspace/home-observer-vllm-venv/bin/python
    ;;
  native)
    bash scripts/runpod_native_setup.sh
    runtime=/workspace/home-observer-venv/bin/python
    ;;
  *) printf '%s\n' 'Expected backend vllm or native' >&2; exit 2 ;;
esac
# Upload only the selected adapter after SSH becomes available. No model runs
# until its six-file checksum manifest has been verified by the transfer tool.
for ((attempt=0;attempt<600;attempt++)); do
  if [[ -s artifacts/serving-adapter/transfer-manifest.json ]]; then
    exec "$runtime" scripts/runpod_adapter_session.py --backend "$backend" \
      --adapter artifacts/serving-adapter --output artifacts/adapter-serving
  fi
  sleep 2
done
printf '%s\n' 'Adapter transfer deadline exceeded (20 minutes)' >&2
exit 1

#!/usr/bin/env bash
# Explicit warm start from v2 checkpoint300; no optimizer-resume claim.
set -euo pipefail
cd /workspace/home-observer
export HF_HOME=/workspace/huggingface
python_bin=/workspace/home-observer-venv/bin/python
completion=artifacts/release-base-v2-policy-exit
deadline=$((SECONDS + 1800))
while [[ ! -f "$completion" ]]; do
  if (( SECONDS > deadline )); then
    printf '%s\n' 'Base evaluation did not finish within the coordination timeout.'
    exit 124
  fi
  sleep 5
done
if [[ "$(cat "$completion")" != "0" ]]; then
  printf '%s\n' 'Base evaluation failed; continuation was not started.'
  exit 1
fi
set +e
timeout 1800 "$python_bin" -m home_observer.train \
  --train data/mixed-v2/train.jsonl \
  --validation data/mixed-v2/validation-dev.jsonl \
  --dataset-root data --policy data/mixed-v2/policy.json \
  --output artifacts/training-v2-continuation/adapter \
  --quantization none --max-steps 100 --rank 16 \
  --gradient-accumulation-steps 2 --learning-rate 2e-5 \
  --decision-field-order actions-first \
  --warm-start-adapter artifacts/training-v2/adapter/checkpoint-300 \
  --generation-validation data/mixed-v2/generation-validation.jsonl \
  --generation-eval-every 50 --checkpoint-every 50 \
  > artifacts/training-v2-continuation.log 2>&1
result=$?
printf '%s\n' "$result" > artifacts/training-v2-continuation-exit
exit "$result"

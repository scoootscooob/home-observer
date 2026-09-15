#!/usr/bin/env bash
# Native image+audio Gemma4 server. The project observer/policy remains separate.
set -uo pipefail
umask 077
cd /workspace/home-observer
VLLM_RUNTIME=/workspace/home-observer-vllm-venv
export PATH="$VLLM_RUNTIME/bin:$PATH"
mkdir -p artifacts/vllm
export HF_HOME=/workspace/huggingface
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=240
export PYTHONUNBUFFERED=1
export MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=2

setup_environment() {
set -e
if ! command -v uv >/dev/null 2>&1; then
    python -m venv /workspace/home-observer-vllm-bootstrap
    /workspace/home-observer-vllm-bootstrap/bin/python -m pip install 'uv==0.10.9'
    UV_BIN=/workspace/home-observer-vllm-bootstrap/bin/uv
else
    UV_BIN="$(command -v uv)"
fi
if [[ ! -x "$VLLM_RUNTIME/bin/python" ]]; then
    "$UV_BIN" venv --python "$(command -v python3)" "$VLLM_RUNTIME"
fi
timeout --signal=TERM --kill-after=15s 300s "$UV_BIN" pip install --python "$VLLM_RUNTIME/bin/python" -r requirements-vllm.txt -e '.[data]'
"$UV_BIN" pip freeze --python "$VLLM_RUNTIME/bin/python" > artifacts/vllm/requirements-resolved.txt
timeout --signal=TERM --kill-after=15s 300s "$VLLM_RUNTIME/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-4-E4B-it', revision='ee0ef6023621cff504d758262d4e04895a5af4a2',
                  ignore_patterns=['*.gguf', '*.md'], max_workers=4)
PY
test -s /dev/shm/home-observer-inference-token
}

(set -e; setup_environment) > artifacts/vllm/bootstrap.log 2>&1
setup_exit=$?
printf '%s\n' "$setup_exit" > artifacts/vllm/setup-exit
printf '%s\n' "$setup_exit" > artifacts/setup-exit
if [[ "${HOME_OBSERVER_SETUP_ONLY:-0}" == 1 ]]; then exit "$setup_exit"; fi
server_pid=''
frontend_pid=''
telemetry_pid=''
server_exit=$setup_exit
if [[ "$setup_exit" == 0 ]]; then
export VLLM_API_KEY="$(cat /dev/shm/home-observer-inference-token)"
export HOME_OBSERVER_ENGINE_TOKEN="$VLLM_API_KEY"
export HOME_OBSERVER_API_TOKEN="$VLLM_API_KEY"
"$VLLM_RUNTIME/bin/vllm" serve google/gemma-4-E4B-it \
    --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
    --served-model-name home-observer \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image":4,"audio":1,"video":0}' \
    --mm-processor-kwargs '{"max_soft_tokens":140}' \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --host 127.0.0.1 \
    --port 8001 > artifacts/vllm/server.log 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > artifacts/vllm/server.pid

# Start the observer protocol only once its single model engine is healthy.
"$VLLM_RUNTIME/bin/python" - > artifacts/vllm/frontend-startup.log 2>&1 <<'PY'
import os, time
import httpx
deadline = time.monotonic() + 240
with httpx.Client(timeout=5, headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}) as client:
    while time.monotonic() < deadline:
        try:
            response = client.get("http://127.0.0.1:8001/health")
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        raise SystemExit("Engine health deadline exceeded; frontend not started")
PY
health_exit=$?
printf '%s\n' "$health_exit" > artifacts/vllm/engine-health-exit
if [[ "$health_exit" != 0 ]]; then server_exit=$health_exit; fi
if [[ "$health_exit" == 0 ]]; then
    "$VLLM_RUNTIME/bin/python" -m home_observer.serve \
        --dataset-root data --backend vllm --quantization none \
        --engine-endpoint http://127.0.0.1:8001/v1 --engine-model home-observer \
        --max-new-tokens 768 --host 127.0.0.1 --port 8000 \
        > artifacts/vllm/frontend.log 2>&1 &
    frontend_pid=$!
    printf '%s\n' "$frontend_pid" > artifacts/vllm/frontend.pid
    "$VLLM_RUNTIME/bin/python" - > artifacts/vllm/frontend-health.log 2>&1 <<'PY'
import os, time
import httpx
with httpx.Client(timeout=5, headers={"Authorization": "Bearer " + os.environ["HOME_OBSERVER_API_TOKEN"]}) as client:
    for _ in range(15):
        try:
            response = client.get("http://127.0.0.1:8000/health")
            if response.status_code == 200 and response.json().get("status") == "ready":
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        raise SystemExit("Frontend health deadline exceeded")
PY
    frontend_health_exit=$?
    printf '%s\n' "$frontend_health_exit" > artifacts/vllm/frontend-health-exit
    if [[ "$frontend_health_exit" == 0 ]]; then
        timeout --signal=TERM --kill-after=5s 1200s nvidia-smi \
            --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw \
            --format=csv -l 1 > artifacts/vllm/gpu-telemetry.csv 2> artifacts/vllm/gpu-telemetry-error.log &
        telemetry_pid=$!
        printf '%s\n' "$telemetry_pid" > artifacts/vllm/telemetry.pid
    else
        server_exit=$frontend_health_exit
    fi
fi
fi

# Keep the disposable Pod available for diagnostics and the separate frontend.
# Provider stopAfter plus the Runpod supervisor remain the hard outer deadline.
while [[ ! -f /workspace/home-observer-inference-finish ]]; do
    if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        server_exit=$?
        printf '%s\n' "$server_exit" > artifacts/vllm/server-exit
        server_pid=''
    fi
    if [[ -n "$frontend_pid" ]] && ! kill -0 "$frontend_pid" 2>/dev/null; then
        wait "$frontend_pid"
        frontend_exit=$?
        printf '%s\n' "$frontend_exit" > artifacts/vllm/frontend-exit
        if [[ "$frontend_exit" != 0 ]]; then server_exit=$frontend_exit; fi
        frontend_pid=''
    fi
    sleep 2
done
if [[ -n "$telemetry_pid" ]]; then
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" || true
fi
if [[ -n "$frontend_pid" ]]; then
    kill -TERM "$frontend_pid" 2>/dev/null || true
    wait "$frontend_pid" || true
fi
if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" || true
fi
if [[ -f artifacts/final-exit ]]; then
    exit "$(cat artifacts/final-exit)"
fi
exit "$server_exit"

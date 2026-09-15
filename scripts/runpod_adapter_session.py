"""Bounded native-LoRA engine/frontend session; keeps base experiment artifacts."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--adapter", required=True)
parser.add_argument("--backend", choices=["vllm", "native"], default="vllm")
parser.add_argument("--output", default="artifacts/adapter-serving")
args = parser.parse_args()
root = Path("/workspace/home-observer")
out = root / args.output
out.mkdir(parents=True, exist_ok=True)
if (out / "started.json").exists():
    raise SystemExit("Refusing duplicate LoRA startup")
runtime = "/workspace/home-observer-vllm-venv" if args.backend == "vllm" else "/workspace/home-observer-venv"
adapter = args.adapter
token = Path("/dev/shm/home-observer-inference-token").read_text().strip()
env = dict(
    os.environ,
    PATH=runtime + "/bin:" + os.environ["PATH"],
    HF_HOME="/workspace/huggingface",
    VLLM_API_KEY=token,
    HOME_OBSERVER_API_TOKEN=token,
    HOME_OBSERVER_ENGINE_TOKEN=token,
    VLLM_NO_USAGE_STATS="1",
    DO_NOT_TRACK="1",
    VLLM_ENGINE_READY_TIMEOUT_S="300",
    PYTHONUNBUFFERED="1",
    MAX_JOBS="2",
    TORCHINDUCTOR_COMPILE_THREADS="2",
)
children = []
status = {
    "controller_pid": os.getpid(),
    "started_at_unix": time.time(),
    "adapter": adapter,
    "adapter_sha256": hashlib.sha256((root / adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
    "adapter_merged": False,
    "backend": args.backend,
    "configured_base_model_id": "google/gemma-4-E4B-it",
    "configured_base_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "prompt_module_sha256": hashlib.sha256((root / "src/home_observer/prompts.py").read_bytes()).hexdigest(),
    "identity_status": "Configured only; requires a real enabled-adapter response",
    "lora_dtype": "auto (base BF16) in vLLM; PEFT FP32 in native. Requires separate validation",
}


def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")


def stop(signum=None, frame=None):
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    if signum:
        raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def start(cmd, logname):
    with (out / logname).open("wb") as log:
        child = subprocess.Popen(
            cmd, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
    children.append(child)
    return child


def ready(url, child, seconds, expected=None):
    deadline = time.monotonic() + seconds
    with httpx.Client(timeout=5, headers={"Authorization": "Bearer " + token}) as client:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Owned server exited during startup: " + str(child.returncode))
            try:
                r = client.get(url)
                if r.status_code == 200 and (expected is None or expected(r.json())):
                    return r
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise RuntimeError("Server readiness deadline exceeded")


try:
    engine = None
    if args.backend == "vllm":
        engine_cmd = [
            runtime + "/bin/vllm",
            "serve",
            "google/gemma-4-E4B-it",
            "--revision",
            "ee0ef6023621cff504d758262d4e04895a5af4a2",
            "--served-model-name",
            "home-observer",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.9",
            "--limit-mm-per-prompt",
            '{"image":4,"audio":1,"video":0}',
            "--mm-processor-kwargs",
            '{"max_soft_tokens":140}',
            "--generation-config",
            "vllm",
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}',
            "--enable-lora",
            "--max-lora-rank",
            "16",
            "--max-loras",
            "1",
            "--lora-modules",
            "observer=" + str(root / adapter),
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ]
        engine = start(engine_cmd, "engine.log")
        status.update(engine_pid=engine.pid, engine_command=engine_cmd)
        save("started.json", status)
        ready("http://127.0.0.1:8001/health", engine, 300)
        with httpx.Client(timeout=10, headers={"Authorization": "Bearer " + token}) as client:
            models = client.get("http://127.0.0.1:8001/v1/models")
            models.raise_for_status()
            model_data = models.json()
            save("registered-models.json", model_data)
            if "observer" not in {x["id"] for x in model_data["data"]}:
                raise RuntimeError("Native LoRA model not registered")
    front_cmd = [
        runtime + "/bin/python",
        "-m",
        "home_observer.serve",
        "--dataset-root",
        "data",
        "--backend",
        "vllm",
        "--quantization",
        "none",
        "--adapter",
        adapter,
        "--no-merge-adapter",
        "--engine-endpoint",
        "http://127.0.0.1:8001/v1",
        "--engine-model",
        "observer",
        "--decision-field-order",
        "actions-first",
        "--max-new-tokens",
        "768",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    if args.backend == "native":
        front_cmd = [
            runtime + "/bin/python",
            "-m",
            "home_observer.serve",
            "--dataset-root",
            "data",
            "--backend",
            "native",
            "--quantization",
            "none",
            "--adapter",
            adapter,
            "--no-merge-adapter",
            "--max-new-tokens",
            "768",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    front = start(front_cmd, "frontend.log")
    status.update(frontend_pid=front.pid, frontend_command=front_cmd)
    save("started.json", status)
    response = ready(
        "http://127.0.0.1:8000/health",
        front,
        180,
        lambda x: (
            x.get("status") == "ready"
            and x.get("adapter_configured") is True
            and (args.backend == "native" or x.get("decision_field_order") == "actions-first")
        ),
    )
    telemetry = start(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "1200s",
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        "gpu-telemetry.csv",
    )
    status.update(telemetry_pid=telemetry.pid, health=response.json(), ready_at_unix=time.time())
    save("ready.json", status)
    while (
        not Path("/workspace/home-observer-inference-finish").exists()
        and (engine is None or engine.poll() is None)
        and front.poll() is None
    ):
        time.sleep(2)
    if not Path("/workspace/home-observer-inference-finish").exists():
        raise RuntimeError("Serving process exited before the finish marker")
except Exception as exc:
    save(
        "failure.json",
        {
            "error_type": type(exc).__name__,
            "error": str(exc).replace(token, "[REDACTED]"),
            "time": time.time(),
        },
    )
    raise
finally:
    stop()
    save("exit.json", {"stopped_at_unix": time.time(), "exit_codes": [p.returncode for p in children]})

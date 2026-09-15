# Run the complete public-data demo

Run these commands from the project directory. The bundled video is one public
EgoLife recording, used in four independent capture processes. It tests transport
and orchestration; it is not synchronized footage from four physical cameras.

## 1. Local tools and private credentials

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev,data]'
mkdir -p work
```

Provide `RUNPOD_API_KEY` through your local secret manager. Generate a separate
inference token and keep it in the shell environment used for both the Runpod
supervisor and local client:

```sh
export HOME_OBSERVER_API_TOKEN="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

The account key stays local. The supervisor sends only the inference token to the
GPU over SSH. The configuration below serves the pinned **base model in BF16**;
saved experimental LoRA adapters have not been accepted for home control.

## 2. Start Runpod inference

```sh
.venv/bin/python scripts/runpod.py keygen --key work/runpod_ssh
.venv/bin/python scripts/runpod.py preflight --config configs/runpod-consumer.json
.venv/bin/python scripts/runpod.py run --config configs/runpod-consumer.json \
  --run-dir work/runpod-consumer --key work/runpod_ssh
```

Keep that supervisor running. On macOS, `caffeinate -i` before the last command
prevents idle sleep. The script installs vLLM, downloads the pinned model, starts
the private engine and observer API, and retains bootstrap logs on failure. This
is a bounded four-hour session. A successful process launch alone does not mean
the model is ready.

In another terminal, read and run the SSH tunnel command from the state file:

```sh
.venv/bin/python -c 'import json; print(json.load(open("work/runpod-consumer/state.json"))["tunnel"])'
```

Use `http://127.0.0.1:8000` through that tunnel. Keep the inference-token environment
available in the client terminal. No public HTTP port is exposed.

## 3. Evaluate a local window through the cloud

```sh
.venv/bin/python -m home_observer.cli evaluate \
  --backend remote --endpoint http://127.0.0.1:8000 \
  --data data/mixed/test.jsonl --dataset-root data \
  --policy data/mixed-v2/policy.json --limit 1 \
  --output work/remote-window
```

Remote evaluation uploads the referenced local image/audio files by default.
`--preloaded-media` instead requires paths already valid on the GPU. Evaluation
simulates device execution; its exit status checks request validity, not an
accuracy acceptance threshold. The generated metrics contain the actual scores.

## 4. Continuous camera/audio capture

```sh
cp configs/demo-device-state.json work/demo-device-state.json
.venv/bin/python -m home_observer.live \
  --sources configs/live-public-demo.json \
  --audio data/public-egolife/raw/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4 \
  --state-file work/demo-device-state.json --policy data/mixed-v2/policy.json \
  --backend remote --endpoint http://127.0.0.1:8000 \
  --output work/live-public --duration 60 --retain-windows 8 --loop
```

The source JSON maps camera IDs to local video paths or RTSP URLs. A camera named
`kitchen` maps to `room.kitchen` and `light.kitchen` in the supplied policy. The
device-state JSON maps exact entity IDs to states, for example
`{"light.kitchen":"off","sensor.ambient_lux":120}`. Keep stream credentials in a
private source file outside the deliverable. Use a fresh output directory per run.

## 5. Actual Home Assistant service calls

With Docker running, start the isolated Home Assistant demo:

```sh
.venv/bin/python scripts/ha_demo.py start
.venv/bin/python scripts/verify_learned_ha.py --endpoint http://127.0.0.1:8000 \
  --policy data/mixed-v2/policy.json --output work/learned-ha
.venv/bin/python scripts/run_live_demo.py --endpoint http://127.0.0.1:8000 \
  --duration 60 --output work/live-ha
```

These scripts check the owned Docker container and read its private credentials
locally. They call real Home Assistant REST endpoints controlling template lights
and a buzzer; no physical household devices are connected. `verify_learned_ha.py`
uses three authored cases and exits unsuccessfully if any case fails. The live
demo reads current HA state each window and executes validated model proposals.

For an actual installation, supply your own sources, entity policy and
`HOME_ASSISTANT_TOKEN`, then use the explicit `--execute-ha --ha-url` options in
[live.md](live.md). Current experimental results do not establish reliable home
control.

## 6. Finish and inspect results

Use the active session's SSH connection to create
`/workspace/home-observer-inference-finish`. The supervisor then stops the servers,
downloads the artifacts, verifies their checksum and deletes the Pod. Check
`cleanup_confirmed` in its state file. Recovery instructions are in
[runpod.md](runpod.md). Stop the local fixture service separately with:

```sh
.venv/bin/python scripts/ha_demo.py stop
```

Read [the experiment dashboard](../reports/index.html) and
[deployment selection](../reports/release-status.json) for measured outcomes.

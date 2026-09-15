# Reproduce unmerged adapter serving

The adapter stays separate from the pinned base weights. Each backend needs its
own sequential validation: native PEFT uses FP32 adapter arithmetic, while vLLM
uses its configured LoRA dtype with BF16 base inference. A configured adapter path
or registered alias does not establish that an actual response used the adapter.
Keep the response metrics and validate the resulting decisions.

Keep the canonical policy serialization in `prompts.py`. JSON objects with equal
values but different key order produce different model input tokens. The matched
policy-order control restored the saved eight-example validation outputs byte
for byte without changing weights. Earlier apparent loading, merging, or batching
regressions therefore must not be attributed to those mechanisms from that
comparison alone.

## Files and exact configuration

- `scripts/runpod_adapter_session.py`: loopback serving, one request at a time,
  private bearer authentication, process supervision, bounded GPU telemetry.
- `scripts/runpod_adapter_job.sh`: isolated runtime setup, then a twenty-minute
  window to receive a verified adapter before starting the model.
- `scripts/runpod_transfer_adapter.py`: transfers exactly six model/processor
  files; optimizer state and other files stay local. Both ends verify SHA256.
- `configs/runpod-adapter-vllm.json`: RTX3090, native LoRA rank16, one sequence.
- `configs/runpod-adapter-native.json`: RTX3090, native PEFT with explicit
  `--no-merge-adapter`, BF16 base, no quantization.

Both use `google/gemma-4-E4B-it` at revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`, four images and one audio window,
8192 input tokens, and at most768 output tokens. vLLM requests the adapter alias
`observer`, uses `--decision-field-order actions-first`, and keeps brightness
controls disabled. Native inference preserves the model's generated field order.
The sequential quality comparisons in this project do not justify batch4 or
batch8 serving equivalence.

## Start a supervised Pod

Run from this project directory. Provide `RUNPOD_API_KEY` in your environment or
in Runpod's existing configuration file. Generate a task SSH key once:

```bash
.venv/bin/python scripts/runpod.py keygen --key work/runpod_ssh
```

Create a separate inference token locally. The cloud launcher sends only this
inference token through encrypted SSH stdin into a private tmpfs file:

```bash
mkdir -p work
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > work/inference-token
export HOME_OBSERVER_API_TOKEN="$(cat work/inference-token)"
```

Choose the validated backend's configuration. For native PEFT:

```bash
.venv/bin/python scripts/runpod.py preflight \
  --config configs/runpod-adapter-native.json
.venv/bin/python scripts/runpod.py run \
  --config configs/runpod-adapter-native.json \
  --run-dir work/deployment --key work/runpod_ssh
```

Keep that supervisor running. In a second terminal, after
`work/deployment/state.json` shows a running Pod and its SSH endpoint, transfer
the selected checkpoint:

```bash
.venv/bin/python scripts/runpod_transfer_adapter.py \
  --source artifacts/training-v2-continuation/adapter/checkpoints/checkpoint-50 \
  --run-dir work/deployment
```

For the vLLM candidate, use `configs/runpod-adapter-vllm.json` in the same steps.
Its clean setup installs the separate vLLM runtime and downloads the base model.
`artifacts/adapter-serving/started.json` records the exact actual command lines;
`ready.json` records configured identity and health, and
`registered-models.json` records vLLM alias registration. Preserve the first real
response's `engine_reported_model` and the validation report before selecting it.

## Connect, validate, and stop

The state file contains the SSH tunnel command. Change its local side to18080
if needed; both remote servers bind loopback, and only SSH is publicly exposed.
Use the private inference token with the local observer's remote backend. Run
`scripts/evaluate_fixed_remote.py` sequentially before the Home Assistant and live
checks. Inspect both decisions and errors; valid JSON is not an accuracy result.

Finish by creating `/workspace/home-observer-inference-finish` through the task's
SSH connection. The adapter session stops its sampler and owned servers. The
supervisor downloads artifacts, verifies their checksum, and only then terminates
the disposable Pod. If collection fails, it stops the Pod and retains the volume
for recovery. The provider and local supervisor enforce the four-hour outer
limit even if a terminal or process fails.

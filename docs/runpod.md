# Runpod training and inference

The runner provisions a single on-demand GPU, uploads the project and selected data through direct SSH, runs the GPU job, downloads its artifacts, verifies SHA-256, and terminates the Pod. Only TCP 22 is exposed. Inference listens on `127.0.0.1:8000` inside the Pod and is reached through the SSH tunnel recorded in `state.json`.

## Credentials and prerequisites

Use Python 3.11+, OpenSSH `ssh`, `scp`, `ssh-keygen`, and a funded Runpod account. The runner reads `RUNPOD_API_KEY` or `api_key` in `~/.runpod/config.toml`. It never prints or sends the Runpod key to the GPU. Use an environment variable provided by your secret manager; do not put a token in a command argument or a committed file.

If model download needs authentication, provide `HF_TOKEN` in the local runner environment. It travels over encrypted SSH stdin to a temporary mode-600 file in the Pod's memory filesystem and is removed after execution. Accept a model's license in Hugging Face before starting a paid job when required.

For an authenticated inference session, supply `HOME_OBSERVER_API_TOKEN` through
the local runner environment. It is transferred over SSH stdin into a private
tmpfs file; it is not part of the provider Pod configuration or uploaded bundle.

The project bundle includes `src`, `scripts`, `configs`, `docs`, `tests`, `data`, `datasets`, `pyproject.toml`, and GPU requirements. It excludes `work`, previous artifacts, hidden files, symlinks, and private-key extensions. Put only intended training/inference inputs in the data folders.

## Run training

From the project directory:

```bash
python scripts/runpod.py plan --config configs/runpod-training.json
python scripts/runpod.py preflight --config configs/runpod-training.json
python scripts/runpod.py keygen --key work/runpod_ssh
python scripts/runpod.py run --config configs/runpod-training.json --run-dir work/runpod-training --key work/runpod_ssh
```

On macOS, prefix the final command with `caffeinate -i` so normal idle sleep does not suspend the supervisor. Keep this process running. The key is scoped to the Pod and the project; no global SSH or Runpod account settings are modified.

The training config selects an H100 80 GB for fast BF16 baseline and LoRA training. Its four-hour job deadline and $64 per-job guard are runaway protection, not a total research budget. Live pricing must fit the configured $4/hour maximum. The actual created Pod rate is checked again before GPU work. Current documented running storage pricing is included conservatively; the provider's eventual billing statement remains the cost authority.

`scripts/runpod_job.sh` installs pinned Torch 2.10/CUDA 12.8 and GPU dependencies and records the software and NVIDIA environment. Its current reference recipe trains on `data/mixed-v2`: 432 authored windows with balanced actions and causal prior contexts, plus 64 human-captioned natural images. It runs 400 steps with actions-first serialization, generation validation and full resumable checkpoints every 100 steps, then evaluates the final candidate on the unchanged held-out suites in batches of 4. The result is an experimental candidate, never an automatic deployment selection. The historical v1 run used 160 steps; the original v2 run was interrupted after saving checkpoint 300. A new clean 400-step recipe does not recreate that interruption. The separate `scripts/model_smoke.py` remains available for a two-step engineering smoke test.

## Warm a GPU while finishing the payload

`configs/runpod-warm.json` runs the same lifecycle, but waits up to 15 minutes for `/workspace/home-observer-launch-ready`. After `state.json` has its SSH address, the controller can upload final code and data, verify the archive, and create that marker. A missing marker exits with failure and triggers ordinary artifact retrieval and cleanup. This mode never waits indefinitely.

## Inference Pod

Copy the verified adapter from the downloaded training artifacts into `data/adapter`. Then:

```bash
python scripts/runpod.py preflight --config configs/runpod-inference.json
python scripts/runpod.py run --config configs/runpod-inference.json --run-dir work/runpod-inference --key work/runpod_ssh
```

The general native inference config targets a 24 GB RTX 4090 and requires the explicitly staged adapter above. The current consumer session uses `configs/runpod-consumer.json`: a Community RTX 3090 with at least 16 GB host RAM and two host vCPUs, running the pinned base model in BF16 through vLLM. It needs no staged adapter. Its GPU has 24 GB VRAM; host RAM is a separate limit. See [the complete demo](quickstart.md) and [vLLM setup](vllm.md). This is a bounded session, not an always-on service. Open the command in `state.json`'s `tunnel` field in another terminal, then use the local `http://127.0.0.1:8000` URL. Validate actual responses and runtime measurements before selecting a smaller GPU.

The consumer config currently launches the isolated [vLLM experiment](vllm.md).
Its engine listens on loopback port 8001; the observer protocol adapter is a
separate process on port 8000. A running lifecycle does not establish that either API is ready.

## Lifecycle evidence and recovery

Each run directory contains a durable `state.json`, the exact uploaded archive, and, after retrieval, `artifacts.tar.gz` plus `downloaded/artifacts`. State includes the live quote, provider Pod ID, deadline, job exit code, transfer checksum and cleanup confirmation. A successful program exit is not a claim that model quality passed; inspect the experiment report inside the artifacts.

- A unique creation intent is written before the provider mutation. An ambiguous response is never retried. Run `python scripts/runpod.py reconcile --run-dir work/runpod-training` to locate that exact Pod by its recorded unique name. No replacement Pod is created.
- Read-only running-job status checks tolerate transient SSH/API errors for up to 300 seconds, bounded by the job deadline, and refresh a changed SSH endpoint. Errors and recovery counts are retained. Provisioning and device mutations are never retried by this mechanism.
- Running-job status reads tolerate up to 300 seconds of consecutive SSH failures,
  bounded by the execution deadline. They refresh the SSH endpoint from Runpod
  and preserve error/recovery counts. Confirmed terminal process/provider state
  still ends the job. Create, upload, launch and cleanup mutations are never
  replayed by this retry loop. An endpoint change also requires refreshing any
  separately opened local tunnel.
- `ssh-errors.jsonl` retains sanitized SSH/SCP stderr in the private run directory
  with mode 600. It excludes commands, stdin and stdout; diagnostics do not appear
  in ordinary status output.
- After any job failure, the runner stops the process group and attempts artifact retrieval before cleanup. A verified download permits Pod deletion.
- If artifacts cannot be retrieved, the Pod is stopped to preserve `/workspace`. **Stopped volume storage remains billed** until the Pod is deleted. The state records `stopped_recovery_required`.
- To recover, resume the recorded Pod in Runpod, run `python scripts/runpod.py recover --run-dir work/runpod-training --key work/runpod_ssh`, then `python scripts/runpod.py cleanup --run-dir work/runpod-training`. Resume is a paid operation; do not create another Pod as a substitute for recovery.
- `python scripts/runpod.py status --run-dir work/runpod-training` inspects current state. `stop` requests a stop preserving the volume. `cleanup` refuses to delete a Pod without verified local artifacts.

Two deadlines protect normal execution: an on-Pod process timeout stops GPU work before a ten-minute download/cleanup reserve, and a provider `stopAfter` timestamp requests Pod shutdown at the final deadline. The local supervisor also enforces that deadline. Provider scheduling/API outages or a crashed/suspended controller can delay cleanup; these controls are not a guaranteed billing cap. A provider stop preserves the volume, so recovery storage is intentionally outside the normal finished-job estimate. Check `cleanup_confirmed` and the Runpod console after an interrupted run.

## API sources

The implementation uses the official [GraphQL schema](https://graphql-spec.runpod.io/) for the account, live GPU price/stock, Pod create with `stopAfter`, and status. It uses REST v1 for [stop](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop) and [delete](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId). See [SSH access](https://docs.runpod.io/pods/configuration/use-ssh), [storage pricing](https://docs.runpod.io/pods/pricing), and the [official verified PyTorch image example](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod/golden-paths/09-custom-serverless-dev-loop/README.md).

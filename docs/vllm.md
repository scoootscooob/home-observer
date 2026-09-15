# Experimental vLLM inference

`scripts/vllm_session.sh` installs `vllm[audio]==0.29.0` in
`/workspace/home-observer-vllm-venv`. It leaves the native inference and training
environments unchanged. The resolver records the exact installed packages in
`artifacts/vllm/requirements-resolved.txt`.

The server uses the same pinned Gemma4 E4B base revision and cached weights, BF16,
an 8192-token context, one concurrent sequence, four images and one audio item.
It binds only `127.0.0.1:8001`. After engine health succeeds, the session starts
the authenticated observer protocol on `127.0.0.1:8000`, using the vLLM backend
and a 768-token output limit. The local observer owns policy validation and Home
Assistant execution. Model health alone does not establish observation or action quality.

The private `/dev/shm/home-observer-inference-token` is loaded through
`VLLM_API_KEY`, never a command-line argument. vLLM API authentication does not
protect every management endpoint, so the loopback bind and SSH access are part
of this configuration. A clean Pod bootstraps from its system Python, installs
the project/data dependencies into the isolated environment, and downloads the
pinned model before engine startup. Package installation and model download each
have five-minute deadlines; engine readiness has a four-minute deadline. Check
bootstrap progress after ten minutes rather than claiming readiness from a
running process. The session remains available for diagnostics until the task
finish marker or its outer provider/job deadline.

The observer frontend uses `HOME_OBSERVER_ENGINE_TOKEN`, loaded from the same
private tmpfs token file, when connecting to the loopback engine. The session
exports it for processes launched within that session; a separately launched
frontend must load it explicitly. Only the inference key is transferred to the
Pod; the Runpod account credential stays local.

## Official references

- [vLLM server quickstart and API authentication](https://docs.vllm.ai/en/stable/getting_started/quickstart/)
- [Multimodal inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- [API-key authentication limitations](https://docs.vllm.ai/en/stable/usage/security/)

Actual startup and image/audio probe outcomes belong in `artifacts/vllm`; this
document does not assert that the experiment succeeded before those checks run.

## Verified startup and retained failure

The replacement RTX 3090 reached engine readiness and authenticated observer
readiness. `artifacts/vllm/startup-recovery.json` records the running configuration
and `frontend-ready.json` records the observed health response. The first startup
failed because FlashInfer could not find `ninja`, despite the package being
installed. The session now prepends its isolated environment's `bin` directory to
`PATH` and lists `ninja` explicitly. The original failure log is retained as
`server-initial-ninja-failure.log`; it is separate from the successful restart.

After frontend readiness, the session samples GPU utilization, memory and power
once per second for at most 1,200 seconds. It stops that sampler before artifact
collection. The raw `gpu-telemetry.csv` supports measured resource reporting;
startup success alone does not establish model quality or streaming throughput.

## Decision contract

The observer frontend constrains JSON generation to the configured entity IDs,
evidence IDs, services and maximum action count. Contract version
`executor_arguments_v3` also enforces the rule that `noop: true` has empty
observations and actions. The normal policy and evidence checks still run after
generation. A syntactically valid response can still make an incorrect decision.

Action data is empty by default, matching the demo's on/off-only lights. Enable
`--allow-brightness` only for an installation whose lights support brightness;
the permitted argument is then an integer `brightness` from 1 through 255.
`brightness_pct` is not accepted. Use `--decision-field-order actions-first`
when testing the adapter trained with that serialization; base measurements use
`summary-first`. Health responses and prediction metrics record these settings.

Gemma4 LoRA support exists in the pinned vLLM version. This does not establish
equivalence to the native PEFT implementation: saved adapters contain FP32
tensors, whereas vLLM LoRA uses the configured lower precision. Compare actual
decisions before selecting a serving backend. Fresh-loaded unmerged single,
batched and merged versions initially differed from the training callback on
the same two visual validation cases. A matched control traced the fresh-single
difference to policy JSON field order: restoring the authored order reproduced
all eight callback outputs byte for byte, with identical media tensors and no
numerical changes. `build_context` now canonicalizes that field order. Earlier
comparisons did not isolate a merge or batch effect and remain historical
variants. Report each configuration and prompt version separately.

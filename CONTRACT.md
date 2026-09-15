# Home Observer implementation contract

Build a replayable home observer with a single native multimodal model, an explicit SQLite journal, and a Home Assistant execution adapter. Primary model candidate: `google/gemma-4-E4B-it` (native image/audio input; confirm runtime support). Qwen3.5-4B may be a visual/text baseline. Do not claim indefinite recurrent streaming: v1 uses bounded windows and explicit state. Run training and learned inference on Runpod; local deterministic fixtures only verify plumbing.

## Shared JSON contracts

`ObservationWindow`: `window_id` string, `started_at` and `ended_at` float Unix seconds, `frames` list of `{camera_id, timestamp, path, evidence_id}`; `audio` list of `{microphone_id, started_at, ended_at, path, evidence_id}`; `device_states` mapping entity IDs to JSON states. Paths can be absolute local paths on the machine running inference; datasets use paths relative to their dataset root. Windows must only include data at or before ended_at.

`Decision`: `summary` string, `observations` list of `{entity_id, attribute, value, confidence, evidence_ids}`; `actions` list of `{domain, service, entity_id, data, reason, evidence_ids}`; `noop` bool. No action is inferred merely from missing annotations. Entity IDs and permitted actions are supplied by the runtime policy. Observations are uncertain and require evidence IDs from the current window. Device-state evidence uses `device:<entity_id>`.

Training JSONL row: `{id, split, source, window: ObservationWindow, target: Decision}`. Include source provenance, annotation method, and group IDs as extra fields. Split by recording/source/scenario family, not random frames. Keep public-data supervision separate from synthetic rule labels. Real replay eval must be separately reported from fixtures. Model inputs must NEVER include labels, future frames, teacher output, or held-out target facts.

Inference request JSON: `{window: ObservationWindow, state: object, recent_events: list, policy: object}`. Inference response: `{decision: Decision, metrics: {latency_s, input_tokens, output_tokens, ...}}`. HTTP `/observe`, `/health`; optional `/reload`. API access requires a bearer token if listening on non-loopback. Backend also callable offline from Python. The model receives raw visual/audio inputs plus bounded state, journal context, policy, and timestamps.

## Ownership

- Root owns `schema.py`, `prompts.py`, journal, engine, policy, HA adapter, local demo/dashboard, packaging and integration.
- Data worker owns `data.py`, `replay.py`, scripts/data*, tests/test_data*, tests/test_replay*, and docs/data.md.
- Model worker owns `model.py`, `train.py`, `serve.py`, scripts/model*, scripts/train*, tests/test_model*, requirements-gpu.txt, docs/model.md.
- Runpod worker owns `cloud.py`, scripts/runpod*, tests/test_cloud*, configs/runpod*, docs/runpod.md.

All workers share this checkout. Do not revert another worker's edits. Avoid changing shared interfaces without messaging root. No cloud provisioning or paid jobs from workers; root handles account access, budget, actual deployment, and termination after artifact retrieval.

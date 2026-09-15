# Production runtime: local perception, enforced export boundary, sandboxed frontier

This is the production-shaped path. It is separate from the experimental
Runpod/media-capable bridge described in [recipe workflow](recipe-workflow.md),
which remains research-only. The production path keeps every sensory byte on
the host and is enforced by code and tests, not only by documentation. It is
still a prototype: model quality is measured separately and remains weak.

## Three enforced layers

| Layer | Mechanism | Where |
|---|---|---|
| Local inference | `PhysicalLocalModel` runs the pinned Gemma 4 E4B weights and the unmerged v1 adapter in-process on Apple Metal. It declares `local_inference=True` and `sensory_media_leaves_host=False`; `assert_production_model` rejects `PhysicalRemoteModel`, `PhysicalVLLMModel`, anything with an HTTP client attribute, or any backend that does not make the declaration. `ProductionRuntimeConfig` refuses configs naming `url`, `engine_url`, `base_url`, `endpoint` or `token_file`. | `physical_local.py`, `production_export.py` |
| Export boundary | `ProductionExporter` consumes private outbox envelopes and builds typed `ExportEnvelope` messages from explicitly selected fields. The sanitized content is hashed after sanitization (`export_sha256`), persisted, leased, retried with backoff and acknowledged by `export_id` + hash. A keyed `audit_ref` maps each export back to the private record locally. Unknown topics and anything failing the leak scanner are blocked and recorded, never sent. | `production_export.py` |
| Host isolation | `ProductionCoordinator` writes only exports, sanitized command outcomes and rejection classes into the bridge directory. The frontier coordinator process (`scripts/frontier_coordinator.py`, standard library only) runs under `scripts/frontier_sandbox.py`: a deny-by-default `sandbox-exec` profile that allows reading its interpreter and the bridge, writing the bridge, and TCP to one local proxy port. A test proves it cannot read a frame outside the bridge or connect to another port. | `production_coordinator.py`, `scripts/frontier_sandbox.py` |

## What the frontier receives

Versioned schema `home-observer.production-export/1`, four message types:

- `physical_event`: opaque `pevt_*`/`phys_*` IDs, `origin`, a `kind` and `object_category` from the local vocabulary (`configs/production/export-policy.json`; anything else becomes `other` with `*_in_vocabulary=false`), an `area` token from the camera map, wall-clock `started_at`/`ended_at`/`available_at`, rounded `confidence`, a bounded `uncertainty_level` (geometric events never export as `low`), `evidence_count`, `audit_ref`.
- `watch_status`: watch/commitment/response IDs, a status enum, event IDs, clocks, `response_timely`, `verification_kind`, `readback_state` (`on`/`off`/`unknown`) and a `failure_class` token. No spec text, no evidence IDs, no executor response body.
- `perception_status`: `completed`/`invalid`/`failed`, a failure class, frame counts, wall-clock window bounds, latency and accepted item counts. No window, no paths, no raw output, no model identifiers.
- `verification_result`: the local verifier verdict (`confirmed`/`contradicted`/`unknown`), confidence, uncertainty band, a bounded `reason_code`, evidence span and frames examined. The private audit (frame IDs, boxes, displacement, learned check details) stays in the export audit table.

The leak scanner (`scan_export`) additionally rejects any string containing a path separator, URL scheme, `data:`/base64 markers, media or file extensions, whitespace free text, non-ASCII, long strings, base64-like runs, denied keys such as `path`, `raw_output`, `description`, `uncertainty`, `spec`, `result`, `window`, and non-finite numbers. Tests replay the eight recorded experimental envelopes, which contain absolute frame paths, a manifest path, raw model output and the adapter path, and assert none of it crosses.

## Commands the frontier may send

`create_watch`, `cancel_watch`, `ack_export` (export ID and hash; a private delivery ID is rejected) and `verify` (a `VerificationRequest`: one of `moved_from_initial_location`, `clear_of_support`, `still_present`, `pickup_completed` for a `phys_*` subject). Command outcomes written back are scanned as well; anything else is suppressed to `{"accepted": true, "detail_suppressed": true}`. Rejections expose only a packet ID and an error class; the detail lives in the private directory next to the run output.

## Run the production-shaped workflow

The runner needs the Metal-capable environment (`work/model-venv`, Python 3.12 with torch, transformers 5.17, peft 0.21) and the sandbox needs `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_BASE_URL` for the local inference proxy in the caller's environment (never printed, passed only to the sandboxed process).

```bash
HF_HUB_OFFLINE=1 work/model-venv/bin/python scripts/run_production.py \
  --video data/continuous-v2/videos/P28_07.mp4 --camera-id recipe --source-origin 0 \
  --runtime configs/production/runtime.json \
  --output reports/production-<name> --bridge reports/production-<name>-bridge \
  --responses configs/recipe-workflow/responses.json \
  --digital-context configs/recipe-workflow/digital-context.json \
  --ha-credentials /absolute/private/path/ha-credentials.json \
  --seed <seed.json> --frontier sandboxed --frontier-model claude-opus-5
```

The runner refuses non-empty output or bridge directories, resets the isolated demo light to off and records it, starts the sandboxed frontier, waits for its watch, streams, drains pending verifications, stops the frontier, scans every frontier-visible file and writes `privacy-evidence.json` and `production-summary.json`. `--motion-threshold` and `--motion-confirm-frames` set the geometric tracker's
sensitivity (defaults 0.003 and 3 consecutive moving frames; a single re-centering step
after seeding is never a motion event). `--frontier none` lets a test or an external
process act as the frontier. `--interrupt-frontier-at-seconds` kills and relaunches the sandboxed coordinator mid-run; its state file lets it resume and acknowledge the exports it missed.

`output_contract` in the runtime config selects `strict` (complete decision required) or `judged-fields` (the adapter's judged fields are accepted and the unsupervised `summary`, `confidence` and `uncertainty` are completed with explicit, recorded placeholders; evidence citations are derived from the claimed interval and the model's own citations are ignored and counted). Placeholder confidence never satisfies a watch that requires a model judgment above 0.5.

## Verification of the boundary

`tests/test_production_export.py`, `tests/test_production_coordinator.py`, `tests/test_frontier_sandbox.py` and `tests/test_run_production.py` cover: replay of the recorded leaked envelopes, adversarial paths/base64/free text through events, failures, diagnostics and unknown topics, hash discipline and acknowledgement binding, lease/retry identity, guard failures for remote inference and media-capable bridges, sandbox confinement (denied read, denied listing, denied write, denied port; allowed bridge and allowed port), and an end-to-end synthetic run whose private outbox still contains paths while the bridge contains none.

What this does not prove: that the model is right, that the tracker identity is correct, or that a household deployment is reliable. Those are measured separately in [physical evaluation](physical-evaluation.md) and the reports.

## Presenting the node to an agent gateway (Agent Hardware Protocol)

The production boundary has the same shape as the
[Agent Hardware Protocol](https://github.com/scoootscooob/agent-bridge-sdk) used by
OpenClaw hardware nodes: a device self-describes typed **properties**, **commands** and
**events** with device-side emission policies, and an agent host discovers and reacts to
them over JSON-RPC. `home_observer.ahp_manifest` derives an AHP manifest from the enforced
export and command schemas, so the self-description cannot drift from what the boundary sends:

```bash
.venv/bin/python -m home_observer.ahp_manifest > home-observer-ahp-manifest.json
```

| AHP primitive | Home Observer equivalent |
|---|---|
| `ahp.event` (`physical_event`, `watch_status`, `perception_status`, `verification_result`) | `ExportEnvelope` messages, scanner-checked before any transport |
| `ahp.command.invoke` (`create_watch`, `cancel_watch`, `verify`, `ack_export`) | `ProductionCommand`, idempotent by command ID |
| Event policies (`debounce_ms`, `max_rate_per_min`, `wake`) | Local watches and bounded outbox emission; no media firehose exists to throttle |
| Properties | `schema_version`, `media_leaves_device=false`, `pending_exports`, `armed_watches` |

In that layout an ESP32 running the AHP SDK is a peer device on the same gateway: it can
carry the camera or microphone into the local sensing server over the LAN, and it can be the
approved responder (a light or LED) that the local watch drives through a typed command.
The gateway and the agent behind it still receive only the manifest and typed events;
frames never enter the protocol. The transport binding (WebSocket to the gateway) is not
implemented here yet; the manifest, schemas and tests are.

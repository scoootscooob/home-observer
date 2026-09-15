# Recipe step → physical observation

**Research path only.** This recorded workflow uploads media to a Runpod-hosted
adapter and lets a frontier subagent inspect images, which production forbids.
The production-shaped workflow (local Metal inference, exports-only bridge,
sandboxed frontier) is `scripts/run_production.py`, described in
[production runtime](production-runtime.md) and reported in
[PRODUCTION_REPORT.md](../PRODUCTION_REPORT.md).

The recorded workflow uses a real, continuous public kitchen clip, a Runpod-hosted
physical adapter, an actual frontier subagent, and an isolated Home Assistant
template light. It does not connect to a user's home or establish whole-house
reliability. See the [physical iteration report](../PHYSICAL_REPORT.md) for results.

## Responsibility and evidence

1. The digital instruction asks for a cue when the nested bowls start moving,
   followed by visual verification of the pickup.
2. A frontier subagent reads that instruction and the approved response registry,
   then creates a one-shot `motion_started` watch. The watch cannot invent a
   Home Assistant service call.
3. An explicitly recorded initial box seeds local CSRT tracking. This is an
   assistant-selected bootstrap, not a model detection or human tracking label.
4. Capture retains the full sequence while tracking and Runpod inference run in
   separate workers. Local geometry can trigger the cue before model inference
   finishes. The model receives physical context, without the digital recipe or
   dataset target labels.
5. The existing executor sends the approved light command once and verifies its
   actual Home Assistant state. The preparation record establishes that it was
   off before the cue.
6. The frontier subagent receives the event and perception envelopes, opens the
   chronological image evidence, acknowledges exact delivery IDs and hashes,
   and records a separate physical-outcome judgment.

Light readback proves the cue's device state. Geometric motion proves neither
pickup nor recipe completion. Invalid model output is retained and delivered as
`perception.failed`; it cannot become an accepted learned event through this
transport. Frontier verification must cite the actual images independently.

## Run another recorded demonstration

Install `.[dev,data,sensing]` and start the authenticated Runpod upload frontend
as described in [physical model](physical-model.md). Forward its loopback port
through SSH. Keep credentials outside this project; no provider credential is
needed in a perception request.

Use a **fresh** output and bridge directory. Supply a fresh coordinator-created
watch with current `created_at` and a bounded TTL. Existing recorded watches and
journals are immutable evidence; do not overwrite or replay them as new runs.
The approved response file in `configs/recipe-workflow` targets only the isolated
demo light. Verify that light is off before starting a cue demonstration.

```bash
.venv/bin/python scripts/run_physical.py \
  --video data/temporal-pilot/clips/030c003244925eb728dd.mp4 \
  --camera-id recipe --source-origin 608.99 \
  --url http://127.0.0.1:18080 \
  --token-file /absolute/private/path/inference-token \
  --output reports/recipe-workflow-new \
  --bridge reports/recipe-workflow-new/bridge \
  --responses configs/recipe-workflow/responses.json \
  --ha-credentials /absolute/private/path/ha-credentials.json \
  --seed configs/recipe-workflow/seed.json \
  --tracker csrt --opencv-threads 1 \
  --wait-for-watch-seconds 60 --post-source-watch-seconds 60
```

The local bridge must keep ticking to process acknowledgements. The recorded
command stops after source EOF, pending inference, and its configured final drain.
It does not keep a watch active after the process exits. A persistent deployment
needs a continuously running capture/watch service and a coordinator delivery
worker; this short replay does not establish that deployment's reliability.

## Inspect a run

- `runtime-ready.json`: source hash, source kind and startup time.
- `bootstrap/seed.json`: explicit box provenance and persistent physical ID.
- `sequences/*/manifest.json`: all retained frames and the subset sent to the model.
- `tracks.jsonl`, `geometric-events.jsonl`: tracking states and local event production.
- `predictions.jsonl`: actual Runpod decisions, raw output, errors and inference clocks.
- `journal.sqlite3`: physical registry, tracking history, watch reservations and deliveries.
- `bridge/results`, `bridge/outbox`: accepted commands and durable delivery envelopes.
- `coordinator/`: actual frontier planning, evidence review and outcome provenance.
- `stream-report.json`: capture losses, uncertainty, watch status and measured delays.

Source time, local event availability, model completion, command readback, transport
acknowledgement and frontier outcome review are separate clocks. In particular,
frontier review delay includes this interactive agent's scheduling and inspection;
it is not a benchmark of a continuously running production coordinator.

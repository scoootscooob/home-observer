# Continuous capture and retention

`python -m home_observer.live` runs independent ffmpeg camera/audio capture while inference consumes the latest available media. This counts captured media that was not submitted to the model; it does not claim that the model understood every captured frame.

The [complete demo](quickstart.md) includes exact setup commands and bundled input templates. `configs/live-public-demo.json` maps camera IDs to the included public video; `configs/demo-device-state.json` supplies simulated device states. Camera ID `kitchen` maps to policy entities `room.kitchen` and `light.kitchen`. Replace the source paths with actual camera URLs and adjust the entity policy for another installation.

## Cloud media transfer

Remote capture uploads each window’s referenced JPEG/PNG and audio bytes to `/observe_upload` over the SSH tunnel, using `HOME_OBSERVER_API_TOKEN` when configured. The server limits each request to 16 MiB decoded media and 24 MiB JSON, stores files only in a temporary directory under its dataset root, and deletes them after inference. Local retained snapshots remain the audit copies. Only the preloaded-data `/observe` endpoint expects files already on the GPU machine.

## Bound the snapshot media

```sh
python -m home_observer.live \
  --sources work/sources.json --audio work/source.mp4 \
  --state-file work/device-state.json --output work/live-session-1 \
  --backend remote --endpoint http://127.0.0.1:8000 \
  --capture-fps 2 --interval 1 --duration 600 \
  --retain-windows 100
```

`--retain-windows` defaults to **100 completed observation windows**, with a minimum of one. During inference, at most one additional active snapshot is protected. Its image/audio copies are never expired while the backend is using them. Once inference returns and the result is written, the oldest completed snapshots expire until the retention count is met. On normal shutdown, the most recent completed snapshots remain available.

The ffmpeg capture buffers separately retain 64 JPEGs per camera and 32 audio chunks. Frame accounting uses three integer counters per camera rather than accumulating every evidence ID. Repeated sampling of the same captured frame increments `sample_events` but not `model_sampled`. Missing capture numbers correctly contribute to `not_model_sampled = captured - model_sampled`.

Capture numbers must increase monotonically for a camera in this invocation. Restarting a camera counter under the same running observer is rejected, rather than silently corrupting omission statistics. Each invocation receives a unique window-ID prefix.

## Historical evidence after expiry

Expiring raw media does not remove its original journal input, decision, facts or action records. The journal receives a `media_expired` event, and the window result gains:

```json
{"media":{"status":"expired","reason":"snapshot_retention","directory":"...","expired_at":1750000000,"capture_ended_at":1749999900}}
```

The original evidence IDs remain part of the historical decision, but their image/audio paths are no longer readable. A result marked `media.status="expired"` must not be presented as currently viewable media. `predictions.jsonl` is a historical append log during the invocation; its earlier `retained` status is a statement at write time. **The SQLite journal result is the current media-status authority.** The final `live_report.json` summarizes retained, active and expired window counts.

## Scope of the bounds

- Retention applies to snapshot directories created by the current invocation. Use a fresh `--output` directory for each run. Unknown files/directories and older runs are deliberately not deleted.
- SQLite journal records, predictions and ffmpeg diagnostic logs are durable and are not rotated by this option. Their disk usage can grow; this implementation bounds live media buffers and in-memory capture accounting, not all audit storage.
- The in-memory error list retains the latest 100 errors; `error_count` preserves the total.
- Source timestamps are receipt wall-clock timestamps at capture, not recovered historical video dates or synchronized original camera PTS.
- The duration stops scheduling new inference windows. An already running inference finishes before shutdown, so elapsed duration may exceed the requested duration.
- Abrupt process termination can leave an active snapshot behind. It is not automatically deleted on a later invocation.

## Verification

`tests/test_live.py` exercises 200,000 repeated/gapped frame-accounting events with fixed-size counters, bounded completed snapshots with active-window protection, and journal expiry markers that preserve original decisions/errors. Its optional real-media test uses the same short EgoLife clip in four independent ffmpeg processes plus one microphone. A deliberately delayed fake backend verifies capture independence, immutable active snapshots, retention rollover, exact omitted-frame counts, causal windows and process shutdown. This is a capture plumbing test, not learned inference or genuine multiview synchronization.

## Read and control a configured Home Assistant

For the project's isolated demo, or an explicitly configured Home Assistant, pass `--execute-ha --ha-url URL` and supply `HOME_ASSISTANT_TOKEN` from your secret manager. The token stays on the capture/controller machine. Each window reads only configured entities into the model input; returned attributes are limited to brightness and illuminance units. Missing devices stay unknown.

```sh
python -m home_observer.live \
  --sources work/sources.json --audio work/source.mp4 \
  --backend remote --endpoint http://127.0.0.1:18080 \
  --policy data/mixed-v2/policy.json \
  --execute-ha --ha-url http://127.0.0.1:8124 \
  --output work/live-ha-session --duration 60 --retain-windows 20
```

The default remains simulation with `--state-file`. The Home Assistant path proposes actions through the same model and policy, reserves each action in SQLite, sends the actual service call, and polls the resulting state without repeating an uncertain command. Capture startup failures and zero-window runs exit unsuccessfully and retain their diagnosis.

## Success and throughput

The final report separates `accepted_windows`, `rejected_windows`,
`uncertain_action_windows` and transport/capture errors. A completed HTTP request
does not establish a usable model decision. The process exits unsuccessfully if
any window is rejected, any action remains uncertain, or capture/transport fails.
`actual_windows_per_s` measures all completed inference windows;
`accepted_windows_per_s` measures only accepted decisions. Neither rate measures
perception accuracy. Earlier historical reports without these counters retain
their original output and should not be interpreted as passing on `error_count`
alone.

Each result records encoded-media SHA-256 hashes and decoded RGB hashes for all
camera frames, preserving source identity even after snapshot expiry. The
four-source benchmark prepared by `scripts/prepare_live_demo.py` uses time
offsets of one public recording. Its provenance file explicitly identifies that
scope; four distinct images are not evidence of four synchronized camera views.

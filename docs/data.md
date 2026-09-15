# Data and causal replay

Three source types are produced and evaluated separately:

| Dataset | What is real | Supervision | Valid conclusion |
| --- | --- | --- | --- |
| `data/fixtures` | Nothing: authored camera drawings, device readings and generated tones | Exact synthetic geometry and explicit device rules | Packaging, causal inputs, training/evaluation plumbing, simple synthetic visual understanding |
| EgoLife public sample | A downloaded household recording with its original audio | None; no action/no-op labels are invented | Media download/extraction and model inference smoke test only |
| `data/coco-household` | COCO photographs and original human captions | Summary only; other Decision fields masked | Natural-image caption adaptation, not temporal home control |

## Generate fixtures

```sh
python -m home_observer.data fixtures data/fixtures
```

The default is 96 windows, split 32/32/32 into train/validation/test. Each window contains four camera IDs (`kitchen`, `entry`, `lounge`, `garage`), source timestamps, one PNG per camera, a 16 kHz mono WAV and device states. PNGs are drawn by a deterministic stdlib renderer; WAVs contain a generated tone or silence, never speech. No target text or action names are painted onto the images.

`policy.json` contains the explicitly authored rule and entity surface. Pass it separately with `--policy data/fixtures/policy.json` to training/evaluation. The target is never used to construct the input policy.

There are two task types, both available as combined and separate split files:

- `device_control`: explicit motion + darkness + light-off readings trigger `light.turn_on`. Negative examples deliberately cover empty rooms, bright occupied rooms and already-on lights. The deterministic device backend is an oracle for this plumbing subset only.
- `visual_occupancy`: motion sensors are removed entirely. `room.<camera_id>.occupied` targets are derived from the authored person geometry; actions must use the camera evidence. Empty and occupied dark-room windows have identical device readings. A device-only baseline cannot distinguish them. No-op and no-external-action are distinct: a decision that records occupancy but takes no action has `noop=false`, following the shared schema.

Entire generated household/routine families and all their variants stay in a single split; the visual and device variants of the same scene stay together. Palette/person-position variations depend on the split/family seed. All splits share the same authored lighting policy, so these are tests of simple synthetic visual inputs, not evidence of generalization to real homes or unseen policies. `task_type`, `group_id`, `recording_id`, annotation method and provenance are row metadata, never model inputs.

The fixture oracle also implements a **local simulated buzzer** rule for an explicitly active leak sensor. The default fixture sequence does not contain a leak event. Generated tone audio is not labeled as a real fire or leak alarm.

## Download and prepare one real public sample

Install `huggingface-hub` and either `imageio-ffmpeg` (portable binary) or a system `ffmpeg`. The project has a data dependency extra; a bare dependency install also works:

```sh
pip install huggingface-hub imageio-ffmpeg
python -m home_observer.data public-sample work/public-egolife --max-mib 32
```

The downloader uses the dedicated Hugging Face Hub SDK and lists only `A1_JAKE/DAY1` in `lmms-lab/EgoLife`. It chooses one MP4 below the byte cap, resolves and pins the revision, and writes a provenance manifest with original path/URL, revision, byte count, SHA-256 and the dataset card's declared license. It never clones the 512 GB corpus. The `--revision` flag allows reproducing a recorded revision, and `--no-prepare` skips extraction.

The first 20 seconds or the available clip duration, whichever is shorter, are extracted into at-most-10-second windows: 384 px-wide JPEG frames sampled every five seconds and 16 kHz mono audio. The source clip's duration/audio presence is probed first. A synthetic UTC origin is explicitly recorded because the clip does not provide a verified UTC date; source-relative offsets are preserved. Image timestamps represent requested sample positions and are frame-quantized by the source video. Extraction is constrained to the current window; images/audio after the window end are not included.

The result is `replay.jsonl` with ObservationWindows **without a `target` field**, and `training_eligible=false` in provenance. All participants/cameras/days from the EgoLife shared household week are one split group to avoid leakage from synchronized views. A single clip cannot fill disjoint training and evaluation sets.

### Verified local download

On 2026-09-15, the public SDK download and extraction completed without an account token:

- Dataset revision: `143fb319be7aa5ae210c936bf4f0f3a86092afb0`.
- File: `A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4`.
- Size: 6,663,507 bytes; duration: 17.63 seconds; original audio present.
- SHA-256: `8d20da0a3385f2f8d870c24c8638852dda14380cefb88ce2dd012677bbd0f2da`.
- Extracted: two causal windows, four JPEG frames total, two WAV chunks. The first extracted frame was visually inspected and shows a real indoor household scene.
- Source card declares `mit`. This records the card's declaration, not an independent rights assessment.

The sample was initially downloaded under the task's `work/public-egolife` directory. A packaged inference copy may be present under `data/public-egolife`; its manifest identifies the same pinned source. Re-run the command to reproduce it. The manifest records the exact source URL and file hash.

## Use your own bounded recording

```python
from home_observer.data import prepare_recording
prepare_recording(
    "kitchen.mp4", "work/my-recording", recording_id="kitchen-session-1",
    camera_id="kitchen", start_time=1750000000, duration_s=60,
    window_s=10, frame_step_s=2,
    provenance={"kind": "user_supplied", "group_id": "household-session-1"},
)
```

Use an accurate duration and source clock. Pass `include_audio=False` for silent video. The current public preparation command handles one camera; the common ObservationWindow format and four-camera fixture demonstrate multiple views. Synchronization of independently recorded cameras must be supplied by the capture system; no alignment is fabricated.

## Replay contract

`iter_replay(path, dataset_root=None)` accepts training rows or bare ObservationWindows and yields **only windows**, with media paths resolved. It rejects missing media, relative paths escaping the dataset root, duplicate window/evidence IDs, non-monotonic recording timestamps, windows longer than 120 seconds, and evidence outside the window. Runtime policy may impose a tighter duration limit. Targets, captions, provenance, split metadata and teacher outputs are not returned to inference.

`assert_disjoint_splits(rows)` validates both group IDs and recording IDs. Keep all frames/windows, overlapping clips, views and annotation variants belonging to the same recording group together.

## What the public annotations do not provide

- [EgoLife](https://egolife-ai.github.io/blog/) has first-person captions/transcripts and a separate third-person release through Baidu. The official Hub download above is one **egocentric** camera; it is not four fixed home cameras. Captions and retrospective QA do not establish Home Assistant service calls or silence targets.
- [EgoServe](https://huggingface.co/datasets/SitongGong/EgoServe/blob/main/README.md) publishes proactive-service annotations separately from source videos. Its event windows/dialogues can inform a future evaluation set, but they are not ready-made HA entity/service targets. It is not used as training data by this implementation.
- [SmartHome-Bench](https://huggingface.co/datasets/violetcliff/SmartHome-Bench) distributes annotation/URL CSVs; the original videos are fetched separately from public platforms, and 180 videos are private. Its current card declares `cc-by-nc-nd-4.0`. It was not downloaded or represented as available training footage here.
- [HOMAGE](https://arxiv.org/html/2105.05226v1) annotates sampled human-object relations, not continuous full-home device state. Graph absence cannot be treated as a no-op label.

No access failure occurred for the selected EgoLife sample. The other sources were inspected for suitability; this implementation does not claim to have downloaded their videos, overcome their access workflows or obtained new action supervision.

## Verification

### Expanded fixtures and image-dependent counterfactuals

The updated generator uses opaque model-visible window/evidence IDs and a deterministic randomized source-clock phase for each episode. It no longer puts the event step or split name in model-visible IDs. `data/fixtures-expanded` contains **432 examples (144 per split)** generated with six families per split and three variants. The original `data/fixtures` remains the earlier smoke-test data; use the expanded set for subsequent training.

```sh
python -m home_observer.data fixtures data/fixtures-expanded --families-per-split 6 --variants 3
python -m home_observer.data counterfactuals data/counterfactuals
```

`data/counterfactuals/pairs.jsonl` contains eight held-out pairs (16 examples). Both members of a pair have identical model-visible text, IDs, timestamps, device readings and silent audio. Only the pixels in one room's camera change from empty to occupied. These rows **must each be inferred with empty state and a fresh journal**, not sent through chronological `iter_replay`: the intentionally reused input window ID would correctly be rejected by that reader. Distinct row IDs and pair IDs are external evaluation metadata. A correct image-dependent result has no action for the empty member and the configured light-on action for the occupied member. This directly tests whether the model uses images rather than metadata shortcuts.

### Natural household-image supervision

```sh
pip install huggingface-hub pyarrow pillow
python -m home_observer.data coco-household data/coco-household --raw-dir work/coco-source
```

The prepared subset contains **64 training, 16 validation and 16 held-out test images**. It uses existing MS-COCO human captions, selecting images whose captions mention household rooms or objects such as kitchens, living rooms, sofas or refrigerators. Original captions are preserved verbatim, with their annotation IDs and image IDs. Images are resized to at most 512 px on either side; no new caption, device state, action or silence label is generated.

Metadata comes from `mlgym/coco-captioning` at pinned revision `96f5cf8784404ffd62ff541c5aeaea7b7a4d550d`; images come from the original public COCO image bucket. The COCO custom HTTPS hostname currently has a certificate-name mismatch, so the downloader accesses the **same bucket via Amazon's valid HTTPS path-style endpoint**, retaining both original and download URLs. It does not disable TLS verification. All 96 images downloaded successfully; total metadata plus image transfer was **27,928,622 bytes**, below the 250 MB enforced budget. Individual image hashes, prepared-file hashes, source metadata hashes and numeric original image-license IDs are in provenance. [COCO caption collection paper](https://arxiv.org/abs/1504.00325)

Every image is one split group. Repeated image IDs and repeated original-image byte hashes are excluded across splits. The preparer uses the source packaging's train/validation/test partitions and selects one caption per image; other original references remain in metadata for inspection. `heldout-replay.jsonl` contains the 16 test windows with targets removed. Still images receive a clearly synthetic one-second timestamp wrapper: no continuous temporal events are claimed.

Each training row has only:

```json
{"target":{"summary":"Original human caption."},"supervision_mask":{"summary":true,"actions":false,"observations":false,"noop":false}}
```

The trainer must support this mask. It supervises the summary prefix/content while masking the partial object's closing brace/end-of-turn, avoiding supervision to terminate an ordinary Decision early. **Do not fill in empty actions, empty observations or a no-op value:** a static caption does not support these labels. Runtime output still follows the full Decision contract. Caption evaluation should use held-out caption/reference review or an appropriate caption metric; action/no-op accuracy is not meaningful for these rows.

`data/coco-household/policy.json` supplies an empty device surface. If combining caption rows with the synthetic policy dataset, use a separately supplied policy and prefix media paths relative to a common dataset root. Captions, source metadata and held-out references must never appear in model inputs.

### Balanced V2 decisions and causal context

`data/mixed-v2` is a training/validation expansion of existing authored fixtures. It contains **496 training rows**: 432 synthetic decisions, evenly split between 216 action and 216 no-action windows, plus 64 unchanged human-caption rows. The 432 synthetic rows comprise 360 device-rule cases and 72 visual-occupancy cases. The validation split contains 144 synthetic decisions plus 16 caption rows from separate families/images. A no-action window may still contain a new observation; only 61 synthetic training rows have `noop=true`.

```sh
python scripts/data_mixed_v2.py --source data/mixed --output data/mixed-v2
```

This preparation reads the original training and validation files only. It does not read, relabel, or overwrite the test set. Camera images remain unchanged: telemetry, timestamps and historical context are varied only under authored cases consistent with the source geometry. Reused image bytes are recorded in each row's provenance. These variants add decision supervision, not new natural visual diversity. An intentionally balanced action distribution does not estimate the frequency of household events.

Exactly 108 synthetic training rows use each of four context conditions: cold start, matching prior state, different prior state, and stale prior state. Cold starts supply empty state and label current supported observations. Historical cases use a versioned `runtime_context` with the source group, source cutoff, Journal-shaped state and prior events. Every source fact timestamp is at or before its cutoff, and the cutoff is strictly before the current window starts. Current targets are computed afterward and omit unchanged observations. `context-sources.jsonl` and row provenance identify the authored prior source; current targets never become current input state. Caption rows receive no invented history.

`generation-validation.jsonl` fixes eight **validation-only** rows before model results: four action/four no-action cases, four device/four visual cases, and four cold starts/four historical contexts. `generation-validation-selection.json` preserves their source row IDs. Checkpoint generation reports and `reports/base-fixed8` use this selection set; they are not independent release-test results. `validation-dev.jsonl` is another bounded subset of validation, not an additional test split. The manifest records counts, source/generator hashes, family/image separation and integrity checks.

### Interpretation of execution and continuous-capture reports

- The Home Assistant reports exercise an actual isolated HA server and its REST API. Its lights/switches are fixture entities backed by `input_boolean`; successful readback does not demonstrate a physical appliance or a deployed household system. Learned-action correctness and REST transport success are reported separately.
- `reports/live-cloud-v1/transport_summary.json` uses one real EgoLife clip repeated as four camera feeds, with audio from that clip. It tests concurrent capture, upload, inference and cleanup, not four-room observation quality. Its recorded 31.85-second run processed nine windows (about 0.283 windows/second), below the requested one window/second. The footage has no action labels, so its nine no-op outputs have no measured accuracy.
- The offline dashboard reads existing reports and the canonical `reports/release-status.json`. It does not select or promote a checkpoint. No 72-hour household stability result is evidenced by these short tests.

Later live demos use four eight-second segments from the **same** included EgoLife recording, starting at source offsets 0, 3, 6 and 9 seconds. `scripts/prepare_live_demo.py` creates these inputs under the task's `work/live-distinct-inputs`, outside the deliverable's media directory. Their original audio is retained; the demo uses one segment as its microphone input. The four camera names are routing labels, not verified rooms or simultaneous cameras. Looping these segments permits a longer capture test without providing new scenes or temporal coverage.

The preparer hashes decoded frames at two positions, and live prediction records include decoded frame hashes for the sampled windows. Different hashes establish different input pixels; they do not establish that the model used those pixels correctly. Each live report's `source-provenance.json` and `validation.json`, when supplied, record the exact source scope and execution checks. Capture/transport errors, rejected decisions and independently verified HA calls must be read separately.

See the [evaluation guide](evaluation-guide.md) for source masks, known-label observation scoring, counterfactual controls, merge/batch comparisons and release completion. The [offline dashboard](../reports/index.html) links each measurement to its original report and records source-file hashes. A selection in a training experiment is a candidate for further evaluation; deployment acceptance comes only from the separate canonical status.

### Tests

`pytest tests/test_data.py tests/test_replay.py` exercises causal bounds, duplicate/time/path rejection, split leakage, absence of target leakage, reproducible fixture generation, explicit negative labels, genuine image-dependent fixture targets, and a generated MP4-to-image/audio extraction round trip. Schema/policy validation additionally checked all 96 generated windows/targets. Public download and extraction were performed separately from unit tests; tests make no network calls.

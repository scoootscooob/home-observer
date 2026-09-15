# Continuous benchmark v2: frozen protocol

This protocol was written before any model prediction on the new footage was
inspected. The split file `data/continuous-v2/split.json` carries the freeze
timestamp. Numbers (window counts, seconds, event counts) come from the
preparation manifest; nothing in this document is adjusted after scoring.

## Source and separation

- **Footage:** untrimmed official EPIC-KITCHENS-100 videos from the public
  University of Bristol server, re-encoded to 854×480 at the source frame rate
  for replay and frame sampling. License CC BY-NC 4.0 with attribution to Dima
  Damen and the EPIC-KITCHENS authors; noncommercial use only.
- **Labels:** the official dense participant narrations (verb, noun, start and
  stop time). Every action in a video was narrated by its participant, which is
  the basis for the exhaustive-coverage claim on the evaluated classes.
- **Split:** by participant, disjoint from the v1 pilot participants (P01, P02,
  P03, P04, P06, P11) and from each other. Test P28, P30, P22, P27; validation
  P26; train P09, P25, P14, P07, P23, P13. Substitutions, if any, are recorded in
  the manifest with reasons.
- **Home separation:** each EPIC participant recorded in their own kitchen; the
  extension table marks which returning participants changed kitchens. Home
  identity is therefore asserted from the dataset's one-participant-one-kitchen
  protocol, not independently verified with shared-home identifiers. Whether the
  pretrained base has seen these public videos cannot be established.

## Windows

Each video is tiled into fixed 3.0 s windows with a 1.5 s stride, independent of
action boundaries. Eight frames are sampled uniformly per window. Window and
evidence IDs are opaque hashes; timestamps are recording-relative seconds. A
narrated action becomes a target of a window when at least half its duration
lies inside; its interval is clipped to the window. Actions overlapping a window
without being targets are removed from that window's covered intervals, so a
prediction there is unscored rather than false. Windows with no narrated action
at all are marked `quiet_window`.

Causal evidence citations are derived deterministically from the annotated
interval (latest sampled frame at or before the start; latest frame inside the
interval) and are supervised only when derivable for every event in the row.
Windows with no target events are supervised as an empty event list (boolean
mask) so the trainer sees abstention. Training and validation use the
`*-supervised.jsonl` subsets: windows with targets plus truly quiet windows.
Windows whose narrated actions overlap only partially are excluded from
supervision, because an empty list there would contradict a visible partial
action; the rule and counts are in `supervised-selection.json`. The test split
is never filtered.

## Frozen metrics (from `physical_evaluate.score_predictions`)

| Metric | Definition | Requires |
|---|---|---|
| Event recall | Matched targets / targets; match = normalized exact verb and noun plus temporal IoU ≥ 0.5 on the clipped interval | targets |
| False events per hour | Unmatched predictions of an evaluated class lying entirely inside covered intervals, per covered hour, per timeline | exhaustive coverage, covered seconds |
| False completion | A `pickup_completed`/`clear_of_support` verdict of `confirmed` on a demonstration clip whose annotated outcome is not a pickup | workflow clips |
| Strict validity | Complete `PhysicalDecision` passing the causal validator | every window |
| Judged-fields validity | Judged fields accepted after explicit placeholder completion | every window |
| Abstention | Fraction of quiet windows with zero predicted events of an evaluated class | quiet windows |
| Calibration | Brier/ECE on scorable positives plus covered false events, reported with the positive-only flag | numeric confidence |
| Delay | Source end → first local availability and → verified response, only in stream replays with clock mapping | stream runs |
| Tracking consistency | Unavailable: no independent boxes/identities are annotated; tracker success flags are not ground truth | boxes, identities |
| Coverage under load | Captured frames retained, sequence/queue losses, lateness from `stream-report.json` | stream runs |

Evaluated classes: `take`, `pick-up`, `put`, `put-down`, `put-on`, `put-in`,
`put-into`, `open`, `close`. Other narrated verbs are targets for recall only
when present, and never count as false events.

## Two evaluation modes

1. **Dense offline tiling.** Every model sees identical windows, frames, prompt
   and greedy decoding on identical hardware. This measures recognition and
   localization inside a bounded window; it cannot see across window edges.
2. **Stream replay.** The production runner replays a video through the actual
   capture, tracking, activity buffer and local model. Learned events are scored
   against narrations of the whole video with the same matching rule; this adds
   detection delay and coverage measurements, but activity gating decides which
   frames the model sees, so recall is not comparable to mode 1.

## Reverse and static controls

Time-reversed and repeated-first-frame variants of test windows are diagnostics
for temporal sensitivity only. They never create negative labels and are
reported separately from the metrics above.

## What is still missing

Independent human boxes, identities and visibility for tracking metrics; human
adjudication of ambiguous outcomes; and more than a few minutes per participant.
The narrations are annotation of actions, not of object state, so outcome
verification is measured only on the explicitly chosen workflow clips.

## Tooling

| Script | Purpose |
|---|---|
| `scripts/continuous_benchmark_prepare.py` | Downloads the frozen videos from the official server, re-encodes to 480p, tiles windows, derives labels/coverage/citations, writes manifests and the demonstration candidate list |
| `scripts/physical_v2_job.sh` + `configs/runpod-physical-v2.json` | Bounded Runpod job: v2 training on the training windows, then matched native evaluation of base / v1 / v2 on the test windows |
| `scripts/stage_runpod_project.py` | Stages code, configs, window frames and the v1 adapter for the job; no videos, credentials, reports or prior artifacts |
| `scripts/compare_continuous_runs.py` | Verifies the three runs share every setting except the adapter and scores strict, judged-fields, false events per hour and quiet-window abstention |
| `scripts/score_stream_run.py` | Scores a production or research stream replay against a video's official narrations (mode 2), including detection delay from the clock mapping |
| `scripts/cut_workflow_clip.py`, `scripts/clip_contact_sheet.py` | Cut a demonstration segment with a provenance sidecar; render a frame grid with normalized coordinates for choosing an explicit seed box |

# Evaluating physical sensing

`physical_evaluate.score_predictions` scores physical events and optional supplied object annotations. It does not use HA actions, device policy or workflow success as perception labels.

```sh
.venv/bin/python scripts/evaluate_physical.py \
  --dataset data/temporal-pilot/test.jsonl \
  --predictions work/physical-predictions.jsonl \
  --output reports/physical-test.json
```

This command reads existing predictions and runs **no inference**. With `--dataset`, source labels, masks and coverage are authoritative: duplicate/unknown IDs, altered targets or altered window timestamps/evidence are rejected. Local media path remapping is allowed. Missing predictions count as missed labeled events. The report hashes its input files.

Prediction rows use `{id, result: {decision, metrics, error?}}`; top-level `decision` and `metrics` also work. A self-contained generation file can carry its source `window`, `target`, `supervision_mask` and `annotation_coverage`; omit `--dataset` in that case. Unlabeled live rows produce no ground-truth recall denominator. Rejected raw text is retained by the caller, but this evaluator does not repair it into an accepted decision. `output_usable_rate` means a decision was supplied without a reported error; strict schema/evidence validation must run upstream.

## Event matching

A match requires the same `kind`, `object_label` and temporal IoU ≥ 0.5 by default. Names are normalized for case and whitespace only. Synonyms are not silently substituted. Temporal IoU is overlap duration divided by union duration. Matching maximizes the number of matched annotations, with deterministic IoU-ordered choices; it does not promise maximal total IoU among tied assignments.

`events.labeled_event_recall` is matched supplied events divided by eligible supplied events. Partial masks must enable kind, noun and both time fields. Unsupported annotations are counted separately. Extra predictions are **unscored** unless exhaustive coverage establishes that they are false.

The EPIC pilot supplies action-trimmed clips, so the input clip boundaries reveal the annotated interval. Its temporal IoU is not proof of event localization in continuous footage. The report exposes `pretrimmed_interval_warning`.

## False events per hour

This metric requires explicit annotations for both present and absent events, covering specified classes and time intervals. An empty `target.events` array can establish a negative only with exhaustive coverage and an enabled event mask. Example:

```json
{
  "group_id": "recording-17",
  "target": {"events": []},
  "supervision_mask": {"events": true},
  "annotation_coverage": {
    "exhaustive": true,
    "event_count_exhaustive": true,
    "evaluated_event_classes": ["put-down", {"kind": "take", "object_label": "bowl"}],
    "covered_intervals": [[10.0, 40.0]]
  }
}
```

A class string means that event kind is exhaustively annotated across nouns; a kind/noun object narrows the claim. An unmatched prediction must lie entirely inside the covered interval and match the covered class before it can be counted false. Events outside that scope remain unscored. Never set exhaustive coverage solely because a dataset contains some positive labels.

The aggregate rate is null with a reason if any evaluated row lacks exhaustive class/time coverage or a stable recording/timeline ID. `exhaustive_subset` separately reports the supported portion, if any. Overlapping annotated intervals are counted once per timeline. The count refers to predicted event proposals in the supplied logs; repeated emitted proposals count separately. It is not an estimate for an unobserved hour or a deployment with different proposal-deduplication behavior.

## Confidence, uncertainty and delay

Numeric event confidence is scored only on matched positives or false events established by exhaustive annotations. Brier score and calibration bins explicitly describe this known-label subset. With positive-only annotations, the report marks the selection bias and does not claim population calibration. Free-text uncertainty has no reference rubric here and remains unscored.

Model latency is separate from detection delay. Detection delay additionally requires measured emission time and a source-to-wall-clock mapping:

```json
{
  "clock_mapping": {"source_origin": 0.0, "wall_origin": 1750000000.0, "scale": 1.0},
  "emitted_at": 1750000004.8,
  "event_emitted_at": {"event-17": 1750000004.7}
}
```

The mapping is `wall = wall_origin + (source - source_origin) * scale`; `scale` must reflect actual playback/capture timing. Per-event emission time overrides the row-wide emission time. The evaluator reports delay from annotated event start and from annotated event end, only for matched events. Without the mapping, delay is null. A network/inference duration alone cannot establish when an event occurred in the source stream.

## Object tracking

Supply explicit GT objects with `box`, `label`, `timestamp` and `frame_evidence_id`; enable those fields in the object mask. Predictions must match the exact frame and label before box IoU is considered. Annotated-box recall and mean IoU of matched boxes measure only these supplied labels. Unannotated image regions do not establish false detections.

Identity switching additionally requires supplied GT `track_id`, predicted track IDs, and a stable timeline/recording ID. The reported count is a change in predicted identity between consecutive matched annotations of the same GT track; unmatched gaps do not reset it. Missing predicted IDs are reported separately. This defined count is not IDF1 or HOTA, and those metrics remain null. No boxes or identities are inferred from EPIC noun labels.

Keep quality, annotation coverage, model latency, source time and workflow outcome separate in reports. A correctly formatted physical decision alone does not prove that a bowl was placed or that a recipe step was completed.

## Produce native predictions

The sequential runner performs pinned BF16 inference with an unmerged optional adapter and the production physical validator:

```sh
python scripts/predict_physical.py \
  --dataset data/temporal-pilot/test.jsonl \
  --dataset-root data/temporal-pilot \
  --adapter artifacts/physical-training-v1/adapter \
  --output artifacts/physical-test-v1
```

Omit `--adapter` for the base-model control. Run validation and test into separate output directories, with matching source/code/model settings. The runner preserves `inputs.jsonl`, `predictions.jsonl` and `native-outputs.jsonl`, including invalid raw generations and their measured latency. `manifest.json` records source, media, adapter, implementation and prompt hashes and stays incomplete until every row finishes. An existing output directory is never replaced. Completion means all requested rows were processed; it does not mean quality passed.

Each request selects only the physical window and empty track/event/focus context. Labels, narration, provenance and supervision masks stay outside inference. The source split is kept for scoring after generation. Generation uses one request at a time; no multi-GPU throughput claim follows from this runner.

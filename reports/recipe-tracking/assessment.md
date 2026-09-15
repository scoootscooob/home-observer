# Recipe object tracking assessment

## Result

The selected nested-bowls clip supports an early local **geometric motion cue**.
It does not support a claim that the classical tracker verifies recipe completion.
The implementation keeps semantic outcome confirmation in the learned observer.

The original white-bowl clip is retained as a failure case: raw CSRT followed the
bowl initially, then continued returning success after the target left view and
the box drifted onto a cutting board. KCF reported loss from frame66; MIL drifted
onto background earlier despite returning success. API success is not confidence.

## Source selection and actual frames

- Original: `a890f0b6cb389efafa60`, public action annotation “put down bowl”,
  78 actual frames at50fps,1.56seconds. Initial assistant-observed box
  `(60,300,295,480)` in854×480 pixels.
- Visibility-selected demo: `030c003244925eb728dd`, public action annotation
  “pick up bowls”,145 actual frames at50fps,2.90seconds. Initial
  assistant-observed box `(319,67,573,327)`. The dark nested bowls/colander stack
  stays visible in first, middle and last frames. This choice was recorded before
  candidate model outputs; no test rows or labels were changed.

The source annotation describes the action. It is not tracking ground truth.
Eight sampled boxes were visually inspected by the assistant; there are no
independent framewise boxes or calibrated identity-confidence labels.

## Measurements on this Mac

| Configuration | Source | Update p95 | Whole assessment loop | Tracking result |
|---|---|---:|---:|---|
| Raw CSRT, full resolution | Original bowl |15.42ms|71.9fps|77/77 API-success updates, but final box drift |
| Raw CSRT, half resolution | Nested bowls |10.44ms|98.7fps|144/144 API-success updates;8sampled boxes remain on visible stack |
| Guarded application CSRT, half resolution | Nested bowls |19.91ms|60.3fps|124 accepted updates; declares uncertainty at frame125 |

The exact guarded application run includes JPEG packing/decoding, background
translation estimation, appearance checks and CSRT. Seven of144 loop updates
exceeded the20ms source-frame budget. The source was decoded once continuously
without frame skipping. The backend stops updating a lost track, so aggregate
throughput includes the final uncertain interval and is not sustained tracking
throughput for an arbitrary scene or many objects.

## Conservative application behavior

`CSRTContinuousTracker` in `src/home_observer/object_tracker.py` matches the
stream's seed/update/report interface and supports `clone_single()` for replaying
a delayed detection. A track holds its CSRT backend and lost timestamp. It checks
visible box fraction, per-step and total scale, center displacement, source-frame
gap and appearance distance from the initial crop. It does not silently reseed
after failure.

The exact guarded nested-bowls run emitted:

- `motion_started` at source0.08seconds, heuristic score0.8813.
- `visibility_lost` at2.50seconds when appearance distance exceeded the preset
  conservative threshold. This means tracking uncertainty, not object absence.
- No `motion_settled` or semantic pickup event.

On the original white-bowl clip, the boundary guard rejected tracking at0.12s,
before the later raw-CSRT drift. This early uncertainty is deliberately retained.

The score is `min(seed confidence, 1 − histogram distance, visible fraction)`.
It is explicitly uncalibrated. Motion events require successful background
translation compensation, which still cannot fully model camera rotation,
parallax, occlusion or identity swaps. Completion must be established separately.

## Reproduction and evidence

The assessment environment is isolated under `work/tracking-venv`, using
OpenCV contrib4.14.0.94, NumPy2.4.6, Python3.11.15 and one OpenCV thread.
The shared project dependency was not changed by this assessment.

```bash
python scripts/assess_recipe_tracking.py \
  --clip data/temporal-pilot/clips/030c003244925eb728dd.mp4 \
  --box 319 67 573 327 --trackers CSRTGuarded --scale 0.5 \
  --output reports/recipe-tracking-guarded-new-run
```

Evidence directories: `reports/recipe-tracking` (original failure and visibility
selection), `reports/recipe-tracking-nested-bowls-half` (raw speed/boxes), and
`reports/recipe-tracking-guarded-final` (exact application tracker). Each contains
timings, per-frame outputs, sampled images and a separate manual-review record.
Six focused tracker tests plus22watch tests pass; Ruff passes for the new files.

The implementation follows the [official OpenCV CSRT API](https://docs.opencv.org/4.12.0/d2/da2/classcv_1_1TrackerCSRT.html).

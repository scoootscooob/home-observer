# Physical temporal pilot

This separate pilot contains **96 real EPIC-KITCHENS clips**: 64 training, 16 validation and 16 test. It supervises observed kitchen interactions only. The existing Home Assistant baseline and datasets remain unchanged.

## What is actually available

The public [Lightly EPIC-KITCHENS clip mirror](https://huggingface.co/datasets/lightly-ai/epic-kitchens-100-clips/tree/e1739df818cbfb7e528fb1ef5005dbb51e96aacd) was downloaded without credentials using `huggingface_hub`. Its clips are trimmed to annotated actions and compressed to 854 × 480. We sampled eight chronological JPEG frames from each clip, retaining the original MP4. No text overlays or synthetic images were added. Audio remains in the source MP4 but is excluded from this visual-interaction pilot.

The annotation CSVs match the [official EPIC-KITCHENS repository](https://github.com/epic-kitchens/epic-kitchens-100-annotations/tree/ea8b40457a400c3fffa1c7f406ef3dc169cc2522) byte for byte. The source supplies participant narration, parsed verb/noun and action start/stop times. Repository revisions, annotation hashes, original rows, clip hashes, frame indices and frame hashes are recorded in the pilot.

The selected source has no access blocker. Downloaded assets and metadata total **48,168,694 bytes**; prepared outputs occupy about **94 MB**. The clips contain **145.76 seconds** of source video. These are measured preparation totals, not training measurements. No full dataset archive was downloaded.

The source-declared license is [CC BY-NC 4.0](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/license.txt), with attribution and noncommercial conditions. This agrees with the saved [license text](../data/temporal-pilot/sources/license.txt), [mirror dataset card](../data/temporal-pilot/sources/README.md), manifest and all 96 row provenance records. Credit: Dima Damen and the EPIC-KITCHENS authors; Lightly produced the compressed action clips. Retain the attribution, license link and change notices with redistributed pilot media: the mirror trimmed, resized and compressed the source; this pilot selected clips and extracted JPEG frames. Public download access does not remove the source's noncommercial condition.

## Split and leakage controls

| Split | Participants | Clips | Recording groups |
| --- | --- | ---: | ---: |
| Train | P01, P02, P04, P06 | 64 | 32 |
| Validation | P11 | 16 | 6 |
| Test | P03 | 16 | 9 |

Each participant contributes four clips per broad interaction family: pickup/take, placement, opening and closing. Exact source verbs are retained in the targets. Selection uses a fixed hash order within each participant/family, with two documented bowl clips selected for the demonstration. No model predictions affected selection.

These are **custom participant-disjoint and recording-disjoint splits**, drawn entirely from the official training annotations. They are not the official EPIC benchmark split. No participant, recording, MP4 hash or sampled-frame hash crosses our split boundary. All 96 windows passed `PhysicalWindow` validation; all 768 sampled images have distinct hashes.

Do not describe this as a held-out-home benchmark. The official [participant extension table](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/Extension_Participants.csv) marks returning/changing kitchens, but does not supply the stable shared-home identifiers needed to establish that claim. This split also cannot establish that the pretrained base model has never seen these public videos.

The saved [manifest](../data/temporal-pilot/manifest.json) and [validation report](../data/temporal-pilot/validation-report.json) describe input-data integrity, not successful model predictions. Every row has `original_split: "train"` and `home_id: null`. The P03 test examples have now been inspected for evaluation and the recipe demonstration. They remain disjoint from the v1 training rows, but must not be presented as a fresh, unseen acceptance set for an iteration designed around those findings. Preserve these rows and original scores; reserve new participants/recordings for the next final evaluation before inspecting model results.

## Labels and masks

Input `window` follows `PhysicalWindow`: a timestamped clip containing ordered frame evidence, plus an empty audio list. The time basis is **seconds within the original recording**, not an invented wall clock. Filenames, window IDs and evidence IDs are opaque hashes. Source participant IDs, narration, labels and original annotation metadata live outside the model input.

The partial target is:

```json
{
  "summary": "put down bowl",
  "events": [{
    "kind": "put-down",
    "object_label": "bowl",
    "started_at": 37.62,
    "ended_at": 39.18,
    "description": "put down bowl"
  }]
}
```

The matching `supervision_mask` enables summary and those five event fields. It does **not** supervise missing detections, boxes, tracks, subjects, confidence, uncertainty, pre/post citations or observations as empty arrays or negative labels. No HA entity, policy, action or `noop` target is present. The physical trainer must understand partial field masks; do not feed these rows to the original full-Decision trainer.

Source intervals remain exact. Video frame timestamps use `action_start + decoded_frame_index / fps`; the final frame can precede the source stop time by a frame period. That does not justify inventing an image at the interval endpoint.

## What the pilot can and cannot measure

It can test whether a model recognizes the annotated interaction and noun from chronological real imagery. It can score recall of those supplied labels. Each row includes explicit coverage metadata.

Every selected video was already trimmed to an annotated action. Its boundaries therefore reveal the supplied action interval. Temporal IoU on these clips is a compatibility check, **not evidence of event localization in an untrimmed stream**. An event outside the supplied annotations is unscored, not automatically a false event. There are no exhaustively annotated negative intervals. False events per hour, continuous detection delay, calibration against complete event truth, dense object tracking accuracy and long occlusion recovery are unavailable from these labels alone.

Additional real continuous footage with exhaustive class/time coverage, object boxes, identities and visibility labels is required for those measurements. The existing short unlabeled EgoLife demo does not supply them. Original full EPIC videos can be obtained through the [official downloader](https://github.com/epic-kitchens/epic-kitchens-download-scripts); this pilot uses individually downloadable clips to keep the initial transfer bounded.

## Next collection and annotation requirements

The 100-step trial reduced supervised-field validation loss from 2.165 to 0.073, but the matched guided evaluation accepted **0/16 validation decisions and 1/16 test decisions**. Of the 32 outputs, 28 were rejected for future-evidence citations, one for a duplicate observation ID, one for an out-of-window assertion and one for truncated JSON. These are complete-decision results; they do not mean every rejected event description was wrong. The [recorded result summary](../artifacts/physical-adapted-controller/result-summary.json) preserves that distinction. The one truncated output contained a complete event array, but its citations also failed causal validation.

This exposes a gap between the source's five annotated event fields and the richer runtime contract. More narration-only clips will not supply the missing evidence, identity or uncertainty labels. The next corpus should include:

| Requirement | What to collect or annotate | What it enables |
| --- | --- | --- |
| Continuous context | Untrimmed recordings with approach, interaction, aftermath and quiet intervals; source frame times, camera identity, recording identity and dropped-frame/gap metadata. Sample windows independently of the action boundaries. Keep gaps explicitly unknown. | Event localization and evaluation on an actual stream. Preserve source-to-wall-clock mapping and measured event emission times if reporting detection delay; model latency alone is a different measure. |
| Causal before/after evidence | For each visible event, annotate its kind, noun, interval and distinct supporting frame IDs. Under the current contract, pre evidence must end at or before event start; post evidence must follow the pre evidence and end at or before event end. If the available frames do not support both sides, mark the event/evidence unresolved instead of inventing citations. | Direct supervision for the causal-citation failure seen here. An interval or narration is not itself a frame-evidence label. |
| Explicit event coverage and negatives | Declare which classes are exhaustively annotated over which uninterrupted intervals. Include confirmed non-events, near misses, hand motion without object movement, repeated handling, camera motion and occlusion. Have annotators review the whole covered interval, including events that were easy to miss. | Recall and false events per hour for the declared classes and time coverage. Unannotated gaps and unlisted classes remain unscored. |
| Independent boxes and identities | Annotate raw video frames with normalized object boxes, camera/frame references, persistent identity where supported, visibility and entry/exit/occlusion intervals. Use independent review; tracker outputs and model boxes are proposals, not ground truth. Reappearance does not prove identity across an unseen gap. | Box overlap, identity-switch and occlusion-recovery measurements against real annotations. |
| Uncertainty and ambiguous outcomes | Mark visible, partially occluded and unresolved cases, including uncertain identity, event boundaries and final object state. Retain annotator disagreement and adjudication. Use independently scored correct/incorrect cases to evaluate model confidence; do not manufacture confidence probabilities from narration or treat missing fields as certain negatives. | Honest abstention and calibration on cases whose correctness is actually scorable. |
| Recipe-step outcome | Separately annotate the requested step and observable completion condition, such as a bowl visibly resting on a specified surface. Label an outcome unknown when contact or the final state is occluded. Keep instruction, event and outcome annotations distinct. | A measured instruction-to-observed-interaction workflow without assuming the narrated action proves completion. |

Assign participant and recording splits **before** window extraction and augmentation. All overlapping windows, alternate camera views, transformed media and repeated appearances from a recording stay in one split. If a future collection has verified shared-kitchen/home identifiers, group by those as well; until then, report only the verified participant/recording separation. Keep labels and annotation provenance outside model inputs, supervise only adjudicated fields, and record annotator/source identity and revision. New physical sensing examples must contain no Home Assistant action targets.

Use the new training split to develop complete-label behavior and validation to choose thresholds or formats. Freeze the next test split and annotation protocol before the final comparison. Temporal reversal and repeated-frame probes are sensitivity controls; they do not create ground-truth reverse actions or negative labels. Keep their results separate from the original event-recall scores.

## One recipe step

`recipe-demo.jsonl` selects **one continuous 1.56-second test clip**, `P03_106_11`, covering source time 37.62–39.18 seconds. Its existing label is “put down bowl.” The matching `recipe-demo.json` carries an authored instruction, “Put the bowl down on the work surface,” clearly separate from source annotation and model input. It is a demonstration instruction, not an original recipe or a proven workflow-completion label.

The contact sheet visually confirms the bowl being handled near the work surface. The bowl moves partly outside the frame as the camera turns. The source provides no exact bowl box, persistent identity, visibility interval or final surface relation, so those remain unannotated. The earlier pickup clip is also in the test split, but is **not stitched** into this replay and does not establish identity across its unobserved gap.

## Files and reproduction

- `data/temporal-pilot/{train,validation,test}.jsonl`: partial supervised rows.
- `data/temporal-pilot/manifest.json`: pinned sources, measured sizes, split counts and limits.
- `data/temporal-pilot/validation-report.json`: schema and split checks.
- `data/temporal-pilot/recipe-demo.jsonl`, `recipe-demo.json`, `recipe-contact-sheet.jpg`: one-step demonstration.
- Each row's `provenance.clip` identifies its original local MP4; `window` paths identify sampled input frames.

From the project directory:

```sh
.venv/bin/python scripts/temporal_data_prepare.py \
  --output data/temporal-pilot-new \
  --cache work/temporal-source-probe \
  --max-download-mb 200
```

The preparer refuses to overwrite an existing output. It downloads individual pinned Hub files with `token=False`, verifies annotation and clip SHA-256 values, checks decoded duration against each source interval, and records completion only after preparation succeeds. The SDK cache permits retries after network failure without redownloading completed assets. Dataset training and evaluation are separate subsequent steps.

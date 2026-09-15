# How to read the evaluation evidence

Open the [offline dashboard](../reports/index.html) for comparisons and prediction inspection. It is a snapshot of local files; rebuild it after copying new reports:

```sh
python scripts/build_report.py
```

## Keep the evaluation sets separate

| Set | What it tests | What it does not establish |
| --- | --- | --- |
| Fixed eight validation rows | Four action/four no-action cases, balanced between device and visual tasks; used to choose checkpoints | Independent test performance or household event frequency |
| Held-out 160 rows | 144 authored synthetic decisions plus 16 natural photographs with human captions | Real household control accuracy |
| Counterfactual 16 rows | Eight authored empty/occupied image pairs with identical other inputs | Generalization to natural camera views |
| Public video two windows | Real video/audio input and chronological journal flow | Perception, action or silence accuracy; no targets exist |
| Isolated Home Assistant demo | Model proposals, allowed REST actions and independent state readback | Operation of physical household appliances |

Synthetic action labels come from explicit authored policies. Natural-image rows have `supervision_mask.actions=false`, `observations=false` and `noop=false`: only their existing human captions are supervised. Missing labels are never evidence that nothing happened. See [data provenance](data.md).

## Three different scores

**Action correctness** compares the exact `(domain, service, entity_id, data)` values with labeled targets. A different argument payload is a different proposed action. Invalid responses and policy-rejected windows do not earn true-positive credit; expected actions remain missed. False-action-window rate counts proposed actions in explicitly labeled no-action windows, including rejected proposals. This measures proposal behavior, not the number of physical actions executed. Undefined precision when no actions were proposed is shown as a dash.

**Known-label observation scores** evaluate only annotated `(entity_id, attribute)` pairs. Each pair is correct, incorrect or missing. Values must match their JSON type: `true`, `"true"` and `1` are different. Coverage is `(correct + incorrect) / annotated`; accuracy when covered is `correct / (correct + incorrect)`. Read both: high accuracy on a handful of reported facts can coexist with many missing facts. `correct / annotated` includes missing facts in its denominator.

Predictions outside those annotated pairs are **unscored extras**, not proven false perceptions. This matters when the current policy asks for facts that older source labels did not exhaustively annotate. The helper can score parsed fact values even if another part of the same response was rejected; its separate original-failure count remains authoritative for whole-response validity. A high fact-value score never clears an invalid evidence citation or an execution rejection.

**Legacy strict-set observation scores** compare full sets of `(entity_id, attribute, value)`. They treat extra unannotated pairs as mismatches and suppress true-positive credit for failed windows. They are retained for compatibility with original reports, but should not be interpreted as the percentage of perceived facts that are false. Original targets and raw predictions are unchanged by the supplementary known-label calculation.

To generate the supplementary report from existing predictions:

```sh
python scripts/model_annotation_metrics.py --predictions reports/EXPERIMENT/predictions.jsonl
python scripts/model_annotation_metrics.py --release-root artifacts/RELEASE
```

## Causal context and image dependence

The fixed validation set mixes cold starts with authored historical context. Historical facts are strictly earlier than the current window and have traceable source groups. Their targets omit unchanged observations. An observation target is a change to record, not necessarily every fact visible now.

The original held-out decision labels describe independent snapshots. Their release evaluation uses empty prior observation history and a fresh journal for each labeled row. Replaying those snapshot labels as if they were chronological change labels would change the question and can penalize correct omission of repeated facts.

Each counterfactual pair has identical model-visible IDs, timestamps, device readings and silent audio. One room's pixels change from empty to occupied. Each member must run in a separate empty journal because its input window ID is intentionally reused. Compare the two decisions together: expected behavior is no light action for the empty member and the policy-specified light action for the occupied member. Per-window scores alone do not establish that both members of a pair succeeded. Drawn geometry remains synthetic evidence.

## Match the model and runtime before comparing

Record the model revision, adapter hash/path, policy and prompt hashes, media bounds, schema contract, context mode and batch size. Matching row-file hashes confirms the selected rows, not identical prompts or runtime settings.

An **unmerged adapter** remains a separate LoRA computation attached to the base model. A **merged adapter** folds its weights into the base model. Their outputs must be checked on the same requests. The recovered checkpoint 300 fixed-eight validation and continuation checkpoint 50 training-callback validation record `adapter_merged=false`. A separate merged checkpoint 300 release run is a different condition. Read each release manifest and actual generation metrics rather than inferring merge mode from the checkpoint name.

The initial fresh-loaded single-request run differed from the callback on two visual cases. The cause was isolated to **policy JSON field order**: values were unchanged, but text token IDs differed while media and mask tensors matched. Restoring the training field order reproduced all eight callback outputs byte for byte in the [single-request control](../artifacts/continuation-policy-order/report.json). Canonical policy serialization now makes training/runtime contexts and encoded tensors match for all eight checked examples; see the [verification](../artifacts/policy-order-diagnostic/canonicalization-verification.json). Earlier reload, merge and batch comparisons used different text inputs and therefore do not isolate merging or batching as the cause. This resolved control does not establish full release performance.

The continuation was initialized from checkpoint 300 adapter weights with a reset optimizer; its local steps 50 and 100 are labeled **continuation steps**. They are not a claim of uninterrupted training from the original run. The continuation's saved checkpoints include separate optimizer, scheduler and random-state artifacts for resumability.

Batched release evaluation is also a distinct runtime condition. A same-request single-versus-batch control on the same loaded model is needed before attributing changed decisions to batching or training. Higher batch throughput does not imply lower event latency; an event waits for its entire batch. Read partial and rejected experiment reports alongside the selected run.

## Completion, execution and acceptance

- A full release needs all three required suites, matching manifest source/selected counts, report counts and locally readable prediction counts. A checkpoint file, eight-row validation report or partial JSONL does not complete a release. A completed evaluation can still fail quality checks.
- `artifacts/training-v2-continuation/selection.json` records a validation candidate. Only [canonical release status](../reports/release-status.json) can state deployment acceptance. Neither the dashboard nor a good validation score promotes a model.
- HA reports use an actual isolated server with fixture entities backed by `input_boolean`. A service proposal, an accepted request and independently verified readback are separate evidence. The initial vLLM live runner reported transport success despite rejected decisions; its `validation.json` records this limitation, and later versions preserve their own reports.
- Live source clips are four time offsets from one public recording, not four simultaneous home cameras. Count captured, sampled and omitted frames separately. Capture timestamps describe the local replay session, not verified wall-clock timing of the original household.
- Zero transport errors does not mean zero rejected decisions. Throughput is not quality, and a short demo is not a one-Hz guarantee or a 72-hour stability test. Expired media references remain in the audit log; recent retained snapshots and their hashes support bounded inspection.

The dashboard embeds at most 250 predictions per source file and a bounded number of thumbnails. Full original JSONL files, source hashes and reported scope remain linked for inspection.

## Combining independent release workers

After both sequential-inference shards finish, combine them without rerunning inference:

```sh
python scripts/model_combine_release_shards.py \
  --shards artifacts/release-continuation-selected-shard0 artifacts/release-continuation-selected-shard1 \
  --output artifacts/release-continuation-selected
python scripts/model_annotation_metrics.py --release-root artifacts/release-continuation-selected
python scripts/build_report.py
```

The combiner requires a fresh output directory and complete, disjoint shard indices. It checks configuration/adapter/implementation hashes, source partitions, native-output counts and order, and actual merge flags. It also checks the original local dataset hashes, targets, masks and model-visible windows; `--source-root` changes the default `data` directory. The entire public chronology belongs to shard 0.

Original input, prediction and raw-generation JSONL lines are retained in source order, including invalid outputs and their original latencies. Source shard files stay unchanged. Reports pool correctness counts but retain per-worker runtime measurements; without a shared wall-clock measurement, combined throughput and single-GPU throughput remain unavailable. A complete combined report establishes dataset coverage, not model acceptance.

# Mixed v2 training data

496 training rows: 432 authored synthetic decisions (216 with actions, 216 without) plus 64 existing human COCO captions. The synthetic rows include 72 visual-only examples with no current motion telemetry. Validation contains 160 rows from separate source families/images; `validation-dev.jsonl` contains 56 of those for bounded training-time checks. `generation-validation.jsonl` is a fixed 8-row validation-only generation check: four action/four no-action rows, both device and visual tasks, four cold starts/four historical contexts. The adjacent selection JSON records exact unchanged validation row IDs; no model result or test row was used to choose them.

Run from the project root:

```sh
.venv/bin/python scripts/data_mixed_v2.py
```

Use `--dataset-root data`, `--policy data/mixed-v2/policy.json`, and train/validation JSONL paths under this directory. The script reads **only** v1 train and validation files. It does not read or change test sets.

All four camera images are reused byte-for-byte from their recorded procedural source scenes. Occupied, empty, bright and already-lit geometry stays consistent with the authored telemetry. These are extra decision examples, not new natural visuals. Audio is same-family generated tone/silence selected independently of the action class. No audio alarm or speech labels are introduced.

`context-sources.jsonl` stores the authored earlier windows used for context. Rows identify their exact parent source IDs and input-file hashes. `runtime_context` is versioned and contains Journal-shaped history with timestamps strictly earlier than the current window. Targets are computed only after history is authored. Each source recording has exactly six cold-start, six unchanged-history, six changed-history and six stale-history examples. Cold-start rows have no runtime context and label all supported current facts. Other targets omit unchanged facts. Current evidence controls action preconditions.

The policy defines binary sensor equivalence (on/true/JSON true and off/false/JSON false), wrapped Home Assistant states, numeric lux strings, the strict below-40 threshold, light/buzzer already-on cases and changed-fact recording. The 50/50 action balance is a training choice, not a household prevalence claim. Natural captions keep summary-only masks and no runtime context.

`manifest.json` records counts, integrity checks and limitations. No v2 evaluation result is implied by preparation.

# What the public repository omits, and why

The repository keeps code, configuration, documentation, synthetic fixtures,
dataset manifests with hashes, labels and provenance, and the JSON evidence of
every run. It omits the items below. Nothing omitted is required to run the
tests; a clean checkout passes with the data-dependent tests skipping.

| Omitted | License or reason | Kept instead | Regenerate |
|---|---|---|---|
| `data/temporal-pilot/{clips,media}`, `recipe-contact-sheet.jpg` | EPIC-KITCHENS-100, CC BY-NC 4.0 | `manifest.json`, `*.jsonl`, `validation-report.json` | `scripts/temporal_data_prepare.py` |
| `data/continuous-v2/{videos,media}` | same | manifest, split, validation report, `supervised-selection.json`, candidates | `scripts/continuous_benchmark_prepare.py --annotations <official CSV dir>` |
| `data/continuous-v2-workflow/*.mp4`, `*.jpg` | same | the `*.json` sidecars with source hashes and offsets | `scripts/cut_workflow_clip.py`, `scripts/clip_contact_sheet.py` |
| `data/coco-household/assets`, `data/public-egolife/{raw,assets}` | per-photo Creative Commons terms; identifiable people | manifests and captions | `home_observer.data` preparers |
| `reports/**` frames, event evidence, bootstrap frames, chronology images | EPIC-derived media | every JSON and Markdown record, sequence manifests, `stream-scores.json` | produced by runs |
| `baselines/*/checkpoints`, `baselines/*/*.tar.gz`, `artifacts/` | Gemma Terms of Use, size | `baseline-final.json`, `checkpoint-manifest.json` with SHA-256 | Runpod jobs (`docs/runpod.md`) |
| `*.sqlite*` journals, `*-private/` directories, sandbox profiles, `work/` | private local state, machine-specific paths, keys | export stores are summarized in `privacy-evidence.json` | produced by runs |

## Residual machine-specific strings

Historical run reports and manifests are preserved byte-identical, because their
SHA-256 values are recorded by other evidence files and the project's rule is never
to alter a historical run. They therefore still contain the producing machine's
absolute paths (a user name and a directory layout) and terminated Runpod pod
identifiers. None of these are secrets or credentials; a later release may
re-record them with a hash-aware scrub if presentation matters.

## Noncommercial labels

The retained `*.jsonl` rows, annotation copies under `data/*/sources` and the
stream scores quote EPIC-KITCHENS narrations. That part of the public tree is
CC BY-NC 4.0, not MIT; commercial users must regenerate labels from a source they
are licensed to use.

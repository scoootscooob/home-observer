# Contributing

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev,data,sensing]'
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

Tests that need excluded data, a GPU, macOS `sandbox-exec` or Home Assistant skip
themselves when the prerequisite is missing; a clean checkout must still pass.

## What is opt-in

- Datasets are regenerated from pinned upstream sources; see `docs/excluded-media.md`.
- GPU training and evaluation run on Runpod through `scripts/runpod.py`; keys and
  provider state stay under the gitignored `work/` directory.
- The production-shaped runner (`scripts/run_production.py`) needs Apple silicon and
  an Anthropic-Messages-compatible endpoint for the sandboxed frontier planner.

## Rules that reviews enforce

- Nothing that leaves the local sensing host may contain media, paths, raw model
  output or free text: every export goes through `production_export.py` and its
  scanner, and `tests/test_production_export.py` replays known leaks.
- A geometric event is not a semantic pickup, a device readback is not a physical
  outcome, and a delivered export is not an acknowledgement. Keep those facts
  separate in code, reports and docs.
- Historical run reports are evidence: never rewrite their scores.

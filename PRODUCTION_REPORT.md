# Production privacy boundary and local perception — iteration 2 (DRAFT, pending sections marked)

**The production privacy boundary is now enforced and demonstrated. Model quality remains weak.**
Every sensory byte stayed on this machine in the production-shaped runs: perception ran
in-process on Apple Metal, the frontier received only typed exports, and the frontier
process ran in a sandbox that cannot read media or reach any other host. The recipe
step remains unverified; the frontier correctly left it pending.

This report separates four things: tested privacy enforcement, measured model
capability, unresolved uncertainty, and planned work. Preserved iteration-1 results
in [PHYSICAL_REPORT.md](PHYSICAL_REPORT.md) are unchanged; their baseline manifest
(`ee233b87…`) and v1 adapter weights (`e46d4192…`) were hash-verified again on 2026-09-15.

## 1. Tested privacy enforcement

| Requirement | Enforcement | Test evidence |
|---|---|---|
| Media never leaves the host | `PhysicalLocalModel` runs the pinned Gemma 4 E4B weights and the unmerged v1 adapter in-process; `assert_production_model` rejects remote/vLLM backends and any object with an HTTP client attribute; `ProductionRuntimeConfig` refuses configs naming a URL, endpoint or token file | `tests/test_production_export.py` |
| Frontier receives selected structured fields only | `ExportEnvelope` (schema `home-observer.production-export/1`) with four typed message kinds, vocabulary-mapped kinds/categories, opaque IDs, rounded times, bounded uncertainty bands | same |
| Hash after sanitization, persist, retry, acknowledge | `export_sha256` covers the sanitized content; exports are leased and retried with backoff; `ack_export` binds export ID + hash and maps to the private delivery through the audit table | same |
| No arbitrary passthrough | Leak scanner rejects paths, URLs, base64, extensions, free text, denied keys, non-finite numbers; unknown topics are blocked and recorded | same |
| Experimental bridge unusable in production | `assert_production_coordinator` rejects `FileCoordinator`/`HTTPCoordinator`; the production bridge writes exports, scanned outcomes and rejection classes only | `tests/test_production_coordinator.py` |
| Host isolation of the frontier | `scripts/frontier_sandbox.py` deny-by-default `sandbox-exec` profile: read/write the bridge, read its own interpreter, TCP to one local port | `tests/test_frontier_sandbox.py` (denied read, listing, write, port; allowed bridge and port) |
| Recorded leaks cannot recur | The eight envelopes recorded in iteration 1 (absolute frame paths, manifest path, raw model output, adapter path, free text) replay through the boundary with zero leaks | `tests/test_production_export.py` |
| End to end | Synthetic production run: private outbox still contains paths, bridge contains none, notify watch verified only after export ACK, verifier result exported without its audit | `tests/test_run_production.py` |

Software verification after this iteration: **296 passed, 5 skipped, Ruff clean** (up from 272/5).

## 2. Measured local capability (pilot test rows, Apple M5 Max, 48 GB)

Same 16 pilot test rows, same prompt, eight frames, greedy decoding, BF16, unmerged adapter.

| Backend / contract | Usable decisions | Labeled event recall | Mean latency |
|---|---:|---:|---:|
| v1 adapter, strict contract | 0/16 | 0.00 | 18.1 s |
| v1 adapter, judged-fields contract | 16/16 | 0.25 | 18.1 s |
| Base, strict, 768-token budget | 0/16 (all truncated) | 0.00 | 37.6 s |
| Base, strict, 1536-token budget | 0/16 (9 truncated, 7 missing required object fields) | 0.00 | 92.3 s (shared GPU) |
| Base, judged-fields, 1536-token budget | 8/16 (no events reported) | 0.00 | 92.3 s (shared GPU) |

The strict failures of the adapter are format failures: it emits only the five supervised
event fields plus copied evidence citations. The judged-fields contract accepts exactly
the fields the adapter judges, derives causal citations from the claimed interval, and
completes `summary`, `confidence` and `uncertainty` with explicit recorded placeholders.
Recall 0.25 is 4 of 16 exact verb+noun matches with temporal IoU at least 0.5 on
pretrimmed clips; it is a content measurement, not evidence of localization.
Without a decoding grammar the base model enumerates scene objects verbosely, runs past
its token budget, and omits required per-object confidence fields, so it is unusable on
this backend even though it produced 15/16 valid decisions under the vLLM grammar on
Runpod in iteration 1. Local constrained decoding is future work; the production
backend therefore depends on the adapter's terse output under the judged contract.
Evidence: `artifacts/local-backend-benchmark/`.

## 3. Production-shaped recipe replays (same public clip as iteration 1)

Run `production-recipe-replay-v2` (first complete run):

- 145/145 frames captured, no losses; capture lateness mean 2.3 ms, max 36 ms.
- Geometric `motion_started` on the seeded bowl; the isolated light verified on with
  source-event-end to verified response **0.138 s** (availability to response 0.071 s).
- 12 exports delivered and all 12 acknowledged by export ID and hash; 0 blocked;
  41 frontier-visible files, 0 scanner violations; the private outbox still holds paths.
- The sandboxed Opus 5 planner (5 turns) created the watch, acknowledged every export,
  asked `pickup_completed` and `clear_of_support` twice, received `unknown`, and left the
  step pending: `recipe_step_complete=false`, `pickup_verified=unknown`.
- Defects found and fixed after this run: the verifier projected a question asked after
  the last frame past the source (now clamped to the newest evidence with staleness
  recorded), and an echoed journal event ID failed the judged parser (now ignored and counted).

Runs v3 to v5 repeated the replay under stress, all with 145/145 frames, no losses, zero
scanner violations and every export acknowledged:

| Run | Condition | Source end → verified cue | Exports acked | Planner turns | Verifier verdicts | Outcome |
|---|---|---:|---:|---:|---|---|
| v3 | verifier clamped to newest evidence; GPU shared with a benchmark | 0.951 s | 11/11 | 8 | 3 × `unknown` (`localization_uncertain`, 125 states examined) | pending, unknown |
| v4 | 20 s artificial model delay | 0.153 s | 9/9 | 5 | 1 × `unknown` (`localization_uncertain`) | pending, unknown |
| v5 | sandboxed frontier killed 25 s in and relaunched 3 s later | 0.075 s | 11/11 | 6 (1 restart) | 3 × `unknown` (`localization_uncertain`) | pending, unknown |

The `unknown` verdicts are correct for this clip: the CSRT track became uncertain at 2.5 s
when the cookware's appearance changed, and the verifier never converts missing
localization into presence or absence. The v3 cue latency shows the local loop is
sensitive to host contention (three model processes shared the GPU); the deadline chosen
by the planner was 1 s and was met. In v5 the relaunched coordinator resumed from its
state file and acknowledged every export it had missed. Stream scoring against the
video's official narrations (`stream-scores.json`) shows one labeled pickup in the replay
span, zero matches (the full-window decision failed validation in every run: an echoed
event ID in v3, trailing characters in v5; both are now tolerated as format noise with the
ignored content counted) and zero false events under coverage. Run v6 with the parser fix: both perception windows validated under the judged
contract (11 exports, all acknowledged, zero violations, cue verified 0.434 s after the
source event under GPU contention). The full-window decision reported `pick-up bowl`
over 608.99 to 611.87 s, which matches the official narration `P03_109_178` exactly, so the
stream score for this replay is 1 of 1 labeled events matched with zero false events; the
interval copies the window bounds, so this is a content match, not localization evidence.
The event carried no subject ID, the tracker had lost the bowl, and the verifier answered
`unknown` (`localization_uncertain`) three times; the planner left the step pending with
`pickup_verified=unknown`. Learned-event availability lagged the source by 193 s because
the first model call shared the GPU with a benchmark; alone, the same call takes 15 to 32 s.

## 4. Production-shaped workflows on fresh footage

Clips were cut from the frozen test videos by inspecting footage and official narrations
only (`data/continuous-v2-workflow/*.json` sidecars record the source hash, offsets and
overlapping narrations). Positive: `take pan` (P27_03, 56.54 to 57.99 s) and `take lid`
(P27_03, 47.68 to 49.90 s); negative: `move plate` (P22_105, the plate is slid twice, never
lifted). Each run used the local Metal backend, the exports-only bridge, the sandboxed
Opus 5 planner, the isolated light and an explicit seed box.

| Run | Frames | Exports acked | Leaks | Cue (source end → verified) | First geometric motion vs. narrated action | Tracker | Verifier | Planner outcome |
|---|---:|---:|---:|---:|---|---|---|---|
| positive pan | 510/510 | 20/20 | 0 | 0.128 s | 52.53 s vs. 56.54 s | lost at 55.84 s (implausible step) | 5 × unknown | pending, unknown |
| positive lid | 509/509 | 18/18 | 0 | 0.078 s | 44.92 s vs. 47.68 s | lost at 46.29 s (appearance) | 3 × unknown | pending, unknown |
| negative plate | 600/600 | 22/22 | 0 | 0.089 s | 55.24 s vs. 57.95 s | lost at 66.34 s (camera pans away) | 1 × unknown | pending, unknown |

What these runs establish and what they do not:

- **Privacy held in every run**: every export acknowledged by hash, zero blocked, zero
  scanner violations, sandboxed planner throughout (exact totals below).
- **No false completion**: the negative run never became complete even though the local
  model claimed `put-down plate` in five consecutive windows and, in an earlier run of the
  same clip, `pick-up plate`. Those claims were exported as learned events with placeholder
  confidence and could not satisfy the planner's completion rule, which requires a
  confirmed verifier verdict.
- **No positive completion either**: the CSRT tracker lost the pan and the lid before the
  narrated pickups under egocentric camera motion, so the verifier answered
  `localization_uncertain` and the learned check, which is gated behind geometric
  confirmation, never ran. The positive workflows therefore end pending. That is the
  honest state: this prototype cannot yet verify a pickup on head-mounted footage.
- **The geometric cue fired early in all runs**, within half a second of seeding and before
  any narrated action. A first cause was a tracker re-centering step right after seeding;
  `motion_started` now requires three consecutive moving frames (regression test added, the
  pre-fix runs are preserved under `*-pre-motion-fix`). The remaining early cue is residual
  camera motion that the translation-only compensation cannot remove; a higher-threshold
  sensitivity run is reported below. On a static household camera this source is absent,
  which is why the earlier bowl replay's 174 ms cue is a pipeline latency, not evidence
  that the cue was caused by the object's motion.
- **Stream scores** (`stream-scores.json`): exact-verb recall 0 on all three clips (the
  model says `pick-up` where the narration says `take`; the family-level diagnostic finds
  one near match on the lid clip); the negative clip records 5 false events over 12 s of
  covered time, so the model's false-event rate on handling-without-pickup footage is high.

Sensitivity runs with `--motion-threshold 0.01 --motion-confirm-frames 5` (runner flags added
for this purpose, recorded in `runtime-ready.json`):

- **Negative plate**: the only geometric `motion_started` fired at 63.76 s, half a second before
  the second narrated `move plate` (64.27 to 65.25 s), and the first slide at 57.95 s produced
  no cue. The cue was verified 0.269 s after the source event. The model reported `move plate`
  in all six windows, the correct verb this time, so the run has 0 false events under
  coverage and its extras are unscored. The verifier still returned `unknown` and the step
  stayed pending: correct outcome, this time for a plausible cue.
- **Positive pan**: motion was detected at 55.74 s with tracker confidence 0.37, below the
  planner's 0.5 threshold, and the track was lost 0.1 s later; no cue fired and the watch stayed
  armed. The same appearance change that starts a pickup is what breaks the tracker.

Across all 8 fresh-footage runs (three before the motion fix, three after, two
sensitivity runs): 4348 frames captured, 155 exports of which 155 acknowledged,
0 blocked, 0 scanner violations across 449 frontier-visible files,
0 completions, 0 verified pickups.

## 5. Fresh continuous benchmark and controlled experiments

Protocol frozen before any prediction: [docs/continuous-benchmark.md](docs/continuous-benchmark.md).
Split frozen 2026-09-15 20:14:36 UTC (P28_10 substituted by P28_07 because of missing
narration timestamps; the preparation was interrupted by a session restart and re-run with
the same deterministic assignment, both freeze records preserved).

| Split | Participants | Videos | Windows | Quiet windows | Target events | Seconds |
|---|---|---:|---:|---:|---:|---:|
| Test | P22, P27, P28, P30 | 4 | 242 | 33 | 146 | 369 |
| Validation | P26 | 1 | 42 | 3 | 20 | 65 |
| Train | P07, P09, P13, P14, P23, P25 | 6 | 380 | 48 | 168 | 579 |

3.8 GB of official 1080p video downloaded and hash-recorded, re-encoded to 480p; 5312 frames,
5301 distinct hashes, participant-, recording-, video- and frame-disjoint across splits and
from the pilot. Home separation rests on the dataset's one-participant-one-kitchen protocol
and the extension notes, not on verified shared-home identifiers. Training supervision uses
windows with targets plus truly quiet windows (195 train, 22 validation rows); windows whose
actions overlap only partially are excluded from supervision so an empty event list never
contradicts a visible partial action (`supervised-selection.json`).

Runpod research job: the first launch failed at payload extraction (the project archive
carried local uid/gid and the pod's `tar` could not chown); the runner stopped that pod with
its volume retained, nothing had run, and the pod was deleted and provider-confirmed
(`reports/physical-v2-failed-upload-cleanup.json`, about $0.10). The packager now writes
root-owned entries and the extractor uses `--no-same-owner`. The relaunched job on a fresh
H100 pod: PENDING (v2 training for 300 steps on the supervised subset, then matched native
evaluation of v2, v1 and base on the 242 test windows).

## 6. Unresolved uncertainty and limits

- The sandbox is a local `sandbox-exec` profile, not a network or hardware guarantee.
- Structured exports still reveal household activity by design.
- Tracker identity, false events per hour, calibration and delay on fresh footage are
  not yet measured; the pilot test rows are diagnostic only.
- The judged-fields contract makes the adapter usable but its confidence is a placeholder;
  watches requiring a model judgment above 0.5 cannot be satisfied by it.

## 7. Planned work

PENDING: filled after the Runpod results and the workflow runs.

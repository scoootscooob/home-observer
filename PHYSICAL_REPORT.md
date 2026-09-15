# Continuous physical sensing — iteration 1

**The sensing architecture, Runpod training/inference, and recipe observation workflow were implemented and exercised. The model remains experimental.** The local cue was verified in **174 ms**, but the frontier coordinator could not verify the requested pickup, so the recipe step remains incomplete.

This is a real public-video replay with an actual isolated Home Assistant server and an actual frontier subagent. It is not a deployment in the user's home. The coordinator runs within this Codex session; no permanent cloud coordinator or camera service remains running.

## What changed

| Requested change | Implemented behavior |
|---|---|
| Preserve the baseline | Frozen source/reports, all 84 checkpoint files, and 803 evaluation files; hashes reverified after the physical iteration. |
| Separate perception from actions | `PhysicalDecision` contains objects, observations, events, evidence and uncertainty. It has no Home Assistant action field. Frontier-created watches select only locally approved responses. |
| Ordinary physical objects | Dedicated `phys_*` registry and tracking history coexist with the existing SQLite journal. Device permissions apply separately to execution. |
| Continuous coverage | Independent capture, tracking and inference workers; full buffered sequences; explicit bounded queues, overload losses, delayed-detection replay and uncertain tracking states. |
| Event-driven coordination | Durable outbox, exact delivery-ID/hash acknowledgements, local time-sensitive watches, and the existing once-only executor with state readback. |
| Temporal training and evaluation | Fresh physical LoRA on labeled real kitchen clips; separate content, validity, event, tracking, uncertainty and timing measurements. |

Architecture and runnable instructions: [continuous sensing](docs/continuous-sensing.md), [physical model](docs/physical-model.md), [recipe workflow](docs/recipe-workflow.md), [watch protocol](docs/watches.md).

## Training and matched evaluation

The pilot contains **96 real EPIC-KITCHENS clips / 768 frames**: 64 training, 16 validation, and 16 test. Participants, recordings, clips and frame hashes are disjoint across splits. These are **not verified held-out homes**. Clips are short and pretrimmed, with human interaction labels but incomplete scene/event annotations. The dataset's recorded license is CC-BY-NC 4.0; see [data provenance and annotation scope](docs/temporal-data.md).

A fresh language LoRA was trained for **100 steps on a Runpod H100** from the pinned Gemma 4 E4B base. Validation loss fell from **2.165 to 0.073**. All four checkpoints include optimizer, scheduler, RNG and trainer state and were recovered with hash verification. Low loss did not translate into a reliable complete sensing contract.

The base and final adapter were evaluated on the same Runpod RTX 3090, vLLM 0.29.0, BF16, eight-frame input budget, prompt, grammar and request sequence. All 32 paired model-visible inputs and decoding configurations were verified identical except the adapter and explicit served alias.

| Measurement | Base | Physical adapter |
|---|---:|---:|
| Strict-valid validation decisions | 15/16 | 0/16 |
| Matched validation events | 0/16 | 0/16 |
| Strict-valid test decisions | 15/16 | 1/16 |
| Matched test events | 0/16 | 1/16 — 6.25% |
| Raw test interaction matches, including rejected output | 0/16 | 6/16 — 37.5% |
| Mean test model-call latency | 5.16 s | 12.65 s |

Primary event matching requires an exact normalized verb+noun match and temporal intersection-over-union of at least 0.5. **The raw-content row is a separate diagnostic, not an accepted event score.** It preserves the observation gain even when complete output validation fails. An assistant audit also identifies vocabulary near-matches without changing the primary metric or claiming independent human review.

The adapter's 31 rejected decisions comprise **28 future-evidence errors, one duplicate ID, one out-of-window assertion and one truncated JSON output**. The full raw outputs remain available. All 16 adapted test intervals copy the supplied clip bounds; their timing does not demonstrate learned event-boundary localization.

Evidence: [paired comparison](artifacts/physical-guided-comparison.json), [content audit](artifacts/physical-semantic-audit/assistant-review.md), [adapted evaluation summary](artifacts/physical-adapted-controller/result-summary.json), [training recovery](artifacts/physical-training-v1/transfer-verification.json).

### Temporal controls and unavailable metrics

Four additional probes reversed the pixels or repeated the first frame for two sequences, preserving chronological metadata and decoder configuration. **Both static sequences still produced pickup claims.** Reversal did not produce a coherent opposite interaction. This shows some sensitivity to image content, but does not establish reliable temporal understanding. These derived controls are not real continuous negative footage. See [probe results](artifacts/physical-temporal-probe/assistant-analysis.md).

- **False events per hour:** unavailable without exhaustive class/time coverage and continuous negative intervals. Unannotated extra predictions are not automatically false events.
- **Tracking accuracy and identity switches:** unavailable without independent frame boxes and track identities. A tracker success flag is not ground truth.
- **Uncertainty calibration:** the test provides only one accepted matched positive and no exhaustively covered negatives; the reported positive-only statistics are not population calibration. Textual uncertainty has no reference rubric.
- **Detection delay on the test set:** unavailable without explicit source-to-wall mapping. Model-call latency is measured separately. The replay below does have clock mapping.

Definitions and supported annotation inputs are in [physical evaluation](docs/physical-evaluation.md). The inspected pilot test set is now a diagnostic set; a later acceptance run needs fresh footage.

## Recorded recipe workflow

The digital commitment asked for an immediate cue when nested bowls began moving, followed by verification of a pickup from a worktop. A frontier subagent created the watch before runtime. An explicitly recorded assistant-selected initial box bootstrapped local tracking; it was not a learned detection.

The 2.9-second continuous clip produced:

- **145/145 captured frames**, all retained in the full sequence, with no tracking-queue, event-queue or sequence losses.
- Two Runpod requests: the initial eight frames and eight selected frames from the full sequence. This is **16 image presentations / 15 unique frames**, not model analysis of all 145 frames.
- A local geometric motion event at source-relative **0.08 s**. The approved isolated light changed **off → on**, with verified readback **173.785 ms after the trigger's source end**, within its one-second deadline.
- Mean capture lateness **2.69 ms**, maximum **35.58 ms**. These measurements cover one short replay, not an indefinite camera deployment.
- One physical track with **125 persisted tracking updates**. Appearance checks declared localization uncertain at **2.50 s**; this was not treated as physical disappearance.
- One strict-valid learned pickup assertion from the initial **0.14-second prefix**, available **12.124 s** after that prefix ended. The full-window decision failed future-evidence validation. Neither established recipe completion.
- Eight real coordinator delivery acknowledgements, accepted with exact IDs and payload hashes. ACKs arrived after the runtime's final drain and were processed against the same journal in a separately recorded post-run receive step.

### Physical outcome: unknown

The actual frontier subagent inspected nine chronological frames, including three at full size, and verified all 145 image hashes. It saw nested cookware supported on a wooden board in an **open lower cupboard**, with hands repositioning/reaching around it. It could not establish clear separation from the support or a carried state. This also contradicted the digital instruction's **worktop** location.

**`pickup_verified=false`; `recipe_step_complete=false`.** The observed workflow reached response, delivery and outcome review; it did not demonstrate successful completion of the physical recipe step. A later unobscured view of the cookware clear of its support is needed to confirm pickup. No new watch was armed after camera EOF.

The coordinator's visual review completed about **208 s after the clip ended**, including interactive agent scheduling and inspection. That is separate from the **174 ms local response**, and is not a production cloud-coordinator latency claim.

Evidence: [frontier outcome and inspected frames](reports/recipe-workflow-trained-v1/coordinator/physical-outcome.md), [stream report](reports/recipe-workflow-trained-v1/stream-report.json), [ACK receipt](reports/recipe-workflow-trained-v1/coordinator/post-run-ack-receipt.json), [original digital plan](reports/recipe-workflow-trained-v1/coordinator/planning.md).

## Verification, preservation and current state

**272 tests passed, 5 skipped; Ruff passed.** Meaningful checks include capture during slow inference and tracking, bounded queues, causal evidence, immutable journal replays, delayed seed reconciliation, exact delivery ACKs and once-only execution. A review-found EOF hang when overload dropped a detection's anchor frame was fixed and covered by a regression test. [Software verification](reports/physical-software-verification.json).

The original sensing-plus-action baseline is preserved at [baseline manifest](baselines/2026-09-15-home-observer/baseline-final.json). The new adapter and all resumable checkpoints are under `artifacts/physical-training-v1/adapter`. Historical frozen source, live frontend source, GPU controllers, logs and evaluation evidence are preserved in `artifacts/physical-runtime` and the associated physical artifact directories.

All task training/evaluation/inference pods were **terminated and provider-confirmed only after their artifacts were recovered and verified**: [training H100](reports/physical-training-h100-cleanup.json), [additional H100](reports/physical-additional-h100-cleanup.json), [consumer inference](reports/physical-consumer-cleanup.json). The Runpod endpoint is no longer running. Reproduction commands are documented; credentials remain outside the deliverables.

The next training priority is continuous, exhaustively annotated real footage with causal pre/post evidence, independently labeled identity/visibility, occlusion and negative intervals, plus verified home-level separation. The current architecture supports those measurements, but this pilot does not supply those labels or establish production reliability.

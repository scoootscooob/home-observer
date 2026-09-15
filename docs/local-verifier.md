# Local outcome verifier

`LocalVerifier` answers a typed question about an opaque physical track using
local evidence only: the physical journal, the tracking history and, optionally,
a focused local re-observation of retained frames. Its answer is
`confirmed`, `contradicted` or `unknown` with a confidence, a bounded
`reason_code`, the evidence span and the number of frames examined. Frame
references, boxes and displacement numbers stay in the private audit record.

## Questions and rules

| Question | Confirmed when | Contradicted when | Otherwise |
|---|---|---|---|
| `moved_from_initial_location` | Latest tracking state is visible, at least `min_visible_states` visible states exist in the lookback, and the box center moved at least `min_center_displacement` (0.15 normalized) with IoU against the initial box at most 0.3. | Visible, settled, displacement at most 0.05 and IoU at least 0.6. | `unknown`: `localization_uncertain`, `insufficient_tracking_history` or `ambiguous_displacement`. |
| `still_present` | A visible state within `stale_after_s` of the question time. | Never from missing updates: unseen is not gone. | `unknown`: `not_observed_in_window`. |
| `clear_of_support`, `pickup_completed` | Geometric movement confirmed **and** agreeing learned evidence: a journaled positive learned event (`pick-up`, `take`, `lift`, ...) for the subject, or a focused local learned check returning true. | Geometry contradicted, or the latest learned event is a placement, or the learned check returns false. | `unknown`: `learned_model_unavailable`, `requires_learned_evidence`, `learned_output_invalid`, or the geometric reason. |

Two-dimensional displacement alone never confirms a lift. A geometric
`motion_started` event, a verified device cue, and a verified recipe step are
three different facts and stay separate.

## Learned check

The production runner supplies an optional `learned_check(subject_id, question,
source_as_of, lookback_s)` built over the retained sequence spool. It selects up
to the model's frame budget from retained frames inside the lookback, asks the
local model for a single `clear_of_support` observation for that subject (and
accepts a visible positive pick-up event for the subject), and returns
`{"value": true|false|null, "confidence", "audit"}`. The request and result
are logged privately in `verifications-private.jsonl`; only the value reaches
the verifier and only the verdict reaches the export. The verifier never
fetches media itself.

## Asynchrony

`ProductionCoordinator(async_verification=True)` answers a `verify` command
with `{"verification_id", "status": "pending"}`, runs the verifier in a
worker thread, and emits `verification.result` into the private outbox on a
later tick. Capture, tracking and fast local watches never wait for it.

## Thresholds and what they are not

`VerifierPolicy` thresholds are local settings; the frontier cannot change them.
The confidence is derived from the tracker's uncalibrated heuristic score and
the learned event confidence; it is a bounded score, not a probability. The
verifier is tested in `tests/test_local_verifier.py` against synthetic tracking
histories; its behavior on real footage is measured in the production runs.

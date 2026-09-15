# Continuous physical watches and a frontier coordinator

The coordinator reads digital context and creates a typed local watch. A local
loop evaluates physical events even when the cloud coordinator or the learned
perception model is busy. The coordinator bridge carries messages; it does not
simulate planning or call another model. In the integrated demo, a real Codex
frontier agent writes the watch command and acknowledges events through files.

## Local approval and identity boundaries

`WatchSpec` accepts event or clock predicates, a lifetime, a response deadline,
and an optional physical outcome. It cannot contain Python, shell commands, or
device actions. Its `response_id` selects an exact response that was approved in
the **local** registry before the watch was created:

```python
from home_observer.coordinator import FileCoordinator
from home_observer.watches import ApprovedResponse, WatchEngine, WatchStore

store = WatchStore("work/physical-watch.sqlite")
engine = WatchEngine(store, [ApprovedResponse(
    response_id="recipe-ready",
    kind="notify",
    message="The observed recipe step is ready to check.",
    approval_ref="user-approved recipe demo",
)])
bridge = FileCoordinator(engine, "work/recipe-coordinator")
```

For an approved device response, use `kind="ha_service"`, an exact
`ha_entity_id`, `domain`, `service`, and optional supported `data`. Pass the
existing `HomeAssistant` executor to `WatchEngine`. Supported commands are
`light`/`switch` `turn_on`/`turn_off`; brightness is optional only for an approved
`light.turn_on`. `approval_ref` records the source of the local approval. A watch
stores an immutable snapshot of its response; changing registry contents cannot
silently change an existing watch.

Physical identities such as `phys_cup_1` describe tracked objects. Home Assistant
identities such as `light.recipe_alert` describe configured devices. Neither can
be used in the other's field. A physical perception decision contains no device
actions. Labels alone are allowed when identity is unknown, but provide weaker
matching than an identified track.

## Actual coordinator exchange

The local process creates `inbox/`, `outbox/`, `results/`, and `rejected/` under
the bridge directory. The frontier agent reads the digital recipe and available
physical evidence, then writes a JSON command to `inbox/`. Use Unix seconds from
the active capture clock for `created_at`; the example times below are schematic
and must be replaced at runtime.

```json
{
  "command_id": "recipe-agent-turn-1",
  "coordinator_id": "actual-frontier-agent-task-id",
  "type": "create_watch",
  "watch": {
    "watch_id": "recipe-cup-settled",
    "context": {
      "commitment_id": "recipe-step-2",
      "text": "When the tracked cup stops moving, alert me to check the next step.",
      "coordinator_id": "actual-frontier-agent-task-id",
      "source_ref": "recipe-context.json#step-2"
    },
    "condition": {
      "type": "event",
      "kind": "motion_settled",
      "object_label": "cup",
      "subject_ids": ["phys_cup_1"],
      "origin": "geometric",
      "min_confidence": 0.8,
      "min_duration_s": 0
    },
    "response_id": "recipe-ready",
    "created_at": 1789490000.0,
    "ttl_s": 180,
    "response_deadline_s": 3,
    "max_event_age_s": 5,
    "max_responses": 1
  }
}
```

`bridge.tick()` accepts bounded JSON packets, calls `engine.advance()`, and
publishes due deliveries. Accepted commands produce `results/<command_id>.json`.
Malformed packets are quarantined with an error; they cannot block later valid
commands. Reusing a command ID with different contents is rejected. Retrying the
same command returns its saved result. Use an atomic file replacement when
writing commands; `coordinator.atomic_json(path, value)` is provided.

Every outbox envelope includes `delivery_id`, `topic`, `payload`,
`payload_sha256`, `created_at`, and a delivery attempt number. An actual
coordinator that has consumed it writes:

```json
{
  "command_id": "recipe-agent-ack-1",
  "coordinator_id": "actual-frontier-agent-task-id",
  "type": "ack_event",
  "delivery_id": "delivery_REPLACE_WITH_32_HEX_CHARACTERS",
  "payload_sha256": "REPLACE_WITH_EXACT_64_HEX_PAYLOAD_HASH"
}
```

The hash and delivery ID must be copied from the real envelope. Successful
transport alone does not acknowledge the event. Files are published under stable
delivery IDs and retried with exponential backoff until acknowledged. Expired
delivery leases are recoverable after restart. A coordinator must deduplicate
by delivery ID before making a new planning decision. Optional `HTTPCoordinator`
uses an explicitly configured client/endpoint and the same contract; an HTTP
success is transport delivery, and a separate ACK is still required.

To cancel an armed watch, send `type="cancel_watch"` and `watch_id`. Cancellation
does not retract a response already reserved or sent.

## Local observation loop

Feed `engine.consume(event)` the flattened events from
`PhysicalJournal.recent_events(...)`, plus their trusted local `origin`
(`"learned"` or `"geometric"`) and `available_at`. Event IDs must be durable
`pevt_*` IDs, subjects must already be resolved `phys_*` IDs, and evidence must
refer to captured source media. `available_at` is when the result first became
available locally; it is distinct from `ended_at`, the source interval's end.
Omitting `available_at` records the first ingestion time. Replays preserve the
original arrival timestamp and cannot rewrite an event's contents.

Run `bridge.tick()` regularly in a local loop independently of cloud inference.
Each process/thread must create its own SQLite connection. Do not execute
blocking model or cloud requests in this loop. The existing HA executor performs
bounded command/readback work; it can occupy the response loop while completing
that one response. Keep frame capture and event production in their independent
workers. Clock deadlines can fire through `engine.advance()` with no new media.

Geometric kinds include `entered_zone`, `motion_started`, `motion_settled`, and
`visibility_lost`. A geometric `motion_settled` event is **not** proof that an
object was placed, food cooked, or a recipe step completed. Use the precise
geometric criterion in the watch and request an independent learned outcome
when semantic confirmation is needed.

`min_duration_s` accumulates matching **observed source intervals** for the same
track(s), separated by at most `max_gap_s`. Wall-clock waiting and missing frames
do not add observed duration. `reset_kinds` can clear that duration when the same
subject produces an explicit contrary event. A positive duration requires
interval-bearing events; repeated point events do not claim continuous evidence.
Stale or future evidence cannot trigger a response. Late learned events are not
discarded merely because an unrelated geometric event has a newer timestamp.

## Response and outcome guarantees

A unique response reservation is committed before any device command. Duplicate
events, repeated coordinator commands, and process restarts do not repeat it.
The executor sends the command once and polls actual HA state for verification.
Readback failure or timeout is `uncertain`; it never causes a command retry. A
crash after reservation may leave a `reserved` watch and `pending` execution,
which must be treated as unresolved and is not automatically retried. This
trades possible missed responses for avoiding duplicate uncertain side effects.

The source-relative response deadline is checked on match and again immediately
before a device send. Verification finishing late records
`response_timely=false`. A notification is queued durably and becomes a verified
**coordinator delivery** only after its matching ACK. That does not prove a human
noticed it. A late ACK remains recorded with `response_timely=false`.

For a separate physical outcome, add:

```json
{
  "outcome": {
    "condition": {
      "type": "event",
      "kind": "placed",
      "object_label": "cup",
      "subject_ids": ["phys_cup_1"],
      "origin": "learned",
      "min_confidence": 0.9
    },
    "within_s": 20
  }
}
```

This is an example schema, not a claim that the model recognizes `placed`
reliably. Only a matching, evidenced event starting after the response can
satisfy it. Missing evidence becomes `outcome_unknown` after the outcome window
or watch expiry. A verified physical outcome does not erase an uncertain HA
execution record; inspect the execution and outcome separately.

## Measured timeline

`store.get(watch_id)` returns the immutable plan, status, execution, progress,
`clocks`, and `latencies_s`. Clocks separately record source start/end, first
local availability, match, physical-event transport delivery/ACK, response
reservation/send/verification, and physical-outcome source/availability/
verification/delivery/ACK. Unavailable clocks remain `null`; the system does not
substitute source time for cloud arrival time.

Latency fields include source→availability, availability→match,
source→event-delivery, source→response, source→verified-response, and
response→verified-physical-outcome. Local response may precede cloud event
delivery by design. Device state readback, coordinator delivery ACK, and physical
outcome verification are labeled separately.

## Validation

```bash
.venv/bin/pytest -q tests/test_watches.py
```

Tests cover reservation/restart behavior, uncertain commands, async learned
arrival, duration/TTL/deadlines, domain separation, hash-bound acknowledgements,
delivery leases and retries, immutable plans, invalid inbox quarantine, and
independent timing fields. These tests exercise the transport and watch logic;
the separate integrated run supplies the actual frontier-agent exchange and
captured-media evidence.

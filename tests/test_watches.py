import json

import pytest
from pydantic import ValidationError

from home_observer.coordinator import CoordinatorCommand, FileCoordinator
from home_observer.ha import SimulatedHomeAssistant
from home_observer.watches import ApprovedResponse, EventCondition, WatchEngine, WatchSpec, WatchStore


class Clock:
    def __init__(self, now=100):
        self.now = now

    def __call__(self):
        return self.now


def plan(kind="ha_service"):
    data = {
        "response_id": "recipe-alert",
        "kind": kind,
        "message": "Recipe step is ready",
        "approval_ref": "local-demo-configuration",
    }
    if kind == "ha_service":
        data.update(ha_entity_id="light.recipe_alert", domain="light", service="turn_on")
    return ApprovedResponse.model_validate(data)


def spec(**changes):
    value = {
        "watch_id": "recipe-watch",
        "created_at": 100,
        "ttl_s": 100,
        "context": {
            "commitment_id": "recipe-step-1",
            "text": "Watch the cup settle, then notify me",
            "coordinator_id": "actual-frontier-task",
        },
        "condition": {
            "type": "event",
            "kind": "motion_settled",
            "object_label": "cup",
            "origin": "geometric",
        },
        "response_id": "recipe-alert",
    }
    value.update(changes)
    return value


def event(number=1, *, start=101, end=101, kind="motion_settled", origin="geometric", **changes):
    value = {
        "event_id": "pevt_" + str(number),
        "source_event_id": str(number),
        "window_id": "w" + str(number),
        "kind": kind,
        "object_label": "cup",
        "subject_ids": ["phys_cup1"],
        "started_at": start,
        "ended_at": end,
        "confidence": 0.95,
        "origin": origin,
        "pre_evidence_ids": ["frame-before-" + str(number)],
        "post_evidence_ids": ["frame-after-" + str(number)],
    }
    value.update(changes)
    return value


def engine(tmp_path, *, response=None, executor=None):
    clock = Clock()
    store = WatchStore(tmp_path / "journal.sqlite")
    executor = executor or SimulatedHomeAssistant()
    runtime = WatchEngine(store, [response or plan()], executor, clock=clock)
    return runtime, clock, executor


def test_local_geometric_event_responds_without_cloud_and_readback_is_verified(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(spec())
    clock.now = 102
    runtime.consume(event(), now=102)
    result = runtime.store.get("recipe-watch")
    assert result["status"] == "verified"
    assert result["execution"]["status"] == "verified"
    assert ha.states["light.recipe_alert"] == "on"
    assert result["progress"]["event_available_at"] == 102
    assert result["progress"]["response_sent_at"] == 102
    assert result["progress"]["response_timely"] is True
    assert result["response"]["ha_entity_id"] != "phys_cup1"
    assert (
        runtime.store.db.execute("SELECT COUNT(*) FROM watch_outbox WHERE acked_at IS NULL").fetchone()[0]
        >= 3
    )


def test_duplicate_event_and_process_restart_never_repeat_response(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(spec())
    clock.now = 102
    runtime.consume(event())
    runtime.consume(event())
    runtime.store.close()
    runtime = WatchEngine(WatchStore(tmp_path / "journal.sqlite"), [plan()], ha, clock=clock)
    runtime.consume(event())
    runtime.consume(event(2, start=102, end=102))
    assert len(ha.calls) == 1
    assert runtime.store.get("recipe-watch")["status"] == "verified"


def test_timeout_is_uncertain_and_not_retryable(tmp_path):
    class TimeoutExecutor:
        calls = 0

        def execute(self, action, response_key):
            self.calls += 1
            raise TimeoutError("response might already have happened")

    executor = TimeoutExecutor()
    runtime, clock, _ = engine(tmp_path, executor=executor)
    runtime.create(spec())
    clock.now = 102
    runtime.consume(event())
    assert runtime.store.get("recipe-watch")["status"] == "uncertain"
    runtime.store.close()
    runtime = WatchEngine(WatchStore(tmp_path / "journal.sqlite"), [plan()], executor, clock=clock)
    runtime.consume(event(2, start=102, end=102))
    assert executor.calls == 1


def test_crash_after_reservation_is_visible_and_never_repeated(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(spec())
    clock.now = 102
    runtime._respond = lambda *args: None
    runtime.consume(event())
    assert runtime.store.get("recipe-watch")["execution"]["status"] == "pending"
    runtime.store.close()
    runtime = WatchEngine(WatchStore(tmp_path / "journal.sqlite"), [plan()], ha, clock=clock)
    runtime.consume(event())
    runtime.consume(event(2, start=102, end=102))
    assert not ha.calls


def test_response_deadline_rechecked_before_each_device_mutation(tmp_path):
    clock = Clock()

    class SlowExecutor(SimulatedHomeAssistant):
        def execute(self, action, response_key):
            result = super().execute(action, response_key)
            clock.now += 20
            return result

    ha = SlowExecutor()
    runtime = WatchEngine(WatchStore(tmp_path / "db"), [plan()], ha, clock=clock)
    runtime.create(spec(watch_id="first", response_deadline_s=5))
    runtime.create(spec(watch_id="second", response_deadline_s=5))
    clock.now = 102
    runtime.consume(event())
    assert len(ha.calls) == 1
    assert runtime.store.get("second")["status"] == "missed_deadline"
    assert runtime.store.get("second")["execution"]["status"] == "not_sent_deadline"


def test_duration_requires_matching_source_intervals_not_wall_clock_or_missing_events(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(
        spec(
            condition={
                "type": "event",
                "kind": "motion_settled",
                "object_label": "cup",
                "min_duration_s": 3,
                "max_gap_s": 0.1,
            }
        )
    )
    clock.now = 102
    runtime.consume(event(start=101, end=102))
    clock.now = 104
    runtime.advance()
    assert not ha.calls
    runtime.consume(event(2, start=103.5, end=104))
    assert not ha.calls
    clock.now = 107
    runtime.consume(event(3, start=104, end=107))
    assert len(ha.calls) == 1
    assert runtime.store.get("recipe-watch")["progress"]["candidate_start"] == 103.5


def test_min_duration_cannot_combine_distinct_tracks(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(spec(condition={"type": "event", "kind": "motion_settled", "min_duration_s": 3}))
    clock.now = 103
    runtime.consume(event(start=101, end=103))
    clock.now = 105
    runtime.consume(event(2, start=103, end=105, subject_ids=["phys_other"]))
    assert not ha.calls


def test_point_events_and_tolerated_gaps_do_not_add_observed_duration(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(
        spec(condition={"type": "event", "kind": "motion_settled", "min_duration_s": 1, "max_gap_s": 0.5})
    )
    for number in range(6):
        clock.now = 101 + number * 0.25
        runtime.consume(event(number, start=clock.now, end=clock.now))
    assert not ha.calls
    assert runtime.store.get("recipe-watch")["progress"]["candidate_observed_s"] == 0
    clock.now = 103
    runtime.consume(event(10, start=102.5, end=103))
    clock.now = 103.75
    runtime.consume(event(11, start=103.5, end=103.75))
    assert not ha.calls
    clock.now = 104
    runtime.consume(event(12, start=103.75, end=104))
    assert len(ha.calls) == 1
    assert runtime.store.get("recipe-watch")["progress"]["candidate_observed_s"] == 1


def test_stale_future_and_wrong_origin_events_do_not_trigger(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(
        spec(max_event_age_s=2, condition={"type": "event", "kind": "placed", "origin": "learned"})
    )
    clock.now = 105
    runtime.consume(event(kind="placed", origin="geometric", start=104, end=105))
    runtime.consume(event(2, kind="placed", origin="learned", start=101, end=101))
    with pytest.raises(ValueError, match="future"):
        runtime.consume(event(3, kind="placed", origin="learned", start=106, end=106))
    assert not ha.calls


def test_deadline_trigger_and_ttl_progress_without_cloud_or_sensing(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(spec(condition={"type": "deadline", "at": 104}))
    clock.now = 104
    runtime.advance()
    assert len(ha.calls) == 1
    runtime.create(spec(watch_id="expires", created_at=104, ttl_s=2))
    clock.now = 107
    runtime.advance()
    assert runtime.store.get("expires")["status"] == "expired"
    assert len(ha.calls) == 1


def test_physical_outcome_separate_from_verified_device_response(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(
        spec(
            outcome={
                "condition": {"type": "event", "kind": "placed", "object_label": "cup", "origin": "learned"},
                "within_s": 10,
            }
        )
    )
    clock.now = 102
    runtime.consume(event())
    assert runtime.store.get("recipe-watch")["status"] == "awaiting_outcome"
    clock.now = 103
    runtime.consume(event(2, kind="placed", origin="geometric", start=103, end=103))
    assert runtime.store.get("recipe-watch")["status"] == "awaiting_outcome"
    clock.now = 104
    runtime.consume(event(3, kind="placed", origin="learned", start=104, end=104))
    result = runtime.store.get("recipe-watch")
    assert result["status"] == "verified"
    assert result["progress"]["outcome_event_id"] == "pevt_3"
    assert result["progress"]["outcome_verified_at"] == 104
    assert len(ha.calls) == 1


def test_outcome_after_ttl_or_before_response_is_not_verification(tmp_path):
    runtime, clock, _ = engine(tmp_path)
    runtime.create(
        spec(ttl_s=5, outcome={"condition": {"kind": "placed", "origin": "learned"}, "within_s": 20})
    )
    clock.now = 102
    runtime.consume(event())
    clock.now = 103
    runtime.consume(event(2, kind="placed", origin="learned", start=101, end=103))
    assert runtime.store.get("recipe-watch")["status"] == "awaiting_outcome"
    clock.now = 106
    runtime.consume(event(3, kind="placed", origin="learned", start=106, end=106))
    assert runtime.store.get("recipe-watch")["status"] == "outcome_unknown"


def test_notification_outbox_and_ack_are_durable_without_repeating_actions(tmp_path):
    runtime, clock, ha = engine(tmp_path, response=plan("notify"))
    runtime.create(spec())
    clock.now = 102
    runtime.consume(event())
    assert runtime.store.get("recipe-watch")["status"] == "awaiting_delivery"
    assert not ha.calls
    bridge = FileCoordinator(runtime, tmp_path / "bridge")
    published = bridge.publish(now=102)
    alert = next(row for row in published if row["topic"] == "watch.triggered")
    assert runtime.acknowledge(alert["delivery_id"], alert["payload_sha256"], now=103)
    assert not runtime.acknowledge(alert["delivery_id"], alert["payload_sha256"], now=104)
    result = runtime.store.get("recipe-watch")
    assert result["status"] == "verified"
    assert result["progress"]["verification_kind"] == "coordinator_delivery_ack"
    assert result["progress"]["response_sent_at"] == 102
    assert result["progress"]["response_verified_at"] == 103


def test_outbox_retry_lease_and_ack_bind_exact_payload(tmp_path):
    runtime, _, _ = engine(tmp_path, response=plan("notify"))
    runtime.create(spec())
    one = runtime.store.claim_events(now=100, lease_s=5)
    assert len(one) == 1
    assert not runtime.store.claim_events(now=101)
    two = runtime.store.claim_events(now=106)
    assert two[0]["delivery_id"] == one[0]["delivery_id"]
    assert two[0]["attempt"] == 2
    assert not runtime.store.delivery_result(
        one[0]["delivery_id"], one[0]["lease_token"], now=106, delivered=True
    )
    assert runtime.store.delivery_result(
        two[0]["delivery_id"], two[0]["lease_token"], now=106, delivered=True
    )
    with pytest.raises(ValueError, match="exact delivered payload"):
        runtime.acknowledge(two[0]["delivery_id"], "0" * 64, now=107)
    runtime.acknowledge(two[0]["delivery_id"], two[0]["payload_sha256"], now=107)
    assert not runtime.store.claim_events(now=1000)


def test_real_agent_file_contract_creates_watch_and_idempotently_accepts_retry(tmp_path):
    runtime, clock, _ = engine(tmp_path)
    bridge = FileCoordinator(runtime, tmp_path / "bridge")
    command = {
        "command_id": "agent-turn-1",
        "coordinator_id": "actual-frontier-task",
        "type": "create_watch",
        "watch": spec(),
    }
    (tmp_path / "bridge/inbox/agent-turn-1.json").write_text(json.dumps(command))
    results = bridge.tick(now=100)
    assert results["commands"][0]["result"]["status"] == "armed"
    assert not (tmp_path / "bridge/inbox/agent-turn-1.json").exists()
    assert bridge.accept(command, now=101) == results["commands"][0]
    command["watch"]["ttl_s"] = 200
    with pytest.raises(ValueError, match="different content"):
        bridge.accept(command, now=102)
    assert runtime.store.db.execute("SELECT COUNT(*) FROM local_watches").fetchone()[0] == 1


def test_cloud_plan_cannot_add_device_targets_or_execute_code(tmp_path):
    runtime, _, _ = engine(tmp_path)
    with pytest.raises(ValidationError):
        WatchSpec.model_validate(spec(python="import os"))
    with pytest.raises(ValidationError):
        EventCondition(kind="motion_settled", subject_ids=["light.kitchen"])
    with pytest.raises(ValidationError):
        ApprovedResponse(
            response_id="bad",
            kind="ha_service",
            domain="light",
            service="turn_on",
            ha_entity_id="phys_pot",
            approval_ref="local",
        )
    with pytest.raises(ValueError, match="local approved registry"):
        runtime.create(spec(response_id="not-approved"))
    with pytest.raises(ValidationError):
        CoordinatorCommand.model_validate(
            {"command_id": "evil", "coordinator_id": "x", "type": "execute", "code": "print(1)"}
        )


def test_delayed_learned_outcome_is_not_lost_to_newer_unrelated_geometric_events(tmp_path):
    runtime, clock, _ = engine(tmp_path)
    runtime.create(spec(outcome={"condition": {"kind": "placed", "origin": "learned"}, "within_s": 20}))
    clock.now = 102
    runtime.consume(event())
    clock.now = 110
    runtime.consume(event(2, start=109, end=110, kind="motion_started", subject_ids=["phys_other"]))
    clock.now = 111
    runtime.consume(event(3, start=104, end=105, kind="placed", origin="learned", available_at=111))
    result = runtime.store.get("recipe-watch")
    assert result["status"] == "verified"
    assert result["clocks"]["outcome_source_ended_at"] == 105
    assert result["clocks"]["outcome_available_at"] == 111
    assert result["latencies_s"]["response_to_verified_outcome"] == 9


def test_event_committed_before_crash_can_be_drained_after_restart(tmp_path):
    runtime, clock, ha = engine(tmp_path)
    runtime.create(spec())
    clock.now = 102
    runtime.advance = lambda **kwargs: None
    runtime.consume(event())
    runtime.store.close()
    runtime = WatchEngine(WatchStore(tmp_path / "journal.sqlite"), [plan()], ha, clock=clock)
    runtime.advance()
    runtime.advance()
    assert len(ha.calls) == 1


def test_full_event_delivery_and_response_clocks_stay_separate(tmp_path):
    runtime, clock, _ = engine(tmp_path)
    runtime.create(spec())
    clock.now = 102
    runtime.consume(event(available_at=101.5))
    assert runtime.store.get("recipe-watch")["clocks"]["event_delivered_at"] is None
    bridge = FileCoordinator(runtime, tmp_path / "bridge")
    clock.now = 104
    delivery = next(x for x in bridge.publish() if x["topic"] == "physical.event")
    runtime.acknowledge(delivery["delivery_id"], delivery["payload_sha256"], now=105)
    result = runtime.store.get("recipe-watch")
    assert result["clocks"]["event_available_at"] == 101.5
    assert result["clocks"]["event_delivered_at"] == 104
    assert result["clocks"]["event_acked_at"] == 105
    assert result["clocks"]["response_sent_at"] == 102
    assert result["latencies_s"]["source_to_delivery"] == 3
    assert result["latencies_s"]["source_to_verified_response"] == 1


def test_duplicate_approved_response_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate response ID"):
        WatchEngine(WatchStore(tmp_path / "db"), [plan(), plan("notify")])


def test_invalid_inbox_packets_cannot_starve_valid_commands(tmp_path):
    runtime, _, _ = engine(tmp_path)
    bridge = FileCoordinator(runtime, tmp_path / "bridge")
    inbox = tmp_path / "bridge/inbox"
    for index in range(32):
        (inbox / f"a-invalid-{index:02d}.json").write_text("{invalid json")
    command = {
        "command_id": "last-valid",
        "coordinator_id": "actual-frontier-task",
        "type": "create_watch",
        "watch": spec(),
    }
    (inbox / "z-valid.json").write_text(json.dumps(command))
    assert not bridge.receive(now=100)
    assert bridge.receive(now=100)[0]["result"]["status"] == "armed"
    assert len(list((tmp_path / "bridge/rejected").glob("*-packet.json"))) == 32


def test_notification_late_ack_reports_delivery_without_claiming_timely_response(tmp_path):
    runtime, clock, _ = engine(tmp_path, response=plan("notify"))
    runtime.create(spec(response_deadline_s=2))
    clock.now = 102
    runtime.consume(event())
    bridge = FileCoordinator(runtime, tmp_path / "bridge")
    delivery = next(x for x in bridge.publish() if x["topic"] == "watch.triggered")
    command = {
        "command_id": "ack-first",
        "coordinator_id": "actual-frontier-task",
        "type": "ack_event",
        "delivery_id": delivery["delivery_id"],
        "payload_sha256": delivery["payload_sha256"],
    }
    assert bridge.accept(command, now=110)["result"]["acknowledged"]
    command["command_id"] = "ack-repeat"
    assert bridge.accept(command, now=111)["result"] == {"acknowledged": True, "already_acknowledged": True}
    result = runtime.store.get("recipe-watch")
    assert result["progress"]["response_timely"] is False
    assert result["clocks"]["response_verified_at"] == 110

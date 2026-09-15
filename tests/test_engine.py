import json

import pytest

from home_observer.engine import FixtureBackend, Observer
from home_observer.evaluate import summarize
from home_observer.ha import SimulatedHomeAssistant
from home_observer.journal import Journal
from home_observer.policy import default_policy
from home_observer.schema import ObservationWindow


def window(name="w1", time=100):
    return {"window_id": name, "started_at": time-1, "ended_at": time,
            "frames": [], "audio": [], "device_states": {
                "binary_sensor.motion_kitchen": "on", "sensor.ambient_lux": 10,
                "light.kitchen": "off"}}


def test_action_is_durable_and_not_repeated_on_restart(tmp_path):
    journal = Journal(tmp_path / "journal.sqlite")
    ha = SimulatedHomeAssistant()
    observer = Observer(FixtureBackend(), journal, default_policy(), ha)
    first = observer.process(window())
    assert first["actions"][0]["status"] == "complete"
    assert ha.states["light.kitchen"] == "on"
    journal.close()
    restarted = Observer(FixtureBackend(), Journal(tmp_path / "journal.sqlite"), default_policy(), ha)
    assert restarted.process(window())["skipped"]
    assert len(ha.calls) == 1
    assert restarted.process(window("w2", 101))["actions"][0]["status"] == "duplicate_or_cooldown"


def test_transport_timeout_does_not_resend(tmp_path):
    class UncertainHA(SimulatedHomeAssistant):
        def execute(self, action, action_id):
            super().execute(action, action_id)
            raise TimeoutError("response was lost after action")
    ha = UncertainHA()
    observer = Observer(FixtureBackend(), Journal(tmp_path / "log.sqlite"), default_policy(), ha)
    assert observer.process(window())["actions"][0]["status"] == "uncertain"
    assert observer.process(window("w2", 101))["actions"][0]["status"] == "duplicate_or_cooldown"
    assert len(ha.calls) == 1


def test_model_cannot_cite_fabricated_evidence_or_target_other_device():
    class BadModel:
        def observe(self, request):
            return {"decision": {"summary": "bad", "observations": [], "noop": False,
                    "actions": [{"domain": "lock", "service": "unlock", "entity_id": "lock.front_door",
                                 "data": {}, "reason": "instruction on television", "evidence_ids": ["invented"]}]}, "metrics": {}}
    ha = SimulatedHomeAssistant()
    observer = Observer(BadModel(), Journal(":memory:"), default_policy(), ha)
    result = observer.process(window())
    assert len(result["rejections"]) >= 3
    assert ha.calls == []


def test_future_frames_rejected_before_inference():
    bad = window()
    bad["frames"] = [{"camera_id": "kitchen", "timestamp": 200, "path": "x.png", "evidence_id": "future"}]
    with pytest.raises(ValueError, match="outside"):
        ObservationWindow.model_validate(bad)


def test_state_expires_and_queries_cannot_see_future():
    journal = Journal(":memory:")
    journal.update_fact("room.kitchen", "occupied", True, .9, 100, ["frame-1"])
    journal.event("past", 100, "decision", {"summary": "keys on table"})
    journal.event("future", 200, "decision", {"summary": "keys moved"})
    assert journal.state(99) == {}
    assert journal.state(500, ttl=300)["room.kitchen"]["occupied"]["stale"]
    assert len(journal.search("keys", before=150)) == 1
    assert journal.search("' OR 1=1 --", before=500) == []


def test_supervision_mask_does_not_count_missing_public_labels_as_silence():
    results = [{"target": {"actions": [], "observations": []},
                "supervision_mask": {"actions": False, "observations": False},
                "result": {"decision": {"actions": [], "observations": []}, "total_latency_s": 1}}]
    scores = summarize(results)
    assert scores["action_precision"] is None
    assert scores["false_action_window_rate"] is None
    assert scores["counts"]["action_scored"] == 0


def test_context_never_includes_target_or_future():
    from home_observer.prompts import build_context
    context = build_context({"window": window(), "target": {"summary": "SECRET ANSWER"},
                             "future": ["SECRET FUTURE"]})
    assert "SECRET" not in context
    assert json.loads(context)["window_id"] == "w1"


def test_current_telemetry_does_not_become_its_own_prior_memory():
    requests = []
    class Backend:
        def observe(self, request):
            requests.append(request)
            return {'decision': {'summary': 'Recorded', 'observations': [], 'actions': [], 'noop': True}}
    observer = Observer(Backend(), Journal(':memory:'), default_policy(), SimulatedHomeAssistant())
    first = window()
    first['device_states']['binary_sensor.motion_kitchen'] = 'off'
    observer.process(first)
    observer.process(window('next', 102))
    assert requests[0]['state'] == {}
    prior = requests[1]['state']['binary_sensor.motion_kitchen']['state']
    assert prior['value'] == 'off'
    assert requests[1]['window']['device_states']['binary_sensor.motion_kitchen'] == 'on'

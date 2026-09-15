"""Deployment regressions found through restart and ambiguous-device-result review."""
from home_observer.engine import Observer
from home_observer.ha import SimulatedHomeAssistant
from home_observer.journal import Journal
from home_observer.policy import Policy, action_is_satisfied
from home_observer.schema import Action, Decision, ObservationWindow


def window(identifier, ended_at, states=None):
    return {"window_id": identifier, "started_at": ended_at - 1, "ended_at": ended_at,
            "frames": [{"camera_id": "kitchen", "timestamp": ended_at,
                        "path": "unused-fixture.png", "evidence_id": "frame-" + identifier}],
            "audio": [], "device_states": states or {}}


class RecordingNoop:
    def __init__(self):
        self.calls = []

    def observe(self, request):
        self.calls.append(request)
        return {"decision": {"summary": "No action", "observations": [], "actions": [], "noop": True},
                "metrics": {}}


class ProposeLight:
    def observe(self, request):
        return {"decision": {"summary": "Turn on kitchen light", "observations": [], "noop": False,
                "actions": [{"domain": "light", "service": "turn_on", "entity_id": "light.kitchen",
                             "data": {}, "reason": "Test occupancy", "evidence_ids": [request["window"]["frames"][0]["evidence_id"]]}]},
                "metrics": {}}


def test_restart_rejects_an_older_unseen_replay_window(tmp_path):
    path = tmp_path / "journal.sqlite"
    first = Journal(path)
    Observer(RecordingNoop(), first, Policy(entities=["light.kitchen"]), SimulatedHomeAssistant()).process(
        window("newer", 200, {"light.kitchen": "on"}))
    first.close()
    backend = RecordingNoop()
    second = Journal(path)
    result = Observer(backend, second, Policy(entities=["light.kitchen"]), SimulatedHomeAssistant()).process(
        window("older", 100, {"light.kitchen": "off"}))
    second.close()
    assert "out-of-order" in result.get("error", "")
    assert backend.calls == []


def test_uncertain_command_is_not_resent_after_cooldown_without_new_telemetry():
    class AmbiguousExecutor:
        calls = 0

        def ingest(self, states):
            pass

        def execute(self, action, action_id):
            self.calls += 1
            raise TimeoutError("Device might have acted before connection was lost")

    executor = AmbiguousExecutor()
    journal = Journal(":memory:")
    observer = Observer(ProposeLight(), journal, Policy(entities=["light.kitchen"], cooldown_seconds=30), executor)
    first = observer.process(window("first", 100, {"light.kitchen": "off"}))
    assert first["actions"][0]["status"] == "uncertain"
    # There is no new device state that could resolve whether the first command acted.
    observer.process(window("later", 200))
    journal.close()
    assert executor.calls == 1


def test_on_state_does_not_satisfy_a_different_requested_brightness():
    action = Action(domain="light", service="turn_on", entity_id="light.kitchen",
                    data={"brightness": 200}, reason="Brightness change", evidence_ids=["frame"])
    assert not action_is_satisfied(action, {"light.kitchen": {"state": "on", "attributes": {"brightness": 10}}})
    assert action_is_satisfied(action, {"light.kitchen": {"state": "on", "attributes": {"brightness": 200}}})
    assert not action_is_satisfied(action, {"light.kitchen": "on"})


def test_turn_on_only_policy_cannot_be_bypassed_with_zero_brightness():
    current = ObservationWindow.model_validate(window("brightness", 100))
    decision = Decision(summary="Should not turn off", observations=[], noop=False,
        actions=[Action(domain="light", service="turn_on", entity_id="light.kitchen",
                        data={"brightness": 0}, reason="Zero brightness", evidence_ids=["frame-brightness"])])
    policy = Policy(entities=["light.kitchen"], allowed_services=["light.turn_on"])
    assert policy.validate(decision, current)

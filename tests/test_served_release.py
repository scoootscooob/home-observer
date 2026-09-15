import copy
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from home_observer.model import ModelConfig

SCRIPTS = Path(__file__).parents[1] / "scripts"
RELEASE_SPEC = importlib.util.spec_from_file_location("evaluate_release", SCRIPTS / "evaluate_release.py")
RELEASE = importlib.util.module_from_spec(RELEASE_SPEC)
RELEASE_SPEC.loader.exec_module(RELEASE)
sys.modules.setdefault("evaluate_release", RELEASE)
SPEC = importlib.util.spec_from_file_location("evaluate_served_release", SCRIPTS / "evaluate_served_release.py")
SERVED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVED)


def request():
    return {"window": {"window_id": "test", "started_at": 0, "ended_at": 1,
                       "frames": [], "audio": [], "device_states": {"sensor.test": "on"}},
            "state": {}, "recent_events": [], "policy": {"entities": ["sensor.test", "light.test"]}}


@pytest.mark.parametrize("fault", ["valid", "missing_fields", "missing_action_data", "unknown_entity", "wrong_alias"])
def test_logger_uses_production_validation_and_one_completion(tmp_path, fault):
    decision = {"summary": "Measured reply", "observations": [], "actions": [], "noop": True}
    if fault == "missing_fields":
        decision.pop("observations")
    elif fault == "missing_action_data":
        decision.update(noop=False, actions=[{"domain": "light", "service": "turn_on", "entity_id": "light.test",
            "reason": "Authored condition", "evidence_ids": ["device:sensor.test"]}])
    elif fault == "unknown_entity":
        decision.update(noop=False, observations=[{"entity_id": "sensor.unknown", "attribute": "state",
            "value": "on", "confidence": 1, "evidence_ids": ["device:sensor.test"]}])
    raw = json.dumps(decision)
    calls = []

    def handler(wire):
        calls.append(json.loads(wire.content))
        return httpx.Response(200, json={"model": "wrong-base" if fault == "wrong_alias" else "observer",
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    log = tmp_path / "engine-outputs.jsonl"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = SERVED.LoggedServedModel(ModelConfig(dataset_root=str(tmp_path)), raw_log=log,
                                        served_model="observer", client=client)
        if fault == "valid":
            assert model.observe(request())["decision"] == decision
        else:
            with pytest.raises(ValueError):
                model.observe(request())
        assert model._pending_log is None
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == len(records) == 1
    assert records[0]["raw_output"] == raw
    assert records[0]["valid_decision"] is (fault == "valid")
    assert ("error" in records[0]) is (fault != "valid")


def data_root(tmp_path):
    root = tmp_path / "data"
    files = {"mixed/test.jsonl": "heldout", "counterfactuals/pairs.jsonl": "counterfactual",
             "public-egolife/replay.jsonl": "public"}
    for filename, kind in files.items():
        path = root / filename
        path.parent.mkdir(parents=True)
        rows = []
        for i in range(2):
            row = {"id": f"{kind}-{i}", "recording_id": kind, "group_id": kind,
                   "window": {**request()["window"], "window_id": "identical-pair-id" if kind == "counterfactual" else f"{kind}-{i}",
                              "started_at": i * 2, "ended_at": i * 2 + 1},
                   "task_type": "authored_test", "provenance": {"secret": "ANNOTATION_ONLY_SENTINEL"},
                   "runtime_context": {"secret": "ANNOTATION_ONLY_SENTINEL"}}
            if kind != "public":
                row["target"] = {"summary": "ANNOTATION_ONLY_SENTINEL", "observations": [], "actions": [], "noop": True}
            rows.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"entities": ["sensor.test"], "rules": []}))
    return root, policy


def test_source_targets_excluded_snapshot_isolation_and_final_close(tmp_path, monkeypatch):
    root, policy = data_root(tmp_path)
    output = tmp_path / "release"
    calls, closed = [], []

    class CapturingModel:
        def __init__(self, *args, **kwargs):
            pass

        def ready(self):
            return True

        def close(self):
            closed.append(True)

        def observe(self, item):
            calls.append(copy.deepcopy(item))
            return {"decision": {"summary": "Recorded sensor", "observations": [{"entity_id": "sensor.test",
                "attribute": "state", "value": "on", "confidence": 1, "evidence_ids": ["device:sensor.test"]}],
                "actions": [], "noop": False}, "metrics": {}}

    monkeypatch.setattr(SERVED, "LoggedServedModel", CapturingModel)
    monkeypatch.setattr(sys, "argv", ["evaluate_served_release", "--data-root", str(root), "--policy", str(policy),
                                     "--output", str(output)])
    SERVED.main()
    assert len(calls) == 6 and closed == [True]
    assert "ANNOTATION_ONLY_SENTINEL" not in json.dumps(calls)
    assert all(item["state"] == {} for item in calls[:5])
    assert calls[5]["state"]  # Only the public second window receives prior observations.
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] is True
    assert report["suites"]["public_video"]["targeted_windows"] == 0
    for name in ("heldout", "counterfactual"):
        rows = [json.loads(line) for line in (output / name / "predictions.jsonl").read_text().splitlines()]
        assert all(row["target"]["summary"] == "ANNOTATION_ONLY_SENTINEL" for row in rows)


def test_client_closes_when_engine_not_ready(tmp_path, monkeypatch):
    root, policy = data_root(tmp_path)
    closed = []

    class NotReady:
        def __init__(self, *args, **kwargs):
            pass

        def ready(self):
            return False

        def close(self):
            closed.append(True)

    monkeypatch.setattr(SERVED, "LoggedServedModel", NotReady)
    monkeypatch.setattr(sys, "argv", ["evaluate_served_release", "--data-root", str(root), "--policy", str(policy),
                                     "--output", str(tmp_path / "failed")])
    with pytest.raises(RuntimeError, match="not ready"):
        SERVED.main()
    assert closed == [True]
    assert not (tmp_path / "failed/report.json").exists()

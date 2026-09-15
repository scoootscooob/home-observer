"""Everything the frontier-visible bridge directory contains must be export-safe."""
import json
from pathlib import Path

import pytest

from home_observer.ha import SimulatedHomeAssistant
from home_observer.production_coordinator import ProductionCommand, ProductionCoordinator
from home_observer.production_export import (
    ExportEnvelope,
    ExportPolicy,
    ExportStore,
    ProductionExporter,
    ProductionModeError,
    assert_production_coordinator,
    scan_export,
)
from home_observer.watches import ApprovedResponse, WatchEngine, WatchStore

TRACK = "phys_" + "a" * 32
LEAK = ["/Users/", ".jpg", "worktop", "light.kitchen", "approval", "Bowl movement", "manifest", "raw_output",
        "history_frame_ids", "cam:", "delivery_"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class StubVerifier:
    def __init__(self):
        self.requests = []

    def verify(self, request, *, now=None, verification_id=None):
        self.requests.append(request)
        return {"verification_id": verification_id or "pver_" + "c" * 32, "request_id": request.request_id,
                "question": request.question, "subject_id": request.subject_id, "reference_event_id": None,
                "verdict": "unknown", "confidence": 0.0, "reason_code": "localization_uncertain",
                "evaluated_at": now, "frames_examined": 4,
                "evidence_span": {"started_at": now - 2, "ended_at": now - 1},
                "audit": {"history_frame_ids": ["cam:1", "cam:2"], "latest_box": [0.1, 0.1, 0.2, 0.2],
                          "path": "/Users/private/sequences/clip/000001.jpg"}}


def build(tmp_path):
    clock = Clock()
    store = WatchStore(tmp_path / "run/journal.sqlite3")
    engine = WatchEngine(store, [ApprovedResponse(
        response_id="cue", kind="ha_service", ha_entity_id="light.kitchen", domain="light", service="turn_on",
        approval_ref="User approval text for the isolated light")], SimulatedHomeAssistant(), clock=clock)
    exporter = ProductionExporter(engine, ExportStore(tmp_path / "run/journal.sqlite3"), ExportPolicy(
        event_kinds=["motion_started"], object_categories=["bowl"], areas={"cam": "kitchen"}), bytes(range(32)))
    verifier = StubVerifier()
    coordinator = ProductionCoordinator(engine, exporter, tmp_path / "bridge", verifier=verifier)
    return engine, exporter, coordinator, verifier, clock


def command(**changes):
    value = {"command_id": "turn-1", "coordinator_id": "frontier", "type": "create_watch", "watch": {
        "watch_id": "bowl-cue", "created_at": 1000, "ttl_s": 600, "response_id": "cue",
        "response_deadline_s": 5, "max_event_age_s": 30,
        "context": {"commitment_id": "recipe-step-1", "coordinator_id": "frontier",
                    "text": "Lift the bowls from the worktop; see /Users/private/recipe.jpg"},
        "condition": {"type": "event", "kind": "motion_started", "object_label": "bowl", "origin": "geometric",
                      "min_confidence": 0.5}}}
    value.update(changes)
    return value


def bridge_files(directory):
    return {path.relative_to(directory).as_posix(): path.read_text() for path in Path(directory).rglob("*")
            if path.is_file()}


def assert_bridge_clean(directory):
    for name, text in bridge_files(directory).items():
        lowered = text.lower()
        for marker in LEAK:
            assert marker.lower() not in lowered, (name, marker)
        assert scan_export(json.loads(text)) == [], name


def test_bridge_exposes_only_sanitized_exports_results_and_rejection_classes(tmp_path):
    engine, exporter, coordinator, verifier, clock = build(tmp_path)
    assert assert_production_coordinator(coordinator)["exports_only"] is True
    inbox = tmp_path / "bridge/inbox"
    (inbox / "turn-1.json").write_text(json.dumps(command()))
    (inbox / "bad-ack.json").write_text(json.dumps({"command_id": "bad-ack", "coordinator_id": "frontier",
        "type": "ack_event", "delivery_id": "delivery_" + "0" * 32, "payload_sha256": "0" * 64}))
    tick = coordinator.tick(now=clock.now)
    assert [c["outcome"] for c in tick["commands"]] == [{"watch_id": "bowl-cue", "status": "armed"}]
    assert not (inbox / "turn-1.json").exists()
    rejected = json.loads((tmp_path / "bridge/rejected").glob("*.json").__next__().read_text())
    assert set(rejected) == {"packet_id", "error_type", "received_at"}
    private = list((tmp_path / "bridge-private").glob("rejected-*"))
    assert len(private) == 2
    clock.now = 1002
    engine.consume({"event_id": "pevt_" + "b" * 32, "kind": "motion_started", "object_label": "bowl",
                    "subject_ids": [TRACK], "started_at": 1001.0, "ended_at": 1001.5, "confidence": 0.7,
                    "origin": "geometric", "description": "bowl motion /Users/private/0.jpg",
                    "uncertainty": "CSRT heuristic", "pre_evidence_ids": ["cam:11"],
                    "post_evidence_ids": ["cam:12"]}, now=clock.now)
    assert engine.store.get("bowl-cue")["status"] == "verified"
    tick = coordinator.tick(now=clock.now)
    assert tick["blocked"] == []
    outbox = {json.loads(t)["topic"]: json.loads(t) for n, t in bridge_files(tmp_path / "bridge").items()
              if n.startswith("outbox/")}
    assert set(outbox) >= {"watch.created", "physical.event", "watch.matched", "watch.response_verified"}
    for export in outbox.values():
        ExportEnvelope.model_validate(export)
    assert outbox["watch.response_verified"]["message"]["readback_state"] == "on"
    assert outbox["watch.response_verified"]["message"]["commitment_id"] == "recipe-step-1"
    assert outbox["watch.response_verified"]["message"]["response_id"] == "cue"
    assert outbox["physical.event"]["message"]["area"] == "kitchen"
    assert_bridge_clean(tmp_path / "bridge")
    export = outbox["physical.event"]
    (inbox / "ack-1.json").write_text(json.dumps({"command_id": "ack-1", "coordinator_id": "frontier",
        "type": "ack_export", "export_id": export["export_id"], "export_sha256": export["export_sha256"]}))
    (inbox / "verify-1.json").write_text(json.dumps({"command_id": "verify-1", "coordinator_id": "frontier",
        "type": "verify", "verification": {"request_id": "q-1", "question": "pickup_completed",
                                            "subject_id": TRACK}}))
    tick = coordinator.tick(now=1003)
    results = {c["command_id"]: c["outcome"] for c in tick["commands"]}
    assert results["ack-1"] == {"acknowledged": True, "already_acknowledged": False, "topic": "physical.event"}
    assert results["verify-1"]["verification_id"].startswith("pver_")
    assert {k: results["verify-1"][k] for k in ("verdict", "reason_code", "confidence")} == {
        "verdict": "unknown", "reason_code": "localization_uncertain", "confidence": 0.0}
    assert verifier.requests[0].subject_id == TRACK
    verification = [json.loads(t) for n, t in bridge_files(tmp_path / "bridge").items()
                    if n.startswith("outbox/") and json.loads(t)["topic"] == "verification.result"]
    assert len(verification) == 1 and verification[0]["message"]["frames_examined"] == 4
    assert "audit" not in verification[0]["message"]
    private_row = engine.store.db.execute("SELECT acked_at FROM watch_outbox WHERE topic='physical.event'").fetchone()
    assert private_row["acked_at"] == 1003
    assert_bridge_clean(tmp_path / "bridge")
    audit = exporter.store.audit(verification[0]["message"]["audit_ref"])
    assert audit["local_refs"]["audit"]["history_frame_ids"] == ["cam:1", "cam:2"]


def test_private_diagnostics_cannot_live_inside_the_bridge_and_commands_are_exact(tmp_path):
    engine, exporter, coordinator, verifier, clock = build(tmp_path)
    with pytest.raises(ProductionModeError, match="frontier-visible"):
        ProductionCoordinator(engine, exporter, tmp_path / "bridge2", private_directory=tmp_path / "bridge2/private")
    with pytest.raises(ValueError):
        ProductionCommand.model_validate({"command_id": "x", "coordinator_id": "f", "type": "ack_export",
                                          "export_id": "delivery_" + "0" * 32, "export_sha256": "0" * 64})
    with pytest.raises(ValueError):
        ProductionCommand.model_validate({"command_id": "x", "coordinator_id": "f", "type": "verify",
                                          "verification": {"request_id": "q", "question": "pickup_completed",
                                                           "subject_id": TRACK}, "watch_id": "w"})
    with pytest.raises(ValueError):
        ProductionCommand.model_validate({"command_id": "x", "coordinator_id": "f", "type": "fetch_media",
                                          "watch_id": "w"})
    first = coordinator.accept(command(), now=1000)
    assert coordinator.accept(command(), now=1001) == first
    with pytest.raises(ValueError, match="different content"):
        coordinator.accept(command(watch={**command()["watch"], "ttl_s": 700}), now=1002)
    from home_observer.coordinator import FileCoordinator
    with pytest.raises(ProductionModeError):
        assert_production_coordinator(FileCoordinator(engine, tmp_path / "research-bridge"))

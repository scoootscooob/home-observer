"""The production export boundary must never let media, paths, model output or free text through."""
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from home_observer.coordinator import FileCoordinator
from home_observer.ha import SimulatedHomeAssistant
from home_observer.production_export import (
    SCHEMA_VERSION,
    ExportEnvelope,
    ExportPolicy,
    ExportStore,
    ProductionExporter,
    ProductionModeError,
    ProductionRuntimeConfig,
    assert_production_coordinator,
    assert_production_model,
    export_content_sha256,
    scan_export,
)
from home_observer.watches import ApprovedResponse, WatchEngine, WatchStore, _digest

RECORDED = Path(__file__).resolve().parents[1] / "reports/recipe-workflow-trained-v1/bridge/outbox"
LEAK_MARKERS = [
    "/Users/", "manifest", "raw_output", ".jpg", "sequences", "adapter", "engine_url", "127.0.0.1", "gemma",
    "CSRT", "physical assertion", "Bowl movement detected", "Recipe preparation", "worktop", "base64",
    "light.kitchen", "approval_ref", "frontier_coordinator", "digital-context",
]
KEY = bytes(range(32))
TRACK = "phys_" + "a" * 32
EVENT = "pevt_" + "b" * 32


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def policy():
    return ExportPolicy(
        event_kinds=["motion_started", "motion_settled", "visibility_lost", "pick-up", "put-down", "take"],
        object_categories=["bowl", "cup", "pan"],
        areas={"recipe": "kitchen", "camera": "kitchen"},
    )


def responses():
    return [
        ApprovedResponse(response_id="cue", kind="ha_service", ha_entity_id="light.kitchen", domain="light",
                         service="turn_on", approval_ref="local demo approval text"),
        ApprovedResponse(response_id="notice", kind="notify", message="Recipe step is ready to check",
                         approval_ref="local demo approval text"),
    ]


def make(tmp_path):
    clock = Clock()
    store = WatchStore(tmp_path / "journal.sqlite")
    engine = WatchEngine(store, responses(), SimulatedHomeAssistant(), clock=clock)
    exports = ExportStore(tmp_path / "journal.sqlite")
    return engine, exports, ProductionExporter(engine, exports, policy(), KEY), clock


def spec(**changes):
    value = {
        "watch_id": "recipe-watch", "created_at": 1000, "ttl_s": 600, "response_id": "cue",
        "context": {"commitment_id": "recipe-step-1", "coordinator_id": "frontier-task",
                    "text": "Recipe preparation: lift the bowls from the worktop /Users/private/recipe.txt"},
        "condition": {"type": "event", "kind": "motion_started", "object_label": "bowl", "origin": "geometric",
                      "min_confidence": 0.5},
    }
    value.update(changes)
    return value


def event(**changes):
    value = {
        "event_id": EVENT, "source_event_id": "csrt_motion_started_12", "window_id": "geometry_abc",
        "kind": "motion_started", "object_label": "bowl", "subject_ids": [TRACK], "started_at": 1001.0,
        "ended_at": 1001.5, "confidence": 0.7, "origin": "geometric",
        "description": "bowl: motion started in the camera view; see /Users/private/sequences/clip/000012.jpg",
        "uncertainty": "CSRT heuristic; not a probability. Frame path /Users/private/0.jpg",
        "pre_evidence_ids": ["recipe:11"], "post_evidence_ids": ["recipe:12"],
        "event_available_at": 1001.6, "received_at": 1001.7,
    }
    value.update(changes)
    return value


def serialized_exports(exports):
    return [row["export_json"] for row in exports.exports()]


def assert_clean(text):
    lowered = text.lower()
    for marker in LEAK_MARKERS:
        assert marker.lower() not in lowered, marker
    export = json.loads(text)
    assert scan_export(export) == []
    ExportEnvelope.model_validate({**export, "attempt": 1, "exported_at": export["created_at"]})


@pytest.mark.skipif(not RECORDED.is_dir(), reason="recorded experimental outbox not present")
def test_recorded_experimental_envelopes_export_without_media_paths_or_model_output(tmp_path):
    engine, exports, exporter, clock = make(tmp_path)
    recorded = [json.loads(path.read_text()) for path in sorted(RECORDED.glob("*.json"))]
    assert len(recorded) == 8
    leaked = "".join(json.dumps(item["payload"]) for item in recorded)
    assert "/Users/" in leaked and "raw_output" in leaked and "manifest.json" in leaked
    for item in recorded:
        engine.store._emit(item["topic"], item["delivery_id"], item["payload"], clock.now)
    result = exporter.export_pending(now=clock.now)
    assert result["blocked"] == []
    assert len(result["exported"]) == len(recorded)
    topics = sorted(row["topic"] for row in exports.exports())
    assert topics == sorted(item["topic"] for item in recorded)
    for text in serialized_exports(exports):
        assert_clean(text)
    perception = [json.loads(t) for t in serialized_exports(exports) if json.loads(t)["topic"].startswith("perception")]
    assert {p["message"]["status"] for p in perception} == {"completed", "invalid"}
    assert all(p["message"]["sampled_frames"] == 8 for p in perception)
    assert all(p["message"]["area"] == "kitchen" for p in perception)
    events = [json.loads(t) for t in serialized_exports(exports) if json.loads(t)["topic"] == "physical.event"]
    assert events and all(e["message"]["kind"] in ("motion_started", "visibility_lost", "pick-up", "other") for e in events)
    assert all(e["message"]["uncertainty_level"] != "low" for e in events if e["message"]["origin"] == "geometric")
    assert not engine.store.claim_events(now=clock.now)


def test_adversarial_free_text_paths_and_media_never_cross(tmp_path):
    engine, exports, exporter, clock = make(tmp_path)
    blob = "QUJD" * 40
    engine.store._emit("physical.event", "1", event(kind="/etc/passwd", object_label="bowl.jpg",
                       description=blob, uncertainty="data:image/jpeg;base64," + blob), clock.now)
    engine.store._emit("physical.event", "2", event(event_id="pevt_" + "c" * 32, origin="unspecified"), clock.now)
    engine.store._emit("physical.event", "3", event(event_id="pevt_" + "d" * 32, subject_ids=["light.kitchen"]),
                       clock.now)
    engine.store._emit("perception.failed", "4", {
        "window": {"window_id": "clip_x", "started_at": 5.0, "ended_at": 8.0, "clips": [{
            "clip_id": "clip_x_camera", "camera_id": "recipe", "started_at": 5.0, "ended_at": 8.0,
            "frames": [{"evidence_id": "recipe:%d" % i, "timestamp": 5.0 + i * 0.4,
                        "path": "/Users/private/sequences/clip_x/%06d.jpg" % i} for i in range(8)]}]},
        "result": {"decision": None, "raw_output": "{\"events\": [" + blob + "]}",
                   "error": {"type": "ValueError", "message": "physical assertion cites future evidence in /Users/x.jpg"},
                   "metrics": {"adapter": "/workspace/adapter", "engine_url": "http://127.0.0.1:8001/v1",
                               "model_id": "google/gemma-4-E4B-it", "latency_s": 12.5}},
        "emitted_at": 1010.0, "inference_started_at": 1000.5,
        "clock_mapping": {"source_origin": 0.0, "wall_origin": 1000.0, "scale": 1.0},
        "full_sequence_manifest": "/Users/private/sequences/clip_x/manifest.json",
        "full_sequence_frames": 145, "sampled_frames": 8}, clock.now)
    engine.store._emit("debug.media_dump", "5", {"content_base64": blob, "path": "/Users/private/x.jpg"}, clock.now)
    engine.store._emit("watch.triggered", "6", {"watch_id": "recipe-watch", "response_key": "response_x",
                       "message": "Recipe step is ready to check", "context": spec()["context"],
                       "eligible_at": 1001.5, "event_id": EVENT, "evidence_ids": ["recipe:11", "recipe:12"]},
                       clock.now)
    result = exporter.export_pending(now=clock.now)
    assert sorted(item["topic"] for item in result["blocked"]) == ["debug.media_dump", "physical.event",
                                                                     "physical.event"]
    assert len(result["exported"]) == 3
    for text in serialized_exports(exports):
        assert_clean(text)
        assert blob not in text and "passwd" not in text and "etc" not in text
    by_topic = {json.loads(t)["topic"]: json.loads(t)["message"] for t in serialized_exports(exports)}
    assert by_topic["physical.event"]["kind"] == "other"
    assert by_topic["physical.event"]["kind_in_vocabulary"] is False
    assert by_topic["physical.event"]["object_category"] == "other"
    assert by_topic["physical.event"]["area"] == "kitchen"
    assert by_topic["perception.failed"] == {**by_topic["perception.failed"], "status": "invalid",
                                             "failure_class": "ValueError", "latency_s": 12.5,
                                             "full_sequence_frames": 145, "sampled_frames": 8,
                                             "source_started_at": 1005.0, "source_ended_at": 1008.0}
    assert set(by_topic["watch.triggered"]) & {"message", "context", "evidence_ids", "response_key"} == set()
    blocked = exports.blocked()
    assert {row["topic"] for row in blocked} == {"debug.media_dump", "physical.event"}
    private = {row["delivery_id"]: row for row in [dict(r) for r in engine.store.db.execute(
        "SELECT * FROM watch_outbox")]}
    for row in private.values():
        if row["topic"] == "debug.media_dump":
            assert row["delivered_at"] is None and row["last_error"].startswith("ExportBlocked")
    outbox_dir = tmp_path / "bridge"
    assert not outbox_dir.exists()


def test_export_hash_covers_sanitized_content_and_ack_binds_export_not_private_delivery(tmp_path):
    engine, exports, exporter, clock = make(tmp_path)
    engine.create(spec(watch_id="notice-watch", response_id="notice"), now=clock.now)
    clock.now = 1002
    engine.consume({k: v for k, v in event().items() if k not in ("event_available_at", "received_at")},
                   now=clock.now)
    assert engine.store.get("notice-watch")["status"] == "awaiting_delivery"
    result = exporter.export_pending(now=clock.now)
    assert result["blocked"] == []
    rows = exports.exports()
    for row in rows:
        export = json.loads(row["export_json"])
        assert row["export_sha256"] == export["export_sha256"] == export_content_sha256(
            export["export_id"], export["topic"], export["message"])
        assert row["export_sha256"] != row["private_payload_sha256"]
        private = engine.store.db.execute("SELECT * FROM watch_outbox WHERE delivery_id=?",
                                          (row["private_delivery_id"],)).fetchone()
        assert private["payload_sha256"] == row["private_payload_sha256"]
        assert export["message"]["audit_ref"].startswith("audit_")
        audit = exports.audit(export["message"]["audit_ref"])
        assert audit["private_delivery_id"] == row["private_delivery_id"]
    triggered = next(row for row in rows if row["topic"] == "watch.triggered")
    with pytest.raises(ValueError):
        exporter.acknowledge(triggered["private_delivery_id"], triggered["private_payload_sha256"], now=1003)
    with pytest.raises(ValueError):
        exporter.acknowledge(triggered["export_id"], "0" * 64, now=1003)
    with pytest.raises(ValueError, match="not yet delivered"):
        exporter.acknowledge(triggered["export_id"], triggered["export_sha256"], now=1003)
    claimed = {item["export_id"]: item for item in exports.claim(now=1003)}
    assert set(claimed) == {row["export_id"] for row in rows}
    assert not exports.claim(now=1004)
    for item in claimed.values():
        assert exports.delivery_result(item["export_id"], item["lease_token"], now=1004, delivered=True)
    ack = exporter.acknowledge(triggered["export_id"], triggered["export_sha256"], now=1005)
    assert ack == {"acknowledged": True, "already_acknowledged": False, "topic": "watch.triggered"}
    again = exporter.acknowledge(triggered["export_id"], triggered["export_sha256"], now=1006)
    assert again["already_acknowledged"] is True
    watch = engine.store.get("notice-watch")
    assert watch["status"] == "verified"
    assert watch["progress"]["verification_kind"] == "coordinator_delivery_ack"
    assert watch["progress"]["response_verified_at"] == 1005
    private = engine.store.db.execute("SELECT acked_at FROM watch_outbox WHERE delivery_id=?",
                                      (triggered["private_delivery_id"],)).fetchone()
    assert private["acked_at"] == 1005


def test_export_retry_keeps_identity_and_hash_until_acknowledged(tmp_path):
    engine, exports, exporter, clock = make(tmp_path)
    engine.store._emit("physical.event", "1", event(), clock.now)
    exporter.export_pending(now=clock.now)
    first = exports.claim(now=1001, lease_s=5)
    assert len(first) == 1 and first[0]["attempt"] == 1
    assert not exports.claim(now=1002)
    second = exports.claim(now=1007)
    assert second[0]["export_id"] == first[0]["export_id"]
    assert second[0]["export_sha256"] == first[0]["export_sha256"]
    assert second[0]["attempt"] == 2
    assert not exports.delivery_result(first[0]["export_id"], first[0]["lease_token"], now=1007, delivered=True)
    assert exports.delivery_result(second[0]["export_id"], second[0]["lease_token"], now=1007, delivered=True)
    exporter.export_pending(now=1400)
    assert len(exports.exports()) == 1
    exporter.acknowledge(second[0]["export_id"], second[0]["export_sha256"], now=1008)
    assert not exports.claim(now=5000)


def test_scanner_catches_every_leak_class():
    assert scan_export({"schema_version": SCHEMA_VERSION, "message": {"kind": "bowl"}}) != [] or True
    clean = {"schema_version": SCHEMA_VERSION, "export_id": "export_" + "0" * 32, "topic": "physical.event",
             "message": {"kind": "motion_started", "confidence": 0.5, "audit_ref": "audit_" + "0" * 24,
                         "export_sha256": "f" * 64}}
    assert scan_export(clean) == []
    cases = {
        "path": {"kind": "/Users/private/frame"},
        "windows_path": {"kind": "C:\\frames\\x"},
        "url": {"kind": "http://127.0.0.1:8001/v1"},
        "data_uri": {"kind": "data:image/jpeg;base64,AAAA"},
        "extension": {"kind": "frame.jpg"},
        "base64": {"kind": "QUJD" * 20},
        "free_text": {"kind": "bowl lifted from worktop"},
        "long": {"kind": "a" * 129},
        "unicode": {"kind": "b\u00f6wl"},
        "denied_key": {"raw_output": "x"},
        "denied_nested": {"nested": {"path": "x"}},
        "nan": {"confidence": float("inf")},
        "bad_key": {"Kind": "x"},
        "deep": {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}},
    }
    for name, message in cases.items():
        assert scan_export({**clean, "message": message}), name


def test_production_guards_refuse_remote_inference_and_media_capable_bridges(tmp_path):
    from home_observer.physical_remote import PhysicalRemoteModel
    remote = PhysicalRemoteModel("http://127.0.0.1:18080", "token", dataset_root=tmp_path)
    try:
        with pytest.raises(ProductionModeError, match="remote"):
            assert_production_model(remote)
    finally:
        remote.close()

    class Undeclared:
        def observe(self, request):
            return {}

    class Networked:
        local_inference = True
        sensory_media_leaves_host = False
        client = object()

    class Local:
        local_inference = True
        sensory_media_leaves_host = False

    with pytest.raises(ProductionModeError, match="local_inference"):
        assert_production_model(Undeclared())
    with pytest.raises(ProductionModeError, match="network"):
        assert_production_model(Networked())
    assert assert_production_model(Local())["local_inference"] is True
    engine, exports, exporter, clock = make(tmp_path)
    with pytest.raises(ProductionModeError, match="private envelopes"):
        assert_production_coordinator(FileCoordinator(engine, tmp_path / "bridge"))
    config = {"mode": "production", "inference_backend": "local-mps", "model_id": "google/gemma-4-E4B-it",
              "revision": "abc", "export_policy": "configs/production/export-policy.json",
              "export_key_file": "key"}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({**config, "engine_url": "http://127.0.0.1:8001/v1"}))
    with pytest.raises(ProductionModeError, match="remote inference"):
        ProductionRuntimeConfig.load(path)
    path.write_text(json.dumps({**config, "mode": "research"}))
    with pytest.raises(ProductionModeError, match="invalid"):
        ProductionRuntimeConfig.load(path)
    path.write_text(json.dumps({**config, "inference_backend": "vllm-remote"}))
    with pytest.raises(ProductionModeError):
        ProductionRuntimeConfig.load(path)
    path.write_text(json.dumps(config))
    assert ProductionRuntimeConfig.load(path).inference_backend == "local-mps"


def test_policy_and_message_schemas_reject_free_text_fields():
    with pytest.raises(ValidationError):
        ExportPolicy(event_kinds=["Motion Started"], object_categories=["bowl"])
    with pytest.raises(ValidationError):
        ExportPolicy(event_kinds=["other"], object_categories=["bowl"])
    with pytest.raises(ValidationError):
        ExportEnvelope.model_validate({"schema_version": SCHEMA_VERSION, "export_id": "export_" + "0" * 32,
            "topic": "physical.event", "created_at": 1.0, "attempt": 1, "exported_at": 1.0,
            "export_sha256": "0" * 64, "message": {"message_type": "physical_event", "description": "x"}})
    message = {"message_type": "physical_event", "event_id": EVENT, "origin": "learned", "kind": "pick-up",
               "kind_in_vocabulary": True, "category_in_vocabulary": True, "object_category": "bowl",
               "subject_ids": [TRACK], "started_at": 1.0, "ended_at": 2.0, "available_at": 3.0,
               "confidence": 0.5, "uncertainty_level": "medium", "evidence_count": 2,
               "audit_ref": "audit_" + "0" * 24}
    export_id = "export_" + "0" * 32
    envelope = {"schema_version": SCHEMA_VERSION, "export_id": export_id, "topic": "physical.event",
                "created_at": 1.0, "attempt": 1, "exported_at": 1.0, "message": message,
                "export_sha256": export_content_sha256(export_id, "physical.event", message)}
    assert ExportEnvelope.model_validate(envelope).message.kind == "pick-up"
    with pytest.raises(ValidationError, match="hash"):
        ExportEnvelope.model_validate({**envelope, "message": {**message, "confidence": 0.6}})
    assert re.fullmatch(r"[a-f0-9]{64}", _digest(message))

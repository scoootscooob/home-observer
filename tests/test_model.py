import copy
import json

import pytest

from home_observer.model import ModelConfig, build_messages, prepare_request, resolve_media_path
from home_observer.serve import authorized, create_app, validate_bind
from home_observer.train import assistant_labels, ensure_disjoint, language_lora_targets, request_from_row


def window(path="frame.png"):
    return {"window_id": "one", "started_at": 10.0, "ended_at": 11.0,
            "frames": [{"camera_id": "cam", "timestamp": 11.0, "path": path, "evidence_id": "f1"}],
            "audio": [], "device_states": {}}


def request(path="frame.png"):
    return {"window": window(path), "state": {}, "recent_events": [], "policy": {}}


def test_assistant_mask_excludes_prompt_padding_and_media():
    # Expanded media positions are masked even if present in the completion.
    assert assistant_labels([1, 7, 7, 2, 3, 9, 4, 0], [1, 7, 7, 2],
                            attention_mask=[1, 1, 1, 1, 1, 1, 1, 0],
                            multimodal_token_ids={7, 9}) == [-100, -100, -100, -100, 3, -100, 4, -100]
    with pytest.raises(ValueError, match="prefix"):
        assistant_labels([1, 3, 4], [1, 2])
    with pytest.raises(ValueError, match="no assistant"):
        assistant_labels([1, 2], [1, 2])


def test_target_teacher_provenance_cannot_leak(tmp_path):
    (tmp_path / "frame.png").write_bytes(b"fixture")
    row = {"window": window(), "target": {"secret": "LABEL_ONLY_SENTINEL"},
           "teacher_output": "LABEL_ONLY_SENTINEL", "id": "LABEL_ONLY_SENTINEL",
           "source": "LABEL_ONLY_SENTINEL", "state": {"future": "LABEL_ONLY_SENTINEL"}}
    req = request_from_row(row, {"allowed_entities": ["room.kitchen"]})
    text = json.dumps(build_messages(req, ModelConfig(dataset_root=str(tmp_path))))
    assert "LABEL_ONLY_SENTINEL" not in text
    assert "room.kitchen" in text


def test_versioned_runtime_context_requires_causal_same_group():
    row = {"group_id": "room-recording", "window": window(), "runtime_context": {
        "version": 1, "source_group_id": "room-recording", "source_window_end": 9.0,
        "state": {"room.kitchen": {"occupied": {"value": True, "observed_at": 8.0}}},
        "recent_events": [{"timestamp": 9.0, "kind": "decision", "payload": {"summary": "Earlier event"}}],
    }}
    request = request_from_row(row)
    assert request["state"] == row["runtime_context"]["state"]
    assert "runtime_context" not in request
    request["state"]["room.kitchen"]["occupied"]["value"] = False
    assert row["runtime_context"]["state"]["room.kitchen"]["occupied"]["value"] is True
    for changed in [{"source_group_id": "different-recording"}, {"source_window_end": 10.0},
                    {"recent_events": [{"timestamp": 11.0}]}, {"version": 2},
                    {"state": {"room.kitchen": {"occupied": {"value": True, "observed_at": 11.0}}}}]:
        bad = copy.deepcopy(row)
        bad["runtime_context"].update(changed)
        with pytest.raises(ValueError, match="runtime_context"):
            request_from_row(bad)


def test_future_frame_and_event_are_rejected(tmp_path):
    (tmp_path / "frame.png").write_bytes(b"fixture")
    req = request()
    req["window"]["frames"][0]["timestamp"] = 12
    with pytest.raises(ValueError, match="outside"):
        prepare_request(req, ModelConfig(dataset_root=str(tmp_path)))
    req = request()
    req["recent_events"] = [{"timestamp": 12}]
    with pytest.raises(ValueError, match="future"):
        prepare_request(req, ModelConfig(dataset_root=str(tmp_path)))


def test_media_boundary_rejects_traversal_and_symlink(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    (root / "link.png").symlink_to(outside)
    for path in ["../secret.png", str(outside), "link.png"]:
        with pytest.raises(ValueError, match="inside"):
            resolve_media_path(path, root)
    with pytest.raises((ValueError, OSError)):
        resolve_media_path("https://example.com/a.png", root)


def test_no_silent_frame_discard(tmp_path):
    (tmp_path / "frame.png").write_bytes(b"fixture")
    req = request()
    req["window"]["frames"].append({**req["window"]["frames"][0], "evidence_id": "f2"})
    with pytest.raises(ValueError, match="max_frames"):
        prepare_request(req, ModelConfig(dataset_root=str(tmp_path), max_frames=1))


def test_lora_targets_only_language_projection():
    names = ["model.language_model.layers.0.self_attn.q_proj",
             "model.vision_tower.layers.0.q_proj", "model.audio_tower.layers.0.q_proj",
             "model.language_model.embed_tokens", "model.language_model.layers.1.mlp.up_proj"]
    assert language_lora_targets(names) == [names[0], names[4]]
    with pytest.raises(ValueError, match="no Gemma4"):
        language_lora_targets([names[1]])


def test_recording_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint([{"id": "a", "group_id": "scene1"}], [{"id": "b", "group_id": "scene1"}])
    with pytest.raises(ValueError, match="group_id"):
        ensure_disjoint([{"id": "a"}], [{"id": "b", "group_id": "scene2"}])


def test_bind_and_token_guard():
    validate_bind("127.0.0.1", None)
    validate_bind("::1", None)
    for host in ["0.0.0.0", "::", "example.com", "192.168.1.2"]:
        with pytest.raises(ValueError, match="TOKEN"):
            validate_bind(host, None)
    validate_bind("0.0.0.0", "secret")
    assert authorized("Bearer secret", "secret")
    assert not authorized("Bearer wrong", "secret")
    assert not authorized(None, "secret")


def test_api_guards_health_and_observe(tmp_path):
    from fastapi.testclient import TestClient
    (tmp_path / "frame.png").write_bytes(b"fixture")
    class Backend:
        def observe(self, req):
            return {"decision": {"summary": "No changes.", "observations": [], "actions": [], "noop": True},
                    "metrics": {"fixture_only": True}}
    app = create_app(ModelConfig(dataset_root=str(tmp_path)), backend=Backend(),
                     host="0.0.0.0", api_token="token")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        assert client.post("/observe", json=request()).status_code == 401
        headers = {"Authorization": "Bearer token"}
        assert client.get("/health", headers=headers).status_code == 200
        reply = client.post("/observe", headers=headers, json=request())
        assert reply.status_code == 200, reply.text
        assert reply.json()["metrics"]["fixture_only"]
        assert client.post("/observe", headers=headers, json=request("../secret")).status_code == 422


def test_summary_only_target_never_invents_action_labels():
    from home_observer.train import training_target
    row = {"target": {"summary": "A chair beside a window."},
           "supervision_mask": {"summary": True, "actions": False, "observations": False, "noop": False}}
    text, partial = training_target(row)
    assert partial and json.loads(text) == row["target"]
    assert "actions" not in text and "noop" not in text
    with pytest.raises(ValueError, match="no other target"):
        training_target({**row, "target": {"summary": "A chair.", "noop": True}})


def test_actions_first_changes_order_only_and_accepts_explicit_full_mask():
    from home_observer.train import training_target
    row = {"target": {"summary": "No change.", "observations": [], "actions": [], "noop": True},
           "supervision_mask": {"summary": True, "observations": True, "actions": True, "noop": True}}
    standard, _ = training_target(row)
    reordered, partial = training_target(row, "actions-first")
    assert not partial and json.loads(standard) == json.loads(reordered)
    assert reordered.startswith('{"actions":')
    assert standard.startswith('{"summary":')

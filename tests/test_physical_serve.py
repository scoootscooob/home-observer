import base64
import copy
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_physical_schema import physical_window

from home_observer.physical_remote import PhysicalRemoteModel
from home_observer.physical_serve import constrained_physical_schema, create_physical_app

TOKEN = "physical-test-token-123456789"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def jpeg():
    content = io.BytesIO()
    Image.new("RGB", (24, 24), "blue").save(content, format="JPEG")
    return content.getvalue()


def envelope():
    return {"request": {"window": physical_window()}, "media": {
        name: {"content_base64": base64.b64encode(jpeg()).decode(), "media_type": "image/jpeg"}
        for name in ("before", "after")}}


class FakeNative:
    def __init__(self, root, raw=None):
        self.root, self.seen = root, []
        self.raw = raw or json.dumps({"summary": "No confident physical claims.", "objects": [],
                                      "observations": [], "events": []})
    def generate_text(self, request):
        for clip in request["window"]["clips"]:
            for frame in clip["frames"]:
                path = Path(frame["path"])
                assert path.is_relative_to(self.root)
                assert path.read_bytes() == jpeg()
                self.seen.append(path)
        return self.raw, {"latency_s": 0.25, "task": "physical_perception"}


def test_authenticated_upload_ignores_client_paths_and_cleans_files(tmp_path):
    backend = FakeNative(tmp_path)
    client = TestClient(create_physical_app(backend, api_token=TOKEN, media_root=tmp_path))
    assert client.get("/health").status_code == 401
    assert client.post("/observe", json=envelope()).status_code == 401
    assert client.get("/health", headers=HEADERS).json()["task"] == "physical_perception"
    request = envelope()
    request["request"]["window"]["clips"][0]["frames"][0]["path"] = "/etc/passwd"
    response = client.post("/observe", json=request, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["decision"]["objects"] == []
    assert backend.seen
    assert all(not path.exists() for path in backend.seen)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    lambda e: e["media"].pop("before"),
    lambda e: e["media"]["before"].update(content_base64="not base64"),
    lambda e: e["request"].update(policy={"entities": ["light.kitchen"]}),
    lambda e: e["request"].update(target={"events": []}),
    lambda e: e.update(actions=[]),
])
def test_transport_rejects_missing_media_and_digital_or_label_context(tmp_path, mutation):
    backend = FakeNative(tmp_path)
    client = TestClient(create_physical_app(backend, api_token=TOKEN, media_root=tmp_path))
    request = copy.deepcopy(envelope())
    mutation(request)
    response = client.post("/observe", json=request, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["error"]
    assert backend.seen == []
    assert list(tmp_path.iterdir()) == []


def test_invalid_generated_output_is_retained_and_remote_client_uploads_media(tmp_path):
    source, server = tmp_path / "client", tmp_path / "server"
    source.mkdir()
    for name in ("before.jpg", "after.jpg"):
        (source / name).write_bytes(jpeg())
    backend = FakeNative(server, raw='{"actions": []}')
    client = TestClient(create_physical_app(backend, api_token=TOKEN, media_root=server))
    remote = PhysicalRemoteModel("http://testserver", TOKEN, dataset_root=source, client=client)
    result = remote.observe({"window": physical_window()})
    assert result["decision"] is None
    assert result["raw_output"] == '{"actions": []}'
    assert result["metrics"]["latency_s"] == .25
    assert result["error"]
    assert all(not path.exists() for path in backend.seen)


def test_body_budget_and_schema_constraints_keep_labels_free(tmp_path):
    client = TestClient(create_physical_app(FakeNative(tmp_path), api_token=TOKEN, media_root=tmp_path,
                                          max_body_bytes=128, max_file_bytes=64))
    assert client.post("/observe", content=b"x" * 129, headers=HEADERS).status_code == 413
    schema = constrained_physical_schema(physical_window(), ["phys_bowl"])
    definitions = schema["$defs"]
    assert "enum" not in definitions["PhysicalObject"]["properties"]["label"]
    assert "enum" not in definitions["TemporalEvent"]["properties"]["kind"]
    assert definitions["PhysicalObject"]["properties"]["frame_evidence_id"]["enum"] == ["after", "before"]
    assert definitions["PhysicalObject"]["properties"]["track_id"]["anyOf"][0]["enum"] == ["phys_bowl"]
    empty = constrained_physical_schema({"window_id": "empty", "started_at": 0., "ended_at": 0.})
    assert empty["properties"]["objects"]["maxItems"] == 0

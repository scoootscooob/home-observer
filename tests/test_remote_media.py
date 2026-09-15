import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from home_observer.model import ModelConfig
from home_observer.serve import create_app
from home_observer.transport import pack_media, unpack_media


def request(path):
    return {'window': {'window_id': 'local-capture', 'started_at': 1, 'ended_at': 2,
        'frames': [{'camera_id': 'kitchen', 'timestamp': 2, 'path': str(path), 'evidence_id': 'f1'}],
        'audio': [], 'device_states': {}}, 'state': {}, 'recent_events': [], 'policy': {}}


def test_remote_capture_bytes_arrive_and_temporary_files_are_removed(tmp_path):
    source = tmp_path / 'mac-frame.jpg'
    source.write_bytes(b'actual image bytes')
    server = tmp_path / 'server-data'
    server.mkdir()
    seen = []
    class Backend:
        def observe(self, req):
            path = Path(req['window']['frames'][0]['path'])
            assert path.is_relative_to(server)
            assert path.read_bytes() == b'actual image bytes'
            seen.append(path)
            return {'decision': {'summary': 'Transport verified', 'noop': True, 'actions': [], 'observations': []}}
    payload, size = pack_media(request(source))
    app = create_app(ModelConfig(dataset_root=str(server)), backend=Backend(), api_token='token')
    with TestClient(app) as client:
        assert client.post('/observe_upload', json=payload).status_code == 401
        response = client.post('/observe_upload', json=payload, headers={'Authorization': 'Bearer token'})
        assert response.status_code == 200, response.text
        assert response.json()['metrics']['received_media_bytes'] == size
    assert seen and all(not path.exists() for path in seen)
    assert list(server.iterdir()) == []
    assert payload['request']['window']['frames'][0]['path'] == str(source)


def test_upload_does_not_write_supplied_paths_or_accept_extra_media(tmp_path):
    payload = {'request': request('../../outside.jpg'), 'media': [
        {'path': '../../outside.jpg', 'data': base64.b64encode(b'content').decode()}]}
    unpacked, _ = unpack_media(payload, tmp_path)
    assert Path(unpacked['window']['frames'][0]['path']).parent == tmp_path
    payload['media'].append({'path': 'extra.jpg', 'data': ''})
    with pytest.raises(ValueError, match='exactly cover'):
        unpack_media(payload, tmp_path)


def test_bad_upload_is_rejected_and_cleans_partial_files(tmp_path):
    class Backend:
        def observe(self, _):
            raise AssertionError('malformed media must not reach model')
    payload = {'request': request('frame.jpg'), 'media': [{'path': 'frame.jpg', 'data': 'not base64!'}]}
    with TestClient(create_app(ModelConfig(dataset_root=str(tmp_path)), backend=Backend())) as client:
        assert client.post('/observe_upload', json=payload).status_code == 422
        assert client.post('/observe_upload', content=b'{}', headers={'Content-Length': str(25*1024*1024)}).status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_client_rejects_large_media_before_reading(tmp_path, monkeypatch):
    import home_observer.transport as transport
    monkeypatch.setattr(transport, 'MAX_MEDIA_BYTES', 4)
    path = tmp_path / 'too-large.jpg'
    path.write_bytes(b'12345')
    with pytest.raises(ValueError, match='byte budget'):
        pack_media(request(path))


def test_engine_failure_returns_gateway_error_and_releases_upload(tmp_path):
    import httpx
    source = tmp_path / "source.jpg"
    source.write_bytes(b"frame")
    server = tmp_path / "server"
    server.mkdir()
    class Backend:
        closed = False
        def ready(self):
            return True
        def observe(self, _):
            raise httpx.ConnectError("private engine diagnostic")
        def close(self):
            self.closed = True
    backend = Backend()
    payload, _ = pack_media(request(source))
    with TestClient(create_app(ModelConfig(dataset_root=str(server)), backend=backend)) as client:
        response = client.post("/observe_upload", json=payload)
        assert response.status_code == 502
        assert "private" not in response.text
        assert list(server.iterdir()) == []
    assert backend.closed


def test_remote_evaluation_resolves_dataset_relative_files_before_upload(tmp_path):
    import httpx

    from home_observer.engine import RemoteBackend
    media = tmp_path / "frame.jpg"
    media.write_bytes(b"image input")
    seen = []
    def handler(req):
        import json
        payload = json.loads(req.content)
        seen.append(payload)
        assert req.url.path == "/observe_upload"
        assert base64.b64decode(payload["media"][0]["data"]) == b"image input"
        return httpx.Response(200, json={"decision": {"summary": "ok", "observations": [], "actions": [], "noop": True}})
    backend = RemoteBackend("http://localhost", "", upload_media=True, media_root=str(tmp_path))
    backend.client.close()
    backend.client = httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(handler))
    original = request("frame.jpg")
    result = backend.observe(original)
    assert result["metrics"]["uploaded_media_bytes"] == len(b"image input")
    assert original["window"]["frames"][0]["path"] == "frame.jpg"
    assert len(seen) == 1
    outside = tmp_path.parent / (tmp_path.name + "-outside.jpg")
    outside.write_bytes(b"must not be uploaded")
    with pytest.raises(ValueError, match="root|outside|escape"):
        backend.observe(request(str(outside)))
    assert len(seen) == 1
    backend.client.close()

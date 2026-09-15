import base64
import copy
import json
import wave

import httpx
import pytest
from PIL import Image

from home_observer.model import ModelConfig
from home_observer.prompts import SYSTEM_PROMPT
from home_observer.vllm_backend import VLLMModel, decision_schema


@pytest.fixture
def media_request(tmp_path):
    frames = []
    for index in range(4):
        path = tmp_path / f"camera-{index}.png"
        Image.new("RGB", (16, 12), (index * 30, 20, 40)).save(path)
        frames.append({"camera_id": f"cam{index}", "timestamp": 11.0,
                       "path": path.name, "evidence_id": f"frame{index}"})
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    return {"window": {"window_id": "opaque", "started_at": 10.0, "ended_at": 11.0,
                       "frames": frames, "audio": [{"microphone_id": "mic", "started_at": 10.0,
                           "ended_at": 11.0, "path": "audio.wav", "evidence_id": "sound"}],
                       "device_states": {"light.kitchen": "off", "sensor.ambient_lux": 5}},
            "state": {}, "recent_events": [],
            "policy": {"entities": ["room.kitchen", "light.kitchen"], "rules": ["Authored test rule."]}}


def decision():
    return {"summary": "The kitchen is occupied.", "observations": [
        {"entity_id": "room.kitchen", "attribute": "occupied", "value": True,
         "confidence": 0.9, "evidence_ids": ["frame0"]}],
        "actions": [{"domain": "light", "service": "turn_on", "entity_id": "light.kitchen",
                     "data": {}, "reason": "Policy condition met.",
                     "evidence_ids": ["frame0", "device:sensor.ambient_lux", "device:light.kitchen"]}],
        "noop": False}


def completion(value=None, **extra):
    return {"choices": [{"message": {"content": json.dumps(value if value is not None else decision())},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 120, "total_tokens": 820}, **extra}


def backend(tmp_path, handler, **kwargs):
    return VLLMModel(ModelConfig(dataset_root=str(tmp_path), max_new_tokens=333),
                     client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_four_images_and_audio_share_one_greedy_native_call(tmp_path, media_request, monkeypatch):
    monkeypatch.setenv("HOME_OBSERVER_ENGINE_TOKEN", "private-engine-token")
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "http://127.0.0.1:8001/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer private-engine-token"
        body = json.loads(request.content)
        assert body["model"] == "home-observer"
        assert body["temperature"] == 0 and body["max_tokens"] == 333
        assert body["n"] == 1 and body["stream"] is False
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["mm_processor_kwargs"] == {"max_soft_tokens": 140}
        assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        parts = body["messages"][1]["content"]
        assert [p["type"] for p in parts] == ["image_url"] * 4 + ["text", "input_audio"]
        for index, part in enumerate(parts[:4]):
            prefix, data = part["image_url"]["url"].split(",", 1)
            assert prefix == "data:image/png;base64"
            assert base64.b64decode(data) == (tmp_path / f"camera-{index}.png").read_bytes()
        context = json.loads(parts[4]["text"])
        assert context["frames"][0]["evidence_id"] == "frame0"
        assert all("path" not in item for item in context["frames"])
        assert context["policy"] == media_request["policy"]
        assert parts[5]["input_audio"]["format"] == "wav"
        assert base64.b64decode(parts[5]["input_audio"]["data"]) == (tmp_path / "audio.wav").read_bytes()
        schema = body["structured_outputs"]["json"]
        assert set(schema["required"]) == {"summary", "observations", "actions", "noop"}
        for name in ("Action", "Observation"):
            fields = schema["$defs"][name]["properties"]
            assert fields["evidence_ids"]["items"]["enum"] == [
                "device:light.kitchen", "device:sensor.ambient_lux", "frame0", "frame1", "frame2", "frame3", "sound"]
        assert schema["$defs"]["Action"]["properties"]["entity_id"]["enum"] == ["light.kitchen", "room.kitchen"]
        return httpx.Response(200, json=completion())

    original = copy.deepcopy(media_request)
    response = backend(tmp_path, handler).observe(media_request)
    assert len(calls) == 1 and response["decision"] == decision()
    assert media_request == original
    metrics = response["metrics"]
    assert metrics["backend"] == "vllm" and metrics["structured_decode"] is True
    assert metrics["input_tokens"] == 700 and metrics["output_tokens"] == 120
    assert metrics["latency_s"] >= metrics["preprocess_s"] >= 0
    assert metrics["model_revision"] and metrics["model_revision_source"] == "deployment_config"
    assert metrics["gpu_memory_available"] is False and "peak_allocated_gb" not in metrics
    assert "private-engine-token" not in json.dumps(metrics)


@pytest.mark.parametrize("constraints", [True, False])
@pytest.mark.parametrize("bad_field", ["evidence", "entity", "noop", "missing_decision", "missing_data"])
def test_invalid_output_is_rejected_even_without_schema(tmp_path, media_request, constraints, bad_field):
    output = decision()
    if bad_field == "evidence":
        output["actions"][0]["evidence_ids"] = ["future-frame"]
    elif bad_field == "entity":
        output["actions"][0]["entity_id"] = "light.invented"
    elif bad_field == "noop":
        output["noop"] = True
    elif bad_field == "missing_decision":
        del output["observations"]
    else:
        del output["actions"][0]["data"]

    def handler(request):
        assert ("structured_outputs" in json.loads(request.content)) is constraints
        return httpx.Response(200, json=completion(output))

    with pytest.raises(ValueError, match="valid grounded Decision"):
        backend(tmp_path, handler, schema_constraints=constraints).observe(media_request)


def test_schema_with_no_evidence_or_entities_allows_empty_lists_only():
    request = {"window": {"window_id": "empty", "started_at": 0, "ended_at": 1}, "policy": {}}
    schema = decision_schema(request)
    assert schema["properties"]["actions"]["maxItems"] == 0
    assert schema["properties"]["observations"]["maxItems"] == 0
    assert '"enum": []' not in json.dumps(schema)


@pytest.mark.parametrize("failure", ["future", "duration", "rate", "traversal", "frame_limit"])
def test_invalid_media_is_rejected_before_network(tmp_path, media_request, failure):
    calls = []
    model = backend(tmp_path, lambda req: calls.append(req))
    if failure == "future":
        media_request["window"]["frames"][0]["timestamp"] = 12
    elif failure == "duration":
        media_request["window"]["audio"][0]["started_at"] = 10.5
    elif failure == "rate":
        with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\0\0" * 8000)
    elif failure == "traversal":
        media_request["window"]["frames"][0]["path"] = "../outside.png"
    else:
        model.config.max_frames = 3
    with pytest.raises((ValueError, OSError)):
        model.generate_text(media_request)
    assert not calls


def test_http_failure_does_not_retry_or_fallback(tmp_path, media_request):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": "engine loading"})

    with pytest.raises(httpx.HTTPStatusError):
        backend(tmp_path, handler).generate_text(media_request)
    assert len(calls) == 1


def test_health_uses_engine_health_and_does_not_require_completion(tmp_path):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200 if len(paths) == 1 else 503)

    model = backend(tmp_path, handler, base_url="http://localhost:8001")
    assert model.ready() is True
    assert model.ready() is False
    assert paths == ["/health", "/health"]


def test_missing_usage_is_unknown_not_zero_and_input_overflow_rejects(tmp_path, media_request):
    model = backend(tmp_path, lambda req: httpx.Response(200, json=completion(usage=None)))
    _, metrics = model.generate_text(media_request)
    assert metrics["input_tokens"] is None and metrics["output_tokens"] is None
    model.config.max_input_tokens = 600
    model.client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=completion())))
    with pytest.raises(ValueError, match="input token budget"):
        model.generate_text(media_request)


def test_structured_actions_match_executor_arguments_and_policy_surface(media_request):
    media_request['policy'].update(allowed_services=['light.turn_on'], max_actions_per_window=1)
    schema = decision_schema(media_request)
    fields = schema['$defs']['Action']['properties']
    assert fields['entity_id']['enum'] == ['light.kitchen']
    assert fields['domain']['enum'] == ['light']
    assert fields['service']['enum'] == ['turn_on']
    assert fields['data']['additionalProperties'] is False
    assert fields['data']['properties'] == {}
    enabled = decision_schema(media_request, allow_brightness=True, decision_field_order='actions-first')
    assert enabled['$defs']['Action']['properties']['data']['properties'] == {'brightness': {'type': 'integer', 'minimum': 1, 'maximum': 255}}
    assert list(enabled['properties']) == ['actions', 'observations', 'noop', 'summary']
    assert schema['properties']['actions']['maxItems'] == 1
    media_request['policy']['allowed_services'] = []
    assert decision_schema(media_request)['properties']['actions']['maxItems'] == 0


def test_structured_noop_cannot_contain_observations_or_actions(media_request):
    schema = decision_schema(media_request)
    active, quiet = schema['anyOf']
    assert active['properties']['noop']['const'] is False
    assert quiet['properties']['noop']['const'] is True
    assert quiet['properties']['observations']['maxItems'] == 0
    assert quiet['properties']['actions']['maxItems'] == 0
    assert set(quiet['required']) == {'summary', 'observations', 'actions', 'noop'}
    assert quiet['additionalProperties'] is False

import copy
import json

import httpx
import pytest
from PIL import Image
from test_physical_schema import physical_decision, physical_window

from home_observer.model import ModelConfig
from home_observer.physical_model import PHYSICAL_SYSTEM_PROMPT, physical_context, prepare_physical_request
from home_observer.physical_vllm import PhysicalVLLMModel, bounded_physical_schema


def guided_decision():
    result = physical_decision()
    result["objects"][0]["detection_id"] = "det_1"
    result["events"][0]["subject_ids"] = ["det_1"]
    result["observations"][0]["subject_id"] = "det_1"
    return result


def config(tmp_path):
    for name, color in (("before.jpg", "red"), ("after.jpg", "blue")):
        Image.new("RGB", (12, 12), color).save(tmp_path / name)
    return ModelConfig(dataset_root=str(tmp_path), quantization="none", max_frames=8,
                       max_input_tokens=8192, max_new_tokens=1024)


def test_grammar_orders_events_bounds_lists_keeps_unknown_labels_and_empty_arrays():
    schema = bounded_physical_schema(physical_window())
    assert list(schema["properties"]) == ["events", "objects", "observations", "summary"]
    assert schema["required"] == list(schema["properties"])
    assert list(schema["$defs"]["TemporalEvent"]["properties"])[:5] == [
        "kind", "object_label", "started_at", "ended_at", "description"]
    for name in ("events", "objects", "observations"):
        assert schema["properties"][name]["maxItems"] == 2
        assert schema["properties"][name].get("minItems", 0) == 0
    for definition, field in ((schema["$defs"]["TemporalEvent"], "kind"),
                              (schema["$defs"]["PhysicalObject"]["anyOf"][0], "label")):
        assert "enum" not in definition["properties"][field]
        assert "uncertainty" in definition["required"]
    assert schema["$defs"]["PhysicalObject"]["anyOf"][0]["properties"]["track_id"] == {"type": "null", "default": None}
    assert schema["$defs"]["TemporalEvent"]["properties"]["post_evidence_ids"]["items"]["enum"] == [
        "after", "before", "clip-window:clip"]
    assert schema["$defs"]["TemporalEvent"]["properties"]["subject_ids"]["items"]["enum"] == ["det_1", "det_2"]
    assert schema["$defs"]["PhysicalObject"]["anyOf"][0]["properties"]["detection_id"]["enum"] == ["det_1", "det_2"]
    branches = schema["$defs"]["PhysicalObject"]["anyOf"]
    assert len(branches) == 2
    assert branches[0]["properties"]["timestamp"]["const"] == 10.0
    assert branches[1]["properties"]["frame_evidence_id"]["const"] == "after"
    assert branches[1]["properties"]["evidence_ids"]["maxItems"] == 1
    empty = bounded_physical_schema({"window_id": "empty", "started_at": 0., "ended_at": 0.})
    assert all(empty["properties"][k]["maxItems"] == 0 for k in ("events", "objects", "observations"))


def test_exact_prompt_context_order_and_model_alias_are_sent(tmp_path):
    settings = config(tmp_path)
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer private-test-token"
        return httpx.Response(200, json={"model": "physical-base", "choices": [{
            "message": {"content": json.dumps(guided_decision())}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = PhysicalVLLMModel(settings, served_model="physical-base", api_token="private-test-token", client=client)
    request = {"window": physical_window()}
    result = model.observe(request)
    payload = seen[0]
    assert payload["messages"][0]["content"] == PHYSICAL_SYSTEM_PROMPT
    content = payload["messages"][1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[-1]["text"] == physical_context(prepare_physical_request(request, settings))
    assert payload["model"] == "physical-base"
    assert payload["structured_outputs"]["json"] == bounded_physical_schema(physical_window())
    assert result["decision"]["objects"][0]["label"] == "mixing bowl"
    assert result["metrics"]["sampled_frames"] == 2
    assert result["metrics"]["frame_evidence_ids"] == ["before", "after"]
    assert result["metrics"]["engine_reported_model"] == "physical-base"
    assert "private-test-token" not in json.dumps(model.backend_config)
    model.close()
    assert not client.is_closed
    client.close()


@pytest.mark.parametrize("case", ["actions", "evidence", "alias", "usage", "missing", "list-cap"])
def test_invalid_result_is_retained_without_repair(tmp_path, case):
    settings = config(tmp_path)
    decision = copy.deepcopy(guided_decision())
    result = {"model": "physical-observer", "usage": {"prompt_tokens": 100}}
    if case == "actions":
        decision["actions"] = []
    elif case == "evidence":
        decision["events"][0]["post_evidence_ids"] = ["unknown"]
    elif case == "alias":
        result["model"] = "unexpected-base"
    elif case == "usage":
        result["usage"]["prompt_tokens"] = 9000
    elif case == "missing":
        del decision["observations"]
    elif case == "list-cap":
        decision["objects"] = [dict(decision["objects"][0], detection_id=f"local-{n}") for n in range(3)]
    raw = json.dumps(decision)
    result["choices"] = [{"message": {"content": raw}, "finish_reason": "stop"}]
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=result))) as client:
        model = PhysicalVLLMModel(settings, client=client)
        with pytest.raises(ValueError):
            model.observe({"window": physical_window()})
        assert model.last_generation["raw_output"] == raw
        assert model.last_generation["metrics"]["task"] == "physical_perception"


def test_inputs_reject_labels_and_ha_policy_before_http(tmp_path):
    settings = config(tmp_path)
    def forbidden(request):
        raise AssertionError("Invalid model input reached HTTP")
    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        model = PhysicalVLLMModel(settings, client=client)
        for extra in ({"target": {"summary": "take knife"}}, {"policy": {"entities": ["light.kitchen"]}}):
            with pytest.raises(ValueError):
                model.generate_text({"window": physical_window(), **extra})

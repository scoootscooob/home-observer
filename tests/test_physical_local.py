"""Judged-fields completion: software owns identifiers, citations and placeholders; never visual claims."""
import json

import pytest

from home_observer.physical_local import PLACEHOLDER_CONFIDENCE, complete_judged_decision
from home_observer.physical_schema import validate_physical_decision

WINDOW = {"window_id": "w", "started_at": 10.0, "ended_at": 13.0, "clips": [{
    "clip_id": "c", "camera_id": "cam", "started_at": 10.0, "ended_at": 13.0,
    "frames": [{"evidence_id": f"cam:{i}", "timestamp": 10.0 + i * 0.5, "path": f"{i}.jpg"} for i in range(7)]}]}


def test_partial_adapter_output_is_completed_with_recorded_placeholders():
    raw = json.dumps({"events": [{"kind": "pick-up", "object_label": "bowl", "started_at": 11.0, "ended_at": 12.4,
                                  "description": "pick up bowl", "pre_evidence_ids": ["cam:5", "cam:6"],
                                  "post_evidence_ids": ["cam:6"]}]})
    decision, completion = complete_judged_decision(raw, WINDOW)
    validate_physical_decision(decision, WINDOW)
    event = decision.events[0]
    assert event.pre_evidence_ids == ["cam:2"] and event.post_evidence_ids == ["cam:4"]
    assert event.confidence == PLACEHOLDER_CONFIDENCE and "placeholder" in event.uncertainty
    assert completion["events"][0]["ignored_model_citations"] == 3
    assert completion["events"][0]["evidence"] == "derived_from_interval"
    assert completion["summary"] == "placeholder" and decision.summary == "pick up bowl"


def test_echoed_identifiers_and_unknown_keys_are_ignored_and_counted():
    raw = json.dumps({"summary": "s", "objects": [{"label": "bowl"}], "events": [{
        "event_id": "pevt_" + "a" * 32, "kind": "put-down", "object_label": "bowl", "started_at": 10.5,
        "ended_at": 12.0, "description": "put down bowl", "subject_ids": [], "actions": ["light.turn_on"]}]})
    decision, completion = complete_judged_decision(raw, WINDOW)
    validate_physical_decision(decision, WINDOW)
    assert decision.events[0].event_id == "judged_1"
    assert decision.objects == [] and decision.observations == []
    assert completion["ignored_top_level_fields"] == ["objects"]
    assert completion["events"][0]["ignored_event_fields"] == ["actions", "event_id"]
    assert not hasattr(decision, "actions")


def test_visual_claims_are_never_invented_or_repaired():
    with pytest.raises(ValueError):
        complete_judged_decision(json.dumps({"events": [{"kind": "", "started_at": 11, "ended_at": 12}]}), WINDOW)
    with pytest.raises(ValueError):
        complete_judged_decision("not json", WINDOW)
    decision, _ = complete_judged_decision(json.dumps({"events": []}), WINDOW)
    assert decision.events == [] and decision.summary == "no event reported"
    # An interval before any sampled frame has no derivable citation and is rejected, not repaired.
    with pytest.raises(ValueError):
        complete_judged_decision(json.dumps({"events": [{"kind": "take", "object_label": "cup",
                                                          "started_at": 9.0, "ended_at": 9.5}]}), WINDOW)


def test_trailing_characters_and_fences_are_format_noise():
    raw = '```json\n{"events": [{"kind": "pick-up", "object_label": "bowl", "started_at": 11.0, "ended_at": 12.4, "description": "pick up bowl"}]}}\n```'
    decision, completion = complete_judged_decision(raw, WINDOW)
    validate_physical_decision(decision, WINDOW)
    assert decision.events[0].kind == "pick-up"
    assert completion["ignored_trailing_characters"] == 1
    with pytest.raises(ValueError):
        complete_judged_decision("no object here", WINDOW)

import copy

import pytest

from home_observer.physical_schema import PhysicalDecision, PhysicalWindow, validate_physical_decision


def physical_window(identifier="clip-window", start=10.0):
    return {"window_id": identifier, "started_at": start, "ended_at": start + 1,
            "clips": [{"clip_id": identifier + ":clip", "camera_id": "kitchen-camera",
                       "started_at": start, "ended_at": start + 1,
                       "frames": [{"evidence_id": "before", "timestamp": start, "path": "before.jpg"},
                                  {"evidence_id": "after", "timestamp": start + 1, "path": "after.jpg"}]}]}


def physical_decision(start=10.0):
    return {"summary": "A bowl was put on the counter.", "objects": [{
        "detection_id": "bowl-1", "label": "mixing bowl", "timestamp": start + 1,
        "box": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6},
        "location": {"camera_id": "kitchen-camera", "area": "counter"}, "frame_evidence_id": "after",
        "visibility": "partially_occluded", "confidence": 0.8,
        "uncertainty": "The hand obscures part of the rim.", "evidence_ids": ["after"]}],
        "observations": [{"observation_id": "position", "subject_id": "bowl-1", "attribute": "support",
                          "value": "counter", "observed_at": start + 1, "confidence": 0.8,
                          "uncertainty": "The bottom is partially obscured.", "evidence_ids": ["after"]}],
        "events": [{"event_id": "placement", "kind": "put", "object_label": "bowl", "subject_ids": ["bowl-1"],
                    "started_at": start, "ended_at": start + 1, "description": "A bowl was put on the counter.",
                    "confidence": 0.8, "uncertainty": "Two sampled frames bound the transition.",
                    "pre_evidence_ids": ["before"], "post_evidence_ids": ["after"]}]}


def test_free_form_physical_objects_need_no_ha_allowlist():
    result = validate_physical_decision(physical_decision(), physical_window())
    assert result.objects[0].label == "mixing bowl"
    assert result.events[0].kind == "put"
    assert result.events[0].event_type == "put"
    assert result.events[0].evidence_ids == ["before", "after"]
    assert "actions" not in PhysicalDecision.model_json_schema()["properties"]
    bad = {**physical_decision(), "actions": []}
    with pytest.raises(ValueError):
        PhysicalDecision.model_validate(bad)


@pytest.mark.parametrize("mutation", [
    lambda d: d["objects"][0].update(confidence=1.1),
    lambda d: d["objects"][0].update(confidence=float("nan")),
    lambda d: d["objects"][0].update(uncertainty=""),
    lambda d: d["objects"][0]["box"].update(x_max=0.05),
    lambda d: d["objects"][0]["box"].update(y_min=-0.1),
    lambda d: d["objects"][0].update(frame_evidence_id="before"),
    lambda d: d["objects"][0]["location"].update(camera_id="different-camera"),
    lambda d: d["objects"][0].update(timestamp=12.0),
    lambda d: d["objects"][0].update(track_id="light.kitchen"),
    lambda d: d["objects"][0].update(track_id="phys_unknown"),
    lambda d: d["observations"][0].update(evidence_ids=["invented"]),
    lambda d: d["observations"][0].update(observed_at=10.0),
    lambda d: d["events"][0].update(pre_evidence_ids=["after"]),
    lambda d: d["events"][0].update(subject_ids=["unknown-local-id"]),
])
def test_invalid_geometry_confidence_evidence_and_causality(mutation):
    bad = copy.deepcopy(physical_decision())
    mutation(bad)
    with pytest.raises(ValueError):
        validate_physical_decision(bad, physical_window())


def test_clips_must_be_ordered_bounded_and_evidence_unique():
    for change in ("reverse", "duplicate", "future"):
        bad = physical_window()
        frames = bad["clips"][0]["frames"]
        if change == "reverse":
            frames.reverse()
        elif change == "duplicate":
            frames[1]["evidence_id"] = frames[0]["evidence_id"]
        else:
            frames[1]["timestamp"] = 12.0
        with pytest.raises(ValueError):
            PhysicalWindow.model_validate(bad)


def test_untracked_event_can_retain_noun_without_inventing_identity():
    data = physical_decision()
    data["objects"] = []
    data["observations"] = []
    data["events"][0]["subject_ids"] = []
    assert validate_physical_decision(data, physical_window()).events[0].object_label == "bowl"

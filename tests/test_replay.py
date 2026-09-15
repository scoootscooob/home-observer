import copy
import json

import pytest

from home_observer.replay import iter_replay, read_jsonl, resolve_window, validate_window


def window():
    return {"window_id": "x", "started_at": 10, "ended_at": 20,
            "frames": [{"camera_id": "entry", "timestamp": 19, "path": "frame.png", "evidence_id": "f1"}],
            "audio": [{"microphone_id": "home", "started_at": 12, "ended_at": 20,
                       "path": "sound.wav", "evidence_id": "a1"}], "device_states": {}}


@pytest.mark.parametrize("fault", ["future_frame", "past_frame", "future_audio", "duplicate", "long", "nan"])
def test_rejects_noncausal_and_unbounded_evidence(fault):
    item = window()
    if fault == "future_frame":
        item["frames"][0]["timestamp"] = 21
    if fault == "past_frame":
        item["frames"][0]["timestamp"] = 9
    if fault == "future_audio":
        item["audio"][0]["ended_at"] = 21
    if fault == "duplicate":
        item["audio"][0]["evidence_id"] = "f1"
    if fault == "long":
        item["ended_at"] = 400
    if fault == "nan":
        item["started_at"] = float("nan")
    with pytest.raises(ValueError):
        validate_window(item)


def test_replay_never_passes_labels_provenance_or_policy_to_model(tmp_path):
    (tmp_path / "frame.png").write_bytes(b"png")
    (tmp_path / "sound.wav").write_bytes(b"wav")
    item = window()
    row = {"window": item, "target": {"answer": "SECRET"}, "teacher": "SECRET", "policy": "SECRET", "group_id": "r"}
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(row) + "\n")
    result = list(iter_replay(source))[0]
    assert result.keys() == item.keys()
    assert "SECRET" not in json.dumps(result)
    assert result["frames"][0]["path"] == str(tmp_path / "frame.png")
    assert item["frames"][0]["path"] == "frame.png"


def test_path_escape_missing_media_duplicate_and_reversed_clock(tmp_path):
    item = window()
    item["frames"][0]["path"] = "../escape.png"
    with pytest.raises(ValueError, match="escapes"):
        resolve_window(item, tmp_path, check_files=False)
    with pytest.raises(FileNotFoundError):
        resolve_window(window(), tmp_path)
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(window()) + "\n" + json.dumps(window()) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        list(iter_replay(source, check_files=False))
    earlier = copy.deepcopy(window())
    earlier["window_id"] = "earlier"
    earlier["ended_at"] = 19
    earlier["audio"][0]["ended_at"] = 19
    source.write_text(json.dumps(window()) + "\n" + json.dumps(earlier) + "\n")
    with pytest.raises(ValueError, match="non-monotonic"):
        list(iter_replay(source, check_files=False))


def test_invalid_json_reports_line(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text("\n{bad}\n")
    with pytest.raises(ValueError, match=":2:"):
        list(read_jsonl(source))

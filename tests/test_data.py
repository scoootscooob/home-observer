import json
import wave

import pytest

from home_observer.data import (
    CAMERAS,
    _coco_download_url,
    authored_decision,
    ffmpeg_executable,
    generate_counterfactual_pairs,
    generate_fixtures,
    group_split,
    prepare_recording,
    sha256_file,
    summary_supervised_row,
)
from home_observer.replay import assert_disjoint_splits, iter_replay, read_jsonl


def test_fixtures_are_causal_group_disjoint_and_reproducible(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    manifest = generate_fixtures(first, families_per_split=1, variants=1)
    generate_fixtures(second, families_per_split=1, variants=1)
    assert manifest["counts"] == {"train": 8, "validation": 8, "test": 8}
    assert sha256_file(first / "all.jsonl") == sha256_file(second / "all.jsonl")
    rows = list(read_jsonl(first / "all.jsonl"))
    assert_disjoint_splits(rows)
    windows = list(iter_replay(first / "all.jsonl"))
    assert len(windows) == 24
    assert {x["camera_id"] for x in windows[0]["frames"]} == set(CAMERAS)
    assert len({row["group_id"] for row in rows}) == 3
    for row in rows:
        window = row["window"]
        evidence = {x["evidence_id"] for x in window["frames"] + window["audio"]}
        evidence |= {f"device:{x}" for x in window["device_states"]}
        for item in row["target"]["actions"] + row["target"]["observations"]:
            assert set(item["evidence_ids"]) <= evidence
        if row["target"]["noop"]:
            assert not row["target"]["observations"]
            assert not row["target"]["actions"]
    with wave.open(windows[0]["audio"][0]["path"]) as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1


def test_visual_suite_requires_images_and_does_not_supply_occupancy(tmp_path):
    generate_fixtures(tmp_path, 1, 1)
    rows = list(read_jsonl(tmp_path / "train-visual_occupancy.jsonl"))
    assert rows[0]["window"]["device_states"] == rows[1]["window"]["device_states"]
    assert not rows[0]["target"]["actions"]
    assert len(rows[1]["target"]["actions"]) == 1
    assert rows[0]["target"]["observations"] != rows[1]["target"]["observations"]
    for row in rows:
        assert all("motion" not in key and "occupied" not in key for key in row["window"]["device_states"])
        assert all(obs["attribute"] == "occupied" for obs in row["target"]["observations"])
    policy = json.loads((tmp_path / "policy.json").read_text())
    assert "light.turn_on" in policy["allowed_services"]


def test_group_split_and_leak_rejection():
    assert group_split("a") == group_split("a")
    with pytest.raises(ValueError, match="leaked"):
        assert_disjoint_splits([{"split": "train", "group_id": "family", "recording_id": "1"},
                                {"split": "test", "group_id": "family", "recording_id": "2"}])
    with pytest.raises(ValueError, match="leaked"):
        assert_disjoint_splits([{"split": "train", "group_id": "a", "recording_id": "same"},
                                {"split": "test", "group_id": "b", "recording_id": "same"}])


def test_leak_rule_is_local_buzzer_and_explicit_negative():
    window = {"device_states": {"binary_sensor.water_leak": "on", "switch.buzzer": "off"}}
    result = authored_decision(window)
    assert result["actions"][0]["entity_id"] == "switch.buzzer"
    assert result["actions"][0]["domain"] == "switch"
    assert authored_decision({"device_states": {}})["noop"]


def test_real_media_preparer_preserves_clock_without_inventing_labels(tmp_path):
    import subprocess
    try:
        executable = ffmpeg_executable()
    except RuntimeError:
        pytest.skip("ffmpeg optional dependency unavailable")
    video = tmp_path / "source.mp4"
    subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=5", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000",
                    "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    rows = prepare_recording(video, tmp_path / "prepared", recording_id="source", duration_s=3,
                             window_s=1.5, frame_step_s=0.75, start_time=1000,
                             provenance={"kind": "public_real", "group_id": "whole-recording"})
    assert len(rows) == 2
    assert all("target" not in row and "noop" not in row for row in rows)
    assert rows[1]["window"]["started_at"] == 1001.5
    assert rows[1]["window"]["ended_at"] == 1003
    assert len(list(iter_replay(tmp_path / "prepared" / "replay.jsonl"))) == 2
    assert_disjoint_splits(rows)


def test_public_summary_supervision_never_invents_decision_fields():
    row = summary_supervised_row(image_id=1, image_path="assets/1.jpg", caption="A person sits on a couch.",
                                 split="train", provenance={"kind": "public_real"})
    assert row["target"] == {"summary": "A person sits on a couch."}
    assert row["supervision_mask"] == {"summary": True, "actions": False, "observations": False, "noop": False}
    assert row["window"]["device_states"] == {}
    assert row["window"]["audio"] == []
    assert "caption" not in json.dumps(row["window"])
    assert row["group_id"] == row["recording_id"] == "coco-1"


def test_public_image_download_stays_on_known_coco_source():
    assert _coco_download_url("http://images.cocodataset.org/train2014/a.jpg") == "https://s3.amazonaws.com/images.cocodataset.org/train2014/a.jpg"
    with pytest.raises(ValueError):
        _coco_download_url("https://untrusted.example/train2014/a.jpg")
    with pytest.raises(ValueError):
        _coco_download_url("https://images.cocodataset.org/other/a.jpg")


def test_counterfactual_pairs_have_identical_context_and_audio_but_changed_pixels(tmp_path):
    from home_observer.prompts import build_context
    generate_counterfactual_pairs(tmp_path, pairs_per_room=1)
    rows = list(read_jsonl(tmp_path / "pairs.jsonl"))
    assert len(rows) == 8
    for left, right in zip(rows[::2], rows[1::2]):
        assert left["id"] != right["id"]
        assert left["window"]["window_id"] == right["window"]["window_id"]
        assert build_context({"window": left["window"]}) == build_context({"window": right["window"]})
        assert left["window"]["audio"] == right["window"]["audio"]
        changes = sum(sha256_file(tmp_path / a["path"]) != sha256_file(tmp_path / b["path"])
                      for a, b in zip(left["window"]["frames"], right["window"]["frames"]))
        assert changes == 1
        assert not left["target"]["actions"]
        assert len(right["target"]["actions"]) == 1
        assert left["split"] == right["split"] == "test"


def test_fixture_model_visible_ids_do_not_encode_split_or_step(tmp_path):
    import re
    generate_fixtures(tmp_path, 1, 1)
    rows = list(read_jsonl(tmp_path / "all.jsonl"))
    phases = set()
    for row in rows:
        window = row["window"]
        assert re.fullmatch(r"[a-f0-9]{24}", window["window_id"])
        assert not any(word in json.dumps(window["frames"][0]["evidence_id"])
                       for word in ["train", "validation", "test", "fixture", "step"])
        phases.add(round(window["started_at"] % 10, 1))
    assert len(phases) > 1

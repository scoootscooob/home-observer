import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("annotation_metrics", Path(__file__).parents[1] / "scripts/model_annotation_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def obs(entity, attribute, value):
    return {"entity_id": entity, "attribute": attribute, "value": value, "confidence": 1,
            "evidence_ids": ["frame"]}


def row(expected, actual, **kwargs):
    return {"id": "one", "target": {"observations": expected, "actions": [], "summary": "", "noop": False},
            "result": {"decision": {"observations": actual, "actions": [], "summary": "", "noop": False}}, **kwargs}


def test_known_labels_separate_correct_incorrect_missing_and_extras():
    expected = [obs("room.kitchen", "occupied", True), obs("room.entry", "occupied", False),
                obs("binary_sensor.water_leak", "state", "on")]
    actual = [expected[0], obs("room.entry", "occupied", True), obs("sensor.lux", "state", 10)]
    rows = [row(expected, actual)]
    original = copy.deepcopy(rows)
    report = metrics.known_label_metrics(rows)
    known = report["known_labels"]
    assert (known["annotated_pairs"], known["correct"], known["incorrect"], known["missing"]) == (3, 1, 1, 1)
    assert known["coverage"] == pytest.approx(2 / 3)
    assert known["accuracy_on_covered"] == .5
    assert report["camera_occupancy"]["annotated_pairs"] == 2
    assert report["camera_occupancy"]["coverage"] == 1
    assert report["unscored_extras"]["pair_instances"] == 1
    assert report["unscored_extras"]["by_entity"] == {"sensor.lux": 1}
    assert report["legacy_strict_set_compatibility"]["metrics"]["counts"]["obs_fp"] == 2
    assert rows == original


def test_unlabeled_and_masked_rows_do_not_create_negative_labels():
    item = obs("room.kitchen", "occupied", True)
    masked = row([item], [item], supervision_mask={"summary": True, "observations": False, "actions": False, "noop": False})
    unlabeled = row([], [item])
    del unlabeled["target"]
    report = metrics.known_label_metrics([masked, unlabeled])
    assert report["observation_labeled_windows"] == 0
    assert report["observation_unscored_windows"] == 2
    assert report["known_labels"]["annotated_pairs"] == 0
    assert report["known_labels"]["coverage"] is None
    assert report["known_labels"]["accuracy_on_covered"] is None
    assert report["unscored_extras"]["pair_instances"] == 2


def test_values_are_type_exact_and_conflicting_predictions_are_incorrect():
    expected = obs("room.kitchen", "occupied", True)
    for value in ["true", 1, False]:
        report = metrics.known_label_metrics([row([expected], [obs("room.kitchen", "occupied", value)])])
        assert report["known_labels"]["incorrect"] == 1
    report = metrics.known_label_metrics([row([expected], [expected, expected])])
    assert report["known_labels"]["correct"] == 1
    report = metrics.known_label_metrics([row([expected], [expected, obs("room.kitchen", "occupied", False)])])
    assert report["known_labels"]["incorrect"] == 1


def test_failure_is_retained_separately_and_no_raw_output_is_repaired():
    item = obs("room.kitchen", "occupied", True)
    parsed = row([item], [item])
    parsed["result"]["rejections"] = ["action entity not configured"]
    failed = row([item], [])
    del failed["result"]["decision"]
    failed["result"]["error"] = "invalid JSON"
    failed["raw_output"] = '{"observations": "do not parse me"}'
    report = metrics.known_label_metrics([parsed, failed])
    assert report["original_invalid_or_failed_windows"] == 2
    assert report["known_labels"]["correct"] == 1
    assert report["known_labels"]["missing"] == 1
    assert report["legacy_strict_set_compatibility"]["metrics"]["counts"]["obs_tp"] == 0


def test_conflicting_labels_and_partial_files_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="conflicting source"):
        metrics.known_label_metrics([row([obs("room.kitchen", "occupied", True),
                                         obs("room.kitchen", "occupied", False)], [])])
    source = tmp_path / "predictions.jsonl"
    source.write_text('{"id":"truncated"')
    with pytest.raises(ValueError, match="incomplete or invalid"):
        metrics.read_predictions(source)


def test_release_cli_requires_all_manifest_rows_before_complete(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"suites": [
        {"name": "heldout", "selected_rows": 1}, {"name": "public_video", "selected_rows": 1}]}))
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    item = obs("room.kitchen", "occupied", True)
    source = heldout / "predictions.jsonl"
    raw = json.dumps(row([item], [item])) + "\n"
    source.write_text(raw)
    monkeypatch.setattr(sys, "argv", ["annotation_metrics", "--release-root", str(tmp_path)])
    metrics.main()
    report = json.loads((tmp_path / "annotation-metrics.json").read_text())
    assert report["complete"] is False
    assert report["suites"]["heldout"]["complete"] is True
    public = tmp_path / "public_video"
    public.mkdir()
    unscored = row([], [])
    del unscored["target"]
    (public / "predictions.jsonl").write_text(json.dumps(unscored) + "\n")
    metrics.main()
    report = json.loads((tmp_path / "annotation-metrics.json").read_text())
    assert report["complete"] is True
    assert report["windows"] == 2 and report["known_labels"]["annotated_pairs"] == 1
    assert source.read_text() == raw

import copy
import importlib.util
from pathlib import Path

import pytest

from home_observer.physical_evaluate import box_iou, score_predictions, temporal_iou


def event(kind="put-down", noun="bowl", start=1.0, end=2.0, **extra):
    return {"kind": kind, "object_label": noun, "started_at": start, "ended_at": end, **extra}


def row(target=None, predicted=None, exhaustive=False):
    return {"id": "one", "group_id": "recording", "window": {"started_at": 0.0, "ended_at": 10.0},
            "target": {"events": target if target is not None else [event()]},
            "supervision_mask": {"events": {"kind": True, "object_label": True, "started_at": True,
                                             "ended_at": True}},
            "annotation_coverage": {"exhaustive": exhaustive, "covered_intervals": [[0.0, 10.0]],
                                    "evaluated_event_classes": ["put-down"]},
            "result": {"decision": {"events": predicted if predicted is not None else [event()]},
                       "metrics": {"latency_s": 0.8}}}


def test_known_positive_and_unannotated_extra_are_not_false_events():
    report = score_predictions([row(predicted=[event(confidence=0.8), event(noun="cup", confidence=0.9)])])
    assert report["events"]["labeled_event_recall"] == 1
    assert report["events"]["unscored_extras"] == 1
    assert report["events"]["false_events_per_hour"] is None
    assert report["event_confidence_calibration"]["scorable_predictions"] == 1
    assert report["event_confidence_calibration"]["positive_only"]
    assert report["event_confidence_calibration"]["brier_known_labels"] == pytest.approx(0.04)


def test_exhaustive_negative_provides_false_event_rate_and_calibration():
    report = score_predictions([row(target=[], predicted=[event(confidence=0.7)], exhaustive=True)])
    assert report["events"]["false_with_exhaustive_coverage"] == 1
    assert report["events"]["false_events_per_hour"] == 360
    assert report["event_confidence_calibration"]["brier_known_labels"] == pytest.approx(0.49)
    assert not report["event_confidence_calibration"]["positive_only"]


def test_uncovered_class_and_partial_interval_remain_unscored():
    result = row(target=[], predicted=[event(kind="open"), event(start=9.0, end=11.0)], exhaustive=True)
    report = score_predictions([result])
    assert report["events"]["unscored_extras"] == 2
    assert report["events"]["false_with_exhaustive_coverage"] == 0


def test_overlapping_coverage_duration_deduplicates_per_recording():
    one = row(target=[], predicted=[], exhaustive=True)
    two = row(target=[], predicted=[event(start=8.0, end=9.0)], exhaustive=True)
    one["annotation_coverage"]["covered_intervals"] = [[0.0, 6.0]]
    two["annotation_coverage"]["covered_intervals"] = [[5.0, 10.0]]
    report = score_predictions([one, two])
    assert report["events"]["exhaustive_subset"]["covered_seconds"] == 10
    assert report["events"]["false_events_per_hour"] == 360


def test_coverage_requires_labels_stable_timeline_and_explicit_exhaustive():
    source = row(target=[], predicted=[event()], exhaustive=True)
    del source["group_id"]
    assert score_predictions([source])["events"]["false_events_per_hour"] is None
    del source["target"]
    report = score_predictions([source])
    assert report["events"]["false_with_exhaustive_coverage"] == 0
    assert report["events"]["labeled_event_recall"] is None


def test_subset_rate_excludes_false_events_without_timeline_denominator():
    known = row(target=[], predicted=[], exhaustive=True)
    unknown = row(target=[], predicted=[event()], exhaustive=True)
    unknown.pop("group_id")
    report = score_predictions([known, unknown])
    assert report["events"]["false_events_per_hour"] is None
    assert report["events"]["false_with_exhaustive_coverage"] == 1
    assert report["events"]["exhaustive_subset"]["false_events_per_hour"] == 0


def test_kind_noun_and_time_are_all_required():
    report = score_predictions([row(predicted=[event(noun="cup"), event(kind="take"),
                                               event(start=3.0, end=4.0)])])
    assert report["events"]["missing"] == 1
    assert report["events"]["matched"] == 0


def test_maximum_cardinality_prevents_greedy_match_loss():
    # First GT matches either prediction; second GT matches only the first.
    report = score_predictions([row(target=[event(start=0.0, end=2.0), event(start=1.0, end=3.0)],
                                     predicted=[event(start=0.5, end=2.5), event(start=0.0, end=1.0)])],
                               temporal_iou_threshold=0.3)
    assert report["events"]["matched"] == 2


def test_unlabeled_mask_fields_do_not_become_negative_labels():
    source = row(predicted=[event()])
    source["supervision_mask"] = {"summary": True}
    report = score_predictions([source])
    assert report["events"]["labeled"] == 0
    assert report["events"]["unscored_extras"] == 1
    assert report["tracking"]["annotated_box_recall"] is None


def test_raw_invalid_decisions_are_not_repaired_for_scoring():
    source = row()
    source["result"]["error"] = "invalid evidence citation"
    source["raw_output"] = '{"events":[]}'
    report = score_predictions([source])
    assert report["usable_decisions"] == 0
    assert report["events"]["missing"] == 1


def test_clock_mapping_is_required_and_latency_is_separate():
    source = row(predicted=[event(event_id="e")])
    source["emitted_at"] = 103.0
    before = score_predictions([source])
    assert before["timing"]["event_start_to_emit_delay_s"]["mean"] is None
    assert before["timing"]["model_latency_s"]["mean"] == 0.8
    source["clock_mapping"] = {"source_origin": 0.0, "wall_origin": 100.0, "scale": 1.0}
    after = score_predictions([source])
    assert after["timing"]["event_start_to_emit_delay_s"]["mean"] == 2.0
    assert after["timing"]["event_end_to_emit_delay_s"]["mean"] == 1.0


def obj(timestamp, frame, track):
    return {"label": "bowl", "track_id": track, "timestamp": timestamp, "frame_evidence_id": frame,
            "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}}


def test_supplied_boxes_and_ids_enable_only_documented_tracking_metrics():
    source = row()
    source.pop("supervision_mask")
    source["target"]["objects"] = [obj(1.0, "f1", "gt_a"), obj(2.0, "f2", "gt_a"), obj(3.0, "f3", "gt_a")]
    source["result"]["decision"]["objects"] = [obj(1.0, "f1", "p_a"), obj(2.0, "f2", "p_b")]
    report = score_predictions([source])
    assert report["tracking"]["annotated_box_recall"] == pytest.approx(2 / 3)
    assert report["tracking"]["mean_matched_iou"] == 1
    assert report["tracking"]["identity_switches"] == 1
    assert report["tracking"]["idf1"] is None


def test_boxes_must_match_exact_frame_not_just_label():
    source = row()
    source.pop("supervision_mask")
    source["target"]["objects"] = [obj(1.0, "f1", "gt")]
    source["result"]["decision"]["objects"] = [obj(1.0, "different_camera_frame", "p")]
    assert score_predictions([source])["tracking"]["matched_boxes"] == 0


def test_iou_rejects_invalid_and_normalizes_boxes():
    assert temporal_iou(event(start=0, end=2), event(start=1, end=3)) == pytest.approx(1 / 3)
    assert temporal_iou(event(start=3, end=2), event()) == 0
    assert box_iou({}, {}) == 0
    assert box_iou(obj(1, "f", "x")["box"], obj(1, "f", "x")["box"]) == 1


def test_join_preserves_source_and_rejects_label_mutation_or_duplicate_ids():
    path = Path(__file__).parents[1] / "scripts/evaluate_physical.py"
    spec = importlib.util.spec_from_file_location("evaluate_physical_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = row()
    prediction = {"id": "one", "result": source.pop("result")}
    assert module.join_predictions([source], [prediction])[0]["target"] == source["target"]
    changed = copy.deepcopy(prediction)
    changed["target"] = {"events": []}
    with pytest.raises(ValueError, match="changed source"):
        module.join_predictions([source], [changed])
    with pytest.raises(ValueError, match="unique"):
        module.join_predictions([source], [prediction, prediction])
    assert "Missing prediction" in module.join_predictions([source], [])[0]["result"]["error"]
    stale_source = {**source, "decision": {"events": [event()]}}
    joined = module.join_predictions([stale_source], [{"id": "one"}])
    assert score_predictions(joined)["usable_decisions"] == 0

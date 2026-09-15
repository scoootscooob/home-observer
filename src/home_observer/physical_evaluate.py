"""Ground-truth-aware physical sensing scores; no HA action metrics.

Missing labels never imply absent physical events. Scoring is independent of
strict output validation; a caller can retain metrics for malformed outputs while
reporting the validation failure separately.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _name(value: Any) -> str:
    return " ".join(value.strip().lower().split()) if isinstance(value, str) else ""


def temporal_iou(left: dict, right: dict) -> float:
    values = [left.get("started_at"), left.get("ended_at"), right.get("started_at"), right.get("ended_at")]
    if not all(_number(x) for x in values):
        return 0.0
    a, b, c, d = values
    if b < a or d < c:
        return 0.0
    if a == b == c == d:
        return 1.0
    union = max(b, d) - min(a, c)
    return max(0.0, min(b, d) - max(a, c)) / union if union else 0.0


def box_iou(left: dict, right: dict) -> float:
    try:
        a, b = [[box[k] for k in ("x_min", "y_min", "x_max", "y_max")] for box in (left, right)]
    except (TypeError, KeyError):
        return 0.0
    if not all(_number(x) and 0 <= x <= 1 for x in a + b):
        return 0.0
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def _matching(scores: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    """Maximum-cardinality bipartite matching, deterministic IoU-ordered neighbors.

    This maximizes matched label count, not the sum of IoUs among tied matchings.
    """
    assigned: dict[int, int] = {}

    def visit(target: int, seen: set[int]) -> bool:
        neighbors = sorted(range(len(scores[target])), key=lambda p: (-scores[target][p], p))
        for prediction in neighbors:
            if scores[target][prediction] < threshold or prediction in seen:
                continue
            seen.add(prediction)
            if prediction not in assigned or visit(assigned[prediction], seen):
                assigned[prediction] = target
                return True
        return False

    for target in range(len(scores)):
        visit(target, set())
    return sorted((target, prediction, scores[target][prediction]) for prediction, target in assigned.items())


def _mask(row: dict, family: str, fields: set[str]) -> bool:
    mask = row.get("supervision_mask")
    if mask is None:
        return True  # Explicit targets with no mask are fully annotated for present fields only.
    if not isinstance(mask, dict):
        return False
    value = mask.get(family, False)
    return value is True or isinstance(value, dict) and all(value.get(field) is True for field in fields)


def _intervals(values: Any) -> list[tuple[float, float]]:
    if not isinstance(values, list):
        return []
    intervals = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return []
        a, b = item
        if not _number(a) or not _number(b) or b <= a:
            return []
        intervals.append((a, b))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def _covered(event: dict, intervals: list[tuple[float, float]]) -> bool:
    start, end = event.get("started_at"), event.get("ended_at")
    return (_number(start) and _number(end) and end >= start
            and any(a <= start <= end <= b for a, b in intervals))


def _class_covered(event: dict, classes: list) -> bool:
    for item in classes:
        if isinstance(item, str) and _name(item) == _name(event.get("kind", event.get("event_type"))):
            return True
        if isinstance(item, dict) and _name(item.get("kind")) == _name(event.get("kind", event.get("event_type"))):
            if item.get("object_label") is None or _name(item["object_label"]) == _name(event.get("object_label")):
                return True
    return False


def _timeline(row: dict) -> str | None:
    value = (row.get("annotation_coverage", {}).get("timeline_id") or row.get("group_id")
             or row.get("provenance", {}).get("recording_id"))
    return str(value) if value is not None else None


def _coverage(row: dict) -> tuple[bool, list, list, str | None]:
    coverage = row.get("annotation_coverage") or {}
    intervals = _intervals(coverage.get("covered_intervals"))
    classes = coverage.get("evaluated_event_classes")
    classes = [c for c in classes if isinstance(c, str) and _name(c)
               or isinstance(c, dict) and _name(c.get("kind"))] if isinstance(classes, list) else []
    window = row.get("window") or {}
    if window and any(a < window.get("started_at", a) or b > window.get("ended_at", b) for a, b in intervals):
        intervals = []
    exhaustive = (coverage.get("exhaustive") is True and coverage.get("event_count_exhaustive") is not False
                  and bool(intervals) and bool(classes) and isinstance(row.get("target"), dict)
                  and isinstance(row["target"].get("events"), list)
                  and _mask(row, "events", {"kind", "object_label", "started_at", "ended_at"}))
    return exhaustive, intervals, classes, _timeline(row)


def _decision(row: dict) -> tuple[dict, dict, bool]:
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    decision = result.get("decision", row.get("decision"))
    metrics = result.get("metrics", row.get("metrics", {}))
    # Do not repair/reparse rejected raw text into apparently accepted predictions.
    usable = isinstance(decision, dict) and not result.get("error") and not row.get("error")
    return decision if usable else {}, metrics if isinstance(metrics, dict) else {}, usable


def _confidence_sample(event: dict, correct: bool) -> tuple[float, int] | None:
    value = event.get("confidence")
    return (float(value), int(correct)) if _number(value) and 0 <= value <= 1 else None


def _clock_delays(row: dict, event: dict, target: dict) -> tuple[float, float] | None:
    mapping = row.get("clock_mapping")
    if not isinstance(mapping, dict):
        return None
    source, wall, scale = (mapping.get("source_origin"), mapping.get("wall_origin"), mapping.get("scale", 1.0))
    emissions = row.get("event_emitted_at")
    emissions = emissions if isinstance(emissions, dict) else {}
    emitted = emissions.get(event.get("event_id"), row.get("emitted_at"))
    if not all(_number(x) for x in [source, wall, scale, emitted]) or scale <= 0:
        return None
    start = emitted - (wall + (target["started_at"] - source) * scale)
    end = emitted - (wall + (target["ended_at"] - source) * scale)
    return (start, end) if start >= 0 else None


def _stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": mean(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def score_predictions(rows: list[dict], temporal_iou_threshold: float = 0.5,
                      box_iou_threshold: float = 0.5) -> dict:
    """Score joined dataset/prediction rows, retaining unsupported metrics as null."""
    if not 0 < temporal_iou_threshold <= 1 or not 0 < box_iou_threshold <= 1:
        raise ValueError("IoU thresholds must be in (0, 1]")
    event_counts, class_counts = Counter(), defaultdict(Counter)
    tracking_counts = Counter()
    event_ious, box_ious, calibration, latency, start_delays, end_delays = [], [], [], [], [], []
    exhaustive_intervals = defaultdict(list)
    all_exhaustive, tracking_matches, details, scope_classes = True, [], [], []
    accepted, rate_false_count = 0, 0
    model_provenance = defaultdict(set)
    for row_index, row in enumerate(rows):
        decision, metrics, usable = _decision(row)
        accepted += usable
        for key in ["backend", "model_id", "model_revision", "adapter_path", "adapter_merged"]:
            if key in metrics:
                model_provenance[key].add(str(metrics[key]))
        value = metrics.get("latency_s", metrics.get("total_latency_s", row.get("latency_s")))
        if _number(value) and value >= 0:
            latency.append(float(value))
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        target_events = target.get("events", [])
        target_events = target_events if isinstance(target_events, list) else []
        supported = _mask(row, "events", {"kind", "object_label", "started_at", "ended_at"})
        labeled = [e for e in target_events if isinstance(e, dict) and supported
                   and _name(e.get("kind", e.get("event_type"))) and _name(e.get("object_label"))
                   and temporal_iou(e, e) == 1]
        predicted = decision.get("events", [])
        predicted = [e for e in predicted if isinstance(e, dict)] if isinstance(predicted, list) else []
        scores = [[temporal_iou(a, b) if (_name(a.get("kind", a.get("event_type")))
                   == _name(b.get("kind", b.get("event_type")))
                   and _name(a.get("object_label")) == _name(b.get("object_label"))) else 0
                   for b in predicted] for a in labeled]
        matches = _matching(scores, temporal_iou_threshold)
        matched_pred = {p for _, p, _ in matches}
        matched_gt = {g for g, _, _ in matches}
        event_counts.update(labeled=len(labeled), matched=len(matches), missing=len(labeled) - len(matches),
                            predicted=len(predicted), unsupported_target_events=len(target_events) - len(labeled))
        exhaustive, intervals, classes, timeline = _coverage(row)
        scope_classes.extend(classes)
        if exhaustive and timeline:
            exhaustive_intervals[timeline].extend(intervals)
        else:
            all_exhaustive = False
        false, extras = 0, 0
        for g, event in enumerate(labeled):
            key = f"{event.get('kind', event.get('event_type'))}:{event['object_label']}"
            class_counts[key].update(labeled=1, matched=int(g in matched_gt))
        for g, p, iou in matches:
            event_ious.append(iou)
            sample = _confidence_sample(predicted[p], True)
            if sample:
                calibration.append(sample)
            delays = _clock_delays(row, predicted[p], labeled[g])
            if delays:
                start_delays.append(delays[0])
                end_delays.append(delays[1])
        for p, event in enumerate(predicted):
            if p in matched_pred:
                continue
            if exhaustive and _covered(event, intervals) and _class_covered(event, classes):
                false += 1
                sample = _confidence_sample(event, False)
                if sample:
                    calibration.append(sample)
            else:
                extras += 1
        event_counts.update(false_with_exhaustive_coverage=false, unscored_extras=extras)
        if exhaustive and timeline:
            rate_false_count += false
        # Only supplied GT boxes are eligible. No detections/identity truth is inferred from EPIC nouns.
        gt_objects = target.get("objects", [])
        gt_objects = gt_objects if isinstance(gt_objects, list) else []
        object_fields = {"box", "label", "timestamp", "frame_evidence_id"}
        gt_objects = [o for o in gt_objects if isinstance(o, dict) and _mask(row, "objects", object_fields)
                      and _name(o.get("label")) and box_iou(o.get("box"), o.get("box")) == 1
                      and _number(o.get("timestamp")) and o.get("frame_evidence_id")]
        pred_objects = decision.get("objects", [])
        pred_objects = [o for o in pred_objects if isinstance(o, dict)] if isinstance(pred_objects, list) else []
        object_scores = [[box_iou(a.get("box"), b.get("box")) if (
            a["frame_evidence_id"] == b.get("frame_evidence_id")
            and a["timestamp"] == b.get("timestamp") and _name(a["label"]) == _name(b.get("label")))
            else 0 for b in pred_objects] for a in gt_objects]
        object_matches = _matching(object_scores, box_iou_threshold)
        tracking_counts.update(annotated_boxes=len(gt_objects), matched_boxes=len(object_matches),
                               missing_boxes=len(gt_objects) - len(object_matches))
        for g, p, iou in object_matches:
            box_ious.append(iou)
            gt, pred = gt_objects[g], pred_objects[p]
            if gt.get("track_id") and timeline and _mask(row, "objects", {"track_id"}):
                tracking_matches.append((timeline, str(gt["track_id"]), gt["timestamp"],
                                         str(gt["frame_evidence_id"]), pred.get("track_id")))
        details.append({"id": row.get("id", str(row_index)), "output_usable": usable,
                        "labeled_events": len(labeled), "matched_events": len(matches),
                        "missing_events": len(labeled) - len(matches), "matches": [
                            {"target_index": g, "prediction_index": p, "temporal_iou": iou}
                            for g, p, iou in matches], "false_events_with_coverage": false,
                        "unscored_extra_events": extras, "coverage_exhaustive": exhaustive,
                        "annotated_boxes": len(gt_objects), "matched_boxes": len(object_matches)})
    duration = sum(sum(b - a for a, b in _intervals(v)) for v in exhaustive_intervals.values())
    previous_tracks, seen_frames = {}, set()
    for timeline, gt_id, timestamp, evidence, prediction_id in sorted(tracking_matches, key=lambda x: x[:4]):
        key = (timeline, gt_id)
        frame_key = (*key, timestamp, evidence)
        if frame_key in seen_frames:
            raise ValueError("Duplicate GT track/frame across evaluation rows; select one prediction per frame")
        seen_frames.add(frame_key)
        if not prediction_id:
            tracking_counts["matched_boxes_missing_predicted_track_id"] += 1
            continue
        if key in previous_tracks and previous_tracks[key] != prediction_id:
            tracking_counts["identity_switches"] += 1
        previous_tracks[key] = prediction_id
        tracking_counts["identity_labeled_matches"] += 1
    false_rate = rate_false_count / (duration / 3600) if duration else None
    recall = event_counts["matched"] / event_counts["labeled"] if event_counts["labeled"] else None
    bins = []
    for index in range(10):
        samples = [(p, y) for p, y in calibration if min(int(p * 10), 9) == index]
        if samples:
            bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": len(samples),
                         "mean_confidence": mean(p for p, _ in samples), "accuracy": mean(y for _, y in samples)})
    return {
        "schema_version": 1, "scope": "physical sensing only; supplied annotations, never HA actions",
        "windows": len(rows), "usable_decisions": accepted,
        "prediction_provenance": {k: sorted(v) for k, v in model_provenance.items()},
        "output_usable_rate": accepted / len(rows) if rows else None,
        "validity_note": "Usable means a supplied decision without a reported error; run strict validation upstream.",
        "event_matching": {"kind_and_object_label": "case/whitespace-normalized exact match; no inferred aliases",
            "temporal_iou_threshold": temporal_iou_threshold,
            "algorithm": "maximum-cardinality matching; descending-IoU deterministic neighbors"},
        "events": {**dict(event_counts), "labeled_event_recall": recall,
            "mean_matched_temporal_iou": mean(event_ious) if event_ious else None,
            "false_events_per_hour": false_rate if all_exhaustive and rows else None,
            "false_events_per_hour_unavailable_reason": None if all_exhaustive and duration else
                "Requires explicit exhaustive class/time coverage, labeled events (including empty negatives), "
                "and stable timeline IDs for every evaluated row; unannotated extras are not false events.",
            "exhaustive_subset": {"covered_seconds": duration, "false_events_per_hour": false_rate,
                "false_events": rate_false_count,
                "scope": "Only explicit exhaustive class/time intervals; overlapping intervals counted once per timeline"},
            "evaluated_classes": sorted({str(c) for c in scope_classes}),
            "per_kind_object": {key: {**dict(value), "recall": value["matched"] / value["labeled"]}
                                for key, value in sorted(class_counts.items())},
            "pretrimmed_interval_warning": any((r.get("annotation_coverage") or {}).get("intervals_are_pretrimmed")
                                                for r in rows)},
        "event_confidence_calibration": {"scorable_predictions": len(calibration),
            "correct": sum(y for _, y in calibration), "incorrect": sum(1 - y for _, y in calibration),
            "positive_only": bool(calibration) and all(y for _, y in calibration),
            "brier_known_labels": mean((p - y) ** 2 for p, y in calibration) if calibration else None,
            "ece_known_labels": sum(b["count"] * abs(b["mean_confidence"] - b["accuracy"])
                                    for b in bins) / len(calibration) if calibration else None,
            "bins": bins, "scope": "Matched positives plus covered false events with numeric confidence only; "
                "selection-biased when annotation coverage is partial. Not population calibration.",
            "textual_uncertainty_calibration": None,
            "textual_uncertainty_reason": "No reference rubric for free-text uncertainty is supplied."},
        "timing": {"model_latency_s": _stats(latency), "event_start_to_emit_delay_s": _stats(start_delays),
            "event_end_to_emit_delay_s": _stats(end_delays),
            "detection_delay_unavailable_reason": None if start_delays else
                "No matched events with explicit source-to-wall clock mapping and measured emission time.",
            "delay_scope": "Only matched events; model latency is reported separately and is not detection delay."},
        "tracking": {**dict(tracking_counts),
            "annotated_box_recall": tracking_counts["matched_boxes"] / tracking_counts["annotated_boxes"]
                if tracking_counts["annotated_boxes"] else None,
            "mean_matched_iou": mean(box_ious) if box_ious else None,
            "identity_switches": tracking_counts["identity_switches"]
                if tracking_counts["identity_labeled_matches"] else None,
            "unavailable_reason": None if tracking_counts["annotated_boxes"] else
                "No supervised frame boxes; source noun labels do not establish detections or tracks.",
            "box_iou_threshold": box_iou_threshold,
            "identity_switch_definition": "Changed predicted track ID between consecutive matched annotations "
                "of the same GT track/timeline; unmatched gaps do not reset identity; not IDF1 or HOTA.",
            "idf1": None, "hota": None},
        "rows": details,
    }

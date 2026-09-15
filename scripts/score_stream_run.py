#!/usr/bin/env python3
"""Score the learned decisions of a stream replay against a video's official dense narrations.

Each model window becomes an evaluation row: narrations with at least half their
duration inside the window are its targets (clipped to the window); overlapping
narrations of evaluated classes that were not selected are removed from the
covered interval so they are unscored rather than false. The run's clock mapping
and emission times give detection delay. Geometric events are reported as counts
only; they are not semantic claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from home_observer.physical_evaluate import score_predictions

CLASSES = ["take", "pick-up", "put", "put-down", "put-on", "put-in", "put-into", "open", "close"]
# Diagnostic only: EPIC verb classes grouped into interaction families. The primary metric
# keeps exact verb matching; this reports how many misses are cross-verb near matches.
FAMILIES = {"pickup": {"take", "pick-up"}, "placement": {"put", "put-down", "put-on", "put-in", "put-into"},
            "opening": {"open"}, "closing": {"close"}}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return round(int(hours) * 3600 + int(minutes) * 60 + float(rest), 6)


def subtract(interval, holes):
    pieces = [tuple(interval)]
    for a, b in holes:
        updated = []
        for start, end in pieces:
            if b <= start or a >= end:
                updated.append((start, end))
            else:
                if a > start:
                    updated.append((start, a))
                if b < end:
                    updated.append((b, end))
        pieces = updated
    return [[round(s, 6), round(e, 6)] for s, e in pieces if e - s > 1e-6]


def build_rows(predictions, narrations, classes):
    rows = []
    for item in predictions:
        window = item["window"]
        start, end = window["started_at"], window["ended_at"]
        targets, holes = [], []
        for n in narrations:
            overlap = min(end, n["stop"]) - max(start, n["start"])
            if overlap <= 0:
                continue
            duration = n["stop"] - n["start"]
            if duration <= 0 or overlap / duration >= 0.5:
                targets.append({"kind": n["verb"], "object_label": n["noun"], "started_at": max(start, n["start"]),
                                "ended_at": min(end, n["stop"]), "description": n["narration"]})
            elif n["verb"] in classes:
                holes.append((max(start, n["start"]), min(end, n["stop"])))
        rows.append({
            "id": item["id"], "window": window, "group_id": narrations[0]["video_id"] if narrations else None,
            "target": {"summary": "; ".join(t["description"] for t in targets) or "no annotated action",
                       "events": targets},
            "supervision_mask": {"summary": True, "events": True},
            "annotation_coverage": {"exhaustive": True, "event_count_exhaustive": True,
                                    "timeline_id": narrations[0]["video_id"] if narrations else "unknown",
                                    "covered_intervals": subtract((start, end), holes),
                                    "evaluated_event_classes": classes, "intervals_are_pretrimmed": False},
            "result": item["result"], "clock_mapping": item.get("clock_mapping"),
            "emitted_at": item.get("emitted_at"),
        })
    return rows


def family_of(kind):
    kind = str(kind or "").strip().lower()
    return next((name for name, verbs in FAMILIES.items() if kind in verbs), None)


def family_diagnostic(rows):
    """Cross-verb near matches within a family, same noun and temporal IoU >= 0.5; not the primary score."""
    from home_observer.physical_evaluate import temporal_iou
    labeled = matched = 0
    for row in rows:
        decision = row["result"].get("decision") if not row["result"].get("error") else None
        predicted = (decision or {}).get("events", [])
        used = set()
        for target in row["target"]["events"]:
            family = family_of(target["kind"])
            if family is None:
                continue
            labeled += 1
            for index, event in enumerate(predicted):
                if index in used or family_of(event.get("kind")) != family:
                    continue
                if str(event.get("object_label", "")).strip().lower() != target["object_label"].strip().lower():
                    continue
                if temporal_iou(event, target) >= 0.5:
                    used.add(index)
                    matched += 1
                    break
    return {"labeled_family_events": labeled, "family_matches": matched,
            "family_recall": matched / labeled if labeled else None,
            "scope": "diagnostic near-match across verbs of one family; never replaces exact-verb recall"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Production or research stream run directory")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True, nargs="+",
                        help="Official narration CSVs (train and/or validation)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    predictions = [json.loads(line) for line in (args.run / "predictions.jsonl").read_text().splitlines() if line.strip()]
    narrations = [{"video_id": r["video_id"], "narration_id": r["narration_id"], "narration": r["narration"],
                   "verb": r["verb"], "noun": r["noun"], "start": seconds(r["start_timestamp"]),
                   "stop": seconds(r["stop_timestamp"])}
                  for path in args.annotations for r in csv.DictReader(path.open()) if r["video_id"] == args.video_id]
    if not narrations:
        raise ValueError("no narrations found for the video in the supplied annotation files")
    rows = build_rows(predictions, narrations, CLASSES)
    scores = score_predictions(rows)
    report = json.loads((args.run / "stream-report.json").read_text())
    geometric = [json.loads(line) for line in (args.run / "geometric-events.jsonl").read_text().splitlines()
                 if line.strip()] if (args.run / "geometric-events.jsonl").is_file() else []
    span = [min(r["window"]["started_at"] for r in rows), max(r["window"]["ended_at"] for r in rows)] if rows else None
    covered = [n for n in narrations if span and n["stop"] >= span[0] and n["start"] <= span[1]]
    result = {
        "run": str(args.run.resolve()), "video_id": args.video_id,
        "annotations_sha256": {str(path): sha(path) for path in args.annotations},
        "model_windows": len(rows), "replay_source_span": span,
        "narrations_in_span": [{k: n[k] for k in ("narration_id", "verb", "noun", "start", "stop")} for n in covered],
        "scores": scores, "family_diagnostic": family_diagnostic(rows),
        "geometric_events": {"count": len(geometric), "kinds": sorted({g["event"]["kind"] for g in geometric})},
        "capture": {"captured_frames": report["captured_frames"], "losses": report["losses"],
                    "tracking_frame_losses": report["tracking_frame_losses"],
                    "capture_lateness_mean_s": report["capture_lateness_mean_s"],
                    "capture_lateness_max_s": report["capture_lateness_max_s"], "spool": report["spool"]},
        "scope": "Learned decisions of activity-gated model windows scored against official narrations; "
                 "recall is not comparable to dense offline tiling; geometric events are not semantic claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    events = scores["events"]
    print(json.dumps({"model_windows": len(rows), "usable": scores["usable_decisions"],
                      "labeled": events["labeled"], "matched": events["matched"],
                      "recall": events["labeled_event_recall"], "false_with_coverage": events["false_with_exhaustive_coverage"],
                      "false_events_per_hour": events["false_events_per_hour"],
                      "family_diagnostic": result["family_diagnostic"],
                      "delay": scores["timing"]["event_end_to_emit_delay_s"]}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score recorded physical predictions; never run inference or send target labels.

Prediction rows use id + result:{decision,metrics,error?}, or top-level
decision/metrics. With --dataset, source labels and coverage are authoritative.
Without it, self-contained prediction rows or unlabeled live logs are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from home_observer.physical_evaluate import score_predictions


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Every JSONL row must be an object")
            rows.append(row)
    return rows


def _without_paths(value):
    if isinstance(value, dict):
        return {k: _without_paths(v) for k, v in value.items() if k != "path"}
    if isinstance(value, list):
        return [_without_paths(v) for v in value]
    return value


def join_predictions(dataset: list[dict], predictions: list[dict]) -> list[dict]:
    def index(rows):
        result = {}
        for row in rows:
            key = row.get("id")
            if not isinstance(key, str) or not key or key in result:
                raise ValueError("Every row needs a unique nonempty string id")
            result[key] = row
        return result

    source, predicted = index(dataset), index(predictions)
    if set(predicted) - set(source):
        raise ValueError("Predictions contain IDs outside the source dataset")
    result = []
    allowed = {"result", "decision", "metrics", "error", "raw_output", "emitted_at", "event_emitted_at",
               "clock_mapping", "latency_s"}
    for key, row in source.items():
        prediction = predicted.get(key)
        if prediction is None:
            result.append(dict(row, result={"error": "Missing prediction"}))
            continue
        for name in ["target", "supervision_mask", "annotation_coverage"]:
            if name in prediction and prediction[name] != row.get(name):
                raise ValueError(f"Prediction changed source {name} for {key}")
        if "window" in prediction and _without_paths(prediction["window"]) != _without_paths(row.get("window")):
            raise ValueError(f"Prediction window differs from source for {key}")
        result.append({**{k: v for k, v in row.items() if k not in allowed},
                       **{k: v for k, v in prediction.items() if k in allowed}})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Authoritative annotated source JSONL; optional for live logs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-iou", type=float, default=0.5)
    parser.add_argument("--box-iou", type=float, default=0.5)
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    rows = join_predictions(read_jsonl(args.dataset), predictions) if args.dataset else predictions
    report = score_predictions(rows, args.temporal_iou, args.box_iou)
    inputs = [args.predictions, *([args.dataset] if args.dataset else [])]
    report["provenance"] = {"inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                       for p in inputs],
                            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "inference_executed": False, "predictions_read": len(predictions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "windows": report["windows"],
                      "labeled_event_recall": report["events"]["labeled_event_recall"],
                      "false_events_per_hour": report["events"]["false_events_per_hour"]}))


if __name__ == "__main__":
    main()

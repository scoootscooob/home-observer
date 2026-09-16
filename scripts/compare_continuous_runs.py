#!/usr/bin/env python3
"""Matched comparison of base / v1 / v2 native evaluations on the frozen continuous test windows.

Every run must share the dataset, media, prompt, implementation, generation settings
and token budgets; only the adapter differs. Strict scores come from the runs'
validated predictions; judged-fields scores are recomputed from the retained raw
generations with the same completion rule the local backend uses. Invalid outputs
stay invalid; nothing is repaired into an accepted event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from home_observer.model import ModelConfig
from home_observer.physical_evaluate import score_predictions
from home_observer.physical_local import complete_judged_decision
from home_observer.physical_model import physical_request_from_row, prepare_physical_request
from home_observer.physical_schema import validate_physical_decision


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"{root}: prediction run is incomplete")
    if sha(root / "predictions.jsonl") != manifest["predictions_sha256"]:
        raise ValueError(f"{root}: predictions hash mismatch")
    raw_name = manifest["raw_outputs_file"]
    if sha(root / raw_name) != manifest["raw_outputs_sha256"]:
        raise ValueError(f"{root}: raw output hash mismatch")
    predictions = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines() if line.strip()]
    raw = {item["id"]: item for item in (json.loads(line) for line in (root / raw_name).read_text().splitlines()
                                          if line.strip())}
    return {"root": root, "manifest": manifest, "predictions": predictions, "raw": raw}


def judged_rows(run: dict, dataset_rows: dict, config: ModelConfig) -> list[dict]:
    rows = []
    for item in run["predictions"]:
        source = dataset_rows[item["id"]]
        generated = run["raw"].get(item["id"], {})
        raw_text = generated.get("raw_output") or ""
        result = {"decision": None, "metrics": generated.get("metrics") or {}, "raw_output": raw_text}
        if item["result"].get("decision") and not item["result"].get("error"):
            result["decision"] = item["result"]["decision"]
        elif raw_text:
            clean = prepare_physical_request(physical_request_from_row(source), config)
            try:
                completed, completion = complete_judged_decision(raw_text, clean["window"])
                completed = validate_physical_decision(completed, clean["window"])
                result["decision"] = completed.model_dump()
                result["metrics"] = {**result["metrics"], "completion": completion}
            except ValueError as exc:
                result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        else:
            result["error"] = item["result"].get("error") or "no generation"
        rows.append({**source, "id": item["id"], "result": result})
    return rows


def summary(scores: dict) -> dict:
    events = scores["events"]
    return {"usable": scores["usable_decisions"], "windows": scores["windows"],
            "usable_rate": scores["output_usable_rate"], "labeled": events["labeled"], "matched": events["matched"],
            "recall": events["labeled_event_recall"], "false_with_coverage": events["false_with_exhaustive_coverage"],
            "false_events_per_hour": events["false_events_per_hour"],
            "false_events_per_hour_exhaustive_subset": events["exhaustive_subset"]["false_events_per_hour"],
            "covered_seconds": events["exhaustive_subset"]["covered_seconds"],
            "unscored_extras": events["unscored_extras"], "mean_iou": events["mean_matched_temporal_iou"],
            "calibration": {k: scores["event_confidence_calibration"][k] for k in ("scorable_predictions", "brier_known_labels", "ece_known_labels", "positive_only")},
            "mean_latency_s": scores["timing"]["model_latency_s"]["mean"], "per_kind": scores["events"]["per_kind_object"]}


def quiet_abstention(rows: list[dict], scored: dict) -> dict:
    classes = {"take", "pick-up", "put", "put-down", "put-on", "put-in", "put-into", "open", "close"}
    quiet = [r for r in rows if (r.get("annotation_coverage") or {}).get("quiet_window")]
    abstained = 0
    for row in quiet:
        decision = row["result"].get("decision") if not row["result"].get("error") else None
        events = (decision or {}).get("events", [])
        if decision is not None and not any(str(e.get("kind", "")).strip().lower() in classes for e in events):
            abstained += 1
    usable_quiet = sum(1 for r in quiet if r["result"].get("decision") and not r["result"].get("error"))
    return {"quiet_windows": len(quiet), "usable_quiet": usable_quiet, "abstained": abstained,
            "abstention_rate_on_usable": abstained / usable_quiet if usable_quiet else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=path, e.g. base=artifacts/physical-v2/eval-base")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset_rows = {row["id"]: row for row in (json.loads(line) for line in args.dataset.read_text().splitlines()
                                               if line.strip())}
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = load_run(Path(path))
    reference = next(iter(runs.values()))["manifest"]
    matched = {}
    for key in ("dataset_sha256", "selected_ids", "prompt_sha256", "media_sha256", "implementation_sha256",
                "generation", "versions", "source_rows", "selection_limit", "backend"):
        matched[key] = all(run["manifest"].get(key) == reference.get(key) for run in runs.values())
    for key in reference["model_config"]:
        if key != "adapter_path":
            matched["model_config." + key] = all(run["manifest"]["model_config"].get(key) == reference["model_config"][key]
                                                 for run in runs.values())
    unmatched = sorted(k for k, ok in matched.items() if not ok)
    results = {}
    for name, run in runs.items():
        config = ModelConfig(dataset_root=str(args.dataset_root.resolve()), quantization="none",
                             max_frames=reference["model_config"]["max_frames"], max_input_tokens=8192,
                             max_new_tokens=reference["model_config"]["max_new_tokens"], merge_adapter=False)
        strict_rows = [{**dataset_rows[item["id"]], "id": item["id"], "result": item["result"]} for item in run["predictions"]]
        strict = score_predictions(strict_rows)
        judged_list = judged_rows(run, dataset_rows, config)
        judged = score_predictions(judged_list)
        results[name] = {"adapter": run["manifest"]["model_config"].get("adapter_path"),
                         "adapter_sha256": run["manifest"].get("adapter_sha256"),
                         "strict": summary(strict), "judged": summary(judged),
                         "abstention_strict": quiet_abstention(strict_rows, strict),
                         "abstention_judged": quiet_abstention(judged_list, judged)}
        (args.output.parent / f"{args.output.stem}-{name}-strict.json").write_text(json.dumps(strict, indent=2) + "\n")
        (args.output.parent / f"{args.output.stem}-{name}-judged.json").write_text(json.dumps(judged, indent=2) + "\n")
    report = {"dataset": str(args.dataset), "dataset_sha256": sha(args.dataset), "runs": results,
              "matched_settings": matched, "unmatched_settings": unmatched,
              "note": "Strict = complete validated decision; judged = adapter-judged fields with recorded placeholders; "
                      "false events per hour counts unmatched evaluated-class predictions inside covered intervals."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Run | Contract | Usable | Recall | False/h (covered subset) | Abstention on quiet | Mean latency |", "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value, digits, suffix=""):
        return "n/a" if value is None else f"{value:.{digits}f}{suffix}"

    for name, r in results.items():
        for contract in ("strict", "judged"):
            s, a = r[contract], r["abstention_" + contract]
            lines.append("| " + " | ".join([name, contract, f"{s['usable']}/{s['windows']}", fmt(s["recall"], 3),
                                            fmt(s["false_events_per_hour_exhaustive_subset"], 1),
                                            fmt(a["abstention_rate_on_usable"], 2),
                                            fmt(s["mean_latency_s"], 1, " s")]) + " |")
    print("\n".join(lines))
    if unmatched:
        print("UNMATCHED SETTINGS:", unmatched)


if __name__ == "__main__":
    main()

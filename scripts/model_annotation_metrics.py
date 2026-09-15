#!/usr/bin/env python3
"""Score only annotated observation pairs; retain legacy strict-set compatibility.

No labels, predictions or model outputs are modified. Extra observation pairs are
unscored, not established errors. Value scoring does not validate action safety or
evidence grounding; those remain separately reported by the existing evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from home_observer.evaluate import summarize


def value_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def counts():
    return {"annotated_pairs": 0, "correct": 0, "incorrect": 0, "missing": 0}


def rates(values):
    result = dict(values)
    covered = values["correct"] + values["incorrect"]
    total = values["annotated_pairs"]
    result.update(covered=covered, coverage=covered / total if total else None,
                  accuracy_on_covered=values["correct"] / covered if covered else None,
                  correct_fraction=values["correct"] / total if total else None)
    return result


def observation_map(observations):
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    result = defaultdict(set)
    for observation in observations:
        if not isinstance(observation, dict) or not all(k in observation for k in ("entity_id", "attribute", "value")):
            raise ValueError("cannot score an observation without entity_id, attribute and value")
        pair = (observation["entity_id"], observation["attribute"])
        if not all(isinstance(part, str) and part for part in pair):
            raise ValueError("observation pair must contain nonempty strings")
        result[pair].add(value_key(observation["value"]))
    return result


def known_label_metrics(rows):
    total, occupancy = counts(), counts()
    per_entity = defaultdict(counts)
    extras_by_entity = Counter()
    extras, extras_windows, labeled_windows, failed_windows = 0, 0, 0, 0
    unscored_windows = 0
    details = []
    for row in rows:
        result = row.get("result", {})
        decision = result.get("decision") or {}
        failure = bool(result.get("error") or result.get("rejections") or result.get("skipped") or not decision)
        failed_windows += failure
        target = row.get("target")
        mask = row.get("supervision_mask", {})
        if not isinstance(mask, dict):
            raise ValueError("supervision_mask must be an object")
        supervised = isinstance(target, dict) and "observations" in target and mask.get("observations", True) is True
        actual = observation_map(decision.get("observations", []))
        if supervised:
            labeled_windows += 1
            expected = observation_map(target["observations"])
            if any(len(values) != 1 for values in expected.values()):
                raise ValueError("conflicting source annotations for the same observation pair")
        else:
            unscored_windows += 1
            expected = {}
        row_counts = counts()
        for (entity, attribute), expected_values in expected.items():
            predicted_values = actual.get((entity, attribute))
            status = "missing" if predicted_values is None else "correct" if predicted_values == expected_values else "incorrect"
            groups = [total, row_counts, per_entity[entity]]
            if entity.startswith("room.") and attribute == "occupied":
                groups.append(occupancy)
            for group in groups:
                group["annotated_pairs"] += 1
                group[status] += 1
        extra_pairs = set(actual) - set(expected)
        extras += len(extra_pairs)
        extras_windows += bool(extra_pairs)
        for entity, _ in extra_pairs:
            extras_by_entity[entity] += 1
        details.append({"id": row.get("id"), "task_type": row.get("task_type"),
                        "observation_supervised": supervised, "original_result_failed": failure,
                        **rates(row_counts), "unscored_extra_pairs": len(extra_pairs)})
    legacy = summarize(rows)
    return {
        "version": 1, "metric": "known_annotated_observation_pairs",
        "scope": "Exact typed values for annotated (entity_id, attribute) pairs only. Additional pairs are unscored extras.",
        "matching": "Exact JSON values; strings, booleans and numbers are distinct. Identical duplicates count once; conflicting predictions are incorrect.",
        "failure_handling": "Uses parsed observation values even when other fields are rejected. Invalid/failed windows are retained separately; no raw output is repaired or parsed here.",
        "windows": len(rows), "observation_labeled_windows": labeled_windows,
        "observation_unscored_windows": unscored_windows, "original_invalid_or_failed_windows": failed_windows,
        "known_labels": rates(total), "camera_occupancy": rates(occupancy),
        "camera_occupancy_definition": "Source-annotated room.* / occupied pairs; this name does not assert that the model used camera evidence.",
        "per_entity": {entity: rates(value) for entity, value in sorted(per_entity.items())},
        "unscored_extras": {"pair_instances": extras, "windows": extras_windows,
                            "by_entity": dict(sorted(extras_by_entity.items())),
                            "interpretation": "No source annotation for these pairs. Neither correct nor incorrect perception is established."},
        "legacy_strict_set_compatibility": {"interpretation": "Original evaluator semantics. Unannotated extras count as strict-set mismatches; this is not proof of false perception.",
                                            "metrics": legacy},
        "rows": details,
    }


def read_predictions(path):
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: incomplete or invalid prediction JSON; retry after copying finishes") from exc
    return rows, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "rows": len(rows)}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--release-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.predictions:
        rows, provenance = read_predictions(args.predictions.resolve())
        report = {**known_label_metrics(rows), "built_at": built_at, "sources": [provenance],
                  "completeness": "File snapshot only; no full-release completion is inferred."}
        output = args.output or args.predictions.parent / "annotation-metrics.json"
        write_report(output, report)
        print(json.dumps({"output": str(output), "windows": len(rows), "known_labels": report["known_labels"]}))
        return
    root = args.release_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    expected = {entry["name"]: entry["selected_rows"] for entry in manifest.get("suites", [])}
    suites, all_rows, sources = {}, [], []
    for path in sorted(root.glob("*/predictions.jsonl")):
        rows, provenance = read_predictions(path)
        suite = {**known_label_metrics(rows), "built_at": built_at, "sources": [provenance],
                 "expected_windows": expected.get(path.parent.name),
                 "complete": path.parent.name in expected and len(rows) == expected[path.parent.name]}
        write_report(path.parent / "annotation-metrics.json", suite)
        suites[path.parent.name] = suite
        all_rows.extend(rows)
        sources.append(provenance)
    if not sources:
        raise ValueError("no suite predictions found under release-root")
    report = {**known_label_metrics(all_rows), "built_at": built_at, "sources": sources,
              "suites": suites, "expected_suites": expected,
              "complete": bool(expected) and set(suites) == set(expected) and all(s["complete"] for s in suites.values())}
    output = args.output or root / "annotation-metrics.json"
    write_report(output, report)
    print(json.dumps({"output": str(output), "windows": len(all_rows), "complete": report["complete"],
                      "known_labels": report["known_labels"]}))


if __name__ == "__main__":
    main()

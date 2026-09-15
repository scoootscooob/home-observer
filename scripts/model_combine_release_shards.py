#!/usr/bin/env python3
"""Verify and combine complete disjoint release shards without rerunning a model.

Input artifacts remain unchanged. A combined evaluation is a union of independently
measured decisions, not a single-GPU throughput or deployment acceptance result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

from home_observer.evaluate import summarize

SUITES = ("heldout", "counterfactual", "public_video")
MODES = {"heldout": "isolated_snapshot", "counterfactual": "isolated_snapshot",
         "public_video": "chronological_unscored"}
SOURCE_PATHS = {"heldout": "mixed/test.jsonl", "counterfactual": "counterfactuals/pairs.jsonl",
                "public_video": "public-egolife/replay.jsonl"}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc


def read_lines(path):
    """Retain original nonempty JSONL lines, including their target/raw text bytes."""
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise ValueError(f"Missing JSONL: {path}") from exc
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Incomplete or invalid JSONL: {path}:{number}") from exc
        require(isinstance(row, dict), f"Expected object: {path}:{number}")
        result.append((row, line if line.endswith(b"\n") else line + b"\n"))
    return result


def hash_value(value, label, *, optional=False):
    if optional and value is None:
        return
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"Invalid SHA-256: {label}")


def positive_elapsed(value, label, *, allow_zero=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and (value >= 0 if allow_zero else value > 0),
            f"Invalid elapsed time: {label}")
    return value


def implementation_signature(mapping):
    require(isinstance(mapping, dict) and mapping, "Implementation hashes are required")
    result = {}
    for name, sha in mapping.items():
        hash_value(sha, name)
        parts = Path(name).parts
        # Only host checkout prefixes are ignored, never module names or hashes.
        start = next((i for i, part in enumerate(parts) if part in ("scripts", "src")), 0)
        key = "/".join(parts[start:])
        require(key not in result, "Duplicate normalized implementation path")
        result[key] = sha
    return result


def identity(manifest):
    config = copy.deepcopy(manifest.get("model_config"))
    require(isinstance(config, dict), "model_config is required")
    require(type(config.get("merge_adapter")) is bool, "Explicit merge_adapter mode is required")
    require(isinstance(config.get("model_id"), str), "model_id is required")
    adapter_path = config.pop("adapter_path", None)
    config.pop("dataset_root", None)
    adapter_sha = manifest.get("adapter_sha256")
    hash_value(adapter_sha, "adapter", optional=not adapter_path)
    require(bool(adapter_path) == bool(adapter_sha), "Adapter path/hash presence mismatch")
    for field in ("system_prompt_sha256", "policy_sha256"):
        hash_value(manifest.get(field), field)
    policy = manifest.get("policy")
    require(isinstance(policy, dict), "Policy contents are required")
    require(hashlib.sha256(canonical(policy).encode()).hexdigest() == manifest["policy_sha256"],
            "Policy contents do not match policy_sha256")
    native_batch = manifest.get("native_batch_size")
    require(type(native_batch) is int and native_batch > 0, "native_batch_size is required")
    require(config.get("revision") == manifest.get("model_revision"), "Model revision mismatch")
    return {"model_config": config, "adapter_sha256": adapter_sha,
            "model_revision": manifest.get("model_revision"), "policy": policy,
            "policy_sha256": manifest["policy_sha256"],
            "system_prompt_sha256": manifest["system_prompt_sha256"],
            "native_batch_size": native_batch,
            "implementation_sha256": implementation_signature(manifest.get("implementation_sha256"))}


def verify_native(raw, prepared, manifest):
    require(raw.get("window_id") == prepared["window"]["window_id"],
            "Raw output order/window ID differs from prepared input")
    require(isinstance(raw.get("raw_output"), str), "Raw generation text is missing")
    require(type(raw.get("valid_decision")) is bool, "Raw validity marker is missing")
    metrics = raw.get("metrics")
    require(isinstance(metrics, dict), "Native generation metrics are missing")
    config = manifest["model_config"]
    require(metrics.get("model_id") == config["model_id"], "Actual model ID mismatch")
    require(metrics.get("adapter") == config.get("adapter_path"), "Actual adapter path mismatch")
    expected_merge = bool(config.get("adapter_path")) and config["merge_adapter"]
    require(type(metrics.get("adapter_merged")) is bool and metrics["adapter_merged"] == expected_merge,
            "Actual adapter merge mode mismatch")
    batch = metrics.get("batch_size", 1)
    require(type(batch) is int and 1 <= batch <= manifest["native_batch_size"],
            "Actual native batch size exceeds configured bound")
    positive_elapsed(metrics.get("latency_s"), "native latency", allow_zero=True)


def same_metric_values(actual, expected, label):
    # Recompute original correctness counts; never silently trust a partial summary.
    for key in ("windows", "targeted_windows", "invalid_or_failed_windows", "counts"):
        require(canonical(actual.get(key)) == canonical(expected[key]), f"{label}: inconsistent {key}")


def load_shard(path):
    manifest = read_json(path / "manifest.json")
    report = read_json(path / "report.json")
    shard = manifest.get("shard", {})
    index, count = shard.get("index"), shard.get("count")
    require(type(index) is int and type(count) is int and count >= 2 and 0 <= index < count,
            "Expected an explicit multi-worker shard index/count")
    ident = identity(manifest)
    require(canonical(report.get("model_config")) == canonical(manifest["model_config"]),
            "Shard report and manifest model configurations differ")
    entries = manifest.get("suites", [])
    require(len(entries) == len(SUITES) and {s.get("name") for s in entries} == set(SUITES),
            "Manifest must include each required suite exactly once")
    require(set(report.get("suites", {})) == set(SUITES), "Shard aggregate report is incomplete")
    require(report.get("complete") is not False, "Shard is explicitly incomplete")
    source_files = [path / "manifest.json", path / "report.json", path / "native-outputs.jsonl"]
    native = read_lines(path / "native-outputs.jsonl")
    suites, execution_order, total_elapsed = {}, [], 0.0
    for name in SUITES:
        entry = next(s for s in entries if s["name"] == name)
        require(entry.get("context_mode") == MODES[name], f"Wrong context mode for {name}")
        require(entry.get("targets_changed") is False and entry.get("window_ids_changed") is False,
                "Changed targets/window IDs cannot be combined as the original release")
        hash_value(entry.get("source_sha256"), f"{name} source")
        ids = entry.get("unsharded_selected_ids")
        require(isinstance(ids, list) and all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids),
                f"Missing or duplicate unsharded IDs: {name}")
        require(type(entry.get("source_rows")) is int and len(ids) == entry["source_rows"],
                f"{name}: a selected subset is not a full release")
        expected_ids = (ids if index == 0 else []) if name == "public_video" else ids[index::count]
        inputs_path, predictions_path = path / f"{name}-inputs.jsonl", path / name / "predictions.jsonl"
        metrics_path = path / name / "metrics.json"
        source_files.extend([inputs_path, predictions_path, metrics_path])
        inputs, predictions = read_lines(inputs_path), read_lines(predictions_path)
        require(entry.get("selected_rows") == len(expected_ids) == len(inputs) == len(predictions),
                f"{name}: missing or extra shard rows")
        require([r["id"] for r, _ in inputs] == expected_ids, f"{name}: input shard partition/order mismatch")
        require([r.get("id") for r, _ in predictions] == expected_ids,
                f"{name}: duplicate/missing/reordered prediction IDs")
        input_positions = {record_id: position for position, record_id in enumerate(ids)}
        groups = set()
        for (prepared, _), (predicted, _) in zip(inputs, predictions):
            require(type(prepared.get("evaluation_source_index")) is int and
                    prepared["evaluation_source_index"] == input_positions[prepared["id"]],
                    f"{name}: source index mismatch")
            require(prepared.get("evaluation_context_mode") == MODES[name], "Input context mode mismatch")
            require(canonical(prepared.get("target")) == canonical(predicted.get("target")),
                    "Prediction target differs from prepared target")
            require(canonical(prepared.get("supervision_mask", {})) ==
                    canonical(predicted.get("supervision_mask", {})), "Supervision mask mismatch")
            require(predicted.get("group_id") == prepared.get("recording_id"), "Journal group mismatch")
            if name != "public_video":
                require(prepared.get("recording_id") not in groups, "Snapshot journals are not independent")
                groups.add(prepared.get("recording_id"))
            else:
                require(prepared.get("target") is None, "Public chronology must remain unscored")
            result = predicted.get("result", {})
            require(result.get("window_id") == prepared["window"]["window_id"], "Result window mismatch")
            positive_elapsed(result.get("total_latency_s"), "event latency", allow_zero=True)
            execution_order.append((name, prepared))
        expected_metrics = summarize([r for r, _ in predictions])
        metrics = read_json(metrics_path)
        same_metric_values(metrics, expected_metrics, f"{name} metrics")
        same_metric_values(report["suites"][name], expected_metrics, f"{name} aggregate report")
        elapsed = positive_elapsed(metrics.get("elapsed_s"), f"{name} suite elapsed", allow_zero=not inputs)
        total_elapsed += elapsed
        suites[name] = {"manifest": entry, "inputs": inputs, "predictions": predictions,
                        "metrics": metrics, "report": report["suites"][name], "raw": []}
    require(len(native) == len(execution_order), "Missing or extra native output rows")
    for native_line, (name, prepared) in zip(native, execution_order):
        verify_native(native_line[0], prepared, manifest)
        suites[name]["raw"].append(native_line)
    return {"path": path, "index": index, "count": count, "manifest": manifest, "report": report,
            "identity": ident, "suites": suites, "total_elapsed_s": total_elapsed,
            "source_files": source_files}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_original_sources(shards, source_root):
    """Check supplied hashes and labels against the original local release inputs."""
    paths = []
    for name, relative in SOURCE_PATHS.items():
        path = source_root / relative
        require(path.is_file(), f"Original release source is missing: {path}; supply --source-root")
        original_lines = read_lines(path)
        original = {row["id"]: row for row, _ in original_lines}
        require(len(original) == len(original_lines), "Original source has duplicate IDs")
        for shard in shards:
            part = shard["suites"][name]
            require(digest(path) == part["manifest"]["source_sha256"], f"{name}: original source hash mismatch")
            require(list(original) == part["manifest"]["unsharded_selected_ids"], "Original source ID/order mismatch")
            for prepared, _ in part["inputs"]:
                source = original[prepared["id"]]
                for key in ("target", "supervision_mask"):
                    default = {} if key == "supervision_mask" else None
                    require(canonical(source.get(key, default)) == canonical(prepared.get(key, default)),
                            f"{name}: prepared {key} differs from original source")
                def without_paths(window):
                    window = copy.deepcopy(window)
                    for media in [*window.get("frames", []), *window.get("audio", [])]:
                        media.pop("path", None)
                    return window
                require(canonical(without_paths(source["window"])) == canonical(without_paths(prepared["window"])),
                        "Prepared model-visible window differs from original source beyond resolved media paths")
        paths.append(path)
    return paths


def combine_shards(shard_paths, output, source_root=None):
    paths, output = [Path(p).resolve() for p in shard_paths], Path(output).resolve()
    require(len(set(paths)) == len(paths), "Duplicate shard directories")
    require(not output.exists(), "Output already exists; preserve it and choose a fresh output directory")
    require(not any(output.is_relative_to(p) for p in paths), "Output must be outside input shards")
    shards = sorted([load_shard(p) for p in paths], key=lambda s: s["index"])
    require(shards, "No input shards supplied")
    count = shards[0]["count"]
    require(len(shards) == count and [s["index"] for s in shards] == list(range(count)) and
            all(s["count"] == count for s in shards), "Missing or duplicated shard indices")
    require(all(canonical(s["identity"]) == canonical(shards[0]["identity"]) for s in shards),
            "Shard model/runtime/adapter/policy/prompt/implementation identities differ")
    source_root = Path(source_root).resolve() if source_root else Path(__file__).resolve().parents[1] / "data"
    original_sources = verify_original_sources(shards, source_root)
    manifest = copy.deepcopy(shards[0]["manifest"])
    manifest.pop("shard")
    manifest["combined_shards"] = {"count": count, "indices": list(range(count)),
        "source_directories": [str(s["path"]) for s in shards],
        "scope": "Complete disjoint source-order union; inputs, targets and generated text unchanged."}
    source_provenance = [{"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
                         for shard in shards for p in shard["source_files"]]
    source_provenance.extend({"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size,
                              "role": "original_release_source"} for p in original_sources)
    manifest["source_artifacts"] = source_provenance
    combined, all_raw, per_worker = {}, [], []
    for shard in shards:
        rows = sum(len(v["inputs"]) for v in shard["suites"].values())
        elapsed = shard["total_elapsed_s"]
        per_worker.append({"shard_index": shard["index"], "source": str(shard["path"]), "windows": rows,
            "sum_suite_elapsed_s": elapsed, "windows_per_second_during_suites": rows / elapsed if elapsed else None,
            "scope": "One shard worker; measured suite elapsed excludes model loading/setup. Not combined GPU throughput."})
    for name in SUITES:
        parts = [s["suites"][name] for s in shards]
        signatures = [{k: v for k, v in p["manifest"].items() if k not in ("selected_rows", "source")} for p in parts]
        require(all(canonical(v) == canonical(signatures[0]) for v in signatures),
                f"{name}: source hashes, IDs, counts or annotation modes differ")
        rows = [(input_row, prediction_row, raw_row) for part in parts for input_row, prediction_row, raw_row
                in zip(part["inputs"], part["predictions"], part["raw"])]
        rows.sort(key=lambda row: row[0][0]["evaluation_source_index"])
        expected_ids = parts[0]["manifest"]["unsharded_selected_ids"]
        require([r[0][0]["id"] for r in rows] == expected_ids, f"{name}: union has duplicate or missing IDs")
        suite_manifest = next(s for s in manifest["suites"] if s["name"] == name)
        suite_manifest["selected_rows"] = len(rows)
        predictions = [r[1][0] for r in rows]
        metrics = summarize(predictions)
        metrics.update(by_task_type={kind: summarize([r for r in predictions if r.get("task_type", "unspecified") == kind])
                                     for kind in sorted({r.get("task_type", "unspecified") for r in predictions})},
            context_mode=MODES[name], native_batch_size=manifest["native_batch_size"],
            accuracy_interpretation=parts[0]["report"].get("accuracy_interpretation", "See source shard evaluation scope."),
            latency_interpretation="Pooled original per-event service latencies from separate workers; no division by workers or batch size.",
            per_worker_runtime=[{"shard_index": s["index"], "windows": len(s["suites"][name]["inputs"]),
                "elapsed_s": s["suites"][name]["metrics"]["elapsed_s"],
                "windows_per_second": len(s["suites"][name]["inputs"]) / s["suites"][name]["metrics"]["elapsed_s"]
                    if s["suites"][name]["metrics"]["elapsed_s"] else None} for s in shards])
        combined[name] = {"rows": rows, "metrics": metrics}
        all_raw.extend(r[2][1] for r in rows)
    execution = {"workers": per_worker, "single_gpu_throughput": None, "parallel_wall_throughput": None,
        "reason": "No shared cross-worker wall-clock measurement is supplied. Rates are reported separately per worker; no summed or single-GPU rate is inferred."}
    report = {"complete": True, "model_config": manifest["model_config"],
              "suites": {name: result["metrics"] for name, result in combined.items()},
              "execution": execution, "combined_shard_count": count,
              "scope": "Full source-row release evaluation union. Completion is not deployment acceptance."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, result in combined.items():
            (temporary / name).mkdir()
            (temporary / f"{name}-inputs.jsonl").write_bytes(b"".join(r[0][1] for r in result["rows"]))
            (temporary / name / "predictions.jsonl").write_bytes(b"".join(r[1][1] for r in result["rows"]))
            write_json(temporary / name / "metrics.json", result["metrics"])
        (temporary / "native-outputs.jsonl").write_bytes(b"".join(all_raw))
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "report.json", report)
        write_json(temporary / "combination-provenance.json", {"source_artifacts": source_provenance,
            "execution": execution, "verification": "Complete matching-config shard union; original JSONL lines retained in source order."})
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Original data directory; defaults to project/data")
    args = parser.parse_args()
    report = combine_shards(args.shards, args.output, args.source_root)
    print(json.dumps({"output": str(args.output.resolve()), "complete": report["complete"],
        "suite_windows": {name: data["windows"] for name, data in report["suites"].items()},
        "shard_count": report["combined_shard_count"]}, indent=2))


if __name__ == "__main__":
    main()

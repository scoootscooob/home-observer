from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from home_observer.evaluate import summarize

SPEC = importlib.util.spec_from_file_location(
    "combine_release_shards", Path(__file__).parents[1] / "scripts/model_combine_release_shards.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_rows(path, rows):
    # Deliberately retain non-default spacing to test byte-preserving combination.
    path.write_text("".join(json.dumps(row, separators=(", ", ":  ")) + "\n" for row in rows))


def mutate_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    write_json(path, value)


def source_rows(name):
    rows = []
    for position in range(4 if name == "heldout" else 2):
        record_id = f"{name}-{position}"
        target = None if name == "public_video" else {
            "summary": "Authored test label.", "observations": [],
            "actions": ([{"domain": "light", "service": "turn_on", "entity_id": "light.test", "data": {}}]
                        if position % 2 else []), "noop": position % 2 == 0,
        }
        row = {"id": record_id, "recording_id": "public-recording" if name == "public_video" else record_id,
               "task_type": "authored_test", "evaluation_source_index": position,
               "evaluation_context_mode": MODULE.MODES[name],
               "window": {"window_id": "same-pair-input-id" if name == "counterfactual" else record_id}}
        if target is not None:
            row["target"] = target
        if name == "heldout" and position == 3:
            row["task_type"] = "natural_image_summary"
            row["target"] = {"summary": "Existing human caption fixture."}
            row["supervision_mask"] = {"summary": True, "actions": False, "observations": False, "noop": False}
        rows.append(row)
    return rows


def make_shards(tmp_path):
    paths = [tmp_path / "shard0", tmp_path / "shard1"]
    source_root = tmp_path / "source-data"
    for name, relative in MODULE.SOURCE_PATHS.items():
        original_path = source_root / relative
        original_path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(original_path, source_rows(name))
    policy = {"entities": ["light.test"], "rules": ["Authored test rule."]}
    for index, path in enumerate(paths):
        path.mkdir()
        config = {"model_id": "example/test-model", "revision": "pinned-test-revision",
                  "adapter_path": f"/host{index}/adapter", "dataset_root": f"/host{index}/data",
                  "merge_adapter": False, "max_new_tokens": 128}
        manifest = {"model_config": config, "suites": [], "model_revision": config["revision"],
                    "policy": policy, "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
                    "system_prompt_sha256": "a" * 64, "adapter_sha256": "b" * 64,
                    "implementation_sha256": {f"/host{index}/scripts/evaluate_release.py": "c" * 64},
                    "shard": {"index": index, "count": 2}, "native_batch_size": 1}
        report, raw_outputs = {"model_config": config, "suites": {}}, []
        for name in MODULE.SUITES:
            rows = source_rows(name)
            selected = (rows if index == 0 else []) if name == "public_video" else rows[index::2]
            manifest["suites"].append({"name": name, "source": f"/host{index}/data/{name}.jsonl",
                "source_sha256": MODULE.digest(source_root / MODULE.SOURCE_PATHS[name]), "source_rows": len(rows), "selected_rows": len(selected),
                "context_mode": MODULE.MODES[name], "unsharded_selected_ids": [row["id"] for row in rows],
                "targets_changed": False, "window_ids_changed": False})
            write_rows(path / f"{name}-inputs.jsonl", selected)
            folder = path / name
            folder.mkdir()
            predictions = []
            for row in selected:
                bad = name == "counterfactual" and row["evaluation_source_index"] == 1
                metric = {"model_id": config["model_id"], "adapter": config["adapter_path"],
                          "adapter_merged": False, "latency_s": 2 + row["evaluation_source_index"]}
                decision = copy.deepcopy(row.get("target") or {"summary": "Unscored", "observations": [], "actions": [], "noop": True})
                decision.setdefault("actions", [])
                decision.setdefault("observations", [])
                result = {"window_id": row["window"]["window_id"], "total_latency_s": metric["latency_s"] + .125,
                          "metrics": metric}
                result.update({"error": "Invalid original JSON"} if bad else {"decision": decision})
                predictions.append({"id": row["id"], "group_id": row["recording_id"],
                    "task_type": row["task_type"], "target": row.get("target"),
                    "supervision_mask": row.get("supervision_mask", {}), "result": result})
                raw_outputs.append({"window_id": row["window"]["window_id"], "metrics": metric,
                    "raw_output": "{broken original response" if bad else json.dumps(decision), "valid_decision": not bad})
            write_rows(folder / "predictions.jsonl", predictions)
            metrics = summarize(predictions)
            metrics["elapsed_s"] = len(selected) * 5 + .25
            write_json(folder / "metrics.json", metrics)
            report["suites"][name] = {**metrics, "context_mode": MODULE.MODES[name],
                                      "accuracy_interpretation": "Authored test scope."}
        write_rows(path / "native-outputs.jsonl", raw_outputs)
        write_json(path / "manifest.json", manifest)
        write_json(path / "report.json", report)
    return paths


def test_combines_exact_union_preserving_invalid_raw_and_latencies(tmp_path):
    paths = make_shards(tmp_path)
    before = {p: p.read_bytes() for root in paths for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "combined"
    report = MODULE.combine_shards(paths[::-1], output, tmp_path / "source-data")
    assert report["complete"] is True
    assert [report["suites"][n]["windows"] for n in MODULE.SUITES] == [4, 2, 2]
    assert report["suites"]["counterfactual"]["invalid_or_failed_windows"] == 1
    assert report["suites"]["heldout"]["counts"]["action_scored"] == 3
    assert report["suites"]["public_video"]["targeted_windows"] == 0
    assert report["execution"]["single_gpu_throughput"] is None
    assert report["execution"]["parallel_wall_throughput"] is None
    assert len(report["execution"]["workers"]) == 2
    raw = [r for r, _ in MODULE.read_lines(output / "native-outputs.jsonl")]
    assert len(raw) == 8
    pair_raw = [r for r in raw if r["window_id"] == "same-pair-input-id"]
    assert len(pair_raw) == 2 and pair_raw[1]["raw_output"] == "{broken original response"
    for name in MODULE.SUITES:
        originals = {}
        for root in paths:
            originals.update({r["id"]: line for r, line in MODULE.read_lines(root / name / "predictions.jsonl")})
        for row, line in MODULE.read_lines(output / name / "predictions.jsonl"):
            assert line == originals[row["id"]]
        source_input_lines = {r["id"]: line for root in paths for r, line in MODULE.read_lines(root / f"{name}-inputs.jsonl")}
        assert all(line == source_input_lines[r["id"]] for r, line in MODULE.read_lines(output / f"{name}-inputs.jsonl"))
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("failure", ["missing_shard", "duplicate_shard", "missing_raw", "reordered_raw", "partial_report",
                                     "source_hash", "prompt_hash", "model_config", "merge_actual", "target_drift",
                                     "prediction_duplicate", "source_index", "truncated_jsonl", "metric_counts",
                                     "prepared_target_drift", "input_telemetry_drift"])
def test_rejects_incomplete_or_incompatible_shards_without_output(tmp_path, failure):
    paths = make_shards(tmp_path)
    first, second = paths
    if failure == "missing_shard":
        paths = paths[:1]
    elif failure == "duplicate_shard":
        paths = [first, first]
    elif failure == "partial_report":
        mutate_json(second / "report.json", lambda d: d["suites"].pop("public_video"))
    elif failure in ("source_hash", "prompt_hash", "model_config"):
        def change(d):
            if failure == "source_hash":
                d["suites"][0]["source_sha256"] = "e" * 64
            elif failure == "prompt_hash":
                d["system_prompt_sha256"] = "e" * 64
            else:
                d["model_config"]["max_new_tokens"] = 256
        mutate_json(second / "manifest.json", change)
    elif failure in ("missing_raw", "reordered_raw", "merge_actual"):
        rows = [r for r, _ in MODULE.read_lines(second / "native-outputs.jsonl")]
        if failure == "missing_raw":
            rows.pop()
        elif failure == "reordered_raw":
            rows.reverse()
        else:
            rows[0]["metrics"]["adapter_merged"] = True
        write_rows(second / "native-outputs.jsonl", rows)
    elif failure in ("target_drift", "prediction_duplicate"):
        rows = [r for r, _ in MODULE.read_lines(second / "heldout/predictions.jsonl")]
        if failure == "target_drift":
            rows[0]["target"]["summary"] = "Silently changed label."
        else:
            rows[1]["id"] = rows[0]["id"]
        write_rows(second / "heldout/predictions.jsonl", rows)
    elif failure == "source_index":
        rows = [r for r, _ in MODULE.read_lines(second / "heldout-inputs.jsonl")]
        rows[0]["evaluation_source_index"] = 0
        write_rows(second / "heldout-inputs.jsonl", rows)
    elif failure == "metric_counts":
        mutate_json(second / "heldout/metrics.json", lambda d: d["counts"].update(action_tp=100))
    elif failure in ("prepared_target_drift", "input_telemetry_drift"):
        inputs = [r for r, _ in MODULE.read_lines(second / "heldout-inputs.jsonl")]
        if failure == "prepared_target_drift":
            predictions = [r for r, _ in MODULE.read_lines(second / "heldout/predictions.jsonl")]
            inputs[0]["target"]["summary"] = "Matching tampered label in both artifacts."
            predictions[0]["target"]["summary"] = inputs[0]["target"]["summary"]
            write_rows(second / "heldout/predictions.jsonl", predictions)
        else:
            inputs[0]["window"]["device_states"] = {"light.test": "on"}
        write_rows(second / "heldout-inputs.jsonl", inputs)
    else:
        with (second / "heldout/predictions.jsonl").open("a") as stream:
            stream.write('{"id":')
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        MODULE.combine_shards(paths, output, tmp_path / "source-data")
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    paths = make_shards(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("previous evidence")
    with pytest.raises(ValueError, match="already exists"):
        MODULE.combine_shards(paths, output, tmp_path / "source-data")
    assert marker.read_text() == "previous evidence"

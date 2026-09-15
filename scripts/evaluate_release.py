#!/usr/bin/env python3
"""Release evaluation: labeled snapshots and separate unscored chronological replay.

Source rows and targets are retained. Snapshot isolation changes only external
recording metadata so independent annotations are not scored as change-only labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from home_observer.cli import load_policy
from home_observer.engine import Observer
from home_observer.evaluate import evaluate, summarize
from home_observer.ha import SimulatedHomeAssistant
from home_observer.journal import Journal
from home_observer.model import ModelConfig, NativeModel, prepare_request, verify_decision_evidence
from home_observer.prompts import SYSTEM_PROMPT
from home_observer.schema import parse_decision
from home_observer.train import training_target


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_subset(rows, per_task):
    """Choose IDs without predictions, then preserve source order/chronology."""
    if not per_task:
        return rows
    selected = set()
    for kind in sorted({row.get("task_type", "unspecified") for row in rows}):
        candidates = [row for row in rows if row.get("task_type", "unspecified") == kind]
        chosen = sorted(candidates, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())[:per_task]
        selected.update(row["id"] for row in chosen)
    return [row for row in rows if row["id"] in selected]


def select_shard(rows, index, count, context_mode):
    """Split independent snapshots without splitting a chronological recording."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    if context_mode == "chronological_unscored":
        return rows if index == 0 else []
    return [row for position, row in enumerate(rows) if position % count == index]


def prepare_suite(rows, media_root, common_root, policy, *, context_mode, config):
    """Resolve files within the common root and validate everything before model load."""
    prepared, last_times, seen = [], {}, set()
    for source_row in rows:
        row = copy.deepcopy(source_row)
        if row["id"] in seen:
            raise ValueError("duplicate source row ID")
        seen.add(row["id"])
        original_recording = str(row.get("recording_id", row.get("group_id", "default")))
        end = row["window"]["ended_at"]
        if end < last_times.get(original_recording, float("-inf")):
            raise ValueError(f"source recording is out of chronological order: {original_recording}")
        last_times[original_recording] = end
        for media in [*row["window"].get("frames", []), *row["window"].get("audio", [])]:
            path = Path(media["path"])
            media["path"] = str((path if path.is_absolute() else Path(media_root) / path).resolve())
        request = {"window": row["window"], "state": {}, "recent_events": [], "policy": policy.as_dict()}
        # Enforces absolute/symlink containment under common_root, timestamp bounds,
        # media budgets and evidence schema. Source annotations are never input.
        prepare_request(request, config)
        if row.get("target") is not None:
            training_target(row)
        if context_mode == "isolated_snapshot":
            row["evaluation_original_recording_id"] = original_recording
            row["recording_id"] = "snapshot-" + hashlib.sha256(
                (original_recording + "\0" + row["id"]).encode()).hexdigest()[:24]
        elif row.get("target") is not None:
            raise ValueError("chronological release suite must be unscored; do not reinterpret snapshot labels")
        row["evaluation_context_mode"] = context_mode
        prepared.append(row)
    return prepared


class LoggedNativeBackend:
    """Preserve raw outputs and metrics even when the strict Decision parser fails."""
    def __init__(self, backend, log_path):
        self.backend, self.log_path = backend, Path(log_path)

    def observe(self, request):
        raw, metrics = self.backend.generate_text(request)
        entry = {"window_id": request["window"]["window_id"], "raw_output": raw,
                 "metrics": metrics, "valid_decision": False}
        try:
            decision = parse_decision(raw)
            verify_decision_evidence(decision, request)
            entry["valid_decision"] = True
            return {"decision": decision.model_dump(), "metrics": metrics}
        except ValueError as exc:
            entry["error"] = str(exc)
            raise
        finally:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def capture_snapshot_request(window, policy):
    """Build requests through the production engine, with a fresh observation journal."""
    class Capture:
        request = None
        def observe(self, request):
            self.request = copy.deepcopy(request)
            raise RuntimeError("snapshot request capture only")
    capture, journal = Capture(), Journal(":memory:")
    try:
        Observer(capture, journal, policy, SimulatedHomeAssistant()).process(window)
        if capture.request is None:
            raise ValueError("snapshot engine did not produce a request")
        return capture.request
    finally:
        journal.close()


def evaluate_snapshot_batches(rows_path, output_dir, model, policy, raw_log, batch_size):
    """Evaluate independent snapshots in genuine native batches, then normal policy checks."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise ValueError("use a fresh snapshot output directory")
    rows = read_rows(rows_path)
    if any(row.get("evaluation_context_mode") != "isolated_snapshot" for row in rows):
        raise ValueError("batch evaluation supports isolated snapshots only")
    groups = [row["recording_id"] for row in rows]
    if len(set(groups)) != len(groups):
        raise ValueError("every batched snapshot must have an independent journal")
    started, results, journal_stats = time.perf_counter(), [], {}
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        offset = 0
        while offset < len(rows):
            def layout(row):
                return len(row["window"].get("frames", [])), len(row["window"].get("audio", []))
            end = offset + 1
            while end < min(len(rows), offset + batch_size) and layout(rows[end]) == layout(rows[offset]):
                end += 1
            batch = rows[offset:end]
            requests = [capture_snapshot_request(row["window"], policy) for row in batch]
            outputs = model.generate_batch(requests)
            if len(outputs) != len(batch):
                raise ValueError("native batch output count mismatch")
            for row, expected_request, (raw, metrics) in zip(batch, requests, outputs):
                class Measured:
                    def generate_text(self, actual_request):
                        if actual_request != expected_request:
                            raise ValueError("production request differs from the measured snapshot")
                        return raw, {**metrics, "precomputed_snapshot_generation": True}
                group = row["recording_id"]
                group_name = hashlib.sha256(group.encode()).hexdigest()[:16]
                journal = Journal(output / f"journal-{group_name}.sqlite")
                try:
                    observer = Observer(LoggedNativeBackend(Measured(), raw_log), journal, policy, SimulatedHomeAssistant())
                    result = observer.process(row["window"])
                    # The cached policy pass is fast; include the measured FULL batch
                    # service time for every event instead of dividing by batch size.
                    result["total_latency_s"] += metrics["latency_s"]
                    result["latency_includes_precomputed_native_generation"] = True
                    result["batch_formation_wait_included"] = False
                    journal.finish(row["window"]["window_id"], result, result.get("error"))
                    journal_stats[group] = journal.stats()
                finally:
                    journal.close()
                item = {"id": row["id"], "group_id": group, "source": row.get("source"),
                        "task_type": row.get("task_type", "unspecified"), "target": row.get("target"),
                        "supervision_mask": row.get("supervision_mask", {}), "result": result}
                results.append(item)
                predictions.write(json.dumps(item, ensure_ascii=False) + "\n")
                predictions.flush()
                os.fsync(predictions.fileno())
            offset = end
            print(json.dumps({"snapshot_progress": offset, "total": len(rows), "batch_size": len(batch),
                              "full_event_batch_latency_s": outputs[0][1]["latency_s"]}), flush=True)
    report = summarize(results)
    report.update(elapsed_s=time.perf_counter() - started,
                  by_task_type={kind: summarize([r for r in results if r["task_type"] == kind])
                                for kind in sorted({r["task_type"] for r in results})},
                  journals=journal_stats, native_batch_size=batch_size,
                  batch_formation_wait_included=False,
                  latency_interpretation="Independent snapshots are already available. Every event waits the full native batch latency; no streaming deadline claim.")
    report["observed_windows_per_second"] = len(results) / report["elapsed_s"] if report["elapsed_s"] else None
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter")
    merging = parser.add_mutually_exclusive_group()
    merging.add_argument("--merge-adapter", dest="merge_adapter", action="store_true",
                         help="Explicit experimental BF16 merge; validate output parity before use")
    merging.add_argument("--no-merge-adapter", dest="merge_adapter", action="store_false")
    parser.set_defaults(merge_adapter=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", help="Deployed policy JSON; defaults to data-root/fixtures-expanded/policy.json")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--sample-per-task", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4, 8], default=1,
                        help="Native batch size for isolated snapshots; public chronology stays sequential")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Disjoint source-order snapshot partition; public replay belongs wholly to shard0")
    args = parser.parse_args()
    if args.sample_per_task < 0:
        parser.error("sample-per-task must be nonnegative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "native-outputs.jsonl").exists() or any(output.glob("*/predictions.jsonl")):
        parser.error("use a fresh output directory")
    data_root = Path(args.data_root).resolve(strict=True)
    config = ModelConfig(adapter_path=args.adapter, dataset_root=str(data_root),
                         quantization=args.quantization, max_new_tokens=args.max_new_tokens,
                         max_batch_size=max(4, args.batch_size),
                         merge_adapter=args.merge_adapter)
    policy = load_policy(Path(args.policy) if args.policy else data_root / "fixtures-expanded/policy.json")
    specifications = [
        ("heldout", data_root / "mixed/test.jsonl", data_root, "isolated_snapshot"),
        ("counterfactual", data_root / "counterfactuals/pairs.jsonl", data_root / "counterfactuals", "isolated_snapshot"),
        ("public_video", data_root / "public-egolife/replay.jsonl", data_root / "public-egolife", "chronological_unscored"),
    ]
    prepared_suites, manifest = [], []
    for name, source, media_root, mode in specifications:
        rows = read_rows(source)
        original_count = len(rows)
        for position, row in enumerate(rows):
            row["evaluation_source_index"] = position
        if name == "heldout":
            rows = stable_subset(rows, args.sample_per_task)
        unsharded_ids = [row["id"] for row in rows]
        rows = select_shard(rows, args.shard_index, args.shard_count, mode)
        rows = prepare_suite(rows, media_root, data_root, policy, context_mode=mode, config=config)
        path = output / (name + "-inputs.jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        prepared_suites.append((name, path, mode))
        manifest.append({"name": name, "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "source_rows": original_count, "selected_rows": len(rows), "context_mode": mode,
                         "unsharded_selected_ids": unsharded_ids,
                         "targets_changed": False, "window_ids_changed": False})
    (output / "manifest.json").write_text(json.dumps({"model_config": asdict(config), "suites": manifest,
        "policy": policy.as_dict(),
        "model_revision": config.revision,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "policy_sha256": hashlib.sha256(json.dumps(policy.as_dict(), sort_keys=True).encode()).hexdigest(),
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "native_batch_size": args.batch_size,
        "adapter_sha256": (hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
                           if args.adapter else None),
        "implementation_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__), Path("src/home_observer/model.py"), Path("src/home_observer/prompts.py"),
             Path("src/home_observer/engine.py")]},
        }, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"prepared": True, "gpu_executed": False, "suites": manifest}, indent=2))
        return
    model = NativeModel(config)
    backend = LoggedNativeBackend(model, output / "native-outputs.jsonl")
    reports = {}
    for name, prepared, mode in prepared_suites:
        reports[name] = (evaluate_snapshot_batches(prepared, output / name, model, policy,
                            output / "native-outputs.jsonl", args.batch_size)
                         if mode == "isolated_snapshot" and args.batch_size > 1 else
                         evaluate(prepared, output / name, backend, policy))
        reports[name]["context_mode"] = mode
        reports[name]["accuracy_interpretation"] = (
            "Independent annotated snapshot accuracy, with empty prior observation history."
            if mode == "isolated_snapshot" else
            "Chronological journal replay without ground-truth targets; timing/schema validity only, no perception accuracy claim."
        )
        (output / "report.json").write_text(json.dumps({"model_config": asdict(config), "suites": reports}, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()

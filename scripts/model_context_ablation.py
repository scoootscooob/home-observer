#!/usr/bin/env python3
"""Compare empty prior state with duplicated current telemetry on identical snapshots."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from model_merge_benchmark import append

from home_observer.cli import load_policy
from home_observer.evaluate import summarize
from home_observer.journal import Journal
from home_observer.model import ModelConfig, NativeModel, build_messages, verify_decision_evidence
from home_observer.schema import InferenceRequest, parse_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", default="data/mixed/test.jsonl")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--policy", default="data/fixtures-expanded/policy.json")
    parser.add_argument("--row-indices", default="0,1,72,73")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "generations.jsonl").exists():
        parser.error("use a fresh output directory")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    selected = [rows[int(i)] for i in args.row_indices.split(",")]
    policy = load_policy(args.policy)
    config = ModelConfig(adapter_path=args.adapter, dataset_root=args.dataset_root,
                         quantization="none", max_new_tokens=512)
    samples = []
    for row in selected:
        window = row["window"]
        journal = Journal(":memory:")
        try:
            for entity, value in window.get("device_states", {}).items():
                value = value.get("state") if isinstance(value, dict) else value
                journal.update_fact(entity, "state", value, 1.0, window["ended_at"], [f"device:{entity}"], "device")
            duplicated = InferenceRequest(window=window, state=journal.state(window["ended_at"], policy.state_ttl_seconds),
                                          recent_events=[], policy=policy.as_dict()).model_dump()
        finally:
            journal.close()
        empty = copy.deepcopy(duplicated)
        empty["state"] = {}
        if not duplicated["state"]:
            raise ValueError("ablation requires current telemetry to duplicate")
        for mode, request in [("duplicated_current_telemetry", duplicated), ("empty_prior_state", empty)]:
            build_messages(request, config)
            samples.append((mode, row, request))
    model = NativeModel(config)
    reports = {}
    for mode, row, request in samples:
        raw, metrics = model.generate_text(request)
        result = {"metrics": metrics, "total_latency_s": metrics["latency_s"], "rejections": []}
        try:
            decision = parse_decision(raw)
            verify_decision_evidence(decision, request)
            result.update(decision=decision.model_dump(), rejections=policy.validate(decision, InferenceRequest.model_validate(request).window))
        except ValueError as exc:
            result["error"] = str(exc)
        item = {"id": row["id"], "mode": mode, "raw_output": raw, "request": request,
                "task_type": row.get("task_type"), "target": row.get("target"),
                "supervision_mask": row.get("supervision_mask", {}), "result": result}
        append(out / "generations.jsonl", item)
        reports.setdefault(mode, []).append(item)
        print(json.dumps({"id": row["id"], "mode": mode, "raw_output": raw, "metrics": metrics}), flush=True)
    report = {mode: summarize(items) for mode, items in reports.items()}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

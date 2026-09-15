"""Replay evaluation with recording separation and explicit supervision masks."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from .engine import Observer
from .ha import SimulatedHomeAssistant
from .journal import Journal
from .policy import Policy, default_policy


def action_set(actions: list[dict]) -> set[str]:
    return {json.dumps([a["domain"], a["service"], a["entity_id"], a.get("data", {})], sort_keys=True)
            for a in actions}


def observation_set(observations: list[dict]) -> set[str]:
    return {json.dumps([a["entity_id"], a["attribute"], a["value"]], sort_keys=True) for a in observations}


def ratio(n: int | float, d: int | float):
    return n / d if d else None


def summarize(results: list[dict]) -> dict:
    measured, latencies, failure_count = [], [], 0
    counts = dict(action_tp=0, action_fp=0, action_fn=0, obs_tp=0, obs_fp=0, obs_fn=0,
                  action_scored=0, observation_scored=0, no_action_windows=0, false_action_windows=0)
    for item in results:
        result = item["result"]
        failed = bool(result.get("error") or result.get("rejections") or result.get("skipped") or not result.get("decision"))
        failure_count += failed
        if "total_latency_s" in result:
            latencies.append(result["total_latency_s"])
        target = item.get("target")
        mask = item.get("supervision_mask", {})
        predicted = result.get("decision") or {"actions": [], "observations": []}
        if target is not None:
            measured.append(item)
            if mask.get("actions", True):
                counts["action_scored"] += 1
                expected, actual = action_set(target.get("actions", [])), action_set(predicted.get("actions", []))
                counts["action_tp"] += len(expected & actual) if not failed else 0
                counts["action_fp"] += len(actual - expected)
                counts["action_fn"] += len(expected) if failed else len(expected - actual)
                if not expected:
                    counts["no_action_windows"] += 1
                    counts["false_action_windows"] += bool(actual)
            if mask.get("observations", True):
                counts["observation_scored"] += 1
                expected, actual = observation_set(target.get("observations", [])), observation_set(predicted.get("observations", []))
                counts["obs_tp"] += len(expected & actual) if not failed else 0
                counts["obs_fp"] += len(actual - expected)
                counts["obs_fn"] += len(expected) if failed else len(expected - actual)
    sorted_latencies = sorted(latencies)
    def percentile(p):
        return sorted_latencies[min(len(sorted_latencies) - 1, math.ceil(len(sorted_latencies) * p) - 1)] if sorted_latencies else None
    return {
        "windows": len(results), "targeted_windows": len(measured), "invalid_or_failed_windows": failure_count,
        "valid_response_rate": ratio(len(results) - failure_count, len(results)),
        "action_precision": ratio(counts["action_tp"], counts["action_tp"] + counts["action_fp"]),
        "action_recall": ratio(counts["action_tp"], counts["action_tp"] + counts["action_fn"]),
        "observation_precision": ratio(counts["obs_tp"], counts["obs_tp"] + counts["obs_fp"]),
        "observation_recall": ratio(counts["obs_tp"], counts["obs_tp"] + counts["obs_fn"]),
        "false_action_window_rate": ratio(counts["false_action_windows"], counts["no_action_windows"]),
        "latency_p50_s": percentile(.5), "latency_p95_s": percentile(.95),
        "latency_mean_s": statistics.mean(latencies) if latencies else None, "counts": counts,
        "scope": "Proposed-action and observation accuracy on labeled windows; not a household reliability guarantee.",
    }


def evaluate(rows_path, output_dir, backend, policy: Policy | None = None, limit: int | None = None,
             task_type: str | None = None, realtime: bool = False) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise ValueError("Evaluation output already exists; use a fresh directory to avoid mixing runs")
    policy = policy or default_policy()
    observers, journals, results = {}, {}, []
    started = time.perf_counter()
    replay_clocks = {}
    with Path(rows_path).open() as source, (output / "predictions.jsonl").open("w") as predictions:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if task_type and row.get("task_type") != task_type:
                continue
            group = row.get("recording_id", row.get("group_id", "default"))
            if group not in observers:
                import hashlib
                group_name = hashlib.sha256(group.encode()).hexdigest()[:16]
                journal = Journal(output / f"journal-{group_name}.sqlite")
                journals[group] = journal
                observers[group] = Observer(backend, journal, policy, SimulatedHomeAssistant())
            source_time = row["window"]["ended_at"]
            if group not in replay_clocks:
                replay_clocks[group] = [time.perf_counter(), source_time, source_time]
            replay_lag = 0
            if realtime:
                clock, first_source, previous_source = replay_clocks[group]
                if source_time - previous_source > 60:
                    raise ValueError("real-time replay source gap exceeds 60 seconds; split the recording or use accelerated replay")
                arrival = clock + source_time - first_source
                wait = arrival - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                replay_lag = max(0, time.perf_counter() - arrival)
                replay_clocks[group][2] = source_time
            result = observers[group].process(row["window"])
            result["replay_lag_s"] = replay_lag if realtime else None
            item = {"id": row["id"], "group_id": group, "source": row.get("source"),
                    "task_type": row.get("task_type", "unspecified"),
                    "target": row.get("target"), "supervision_mask": row.get("supervision_mask", {}),
                    "result": result}
            results.append(item)
            predictions.write(json.dumps(item) + "\n")
            predictions.flush()
            print(json.dumps({"progress": len(results), "id": row["id"],
                              "latency_s": result.get("total_latency_s"), "error": result.get("error")}), flush=True)
            if limit and len(results) >= limit:
                break
    summary = summarize(results)
    summary["elapsed_s"] = time.perf_counter() - started
    summary["by_task_type"] = {kind: summarize([r for r in results if r["task_type"] == kind])
                                for kind in sorted({r["task_type"] for r in results})}
    summary["journals"] = {group: journal.stats() for group, journal in journals.items()}
    for journal in journals.values():
        journal.close()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

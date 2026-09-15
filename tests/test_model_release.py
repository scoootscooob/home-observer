"""Snapshot batching retains request isolation, target values and event latency."""
import importlib.util
import json
from pathlib import Path

import pytest

from home_observer.policy import Policy

spec = importlib.util.spec_from_file_location("release_script", Path(__file__).parents[1] / "scripts/evaluate_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Native:
    calls = None
    def __init__(self):
        self.calls = []

    def generate_batch(self, requests):
        self.calls.append([request["window"]["window_id"] for request in requests])
        assert all(request["state"] == {} for request in requests)
        return [(json.dumps({"summary": request["window"]["window_id"], "observations": [],
                             "actions": [], "noop": True}),
                 {"latency_s": 2.5, "batch_size": len(requests)}) for request in requests]


def rows():
    return [{"id": str(i), "recording_id": f"snapshot-{i}", "evaluation_context_mode": "isolated_snapshot",
             "task_type": "device_control", "target": {"summary": "Expected summary", "observations": [], "actions": [], "noop": True},
             "window": {"window_id": str(i), "started_at": float(i), "ended_at": i + 0.5,
                        "frames": [], "audio": [], "device_states": {}}} for i in range(3)]


def test_snapshot_batch_preserves_targets_and_full_event_latency(tmp_path):
    source = tmp_path / "source.jsonl"
    original = rows()
    source.write_text("".join(json.dumps(row) + "\n" for row in original))
    native = Native()
    report = release.evaluate_snapshot_batches(source, tmp_path / "results", native, Policy(entities=[]),
                                                tmp_path / "raw.jsonl", 2)
    predictions = [json.loads(line) for line in (tmp_path / "results/predictions.jsonl").read_text().splitlines()]
    raw = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
    assert native.calls == [["0", "1"], ["2"]]
    assert [row["target"] for row in predictions] == [row["target"] for row in original]
    assert [row["result"]["decision"]["summary"] for row in predictions] == ["0", "1", "2"]
    assert all(row["result"]["total_latency_s"] >= 2.5 for row in predictions)
    assert all(row["valid_decision"] for row in raw)
    assert report["valid_response_rate"] == 1
    assert report["latency_p50_s"] >= 2.5


def test_snapshot_batch_rejects_chronological_sessions(tmp_path):
    source = tmp_path / "source.jsonl"
    row = rows()[0]
    row["evaluation_context_mode"] = "chronological_unscored"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="isolated snapshots"):
        release.evaluate_snapshot_batches(source, tmp_path / "results", Native(), Policy(entities=[]),
                                            tmp_path / "raw.jsonl", 2)


def test_release_shards_partition_snapshots_and_keep_replay_whole():
    original = rows()
    left = release.select_shard(original, 0, 2, "isolated_snapshot")
    right = release.select_shard(original, 1, 2, "isolated_snapshot")
    assert [r["id"] for r in left] == ["0", "2"]
    assert [r["id"] for r in right] == ["1"]
    assert sorted(left + right, key=lambda r: r["id"]) == original
    assert release.select_shard(original, 0, 2, "chronological_unscored") == original
    assert release.select_shard(original, 1, 2, "chronological_unscored") == []
    with pytest.raises(ValueError):
        release.select_shard(original, 2, 2, "isolated_snapshot")

"""Causal, bounded replay readers. Targets never enter the replay input."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield item


def validate_window(window: dict[str, Any], max_duration_s: float = 120) -> None:
    """Reject future evidence, duplicate evidence IDs and malformed time ranges."""
    if not isinstance(window.get("window_id"), str) or not window["window_id"]:
        raise ValueError("window_id must be a nonempty string")
    start, end = float(window["started_at"]), float(window["ended_at"])
    if not all(math.isfinite(x) for x in (start, end)) or not 0 < end - start <= max_duration_s:
        raise ValueError("window must have a positive, bounded finite duration")
    seen: set[str] = set()
    for frame in window.get("frames", []):
        if not start <= float(frame["timestamp"]) <= end:
            raise ValueError("frame evidence falls outside the current window")
        if not frame.get("camera_id") or not frame.get("path"):
            raise ValueError("frame requires camera_id and path")
        _check_evidence(frame, seen)
    for audio in window.get("audio", []):
        if not start <= float(audio["started_at"]) < float(audio["ended_at"]) <= end:
            raise ValueError("audio evidence falls outside the current window")
        if not audio.get("microphone_id") or not audio.get("path"):
            raise ValueError("audio requires microphone_id and path")
        _check_evidence(audio, seen)
    if not isinstance(window.get("device_states", {}), dict):
        raise ValueError("device_states must be an object")


def _check_evidence(item: dict[str, Any], seen: set[str]) -> None:
    evidence_id = item.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id or evidence_id in seen:
        raise ValueError("evidence IDs must be unique nonempty strings")
    seen.add(evidence_id)


def resolve_window(window: dict[str, Any], dataset_root: str | Path,
                   check_files: bool = True) -> dict[str, Any]:
    """Copy an input window, resolving media paths without exposing row metadata."""
    result = copy.deepcopy(window)
    validate_window(result)
    root = Path(dataset_root).resolve()
    for item in result.get("frames", []) + result.get("audio", []):
        path = Path(item["path"])
        if not path.is_absolute():
            path = (root / path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("relative media path escapes dataset root")
        if check_files and not path.is_file():
            raise FileNotFoundError(path)
        item["path"] = str(path)
    return result


def iter_replay(path: str | Path, dataset_root: str | Path | None = None,
                check_files: bool = True) -> Iterator[dict[str, Any]]:
    """Yield only ObservationWindows, in file order, checking each recording's clock.

    Training JSONL rows and raw-window JSONL are both accepted. Source targets,
    annotation text and split labels are deliberately not propagated.
    """
    root = Path(dataset_root) if dataset_root is not None else Path(path).parent
    last_end: dict[str, float] = {}
    seen: set[str] = set()
    for row in read_jsonl(path):
        window = row.get("window", row)
        result = resolve_window(window, root, check_files=check_files)
        recording = str(row.get("recording_id", row.get("group_id", "stream")))
        end = float(result["ended_at"])
        if end < last_end.get(recording, float("-inf")):
            raise ValueError(f"non-monotonic replay timestamps in {recording}")
        if result["window_id"] in seen:
            raise ValueError("duplicate replay window_id")
        seen.add(result["window_id"])
        last_end[recording] = end
        yield result


def assert_disjoint_splits(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Require a source/recording/scenario group to belong to exactly one split."""
    groups: dict[str, str] = {}
    recordings: dict[str, str] = {}
    counts: dict[str, int] = {}
    for row in rows:
        split, group = row["split"], str(row["group_id"])
        for mapping, key in ((groups, group), (recordings, str(row.get("recording_id", group)))):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"group or recording leaked across splits: {key}")
            mapping[key] = split
        counts[split] = counts.get(split, 0) + 1
    return counts

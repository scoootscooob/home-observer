import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from home_observer.physical_journal import PhysicalJournal
from home_observer.temporal_buffer import CapturedFrame
from home_observer.tracking_journal import TrackingJournal


def seed(journal, timestamp=1):
    evidence = f"seed:{timestamp}"
    window_id = f"seed-window-{timestamp}"
    window = {
        "window_id": window_id,
        "started_at": timestamp,
        "ended_at": timestamp,
        "clips": [
            {
                "clip_id": window_id,
                "camera_id": "ego",
                "started_at": timestamp,
                "ended_at": timestamp,
                "frames": [{"evidence_id": evidence, "timestamp": timestamp, "path": "seed.jpg"}],
            }
        ],
    }
    decision = {
        "summary": "Bowl observed",
        "objects": [
            {
                "detection_id": "bowl",
                "label": "bowl",
                "timestamp": timestamp,
                "box": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6},
                "location": {"camera_id": "ego", "area": "counter"},
                "frame_evidence_id": evidence,
                "visibility": "visible",
                "confidence": 0.8,
                "uncertainty": "Approximate visible box",
                "evidence_ids": [evidence],
            }
        ],
    }
    return journal.ingest(window, decision)["bowl"]


def frame(index):
    return CapturedFrame("ego", index, float(index), 100 + index, f"source bytes {index}".encode())


def state(track, source, *, uncertain=False, box=None):
    return {
        "track_id": track,
        "label": "bowl",
        "timestamp": source.timestamp,
        "evidence_id": source.evidence_id,
        "visibility": "uncertain" if uncertain else "visible",
        "box": None if uncertain else (box or [0.1, 0.2, 0.4, 0.6]),
        "confidence": 0 if uncertain else 0.8,
        "confidence_kind": "uncalibrated_heuristic",
    }


def test_historical_queries_use_latest_tracking_evidence_and_preserve_uncertainty(tmp_path):
    path = tmp_path / "physical.sqlite"
    physical = PhysicalJournal(path)
    track = seed(physical)
    journal = TrackingJournal(path)
    journal.record([state(track, frame(2))], frame(2))
    journal.record([state(track, frame(4), uncertain=True)], frame(4))
    # A late-arriving older observation must not replace a later uncertainty.
    journal.record([state(track, frame(3), box=[0.2, 0.2, 0.5, 0.6])], frame(3))
    assert journal.latest(1.5) == []
    assert journal.latest(2.5, [track])[0]["box"] == [0.1, 0.2, 0.4, 0.6]
    assert journal.latest(3.5, [track])[0]["box"] == [0.2, 0.2, 0.5, 0.6]
    current = journal.latest(100, [track])[0]
    assert current["timestamp"] == 4 and current["age_seconds"] == 96
    assert current["visibility"] == "uncertain" and current["box"] is None
    assert current["evidence_sha256"] == hashlib.sha256(frame(4).jpeg).hexdigest()
    assert current["details"]["calibrated_confidence"] is None
    assert journal.record([], frame(5)) == 0
    assert journal.latest(100, [track])[0] == current
    journal.close()
    physical.close()


def test_replay_is_idempotent_conflicts_and_unknown_tracks_roll_back_entire_batch(tmp_path):
    path = tmp_path / "physical.sqlite"
    physical = PhysicalJournal(path)
    track, future = seed(physical), seed(physical, 10)
    journal = TrackingJournal(path)
    source = frame(2)
    assert journal.record([state(track, source)], source) == 1
    assert journal.record([state(track, source)], replace(source, captured_at=999)) == 0
    assert journal.latest(2)[0]["captured_at"] == 102
    with pytest.raises(ValueError, match="conflicting content"):
        journal.record([state(track, source, box=[0.2, 0.2, 0.5, 0.6])], source)
    with pytest.raises(ValueError, match="conflicting content"):
        journal.record([state(track, source)], replace(source, jpeg=b"different frame"))
    for identifier in ("phys_unknown", future):
        with pytest.raises(ValueError, match="unknown or future-born"):
            journal.record([state(track, frame(3)), state(identifier, frame(3))], frame(3))
    assert journal.stats()["updates"] == 1
    with pytest.raises(ValueError, match="never an HA"):
        journal.record([state("light.kitchen", frame(3))], frame(3))
    invalid = {**state(track, frame(3), uncertain=True), "box": [0.1, 0.2, 0.4, 0.6]}
    with pytest.raises(ValueError, match="stale visible box"):
        journal.record([invalid], frame(3))
    journal.close()
    physical.close()


def test_capture_and_model_threads_share_helper_without_future_state_leakage(tmp_path):
    path = tmp_path / "physical.sqlite"
    physical = PhysicalJournal(path)
    track = seed(physical)
    journal = TrackingJournal(path)

    def write(index):
        source = frame(index)
        return journal.record([state(track, source)], source)

    def read(index):
        return all(item["timestamp"] <= index for item in journal.latest(index, [track]))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(write if index % 2 == 0 else read, index) for index in range(2, 42)]
        assert all(future.result() for future in futures)
    assert journal.stats()["updates"] == 20
    assert journal.latest(11, [track])[0]["timestamp"] == 10
    assert journal.latest(100, [track], camera_id="different-camera") == []
    journal.close()
    physical.close()

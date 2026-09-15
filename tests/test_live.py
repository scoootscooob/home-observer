"""Real ffmpeg capture plumbing tests. The delayed backend is explicitly NOT AI.

Set OBSERVER_TEST_VIDEO to an existing household clip with audio. The local
EgoLife sample is auto-detected; CI without this optional media skips the slow
capture test. Four repeated inputs test concurrency, not multi-view alignment.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import wave
from collections import deque
from pathlib import Path

import pytest

from home_observer import live
from home_observer.schema import ObservationWindow


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def real_video():
    configured = os.environ.get("OBSERVER_TEST_VIDEO")
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            pytest.fail(f"OBSERVER_TEST_VIDEO does not exist: {candidate}")
        return candidate
    project = Path(__file__).resolve().parents[1]
    candidates = list((project.parents[1] / "work" / "public-egolife" / "raw").rglob("*.mp4"))
    candidates += list((project / "data" / "public-egolife" / "raw").rglob("*.mp4"))
    if not candidates:
        pytest.skip("optional real EgoLife clip not present; set OBSERVER_TEST_VIDEO")
    return candidates[0]


@pytest.mark.parametrize("reject_decision", [False, True])
def test_four_real_camera_processes_audio_and_slow_inference_are_independent(tmp_path, monkeypatch, reject_decision):
    source = real_video()
    cameras, microphones, calls, progress = [], [], [], []
    captured_type, microphone_type = live.Camera, live.Microphone

    class TrackedCamera(captured_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Force retention rollover during five seconds; window copies must
            # remain intact after their capture-ring originals are deleted.
            self.files = deque(maxlen=2)
            cameras.append(self)

    class TrackedMicrophone(microphone_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            microphones.append(self)

    class DelayedBackend:
        def observe(self, request):
            window = request["window"]
            ObservationWindow.model_validate(window)
            hashes = {item["path"]: digest(item["path"]) for item in window["frames"] + window["audio"]}
            before = {camera.camera_id: camera.count for camera in cameras}
            time.sleep(1.35)  # Controlled latency; capture must continue in parallel.
            after = {camera.camera_id: camera.count for camera in cameras}
            for path, original_hash in hashes.items():
                assert digest(path) == original_hash
            progress.append((before, after))
            calls.append({"window": window, "hashes": hashes})
            return {"decision": {"summary": "Test-only delayed backend; no media interpretation.",
                                 "observations": ([{"entity_id": "room.unconfigured", "attribute": "occupied", "value": True,
                                      "confidence": 1, "evidence_ids": [window["frames"][0]["evidence_id"]]}]
                                     if reject_decision else []), "actions": [], "noop": not reject_decision},
                    "metrics": {"backend": "test_delay", "learned": False, "latency_s": 1.35}}

    monkeypatch.setattr(live, "Camera", TrackedCamera)
    monkeypatch.setattr(live, "Microphone", TrackedMicrophone)
    monkeypatch.setattr(live, "RemoteBackend", lambda *args, **kwargs: DelayedBackend())
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({name: str(source) for name in ["kitchen", "entry", "lounge", "garage"]}))
    state = tmp_path / "state.json"
    state.write_text("{}")
    output = tmp_path / "live-output"
    started = time.time()
    try:
        result = live.main(["--sources", str(sources), "--audio", str(source), "--state-file", str(state),
                            "--output", str(output), "--backend", "remote", "--capture-fps", "4",
                            "--interval", "0.1", "--duration", "5", "--retain-windows", "2"])
        ended = time.time()
        assert result == int(reject_decision)
        assert len(cameras) == 4 and len(microphones) == 1
        assert len({camera.process.pid for camera in cameras}) == 4
        assert microphones[0].process.pid not in {camera.process.pid for camera in cameras}
        assert calls and any(len(call["window"]["frames"]) == 4 for call in calls)
        assert any(call["window"]["audio"] for call in calls)
        assert all(any(after[name] > before[name] for name in before) for before, after in progress)
        # Capture timestamp is receipt wall-clock time, not the embedded EgoLife
        # video clock or historical recording date. Every input is causal.
        for call in calls:
            window = ObservationWindow.model_validate(call["window"])
            assert started <= window.ended_at <= ended
            for frame in window.frames:
                assert started <= frame.timestamp <= window.ended_at
            for audio in window.audio:
                assert audio.started_at < audio.ended_at <= window.ended_at
                if call in calls[-2:]:
                    with wave.open(audio.path, "rb") as stream:
                        assert stream.getnchannels() == 1
                        assert stream.getframerate() == 16000
                        assert stream.getnframes() / stream.getframerate() == pytest.approx(2)
            for path, original_hash in call["hashes"].items():
                if call in calls[-2:]:
                    assert digest(path) == original_hash
                else:
                    assert not Path(path).exists()
        report = json.loads((output / "live_report.json").read_text())
        predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        assert report["model_windows"] == len(calls) == len(predictions)
        assert not report["errors"]
        assert report["rejected_windows"] == (len(calls) if reject_decision else 0)
        assert report["accepted_windows"] == (0 if reject_decision else len(calls))
        for prediction, call in zip(predictions, calls, strict=True):
            window = call["window"]
            expected_hashes = {item["evidence_id"]: call["hashes"][item["path"]]
                               for item in window["frames"] + window["audio"]}
            assert prediction["result"]["media"]["sha256_by_evidence"] == expected_hashes
        assert report["actual_windows_per_s"] < report["requested_windows_per_s"]
        assert report["snapshot_retention"]["retained_completed_windows"] == min(2, len(calls))
        assert report["snapshot_retention"]["expired_windows"] == max(0, len(calls) - 2)
        assert report["snapshot_retention"]["active_windows"] == 0
        assert len(list((output / "windows").iterdir())) == min(2, len(calls))
        with sqlite3.connect(output / "journal.sqlite") as database:
            for call in calls:
                row = database.execute("SELECT result_json FROM windows WHERE id=?", (call["window"]["window_id"],)).fetchone()
                result = json.loads(row[0])
                expected = "retained" if call in calls[-2:] else "expired"
                assert result["media"]["status"] == expected
            expired = database.execute("SELECT COUNT(*) FROM events WHERE kind='media_expired'").fetchone()[0]
            assert expired == max(0, len(calls) - 2)
        for camera in cameras:
            evidence = {frame["evidence_id"] for call in calls for frame in call["window"]["frames"]
                        if frame["camera_id"] == camera.camera_id}
            metrics = report["cameras"][camera.camera_id]
            assert metrics["captured"] == camera.count
            assert metrics["model_sampled"] == len(evidence)
            assert metrics["not_model_sampled"] == camera.count - len(evidence)
            assert metrics["not_model_sampled"] > 0
            assert metrics["capture_error"] is None
        for stream in [*cameras, *microphones]:
            assert not stream.thread.is_alive()
            deadline = time.monotonic() + 2
            while stream.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert stream.process.poll() is not None, "orphan ffmpeg process after close"
    finally:
        # A failed assertion must never leave test capture processes running.
        for stream in [*cameras, *microphones]:
            stream.close()
            if stream.process is not None and stream.process.poll() is None:
                stream.process.kill()
                stream.process.wait(timeout=3)


def test_camera_snapshot_rejects_evidence_newer_than_requested_window(tmp_path):
    camera = live.Camera("entry", "unused", tmp_path / "capture", 2, False)
    source = tmp_path / "capture" / "future.jpg"
    source.write_bytes(b"future-frame")
    camera.latest = {"camera_id": "entry", "timestamp": 20.001, "path": str(source), "evidence_id": "entry:1"}
    output = tmp_path / "window"
    output.mkdir()
    assert camera.snapshot(output, now=20, max_age=3) is None


def test_microphone_snapshot_rejects_audio_ending_after_requested_window(tmp_path):
    microphone = live.Microphone("unused", tmp_path / "capture", False)
    source = tmp_path / "capture" / "future.wav"
    source.write_bytes(b"future-audio")
    microphone.latest = {"microphone_id": "home", "started_at": 18.001, "ended_at": 20.001,
                         "path": str(source), "evidence_id": "audio:1"}
    output = tmp_path / "window"
    output.mkdir()
    assert microphone.snapshot(output, now=20) is None


def test_old_camera_and_audio_evidence_is_not_resampled(tmp_path):
    camera = live.Camera("entry", "unused", tmp_path / "camera", 2, False)
    microphone = live.Microphone("unused", tmp_path / "microphone", False)
    camera.latest = {"timestamp": 10}
    microphone.latest = {"ended_at": 10}
    assert camera.snapshot(tmp_path, now=20, max_age=3) is None
    assert microphone.snapshot(tmp_path, now=20) is None


def test_sample_counter_stays_constant_size_and_counts_gaps_and_repeats():
    counter = live.SampleCounter()
    for number in range(2, 200001, 2):
        counter.record(number)
        counter.record(number)
    assert counter.unique_count == 100000
    assert counter.sample_events == 200000
    assert counter.last_capture == 200000
    assert counter.omitted(200005) == 100005
    assert not hasattr(counter, "__dict__")
    assert all(isinstance(getattr(counter, field), int) for field in counter.__slots__)
    with pytest.raises(ValueError, match="nondecreasing"):
        counter.record(1)
    with pytest.raises(ValueError, match="capture total"):
        counter.omitted(100)


def test_retention_protects_active_snapshot_and_bounds_completed_history(tmp_path):
    expired = []
    retention = live.SnapshotRetention(tmp_path / "windows", keep=2,
                                       on_expire=lambda *args: expired.append(args))
    for index in range(30):
        window_id = f"test-{index}"
        folder = retention.begin(window_id)
        frame = folder / "camera.jpg"
        frame.write_bytes(str(index).encode())
        with pytest.raises(ValueError, match="already active"):
            retention.begin("concurrent")
        with pytest.raises(ValueError, match="not active"):
            retention.complete("wrong-id", 10)
        assert frame.read_bytes() == str(index).encode()
        # At most two complete windows plus this inference's immutable snapshot.
        assert len(retention.completed) <= 2
        assert len(list(retention.root.iterdir())) <= 3
        retention.complete(window_id, index + 10)
        assert frame.read_bytes() == str(index).encode()
        assert len(retention.completed) <= 2
        assert len(list(retention.root.iterdir())) <= 2
    assert retention.expired_count == len(expired) == 28
    assert retention.active is None
    assert [item[0] for item in retention.completed] == ["test-28", "test-29"]
    assert all(not item[1].exists() for item in expired)
    empty = retention.begin("empty-unsubmitted-window")
    retention.discard_active()
    assert not empty.exists()
    assert len(retention.completed) == 2


def test_expiry_marks_journal_without_changing_decision_or_error(tmp_path):
    from home_observer.journal import Journal
    journal = Journal(tmp_path / "journal.sqlite")
    try:
        original = {"window_id": "recorded", "decision": {"summary": "Historical description."}, "error": "original-error"}
        journal.begin({"window_id": "recorded", "started_at": 10, "ended_at": 20,
                       "frames": [], "audio": [], "device_states": {}})
        journal.finish("recorded", original, "original-error")
        live.record_media_expiry(journal, "recorded", tmp_path / "deleted", 20)
        record = journal.get_window("recorded")
        assert record["status"] == "error"
        assert record["error"] == "original-error"
        assert record["result"]["decision"] == original["decision"]
        assert record["result"]["media"]["status"] == "expired"
        events = journal.search("snapshot_retention", time.time() + 1)
        assert len(events) == 1 and events[0]["kind"] == "media_expired"
    finally:
        journal.close()


def test_capture_startup_failure_is_reported_and_exits_nonzero(tmp_path, monkeypatch):
    import json

    from home_observer import live
    sources = tmp_path / 'sources.json'
    sources.write_text(json.dumps({'kitchen': 'missing.mp4'}))
    state = tmp_path / 'state.json'
    state.write_text('{}')
    def missing_ffmpeg():
        raise FileNotFoundError('ffmpeg unavailable')
    monkeypatch.setattr(live, 'ffmpeg_binary', missing_ffmpeg)
    output = tmp_path / 'failed-run'
    result = live.main(['--sources', str(sources), '--state-file', str(state), '--output', str(output),
                        '--audio', 'missing.mp4', '--duration', '2', '--backend', 'remote'])
    report = json.loads((output / 'live_report.json').read_text())
    assert result == 1 and report['model_windows'] == 0
    assert report['error_count'] >= 2
    assert 'ffmpeg unavailable' in report['cameras']['kitchen']['capture_error']
    assert 'ffmpeg unavailable' in report['microphone']['capture_error']

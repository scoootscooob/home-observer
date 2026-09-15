"""End-to-end production shape without a GPU: local test backend, exports-only bridge, scripted frontier."""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from home_observer.production_export import ExportStore, scan_export

ROOT = Path(__file__).resolve().parents[1]
LEAK = ["/Users/", "/private/", ".jpg", "manifest", "raw_output", "sequences", "approval", "worktop"]


def synthetic_video(path: Path, *, seconds=2.0, fps=20, size=(160, 120)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frames = int(seconds * fps)
    rng = np.random.default_rng(7)
    background = rng.integers(40, 90, (size[1], size[0], 3), dtype=np.uint8)
    for index in range(frames):
        image = background.copy()
        shift = 0 if index < frames // 2 else min(50, (index - frames // 2) * 4)
        x0, y0 = 40 + shift, 40
        image[y0:y0 + 40, x0:x0 + 40] = (30, 200, 230)
        cv2.circle(image, (x0 + 20, y0 + 20), 8, (255, 255, 255), -1)
        writer.write(image)
    writer.release()


def wait_for(predicate, timeout, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def scripted_frontier(bridge: Path, coordinator_id: str, log: dict):
    """Acts as the sandboxed frontier: creates the watch, acknowledges exports, asks the verifier."""
    inbox, outbox = bridge / "inbox", bridge / "outbox"
    wait_for(lambda: (bridge / "context/responses.json").is_file(), 30)
    context = json.loads((bridge / "context/digital-context.json").read_text())
    command = {"command_id": "frontier-turn-1", "coordinator_id": coordinator_id, "type": "create_watch", "watch": {
        "watch_id": "cue-on-motion", "created_at": time.time(), "ttl_s": 600, "response_id": "notice",
        "response_deadline_s": 5, "max_event_age_s": 30,
        "context": {"commitment_id": context["commitment_id"], "coordinator_id": coordinator_id,
                    "text": "Cue when the tracked object starts moving"},
        "condition": {"type": "event", "kind": "motion_started", "object_label": "cube", "origin": "geometric",
                      "min_confidence": 0.3}}}
    (inbox / "frontier-turn-1.json").write_text(json.dumps(command))
    acked, verified_subject = set(), None
    deadline = time.time() + 90
    while time.time() < deadline and not (bridge / "coordinator/stop").exists():
        for path in sorted(outbox.glob("*.json")):
            export = json.loads(path.read_text())
            if export["export_id"] in acked:
                continue
            acked.add(export["export_id"])
            (inbox / f"ack-{export['export_id'][7:23]}.json").write_text(json.dumps({
                "command_id": f"ack-{export['export_id'][7:23]}", "coordinator_id": coordinator_id,
                "type": "ack_export", "export_id": export["export_id"], "export_sha256": export["export_sha256"]}))
            message = export["message"]
            if export["topic"] == "physical.event" and message["kind"] == "motion_started" and verified_subject is None:
                verified_subject = message["subject_ids"][0]
                (inbox / "verify-1.json").write_text(json.dumps({
                    "command_id": "verify-1", "coordinator_id": coordinator_id, "type": "verify",
                    "verification": {"request_id": "q-moved", "question": "moved_from_initial_location",
                                     "subject_id": verified_subject, "lookback_s": 30}}))
        time.sleep(0.1)
    log["acked"] = acked
    log["verified_subject"] = verified_subject


@pytest.mark.skipif(not hasattr(cv2, "TrackerCSRT_create"), reason="CSRT requires opencv-contrib")
def test_production_runner_keeps_media_private_end_to_end(tmp_path):
    video = tmp_path / "synthetic.mp4"
    synthetic_video(video)
    runtime = {"mode": "production", "inference_backend": "local-test", "model_id": "google/gemma-4-E4B-it",
               "revision": "test", "adapter_path": None, "export_policy": str(ROOT / "configs/production/export-policy.json"),
               "export_key_file": str(tmp_path / "export.key"), "max_frames": 8, "max_new_tokens": 64,
               "output_contract": "judged-fields"}
    (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    (tmp_path / "responses.json").write_text(json.dumps([{"response_id": "notice", "kind": "notify",
        "message": "Object movement detected; checking the step.", "approval_ref": "local test approval"}]))
    (tmp_path / "context.json").write_text(json.dumps({"commitment_id": "step-1", "text": "Lift the cube",
                                                       "coordinator_id": "scripted"}))
    (tmp_path / "seed.json").write_text(json.dumps({"label": "cube", "box": {"x_min": 0.25, "y_min": 0.33,
        "x_max": 0.5, "y_max": 0.67}, "confidence": 0.9, "provenance": "synthetic test seed",
        "uncertainty": "test"}))
    output, bridge = tmp_path / "run", tmp_path / "bridge"
    import hashlib
    coordinator_id = "sandboxed-frontier-" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    log = {}
    thread = threading.Thread(target=scripted_frontier, args=(bridge, coordinator_id, log), daemon=True)
    thread.start()
    command = [sys.executable, str(ROOT / "scripts/run_production.py"), "--video", str(video), "--camera-id", "cam",
               "--runtime", str(tmp_path / "runtime.json"), "--output", str(output), "--bridge", str(bridge),
               "--responses", str(tmp_path / "responses.json"), "--digital-context", str(tmp_path / "context.json"),
               "--seed", str(tmp_path / "seed.json"), "--frontier", "none", "--wait-for-watch-seconds", "30",
               "--post-source-watch-seconds", "3", "--verification-drain-seconds", "10"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=ROOT)
    (bridge / "coordinator").mkdir(exist_ok=True)
    (bridge / "coordinator/stop").write_text("stop")
    thread.join(timeout=10)
    assert completed.returncode == 0, completed.stderr[-3000:]
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    assert summary["captured_frames"] == 40 and summary["errors"] == []
    assert summary["bridge_violations"] == {}
    assert summary["watches"] == [{"watch_id": "cue-on-motion", "status": "verified"}]
    evidence = json.loads((output / "privacy-evidence.json").read_text())
    assert evidence["declaration"]["model_class"] == "LocalTestModel"
    assert evidence["private_outbox_contains_paths"] is True
    assert evidence["export_store"]["blocked"] == 0 and evidence["export_store"]["exports"] >= 4
    assert evidence["export_store"]["acked"] >= 1
    for path in bridge.rglob("*.json"):
        relative = path.relative_to(bridge).as_posix()
        if relative.startswith("context/"):
            continue
        text = path.read_text()
        for marker in LEAK:
            assert marker not in text, (relative, marker)
        assert scan_export(json.loads(text)) == [], relative
    exports = ExportStore(output / "journal.sqlite3")
    topics = {row["topic"] for row in exports.exports()}
    assert {"watch.created", "physical.event", "watch.matched", "watch.triggered", "watch.response_verified",
            "perception.completed"} <= topics
    assert log["verified_subject"] is not None
    verification = [json.loads(r["export_json"]) for r in exports.exports() if r["topic"] == "verification.result"]
    assert verification and verification[0]["message"]["verdict"] in ("confirmed", "contradicted", "unknown")
    assert verification[0]["message"]["question"] == "moved_from_initial_location"
    assert not (output / "verifications-private.jsonl").exists()
    assert (output / "sequences").is_dir() and any((output / "sequences").rglob("*.jpg"))

"""Continuous file/RTSP capture with explicit freshness and skipped-frame accounting.

Capture runs continuously, independent of model latency. The observer samples the
latest frames; all capture-to-model omissions are counted. This is bounded-window
monitoring, not a claim that every captured frame was understood.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .cli import load_policy
from .engine import Observer, RemoteBackend
from .ha import HomeAssistant, SimulatedHomeAssistant
from .journal import Journal


@dataclass(slots=True)
class SampleCounter:
    """Exact unique-frame accounting for a monotonic latest-frame source.

    A source may be sampled repeatedly before its next frame, but it can never
    revisit an earlier capture number. Three integers replace an unbounded set.
    """
    last_capture: int = 0
    unique_count: int = 0
    sample_events: int = 0

    def record(self, capture_number: int) -> None:
        if capture_number < self.last_capture or capture_number <= 0:
            raise ValueError("capture numbers must be positive and nondecreasing")
        if capture_number > self.last_capture:
            self.unique_count += 1
            self.last_capture = capture_number
        self.sample_events += 1

    def omitted(self, captured: int) -> int:
        if captured < self.last_capture:
            raise ValueError("sampled frame exceeds capture total")
        return captured - self.unique_count


class SnapshotRetention:
    """Retain at most N completed windows plus one protected active window.

    Snapshot ownership is local to this run. Unknown directories in a reused
    output are never deleted. Use a fresh output directory for each invocation.
    """
    def __init__(self, root: Path, keep: int = 100, on_expire=None):
        if keep < 1:
            raise ValueError("retain-windows must be at least one")
        self.root, self.keep, self.on_expire = root.resolve(), keep, on_expire
        self.root.mkdir(parents=True, exist_ok=True)
        self.completed = deque()
        self.active = None
        self.expired_count = 0

    def begin(self, window_id: str) -> Path:
        if self.active is not None:
            raise ValueError("a snapshot is already active")
        if not window_id or Path(window_id).name != window_id or window_id in {".", ".."}:
            raise ValueError("invalid snapshot window ID")
        folder = self.root / window_id
        folder.mkdir(exist_ok=False)
        self.active = (window_id, folder)
        return folder

    def complete(self, window_id: str, ended_at: float) -> None:
        if self.active is None or self.active[0] != window_id:
            raise ValueError("cannot complete a snapshot that is not active")
        self.completed.append((window_id, self.active[1], ended_at))
        self.active = None
        while len(self.completed) > self.keep:
            expired_id, folder, capture_ended_at = self.completed[0]
            # A failure leaves the queue entry intact and stops the run rather
            # than quietly claiming a retention limit that was not enforced.
            shutil.rmtree(folder)
            self.completed.popleft()
            self.expired_count += 1
            if self.on_expire is not None:
                self.on_expire(expired_id, folder, capture_ended_at)

    def discard_active(self) -> None:
        """Drop an empty/unsubmitted snapshot, never call during inference."""
        if self.active is not None:
            shutil.rmtree(self.active[1])
            self.active = None

    def report(self) -> dict:
        return {"retained_completed_windows": len(self.completed), "limit": self.keep,
                "active_windows": int(self.active is not None), "expired_windows": self.expired_count,
                "scope": "Snapshot media from this invocation; text audit logs and older runs are retained separately."}


def record_media_expiry(journal: Journal, window_id: str, folder: Path, capture_ended_at: float) -> None:
    """Keep the historical decision, explicitly marking its raw media expired."""
    media = {"status": "expired", "reason": "snapshot_retention", "directory": str(folder),
             "expired_at": time.time(), "capture_ended_at": capture_ended_at}
    journal.event(window_id, media["expired_at"], "media_expired", media)
    historical = journal.get_window(window_id)
    if historical is not None and historical["result"] is not None:
        result = historical["result"]
        result["media"] = media
        journal.finish(window_id, result, historical["error"])


def ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def input_args(source: str, loop: bool) -> list[str]:
    if source.startswith(("rtsp://", "rtsps://")):
        return ["-rtsp_transport", "tcp", "-i", source]
    return (["-re", "-stream_loop", "-1"] if loop else ["-re"]) + ["-i", source]


class Camera:
    def __init__(self, camera_id: str, source: str, output: Path, fps: float, loop: bool):
        self.camera_id, self.source, self.output = camera_id, source, output
        self.fps, self.loop = fps, loop
        self.output.mkdir(parents=True, exist_ok=True)
        self.latest = None
        self.count = 0
        self.error = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.files = deque(maxlen=64)
        self.process = None
        self.thread = threading.Thread(target=self._capture, daemon=True)

    def start(self):
        self.thread.start()

    def _capture(self):
        try:
            command = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", *input_args(self.source, self.loop),
                       "-an", "-vf", f"fps={self.fps},scale=640:-2", "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1"]
            with (self.output / "capture.log").open("wb") as log:
                self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
                buffer = b""
                while not self.stop_event.is_set():
                    block = self.process.stdout.read1(65536)
                    if not block:
                        break
                    buffer += block
                    if len(buffer) > 8 * 1024 * 1024:
                        raise ValueError("camera emitted an oversized/incomplete JPEG")
                    while b"\xff\xd9" in buffer:
                        end = buffer.index(b"\xff\xd9") + 2
                        raw, buffer = buffer[:end], buffer[end:]
                        start = raw.find(b"\xff\xd8")
                        if start < 0:
                            continue
                        with self.lock:
                            self.count += 1
                            timestamp = time.time()
                            path = self.output / f"frame-{self.count:09d}.jpg"
                            path.write_bytes(raw[start:])
                            if len(self.files) == self.files.maxlen:
                                self.files[0].unlink(missing_ok=True)
                            self.files.append(path)
                            self.latest = {"camera_id": self.camera_id, "timestamp": timestamp,
                                           "path": str(path), "evidence_id": f"{self.camera_id}:{self.count}"}
                code = self.process.wait(timeout=5)
                if code and not self.stop_event.is_set():
                    self.error = f"ffmpeg exited {code}; inspect capture.log"
        except Exception as exc:
            self.error = type(exc).__name__ + ": " + str(exc)

    def snapshot(self, destination: Path, now: float, max_age: float) -> dict | None:
        with self.lock:
            if not self.latest or self.latest["timestamp"] > now or now - self.latest["timestamp"] > max_age:
                return None
            item = dict(self.latest)
            path = destination / (self.camera_id + ".jpg")
            shutil.copyfile(item["path"], path)
            item["path"] = str(path)
            return item

    def close(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=5)


class Microphone:
    def __init__(self, source: str, output: Path, loop: bool, chunk_seconds: int = 2):
        self.source, self.output, self.loop, self.chunk_seconds = source, output, loop, chunk_seconds
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.latest, self.process = None, None
        self.error = None
        self.count = 0
        self.files = deque(maxlen=32)
        self.thread = threading.Thread(target=self._capture, daemon=True)

    def start(self):
        self.thread.start()

    def _capture(self):
        try:
            self._capture_audio()
        except Exception as exc:
            self.error = type(exc).__name__ + ": " + str(exc)

    def _capture_audio(self):
        cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", *input_args(self.source, self.loop),
               "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
        with (self.output / "capture.log").open("wb") as log:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
            length = 16000 * 2 * self.chunk_seconds
            while not self.stop_event.is_set():
                samples = self.process.stdout.read(length)
                if len(samples) != length:
                    break
                with self.lock:
                    self.count += 1
                    ended = time.time()
                    path = self.output / f"chunk-{self.count:09d}.wav"
                    with wave.open(str(path), "wb") as audio:
                        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        audio.writeframes(samples)
                    if len(self.files) == self.files.maxlen:
                        self.files[0].unlink(missing_ok=True)
                    self.files.append(path)
                    self.latest = {"microphone_id": "home", "started_at": ended - self.chunk_seconds,
                                   "ended_at": ended, "path": str(path), "evidence_id": f"audio:{self.count}"}

    def snapshot(self, destination: Path, now: float) -> dict | None:
        with self.lock:
            if not self.latest or self.latest["ended_at"] > now or now - self.latest["ended_at"] > self.chunk_seconds * 2:
                return None
            item = dict(self.latest)
            path = destination / "audio.wav"
            shutil.copyfile(item["path"], path)
            item["path"] = str(path)
            return item

    def close(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="JSON map of camera_id to RTSP URL or video file")
    parser.add_argument("--audio", help="Audio/video file or RTSP stream containing the microphone track")
    parser.add_argument("--state-file", help="JSON state for the simulated Home Assistant")
    parser.add_argument("--execute-ha", action="store_true", help="Execute against the explicitly configured Home Assistant")
    parser.add_argument("--ha-url", help="Home Assistant API URL, required with --execute-ha")
    parser.add_argument("--ha-token-env", default="HOME_ASSISTANT_TOKEN")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--capture-fps", type=float, default=2)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--retain-windows", type=int, default=100,
                        help="Retain this many completed snapshot windows, plus any active inference window (default: 100)")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--backend", choices=["native", "remote"], default="native")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--adapter")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    args = parser.parse_args(argv)
    if min(args.capture_fps, args.interval, args.duration) <= 0:
        raise ValueError("capture rate, interval and duration must be positive")
    if args.retain_windows < 1:
        raise ValueError("retain-windows must be at least one")
    if args.execute_ha and not args.ha_url:
        parser.error("--execute-ha requires --ha-url")
    if not args.execute_ha and not args.state_file:
        parser.error("simulation requires --state-file")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text())
    if not 1 <= len(sources) <= 8 or any(not k.replace("_", "").isalnum() for k in sources):
        raise ValueError("use one to eight alphanumeric camera IDs")
    if args.backend == "native":
        from .model import ModelConfig, NativeModel
        backend = NativeModel(ModelConfig(adapter_path=args.adapter, dataset_root=str(output), quantization=args.quantization))
    else:
        backend = RemoteBackend(args.endpoint, os.environ.get("HOME_OBSERVER_API_TOKEN", ""), upload_media=True)
    cameras = [Camera(name, source, output / "capture" / name, args.capture_fps, args.loop)
               for name, source in sources.items()]
    microphone = Microphone(args.audio, output / "capture" / "audio", args.loop) if args.audio else None
    journal = Journal(output / "journal.sqlite")
    executor = (HomeAssistant(args.ha_url, os.environ.get(args.ha_token_env, ""), execute_real=True)
                if args.execute_ha else SimulatedHomeAssistant(args.state_file))
    policy = load_policy(args.policy)
    observer = Observer(backend, journal, policy, executor)
    retention = SnapshotRetention(output / "windows", args.retain_windows,
                                  lambda *values: record_media_expiry(journal, *values))
    run_id = secrets.token_hex(6)
    started, tick, sampled = time.monotonic(), 0, {name: SampleCounter() for name in sources}
    errors, error_count = deque(maxlen=100), 0
    rejected_windows, uncertain_action_windows, accepted_windows = 0, 0, 0
    try:
        for stream in [*cameras, *([microphone] if microphone else [])]:
            stream.start()
        with (output / "predictions.jsonl").open("w") as predictions:
            while time.monotonic() - started < args.duration:
                now = time.time()
                window_id = f"live-{run_id}-{tick:07d}"
                folder = retention.begin(window_id)
                frames = [f for camera in cameras if (f := camera.snapshot(folder, now, max(3, 3 / args.capture_fps)))]
                audio = microphone.snapshot(folder, now) if microphone else None
                if not frames:
                    retention.discard_active()
                    if all(not camera.thread.is_alive() for camera in cameras):
                        break
                    time.sleep(.2)
                    continue
                beginning = min([f["timestamp"] for f in frames] + ([audio["started_at"]] if audio else []))
                window = {"window_id": window_id, "started_at": beginning, "ended_at": now,
                          "frames": frames, "audio": [audio] if audio else [],
                          "device_states": executor.read_states(policy.entities) if args.execute_ha else dict(executor.states)}
                if args.execute_ha:
                    # Telemetry arrives after frame sampling; the envelope ends
                    # after that read, never before the evidence was available.
                    window["ended_at"] = time.time()
                for frame in frames:
                    camera_id, capture_number = frame["evidence_id"].rsplit(":", 1)
                    if camera_id != frame["camera_id"]:
                        raise ValueError("capture evidence ID does not match camera")
                    sampled[camera_id].record(int(capture_number))
                result = observer.process(window)
                from PIL import Image
                encoded_hashes = {item['evidence_id']: hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
                                  for item in frames + ([audio] if audio else [])}
                decoded_frames = {}
                for frame in frames:
                    with Image.open(frame['path']) as image:
                        decoded_frames[frame['evidence_id']] = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
                result["media"] = {"status": "retained", "directory": str(folder),
                                   "retention_limit_windows": args.retain_windows,
                                   "sha256_by_evidence": encoded_hashes,
                                   "decoded_rgb_sha256_by_frame": decoded_frames,
                                   "distinct_decoded_frames": len(set(decoded_frames.values())),
                                   "note": "Check journal result for current retention status; this prediction is historical."}
                journal.finish(window_id, result, result.get("error"))
                predictions.write(json.dumps({"window": window, "result": result}) + "\n")
                predictions.flush()
                # Inference has returned; its copied assets are no longer active.
                retention.complete(window_id, now)
                print(json.dumps({"tick": tick, "frames": len(frames), "audio": bool(audio),
                                  "latency_s": result.get("total_latency_s"), "error": result.get("error")}), flush=True)
                if result.get("error"):
                    errors.append(result["error"])
                    error_count += 1
                rejected = bool(result.get('rejections'))
                uncertain = any(action.get('status') == 'uncertain' for action in result.get('actions', []))
                rejected_windows += rejected
                uncertain_action_windows += uncertain
                accepted_windows += not (result.get('error') or rejected or uncertain)
                tick += 1
                wait = started + tick * args.interval - time.monotonic()
                if wait > 0:
                    time.sleep(min(wait, max(0, args.duration - (time.monotonic() - started))))
    finally:
        for stream in [*cameras, *([microphone] if microphone else [])]:
            stream.close()
        elapsed = time.monotonic() - started
        capture_failures = [f"{c.camera_id}: {c.error or 'no frames captured'}" for c in cameras if c.error or not c.count]
        if microphone and (microphone.error or not microphone.count):
            capture_failures.append(f"microphone: {microphone.error or 'no audio captured'}")
        if not tick:
            capture_failures.append("no observation windows reached inference")
        errors.extend(capture_failures)
        error_count += len(capture_failures)
        report = {"elapsed_s": elapsed, "model_windows": tick, "actual_windows_per_s": tick / elapsed,
                  "accepted_windows": accepted_windows, "rejected_windows": rejected_windows,
                  "uncertain_action_windows": uncertain_action_windows,
                  "accepted_windows_per_s": accepted_windows / elapsed,
                  "requested_windows_per_s": 1 / args.interval, "errors": list(errors), "error_count": error_count,
                  "errors_retained_limit": 100, "snapshot_retention": retention.report(),
                  "cameras": {c.camera_id: {"captured": c.count, "model_sampled": sampled[c.camera_id].unique_count,
                                             "not_model_sampled": sampled[c.camera_id].omitted(c.count),
                                             "sample_events": sampled[c.camera_id].sample_events,
                                             "capture_error": c.error} for c in cameras},
                  "microphone": {"captured_chunks": microphone.count, "capture_error": microphone.error} if microphone else None,
                  "journal": journal.stats(),
                  "scope": "Continuous capture with latest-frame model sampling. Captured frames omitted by the model are counted; no full-rate awareness claim."}
        (output / "live_report.json").write_text(json.dumps(report, indent=2))
        journal.close()
        if args.execute_ha:
            executor.close()
        if hasattr(backend, 'close'):
            backend.close()
    return 1 if error_count or rejected_windows or uncertain_action_windows else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure OpenCV tracking on every actual clip frame; save boxes for inspection.

Only the initial box is supplied. Success means the tracker API returned a box,
not that identity/localization is correct. No semantic events or actions are made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class GuardedAdapter:
    """Run the exact application tracker, including its JPEG/camera-motion work."""

    def __init__(self, scale, fps):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from home_observer.object_tracker import CSRTContinuousTracker
        from home_observer.temporal_buffer import CapturedFrame

        self.frame_type = CapturedFrame
        self.tracker = CSRTContinuousTracker(scale=scale)
        self.fps, self.index = fps, 0
        self.states, self.events = [], []

    def _frame(self, image):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("frame JPEG encoding failed")
        return self.frame_type("ego", self.index + 1, self.index / self.fps, time.time(), encoded.tobytes())

    def init(self, image, box):
        height, width = image.shape[:2]
        x, y, w, h = box
        self.tracker.seed(
            "phys_assessed_object",
            "assessed object",
            np.array([x, y, x + w, y + h]) / [width, height, width, height],
            self._frame(image),
            confidence=0.9,
        )

    def update(self, image):
        self.index += 1
        self.states, self.events = self.tracker.update(self._frame(image))
        track = self.tracker.tracks.get("phys_assessed_object")
        if track is None or track.lost_since is not None:
            return False, None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = track.box * [width, height, width, height]
        return True, (x1, y1, x2 - x1, y2 - y1)


def tracker_factory(name):
    factory = getattr(cv2, f"Tracker{name}_create", None)
    if factory is None:
        raise RuntimeError(
            f"{name} is unavailable; use an isolated opencv-contrib-python-headless environment"
        )
    return factory()


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def run(args, name):
    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        raise ValueError("could not open the actual video clip")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or frame_count < 2:
        capture.release()
        raise ValueError("assessment needs a finite positive source FPS and at least two frames")
    indices = {round(i * (frame_count - 1) / 7) for i in range(8)}
    output = args.output / name.lower()
    output.mkdir(parents=True, exist_ok=False)
    guarded = name == "CSRTGuarded"
    tracker = GuardedAdapter(args.scale, fps) if guarded else tracker_factory(name)
    coordinate_scale = 1.0 if guarded else args.scale
    status_key = "localization_accepted" if guarded else "tracker_reported_success"
    rows, sampled, latencies, decode_times = [], [], [], []
    initial = None
    index = 0
    started = time.perf_counter()
    try:
        while True:
            before = time.perf_counter()
            ok, frame = capture.read()
            decode_s = time.perf_counter() - before
            if not ok:
                break
            decode_times.append(decode_s)
            height, width = frame.shape[:2]
            tracked = (
                cv2.resize(frame, None, fx=coordinate_scale, fy=coordinate_scale)
                if coordinate_scale != 1
                else frame
            )
            before = time.perf_counter()
            if index == 0:
                x1, y1, x2, y2 = args.box
                if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                    raise ValueError("initial box is outside the source image")
                box = tuple(round(x * coordinate_scale) for x in (x1, y1, x2 - x1, y2 - y1))
                tracker.init(tracked, box)
                success = True
                initial = time.perf_counter() - before
                update_s = None
            else:
                success, box = tracker.update(tracked)
                update_s = time.perf_counter() - before
                latencies.append(update_s)
            pixels = [float(x / coordinate_scale) for x in box] if success else None
            row = {
                "frame_index": index,
                "source_time_s": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                status_key: bool(success),
                "bbox_xywh": pixels,
                "update_s": update_s,
                "decode_s": decode_s,
                "calibrated_confidence": None,
            }
            if guarded:
                row["guarded_states"] = tracker.states
                row["guarded_events"] = [
                    {key: value for key, value in event.items() if key not in ("pre_frame", "post_frame")}
                    for event in tracker.events
                ]
            rows.append(row)
            if index in indices:
                shown = frame.copy()
                if pixels:
                    x, y, w, h = map(round, pixels)
                    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 220, 255), 3)
                label = f"{name} frame {index} t={row['source_time_s']:.2f}s accepted={bool(success)}"
                cv2.rectangle(shown, (0, 0), (width, 35), (0, 0, 0), -1)
                cv2.putText(shown, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                sampled.append((index, shown))
            index += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if len(rows) < 2:
        raise ValueError("video decoded fewer than two actual frames")
    (output / "frames.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "frames.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tiles = []
    for sample_index, shown in sampled:
        cv2.imwrite(str(output / f"frame-{sample_index:03d}.jpg"), shown)
        tiles.append(cv2.resize(shown, (427, 240)))
    if tiles:
        while len(tiles) < 8:
            tiles.append(np.zeros_like(tiles[0]))
        contact = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, 8, 2)])
        cv2.imwrite(str(output / "contact-sheet.jpg"), contact)
    result = {
        "tracker": name,
        "status_meaning": "guarded localization accepted" if guarded else "OpenCV update success only",
        "source_fps": fps,
        "source_declared_frames": frame_count,
        "frames_decoded": len(rows),
        "frames_updated": len(latencies),
        "accepted_updates" if guarded else "api_success_updates": sum(row[status_key] for row in rows[1:]),
        "first_rejected_frame" if guarded else "first_api_failure_frame": next(
            (row["frame_index"] for row in rows[1:] if not row[status_key]), None
        ),
        "init_s": initial,
        "update_total_s": sum(latencies),
        "decode_total_s": sum(decode_times),
        "update_median_ms": percentile(latencies, 50) * 1000,
        "update_p95_ms": percentile(latencies, 95) * 1000,
        "update_max_ms": max(latencies) * 1000,
        "update_only_fps": len(latencies) / sum(latencies),
        "full_decode_track_inspection_loop_s": elapsed,
        "full_decode_track_inspection_loop_fps": len(rows) / elapsed,
        "frame_budget_ms": 1000 / fps,
        "updates_exceeding_source_frame_budget": sum(value > 1 / fps for value in latencies),
        "sampled_frames": [value[0] for value in sampled],
        "manual_localization_quality": "pending visual inspection; no supplied tracking ground truth",
        "scope": "Actual continuous frames, one initial assistant-observed box, no reseeding or model detections. "
        "API success does not prove identity or localization; no calibrated confidence or semantic event claim.",
    }
    if guarded:
        result["guarded_tracker"] = tracker.tracker.report()
        result["scope"] += (
            " Guarded timings include JPEG packing, decode, appearance checks and camera compensation; lost tracks stop backend updates."
        )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", type=Path, default=Path("data/temporal-pilot/clips/a890f0b6cb389efafa60.mp4")
    )
    parser.add_argument(
        "--box", type=int, nargs=4, default=[60, 300, 295, 480], metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument(
        "--trackers", nargs="+", choices=["CSRT", "KCF", "MIL", "CSRTGuarded"], default=["CSRT", "KCF", "MIL"]
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/recipe-tracking"))
    args = parser.parse_args()
    if not 0 < args.scale <= 1 or args.threads < 1:
        parser.error("scale must be in (0, 1], threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(0)
    result = {
        "clip": str(args.clip.resolve()),
        "clip_sha256": hashlib.sha256(args.clip.read_bytes()).hexdigest(),
        "initial_box_xyxy": args.box,
        "initial_box_source": "assistant-observed initial box passed on command line; no later correction",
        "opencv": cv2.__version__,
        "contrib_distribution": importlib.metadata.version("opencv-contrib-python-headless"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "scale": args.scale,
        "opencv_threads": args.threads,
        "source_docs": ["https://docs.opencv.org/4.12.0/d2/da2/classcv_1_1TrackerCSRT.html"],
        "trackers": [run(args, name) for name in args.trackers],
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

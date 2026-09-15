#!/usr/bin/env python3
"""Bounded GPU replay benchmark with FIFO backpressure and recoverable request logs.

A source window scheduled every second is NOT a measured processing rate. All due
windows remain accounted for when inference falls behind. Fixtures and real public
replays are reported separately; no images are duplicated to fake camera coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from home_observer.model import DEFAULT_MODEL, DEFAULT_REVISION, ModelConfig


def append_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def arrivals_due(elapsed, interval, total):
    """Independent source clock, including the arrival at time zero."""
    return min(total, max(0, math.floor(max(0, elapsed) / interval) + 1))


def memory_stats():
    import torch

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    }


def power_sampler(path, stop, gpu_index):
    """Optional device-wide telemetry, never confused with this process's usage."""
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            parts = [part.strip() for part in result.stdout.strip().split(",")]
            append_jsonl(
                path,
                {
                    "time": time.time(),
                    "gpu_index": gpu_index,
                    "timestamp": parts[0],
                    "watts": float(parts[1]),
                    "gpu_utilization_percent": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "scope": "whole_GPU_including_other_processes",
                },
            )
        except (OSError, ValueError, subprocess.SubprocessError, IndexError) as exc:
            append_jsonl(path, {"time": time.time(), "error": str(exc), "telemetry_unavailable": True})
            return
        stop.wait(1.0)


def summarize_power(path, started_wall, ended_wall):
    samples = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # Concurrent telemetry may still be appending its last line.
            if started_wall <= row.get("time", 0) <= ended_wall and "watts" in row:
                samples.append(row["watts"])
    return {
        "samples": len(samples),
        "mean_watts": sum(samples) / len(samples) if samples else None,
        "p95_watts": percentile(samples, 0.95),
        "max_watts": max(samples, default=None),
        "scope": "whole_GPU_including_other_processes",
    }


def track_processor_inputs(backend):
    """Record the actual returned tensors without extra encoding or changing them."""
    original = backend.processor.apply_chat_template

    def tracked(*args, **kwargs):
        batch = original(*args, **kwargs)
        ids = batch["input_ids"]
        image_id = backend.processor.image_token_id
        audio_id = getattr(backend.processor, "audio_token_id", None)
        pixel_values = batch.get("pixel_values")
        backend._benchmark_encoding = {
            "image_items": int(pixel_values.shape[0]) if pixel_values is not None else 0,
            "vision_token_count": int((ids == image_id).sum().item()),
            "audio_token_count": int((ids == audio_id).sum().item()) if audio_id is not None else 0,
            "input_tokens": int(ids.shape[-1]),
        }
        return batch

    backend.processor.apply_chat_template = tracked


def execute_request(backend, request, metadata, path, events):
    """One native generation; parse separately so invalid output remains available."""
    from home_observer.model import verify_decision_evidence
    from home_observer.schema import parse_decision

    started = time.monotonic()
    result = {
        **metadata,
        "request": request,
        "started_wall_time": time.time(),
        "memory_before": memory_stats(),
        "valid_decision": False,
    }
    events.put({"event": "request_started", "window_id": request["window"]["window_id"], "time": started})
    try:
        raw, metrics = backend.generate_text(request)
        result.update(raw_output=raw, metrics=metrics, inference_completed=True)
        encoding = dict(backend._benchmark_encoding)
        result["processor_encoding"] = encoding
        if encoding["image_items"] != len(request["window"].get("frames", [])):
            raise RuntimeError("actual encoded image count differs from scheduled camera frames")
        # NativeModel passes every listed image through the processor and rejects
        # over-budget requests. A successful generation therefore confirms these
        # frame inputs reached inference; nominal source frames are never counted.
        result["encoded_frames_by_camera"] = dict(
            Counter(f["camera_id"] for f in request["window"].get("frames", []))
        )
        result["encoded_audio_chunks"] = len(request["window"].get("audio", []))
        try:
            decision = parse_decision(raw)
            verify_decision_evidence(decision, request)
            result.update(valid_decision=True, decision=decision.model_dump())
        except ValueError as exc:
            result["validation_error"] = str(exc)
    except Exception as exc:
        result.update(inference_completed=False, error=f"{type(exc).__name__}: {exc}")
    result["observed_service_s"] = time.monotonic() - started
    result["memory_after"] = memory_stats()
    result["finished_wall_time"] = time.time()
    append_jsonl(path, result)
    events.put(
        {"event": "request_finished", "window_id": request["window"]["window_id"], "time": time.monotonic()}
    )
    return result


def summarize_case(case, records, *, total, due, interval, elapsed, reason, pending_ids):
    measured = [r for r in records if not r.get("warmup")]
    success = [r for r in measured if r.get("inference_completed")]
    metrics = [r["metrics"] for r in success]
    service = [r["observed_service_s"] for r in measured]
    preprocess = [m["preprocess_s"] for m in metrics]
    inference = [max(0, m["latency_s"] - m["preprocess_s"]) for m in metrics]
    lag = [r["queue_wait_s"] + r["observed_service_s"] for r in measured]
    frames = Counter()
    for record in success:
        frames.update(record["encoded_frames_by_camera"])
    memory = [r["memory_after"] for r in measured]
    beyond_interval = sum(t > interval for t in service)
    pending = max(0, due - len(measured))
    can_keep_pace = (
        bool(success)
        and not pending
        and not beyond_interval
        and len(success) == len(measured)
        and max(lag, default=0) <= interval
        and len(success) / elapsed >= 0.99 / interval
    )
    return {
        "case": case,
        "stop_reason": reason,
        "requested_arrival_interval_s": interval,
        "nominal_source_windows_per_second": 1 / interval,
        "scheduled_windows": total,
        "arrived_windows": due,
        "attempted_windows": len(measured),
        "completed_inference_windows": len(success),
        "invalid_decisions": sum(not r["valid_decision"] for r in measured),
        "pending_arrived_windows": pending,
        "not_yet_arrived_windows": total - due,
        "unprocessed_window_ids": pending_ids,
        "silently_dropped_windows": 0,
        "elapsed_s": elapsed,
        "observed_windows_per_second": len(success) / elapsed if elapsed else 0,
        "encoded_frames_by_camera": dict(frames),
        "encoded_vision_tokens": sum(r["processor_encoding"]["vision_token_count"] for r in success),
        "encoded_audio_tokens": sum(r["processor_encoding"]["audio_token_count"] for r in success),
        "observed_encoded_fps_by_camera": {k: v / elapsed for k, v in frames.items()} if elapsed else {},
        "four_camera_coverage": bool(success)
        and all(len(r["encoded_frames_by_camera"]) == 4 for r in success),
        "latency_s": {
            "p50": percentile(service, 0.50),
            "p95": percentile(service, 0.95),
            "max": max(service, default=None),
        },
        "preprocess_s": {"p50": percentile(preprocess, 0.50), "p95": percentile(preprocess, 0.95)},
        "inference_s": {"p50": percentile(inference, 0.50), "p95": percentile(inference, 0.95)},
        "arrival_to_completion_s": {"p50": percentile(lag, 0.50), "p95": percentile(lag, 0.95)},
        "max_backlog_windows": max([r["backlog_at_start"] for r in measured] + [pending]),
        "requests_slower_than_arrival_interval": beyond_interval,
        "can_keep_pace": can_keep_pace,
        "pace_verdict": "kept_up_for_this_bounded_run"
        if can_keep_pace
        else "cannot_claim_source_rate_processing",
        "input_tokens": {
            "sum": sum(m["input_tokens"] for m in metrics),
            "p50": percentile([m["input_tokens"] for m in metrics], 0.5),
        },
        "output_tokens": {
            "sum": sum(m["output_tokens"] for m in metrics),
            "p50": percentile([m["output_tokens"] for m in metrics], 0.5),
        },
        "memory_trend_gb": {
            key: {
                "first": memory[0][key] if memory else None,
                "last": memory[-1][key] if memory else None,
                "min": min((m[key] for m in memory), default=None),
                "max": max((m[key] for m in memory), default=None),
                "last_minus_first": memory[-1][key] - memory[0][key] if memory else None,
            }
            for key in ("allocated_gb", "reserved_gb")
        },
        "max_peak_allocated_gb": max((m["peak_allocated_gb"] for m in metrics), default=None),
    }


def worker(config_values, cases, options, output, events):
    """Own exactly one NativeModel, CUDA context, and model weight load per run."""
    stop = threading.Event()
    telemetry = None
    try:
        from home_observer.model import NativeModel

        backend = NativeModel(ModelConfig(**config_values))
        track_processor_inputs(backend)
        events.put({"event": "loaded", "time": time.monotonic()})
        if options["power"]:
            telemetry = threading.Thread(
                target=power_sampler,
                args=(Path(output) / "power.jsonl", stop, options["gpu_index"]),
                daemon=True,
            )
            telemetry.start()
        summaries = []
        requests_path = Path(output) / "requests.jsonl"
        for case in cases:
            rows = case["rows"]
            for index in range(options["warmup_windows"]):
                execute_request(
                    backend,
                    rows[index % len(rows)]["request"],
                    {"case": case["name"], "source_kind": case["source_kind"], "warmup": True},
                    requests_path,
                    events,
                )
            started = time.monotonic()
            started_wall = time.time()
            source_span = len(rows) * options["interval"]
            deadline = started + source_span + options["drain_s"]
            records, next_index, due = [], 0, 0
            reason = "all_scheduled_windows_processed"
            while next_index < len(rows):
                now = time.monotonic()
                due = arrivals_due(now - started, options["interval"], len(rows))
                if now >= deadline:
                    reason = "drain_deadline"
                    break
                backlog = due - next_index
                if backlog > options["max_backlog"]:
                    reason = "backpressure_limit"
                    break
                if next_index >= due:
                    time.sleep(min(0.1, started + next_index * options["interval"] - now))
                    continue
                row = rows[next_index]
                queue_wait = max(0, now - (started + next_index * options["interval"]))
                record = execute_request(
                    backend,
                    row["request"],
                    {
                        "case": case["name"],
                        "source_kind": case["source_kind"],
                        "source": row.get("source"),
                        "source_row_id": row.get("id"),
                        "index": next_index,
                        "warmup": False,
                        "scheduled_arrival_elapsed_s": next_index * options["interval"],
                        "queue_wait_s": queue_wait,
                        "backlog_at_start": backlog,
                    },
                    requests_path,
                    events,
                )
                records.append(record)
                next_index += 1
                if not record.get("inference_completed"):
                    reason = "inference_error"
                    break
            if next_index == len(rows):
                # Observe the complete final source interval rather than inflate
                # throughput by counting its frame at the beginning of the interval.
                remaining = started + source_span - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            elapsed = time.monotonic() - started
            due = arrivals_due(elapsed, options["interval"], len(rows))
            # Include the source interval after the last scheduled frame when the
            # model is faster than source; rates must not exceed nominal due to
            # dividing N arrivals by just (N-1) intervals.
            reporting_elapsed = max(elapsed, source_span) if next_index == len(rows) else elapsed
            summary = summarize_case(
                case["name"],
                records,
                total=len(rows),
                due=due,
                interval=options["interval"],
                elapsed=reporting_elapsed,
                reason=reason,
                pending_ids=[r["request"]["window"]["window_id"] for r in rows[next_index:due]],
            )
            summary["power"] = (
                summarize_power(Path(output) / "power.jsonl", started_wall, time.time())
                if options["power"]
                else None
            )
            summary["source_kind"] = case["source_kind"]
            summary["wall_processing_elapsed_s"] = elapsed
            summaries.append(summary)
            append_jsonl(Path(output) / "cases.jsonl", summary)
            events.put({"event": "case_finished", "summary": summary, "time": time.monotonic()})
        (Path(output) / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "model_loads": 1,
                    "model_config": config_values,
                    "options": options,
                    "cases": summaries,
                    "interpretation": "Measured replay. Fixture timing is not real-video accuracy. No infinite-state claim.",
                },
                indent=2,
            )
            + "\n"
        )
        events.put({"event": "finished", "time": time.monotonic()})
    except BaseException as exc:
        append_jsonl(Path(output) / "errors.jsonl", {"error": str(exc), "traceback": traceback.format_exc()})
        events.put({"event": "error", "error": str(exc), "time": time.monotonic()})
        raise
    finally:
        stop.set()
        if telemetry:
            telemetry.join(timeout=3)


def fixture_cases(output, names, total, interval, policy, with_audio):
    """Procedural sensor fixtures. Four independent camera views; no real-media claims."""
    from PIL import Image, ImageDraw

    root = Path(output) / "fixture_media"
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_name in names:
        rows = []
        for index in range(total):
            timestamp = 1700000000.0 + index * interval
            frames = []
            for camera in range(4):
                phase = index if case_name == "moving" else 0
                filename = f"{case_name}-cam{camera}-{phase:05d}.png"
                path = root / filename
                if not path.exists():
                    image = Image.new("RGB", (384, 256), (20 + camera * 30, 25 + phase * 7 % 100, 40))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((12, 25, 372, 246), outline=(230, 230, 230), width=3)
                    for shape in range(8):
                        x = (phase * (7 + shape) + camera * 41 + shape * 43) % 310 + 12
                        y = (phase * (3 + shape) + camera * 19 + shape * 17) % 195 + 30
                        draw.rectangle(
                            (x, y, x + 34, y + 22), fill=(90 + shape * 20, 180 - camera * 25, 35 + shape * 20)
                        )
                    draw.text((16, 8), f"SYNTHETIC CAMERA {camera} {case_name.upper()}", fill="white")
                    image.save(path)
                frames.append(
                    {
                        "camera_id": f"camera_{camera}",
                        "timestamp": timestamp,
                        "path": str(path.resolve()),
                        "evidence_id": f"{case_name}:{index}:camera{camera}",
                    }
                )
            audio = []
            if with_audio:
                import wave

                audio_path = root / "synthetic-silence-1s.wav"
                if not audio_path.exists():
                    with wave.open(str(audio_path), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(b"\0\0" * 16000)
                audio = [
                    {
                        "microphone_id": "fixture_microphone",
                        "started_at": timestamp - 1,
                        "ended_at": timestamp,
                        "path": str(audio_path.resolve()),
                        "evidence_id": f"{case_name}:{index}:microphone",
                    }
                ]
            req = {
                "window": {
                    "window_id": f"{case_name}-{index:05d}",
                    "started_at": timestamp - max(1, interval),
                    "ended_at": timestamp,
                    "frames": frames,
                    "audio": audio,
                    "device_states": {},
                },
                "state": {},
                "recent_events": [],
                "policy": policy,
            }
            rows.append(
                {
                    "request": req,
                    "id": req["window"]["window_id"],
                    "source": {
                        "type": "procedural_geometry",
                        "case": case_name,
                        "all_cameras_change_pixels": case_name == "moving",
                        "audio": "synthetic_silence" if with_audio else "none",
                    },
                }
            )
        cases.append({"name": case_name, "source_kind": "synthetic_fixture", "rows": rows})
    return cases, root


def input_case(path, kind, total, policy):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # Targets, annotations, captions and teacher outputs never enter requests.
            req = {"window": row["window"], "state": {}, "recent_events": [], "policy": policy}
            rows.append({"request": req, "id": row.get("id"), "source": row.get("source")})
            if len(rows) >= total:
                break
    if not rows:
        raise ValueError("empty replay input")
    return [{"name": "input_replay", "source_kind": kind, "rows": rows}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prepare-only", action="store_true", help="write source manifest/media without loading model"
    )
    parser.add_argument("--adapter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-soft-tokens", type=int, default=140)
    parser.add_argument(
        "--duration-s", type=float, default=60, help="nominal source duration per case, <=3600"
    )
    parser.add_argument("--arrival-interval-s", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=600, help="finite source bound per case, <=10000")
    parser.add_argument("--max-backlog", type=int, default=120)
    parser.add_argument("--drain-s", type=float, default=30)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--load-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=120)
    parser.add_argument("--power", action="store_true")
    parser.add_argument(
        "--gpu-index", type=int, default=0, help="nvidia-smi physical GPU index for telemetry"
    )
    parser.add_argument("--fixture-cases", default="quiet,moving")
    parser.add_argument("--fixture-audio", action="store_true", help="add native 1s synthetic silence input")
    parser.add_argument("--input", help="JSONL windows, used as-is; never duplicated to pretend four cameras")
    parser.add_argument("--dataset-root")
    parser.add_argument("--source-kind", choices=["real_public", "private_recording", "synthetic_fixture"])
    parser.add_argument("--policy")
    args = parser.parse_args()
    if not 0 < args.duration_s <= 3600 or not 0.1 <= args.arrival_interval_s <= 60:
        parser.error("duration must be (0,3600], interval [0.1,60]")
    if not 1 <= args.max_windows <= 10000 or args.max_backlog < 1:
        parser.error("max-windows must be [1,10000] and max-backlog positive")
    if not 0 <= args.drain_s <= 3600 or not 0 <= args.warmup_windows <= 10:
        parser.error("drain must be [0,3600], warmup [0,10]")
    if not 1 <= args.request_timeout_s <= 600 or not 1 <= args.load_timeout_s <= 3600:
        parser.error("request timeout must be [1,600], load timeout [1,3600]")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "requests.jsonl").exists():
        parser.error("output already contains requests.jsonl; use a fresh directory")
    requested_total = math.ceil(args.duration_s / args.arrival_interval_s)
    total = min(args.max_windows, requested_total)
    policy = (
        json.loads(Path(args.policy).read_text())
        if args.policy
        else {"allowed_entities": [], "allowed_services": []}
    )
    if args.input:
        if not args.dataset_root or not args.source_kind:
            parser.error("input replay requires --dataset-root and explicit --source-kind")
        cases = input_case(args.input, args.source_kind, total, policy)
        root = Path(args.dataset_root).resolve()
    else:
        names = args.fixture_cases.split(",")
        if any(name not in {"quiet", "moving"} for name in names) or len(set(names)) != len(names):
            parser.error("fixture-cases must contain quiet and/or moving once each")
        cases, root = fixture_cases(output, names, total, args.arrival_interval_s, policy, args.fixture_audio)
    config = ModelConfig(
        model_id=args.model,
        revision=args.revision,
        adapter_path=args.adapter,
        dataset_root=str(root),
        quantization=args.quantization,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_soft_tokens=args.max_soft_tokens,
    )
    options = {
        "interval": args.arrival_interval_s,
        "drain_s": args.drain_s,
        "max_backlog": args.max_backlog,
        "warmup_windows": args.warmup_windows,
        "power": args.power,
        "gpu_index": args.gpu_index,
    }
    manifest = {
        "model_config": asdict(config),
        "options": vars(args),
        "requested_windows_per_case": requested_total,
        "bounded_windows_per_case": total,
        "source_mode": "prerecorded_replay_independent_arrival_clock",
        "cases": [
            {"name": c["name"], "source_kind": c["source_kind"], "windows": len(c["rows"])} for c in cases
        ],
    }
    if args.input:
        digest = hashlib.sha256()
        with open(args.input, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["input_sha256"] = digest.hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for case in cases:
        for row in case["rows"]:
            append_jsonl(output / "scheduled_windows.jsonl", {"case": case["name"], **row})
    if args.prepare_only:
        print(
            json.dumps(
                {"prepared": True, "gpu_executed": False, "manifest": str(output / "manifest.json")}, indent=2
            )
        )
        return
    context = mp.get_context("spawn")
    events = context.Queue()
    child = context.Process(target=worker, args=(asdict(config), cases, options, str(output), events))
    child.start()
    launched = time.monotonic()
    loaded, request_started = None, None
    failure = None
    # Includes all warmups, source spans, drain grace, and one bounded in-flight
    # request per case. Parent terminates CUDA child if a request hangs.
    max_run_s = (
        sum(
            len(c["rows"]) * args.arrival_interval_s
            + args.drain_s
            + (args.warmup_windows + 1) * args.request_timeout_s
            for c in cases
        )
        + 30
    )
    while child.is_alive():
        try:
            event = events.get(timeout=0.25)
            append_jsonl(output / "lifecycle.jsonl", event)
            if event["event"] == "loaded":
                loaded = time.monotonic()
            elif event["event"] == "request_started":
                request_started = time.monotonic()
            elif event["event"] == "request_finished":
                request_started = None
            elif event["event"] == "error":
                failure = event["error"]
        except queue.Empty:
            pass
        now = time.monotonic()
        if loaded is None and now - launched > args.load_timeout_s:
            failure = "model_load_timeout"
        elif request_started is not None and now - request_started > args.request_timeout_s:
            failure = "request_timeout"
        elif loaded is not None and now - loaded > max_run_s:
            failure = "total_run_timeout"
        if failure:
            child.terminate()
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
            break
    child.join(timeout=5)
    if failure or child.exitcode != 0:
        interruption = {
            "completed": False,
            "error": failure or f"child_exit_{child.exitcode}",
            "recoverable_logs": ["requests.jsonl", "scheduled_windows.jsonl", "lifecycle.jsonl"],
            "note": "Scheduled windows absent from completed request logs remain unprocessed, not dropped.",
        }
        (output / "interrupted.json").write_text(json.dumps(interruption, indent=2) + "\n")
        raise SystemExit(f"Benchmark interrupted: {interruption['error']}; see {output}")
    print(json.dumps({"completed": True, "report": str(output / "benchmark_report.json")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare batch 1/2/4 throughput and actual per-event latency on one native model.

Events in a batch are assumed ready together. Report full batch time as event
latency; batch_time / batch_size is only amortized capacity, never event latency.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from model_benchmark import (
    append_jsonl,
    fixture_cases,
    input_case,
    memory_stats,
    percentile,
    power_sampler,
    summarize_power,
)

from home_observer.model import DEFAULT_MODEL, DEFAULT_REVISION, ModelConfig


def worker(config_dict, cases, options, output, events):
    stop = threading.Event()
    power = None
    try:
        from home_observer.model import NativeModel, verify_decision_evidence
        from home_observer.schema import parse_decision

        backend = NativeModel(ModelConfig(**config_dict))
        events.put({"event": "loaded", "time": time.monotonic()})
        if options["power"]:
            power = threading.Thread(
                target=power_sampler,
                args=(Path(output) / "power.jsonl", stop, options["gpu_index"]),
                daemon=True,
            )
            power.start()
        comparisons = []
        for case in cases:
            for size in options["batch_sizes"]:
                records, cursor = [], 0
                started_wall = time.time()
                for iteration in range(options["warmup_batches"] + options["iterations"]):
                    chosen = case["rows"][cursor : cursor + size]
                    cursor += size
                    if len(chosen) != size:
                        raise ValueError("insufficient unique source windows for batch comparison")
                    requests = [row["request"] for row in chosen]
                    start = time.monotonic()
                    events.put(
                        {"event": "batch_started", "batch_size": size, "case": case["name"], "time": start}
                    )
                    record = {
                        "case": case["name"],
                        "source_kind": case["source_kind"],
                        "batch_size": size,
                        "iteration": iteration,
                        "warmup": iteration < options["warmup_batches"],
                        "memory_before": memory_stats(),
                        "windows": [],
                        "inference_completed": False,
                        "input_window_ids": [r["window"]["window_id"] for r in requests],
                    }
                    try:
                        outputs = backend.generate_batch(requests)
                        record["inference_completed"] = True
                        for (raw, metrics), request, row in zip(outputs, requests, chosen):
                            result = {
                                "window_id": request["window"]["window_id"],
                                "raw_output": raw,
                                "metrics": metrics,
                                "valid_decision": False,
                                "request": request,
                                "source": row.get("source"),
                            }
                            try:
                                decision = parse_decision(raw)
                                verify_decision_evidence(decision, request)
                                result.update(valid_decision=True, decision=decision.model_dump())
                            except ValueError as exc:
                                result["error"] = str(exc)
                            record["windows"].append(result)
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["batch_wall_s"] = time.monotonic() - start
                    record["memory_after"] = memory_stats()
                    append_jsonl(Path(output) / "batches.jsonl", record)
                    events.put({"event": "batch_finished", "time": time.monotonic()})
                    if not record["warmup"]:
                        records.append(record)
                    if not record["inference_completed"]:
                        raise RuntimeError(record["error"])
                latencies = [record["batch_wall_s"] for record in records]
                windows = [window for record in records for window in record["windows"]]
                model_metrics = [record["windows"][0]["metrics"] for record in records]
                total_time = sum(latencies)
                summary = {
                    "case": case["name"],
                    "source_kind": case["source_kind"],
                    "batch_size": size,
                    "batches": len(records),
                    "windows": len(windows),
                    "wall_processing_s": total_time,
                    "observed_windows_per_second": len(windows) / total_time,
                    "per_event_service_latency_s": {
                        "p50": percentile(latencies, 0.5),
                        "p95": percentile(latencies, 0.95),
                    },
                    "batch_formation_wait_included": False,
                    "consecutive_1Hz_windows_additional_wait_s": {
                        "first_event": size - 1,
                        "mean": (size - 1) / 2,
                    },
                    "amortized_seconds_per_window": total_time / len(windows),
                    "batch_preprocess_s": {
                        "p50": percentile([m["preprocess_s"] for m in model_metrics], 0.5),
                        "p95": percentile([m["preprocess_s"] for m in model_metrics], 0.95),
                    },
                    "batch_inference_s": {
                        "p50": percentile([m["batch_inference_s"] for m in model_metrics], 0.5),
                        "p95": percentile([m["batch_inference_s"] for m in model_metrics], 0.95),
                    },
                    "invalid_windows": sum(not w["valid_decision"] for w in windows),
                    "input_tokens": sum(w["metrics"]["input_tokens"] for w in windows),
                    "input_padding_tokens": sum(w["metrics"]["input_padding_tokens"] for w in windows),
                    "output_tokens": sum(w["metrics"]["output_tokens"] for w in windows),
                    "vision_tokens": sum(w["metrics"]["vision_tokens"] for w in windows),
                    "audio_tokens": sum(w["metrics"]["audio_tokens"] for w in windows),
                    "max_peak_allocated_gb": max(w["metrics"]["peak_allocated_gb"] for w in windows),
                    "max_reserved_gb": max(r["memory_after"]["reserved_gb"] for r in records),
                    "power": summarize_power(Path(output) / "power.jsonl", started_wall, time.time())
                    if options["power"]
                    else None,
                    "latency_interpretation": "Each co-arriving event waits full batch time. Amortized time measures capacity only.",
                }
                comparisons.append(summary)
                append_jsonl(Path(output) / "comparisons.jsonl", summary)
                events.put({"event": "comparison_finished", "summary": summary, "time": time.monotonic()})
        for item in comparisons:
            single = next(
                (
                    other
                    for other in comparisons
                    if other["case"] == item["case"] and other["batch_size"] == 1
                ),
                None,
            )
            if single:
                item["throughput_relative_to_batch1"] = (
                    item["observed_windows_per_second"] / single["observed_windows_per_second"]
                )
        (Path(output) / "batch_benchmark_report.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "model_loads": 1,
                    "model_config": config_dict,
                    "comparisons": comparisons,
                    "caveat": "Throughput benchmark of ready independent requests. Does not establish a live 1Hz deadline or consumer-GPU performance.",
                },
                indent=2,
            )
            + "\n"
        )
        events.put({"event": "finished", "time": time.monotonic()})
    except BaseException as exc:
        append_jsonl(Path(output) / "errors.jsonl", {"error": str(exc), "traceback": traceback.format_exc()})
        events.put({"event": "error", "error": str(exc)})
        raise
    finally:
        stop.set()
        if power:
            power.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-soft-tokens", type=int, default=140)
    parser.add_argument("--fixture-cases", default="quiet,moving")
    parser.add_argument("--fixture-audio", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--dataset-root")
    parser.add_argument("--source-kind", choices=["real_public", "private_recording", "synthetic_fixture"])
    parser.add_argument("--policy")
    parser.add_argument("--power", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--load-timeout-s", type=float, default=900)
    parser.add_argument("--batch-timeout-s", type=float, default=180)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not sizes or any(size not in (1, 2, 4) for size in sizes) or len(set(sizes)) != len(sizes):
        parser.error("batch-sizes must contain unique sizes from 1,2,4")
    if not 1 <= args.iterations <= 100 or not 0 <= args.warmup_batches <= 10:
        parser.error("iterations must be [1,100], warmup-batches [0,10]")
    if not 1 <= args.load_timeout_s <= 3600 or not 1 <= args.batch_timeout_s <= 600:
        parser.error("load timeout must be [1,3600], batch timeout [1,600]")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "batches.jsonl").exists():
        parser.error("use a fresh output directory")
    total = max(sizes) * (args.iterations + args.warmup_batches)
    policy = (
        json.loads(Path(args.policy).read_text())
        if args.policy
        else {"allowed_entities": [], "allowed_services": []}
    )
    if args.input:
        if not args.dataset_root or not args.source_kind:
            parser.error("input requires dataset-root and explicit source-kind")
        cases = input_case(args.input, args.source_kind, total, policy)
        if len(cases[0]["rows"]) < total:
            parser.error(f"input requires {total} unique windows; decrease iterations or batch sizes")
        root = Path(args.dataset_root).resolve()
    else:
        names = args.fixture_cases.split(",")
        if any(name not in {"quiet", "moving"} for name in names) or len(names) != len(set(names)):
            parser.error("fixture-cases supports quiet,moving once each")
        cases, root = fixture_cases(output, names, total, 1.0, policy, args.fixture_audio)
    config = ModelConfig(
        model_id=args.model,
        revision=args.revision,
        adapter_path=args.adapter,
        quantization=args.quantization,
        dataset_root=str(root),
        max_batch_size=max(sizes),
        max_new_tokens=args.max_new_tokens,
        max_soft_tokens=args.max_soft_tokens,
    )
    options = {
        "batch_sizes": sizes,
        "iterations": args.iterations,
        "warmup_batches": args.warmup_batches,
        "power": args.power,
        "gpu_index": args.gpu_index,
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "model_config": asdict(config),
                "options": vars(args),
                "source_cases": [
                    {"name": c["name"], "source_kind": c["source_kind"], "windows": len(c["rows"])}
                    for c in cases
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if args.prepare_only:
        print(json.dumps({"prepared": True, "gpu_executed": False, "output": str(output)}))
        return
    context = mp.get_context("spawn")
    events = context.Queue()
    child = context.Process(target=worker, args=(asdict(config), cases, options, str(output), events))
    child.start()
    launched, loaded, active_batch = time.monotonic(), None, None
    failure = None
    while child.is_alive():
        try:
            event = events.get(timeout=0.25)
            append_jsonl(output / "lifecycle.jsonl", event)
            if event["event"] == "loaded":
                loaded = time.monotonic()
            elif event["event"] == "batch_started":
                active_batch = time.monotonic()
            elif event["event"] == "batch_finished":
                active_batch = None
            elif event["event"] == "error":
                failure = event["error"]
        except queue.Empty:
            pass
        now = time.monotonic()
        if loaded is None and now - launched > args.load_timeout_s:
            failure = "model_load_timeout"
        elif active_batch is not None and now - active_batch > args.batch_timeout_s:
            failure = "batch_timeout"
        elif (
            loaded is not None
            and now - loaded
            > len(cases) * len(sizes) * (args.iterations + args.warmup_batches) * args.batch_timeout_s + 30
        ):
            failure = "total_run_timeout"
        if failure:
            child.terminate()
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
            break
    child.join(timeout=5)
    if failure or child.exitcode != 0:
        (output / "interrupted.json").write_text(
            json.dumps({"completed": False, "error": failure or f"child_exit_{child.exitcode}"}) + "\n"
        )
        raise SystemExit(f"batch benchmark interrupted: {failure or child.exitcode}")
    print(json.dumps({"completed": True, "report": str(output / "batch_benchmark_report.json")}, indent=2))


if __name__ == "__main__":
    main()

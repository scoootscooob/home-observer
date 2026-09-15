#!/usr/bin/env python3
"""Bounded, inference-only dynamic/static/compiled cache comparison.

Uses the existing NativeModel with a local generate wrapper. Does not change the
production backend. Compilation/cold-start work is recorded separately from warm
calls. All comparisons preserve greedy output, media count, and raw failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import time
import traceback
from pathlib import Path


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def worker(options):
    os.setsid()
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    os.environ.setdefault("MAX_JOBS", "2")
    output = Path(options["output"])
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor-cache")
    import torch
    import transformers
    from transformers import CompileConfig

    from home_observer.cli import load_policy
    from home_observer.model import ModelConfig, NativeModel, prepare_request, verify_decision_evidence
    from home_observer.schema import parse_decision

    config = ModelConfig(dataset_root=options["dataset_root"], quantization="none",
                         max_new_tokens=options["max_new_tokens"])
    rows = [json.loads(line) for line in Path(options["dataset"]).read_text().splitlines()]
    cases = {"quiet": next(r for r in rows if r["task_type"] == "device_control" and not r["target"]["actions"]),
             "active": next(r for r in rows if r["task_type"] == "device_control" and r["target"]["actions"])}
    policy = load_policy(options["policy"]).as_dict()
    started = time.perf_counter()
    backend = NativeModel(config)
    append(output / "events.jsonl", {"event": "loaded", "model_load_s": time.perf_counter() - started,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "quantization": "none", "adapter": None,
        "max_cache_len": options["max_cache_len"], "compile_fullgraph": True,
        "compile_mode": "reduce-overhead", "compile_dynamic": False})
    original = backend.model.generate
    baseline = {}
    for phase in ("dynamic_eager", "static_eager", "static_compile"):
        kwargs = {"disable_compile": phase != "static_compile"}
        if phase != "dynamic_eager":
            kwargs.update(cache_implementation="static", max_cache_len=options["max_cache_len"])
        if phase == "static_compile":
            kwargs["compile_config"] = CompileConfig(fullgraph=True, dynamic=False, mode="reduce-overhead")
        backend.model.generate = lambda *args, _kwargs=kwargs, **kw: original(*args, **kw, **_kwargs)
        # First compiled call includes compilation. The next two are warm calls.
        schedule = [("quiet", "cold"), ("active", "measured")]
        if phase == "static_compile":
            schedule = [("quiet", "compile_startup"), ("quiet", "warm"), ("active", "warm")]
        for case, temperature in schedule:
            row = cases[case]
            request = prepare_request({"window": row["window"], "state": {}, "recent_events": [], "policy": policy}, config)
            record = {"phase": phase, "case": case, "temperature": temperature,
                      "window_id": request["window"]["window_id"],
                      "camera_ids": [f["camera_id"] for f in request["window"]["frames"]],
                      "audio_items": len(request["window"]["audio"])}
            append(output / "events.jsonl", {"event": "generation_start", **record})
            started = time.perf_counter()
            try:
                raw, metrics = backend.generate_text(request)
                record.update(raw_output=raw, metrics=metrics, completed=True,
                              output_sha256=hashlib.sha256(raw.encode()).hexdigest())
                if phase == "dynamic_eager":
                    baseline[case] = raw
                else:
                    record["greedy_output_matches_baseline"] = raw == baseline[case]
                try:
                    decision = parse_decision(raw)
                    verify_decision_evidence(decision, request)
                    record.update(valid_grounded_decision=True, decision=decision.model_dump())
                except ValueError as exc:
                    record.update(valid_grounded_decision=False, validation_error=str(exc))
            except Exception as exc:
                record.update(completed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            record["wall_s"] = time.perf_counter() - started
            append(output / "requests.jsonl", record)
            print(json.dumps({k: record[k] for k in ("phase", "case", "temperature", "completed", "wall_s")}), flush=True)
            if not record["completed"]:
                return  # One attempt only; do not repeat unsupported compiler failures.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/mixed/test.jsonl")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--policy", default="data/fixtures-expanded/policy.json")
    parser.add_argument("--output", default="artifacts/consumer-compile")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-cache-len", type=int, default=4096)
    parser.add_argument("--deadline-seconds", type=float, default=300)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "requests.jsonl").exists():
        raise ValueError("Use a fresh output directory")
    args.output = str(output)
    started = time.monotonic()
    child = mp.get_context("spawn").Process(target=worker, args=(vars(args),))
    child.start()
    child.join(args.deadline_seconds)
    timed_out = child.is_alive()
    if timed_out:
        os.killpg(child.pid, signal.SIGTERM)
        child.join(5)
        if child.is_alive():
            os.killpg(child.pid, signal.SIGKILL)
            child.join(5)
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()] if (output / "requests.jsonl").exists() else []
    summary = {"elapsed_s": time.monotonic() - started, "timed_out": timed_out, "child_exit_code": child.exitcode,
               "scope": "One base model, four authored camera fixtures plus audio; no Home Assistant actions. Exact greedy text parity required before adopting an optimization.",
               "compile_startup_separate": True,
               "records": [{k: r[k] for k in ("phase", "case", "temperature", "completed", "wall_s", "greedy_output_matches_baseline", "error") if k in r} for r in records],
               "sources": ["https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/gemma4",
                           "https://huggingface.co/docs/transformers/v5.17.0/en/kv_cache",
                           "https://github.com/huggingface/transformers/issues/48501"],
               "notes": "Installed5.17 supports Gemma4 fullgraph compile and max_cache_len for cache reuse. Chunked prefill is deliberately disabled; compile errors are retained without retry."}
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 1 if timed_out or child.exitcode or any(not r["completed"] for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())

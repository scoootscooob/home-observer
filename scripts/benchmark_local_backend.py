#!/usr/bin/env python3
"""Measure a local on-host physical backend on labeled rows, retaining every raw generation.

One greedy generation per row. The same raw text is then scored twice: under
the strict complete-decision contract and under the judged-fields contract
that completes unsupervised fields with recorded placeholders. Labels never
enter inference; the output directory is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import ValidationError

from home_observer.model import ModelConfig
from home_observer.physical_evaluate import score_predictions
from home_observer.physical_local import PhysicalLocalModel, complete_judged_decision, local_model_manifest
from home_observer.physical_model import physical_request_from_row, prepare_physical_request
from home_observer.physical_schema import PhysicalDecision, validate_physical_decision


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def score_rows(rows, config, generations):
    """Apply both contracts to saved raw generations; labels are only used for scoring."""
    strict_rows, judged_rows = [], []
    for row in rows:
        clean = prepare_physical_request(physical_request_from_row(row), config)
        item = generations[row["id"]]
        raw, metrics, error = item.get("raw_output") or "", item.get("metrics") or {}, item.get("error")
        strict = {"id": row["id"], "result": {"decision": None, "metrics": metrics, "raw_output": raw}}
        judged = {"id": row["id"], "result": {"decision": None, "metrics": metrics, "raw_output": raw}}
        # A recorded error next to retained raw text is a validation failure to re-judge, not a
        # missing generation; only an empty generation is a backend failure.
        if not raw:
            strict["result"]["error"] = judged["result"]["error"] = error or "no generation"
        else:
            try:
                decision = validate_physical_decision(PhysicalDecision.model_validate_json(raw), clean["window"])
                strict["result"]["decision"] = decision.model_dump()
                judged["result"]["decision"] = decision.model_dump()
                judged["result"]["metrics"] = {**metrics, "completion": {"contract": "judged-fields",
                                                                        "note": "strict output accepted"}}
            except (ValidationError, ValueError) as exc:
                strict["result"]["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
                try:
                    completed, completion = complete_judged_decision(raw, clean["window"])
                    completed = validate_physical_decision(completed, clean["window"])
                    judged["result"]["decision"] = completed.model_dump()
                    judged["result"]["metrics"] = {**metrics, "completion": completion}
                except (ValidationError, ValueError) as inner:
                    judged["result"]["error"] = f"{type(inner).__name__}: {str(inner)[:1500]}"
        strict_rows.append(strict)
        judged_rows.append(judged)
    return strict_rows, judged_rows


def rescore(args, rows):
    source = args.rescore_from
    generations = {item["id"]: item for item in (json.loads(line) for line in
                   (source / "native-outputs.jsonl").read_text().splitlines() if line.strip())}
    missing = [row["id"] for row in rows if row["id"] not in generations]
    if missing:
        raise ValueError(f"native outputs missing rows: {missing[:5]}")
    config = ModelConfig(adapter_path=args.adapter, dataset_root=str(args.dataset_root.resolve()),
                         quantization="none", max_frames=args.max_frames, max_input_tokens=8192,
                         max_new_tokens=args.max_new_tokens, merge_adapter=False)
    args.output.mkdir(parents=True)
    strict_rows, judged_rows = score_rows(rows, config, generations)
    by_id = {row["id"]: row for row in rows}
    scores, summary = {}, {}
    for name, items in (("strict", strict_rows), ("judged", judged_rows)):
        (args.output / f"predictions-{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        scores[name] = score_predictions([{**by_id[item["id"]], **item} for item in items])
        write_json(args.output / f"scores-{name}.json", scores[name])
        summary[name] = {"usable_decisions": scores[name]["usable_decisions"], "windows": scores[name]["windows"],
                         "labeled_event_recall": scores[name]["events"]["labeled_event_recall"],
                         "mean_latency_s": scores[name]["timing"]["model_latency_s"]["mean"]}
    source_manifest = json.loads((source / "manifest.json").read_text())
    write_json(args.output / "manifest.json", {**source_manifest, "rescored_from": str(source.resolve()),
               "rescore_source_sha256": sha(source / "native-outputs.jsonl"),
               "implementation_sha256": sha(Path(__file__)), "complete": True, "summary": summary})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--rescore-from", type=Path, default=None,
                        help="Rebuild predictions/scores from a previous run's native-outputs.jsonl without a model")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark run: {args.output}")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if args.rescore_from is not None:
        rescore(args, rows)
        return
    config = ModelConfig(adapter_path=args.adapter, dataset_root=str(args.dataset_root.resolve()),
                         quantization="none", max_frames=args.max_frames, max_input_tokens=8192,
                         max_new_tokens=args.max_new_tokens, merge_adapter=False)
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    args.output.mkdir(parents=True)
    loaded = time.perf_counter()
    model = PhysicalLocalModel(config, device=args.device, contract="strict")
    load_s = time.perf_counter() - loaded
    manifest = {"version": 1, "complete": False, "task": "physical_perception_local_benchmark",
                "dataset": str(args.dataset.resolve()), "dataset_sha256": sha(args.dataset),
                "rows": len(rows), "selected_ids": [r["id"] for r in rows], "model": local_model_manifest(model),
                "model_config": asdict(config), "versions": versions, "host": {"platform": platform.platform(),
                "machine": platform.machine()}, "load_s": load_s, "labels_in_inference": False,
                "generation": {"greedy": True, "requests_in_flight": 1, "structured_decoding": False}}
    write_json(args.output / "manifest.json", manifest)
    strict_rows, judged_rows, raw_rows = [], [], []
    for index, row in enumerate(rows):
        request = physical_request_from_row(row)
        clean = prepare_physical_request(request, config)
        started = time.perf_counter()
        try:
            raw, metrics = model.generate_text(clean)
            error = None
        except Exception as exc:  # backend failure is recorded, never repaired
            raw, metrics, error = "", {"latency_s": time.perf_counter() - started}, f"{type(exc).__name__}: {exc}"
        raw_rows.append({"id": row["id"], "raw_output": raw, "metrics": metrics, "error": error})
        strict = {"id": row["id"], "result": {"decision": None, "metrics": metrics, "raw_output": raw}}
        judged = {"id": row["id"], "result": {"decision": None, "metrics": metrics, "raw_output": raw}}
        if error:
            strict["result"]["error"] = judged["result"]["error"] = error
        else:
            try:
                decision = validate_physical_decision(PhysicalDecision.model_validate_json(raw), clean["window"])
                strict["result"]["decision"] = decision.model_dump()
                judged["result"]["decision"] = decision.model_dump()
                judged["result"]["metrics"] = {**metrics, "completion": {"contract": "judged-fields",
                                                                        "note": "strict output accepted"}}
            except (ValidationError, ValueError) as exc:
                strict["result"]["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
                try:
                    completed, completion = complete_judged_decision(raw, clean["window"])
                    completed = validate_physical_decision(completed, clean["window"])
                    judged["result"]["decision"] = completed.model_dump()
                    judged["result"]["metrics"] = {**metrics, "completion": completion}
                except (ValidationError, ValueError) as inner:
                    judged["result"]["error"] = f"{type(inner).__name__}: {str(inner)[:1500]}"
        strict_rows.append(strict)
        judged_rows.append(judged)
        for name, items in (("predictions-strict.jsonl", strict_rows), ("predictions-judged.jsonl", judged_rows),
                            ("native-outputs.jsonl", raw_rows)):
            (args.output / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        print(json.dumps({"row": index + 1, "of": len(rows), "id": row["id"], "latency_s": metrics.get("latency_s"),
                          "strict_valid": strict["result"]["decision"] is not None,
                          "judged_valid": judged["result"]["decision"] is not None}), flush=True)
    by_id = {row["id"]: row for row in rows}
    scores = {}
    for name, items in (("strict", strict_rows), ("judged", judged_rows)):
        joined = [{**by_id[item["id"]], **item} for item in items]
        scores[name] = score_predictions(joined)
        write_json(args.output / f"scores-{name}.json", scores[name])
    manifest.update(complete=True, finished_at=time.time(), summary={
        name: {"usable_decisions": scores[name]["usable_decisions"], "windows": scores[name]["windows"],
               "labeled_event_recall": scores[name]["events"]["labeled_event_recall"],
               "mean_latency_s": scores[name]["timing"]["model_latency_s"]["mean"]} for name in scores})
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()

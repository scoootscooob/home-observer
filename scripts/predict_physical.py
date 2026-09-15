#!/usr/bin/env python3
"""Sequential physical inference, retaining every raw generation.

Native mode loads pinned BF16 weights; vLLM uses the explicitly named engine
model. Both use production observe validation and no label fields in requests.
Output is isolated and never overwrites a prior run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import closing, nullcontext
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from home_observer.model import ModelConfig
from home_observer.physical_evaluate import score_predictions
from home_observer.physical_model import (
    PHYSICAL_SYSTEM_PROMPT,
    PhysicalNativeModel,
    physical_request_from_row,
    prepare_physical_request,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class LoggedPhysicalModel(PhysicalNativeModel):
    def generate_text(self, request):
        raw, metrics = super().generate_text(request)
        self.last_generation = {"raw_output": raw, "metrics": metrics}
        return raw, metrics


def run_predictions(dataset: Path, config: ModelConfig, output: Path, *, native_factory=None,
                    backend="native", engine_url="http://127.0.0.1:8001/v1", served_model="physical-observer",
                    schema_constraints=True, limit=None) -> dict:
    if backend not in {"native", "vllm"}:
        raise ValueError("Unknown physical backend")
    if config.quantization != "none" or config.should_merge_adapter:
        raise ValueError("This baseline runner requires BF16 with unmerged adapters")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prediction run: {output}")
    source_lines = [line for line in dataset.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in source_lines]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
        raise ValueError("Nonempty dataset with string row IDs required")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate dataset IDs")
    source_count = len(rows)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Selection limit must be a positive integer")
        rows = rows[:limit]
    root = Path(config.dataset_root).resolve()
    media = {}
    # Preflight only the input selector; labels and provenance are not passed to the backend.
    for row in rows:
        clean = prepare_physical_request(physical_request_from_row(row), config)
        for item in [*[frame for clip in clean["window"]["clips"] for frame in clip["frames"]],
                     *clean["window"]["audio"]]:
            path = Path(item["path"])
            media[str(path.relative_to(root))] = sha(path)
    adapter = {}
    if config.adapter_path:
        path = Path(config.adapter_path)
        for name in ["adapter_model.safetensors", "adapter_config.json"]:
            file = path / name
            if not file.is_file():
                raise ValueError(f"Adapter missing {name}")
            adapter[name] = sha(file)
    source = Path(__file__).resolve().parents[1] / "src/home_observer"
    implementation = {str(Path("src/home_observer") / name): sha(source / name) for name in
                      ["physical_model.py", "physical_schema.py", "physical_evaluate.py", "model.py"]}
    implementation["scripts/predict_physical.py"] = sha(Path(__file__))
    if backend == "vllm":
        for name in ["physical_vllm.py", "physical_serve.py", "vllm_backend.py"]:
            implementation[str(Path("src/home_observer") / name)] = sha(source / name)
    versions = {}
    for name in ["torch", "transformers", "peft", "accelerate"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    manifest = {"version": 1, "complete": False, "task": "physical_perception",
                "dataset": str(dataset.resolve()), "dataset_sha256": sha(dataset),
                "expected_rows": len(rows), "source_rows": source_count, "selection_limit": limit,
                "selected_ids": [r["id"] for r in rows], "backend": backend,
                "splits": sorted({str(r.get("split", "unspecified")) for r in rows}),
                "model_config": asdict(config), "adapter_sha256": adapter,
                "implementation_sha256": implementation, "versions": versions,
                "prompt_sha256": hashlib.sha256(PHYSICAL_SYSTEM_PROMPT.encode()).hexdigest(),
                "media_sha256": media, "labels_in_inference": False,
                "mode": "Independent cold-start snapshots, sequential one-request inference",
                "generation": {"greedy": True, "adapter_merged": False,
                               "dtype": "bfloat16" if backend == "native" else "engine_configured",
                               "requested_dtype": "bfloat16",
                               "adapter_execution": "native_peft_unmerged" if backend == "native" else "engine_served",
                               "structured_decoding": backend == "vllm" and schema_constraints,
                               "requests_in_flight": 1}}
    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    (output / "inputs.jsonl").write_text("\n".join(source_lines[:len(rows)]) + "\n")
    started = time.perf_counter()
    if native_factory is None:
        if backend == "native":
            native_factory = LoggedPhysicalModel
        else:
            from home_observer.physical_vllm import PhysicalVLLMModel

            def native_factory(conf):
                return PhysicalVLLMModel(conf, base_url=engine_url, served_model=served_model,
                                         schema_constraints=schema_constraints)
    native = native_factory(config)
    if hasattr(native, "backend_config"):
        manifest["backend_config"] = native.backend_config
        atomic_json(output / "manifest.json", manifest)
    loaded = time.perf_counter()
    results = []
    raw_name = "native-outputs.jsonl" if backend == "native" else "engine-outputs.jsonl"
    client_guard = closing(native) if hasattr(native, "close") else nullcontext()
    with client_guard, (output / "predictions.jsonl").open("w") as predictions, (
            output / raw_name).open("w") as raw_file:
        for row in rows:
            request = physical_request_from_row(row)
            native.last_generation = None
            begin = time.perf_counter()
            result = {}
            try:
                observed = native.observe(request)
                result = {"decision": observed["decision"], "metrics": observed["metrics"]}
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            generated = native.last_generation
            result["total_latency_s"] = time.perf_counter() - begin
            if generated:
                result["metrics"] = generated["metrics"]
            raw = generated["raw_output"] if generated else None
            item = {**row, "raw_output": raw, "result": result}
            results.append(item)
            predictions.write(json.dumps(item, ensure_ascii=False) + "\n")
            predictions.flush()
            os.fsync(predictions.fileno())
            raw_file.write(json.dumps({"id": row["id"], "window_id": request["window"]["window_id"],
                                       "raw_output": raw, "metrics": result.get("metrics"),
                                       "error": result.get("error")}, ensure_ascii=False) + "\n")
            raw_file.flush()
            os.fsync(raw_file.fileno())
            print(json.dumps({"completed": len(results), "total": len(rows), "id": row["id"],
                              "strict_valid": "error" not in result,
                              "latency_s": result["total_latency_s"]}), flush=True)
    report = score_predictions(results)
    report.update(complete=True, dataset_complete=len(rows) == source_count, source_rows=source_count,
                  backend=backend, expected_rows=len(rows),
                  strictly_valid=sum("error" not in r["result"] for r in results),
                  inference_executed=True, model_load_s=loaded - started, run_wall_s=time.perf_counter() - started,
                  inference_wall_s=time.perf_counter() - loaded)
    report["predictions_sha256"] = sha(output / "predictions.jsonl")
    atomic_json(output / "report.json", report)
    manifest.update(complete=True, dataset_complete=len(rows) == source_count, completed_rows=len(results),
                    predictions_sha256=report["predictions_sha256"],
                    raw_outputs_file=raw_name, raw_outputs_sha256=sha(output / raw_name))
    atomic_json(output / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="First N source rows; manifest marks partial dataset coverage")
    parser.add_argument("--backend", choices=["native", "vllm"], default="native")
    parser.add_argument("--engine-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--served-model", default="physical-observer")
    parser.add_argument("--no-schema-constraints", action="store_true")
    args = parser.parse_args()
    config = ModelConfig(dataset_root=str(args.dataset_root.resolve()),
                         adapter_path=str(args.adapter.resolve()) if args.adapter else None,
                         quantization="none", merge_adapter=False, max_batch_size=1,
                         max_frames=args.max_frames, max_input_tokens=args.max_input_tokens,
                         max_new_tokens=args.max_new_tokens)
    report = run_predictions(args.dataset, config, args.output.resolve(), backend=args.backend,
                             engine_url=args.engine_url, served_model=args.served_model,
                             schema_constraints=not args.no_schema_constraints, limit=args.limit)
    print(json.dumps({"complete": report["complete"], "windows": report["windows"],
                      "strictly_valid": report["strictly_valid"],
                      "labeled_event_recall": report["events"]["labeled_event_recall"]}))


if __name__ == "__main__":
    main()

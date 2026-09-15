#!/usr/bin/env python3
"""Export merged BF16 HF weights and compare fixed validation with the saved adapter run."""
import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from model_batch_parity import inspect_output

from home_observer.cli import load_policy
from home_observer.evaluate import summarize
from home_observer.model import ModelConfig, NativeModel
from home_observer.prompts import SYSTEM_PROMPT
from home_observer.train import file_sha256, request_from_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation", default="data/mixed-v2/generation-validation.jsonl")
    parser.add_argument("--reference", required=True, help="Saved unmerged generation-validation.jsonl")
    parser.add_argument("--policy", default="data/mixed-v2/policy.json")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--batch-smoke", action="store_true", help="Compare one native batch8 with merged singles before selecting batch8")
    args = parser.parse_args()
    output, adapter = Path(args.output), Path(args.adapter)
    if output.exists():
        raise ValueError("use a fresh export directory")
    rows = [json.loads(line) for line in Path(args.validation).read_text().splitlines() if line.strip()]
    reference = {row["id"]: row for row in
                 (json.loads(line) for line in Path(args.reference).read_text().splitlines() if line.strip())}
    if not 1 <= len(rows) <= 16 or set(reference) != {row["id"] for row in rows}:
        raise ValueError("reference must cover the exact bounded fixed validation rows")
    policy = load_policy(Path(args.policy))
    config = ModelConfig(adapter_path=str(adapter), dataset_root=args.dataset_root,
                         quantization="none", merge_adapter=True, max_new_tokens=768,
                         max_batch_size=8 if args.batch_smoke else 4)
    model = NativeModel(config)
    if not model.adapter_merged:
        raise RuntimeError("export requires a merged unquantized model")
    output.mkdir(parents=True)
    results, comparisons = [], []
    with (output / "merged-validation.jsonl").open("w") as stream:
        for row in rows:
            request = request_from_row(row, policy.as_dict())
            raw, metrics = model.generate_text(request)
            parsed = inspect_output(raw, request, policy)
            result = {"metrics": metrics, "total_latency_s": metrics["latency_s"],
                      "rejections": parsed["rejections"]}
            if parsed["evidence_valid"]:
                result["decision"] = parsed["decision"]
            else:
                result["error"] = parsed.get("error", "invalid output")
            item = {"id": row["id"], "task_type": row.get("task_type"), "target": row["target"],
                    "supervision_mask": row.get("supervision_mask", {}), "raw_output": raw, "result": result}
            results.append(item)
            old = reference[row["id"]]
            old_parsed = inspect_output(old["raw_output"], request, policy)
            comparisons.append({"id": row["id"], "raw_exact_match": raw == old["raw_output"],
                                "actions_match": parsed["actions"] == old_parsed["actions"],
                                "observations_match": parsed["observations"] == old_parsed["observations"],
                                "validity_match": all(parsed[k] == old_parsed[k] for k in
                                                      ("schema_valid", "evidence_valid", "rejections"))})
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(comparisons[-1]), flush=True)
    report = {"model_config": asdict(config), "quality": summarize(results), "comparisons": comparisons,
              "material_differences": [item["id"] for item in comparisons
                                       if not all(item[k] for k in ("actions_match", "observations_match", "validity_match"))],
              "reference_sha256": file_sha256(args.reference),
              "validation_sha256": file_sha256(args.validation),
              "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "parent_adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
              "export_kind": "Merged BF16 full Hugging Face model, no quantization; not a running server."}
    if args.batch_smoke:
        smoke = {"acceptable": False, "batch_size": 8, "rows": []}
        if len(rows) != 8:
            raise ValueError("batch8 smoke requires exactly8 validation rows")
        requests = [request_from_row(row, policy.as_dict()) for row in rows]
        try:
            outputs = model.generate_batch(requests)
            for row, request, (raw, metrics), single in zip(rows, requests, outputs, results, strict=True):
                batched = inspect_output(raw, request, policy)
                baseline = inspect_output(single["raw_output"], request, policy)
                same = all(batched[k] == baseline[k] for k in
                           ("actions", "observations", "schema_valid", "evidence_valid", "rejections"))
                smoke["rows"].append({"id": row["id"], "matches_merged_single": same,
                                      "raw_output": raw, "metrics": metrics, "parsed": batched})
            smoke["acceptable"] = all(item["matches_merged_single"] for item in smoke["rows"])
            smoke["peak_allocated_gb"] = outputs[0][1]["peak_allocated_gb"]
            smoke["full_batch_event_latency_s"] = outputs[0][1]["latency_s"]
        except RuntimeError as exc:
            smoke["error"] = str(exc)
            import torch
            torch.cuda.empty_cache()
        report["batch8_smoke"] = smoke
        print(json.dumps({"batch8_smoke": {k: v for k, v in smoke.items() if k != "rows"}}), flush=True)
    (output / "merge-validation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    model.model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    model.processor.save_pretrained(output)
    files = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
             for path in output.iterdir() if path.is_file()}
    (output / "export-manifest.json").write_text(json.dumps({"complete": True, "report": report, "files": files}, indent=2) + "\n")
    print(json.dumps({"export_complete": True, "output": str(output), "quality": report["quality"],
                      "material_differences": report["material_differences"]}), flush=True)


if __name__ == "__main__":
    main()

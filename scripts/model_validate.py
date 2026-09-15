#!/usr/bin/env python3
"""Evaluate unchanged labeled validation rows with one loaded native model."""
import argparse
import json
from pathlib import Path

from model_batch_parity import inspect_output
from model_merge_benchmark import append

from home_observer.cli import load_policy
from home_observer.evaluate import summarize
from home_observer.model import ModelConfig, NativeModel, verify_decision_evidence
from home_observer.schema import InferenceRequest, parse_decision
from home_observer.train import load_rows, request_from_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter")
    parser.add_argument("--input", default="data/mixed-v2/generation-validation.jsonl")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--policy", default="data/mixed-v2/policy.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--reference", help="Saved single-request outputs for bounded batch comparison")
    parser.add_argument("--policy-serialization", choices=["runtime", "file"], default="runtime",
                        help="Diagnostic control: use policy file insertion order from training rather than Policy.as_dict order")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predictions.jsonl").exists():
        parser.error("use a fresh output directory")
    rows = load_rows(args.input, "validation")
    policy = load_policy(args.policy)
    input_policy = (json.loads(Path(args.policy).read_text()) if args.policy_serialization == "file"
                    else policy.as_dict())
    if input_policy != policy.as_dict():
        raise ValueError("diagnostic serialization must preserve the same policy values")
    model = NativeModel(ModelConfig(adapter_path=args.adapter, dataset_root=args.dataset_root,
                        quantization="none", merge_adapter=False, max_new_tokens=args.max_new_tokens,
                        max_batch_size=max(4, args.batch_size)))
    reference = ({r["id"]: r for r in (json.loads(line) for line in Path(args.reference).read_text().splitlines() if line.strip())}
                 if args.reference else None)
    if reference is not None and set(reference) != {r["id"] for r in rows}:
        raise ValueError("reference must contain the same validation rows")
    requests = [request_from_row(row, input_policy) for row in rows]
    def generate():
        for start in range(0, len(rows), args.batch_size):
            group = requests[start:start + args.batch_size]
            yield from model.generate_batch(group) if args.batch_size > 1 else [model.generate_text(group[0])]
    results = []
    comparisons = []
    for row, request, (raw, metrics) in zip(rows, requests, generate(), strict=True):
        result = {"metrics": metrics, "total_latency_s": metrics["latency_s"], "rejections": []}
        try:
            decision = parse_decision(raw)
            verify_decision_evidence(decision, request)
            result.update(decision=decision.model_dump(), rejections=policy.validate(decision, InferenceRequest.model_validate(request).window))
        except ValueError as exc:
            result["error"] = str(exc)
        item = {"id": row["id"], "task_type": row.get("task_type", "unspecified"), "target": row["target"],
                "supervision_mask": row.get("supervision_mask", {}), "raw_output": raw, "result": result}
        results.append(item)
        if reference is not None:
            current = inspect_output(raw, request, policy)
            old = inspect_output(reference[row["id"]]["raw_output"], request, policy)
            comparisons.append({"id": row["id"], "raw_exact_match": raw == reference[row["id"]]["raw_output"],
                                "semantic_validity_match": all(current[key] == old[key] for key in
                                    ("actions", "observations", "schema_valid", "evidence_valid", "rejections"))})
        append(out / "predictions.jsonl", item)
        print(json.dumps({"id": row["id"], "raw_output": raw, "metrics": metrics}), flush=True)
    report = summarize(results)
    report.update(batch_size=args.batch_size, merge_adapter=False, comparisons=comparisons,
                  policy_serialization=args.policy_serialization,
                  batch_semantics_match_reference=all(c["semantic_validity_match"] for c in comparisons) if comparisons else None,
                  peak_allocated_gb=max(item["result"]["metrics"]["peak_allocated_gb"] for item in results))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

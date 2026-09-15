#!/usr/bin/env python3
"""Compare one exact release batch with single requests on the same loaded model."""
import argparse
import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from evaluate_release import capture_snapshot_request

from home_observer.cli import load_policy
from home_observer.evaluate import action_set, observation_set
from home_observer.model import ModelConfig, NativeModel, verify_decision_evidence
from home_observer.prompts import SYSTEM_PROMPT
from home_observer.schema import InferenceRequest, parse_decision


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inspect_output(raw, request, policy):
    result = {"raw_output": raw, "schema_valid": False, "evidence_valid": False,
              "actions": None, "observations": None, "rejections": []}
    try:
        decision = parse_decision(raw)
        result.update(schema_valid=True, decision=decision.model_dump(),
                      actions=sorted(action_set(decision.model_dump()["actions"])),
                      observations=sorted(observation_set(decision.model_dump()["observations"])))
        verify_decision_evidence(decision, request)
        result["evidence_valid"] = True
        result["rejections"] = policy.validate(decision, InferenceRequest.model_validate(request).window)
    except ValueError as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/mixed/test.jsonl")
    parser.add_argument("--start", type=int, default=12)
    parser.add_argument("--policy", default="data/mixed-v2/policy.json")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-requests-sha256", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("use a fresh parity output directory")
    policy = load_policy(Path(args.policy))
    root = Path(args.dataset_root).resolve()
    source = Path(args.source)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    selected = rows[args.start:args.start + 4]
    if args.start < 0 or len(selected) != 4:
        raise ValueError("parity requires exactly four source rows")
    requests = []
    for row in selected:
        window = copy.deepcopy(row["window"])
        for media in [*window["frames"], *window["audio"]]:
            path = Path(media["path"])
            media["path"] = str((path if path.is_absolute() else root / path).resolve())
        requests.append(capture_snapshot_request(window, policy))
    digest = fingerprint(requests)
    if digest != args.expected_requests_sha256:
        raise ValueError(f"requests differ from the exact scored batch: {digest}")
    config = ModelConfig(dataset_root=str(root), adapter_path=args.adapter, quantization="none",
                         max_new_tokens=args.max_new_tokens)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({
        "model_config": asdict(config), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "requests_sha256": digest, "source_start": args.start,
        "source_ids": [row["id"] for row in selected], "requests": requests,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "selection": "Exact scored batch containing false actions, chosen for implementation diagnosis; no training or target changes.",
    }, indent=2) + "\n")
    model = NativeModel(config)
    comparisons = []
    with (output / "results.jsonl").open("w") as stream:
        def record(item):
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        batch = model.generate_batch(requests)
        batch_results = []
        for row, request, (raw, metrics) in zip(selected, requests, batch, strict=True):
            item = inspect_output(raw, request, policy)
            item.update(mode="batch4", id=row["id"], metrics=metrics)
            batch_results.append(item)
            record(item)
        for row, request, batched in zip(selected, requests, batch_results, strict=True):
            raw, metrics = model.generate_text(request)
            single = inspect_output(raw, request, policy)
            single.update(mode="single", id=row["id"], metrics=metrics)
            record(single)
            comparison = {"id": row["id"], "raw_exact_match": batched["raw_output"] == raw,
                          "action_sets_match": batched["actions"] == single["actions"],
                          "observation_sets_match": batched["observations"] == single["observations"],
                          "validity_match": all(batched[k] == single[k] for k in
                                                ("schema_valid", "evidence_valid", "rejections")),
                          "expected_actions": sorted(action_set(row["target"].get("actions", []))),
                          "batch": batched, "single": single}
            comparisons.append(comparison)
            print(json.dumps({k: v for k, v in comparison.items() if k not in {"batch", "single"}}), flush=True)
    report = {"requests_sha256": digest, "comparisons": comparisons,
              "raw_exact_matches": sum(x["raw_exact_match"] for x in comparisons),
              "action_set_matches": sum(x["action_sets_match"] for x in comparisons),
              "observation_set_matches": sum(x["observation_sets_match"] for x in comparisons),
              "validity_matches": sum(x["validity_match"] for x in comparisons),
              "scope": "Four exact requests, one loaded model, greedy BF16 generation. Divergence requires diagnosis; finite parity is not universal equivalence."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    main()

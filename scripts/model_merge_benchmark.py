#!/usr/bin/env python3
"""Measure BF16 LoRA merge parity and a small unchanged full-policy snapshot suite."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from home_observer.cli import load_policy
from home_observer.engine import Observer
from home_observer.evaluate import summarize
from home_observer.ha import SimulatedHomeAssistant
from home_observer.journal import Journal
from home_observer.model import (
    ModelConfig,
    NativeModel,
    build_messages,
    encode_messages,
    verify_decision_evidence,
)
from home_observer.schema import parse_decision


def snapshot_request(window, policy):
    """Capture the exact engine request after current device telemetry is ingested."""
    class Capture:
        request = None
        def observe(self, request):
            self.request = copy.deepcopy(request)
            raise RuntimeError("request capture only")
    capture, journal = Capture(), Journal(":memory:")
    try:
        Observer(capture, journal, policy, SimulatedHomeAssistant()).process(window)
        if capture.request is None:
            raise ValueError("engine did not produce an inference request")
        return capture.request
    finally:
        journal.close()


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", default="data/mixed/test.jsonl")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--policy", default="data/fixtures-expanded/policy.json")
    parser.add_argument("--row-indices", default="0,1,72,73")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "generations.jsonl").exists():
        parser.error("use a fresh output directory")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    selected = [rows[int(i)] for i in args.row_indices.split(",")]
    if len(selected) < 2:
        parser.error("at least two rows are required for merge parity")
    policy = load_policy(args.policy)
    config = ModelConfig(adapter_path=args.adapter, dataset_root=args.dataset_root,
                         quantization="none", merge_adapter=False, max_new_tokens=args.max_new_tokens)
    requests = [snapshot_request(row["window"], policy) for row in selected]
    for request in requests:
        build_messages(request, config)
    (out / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    (out / "manifest.json").write_text(json.dumps({"config": asdict(config), "indices": args.row_indices,
                                                 "context": "independent_full_policy_snapshots",
                                                 "targets_changed": False}, indent=2) + "\n")
    model = NativeModel(config)

    def logits(request):
        batch = encode_messages(model.processor, build_messages(request, config), config, generation=True)
        batch = batch.to(device=model.model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            return model.model(**batch, use_cache=True, logits_to_keep=1).logits.float().cpu()

    def generation(index, phase):
        raw, metrics = model.generate_text(requests[index])
        entry = {"phase": phase, "id": selected[index]["id"], "raw_output": raw, "metrics": metrics}
        append(out / "generations.jsonl", entry)
        print(json.dumps({"phase": phase, "id": entry["id"], "metrics": metrics}), flush=True)
        return entry

    # Warmup is measured and retained but excluded from before/after comparisons.
    generation(0, "unmerged_warmup")
    before_logits = [logits(request) for request in requests[:2]]
    before = [generation(i, "unmerged") for i in range(2)]
    started = time.perf_counter()
    model.merge_adapter_weights()
    torch.cuda.synchronize()
    merge_seconds = time.perf_counter() - started
    after_logits = [logits(request) for request in requests[:2]]
    generation(0, "merged_warmup")
    after = [generation(i, "merged") for i in range(len(selected))]
    comparisons = []
    for i in range(2):
        a, b = before_logits[i], after_logits[i]
        comparisons.append({"id": selected[i]["id"], "raw_greedy_exact_match": before[i]["raw_output"] == after[i]["raw_output"],
                            "first_token_logits_max_abs_diff": (a-b).abs().max().item(),
                            "first_token_logits_mean_abs_diff": (a-b).abs().mean().item(),
                            "first_token_argmax_equal": bool(torch.equal(a.argmax(-1), b.argmax(-1))),
                            "unmerged_latency_s": before[i]["metrics"]["latency_s"],
                            "merged_latency_s": after[i]["metrics"]["latency_s"],
                            "unmerged_output_tokens": before[i]["metrics"]["output_tokens"],
                            "merged_output_tokens": after[i]["metrics"]["output_tokens"]})
    results = []
    for row, request, output in zip(selected, requests, after):
        class Cached:
            def observe(self, actual_request):
                if actual_request != request:
                    raise ValueError("engine request differs from measured snapshot")
                decision = parse_decision(output["raw_output"])
                verify_decision_evidence(decision, actual_request)
                return {"decision": decision.model_dump(), "metrics": output["metrics"]}
        journal = Journal(":memory:")
        try:
            result = Observer(Cached(), journal, policy, SimulatedHomeAssistant()).process(row["window"])
        finally:
            journal.close()
        result["total_latency_s"] += output["metrics"]["latency_s"]
        result["latency_includes_precomputed_native_generation"] = True
        item = {"id": row["id"], "task_type": row.get("task_type", "unspecified"), "target": row.get("target"),
                "supervision_mask": row.get("supervision_mask", {}), "result": result}
        results.append(item)
        append(out / "predictions.jsonl", item)
    report = {"merge_seconds": merge_seconds, "comparisons": comparisons, "snapshot_quality": summarize(results),
              "greedy_matches_all_samples": all(x["raw_greedy_exact_match"] for x in comparisons),
              "parity_scope": "Two BF16 samples only. Merge rounds weights; exact parity is not guaranteed generally.",
              "source_rows": len(selected), "all_rows_preserved": True}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

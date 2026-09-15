#!/usr/bin/env python3
"""Run real GPU baseline -> LoRA update -> save -> reload -> Decision/API checks.

All inputs remain labeled as synthetic/public/human in reports. Two steps test the
mechanics; they do not establish perception accuracy or an always-on service SLA.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from home_observer.model import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    ModelConfig,
    NativeModel,
    build_messages,
    encode_messages,
    verify_decision_evidence,
)
from home_observer.schema import parse_decision
from home_observer.train import load_rows, request_from_row, train_adapter


def evaluate(backend, rows, policy):
    results = []
    for row in rows:
        request = request_from_row(row, policy)
        text, metrics = backend.generate_text(request)
        result = {"id": row["id"], "source": row.get("source"),
                  "annotation_method": row.get("annotation_method"),
                  "raw_output": text, "metrics": metrics, "valid_decision": False}
        try:
            decision = parse_decision(text)
            verify_decision_evidence(decision, request)
            result.update(valid_decision=True, decision=decision.model_dump())
        except ValueError as exc:
            result["error"] = str(exc)
        results.append(result)
    return results


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="none")
    parser.add_argument("--inference-quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--eval-examples", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    policy = json.loads(Path(args.policy).read_text()) if args.policy else {}
    rows = load_rows(args.validation, "validation")[:args.eval_examples]
    if not rows or not any(r["window"].get("frames") for r in rows):
        raise ValueError("GPU smoke requires visual validation examples")
    config = ModelConfig(model_id=args.model, revision=args.revision, dataset_root=args.dataset_root,
                         quantization=args.quantization)
    baseline = NativeModel(config)
    request = request_from_row(rows[0], policy)
    encoded = encode_messages(baseline.processor, build_messages(request, config), config, generation=True)
    encoded = encoded.to(device=baseline.model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        a = baseline.model(**encoded, use_cache=True, logits_to_keep=1).logits.float()
        b = baseline.model(**encoded, use_cache=False, logits_to_keep=1).logits.float()
    parity = {"max_abs_diff": (a - b).abs().max().item(),
              "same_argmax": bool(torch.equal(a.argmax(-1), b.argmax(-1)))}
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or parity["max_abs_diff"] > 0.2 or not parity["same_argmax"]:
        raise RuntimeError(f"Gemma4 cache/no-cache parity gate failed: {parity}")
    before = evaluate(baseline, rows, policy)
    (out / "baseline.json").write_text(json.dumps(before, indent=2) + "\n")
    del a, b, encoded, baseline
    gc.collect()
    torch.cuda.empty_cache()
    training = train_adapter(train_path=args.dataset, validation_path=args.validation,
                             output=str(out / "adapter"), config=config, policy=policy,
                             max_steps=args.steps, gradient_accumulation_steps=1)
    gc.collect()
    torch.cuda.empty_cache()
    config.adapter_path = str(out / "adapter")
    config.quantization = args.inference_quantization
    tuned = NativeModel(config)
    after = evaluate(tuned, rows, policy)
    (out / "after.json").write_text(json.dumps(after, indent=2) + "\n")
    from fastapi.testclient import TestClient

    from home_observer.serve import create_app
    with TestClient(create_app(config, backend=tuned, api_token="smoke-test-token")) as client:
        assert client.get("/health").status_code == 401
        headers = {"Authorization": "Bearer smoke-test-token"}
        assert client.get("/health", headers=headers).json()["status"] == "ready"
        response = client.post("/observe", headers=headers, json=request)
        api = {"status_code": response.status_code, "response": response.json()}
    from safetensors.torch import load_file
    tensors = load_file(str(out / "adapter" / "adapter_model.safetensors"))
    changed = sum(bool(torch.count_nonzero(tensor)) for name, tensor in tensors.items() if "lora_B" in name)
    if not changed:
        raise RuntimeError("saved LoRA B weights remain zero; no learned update was persisted")
    report = {
        "pipeline_completed": True,
        "acceptance_pass": all(r["valid_decision"] for r in after) and api["status_code"] == 200,
        "cache_parity": parity, "nonzero_lora_B_tensors": changed,
        "training": training, "baseline": before, "after": after, "api": api,
        "caveat": "Smoke execution and JSON validity are not perception/action accuracy. Synthetic fixtures test plumbing only.",
    }
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pipeline_completed": True, "acceptance_pass": report["acceptance_pass"],
                      "report": str(out / "smoke_report.json")}, indent=2))


if __name__ == "__main__":
    main()

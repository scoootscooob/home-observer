#!/usr/bin/env python3
"""Check trained cache parity and action-branch conditioning without changing weights."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from model_merge_benchmark import append

from home_observer.model import (
    ModelConfig,
    NativeModel,
    build_messages,
    encode_messages,
    validate_audio_files,
)
from home_observer.train import assistant_labels, request_from_row

NOOP = '{"actions":[]'
ACTION_PREFIX = '{"actions":[{'


def action_branch(processor):
    """Find the shared token prefix of the two explicit JSON branch probes."""
    empty = processor.tokenizer(NOOP, add_special_tokens=False)["input_ids"]
    active = processor.tokenizer(ACTION_PREFIX, add_special_tokens=False)["input_ids"]
    common = 0
    while common < min(len(empty), len(active)) and empty[common] == active[common]:
        common += 1
    if common == min(len(empty), len(active)):
        raise ValueError("action/noop probes do not have an aligned divergent token")
    return {"common_tokens": empty[:common], "empty_token_id": empty[common],
            "action_token_id": active[common], "noop_tokens": empty, "action_tokens": active}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter")
    parser.add_argument("--input", default="data/mixed-v2/generation-validation.jsonl")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--policy", default="data/mixed-v2/policy.json")
    parser.add_argument("--row-indices", default="0,2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("diagnostic generation is bounded to 1..64 tokens")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.jsonl").exists():
        parser.error("use a fresh output directory")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    rows = [rows[int(index)] for index in args.row_indices.split(",")]
    policy = json.loads(Path(args.policy).read_text())
    config = ModelConfig(adapter_path=args.adapter, dataset_root=args.dataset_root,
                         quantization="none", merge_adapter=False, max_new_tokens=args.max_new_tokens)
    requests = [request_from_row(row, policy) for row in rows]
    for request in requests:
        build_messages(request, config)
        validate_audio_files(request, config)
    native = NativeModel(config)
    probe = action_branch(native.processor)
    (out / "manifest.json").write_text(json.dumps({"config": asdict(config), "indices": args.row_indices,
        "action_branch": probe,
        "empty_token_text": native.processor.decode([probe["empty_token_id"]]),
        "action_token_text": native.processor.decode([probe["action_token_id"]]),
        "scope": "Diagnostic partial generations and two canonical JSON branch prefixes; not a release accuracy evaluation."}, indent=2) + "\n")
    results = []
    for row, request in zip(rows, requests):
        messages = build_messages(request, config)
        prefix = encode_messages(native.processor, messages, config, generation=True)
        prompt_tokens = prefix["input_ids"][0].tolist()
        candidates = {}
        for name, text, ids in [("empty", NOOP, probe["noop_tokens"]), ("action", ACTION_PREFIX, probe["action_tokens"])]:
            full_messages = messages + [{"role": "assistant", "content": [{"type": "text", "text": text}]}]
            full = encode_messages(native.processor, full_messages, config, generation=False)
            assistant_labels(full["input_ids"][0].tolist(), prompt_tokens)
            if full["input_ids"][0, len(prompt_tokens):len(prompt_tokens)+len(ids)].tolist() != ids:
                raise ValueError("candidate target boundary tokenization mismatch")
            candidates[name] = (full, ids)
        branch_vectors, first_vectors, modes = {}, {}, {}
        for cache in (True, False):
            name = "cache_on" if cache else "cache_off"
            try:
                fresh = encode_messages(native.processor, messages, config, generation=True)
                fresh = fresh.to(device=native.model.device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    first = native.model(**fresh, use_cache=cache, logits_to_keep=1).logits[0, -1].float().cpu()
                    candidate_scores = {}
                    for candidate, (full, ids) in candidates.items():
                        target = full.to(device=native.model.device, dtype=torch.bfloat16)
                        common = len(probe["common_tokens"])
                        positions = torch.arange(len(prompt_tokens)+common-1, len(prompt_tokens)+len(ids)-1, device=native.model.device)
                        vectors = native.model(**target, use_cache=cache, logits_to_keep=positions).logits[0].float()
                        if not torch.isfinite(vectors).all():
                            raise ValueError("non-finite candidate logits")
                        token_ids = torch.tensor(ids[common:], device=vectors.device)
                        score = torch.log_softmax(vectors, dim=-1).gather(-1, token_ids[:, None]).sum().item()
                        candidate_scores[candidate] = {"conditional_prefix_log_probability": score,
                                                       "scored_suffix_tokens": len(token_ids)}
                        if candidate == "empty":
                            branch = vectors[0].cpu()
                if not torch.isfinite(first).all() or not torch.isfinite(branch).all():
                    raise ValueError("non-finite diagnostic logits")
                first_vectors[name], branch_vectors[name] = first, branch
                margin = candidate_scores["action"]["conditional_prefix_log_probability"] - candidate_scores["empty"]["conditional_prefix_log_probability"]
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = native.model.generate(**fresh, max_new_tokens=args.max_new_tokens,
                                                       do_sample=False, use_cache=cache)
                torch.cuda.synchronize()
                ids = generated[0, len(prompt_tokens):].tolist()
                modes[name] = {"generated_token_ids": ids,
                    "raw_partial_output": native.processor.decode(ids, skip_special_tokens=True),
                    "generation_latency_s": time.perf_counter() - started,
                    "max_new_tokens": args.max_new_tokens, "hit_token_limit": len(ids) >= args.max_new_tokens,
                    "candidate_prefix_scores": candidate_scores,
                    "action_minus_empty_prefix_log_probability": margin,
                    "action_probability_among_two_canonical_prefixes": torch.sigmoid(torch.tensor(margin)).item(),
                    "prefix_score_interpretation": "Probability of two specific tokenized JSON prefixes, excluding their shared prefix; not a complete semantic action probability across alternative tokenizations."}
            except Exception as exc:
                modes[name] = {"error": type(exc).__name__ + ": " + str(exc)}
            append(out / "results.jsonl", {"id": row["id"], "mode": name, **modes[name]})
            print(json.dumps({"id": row["id"], "mode": name, **modes[name]}), flush=True)
        comparison = {"id": row["id"], "expected_action": bool(row["target"].get("actions")), "modes": modes}
        if all("generated_token_ids" in modes[name] for name in ("cache_on", "cache_off")):
            a, b = modes["cache_on"]["generated_token_ids"], modes["cache_off"]["generated_token_ids"]
            comparison["generated_tokens_exact_match"] = a == b
            comparison["first_divergence_index"] = next((i for i, (x, y) in enumerate(zip(a,b)) if x != y),
                                                        min(len(a),len(b)) if len(a) != len(b) else None)
        if len(first_vectors) == 2:
            comparison["first_logits_max_abs_diff"] = (first_vectors["cache_on"]-first_vectors["cache_off"]).abs().max().item()
            comparison["first_logits_argmax_equal"] = bool(first_vectors["cache_on"].argmax() == first_vectors["cache_off"].argmax())
        if len(branch_vectors) == 2:
            comparison["branch_logits_max_abs_diff"] = (branch_vectors["cache_on"]-branch_vectors["cache_off"]).abs().max().item()
        results.append(comparison)
        (out / "report.json").write_text(json.dumps({"samples": results, "adapter_merged": False}, indent=2) + "\n")
    print(json.dumps({"samples": results, "adapter_merged": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()

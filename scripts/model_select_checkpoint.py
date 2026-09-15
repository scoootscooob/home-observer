#!/usr/bin/env python3
"""Select continuation checkpoint50/100 from fixed validation, never heldout tests."""
import argparse
import json
from pathlib import Path

from home_observer.train import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/training-v2-continuation/adapter/checkpoints")
    parser.add_argument("--output", default="artifacts/training-v2-continuation/selection.json")
    args = parser.parse_args()
    candidates = []
    for step in (50, 100):
        checkpoint = Path(args.root) / f"checkpoint-{step}"
        path = checkpoint / "generation-report.json"
        report = json.loads(path.read_text())
        rows = [json.loads(line) for line in (checkpoint / "generation-validation.jsonl").read_text().splitlines() if line.strip()]
        q = report["quality"]
        if report["step"] != step or q["windows"] != 8 or len(rows) != 8:
            raise ValueError("each candidate requires complete fixed8 generation validation")
        counts = q["counts"]
        precision, recall = q["action_precision"] or 0., q["action_recall"] or 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        score = [-counts["action_fp"], f1, q["valid_response_rate"], q["observation_recall"] or 0., -step]
        candidates.append({"step": step, "checkpoint": str(checkpoint), "quality": q, "score": score,
                           "report_sha256": file_sha256(path), "validation_sha256": report["validation_sha256"]})
    if len({candidate["validation_sha256"] for candidate in candidates}) != 1:
        raise ValueError("candidates were evaluated on different validation data")
    selected = max(candidates, key=lambda item: item["score"])
    result = {"selected_checkpoint": selected["checkpoint"], "selected_step": selected["step"],
              "candidates": candidates, "selection_data": "Fixed8 validation only; no heldout test data read.",
              "criteria_in_order": ["fewest false proposed actions", "highest action F1", "highest valid response rate",
                                    "highest supervised observation recall", "earlier checkpoint on exact tie"],
              "status": "Candidate for heldout evaluation; not deployment acceptance."}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

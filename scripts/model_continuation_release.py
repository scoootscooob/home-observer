#!/usr/bin/env python3
"""Bounded sequence: fixed-validation selection, unmerged batch smoke, full heldout."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path("/workspace/home-observer")
    os.chdir(root)
    os.environ["HF_HOME"] = "/workspace/huggingface"
    deadline = time.monotonic() + 1800
    marker = root / "artifacts/training-v2-continuation-exit"
    while not marker.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("continuation did not finish within coordination deadline")
        time.sleep(5)
    if marker.read_text().strip() != "0":
        raise RuntimeError("continuation failed; inspect retained logs before further evaluation")

    def run(name, arguments, timeout=2400, allow_failure=False):
        with Path(f"artifacts/{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        Path(f"artifacts/{name}-exit").write_text(str(result.returncode) + "\n")
        if result.returncode and not allow_failure:
            raise RuntimeError(f"{name} failed with code {result.returncode}")
        return result.returncode
    run("continuation-selection", ["scripts/model_select_checkpoint.py"], 60)
    selection = json.loads(Path("artifacts/training-v2-continuation/selection.json").read_text())
    adapter = selection["selected_checkpoint"]
    def smoke_batch(size):
        name = f"continuation-unmerged-control{size}"
        smoke_output = "artifacts/" + name
        code = run(name, ["scripts/model_validate.py", "--adapter", adapter,
            "--reference", str(Path(adapter) / "generation-validation.jsonl"),
            "--output", smoke_output, "--batch-size", str(size)], 600, allow_failure=True)
        if code:
            log = Path(f"artifacts/{name}.log").read_text()
            if size == 1 or "out of memory" not in log.lower():
                raise RuntimeError("native validation failed; inspect retained logs")
            return False
        return json.loads((Path(smoke_output) / "report.json").read_text())["batch_semantics_match_reference"]

    if not smoke_batch(1):
        raise RuntimeError("freshly loaded unmerged singles differ from selected validation; do not start full release")
    batch = 1
    for size in (8, 4):
        if smoke_batch(size):
            batch = size
            break
    Path("artifacts/training-v2-continuation/release-plan.json").write_text(json.dumps({
        "selected_checkpoint": adapter, "merge_adapter": False, "full_evaluation_batch_size": batch,
        "batch_size_basis": "Fresh single loading parity, then exact fixed-validation semantic/validity controls for each candidate batch size. Failed batch controls fall back to matched singles.",
        "merged_export_status": "Rejected alternative: material visual action/observation differences.",
        "full_evaluation_output": "artifacts/release-continuation-selected",
    }, indent=2) + "\n")
    run("release-continuation-selected", ["scripts/evaluate_release.py", "--adapter", adapter,
        "--batch-size", str(batch), "--policy", "data/mixed-v2/policy.json", "--max-new-tokens", "768",
        "--output", "artifacts/release-continuation-selected", "--quantization", "none", "--no-merge-adapter"])
    run("release-continuation-annotations", ["scripts/model_annotation_metrics.py", "--release-root",
                                           "artifacts/release-continuation-selected"], 60)


if __name__ == "__main__":
    main()

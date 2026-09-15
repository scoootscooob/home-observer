#!/usr/bin/env python3
"""Stage a minimal research payload for a Runpod job: code, configs and only the data it needs.

The staged tree contains no credentials, no reports, no previous artifacts and no
continuous videos (only the sampled window frames). The v1 adapter is copied to
data/adapter-v1 because the packager excludes artifacts/. Hashes are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

CODE = ["src", "scripts", "configs", "pyproject.toml", "requirements-gpu.txt", "requirements-vllm.txt", "README.md",
        "CONTRACT.md"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree(source: Path, target: Path, *, skip_dirs=()):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in skip_dirs or part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not path.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--dataset", default="data/continuous-v2")
    parser.add_argument("--adapter", default="artifacts/physical-training-v1/adapter")
    args = parser.parse_args()
    project, stage = args.project.resolve(), args.stage.resolve()
    if stage.exists():
        raise FileExistsError(f"refusing to reuse staging directory {stage}")
    stage.mkdir(parents=True)
    for name in CODE:
        source = project / name
        if source.is_dir():
            copy_tree(source, stage / name)
        elif source.is_file():
            shutil.copy2(source, stage / name)
    dataset = project / args.dataset
    copy_tree(dataset, stage / args.dataset, skip_dirs=("videos",))
    adapter = project / args.adapter
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (stage / "data/adapter-v1").mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter / name, stage / "data/adapter-v1" / name)
    forbidden = [str(p.relative_to(stage)) for p in stage.rglob("*")
                 if p.is_file() and (p.suffix in (".mp4", ".MP4", ".key", ".pem") or "credential" in p.name.lower()
                                     or "secret" in p.name.lower())]
    if forbidden:
        raise ValueError("staged payload contains forbidden files: " + ", ".join(forbidden[:5]))
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = {"staged_at": time.time(), "project": str(project), "dataset": args.dataset, "adapter": args.adapter,
                "file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files),
                "hashes": {str(p.relative_to(stage)): sha(p) for p in files
                           if p.suffix in (".jsonl", ".json", ".py", ".sh", ".toml", ".txt", ".safetensors")
                           and "media" not in p.parts}}
    (stage / "STAGING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "hashes"}, indent=2))


if __name__ == "__main__":
    main()

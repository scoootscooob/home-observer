#!/usr/bin/env python3
"""Transfer one selected adapter to an existing supervised Pod, with SHA256 checks."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from home_observer.cloud import SSH

FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)


def build_archive(source, archive):
    source, archive = Path(source), Path(archive)
    files = {}
    for name in FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required regular adapter file missing: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    config = json.loads((source / "adapter_config.json").read_text())
    manifest = {
        "files": files,
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "lora_rank": config.get("r"),
        "contains_optimizer_states": False,
        "remote_path": "artifacts/serving-adapter",
    }
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for name in FILES:
            tar.add(source / name, arcname=name, recursive=False)
    archive.chmod(0o600)
    manifest.update(
        archive_bytes=archive.stat().st_size, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local selected checkpoint directory")
    parser.add_argument("--run-dir", required=True, help="Existing supervised Runpod state directory")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    state = json.loads((run_dir / "state.json").read_text())
    if state.get("status") not in ("uploading", "starting_job", "running") or not state.get("ssh"):
        raise ValueError("Existing supervised Pod must have a current SSH endpoint")
    connection = state["ssh"]
    ssh = SSH(connection["host"], connection["port"], connection["key"], run_dir)
    # Refuse replacing an adapter already being served. A fresh run is reproducible.
    ssh.call("test ! -e /workspace/home-observer/artifacts/serving-adapter", timeout=25)
    archive = run_dir / "adapter-transfer.tar.gz"
    manifest = build_archive(args.source, archive)
    local_manifest = run_dir / "adapter-transfer-manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    local_manifest.chmod(0o600)
    ssh.copy(archive, "/workspace/adapter-transfer.tar.gz", timeout=300)
    ssh.copy(local_manifest, "/workspace/adapter-transfer-manifest.json", timeout=30)
    # Publish transfer-manifest.json last; the waiting server treats it as ready.
    code = """import hashlib,json,tarfile
from pathlib import Path
root=Path('/workspace/home-observer/artifacts/serving-adapter')
if root.exists():raise ValueError('Refusing to replace existing adapter')
manifest=json.loads(Path('/workspace/adapter-transfer-manifest.json').read_text())
archive=Path('/workspace/adapter-transfer.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
with tarfile.open(archive) as tar:
    members=tar.getmembers()
    if {m.name for m in members}!=set(manifest['files']) or not all(m.isfile() for m in members):
        raise ValueError('Unexpected adapter archive members')
    root.mkdir(parents=True)
    tar.extractall(root,filter='data')
for name,expected in manifest['files'].items():
    path=root/name
    if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
        raise ValueError('Adapter checksum mismatch: '+name)
manifest['remote_verified']=True
(root/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps({'remote_verified':True,'adapter_sha256':manifest['files']['adapter_model.safetensors']['sha256']}))
"""
    result = json.loads(ssh.call("python3 -", stdin=code.encode(), timeout=60))
    manifest.update(result)
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pod_id": state["pod_id"], "transfer_manifest": str(local_manifest), **result}))


if __name__ == "__main__":
    main()

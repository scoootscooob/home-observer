import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "transfer_adapter", Path(__file__).parents[1] / "scripts/runpod_transfer_adapter.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def selected_checkpoint(tmp_path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    for name in module.FILES:
        (source / name).write_bytes(b"adapter-artifact")
    (source / "adapter_config.json").write_text(
        json.dumps({"r": 16, "base_model_name_or_path": "google/gemma-4-E4B-it"})
    )
    return source


def test_archive_contains_only_inference_files_and_verified_hashes(tmp_path):
    source = selected_checkpoint(tmp_path)
    (source / "optimizer.pt").write_bytes(b"not-for-inference")
    (source / "credentials.json").write_bytes(b"private")
    archive = tmp_path / "adapter.tar.gz"
    manifest = module.build_archive(source, archive)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(module.FILES)
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (
        manifest["files"]["adapter_model.safetensors"]["sha256"]
        == hashlib.sha256(b"adapter-artifact").hexdigest()
    )
    assert manifest["contains_optimizer_states"] is False
    assert archive.stat().st_mode & 0o777 == 0o600


def test_archive_rejects_missing_or_symlinked_weights(tmp_path):
    source = selected_checkpoint(tmp_path)
    weights = source / "adapter_model.safetensors"
    weights.unlink()
    with pytest.raises(ValueError, match="regular adapter file"):
        module.build_archive(source, tmp_path / "adapter.tar.gz")
    elsewhere = tmp_path / "private-file"
    elsewhere.write_bytes(b"do-not-package")
    weights.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="regular adapter file"):
        module.build_archive(source, tmp_path / "adapter.tar.gz")

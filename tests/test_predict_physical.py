import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from home_observer.model import ModelConfig
from home_observer.physical_model import PhysicalNativeModel


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "scripts/predict_physical.py"
    spec = importlib.util.spec_from_file_location("predict_physical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_native_observe_per_row_retains_invalid_raw_and_excludes_labels(runner, tmp_path, monkeypatch):
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8)).save(image)
    window = {"window_id": "w", "started_at": 0.0, "ended_at": 1.0,
              "clips": [{"clip_id": "c", "camera_id": "ego", "started_at": 0.0, "ended_at": 1.0,
                         "frames": [{"evidence_id": "f", "timestamp": 0.0, "path": "frame.png"}]}],
              "audio": []}
    rows = [{"id": str(i), "split": "test", "window": window,
             "target": {"summary": "SECRET_TARGET_SENTINEL"}, "supervision_mask": {"summary": True},
             "provenance": {"narration": "SECRET_SOURCE_SENTINEL"}} for i in range(2)]
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    responses = ['{"summary":"visible scene","events":[],"objects":[],"observations":[]}',
                 '{"summary":"partial","events":[{"kind":"put-down"}]}']
    calls = []

    def generate(self, request):
        calls.append(request)
        return responses[len(calls) - 1], {"latency_s": 0.1, "output_tokens": 10}

    monkeypatch.setattr(PhysicalNativeModel, "generate_text", generate)

    def factory(config):
        native = runner.LoggedPhysicalModel.__new__(runner.LoggedPhysicalModel)
        native.config = config
        return native

    config = ModelConfig(dataset_root=str(tmp_path), quantization="none")
    output = tmp_path / "run"
    report = runner.run_predictions(dataset, config, output, native_factory=factory)
    assert report["complete"] and report["strictly_valid"] == 1
    assert len(calls) == 2
    assert "SECRET_" not in json.dumps(calls)
    assert all(set(request) == {"window", "tracks", "recent_events", "focus"} for request in calls)
    saved = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert saved[1]["raw_output"] == responses[1]
    assert "error" in saved[1]["result"] and saved[1]["result"]["metrics"]["latency_s"] == 0.1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["completed_rows"] == 2
    assert manifest["model_config"]["quantization"] == "none"
    assert manifest["media_sha256"]["frame.png"] == runner.sha(image)
    with pytest.raises(FileExistsError):
        runner.run_predictions(dataset, config, output, native_factory=factory)
    calls.clear()
    partial = runner.run_predictions(dataset, config, tmp_path / "partial", native_factory=factory, limit=1)
    assert partial["complete"] and not partial["dataset_complete"] and partial["source_rows"] == 2
    assert len((tmp_path / "partial/inputs.jsonl").read_text().splitlines()) == 1


def test_runner_rejects_quantization_and_bad_adapter_before_inference(runner, tmp_path):
    with pytest.raises(ValueError, match="BF16"):
        runner.run_predictions(tmp_path / "unused", ModelConfig(), tmp_path / "output")

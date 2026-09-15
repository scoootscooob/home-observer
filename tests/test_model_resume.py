"""Resume integrity checks; optional real tiny Gemma4 + PEFT + Trainer recovery."""
import copy
import json
import os
from pathlib import Path

import pytest

from home_observer.model import ModelConfig
from home_observer.train import seal_checkpoint, training_manifest, validate_resume_checkpoint


def manifest_for(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"example":1}\n')
    (tmp_path / "frame.png").write_bytes(b"original image bytes")
    return training_manifest(
        paths={"train": str(data)},
        rows=[{"window": {"frames": [{"path": "frame.png"}], "audio": []}}],
        config=ModelConfig(dataset_root=str(tmp_path)), policy={"version": 1},
        hyperparameters={"max_steps": 4}, versions={"test": "1"},
    )


def checkpoint_for(tmp_path, manifest):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
                 "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"opaque trusted local artifact")
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    seal_checkpoint(checkpoint, manifest, 2)
    return checkpoint


def test_resume_rejects_input_policy_prompt_and_media_changes(tmp_path):
    manifest = manifest_for(tmp_path)
    checkpoint = checkpoint_for(tmp_path, manifest)
    assert validate_resume_checkpoint(checkpoint, manifest, 4) == checkpoint
    for field in ("policy_sha256", "system_prompt_sha256", "datasets", "hyperparameters", "versions"):
        changed = copy.deepcopy(manifest)
        changed[field] = "changed"
        with pytest.raises(ValueError, match="mismatch"):
            validate_resume_checkpoint(checkpoint, changed, 4)
    (tmp_path / "frame.png").write_bytes(b"changed image bytes")
    # Keep row JSON unchanged: replacing the actual image must still block resume.
    changed = training_manifest(
        paths={"train": str(tmp_path / "train.jsonl")},
        rows=[{"window": {"frames": [{"path": "frame.png"}], "audio": []}}],
        config=ModelConfig(dataset_root=str(tmp_path)), policy={"version": 1},
        hyperparameters={"max_steps": 4}, versions={"test": "1"},
    )
    assert changed["datasets"] == manifest["datasets"]
    with pytest.raises(ValueError, match="media_sha256"):
        validate_resume_checkpoint(checkpoint, changed, 4)


def test_resume_rejects_inference_only_corruption_and_finished_runs(tmp_path):
    manifest = manifest_for(tmp_path)
    adapter = tmp_path / "old-adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="inference-only"):
        validate_resume_checkpoint(adapter, manifest, 4)
    checkpoint = checkpoint_for(tmp_path, manifest)
    with pytest.raises(ValueError, match="unfinished"):
        validate_resume_checkpoint(checkpoint, manifest, 2)
    (checkpoint / "optimizer.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="optimizer.pt"):
        validate_resume_checkpoint(checkpoint, manifest, 4)


@pytest.mark.skipif(os.environ.get("HOME_OBSERVER_TEST_TRAINER") != "1",
                    reason="requires optional torch/Transformers/PEFT CPU test environment")
def test_actual_tiny_gemma_trainer_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig
    from transformers.trainer import Trainer as real_trainer

    from home_observer import train

    torch.set_num_threads(1)
    class Processor:
        def save_pretrained(self, path):
            (Path(path) / "processor_config.json").write_text('{"tiny_test":true}')
    def components(*args, **kwargs):
        text = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, global_head_dim=16, max_position_embeddings=128,
                               vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
                               layer_types=["sliding_attention", "full_attention"], sliding_window=32)
        config = Gemma4Config(text_config=text, image_token_id=60, audio_token_id=61)
        model = Gemma4ForConditionalGeneration(config)
        model.name_or_path = ModelConfig().model_id
        return model, Processor()
    class Collator:
        def __init__(self, *args):
            pass
        def __call__(self, rows):
            index = rows[0]["index"]
            ids = torch.tensor([[2, 3 + index, 11, 12 + index, 1]])
            labels = ids.clone()
            labels[:, :2] = -100
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    monkeypatch.setattr(train, "load_components", components)
    monkeypatch.setattr(train, "DecisionCollator", Collator)
    real_arguments = transformers.TrainingArguments
    def cpu_arguments(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, gradient_checkpointing=False,
                      disable_tqdm=True, dataloader_pin_memory=False)
        return real_arguments(**kwargs)
    def components_for(trainer):
        return lambda: (trainer, transformers.TrainerCallback, cpu_arguments, transformers.set_seed)
    monkeypatch.setattr(train, "trainer_components", components_for(real_trainer))
    paths = {}
    for split in ("train", "validation"):
        rows = [{"id": f"{split}-{i}", "group_id": split, "split": split, "index": i,
                 "window": {"window_id": f"{split}-{i}", "started_at": 0., "ended_at": 1.,
                            "frames": [], "audio": [], "device_states": {}},
                 "target": {"summary": "No change.", "actions": [], "observations": [], "noop": True}}
                for i in range(4)]
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[split].write_text("".join(json.dumps(row) + "\n" for row in rows))
    kwargs = dict(train_path=str(paths["train"]), validation_path=str(paths["validation"]),
                  config=ModelConfig(dataset_root=str(tmp_path), quantization="none"),
                  max_steps=4, checkpoint_every=2, rank=2, gradient_accumulation_steps=1)
    complete_report = train.train_adapter(output=str(tmp_path / "complete"), **kwargs)
    class Interrupted(Exception):
        pass
    class StopAfterSave(transformers.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 2:
                raise Interrupted("simulated process failure after durable checkpoint")
    class InterruptingTrainer(real_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(StopAfterSave())
    monkeypatch.setattr(train, "trainer_components", components_for(InterruptingTrainer))
    with pytest.raises(Interrupted):
        train.train_adapter(output=str(tmp_path / "interrupted"), **kwargs)
    checkpoint = tmp_path / "interrupted/checkpoints/checkpoint-2"
    saved_optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    assert {int(state["step"]) for state in saved_optimizer["state"].values()} == {2}
    monkeypatch.setattr(train, "trainer_components", components_for(real_trainer))
    report = train.train_adapter(output=str(tmp_path / "interrupted"),
                                 resume_from_checkpoint=str(checkpoint), **kwargs)
    assert report["before"] is None and report["train"]["train_loss"] >= 0
    full = load_file(tmp_path / "complete/adapter_model.safetensors")
    resumed = load_file(tmp_path / "interrupted/adapter_model.safetensors")
    assert full.keys() == resumed.keys()
    assert all(torch.equal(full[name], resumed[name]) for name in full)
    state = json.loads((tmp_path / "interrupted/checkpoints/checkpoint-4/trainer_state.json").read_text())
    assert state["global_step"] == 4
    warm = train.train_adapter(output=str(tmp_path / "warm"), warm_start_adapter=str(tmp_path / "complete"),
                                **{**kwargs, "max_steps": 2})
    assert warm["initialization"]["kind"] == "adapter_weights_only_optimizer_reset"
    assert warm["before"]["eval_loss"] == pytest.approx(complete_report["after"]["eval_loss"], abs=1e-6)
    optimizer = torch.load(tmp_path / "warm/checkpoints/checkpoint-2/optimizer.pt", weights_only=True)
    assert {int(state["step"]) for state in optimizer["state"].values()} == {2}
    warm_weights = load_file(tmp_path / "warm/adapter_model.safetensors")
    assert any(not torch.equal(full[name], warm_weights[name]) for name in full)

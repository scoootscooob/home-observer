import copy
import json
import os
from pathlib import Path

import pytest
from test_physical_schema import physical_window

from home_observer.model import ModelConfig
from home_observer.physical_model import build_physical_messages, physical_request_from_row
from home_observer.physical_train import partial_label_mask, physical_training_target


def weak_row():
    return {"window": physical_window(), "target": {"summary": "put bowl", "events": [{
        "kind": "put", "object_label": "bowl", "started_at": 10.0, "ended_at": 11.0,
        "description": "put bowl"}]}, "supervision_mask": {"summary": True,
        "events": {"kind": True, "object_label": True, "started_at": True, "ended_at": True, "description": True},
        "objects": False, "observations": False}}


def test_partial_fields_never_create_missing_negative_or_identity_labels():
    text, partial = physical_training_target(weak_row())
    assert partial
    value = json.loads(text)
    assert set(value) == {"events", "summary"}
    assert set(value["events"][0]) == {"kind", "object_label", "started_at", "ended_at", "description"}
    for missing in ("objects", "observations", "confidence", "uncertainty", "subject_ids", "event_id", "pre_evidence_ids"):
        assert '"' + missing + '"' not in text
    bad = copy.deepcopy(weak_row())
    bad["supervision_mask"]["events"]["confidence"] = True
    with pytest.raises(ValueError, match="lacks an annotation"):
        physical_training_target(bad)


def test_partial_mask_excludes_prompt_media_closures_and_eos_but_keeps_string_braces():
    text = '{"events":[{"description":"use } shaped cup"}]}'
    ids = list(range(100, 100 + len(text)))
    full = [1, 7, 7, *ids, 2, 0]
    labels = partial_label_mask(full, [1, 7, 7], ids, [(i, i + 1) for i in range(len(text))], text,
                                attention_mask=[1] * (len(full) - 1) + [0], media_ids={7})
    assert labels[:3] == [-100] * 3
    assert labels[-2:] == [-100, -100]
    assert labels[3 + text.index("}")] != -100  # String content is an annotation.
    assert labels[3 + len(text) - 3:3 + len(text)] == [-100] * 3


def test_training_input_ignores_all_source_annotation_and_teacher_fields(tmp_path):
    (tmp_path / "before.jpg").write_bytes(b"fixture")
    (tmp_path / "after.jpg").write_bytes(b"fixture")
    row = weak_row()
    row.update(target={"summary": "LABEL_SENTINEL"}, narration="LABEL_SENTINEL",
               source_annotations={"answer": "LABEL_SENTINEL"}, tracks=[{"label": "LABEL_SENTINEL"}],
               teacher_output="LABEL_SENTINEL", focus=["LABEL_SENTINEL"])
    messages = build_physical_messages(physical_request_from_row(row), ModelConfig(dataset_root=str(tmp_path)))
    assert "LABEL_SENTINEL" not in json.dumps(messages)
    assert len([item for item in messages[1]["content"] if item["type"] == "image"]) == 2


@pytest.mark.skipif(os.environ.get("HOME_OBSERVER_TEST_TRAINER") != "1", reason="optional native CPU Trainer smoke")
def test_actual_physical_trainer_saves_full_checkpoint(tmp_path, monkeypatch):
    import torch
    import transformers
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    from home_observer import physical_train

    torch.set_num_threads(1)
    class Processor:
        def save_pretrained(self, directory):
            (Path(directory) / "processor_config.json").write_text('{"tiny_test":true}')
    def components(*args, **kwargs):
        text = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, global_head_dim=16, max_position_embeddings=128,
                               vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
                               layer_types=["sliding_attention", "full_attention"], sliding_window=32)
        model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, image_token_id=60, audio_token_id=61))
        model.name_or_path = ModelConfig().model_id
        return model, Processor()
    class Collator:
        def __init__(self, *args):
            pass
        def __call__(self, rows):
            ids = torch.tensor([[2, 3, 11, 12, 1]])
            labels = ids.clone()
            labels[:, :2] = -100
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    monkeypatch.setattr(physical_train, "load_components", components)
    monkeypatch.setattr(physical_train, "PhysicalCollator", Collator)
    original_arguments = transformers.TrainingArguments
    def cpu_arguments(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, gradient_checkpointing=False,
                      disable_tqdm=True, dataloader_pin_memory=False)
        return original_arguments(**kwargs)
    monkeypatch.setattr(transformers, "TrainingArguments", cpu_arguments)
    paths = {}
    for split in ("train", "validation"):
        rows = [{"id": split, "group_id": split, "split": split,
                 "window": {"window_id": split, "started_at": 0.0, "ended_at": 1.0, "clips": []},
                 "target": {"summary": "put bowl"}, "supervision_mask": {"summary": True}}]
        paths[split] = tmp_path / (split + ".jsonl")
        paths[split].write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = physical_train.train_physical(train_path=paths["train"], validation_path=paths["validation"],
        output=tmp_path / "new-physical-adapter", config=ModelConfig(dataset_root=str(tmp_path), quantization="none"),
        max_steps=2, checkpoint_every=1, rank=2, gradient_accumulation_steps=1, generation_limit=0)
    assert report["task"] == "physical_perception"
    assert report["partial_train_rows"] == 1
    checkpoint = tmp_path / "new-physical-adapter/checkpoints/checkpoint-2"
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "adapter_model.safetensors", "resume_manifest.json"):
        assert (checkpoint / filename).is_file()
    state = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    assert {int(item["step"]) for item in state["state"].values()} == {2}

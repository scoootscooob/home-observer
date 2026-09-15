"""Fresh language-only LoRA for physical perception with explicit partial labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .model import ModelConfig, encode_messages, load_components
from .physical_model import (
    PHYSICAL_SYSTEM_PROMPT,
    PhysicalNativeModel,
    build_physical_messages,
    physical_request_from_row,
    prepare_physical_request,
    validate_physical_audio,
)
from .physical_schema import (
    PhysicalDecision,
    PhysicalObject,
    PhysicalObservation,
    TemporalEvent,
    validate_physical_decision,
)
from .train import (
    assistant_labels,
    atomic_json,
    ensure_disjoint,
    file_sha256,
    language_lora_targets,
    modality_token_ids,
    seal_checkpoint,
    validate_resume_checkpoint,
)

FIELD_MODELS = {"objects": PhysicalObject, "observations": PhysicalObservation, "events": TemporalEvent}


def physical_training_target(row: dict) -> tuple[str, bool]:
    """Serialize only positively selected labels; omitted arrays never become negatives."""
    target, mask = row.get("target"), row.get("supervision_mask")
    if not isinstance(target, dict) or not target:
        raise ValueError("physical training requires explicit target annotations")
    if mask is None:
        parsed = validate_physical_decision(target, row["window"])
        complete = parsed.model_dump()
        return json.dumps({key: complete[key] for key in ("events", "objects", "observations", "summary")},
                          ensure_ascii=False, separators=(",", ":")), False
    if not isinstance(mask, dict) or set(mask) - {"summary", *FIELD_MODELS}:
        raise ValueError("unsupported physical supervision mask")
    selected = {}
    # Events lead to avoid a repeated summary deciding all subsequent output.
    for key in ("events", "objects", "observations", "summary"):
        choice = mask.get(key, False)
        if choice is False:
            continue
        if key not in target:
            raise ValueError("supervision mask selects a missing target: " + key)
        value = target[key]
        if key == "summary":
            if choice is not True or not isinstance(value, str) or not 0 < len(value.strip()) <= 3000:
                raise ValueError("summary supervision requires explicit nonempty text")
            selected[key] = value
        elif choice is True:
            if not isinstance(value, list):
                raise ValueError("physical collection label must be a list")
            selected[key] = [FIELD_MODELS[key].model_validate(item).model_dump() for item in value]
        elif isinstance(choice, dict):
            fields = FIELD_MODELS[key].model_fields
            if not choice or set(choice) - fields.keys() or any(flag not in (True, False) for flag in choice.values()):
                raise ValueError("unknown or invalid supervised physical field")
            if not isinstance(value, list) or not value:
                raise ValueError("partial field supervision requires positive annotated items, not an empty collection")
            items = []
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("partial physical target must be an object")
                kept = {}
                for field, flag in choice.items():
                    if flag:
                        if field not in item:
                            raise ValueError("selected physical field lacks an annotation: " + field)
                        # Validate individual field types/bounds without creating defaults
                        # for the unannotated remainder of the runtime object.
                        from pydantic import TypeAdapter
                        kept[field] = TypeAdapter(fields[field].rebuild_annotation()).validate_python(item[field])
                if not kept:
                    raise ValueError("partial physical item has no supervised fields")
                if "started_at" in kept and "ended_at" in kept:
                    window = row["window"]
                    if not window["started_at"] <= kept["started_at"] <= kept["ended_at"] <= window["ended_at"]:
                        raise ValueError("annotated event interval is outside its source clip")
                items.append(kept)
            selected[key] = items
        else:
            raise ValueError("supervision mask values must be booleans or field masks")
    if not selected:
        raise ValueError("physical row has no supervised labels")
    text = json.dumps(selected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text) > 64000:
        raise ValueError("physical target exceeds text budget")
    return text, True


def partial_label_mask(full_ids, prefix_ids, target_ids, offsets, target_text, *, attention_mask, media_ids):
    """Mask incomplete container endings/EOS, plus every prompt/media/padding token."""
    labels = assistant_labels(full_ids, prefix_ids, attention_mask=attention_mask, multimodal_token_ids=media_ids)
    start = len(prefix_ids)
    if full_ids[start:start + len(target_ids)] != target_ids:
        raise ValueError("physical target boundary tokenization mismatch")
    closing = set()
    quoted, escaped = False, False
    for index, char in enumerate(target_text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "}]":
            closing.add(index)
    for position in range(start, len(labels)):
        relative = position - start
        if relative >= len(target_ids) or any(offsets[relative][0] <= char < offsets[relative][1] for char in closing):
            labels[position] = -100
    if all(label == -100 for label in labels):
        raise ValueError("partial physical row has no supervised tokens")
    return labels


class PhysicalCollator:
    def __init__(self, processor, config: ModelConfig):
        self.processor, self.config = processor, config

    def __call__(self, rows):
        import torch
        if len(rows) != 1:
            raise ValueError("physical multimodal training uses batch size one")
        row = rows[0]
        text, partial = physical_training_target(row)
        request = physical_request_from_row(row)
        messages = build_physical_messages(request, self.config)
        validate_physical_audio(request, self.config)
        prefix = encode_messages(self.processor, messages, self.config, generation=True)
        full = [*messages, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
        batch = encode_messages(self.processor, full, self.config, generation=False)
        full_ids, prefix_ids = batch["input_ids"][0].tolist(), prefix["input_ids"][0].tolist()
        attention = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))[0].tolist()
        media = modality_token_ids(self.processor)
        if partial:
            tokens = self.processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            labels = partial_label_mask(full_ids, prefix_ids, tokens["input_ids"], tokens["offset_mapping"], text,
                                        attention_mask=attention, media_ids=media)
        else:
            labels = assistant_labels(full_ids, prefix_ids, attention_mask=attention, multimodal_token_ids=media)
        batch["labels"] = torch.tensor([labels], dtype=torch.long)
        return dict(batch)


def load_physical_rows(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError("physical dataset must contain the requested nonempty split")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate physical dataset row ID")
    for row in rows:
        physical_training_target(row)
        physical_request_from_row(row)
    return rows


def train_physical(*, train_path, validation_path, output, config, max_steps=100, rank=16,
                   learning_rate=1e-4, gradient_accumulation_steps=2, checkpoint_every=25,
                   generation_limit=4, seed=42, resume_from_checkpoint=None):
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    if config.adapter_path or config.should_merge_adapter:
        raise ValueError("physical training starts from the pinned base, not a home-control adapter")
    if max_steps < 1 or rank < 1 or checkpoint_every < 1 or not 0 <= generation_limit <= 8:
        raise ValueError("invalid physical training limits")
    train_rows, val_rows = load_physical_rows(train_path, "train"), load_physical_rows(validation_path, "validation")
    ensure_disjoint(train_rows, val_rows)
    media = {}
    for row in [*train_rows, *val_rows]:
        clean = prepare_physical_request(physical_request_from_row(row), config)
        for item in [*[frame for clip in clean["window"]["clips"] for frame in clip["frames"]],
                     *clean["window"]["audio"]]:
            path = Path(item["path"])
            relative = str(path.relative_to(Path(config.dataset_root).resolve()))
            media[relative] = file_sha256(path)
    model_config = asdict(config)
    model_config.pop("dataset_root")
    manifest = {"version": 1, "task": "physical_perception", "initialization": {"kind": "base_model"},
                "datasets": {"train": file_sha256(train_path), "validation": file_sha256(validation_path)},
                "media_sha256": hashlib.sha256(json.dumps(media, sort_keys=True).encode()).hexdigest(),
                "model_config": model_config, "physical_prompt_sha256": hashlib.sha256(PHYSICAL_SYSTEM_PROMPT.encode()).hexdigest(),
                "hyperparameters": {"max_steps": max_steps, "rank": rank, "learning_rate": learning_rate,
                    "gradient_accumulation_steps": gradient_accumulation_steps, "seed": seed,
                    "checkpoint_every": checkpoint_every, "generation_limit": generation_limit},
                "implementation": {name: file_sha256(Path(__file__).with_name(name)) for name in
                    ["physical_model.py", "physical_schema.py", "physical_train.py", "model.py"]},
                "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")}}
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()) and not resume_from_checkpoint:
        raise ValueError("physical training requires a fresh output directory; baseline adapters remain untouched")
    resume = validate_resume_checkpoint(resume_from_checkpoint, manifest, max_steps) if resume_from_checkpoint else None
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "training_manifest.json", manifest)
    set_seed(seed)
    model, processor = load_components(config, training=True)
    model.requires_grad_(False)
    if config.quantization == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        for parameter in model.parameters():
            if not parameter.requires_grad and parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
    targets = language_lora_targets([name for name, _ in model.named_modules()])
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=.05,
                                           target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or "language_model" not in name for name, _ in trainable):
        raise RuntimeError("physical training unexpectedly enabled non-language-LoRA parameters")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    collator = PhysicalCollator(processor, config)

    class SavePhysical(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            processor.save_pretrained(checkpoint)
            seal_checkpoint(checkpoint, manifest, state.global_step)
            if not generation_limit:
                return control
            active = kwargs["model"]
            rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
            was_training = active.training
            active.eval()
            results = []
            try:
                native = PhysicalNativeModel.from_loaded(active, processor, config)
                with (checkpoint / "generation-validation.jsonl").open("w") as stream:
                    for row in val_rows[:generation_limit]:
                        request = physical_request_from_row(row)
                        raw, metrics = native.generate_text(request)
                        result = {"metrics": metrics, "total_latency_s": metrics["latency_s"]}
                        try:
                            decision = validate_physical_decision(PhysicalDecision.model_validate_json(raw), request["window"])
                            result["decision"] = decision.model_dump()
                        except ValueError as exc:
                            result["error"] = str(exc)
                        item = {key: row[key] for key in ("id", "window", "target", "supervision_mask", "annotation_coverage") if key in row}
                        item.update(raw_output=raw, result=result)
                        results.append(item)
                        stream.write(json.dumps(item) + "\n")
                        stream.flush()
                report = {"step": state.global_step, "rows": len(results),
                          "strictly_valid": sum("error" not in item["result"] for item in results),
                          "scope": "Weak source interaction labels do not validate boxes, identities, silence, or confidence."}
                try:
                    from .physical_evaluate import score_predictions
                    report["quality"] = score_predictions(results)
                except ImportError:
                    report["quality"] = None
                atomic_json(checkpoint / "generation-report.json", report)
                print(json.dumps({"physical_generation_validation": report}), flush=True)
            finally:
                active.train(was_training)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                torch.cuda.set_rng_state_all(rng[3])
            return control

    args = TrainingArguments(output_dir=str(destination / "checkpoints"), max_steps=max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate,
        warmup_steps=math.ceil(max_steps * .05), weight_decay=.01, bf16=True, fp16=False,
        optim="adamw_torch", lr_scheduler_type="cosine", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="steps",
        save_steps=checkpoint_every, save_only_model=False, eval_strategy="no", report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=val_rows,
                      data_collator=collator, callbacks=[SavePhysical()])
    before = None if resume else trainer.evaluate()
    trained = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    after = trainer.evaluate()
    if not math.isfinite(after["eval_loss"]):
        raise RuntimeError("nonfinite physical validation loss")
    if any(not torch.isfinite(parameter).all().item() for _, parameter in trainable):
        raise RuntimeError("nonfinite physical adapter weights; refusing final export")
    model.save_pretrained(destination)
    processor.save_pretrained(destination)
    report = {"task": "physical_perception", "train_rows": len(train_rows), "validation_rows": len(val_rows),
              "partial_train_rows": sum(physical_training_target(row)[1] for row in train_rows),
              "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
              "lora_targets": targets, "before": before, "train": trained.metrics, "after": after,
              "scope": "Loss on observed source annotation fields only; no unannotated object/track/absence accuracy claim."}
    atomic_json(destination / "training_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--generation-limit", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = ModelConfig(dataset_root=args.dataset_root, quantization=args.quantization,
                         max_frames=args.max_frames, max_input_tokens=args.max_input_tokens,
                         max_new_tokens=args.max_new_tokens, merge_adapter=False)
    print(json.dumps(train_physical(train_path=args.train, validation_path=args.validation,
        output=args.output, config=config, max_steps=args.max_steps, rank=args.rank,
        learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_every=args.checkpoint_every, generation_limit=args.generation_limit,
        resume_from_checkpoint=args.resume_from_checkpoint), indent=2))


if __name__ == "__main__":
    main()

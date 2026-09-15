"""QLoRA training on causal windows, supervised only on assistant Decision tokens."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import threading
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from .model import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    ModelConfig,
    NativeModel,
    build_messages,
    encode_messages,
    load_components,
    resolve_media_path,
    validate_audio_files,
    verify_decision_evidence,
)
from .prompts import SYSTEM_PROMPT
from .schema import Decision


def request_from_row(row: dict, policy: dict | None = None) -> dict:
    """Labels, teacher outputs, annotations, provenance and IDs never enter model inputs."""
    state, recent_events = training_context(row)
    return {"window": row["window"], "state": state, "recent_events": recent_events, "policy": policy or {}}


def training_context(row: dict) -> tuple[dict, list[dict]]:
    """Only versioned, same-recording historical context is an authorized input."""
    context = row.get("runtime_context")
    if context is None:
        return {}, []
    fields = {"version", "state", "recent_events", "source_window_end", "source_group_id"}
    if (not isinstance(context, dict) or set(context) != fields or
            type(context["version"]) is not int or context["version"] != 1):
        raise ValueError("runtime_context requires the exact version 1 fields")
    if not row.get("group_id") or context["source_group_id"] != row["group_id"]:
        raise ValueError("runtime_context must come from the same group")
    cutoff = context["source_window_end"]
    def finite_time(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not finite_time(cutoff) or not cutoff < row["window"]["started_at"]:
        raise ValueError("runtime_context source must end strictly before the current window starts")
    state, events = context["state"], context["recent_events"]
    if not isinstance(state, dict) or not isinstance(events, list):
        raise ValueError("runtime_context state/events have invalid types")
    for entity, attributes in state.items():
        if not isinstance(entity, str) or not isinstance(attributes, dict):
            raise ValueError("runtime_context state requires Journal-shaped entities and attributes")
        for attribute, fact in attributes.items():
            if not isinstance(attribute, str) or not isinstance(fact, dict):
                raise ValueError("runtime_context state requires Journal-shaped facts")
            observed = fact.get("observed_at")
            if not finite_time(observed) or observed > cutoff:
                raise ValueError("runtime_context fact is missing a causal observed_at timestamp")
    for event in events:
        if not isinstance(event, dict) or not finite_time(event.get("timestamp")) or event["timestamp"] > cutoff:
            raise ValueError("runtime_context event is missing a causal timestamp")
        for field in ("ended_at", "observed_at"):
            if field in event and (not finite_time(event[field]) or event[field] > cutoff):
                raise ValueError("runtime_context event includes a future timestamp")
    return copy.deepcopy(state), copy.deepcopy(events)


def assistant_labels(input_ids: list[int], prompt_ids: list[int], *,
                     attention_mask: list[int] | None = None,
                     multimodal_token_ids: set[int] | None = None) -> list[int]:
    """Fail closed if the real expanded multimodal prompt is not a token-exact prefix."""
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not preserve generation prefix; refusing unsafe loss masking")
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("training row contains no assistant completion")
    masked_ids = multimodal_token_ids or set()
    labels = [(-100 if i < len(prompt_ids) or token in masked_ids or
               (attention_mask is not None and not attention_mask[i]) else token)
              for i, token in enumerate(input_ids)]
    if all(label == -100 for label in labels):
        raise ValueError("training row has zero supervised tokens")
    return labels


def modality_token_ids(processor) -> set[int]:
    tokenizer = processor.tokenizer
    ids = set()
    for token in tokenizer.all_special_tokens:
        if any(part in token.lower() for part in ("image", "audio", "video", "boi", "eoi", "boa", "eoa")):
            ids.add(tokenizer.convert_tokens_to_ids(token))
    for obj in (processor, tokenizer):
        for name in ("image_token_id", "audio_token_id", "video_token_id"):
            value = getattr(obj, name, None)
            if isinstance(value, int):
                ids.add(value)
    return ids


def language_lora_targets(module_names: list[str]) -> list[str]:
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name in module_names
               if "language_model.layers." in name and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise ValueError("no Gemma4 language projection modules found; inspect model structure")
    if any("vision" in name or "audio" in name for name in targets):
        raise ValueError("LoRA target unexpectedly includes a modality encoder")
    return targets


def training_target(row: dict, decision_field_order: str = "summary-first") -> tuple[str, bool]:
    """Partial human captions supervise summary only, never missing action/noop fields."""
    mask = row.get("supervision_mask")
    if mask == ["summary"] or mask == {"summary": True, "actions": False, "observations": False, "noop": False}:
        target = row["target"]
        if set(target) != {"summary"} or not isinstance(target["summary"], str) or not target["summary"].strip():
            raise ValueError("summary-only supervision requires a nonempty summary and no other target fields")
        if len(target["summary"]) > 3000:
            raise ValueError("summary exceeds Decision schema limit")
        return json.dumps(target, ensure_ascii=False, separators=(",", ":")), True
    if mask not in (None, [], {"summary": True, "actions": True, "observations": True, "noop": True}):
        raise ValueError("unsupported supervision_mask")
    target = Decision.model_validate(row["target"])
    values = target.model_dump()
    if decision_field_order == "actions-first":
        values = {key: values[key] for key in ("actions", "observations", "noop", "summary")}
    elif decision_field_order != "summary-first":
        raise ValueError("unsupported Decision field order")
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")), False


class DecisionCollator:
    """Batch size one avoids incorrect padding/flattening across variable image/audio counts."""
    def __init__(self, processor, config: ModelConfig, policy: dict | None = None, decision_field_order="summary-first"):
        self.processor, self.config, self.policy = processor, config, policy or {}
        self.decision_field_order = decision_field_order

    def __call__(self, rows: list[dict]) -> dict:
        import torch
        if len(rows) != 1:
            raise ValueError("multimodal collator requires per_device_train_batch_size=1")
        row = rows[0]
        target_text, summary_only = training_target(row, self.decision_field_order)
        request = request_from_row(row, self.policy)
        if not summary_only:
            verify_decision_evidence(Decision.model_validate(row["target"]), request)
        messages = build_messages(request, self.config)
        validate_audio_files(request, self.config)
        prefix = encode_messages(self.processor, messages, self.config, generation=True)
        full = messages + [{"role": "assistant", "content": [
            {"type": "text", "text": target_text}
        ]}]
        batch = encode_messages(self.processor, full, self.config, generation=False)
        labels = assistant_labels(
            batch["input_ids"][0].tolist(), prefix["input_ids"][0].tolist(),
            attention_mask=batch.get("attention_mask", torch.ones_like(batch["input_ids"]))[0].tolist(),
            multimodal_token_ids=modality_token_ids(self.processor),
        )
        if summary_only:
            # Token-exact standalone target offsets identify the terminal object brace.
            # Mask that brace and turn closure: a caption does not teach early termination
            # of the complete runtime Decision, or imply empty observations/actions.
            target_tokens = self.processor.tokenizer(target_text, add_special_tokens=False,
                                                       return_offsets_mapping=True)
            start = prefix["input_ids"].shape[-1]
            token_ids = target_tokens["input_ids"]
            if batch["input_ids"][0, start:start + len(token_ids)].tolist() != token_ids:
                raise ValueError("summary target boundary tokenization mismatch")
            for offset in range(start, len(labels)):
                relative = offset - start
                if relative >= len(token_ids) or target_tokens["offset_mapping"][relative][1] > len(target_text) - 1:
                    labels[offset] = -100
            if all(label == -100 for label in labels):
                raise ValueError("summary-only row has zero supervised tokens")
        batch["labels"] = torch.tensor([labels], dtype=torch.long)
        return dict(batch)


def load_rows(path: str, expected_split: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"{path}:{line_number}: expected split {expected_split}")
            if "target" not in row:
                raise ValueError("unannotated public windows are evaluation inputs, not negative labels")
            training_target(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"empty {expected_split} dataset")
    return rows


def ensure_disjoint(train: list[dict], validation: list[dict]) -> None:
    def groups(rows):
        result = set()
        for row in rows:
            group = row.get("group_id") or row.get("recording_id")
            if not group:
                raise ValueError("every training/evaluation row requires a recording/scenario group_id")
            result.add(str(group))
        return result
    overlap = groups(train) & groups(validation)
    if overlap:
        raise ValueError("train/validation recording or scenario groups overlap: " + ", ".join(sorted(overlap)))
    if {row["id"] for row in train} & {row["id"] for row in validation}:
        raise ValueError("train/validation IDs overlap")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_manifest(*, paths: dict[str, str | None], rows: list[dict], config: ModelConfig,
                      policy: dict | None, hyperparameters: dict, versions: dict) -> dict:
    """Fingerprint inputs and media bytes; roots may move, training semantics may not."""
    media = {}
    root = Path(config.dataset_root).resolve()
    for row in rows:
        for item in [*row["window"]["frames"], *row["window"]["audio"]]:
            path = resolve_media_path(item["path"], root, config.max_media_bytes)
            relative = str(path.relative_to(root))
            if relative not in media:
                media[relative] = file_sha256(path)
    model_config = asdict(config)
    model_config.pop("dataset_root")
    return {
        "version": 1,
        "datasets": {name: file_sha256(path) if path else None for name, path in paths.items()},
        "media_sha256": hashlib.sha256(json.dumps(media, sort_keys=True).encode()).hexdigest(),
        "media_files": len(media), "model_config": model_config,
        "policy_sha256": hashlib.sha256(json.dumps(policy or {}, sort_keys=True).encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "implementation": {name: file_sha256(Path(__file__).with_name(name))
                           for name in ("train.py", "model.py", "prompts.py", "schema.py")},
        "hyperparameters": hyperparameters, "versions": versions,
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal_checkpoint(checkpoint: Path, manifest: dict, step: int) -> None:
    """Run after Trainer saved optimizer/scheduler/RNG; marker is written last."""
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
                "adapter_config.json", "adapter_model.safetensors"}
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise ValueError("incomplete resumable checkpoint: missing " + ", ".join(missing))
    artifacts = {path.name: file_sha256(path) for path in checkpoint.iterdir()
                 if path.is_file() and path.name not in {"resume_manifest.json", "resume_manifest.json.tmp"}}
    atomic_json(checkpoint / "resume_manifest.json", {"training": manifest, "global_step": step,
                                                      "artifacts": artifacts})


def validate_resume_checkpoint(checkpoint: str | Path, manifest: dict, max_steps: int) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    marker = checkpoint / "resume_manifest.json"
    if not marker.is_file():
        raise ValueError("checkpoint has no complete resume manifest; inference-only adapters cannot resume training")
    saved = json.loads(marker.read_text())
    if saved.get("training") != manifest:
        changed = sorted(key for key in set(manifest) | set(saved.get("training", {}))
                         if saved.get("training", {}).get(key) != manifest.get(key))
        raise ValueError("resume configuration/input mismatch: " + ", ".join(changed))
    for name, expected in saved.get("artifacts", {}).items():
        if Path(name).name != name or not (checkpoint / name).is_file() or file_sha256(checkpoint / name) != expected:
            raise ValueError("checkpoint artifact missing or changed: " + name)
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
                "adapter_config.json", "adapter_model.safetensors"}
    if not required <= saved.get("artifacts", {}).keys():
        raise ValueError("checkpoint manifest does not cover all required training states")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = state.get("global_step")
    if type(step) is not int or step != saved.get("global_step") or not 0 < step < max_steps:
        raise ValueError("resume requires a consistent unfinished global_step below max_steps")
    return checkpoint


def warm_start_lineage(path: str | None, config: ModelConfig, rank: int) -> dict | None:
    """Warm start imports adapter weights only and explicitly resets optimizer state."""
    if not path:
        return None
    parent = Path(path).expanduser().resolve(strict=True)
    settings = json.loads((parent / "adapter_config.json").read_text())
    expected = {"peft_type": "LORA", "r": rank, "lora_alpha": rank * 2, "lora_dropout": 0.05,
                "bias": "none", "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("warm-start adapter LoRA settings differ from this training configuration")
    if settings.get("base_model_name_or_path") not in {config.model_id, ""}:
        raise ValueError("warm-start adapter base model differs")
    return {"kind": "adapter_weights_only_optimizer_reset",
            "adapter_config_sha256": file_sha256(parent / "adapter_config.json"),
            "adapter_weights_sha256": file_sha256(parent / "adapter_model.safetensors")}


def trainer_components():
    """Lazy framework imports, also the integration seam for CPU recovery tests."""
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    return Trainer, TrainerCallback, TrainingArguments, set_seed


def train_adapter(*, train_path: str, validation_path: str, output: str, config: ModelConfig,
                  policy: dict | None = None, max_steps: int = 100, learning_rate: float = 1e-4,
                  rank: int = 8, gradient_accumulation_steps: int = 4, seed: int = 42,
                  decision_field_order: str = "summary-first", generation_validation_path: str | None = None,
                  generation_eval_every: int = 100, checkpoint_every: int = 100,
                  resume_from_checkpoint: str | None = None, warm_start_adapter: str | None = None) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    Trainer, TrainerCallback, TrainingArguments, set_seed = trainer_components()
    if config.adapter_path:
        raise ValueError("use resume_from_checkpoint for training state; adapter_path is inference-only")
    if min(max_steps, rank, gradient_accumulation_steps, checkpoint_every) <= 0:
        raise ValueError("steps, rank, accumulation and checkpoint interval must be positive")
    train_rows, val_rows = load_rows(train_path, "train"), load_rows(validation_path, "validation")
    ensure_disjoint(train_rows, val_rows)
    if decision_field_order not in {"summary-first", "actions-first"}:
        raise ValueError("unsupported Decision field order")
    generation_rows = load_rows(generation_validation_path, "validation") if generation_validation_path else []
    if generation_rows:
        ensure_disjoint(train_rows, generation_rows)
        if not 1 <= len(generation_rows) <= 16 or generation_eval_every <= 0:
            raise ValueError("generation validation requires 1..16 rows and a positive interval")
        if generation_eval_every % checkpoint_every:
            raise ValueError("generation validation interval must be a multiple of checkpoint interval")
    # Validate evidence and file boundaries before acquiring model memory.
    for row in [*train_rows, *val_rows, *generation_rows]:
        req = request_from_row(row, policy)
        build_messages(req, config)
        if not training_target(row)[1]:
            verify_decision_evidence(Decision.model_validate(row["target"]), req)
    manifest = training_manifest(
        paths={"train": train_path, "validation": validation_path, "generation": generation_validation_path},
        rows=[*train_rows, *val_rows, *generation_rows], config=config, policy=policy,
        hyperparameters={"max_steps": max_steps, "learning_rate": learning_rate, "rank": rank,
                         "gradient_accumulation_steps": gradient_accumulation_steps, "seed": seed,
                         "decision_field_order": decision_field_order, "checkpoint_every": checkpoint_every,
                         "generation_eval_every": generation_eval_every},
        versions={name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
    )
    lineage = warm_start_lineage(warm_start_adapter, config, rank)
    manifest["initialization"] = lineage or {"kind": "base_model"}
    resume_path = validate_resume_checkpoint(resume_from_checkpoint, manifest, max_steps) if resume_from_checkpoint else None
    output_path = Path(output)
    if not resume_path and (output_path / "training_manifest.json").exists():
        raise ValueError("output already contains a training run; use a fresh output or explicit resume checkpoint")
    output_path.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path / "training_manifest.json", manifest)
    set_seed(seed)
    model, processor = load_components(config, training=True)
    if config.quantization == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        # PEFT upcasts all non-quantized parameters to FP32, including Gemma's huge
        # frozen embedding tables. BF16 encoders/embeddings preserve the memory budget.
        for parameter in model.parameters():
            if not parameter.requires_grad and parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
    else:
        model.requires_grad_(False)
    targets = language_lora_targets([name for name, _ in model.named_modules()])
    if warm_start_adapter:
        parent_config = json.loads((Path(warm_start_adapter) / "adapter_config.json").read_text())
        if set(parent_config["target_modules"]) != set(targets):
            raise ValueError("warm-start adapter targets differ from verified language-only projections")
        model = PeftModel.from_pretrained(model, warm_start_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ))
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or "language_model" not in name for name, _ in trainable):
        raise RuntimeError("unexpected trainable parameters: only language LoRA is permitted")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    collator = DecisionCollator(processor, config, policy, decision_field_order)
    generation_reports = []

    class GenerationCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
            processor.save_pretrained(checkpoint)
            seal_checkpoint(checkpoint, manifest, step)
            if not generation_rows or (step % generation_eval_every and step != max_steps):
                return control
            from .evaluate import summarize
            from .policy import Policy
            from .schema import InferenceRequest, parse_decision
            active_model = kwargs["model"]
            # Generation diagnostics must not alter the training RNG relative to
            # the saved checkpoint, including future diagnostic code changes.
            import random

            import numpy as np
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            was_training = active_model.training
            active_model.eval()
            native = NativeModel.__new__(NativeModel)
            native.config = replace(config, adapter_path=str(checkpoint), merge_adapter=False)
            native.model, native.processor = active_model, processor
            native._lock, native.adapter_merged = threading.Lock(), False
            results = []
            try:
                configured_policy = Policy(**policy) if policy and "entities" in policy else None
                with (checkpoint / "generation-validation.jsonl").open("w", encoding="utf-8") as stream:
                    for row in generation_rows:
                        request = request_from_row(row, policy)
                        raw, metrics = native.generate_text(request)
                        result = {"metrics": metrics, "total_latency_s": metrics["latency_s"], "rejections": []}
                        try:
                            decision = parse_decision(raw)
                            verify_decision_evidence(decision, request)
                            result["decision"] = decision.model_dump()
                            if configured_policy:
                                result["rejections"] = configured_policy.validate(decision, InferenceRequest.model_validate(request).window)
                        except ValueError as exc:
                            result["error"] = str(exc)
                        item = {"id": row["id"], "task_type": row.get("task_type", "unspecified"),
                                "target": row["target"], "supervision_mask": row.get("supervision_mask", {}),
                                "raw_output": raw, "result": result}
                        results.append(item)
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        print(json.dumps({"generation_validation_step": step, "id": row["id"],
                                          "raw_output": raw, "latency_s": metrics["latency_s"]}), flush=True)
                report = {"step": step, "checkpoint": str(checkpoint), "quality": summarize(results),
                          "decision_field_order": decision_field_order,
                          "validation_sha256": hashlib.sha256(Path(generation_validation_path).read_bytes()).hexdigest()}
                (checkpoint / "generation-report.json").write_text(json.dumps(report, indent=2) + "\n")
                generation_reports.append(report)
                print(json.dumps({"generation_validation": report}), flush=True)
            finally:
                active_model.train(was_training)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state_all(rng[3])
                del native
            return control
    args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"), max_steps=max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, weight_decay=0.01, warmup_steps=math.ceil(max_steps * 0.05),
        bf16=True, fp16=False, optim="adamw_torch", lr_scheduler_type="cosine",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=checkpoint_every,
        save_only_model=False, eval_strategy="no", report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_rows,
                      eval_dataset=val_rows, data_collator=collator,
                      callbacks=[GenerationCheckpoint()])
    before = None if resume_path else trainer.evaluate()
    result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    after = trainer.evaluate()
    if not all(math.isfinite(float(item["eval_loss"])) for item in (before, after) if item is not None):
        raise RuntimeError("non-finite validation loss; refusing to save a broken adapter")
    if any(not torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
        raise RuntimeError("non-finite LoRA weights; refusing to save a broken adapter")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    import peft
    import transformers
    report = {
        "model_id": config.model_id, "revision": config.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
        "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "trainable_parameters": sum(n for _, n in trainable), "lora_targets": targets,
        "steps": max_steps, "seed": seed, "quantization": config.quantization,
        "learning_rate": learning_rate, "rank": rank,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "policy_sha256": hashlib.sha256(json.dumps(policy or {}, sort_keys=True).encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "decision_field_order": decision_field_order,
        "generation_validation": generation_reports,
        "checkpoint_every": checkpoint_every,
        "resume_from_checkpoint": str(resume_path) if resume_path else None,
        "resumable_checkpoint_root": str(output_path / "checkpoints"),
        "initialization": manifest["initialization"],
        "before": before, "train": result.metrics, "after": after,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "summary_only_rows": sum(training_target(r)[1] for r in train_rows),
        "runtime_context_rows": sum(r.get("runtime_context") is not None for r in train_rows),
        "annotation_methods": sorted({str(r.get("annotation_method", "unspecified")) for r in train_rows}),
        "validation_kind": "held_out_groups; label provenance retained; not proof of deployment accuracy",
    }
    (output_path / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--decision-field-order", choices=["summary-first", "actions-first"], default="summary-first")
    parser.add_argument("--generation-validation")
    parser.add_argument("--generation-eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--warm-start-adapter", help="Load trainable adapter weights with a new optimizer; not exact resume")
    args = parser.parse_args()
    config = ModelConfig(model_id=args.model, revision=args.revision, dataset_root=args.dataset_root,
                         quantization=args.quantization, max_input_tokens=args.max_input_tokens)
    policy = json.loads(Path(args.policy).read_text()) if args.policy else {}
    report = train_adapter(train_path=args.train, validation_path=args.validation, output=args.output,
                           config=config, policy=policy, max_steps=args.max_steps,
                           learning_rate=args.learning_rate, rank=args.rank,
                           gradient_accumulation_steps=args.gradient_accumulation_steps,
                           decision_field_order=args.decision_field_order,
                           generation_validation_path=args.generation_validation,
                           generation_eval_every=args.generation_eval_every,
                           checkpoint_every=args.checkpoint_every,
                           resume_from_checkpoint=args.resume_from_checkpoint,
                           warm_start_adapter=args.warm_start_adapter)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

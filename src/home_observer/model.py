"""Bounded native image/audio inference. GPU imports are lazy for local tooling."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import SYSTEM_PROMPT, build_context
from .schema import Decision, InferenceRequest, parse_decision

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    adapter_path: str | None = None
    dataset_root: str = "."
    quantization: str = "4bit"
    max_frames: int = 8
    max_audio_seconds: float = 30.0
    max_input_tokens: int = 8192
    max_new_tokens: int = 768
    max_soft_tokens: int = 140
    max_context_chars: int = 24000
    max_media_bytes: int = 32 * 1024 * 1024
    max_batch_size: int = 4
    merge_adapter: bool | None = False

    def __post_init__(self):
        if self.quantization not in {"4bit", "none"}:
            raise ValueError("quantization must be 4bit or none")
        if self.merge_adapter is True and self.quantization != "none":
            raise ValueError("adapter merging is supported only for unquantized BF16 weights")
        if self.max_soft_tokens not in {70, 140, 280, 560, 1120}:
            raise ValueError("unsupported Gemma4 image token budget")
        if not 0 < self.max_audio_seconds <= 30:
            raise ValueError("Gemma4 audio budget must be in (0, 30] seconds")
        if not 1 <= self.max_batch_size <= 8:
            raise ValueError("max_batch_size must be between 1 and 8")
        if min(self.max_frames, self.max_input_tokens, self.max_new_tokens) <= 0:
            raise ValueError("model limits must be positive")

    @property
    def should_merge_adapter(self) -> bool:
        return self.quantization == "none" and self.merge_adapter is True


def resolve_media_path(path: str, root: str | Path, max_bytes: int = 32 * 1024 * 1024) -> Path:
    """Local files only, including symlink resolution. Never fetch media URLs."""
    root = Path(root).expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("media path must be a file inside dataset_root")
    if candidate.stat().st_size > max_bytes:
        raise ValueError("media file exceeds size limit")
    return candidate


def prepare_request(request: dict, config: ModelConfig) -> dict:
    """Reject future observations and oversize windows; never silently discard evidence."""
    clean = InferenceRequest.model_validate(request).model_dump()
    window = clean["window"]
    if len(window["frames"]) > config.max_frames:
        raise ValueError(f"window exceeds max_frames={config.max_frames}; sample upstream")
    duration = sum(a["ended_at"] - a["started_at"] for a in window["audio"])
    if duration > config.max_audio_seconds:
        raise ValueError("total audio exceeds configured duration budget")
    for event in clean["recent_events"]:
        for key in ("timestamp", "ended_at", "observed_at"):
            if key in event and isinstance(event[key], (int, float)):
                if not math.isfinite(event[key]) or event[key] > window["ended_at"]:
                    raise ValueError("recent_events includes future or invalid timestamps")
    if len(build_context(clean)) > config.max_context_chars:
        raise ValueError("state, events, policy and metadata exceed bounded context budget")
    for item in [*window["frames"], *window["audio"]]:
        item["path"] = str(resolve_media_path(item["path"], config.dataset_root, config.max_media_bytes))
    return clean


def build_messages(request: dict, config: ModelConfig) -> list[dict]:
    """Images first, timestamp/evidence map next, audio last, per the Google model card."""
    clean = prepare_request(request, config)
    content = [{"type": "image", "path": f["path"]} for f in clean["window"]["frames"]]
    content.append({"type": "text", "text": build_context(clean)})
    content.extend({"type": "audio", "path": a["path"]} for a in clean["window"]["audio"])
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def validate_audio_files(request: dict, config: ModelConfig) -> None:
    """Validate actual duration too: request metadata cannot bypass the audio cap."""
    import soundfile as sf
    total = 0.0
    for item in request["window"].get("audio", []):
        path = resolve_media_path(item["path"], config.dataset_root, config.max_media_bytes)
        info = sf.info(path)
        actual = info.frames / info.samplerate
        declared = item["ended_at"] - item["started_at"]
        if abs(actual - declared) > 0.10:
            raise ValueError("audio file duration disagrees with source timestamps")
        total += actual
    if total > config.max_audio_seconds + 0.001:
        raise ValueError("audio file durations exceed configured budget")


def encode_messages(processor, messages: list[dict], config: ModelConfig, *, generation: bool):
    batch = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation,
        return_dict=True, return_tensors="pt", enable_thinking=False,
        processor_kwargs={"max_soft_tokens": config.max_soft_tokens, "padding": False},
    )
    count = batch["input_ids"].shape[-1]
    if count > config.max_input_tokens:
        raise ValueError(f"encoded input has {count} tokens, budget is {config.max_input_tokens}; shorten upstream")
    return batch



def encode_batch_messages(processor, conversations: list[list[dict]], config: ModelConfig):
    """Left-pad independent sessions; preserve actual lengths and modality masks."""
    if not 1 <= len(conversations) <= config.max_batch_size:
        raise ValueError(f"batch must contain 1..{config.max_batch_size} independent windows")
    old_padding = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        batch = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
            processor_kwargs={"max_soft_tokens": config.max_soft_tokens, "padding": True},
        )
    finally:
        processor.tokenizer.padding_side = old_padding
    lengths = batch["attention_mask"].sum(dim=-1).tolist()
    if len(lengths) != len(conversations):
        raise ValueError("processor returned an incorrect conversation batch size")
    if any(length > config.max_input_tokens for length in lengths):
        raise ValueError("an encoded batch window exceeds its input token budget")
    if batch["input_ids"].shape[-1] > config.max_input_tokens:
        raise ValueError("padded batch exceeds input token budget")
    return batch


def trim_generation_padding(token_ids: list[int], pad_token_id: int | None) -> list[int]:
    """Count each completion separately, excluding padding after its end token."""
    end = len(token_ids)
    if pad_token_id is not None:
        while end and token_ids[end - 1] == pad_token_id:
            end -= 1
    return token_ids[:end]


def load_components(config: ModelConfig, *, training: bool = False):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("NativeModel requires an NVIDIA CUDA GPU; run scripts/model_smoke.py on the pod")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Gemma4 audio training requires BF16-capable GPU (Ampere or newer)")
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision)
    kwargs: dict[str, Any] = {
        "revision": config.revision, "dtype": torch.bfloat16,
        "device_map": {"": torch.cuda.current_device()}, "attn_implementation": "sdpa",
    }
    if config.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["vision_tower", "audio_tower", "embed_vision", "embed_audio", "embed_tokens", "lm_head"],
        )
    model = AutoModelForMultimodalLM.from_pretrained(config.model_id, **kwargs)
    if config.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=training)
        if not training and config.should_merge_adapter:
            model = model.merge_and_unload(safe_merge=True)
    if not training:
        model.eval()
    return model, processor


def verify_decision_evidence(decision: Decision, request: dict) -> None:
    ids = InferenceRequest.model_validate(request).window.evidence_ids()
    for item in [*decision.observations, *decision.actions]:
        unknown = set(item.evidence_ids) - ids
        if unknown:
            raise ValueError("model cited evidence outside the current window: " + ", ".join(sorted(unknown)))


class NativeModel:
    """One model, fresh bounded context per request, explicit journal state supplied by caller."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model, self.processor = load_components(config)
        self._lock = threading.Lock()
        self.adapter_merged = bool(config.adapter_path and config.should_merge_adapter)

    def merge_adapter_weights(self) -> None:
        """Merge a loaded BF16 adapter under the inference lock; no quantized merge."""
        if self.config.quantization != "none":
            raise ValueError("adapter merging is supported only for unquantized BF16 weights")
        with self._lock:
            if not self.config.adapter_path or self.adapter_merged:
                return
            self.model = self.model.merge_and_unload(safe_merge=True).eval()
            self.adapter_merged = True
            self.config.merge_adapter = True

    def generate_text(self, request: dict) -> tuple[str, dict]:
        import torch
        with self._lock:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            messages = build_messages(request, self.config)
            validate_audio_files(request, self.config)
            inputs = encode_messages(self.processor, messages, self.config, generation=True)
            input_tokens = inputs["input_ids"].shape[-1]
            # Preserve integer IDs and masks; cast only floating modalities to model dtype.
            inputs = inputs.to(device=self.model.device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            preprocessed = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.config.max_new_tokens,
                    do_sample=False, use_cache=True,
                )
            torch.cuda.synchronize()
            output_ids = outputs[0, input_tokens:]
            text = self.processor.decode(output_ids, skip_special_tokens=True).strip()
            metrics = {
                "latency_s": time.perf_counter() - started,
                "preprocess_s": preprocessed - started,
                "input_tokens": input_tokens, "output_tokens": len(output_ids),
                "model_id": self.config.model_id, "adapter": self.config.adapter_path,
                "adapter_merged": self.adapter_merged,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "context_mode": "bounded_window",
            }
            return text, metrics

    def generate_batch(self, requests: list[dict]) -> list[tuple[str, dict]]:
        """Batched independent windows; every event waits for full batch completion.

        Initial supported layout has equal image and audio item counts per window.
        Token lengths may vary and are left-padded. No histories are shared across
        rows. This measures throughput; batch_time / batch_size is not event latency.
        """
        import torch
        if not 1 <= len(requests) <= self.config.max_batch_size:
            raise ValueError(f"batch must contain 1..{self.config.max_batch_size} windows")
        with self._lock:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            conversations = [build_messages(request, self.config) for request in requests]
            layouts = {(len(request["window"].get("frames", [])),
                        len(request["window"].get("audio", []))) for request in requests}
            if len(layouts) != 1:
                raise ValueError("batch windows currently require equal frame/audio item counts")
            for request in requests:
                validate_audio_files(request, self.config)
            inputs = encode_batch_messages(self.processor, conversations, self.config)
            padded_input_tokens = inputs["input_ids"].shape[-1]
            lengths = inputs["attention_mask"].sum(dim=-1).tolist()
            vision_counts = (inputs["input_ids"] == self.processor.image_token_id).sum(dim=-1).tolist()
            audio_id = getattr(self.processor, "audio_token_id", None)
            audio_counts = ((inputs["input_ids"] == audio_id).sum(dim=-1).tolist()
                            if audio_id is not None else [0] * len(requests))
            inputs = inputs.to(device=self.model.device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            preprocessed = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens,
                                              do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            generated = time.perf_counter()
            if len(outputs) != len(requests):
                raise RuntimeError("generation returned an incorrect batch size")
            decoded = []
            pad_id = self.processor.tokenizer.pad_token_id
            for index, output in enumerate(outputs):
                # HF generation returns the entire LEFT-PADDED input prefix.
                ids = trim_generation_padding(output[padded_input_tokens:].tolist(), pad_id)
                text = self.processor.decode(ids, skip_special_tokens=True).strip()
                decoded.append((text, ids))
            latency = time.perf_counter() - started
            shared = {"latency_s": latency, "batch_latency_s": latency,
                      "preprocess_s": preprocessed - started,
                      "batch_inference_s": generated - preprocessed,
                      "batch_size": len(requests), "windows_per_second": len(requests) / latency,
                      "amortized_seconds_per_window": latency / len(requests),
                      "event_latency_excludes_batch_formation_wait": True,
                      "padded_input_tokens": padded_input_tokens,
                      "model_id": self.config.model_id, "adapter": self.config.adapter_path,
                      "adapter_merged": self.adapter_merged,
                      "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                      "context_mode": "independent_bounded_windows"}
            return [(text, {**shared, "batch_index": i, "input_tokens": lengths[i],
                            "input_padding_tokens": padded_input_tokens - lengths[i],
                            "output_tokens": len(ids), "vision_tokens": vision_counts[i],
                            "audio_tokens": audio_counts[i]})
                    for i, (text, ids) in enumerate(decoded)]

    def observe_batch(self, requests: list[dict]) -> list[dict]:
        """Validate each result against its own evidence; reject any invalid item."""
        responses = []
        for index, ((text, metrics), request) in enumerate(zip(self.generate_batch(requests), requests)):
            try:
                decision = parse_decision(text)
                verify_decision_evidence(decision, request)
            except ValueError as exc:
                raise ValueError(f"batch item {index} did not return a valid grounded Decision: {exc}") from exc
            responses.append({"decision": decision.model_dump(), "metrics": metrics})
        return responses

    def observe(self, request: dict) -> dict:
        text, metrics = self.generate_text(request)
        try:
            decision = parse_decision(text)
            verify_decision_evidence(decision, request)
        except ValueError as exc:
            # Do not fabricate a no-op or repair action JSON after a malformed generation.
            raise ValueError(f"model did not return a valid grounded Decision: {exc}") from exc
        return {"decision": decision.model_dump(), "metrics": metrics}

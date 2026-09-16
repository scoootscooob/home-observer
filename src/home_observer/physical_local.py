"""Local Apple-silicon physical perception: pinned weights, unmerged adapter, no network.

This backend keeps every sensory byte on the host. It declares that explicitly
so the production guard can verify it. Two output contracts are supported:

* ``strict`` requires the complete ``PhysicalDecision`` from the model.
* ``judged-fields`` accepts the fields a partially supervised adapter can
  actually judge (event kind, object label, interval, description, optional
  evidence citations) and completes the remaining fields with explicit,
  recorded placeholders. It never invents an event, a label, an interval or a
  visual claim, and the completed decision still passes the causal validator.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError

from .model import ModelConfig, encode_messages
from .physical_model import (
    PHYSICAL_SYSTEM_PROMPT,
    build_physical_messages,
    prepare_physical_request,
    validate_physical_audio,
)
from .physical_schema import PhysicalDecision, PhysicalModel, PhysicalWindow, validate_physical_decision

PLACEHOLDER_CONFIDENCE = 0.5
START_SNAP_TOLERANCE_S = 0.05  # frame timestamps are index/fps; a claimed start may precede the first frame by ms
PLACEHOLDER_UNCERTAINTY = ("confidence and uncertainty were not supervised for this adapter; "
                           "this value is an explicit placeholder, not a model judgment")


class JudgedEvent(PhysicalModel):
    """Software-owned identifiers or unknown keys the model echoes are ignored and counted."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False, str_strip_whitespace=True)
    kind: str = Field(min_length=1, max_length=160)
    object_label: str | None = Field(default=None, max_length=160)
    started_at: float = Field(strict=True)
    ended_at: float = Field(strict=True)
    description: str = Field(default="", max_length=2000)
    subject_ids: list[str] = Field(default_factory=list, max_length=32)
    pre_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    post_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    confidence: float | None = Field(default=None, strict=True, ge=0, le=1)
    uncertainty: str | None = Field(default=None, max_length=1000)


class JudgedDecision(PhysicalModel):
    """Only fields a model may judge; software owns identifiers and placeholders."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False, str_strip_whitespace=True)
    summary: str | None = Field(default=None, max_length=3000)
    events: list[JudgedEvent] = Field(default_factory=list, max_length=64)


def extract_json_object(raw: str) -> tuple[str, int]:
    """First complete JSON object in the text; fences and trailing characters are format noise."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text[start:])
    body = text[start:start + end]
    trailing = text[start + end:].strip()
    return body, len(trailing)


def complete_judged_decision(raw: str, window: PhysicalWindow | dict) -> tuple[PhysicalDecision, dict]:
    """Deterministic completion of unsupervised fields, recorded field by field.

    Under this contract evidence citations are not a model judgment: the latest
    sampled frame at or before the claimed start and the latest frame inside the
    interval are cited, so the causal validator judges the claimed interval
    against real frames. Citations the model emitted are recorded, not used.
    Objects/observations are never added.
    """
    window = PhysicalWindow.model_validate(window)
    body, ignored_trailing = extract_json_object(raw)
    judged = JudgedDecision.model_validate_json(body)
    frames = sorted((frame for clip in window.clips for frame in clip.frames), key=lambda f: f.timestamp)
    completion = {"contract": "judged-fields", "summary": "model", "events": [],
                  "ignored_top_level_fields": sorted(judged.model_extra or {}),
                  "ignored_trailing_characters": ignored_trailing}
    events = []
    for index, item in enumerate(judged.events):
        record = {"index": index, "confidence": "model" if item.confidence is not None else "placeholder",
                  "uncertainty": "model" if item.uncertainty else "placeholder",
                  "evidence": "derived_from_interval",
                  "ignored_model_citations": len(item.pre_evidence_ids) + len(item.post_evidence_ids),
                  "ignored_event_fields": sorted(item.model_extra or {})}
        start = item.started_at
        before = [f.evidence_id for f in frames if f.timestamp <= start]
        if not before and frames and 0 < frames[0].timestamp - start <= START_SNAP_TOLERANCE_S:
            # The claim starts a few milliseconds before the first sampled frame: snap it onto
            # that frame rather than reject it; the shift is recorded, never hidden.
            record["start_snapped_s"] = round(frames[0].timestamp - start, 4)
            start = frames[0].timestamp
            before = [frames[0].evidence_id]
        inside = [f.evidence_id for f in frames if start < f.timestamp <= item.ended_at]
        pre, post = before[-1:], inside[-1:]
        events.append({
            "event_id": f"judged_{index + 1}", "kind": item.kind, "object_label": item.object_label,
            "subject_ids": item.subject_ids, "started_at": start, "ended_at": item.ended_at,
            "description": item.description or f"{item.kind} {item.object_label or ''}".strip(),
            "pre_evidence_ids": pre, "post_evidence_ids": post,
            "confidence": item.confidence if item.confidence is not None else PLACEHOLDER_CONFIDENCE,
            "uncertainty": item.uncertainty or PLACEHOLDER_UNCERTAINTY,
        })
        completion["events"].append(record)
    if judged.summary is None:
        completion["summary"] = "placeholder"
    summary = judged.summary if judged.summary is not None else "; ".join(e["description"] for e in events) or (
        "no event reported")
    return PhysicalDecision.model_validate({"summary": summary, "events": events}), completion


def load_local_components(config: ModelConfig, device: str = "mps"):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("local Metal inference requires an Apple-silicon MPS device")
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model_id, revision=config.revision, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="sdpa")
    if config.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=False)
    return model.eval(), processor


class PhysicalLocalModel:
    """Serialized local generation with strict post-validation; raw output is always retained."""

    local_inference = True
    sensory_media_leaves_host = False
    backend = "transformers-local"

    def __init__(self, config: ModelConfig | None = None, *, device: str = "mps", components=None,
                 contract: Literal["strict", "judged-fields"] = "strict"):
        self.config = config or ModelConfig(quantization="none", max_frames=8, max_input_tokens=8192,
                                            max_new_tokens=768, merge_adapter=False)
        if self.config.quantization != "none" or self.config.should_merge_adapter:
            raise ValueError("local physical inference uses unquantized BF16 weights with an unmerged adapter")
        if contract not in ("strict", "judged-fields"):
            raise ValueError("unknown output contract")
        self.device, self.contract = device, contract
        self.model, self.processor = components if components is not None else load_local_components(
            self.config, device)
        self._lock = threading.Lock()
        self.last_generation = None
        self.backend_config = {
            "backend": self.backend, "device": device, "contract": contract, "greedy": True,
            "adapter_merged": False, "dtype": "bfloat16", "structured_decoding": False,
            "system_prompt_sha256": hashlib.sha256(PHYSICAL_SYSTEM_PROMPT.encode()).hexdigest(),
            "local_inference": True, "sensory_media_leaves_host": False,
        }

    def _synchronize(self):
        import torch
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def generate_text(self, request: dict) -> tuple[str, dict]:
        import torch
        with self._lock:
            self.last_generation = None
            started = time.perf_counter()
            clean = prepare_physical_request(request, self.config)
            validate_physical_audio(clean, self.config)
            messages = build_physical_messages(clean, self.config)
            inputs = encode_messages(self.processor, messages, self.config, generation=True)
            input_tokens = int(inputs["input_ids"].shape[-1])
            inputs = inputs.to(device=self.device, dtype=torch.bfloat16)
            self._synchronize()
            preprocessed = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens,
                                             do_sample=False, use_cache=True)
            self._synchronize()
            ids = output[0, input_tokens:]
            text = self.processor.decode(ids, skip_special_tokens=True).strip()
            frames = [frame for clip in clean["window"]["clips"] for frame in clip["frames"]]
            metrics = {**self.backend_config, "task": "physical_perception",
                       "latency_s": time.perf_counter() - started, "preprocess_s": preprocessed - started,
                       "input_tokens": input_tokens, "output_tokens": int(len(ids)),
                       "sampled_frames": len(frames), "frame_evidence_ids": [f["evidence_id"] for f in frames],
                       "model_id": self.config.model_id, "revision": self.config.revision,
                       "adapter": self.config.adapter_path,
                       "finish_reason": "length" if len(ids) >= self.config.max_new_tokens else "stop"}
            self.last_generation = {"raw_output": text, "metrics": metrics}
            return text, metrics

    def observe(self, request: dict) -> dict:
        """Return the validated decision, or the validation error with the raw text retained."""
        clean = prepare_physical_request(request, self.config)
        raw, metrics = self.generate_text(clean)
        known = {item["track_id"] for item in clean["tracks"]}
        completion = None
        try:
            if self.contract == "strict":
                decision = PhysicalDecision.model_validate_json(raw)
            else:
                try:
                    decision = PhysicalDecision.model_validate_json(raw)
                    completion = {"contract": "judged-fields", "summary": "model", "events": [
                        {"index": i, "confidence": "model", "uncertainty": "model", "evidence": "model"}
                        for i in range(len(decision.events))]}
                except ValidationError:
                    decision, completion = complete_judged_decision(raw, clean["window"])
            decision = validate_physical_decision(decision, clean["window"], known_track_ids=known)
        except (ValidationError, ValueError) as exc:
            return {"decision": None, "metrics": {**metrics, "completion": completion},
                    "error": {"type": type(exc).__name__, "message": str(exc)[:2000]}, "raw_output": raw}
        metrics = {**metrics, "completion": completion}
        return {"decision": decision.model_dump(), "metrics": metrics, "raw_output": raw}


def local_model_manifest(model: PhysicalLocalModel) -> dict:
    """Provenance for reports: hashes of the adapter weights and the prompt, never media."""
    adapter = {}
    if model.config.adapter_path:
        from pathlib import Path
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = Path(model.config.adapter_path) / name
            if path.is_file():
                adapter[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**model.backend_config, "model_id": model.config.model_id, "revision": model.config.revision,
            "adapter_path": model.config.adapter_path, "adapter_sha256": adapter,
            "max_frames": model.config.max_frames, "max_new_tokens": model.config.max_new_tokens,
            "prompt_sha256": hashlib.sha256(PHYSICAL_SYSTEM_PROMPT.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(PhysicalDecision.model_json_schema(),
                                                       separators=(",", ":")).encode()).hexdigest()}

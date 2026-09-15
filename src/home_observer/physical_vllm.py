"""Bounded sensing-only client for an already running native multimodal vLLM engine.

This guided configuration is a separate experiment from native unconstrained
inference. The exact physical system/context messages are shared; the grammar
orders events first and bounds output lists, without an object/verb vocabulary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time

import httpx

from .model import ModelConfig
from .physical_model import (
    PHYSICAL_SYSTEM_PROMPT,
    physical_context,
    prepare_physical_request,
    validate_physical_audio,
)
from .physical_schema import PhysicalDecision, PhysicalWindow, validate_physical_decision
from .physical_serve import constrained_physical_schema
from .vllm_backend import _audio_part, _image_part

SCHEMA_CONTRACT = "physical_events_first_bounded_v3"
FIELD_ORDER = ("events", "objects", "observations", "summary")
MAX_ITEMS = 2


def bounded_physical_schema(window, known_track_ids=()) -> dict:
    """Constrain format and current evidence only; physical truth remains unproven.

    The cap is an explicit reporting budget, not an assertion that no additional
    things exist. All lists can be empty and labels/verbs remain free text.
    """
    window = PhysicalWindow.model_validate(window)
    known_track_ids = tuple(known_track_ids)
    schema = constrained_physical_schema(window, known_track_ids)
    schema["properties"] = {key: schema["properties"][key] for key in FIELD_ORDER}
    schema["required"] = list(FIELD_ORDER)
    for field in ("events", "objects", "observations"):
        prop = schema["properties"][field]
        prop["maxItems"] = min(MAX_ITEMS, prop.get("maxItems", MAX_ITEMS))
    # Explicit required nested fields prevent omission of uncertainty/confidence.
    # Nullable/defaulted fields remain representable; free-form dictionaries are
    # not turned into an ontology or used to force invented object identities.
    for name in ("TemporalEvent", "PhysicalObject", "PhysicalObservation"):
        definition = schema["$defs"][name]
        if name == "TemporalEvent":
            # Match the annotated prefix used by partial event supervision. The
            # complete output still requires evidence and uncertainty afterward.
            prefix = ("kind", "object_label", "started_at", "ended_at", "description")
            properties = definition["properties"]
            definition["properties"] = {key: properties[key] for key in
                (*prefix, *(key for key in properties if key not in prefix))}
        definition["required"] = list(definition["properties"])
    # Allocate only local output slots, not physical labels or persistent IDs.
    # A later validator still rejects a reference whose slot was not emitted.
    local_ids = [f"det_{index + 1}" for index in range(MAX_ITEMS)]
    definitions = schema["$defs"]
    definitions["PhysicalObject"]["properties"]["detection_id"]["enum"] = local_ids
    subject_choices = sorted({*local_ids, *known_track_ids})
    definitions["TemporalEvent"]["properties"]["subject_ids"]["items"] = {
        "type": "string", "enum": subject_choices}
    definitions["TemporalEvent"]["properties"]["subject_ids"]["maxItems"] = MAX_ITEMS
    definitions["PhysicalObservation"]["properties"]["subject_id"] = {
        "anyOf": [{"type": "string", "enum": subject_choices}, {"type": "null"}], "default": None}
    # Geometry belongs to one real sampled frame. Bind only observed metadata;
    # the model still chooses the frame, box, label and whether an object exists.
    branches = []
    for clip in window.clips:
        for frame in clip.frames:
            branch = copy.deepcopy(definitions["PhysicalObject"])
            properties = branch["properties"]
            properties["timestamp"] = {"type": "number", "const": frame.timestamp}
            properties["frame_evidence_id"] = {"type": "string", "const": frame.evidence_id}
            properties["evidence_ids"] = {"type": "array", "items": {"type": "string", "const": frame.evidence_id},
                                           "minItems": 1, "maxItems": 1}
            location = copy.deepcopy(definitions["PhysicalLocation"])
            location["properties"]["camera_id"] = {"type": "string", "const": clip.camera_id}
            properties["location"] = location
            branches.append(branch)
    if branches:
        definitions["PhysicalObject"] = {"title": "PhysicalObject", "anyOf": branches}
    return schema


class PhysicalVLLMModel:
    """One serialized physical request at a time; no retry or model fallback."""

    schema_contract = SCHEMA_CONTRACT

    def __init__(self, config: ModelConfig, *, base_url="http://127.0.0.1:8001/v1",
                 served_model="physical-observer", api_token=None, timeout=180.0,
                 schema_constraints=True, client: httpx.Client | None = None):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("physical engine URL must be HTTP(S) without credentials, query or fragment")
        if not isinstance(served_model, str) or not served_model.strip():
            raise ValueError("physical served model alias is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("physical engine timeout must be positive and finite")
        if config.should_merge_adapter:
            raise ValueError("physical serving requires an explicit unmerged/native adapter configuration")
        prefix = str(url).rstrip("/")
        self.base_url = prefix if prefix.endswith("/v1") else prefix + "/v1"
        self.health_url = self.base_url[:-3] + "/health"
        self.config, self.served_model = config, served_model
        self.timeout, self.schema_constraints = timeout, bool(schema_constraints)
        token = api_token if api_token is not None else os.environ.get("HOME_OBSERVER_ENGINE_TOKEN")
        self._headers = {"Authorization": "Bearer " + token} if token else {}
        self._owned_client = client is None
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False)
        self._lock = threading.Lock()
        self.last_generation = None
        self.backend_config = {
            "backend": "vllm", "served_model": served_model, "engine_url": self.base_url,
            "schema_constraints": self.schema_constraints,
            "structured_decoding": self.schema_constraints,
            "schema_contract": SCHEMA_CONTRACT if self.schema_constraints else None,
            "decision_field_order": list(FIELD_ORDER) if self.schema_constraints else None,
            "max_items_per_list": MAX_ITEMS if self.schema_constraints else None,
            "native_batch_size": 1, "greedy": True, "adapter_merged": False,
            "model_revision_source": "deployment_config_not_remote_attestation",
            "system_prompt_sha256": hashlib.sha256(PHYSICAL_SYSTEM_PROMPT.encode()).hexdigest(),
        }

    def close(self):
        if self._owned_client:
            self.client.close()

    def ready(self):
        try:
            result = self.client.get(self.health_url, headers=self._headers,
                                     timeout=min(self.timeout, 5), follow_redirects=False)
            return result.status_code == 200
        except httpx.HTTPError:
            return False

    def generate_text(self, request: dict) -> tuple[str, dict]:
        with self._lock:
            self.last_generation = None
            started = time.perf_counter()
            clean = prepare_physical_request(request, self.config)
            validate_physical_audio(clean, self.config)
            frames = [frame for clip in clean["window"]["clips"] for frame in clip["frames"]]
            context = physical_context(clean)
            content = [_image_part(frame["path"], self.config.max_media_bytes) for frame in frames]
            content.append({"type": "text", "text": context})
            content.extend(_audio_part(item["path"], self.config.max_media_bytes)
                           for item in clean["window"]["audio"])
            payload = {
                "model": self.served_model,
                "messages": [{"role": "system", "content": PHYSICAL_SYSTEM_PROMPT},
                             {"role": "user", "content": content}],
                "temperature": 0, "max_tokens": self.config.max_new_tokens, "n": 1, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": {"max_soft_tokens": self.config.max_soft_tokens},
            }
            schema = None
            if self.schema_constraints:
                schema = bounded_physical_schema(clean["window"], [t["track_id"] for t in clean["tracks"]])
                payload["structured_outputs"] = {"json": schema}
            # Bound the actual serialized upload including base64 expansion.
            wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            if len(wire) > 64 * 1024 * 1024:
                raise ValueError("physical engine upload exceeds 64 MiB request budget")
            preprocessed = time.perf_counter()
            response = self.client.post(self.base_url + "/chat/completions", content=wire,
                headers={**self._headers, "Content-Type": "application/json"},
                timeout=self.timeout, follow_redirects=False)
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("physical engine must return exactly one completion")
            choice = choices[0]
            raw = choice.get("message", {}).get("content")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("physical engine returned no text completion")
            metrics = {
                **self.backend_config, "task": "physical_perception",
                "latency_s": time.perf_counter() - started,
                "preprocess_s": preprocessed - started,
                "sampled_frames": len(frames),
                "frame_evidence_ids": [frame["evidence_id"] for frame in frames],
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "schema_sha256": hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
                    if schema is not None else None,
                "engine_reported_model": result.get("model"),
                "model_id": self.config.model_id, "revision": self.config.revision,
                "adapter": self.config.adapter_path, "finish_reason": choice.get("finish_reason"),
                "gpu_memory_available": False,
            }
            # Preserve raw output even when usage/alias/physical validation fails.
            self.last_generation = {"raw_output": raw, "metrics": metrics}
            usage = result.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("physical engine returned invalid token usage")
            metrics["token_usage"] = usage
            for wire_name, local_name in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                count = usage.get(wire_name)
                if count is not None and (type(count) is not int or count < 0):
                    raise ValueError("physical engine returned invalid token usage")
                metrics[local_name] = count
            if metrics["input_tokens"] is not None and metrics["input_tokens"] > self.config.max_input_tokens:
                raise ValueError("physical input token budget exceeded; shorten upstream")
            if result.get("model") != self.served_model:
                raise ValueError("physical engine returned a different model alias")
            return raw, metrics

    def observe(self, request: dict) -> dict:
        clean = prepare_physical_request(request, self.config)
        raw, metrics = self.generate_text(clean)
        decision = PhysicalDecision.model_validate_json(raw)
        if decision.model_fields_set != set(PhysicalDecision.model_fields):
            raise ValueError("physical engine omitted required top-level fields")
        if self.schema_constraints and any(len(getattr(decision, name)) > MAX_ITEMS
                                          for name in ("events", "objects", "observations")):
            raise ValueError("physical engine violated bounded reporting budget")
        if self.schema_constraints:
            local_ids = {f"det_{index + 1}" for index in range(MAX_ITEMS)}
            if any(obj.detection_id not in local_ids for obj in decision.objects):
                raise ValueError("physical engine violated local detection slot contract")
            if any(len(event.subject_ids) > MAX_ITEMS for event in decision.events):
                raise ValueError("physical engine violated bounded subject reference contract")
        decision = validate_physical_decision(decision, clean["window"],
                                              known_track_ids={t["track_id"] for t in clean["tracks"]})
        return {"decision": decision.model_dump(), "metrics": metrics, "raw_output": raw}

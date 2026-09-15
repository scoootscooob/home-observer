"""Single-engine Gemma4 image/audio client; no GPU or Transformers imports.

The engine must serve the configured base or registered native LoRA model.
Its model revision is supplied deployment metadata, not remotely attested by this
API. Completion usage is available; GPU memory is not available from this API.
"""
from __future__ import annotations

import base64
import copy
import os
import time
from pathlib import Path

import httpx

from .model import ModelConfig, prepare_request, validate_audio_files, verify_decision_evidence
from .prompts import SYSTEM_PROMPT, build_context
from .schema import Decision, InferenceRequest, parse_decision


def _entities(request: dict) -> tuple[set[str], set[str]]:
    configured = request.get("policy", {}).get("entities", [])
    if not isinstance(configured, list) or any(not isinstance(x, str) or not x for x in configured):
        raise ValueError("policy.entities must be a list of nonempty entity IDs")
    actions = set(configured)
    observations = actions | set(request["window"].get("device_states", {}))
    return observations, actions


def decision_schema(request: dict, *, allow_brightness: bool = False,
                    decision_field_order: str = 'summary-first') -> dict:
    """Constrain syntax/current IDs, never infer policy satisfaction or labels.

    Empty evidence/entity surfaces permit only empty corresponding arrays. Avoid
    empty enums, which some grammar compilers reject even in unreachable items.
    Cross-field noop consistency is constrained and checked again by Pydantic.
    """
    parsed = InferenceRequest.model_validate(request)
    evidence = sorted(parsed.window.evidence_ids())
    observation_entities, action_entities = _entities(request)
    schema = Decision.model_json_schema()
    if decision_field_order == 'actions-first':
        schema['properties'] = {name: schema['properties'][name]
                                for name in ('actions', 'observations', 'noop', 'summary')}
    elif decision_field_order != 'summary-first':
        raise ValueError('unsupported decision field order')
    schema["required"] = list(schema["properties"])
    action_properties = schema['$defs']['Action']['properties']
    # Match the existing executor contract; never generate unsupported HA
    # arguments such as brightness_pct and then silently rewrite them.
    action_properties['data'] = {
        'type': 'object', 'properties': ({
            'brightness': {'type': 'integer', 'minimum': 1, 'maximum': 255}} if allow_brightness else {}),
        'additionalProperties': False,
    }
    services = request.get('policy', {}).get('allowed_services')
    if services is not None:
        if not isinstance(services, list) or any(not isinstance(s, str) or s.count('.') != 1 for s in services):
            raise ValueError('policy.allowed_services must contain domain.service names')
        if services:
            domains, names = zip(*(service.split('.') for service in services))
            action_properties['domain']['enum'] = sorted(set(domains))
            action_properties['service']['enum'] = sorted(set(names))
            action_entities = {entity for entity in action_entities if entity.split('.')[0] in domains}
        else:
            action_entities = set()
    maximum = request.get('policy', {}).get('max_actions_per_window', 8)
    if type(maximum) is not int or maximum < 0:
        raise ValueError('policy.max_actions_per_window must be a nonnegative integer')
    schema['properties']['actions']['maxItems'] = min(8, maximum)
    for name, field, entities in (("Observation", "observations", observation_entities),
                                  ("Action", "actions", action_entities)):
        definition = schema["$defs"][name]
        definition["required"] = list(definition["properties"])
        if evidence:
            definition["properties"]["evidence_ids"]["items"]["enum"] = evidence
        if entities:
            definition["properties"]["entity_id"]["enum"] = sorted(entities)
        if not evidence or not entities:
            schema["properties"][field]["maxItems"] = 0
    # Pydantic's cross-field validator is not emitted by model_json_schema.
    # Represent its exact rule in the decoding grammar as two full branches.
    active = copy.deepcopy({key: value for key, value in schema.items() if key != '$defs'})
    active['properties']['noop'] = {'const': False, 'type': 'boolean'}
    quiet = copy.deepcopy(active)
    quiet['properties']['noop'] = {'const': True, 'type': 'boolean'}
    quiet['properties']['observations']['maxItems'] = 0
    quiet['properties']['actions']['maxItems'] = 0
    schema['anyOf'] = [active, quiet]
    return schema


def _image_part(path: str, max_bytes: int) -> dict:
    from PIL import Image

    with Image.open(path) as source:
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(source.format)
        if mime is None:
            raise ValueError("vLLM frames must be JPEG, PNG, or WebP images")
        source.verify()
    payload = Path(path).read_bytes()
    if len(payload) > max_bytes:
        raise ValueError("media file exceeds size limit")
    encoded = base64.b64encode(payload).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _audio_part(path: str, max_bytes: int) -> dict:
    import soundfile as sf

    info = sf.info(path)
    if info.format not in {"WAV", "WAVEX"} or info.samplerate != 16000 or info.channels != 1:
        raise ValueError("vLLM audio must be native 16 kHz mono WAV; convert upstream")
    payload = Path(path).read_bytes()
    if len(payload) > max_bytes:
        raise ValueError("media file exceeds size limit")
    return {"type": "input_audio", "input_audio": {
        "data": base64.b64encode(payload).decode("ascii"), "format": "wav"}}


class VLLMModel:
    """NativeModel-compatible client for one already-running Gemma4 vLLM engine.

    ``schema_constraints=False`` changes decoding only; validation never turns
    off. A supplied client remains caller-owned. No automatic retries or fallback
    model calls occur. Exact input tokens are checked from returned usage because
    local pre-tokenization would require another model processor or API request.
    """

    schema_contract = 'executor_arguments_v3'

    def __init__(self, config: ModelConfig, *, base_url: str = "http://127.0.0.1:8001/v1",
                 served_model: str = "home-observer", schema_constraints: bool = True,
                 timeout: float = 180.0, client: httpx.Client | None = None,
                 validate_audio: bool = True, allow_brightness: bool = False,
                 decision_field_order: str = 'summary-first'):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("engine base_url must be an HTTP(S) URL without credentials, query, or fragment")
        if timeout <= 0 or not served_model:
            raise ValueError("engine timeout and served_model must be set")
        prefix = str(url).rstrip("/")
        self.base_url = prefix if prefix.endswith("/v1") else prefix + "/v1"
        self.health_url = self.base_url[:-3] + "/health"
        self.config = config
        self.served_model = served_model
        self.schema_constraints = schema_constraints
        self.allow_brightness = allow_brightness
        if decision_field_order not in {'summary-first', 'actions-first'}:
            raise ValueError('unsupported decision field order')
        self.decision_field_order = decision_field_order
        self.validate_audio = validate_audio
        self.timeout = timeout
        self._owned_client = client is None
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False)
        token = os.environ.get("HOME_OBSERVER_ENGINE_TOKEN")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    def close(self) -> None:
        if self._owned_client:
            self.client.close()

    def ready(self) -> bool:
        try:
            response = self.client.get(self.health_url, headers=self._headers,
                                       timeout=min(self.timeout, 5.0), follow_redirects=False)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def generate_text(self, request: dict) -> tuple[str, dict]:
        started = time.perf_counter()
        clean = prepare_request(request, self.config)
        _entities(clean)
        if self.validate_audio:
            validate_audio_files(clean, self.config)
        content = [_image_part(frame["path"], self.config.max_media_bytes)
                   for frame in clean["window"]["frames"]]
        content.append({"type": "text", "text": build_context(clean)})
        content.extend(_audio_part(audio["path"], self.config.max_media_bytes)
                       for audio in clean["window"]["audio"])
        payload = {
            "model": self.served_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
            "temperature": 0, "max_tokens": self.config.max_new_tokens, "n": 1, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"max_soft_tokens": self.config.max_soft_tokens},
        }
        if self.schema_constraints:
            payload["structured_outputs"] = {"json": decision_schema(
                clean, allow_brightness=self.allow_brightness, decision_field_order=self.decision_field_order)}
        preprocessed = time.perf_counter()
        response = self.client.post(self.base_url + "/chat/completions", json=payload,
                                    headers=self._headers, timeout=self.timeout, follow_redirects=False)
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("engine must return exactly one completion")
        choice = choices[0]
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("engine returned no text completion")
        usage = result.get("usage") or {}
        counts = {}
        for wire, local in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
            value = usage.get(wire)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("engine returned invalid token usage")
            counts[local] = value
        if counts["input_tokens"] is not None and counts["input_tokens"] > self.config.max_input_tokens:
            raise ValueError("engine prompt usage exceeds configured input token budget; shorten upstream")
        metrics = {
            "latency_s": time.perf_counter() - started,
            "preprocess_s": preprocessed - started,
            **counts, "token_usage": usage,
            "backend": "vllm", "served_model": self.served_model,
            "engine_reported_model": result.get("model"),
            "model_id": self.config.model_id, "model_revision": self.config.revision,
            "model_revision_source": "deployment_config",
            "adapter": self.config.adapter_path, "structured_decode": self.schema_constraints,
            "schema_contract": self.schema_contract if self.schema_constraints else None,
            "brightness_enabled": self.allow_brightness,
            "decision_field_order": self.decision_field_order,
            "finish_reason": choice.get("finish_reason"), "context_mode": "bounded_window",
            "gpu_memory_available": False,
            "gpu_memory_unavailable_reason": "chat completions API does not report GPU memory",
        }
        return text, metrics

    def observe(self, request: dict) -> dict:
        text, metrics = self.generate_text(request)
        try:
            decision = parse_decision(text)
            if decision.model_fields_set != set(Decision.model_fields):
                raise ValueError("model omitted required Decision fields")
            verify_decision_evidence(decision, request)
            observation_entities, action_entities = _entities(request)
            for items, allowed in ((decision.observations, observation_entities),
                                   (decision.actions, action_entities)):
                unknown = {item.entity_id for item in items} - allowed
                if unknown:
                    raise ValueError("model used unknown entity IDs: " + ", ".join(sorted(unknown)))
            if any("data" not in action.model_fields_set for action in decision.actions):
                raise ValueError("model omitted required action data field")
        except ValueError as exc:
            raise ValueError(f"model did not return a valid grounded Decision: {exc}") from exc
        return {"decision": decision.model_dump(), "metrics": metrics}

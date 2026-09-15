"""Sequential observer loop; every action has evidence and a durable execution record."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol

import httpx

from .journal import Journal
from .policy import Policy, action_is_satisfied
from .schema import InferenceRequest, InferenceResponse, ObservationWindow


class Backend(Protocol):
    def observe(self, request: dict) -> dict: ...


class RemoteBackend:
    def __init__(self, url: str, token: str, timeout: float = 120, *, upload_media: bool = False,
                 media_root: str | None = None):
        self.upload_media = upload_media
        self.media_config = None
        if media_root is not None:
            from .model import ModelConfig
            self.media_config = ModelConfig(dataset_root=media_root)
        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout,
                                  headers={"Authorization": f"Bearer {token}"} if token else {})

    def close(self):
        self.client.close()

    def observe(self, request: dict) -> dict:
        started = time.perf_counter()
        if self.upload_media:
            if self.media_config is not None:
                from .model import prepare_request
                request = prepare_request(request, self.media_config)
            from .transport import pack_media
            payload, size = pack_media(request)
            response = self.client.post("/observe_upload", json=payload)
        else:
            size = 0
            response = self.client.post("/observe", json=request)
        if response.is_error:
            # Keep the server's parse/evidence diagnosis, without auth headers.
            # A bare HTTPStatusError hides why a generated decision was rejected.
            raise ValueError(f"inference HTTP {response.status_code}: {response.text[:1800]}")
        result = response.json()
        result.setdefault("metrics", {}).update(remote_roundtrip_s=time.perf_counter() - started,
                                                 uploaded_media_bytes=size)
        return result


class Observer:
    def __init__(self, backend: Backend, journal: Journal, policy: Policy, executor):
        self.backend, self.journal, self.policy, self.executor = backend, journal, policy, executor
        self.last_timestamp = journal.latest_timestamp()

    def process(self, raw_window: dict) -> dict:
        window = ObservationWindow.model_validate(raw_window)
        if not self.journal.begin(window.model_dump()):
            return {"window_id": window.window_id, "skipped": "already recorded", "existing": self.journal.get_window(window.window_id)}
        started = time.perf_counter()
        result = {"window_id": window.window_id, "started_at": window.started_at,
                  "ended_at": window.ended_at, "actions": [], "rejections": []}
        try:
            if window.ended_at < self.last_timestamp:
                raise ValueError("out-of-order window; start a new journal for an independent replay")
            self.last_timestamp = window.ended_at
            # Prior memory must precede this window. Current telemetry belongs in
            # window.device_states; copying it into the past obscures changes.
            prior_state = self.journal.state(window.ended_at, self.policy.state_ttl_seconds)
            prior_events = self.journal.recent(window.ended_at)
            self.executor.ingest(window.device_states)
            for entity, value in window.device_states.items():
                state = value.get("state") if isinstance(value, dict) else value
                self.journal.update_fact(entity, "state", state, 1.0, window.ended_at,
                                         [f"device:{entity}"], "device")
            request = InferenceRequest(
                window=window, state=prior_state, recent_events=prior_events, policy=self.policy.as_dict(),
            ).model_dump()
            result["request_sha256"] = hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            result["prior_context"] = {"state": prior_state, "recent_events": prior_events}
            response = InferenceResponse.model_validate(self.backend.observe(request))
            decision = response.decision
            errors = self.policy.validate(decision, window)
            result.update({"decision": decision.model_dump(), "metrics": response.metrics, "rejections": errors})
            if errors:
                self.journal.event(window.window_id, window.ended_at, "rejected", {"errors": errors})
            else:
                for obs in decision.observations:
                    if obs.confidence < self.policy.min_confidence:
                        continue
                    # Device telemetry is authoritative for the device's state at this instant.
                    if obs.attribute == "state" and obs.entity_id in window.device_states:
                        continue
                    self.journal.update_fact(obs.entity_id, obs.attribute, obs.value, obs.confidence,
                                             window.ended_at, obs.evidence_ids)
                self.journal.event(window.window_id, window.ended_at, "decision", decision.model_dump())
                for action in decision.actions:
                    signature = json.dumps([action.domain, action.service, action.entity_id, action.data], sort_keys=True)
                    action_id = hashlib.sha256((window.window_id + signature).encode()).hexdigest()[:24]
                    if action_is_satisfied(action, getattr(self.executor, "states", window.device_states)):
                        result["actions"].append({"action_id": action_id, "status": "already_satisfied"})
                        continue
                    if not self.journal.reserve_action(action_id, window.window_id, window.ended_at,
                                                       signature, action.model_dump(), self.policy.cooldown_seconds):
                        result["actions"].append({"action_id": action_id, "status": "duplicate_or_cooldown"})
                        continue
                    try:
                        executed = self.executor.execute(action, action_id)
                        status = "complete" if executed.get("verified", True) else "uncertain"
                    except Exception as exc:
                        # A timeout can occur after the device acted. Never resend automatically.
                        status, executed = "uncertain", {"error_type": type(exc).__name__}
                    self.journal.complete_action(action_id, status, executed)
                    self.journal.event(window.window_id, window.ended_at, "action", {"status": status, **executed})
                    if status == "complete" and "state" in executed:
                        self.journal.update_fact(action.entity_id, "state", executed["state"], 1,
                                                 window.ended_at, action.evidence_ids, "action_verified")
                    result["actions"].append({"action_id": action_id, "status": status, "result": executed})
            result["total_latency_s"] = time.perf_counter() - started
            self.journal.finish(window.window_id, result)
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)[:400]
            result["total_latency_s"] = time.perf_counter() - started
            self.journal.finish(window.window_id, result, result["error"])
        return result


class FixtureBackend:
    """Device-only authored rules for integration tests. This is explicitly NOT learned inference."""
    def observe(self, request: dict) -> dict:
        states = request["window"].get("device_states", {})
        def value(entity):
            item = states.get(entity)
            return item.get("state") if isinstance(item, dict) else item
        actions = []
        try:
            dark = float(value("sensor.ambient_lux")) < 40
        except (TypeError, ValueError):
            dark = False
        for room in ["kitchen", "entry", "lounge", "garage"]:
            sensor, light = f"binary_sensor.motion_{room}", f"light.{room}"
            if dark and value(sensor) == "on" and value(light) == "off":
                actions.append({"domain": "light", "service": "turn_on", "entity_id": light,
                                "data": {}, "reason": "Configured motion and darkness rule",
                                "evidence_ids": [f"device:{sensor}", "device:sensor.ambient_lux", f"device:{light}"]})
        if value("binary_sensor.water_leak") == "on" and value("switch.buzzer") == "off":
            actions.append({"domain": "switch", "service": "turn_on", "entity_id": "switch.buzzer",
                            "data": {}, "reason": "Configured leak rule",
                            "evidence_ids": ["device:binary_sensor.water_leak", "device:switch.buzzer"]})
        # Deliberately no visual/audio conclusions: this backend does not inspect media.
        return {"decision": {"summary": "Deterministic device-rule fixture", "observations": [],
                              "actions": actions[:request.get("policy", {}).get("max_actions_per_window", 2)],
                              "noop": not actions}, "metrics": {"backend": "fixture", "learned": False}}

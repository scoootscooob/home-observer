"""Simulated devices and an explicitly configured Home Assistant service adapter."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from .policy import action_is_satisfied
from .schema import Action


class SimulatedHomeAssistant:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.states = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}
        self.calls: list[dict] = []

    def ingest(self, states: dict):
        self.states.update(states)
        self._save()

    def execute(self, action: Action, action_id: str) -> dict:
        if action.service not in {"turn_on", "turn_off"} or action.domain not in {"light", "switch"}:
            raise ValueError("unsupported simulated service")
        new_state = "on" if action.service == "turn_on" else "off"
        attributes = {}
        previous = self.states.get(action.entity_id)
        if isinstance(previous, dict):
            attributes.update(previous.get("attributes", {}))
        if "brightness" in action.data:
            attributes["brightness"] = action.data["brightness"]
        self.states[action.entity_id] = {"state": new_state, "attributes": attributes} if attributes else new_state
        call = {"action_id": action_id, "entity_id": action.entity_id, "state": new_state,
                "domain": action.domain, "service": action.service, "simulated": True,
                "attributes": attributes, "verified": action_is_satisfied(action, self.states)}
        self.calls.append(call)
        self._save()
        return call

    def _save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.states, indent=2))
            temp.replace(self.path)


class HomeAssistant:
    def __init__(self, base_url: str, token: str, *, execute_real: bool = False,
                 verification_timeout_s: float = 3.0):
        if not execute_real:
            raise ValueError("real Home Assistant execution must be explicitly enabled")
        if not token:
            raise ValueError("Home Assistant token is missing")
        if not 0 <= verification_timeout_s <= 30:
            raise ValueError("Home Assistant verification timeout must be between 0 and 30 seconds")
        self.verification_timeout_s = verification_timeout_s
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=15,
                                   headers={"Authorization": f"Bearer {token}"})

    def ingest(self, states: dict):
        pass

    def read_states(self, entities: list[str]) -> dict:
        """Read current configured devices; missing entities remain unknown."""
        response = self.client.get("/api/states")
        response.raise_for_status()
        allowed = set(entities)
        states = {}
        for row in response.json():
            entity = row.get("entity_id")
            if entity in allowed:
                attributes = {key: value for key, value in row.get("attributes", {}).items()
                              if key in {"brightness", "unit_of_measurement"}}
                states[entity] = {"state": row.get("state"), "attributes": attributes}
        return states

    def execute(self, action: Action, action_id: str) -> dict:
        response = self.client.post(f"/api/services/{action.domain}/{action.service}",
                                    json={**action.data, "entity_id": action.entity_id})
        response.raise_for_status()
        started = time.monotonic()
        deadline = started + self.verification_timeout_s
        # HA service completion can precede an integration's state-changed event.
        # Poll only the readback; never repeat a possibly successful device command.
        while True:
            state_response = self.client.get(f"/api/states/{action.entity_id}")
            state_response.raise_for_status()
            state_json = state_response.json()
            state = state_json.get("state")
            verified = action_is_satisfied(action, {action.entity_id: state_json})
            if verified or time.monotonic() >= deadline:
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        return {"action_id": action_id, "entity_id": action.entity_id, "state": state,
                "simulated": False, "verified": verified, "attributes": state_json.get("attributes", {}),
                "verification_latency_s": time.monotonic() - started}

    def close(self):
        self.client.close()

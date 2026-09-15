"""Validate proposed actions against the configured device surface and evidence."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Action, Decision, ObservationWindow


@dataclass
class Policy:
    entities: list[str]
    rules: list[str] = field(default_factory=list)
    allowed_services: list[str] = field(default_factory=lambda: ["light.turn_on", "light.turn_off", "switch.turn_on", "switch.turn_off"])
    min_confidence: float = 0.8
    cooldown_seconds: float = 30
    state_ttl_seconds: float = 300
    max_actions_per_window: int = 2
    max_window_seconds: float = 60

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def validate(self, decision: Decision, window: ObservationWindow) -> list[str]:
        errors = []
        evidence = window.evidence_ids()
        allowed_entities = set(self.entities) | set(window.device_states)
        if window.ended_at - window.started_at > self.max_window_seconds:
            errors.append("window exceeds maximum permitted duration")
        for obs in decision.observations:
            if obs.entity_id not in allowed_entities:
                errors.append(f"unknown observation entity: {obs.entity_id}")
            if not set(obs.evidence_ids) <= evidence:
                errors.append("observation cites unknown evidence")
        if len(decision.actions) > self.max_actions_per_window:
            errors.append("too many actions in one window")
        for action in decision.actions:
            if f"{action.domain}.{action.service}" not in self.allowed_services:
                errors.append(f"service not allowed: {action.domain}.{action.service}")
            if action.entity_id not in self.entities:
                errors.append(f"action entity not configured: {action.entity_id}")
            if not action.entity_id.startswith(action.domain + "."):
                errors.append("action domain does not match entity")
            if not set(action.evidence_ids) <= evidence:
                errors.append("action cites unknown evidence")
            if action.data:
                if action.domain != "light" or action.service != "turn_on" or set(action.data) != {"brightness"}:
                    errors.append("unsupported action arguments")
                elif type(action.data["brightness"]) is not int or not 1 <= action.data["brightness"] <= 255:
                    errors.append("invalid brightness")
        return errors


def default_policy() -> Policy:
    rooms = ["kitchen", "entry", "lounge", "garage"]
    return Policy(
        entities=[*[f"light.{r}" for r in rooms], *[f"room.{r}" for r in rooms],
                  *[f"binary_sensor.motion_{r}" for r in rooms], "sensor.ambient_lux",
                  "binary_sensor.water_leak", "switch.buzzer"],
        rules=[
            "If current evidence shows occupancy or motion in a room, ambient lux is below 40, and that room's light is off, turn its light on.",
            "If binary_sensor.water_leak is on and switch.buzzer is off, turn switch.buzzer on.",
            "Otherwise propose no device action. Never turn lights off just because occupancy is unknown.",
        ],
    )


def action_is_satisfied(action: Action, states: dict) -> bool:
    state = states.get(action.entity_id)
    attributes = {}
    if isinstance(state, dict):
        attributes = state.get("attributes", {})
        state = state.get("state")
    if action.service == "turn_on" and "brightness" in action.data:
        return state == "on" and attributes.get("brightness") == action.data["brightness"]
    return (action.service == "turn_on" and state == "on") or (action.service == "turn_off" and state == "off")

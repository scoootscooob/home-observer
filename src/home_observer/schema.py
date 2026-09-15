"""Shared wire schemas. Timestamps are source-time Unix seconds, never wall-clock guesses."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Frame(StrictModel):
    camera_id: str
    timestamp: float
    path: str
    evidence_id: str


class AudioChunk(StrictModel):
    microphone_id: str
    started_at: float
    ended_at: float
    path: str
    evidence_id: str


class ObservationWindow(StrictModel):
    window_id: str
    started_at: float
    ended_at: float
    frames: list[Frame] = Field(default_factory=list, max_length=64)
    audio: list[AudioChunk] = Field(default_factory=list, max_length=16)
    device_states: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_times_and_ids(self):
        if self.ended_at < self.started_at:
            raise ValueError("window ends before it starts")
        ids = []
        for frame in self.frames:
            if not self.started_at <= frame.timestamp <= self.ended_at:
                raise ValueError("frame lies outside the observation window")
            ids.append(frame.evidence_id)
        for audio in self.audio:
            if not self.started_at <= audio.started_at <= audio.ended_at <= self.ended_at:
                raise ValueError("audio lies outside the observation window")
            ids.append(audio.evidence_id)
        if len(ids) != len(set(ids)) or any(x.startswith("device:") for x in ids):
            raise ValueError("media evidence IDs must be unique and cannot use device: prefix")
        return self

    def evidence_ids(self) -> set[str]:
        return {x.evidence_id for x in [*self.frames, *self.audio]} | {
            f"device:{entity}" for entity in self.device_states
        }


class Observation(StrictModel):
    entity_id: str
    attribute: str
    value: Any
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)


class Action(StrictModel):
    domain: str
    service: str
    entity_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    reason: str
    evidence_ids: list[str] = Field(min_length=1)


class Decision(StrictModel):
    summary: str = Field(max_length=3000)
    observations: list[Observation] = Field(default_factory=list, max_length=64)
    actions: list[Action] = Field(default_factory=list, max_length=8)
    noop: bool

    @model_validator(mode="after")
    def consistent_noop(self):
        if self.noop and (self.observations or self.actions):
            raise ValueError("noop cannot include observations or actions")
        return self


class InferenceRequest(StrictModel):
    window: ObservationWindow
    state: dict[str, Any] = Field(default_factory=dict)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(StrictModel):
    decision: Decision
    metrics: dict[str, Any] = Field(default_factory=dict)


def parse_decision(text: str) -> Decision:
    """Permit one fenced JSON object, never repair/invent a model's action."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return Decision.model_validate_json(text)

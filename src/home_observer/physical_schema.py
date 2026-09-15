"""Perception-only temporal evidence and physical objects, independent of HA IDs.

Validation establishes reference/time/geometry consistency, not visual correctness.
The capture/backend layer remains responsible for trusted media paths and decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=160)]
TrackID = Annotated[str, Field(pattern=r"^phys_[A-Za-z0-9_-]{1,96}$")]
Confidence = Annotated[float, Field(strict=True, ge=0, le=1)]
Timestamp = Annotated[float, Field(strict=True)]


class PhysicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class NormalizedBox(PhysicalModel):
    x_min: float = Field(strict=True, ge=0, le=1)
    y_min: float = Field(strict=True, ge=0, le=1)
    x_max: float = Field(strict=True, ge=0, le=1)
    y_max: float = Field(strict=True, ge=0, le=1)

    @model_validator(mode="after")
    def nonempty(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("box must have positive width and height")
        return self


class ClipFrame(PhysicalModel):
    evidence_id: Identifier
    timestamp: Timestamp
    path: str = Field(min_length=1, max_length=4096)


class TemporalClip(PhysicalModel):
    clip_id: Identifier
    camera_id: Identifier
    started_at: Timestamp
    ended_at: Timestamp
    frames: list[ClipFrame] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def ordered_frames(self):
        if self.ended_at < self.started_at:
            raise ValueError("clip ends before it starts")
        times = [frame.timestamp for frame in self.frames]
        if any(not self.started_at <= timestamp <= self.ended_at for timestamp in times):
            raise ValueError("frame is outside its clip")
        if any(left >= right for left, right in zip(times, times[1:])):
            raise ValueError("clip frames must be strictly chronological")
        return self


class TemporalAudio(PhysicalModel):
    evidence_id: Identifier
    microphone_id: Identifier
    started_at: Timestamp
    ended_at: Timestamp
    path: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def ordered_time(self):
        if self.ended_at <= self.started_at:
            raise ValueError("audio duration must be positive")
        return self


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    kind: str
    source_id: str
    started_at: float
    ended_at: float
    path: str | None = None


class PhysicalWindow(PhysicalModel):
    window_id: Identifier
    started_at: Timestamp
    ended_at: Timestamp
    clips: list[TemporalClip] = Field(default_factory=list, max_length=16)
    audio: list[TemporalAudio] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def bounded_evidence(self):
        if not 0 <= self.ended_at - self.started_at <= 300:
            raise ValueError("physical window duration must be between 0 and 300 seconds")
        ids = []
        frames = 0
        for item in [*self.clips, *self.audio]:
            if not self.started_at <= item.started_at <= item.ended_at <= self.ended_at:
                raise ValueError("clip/audio is outside the physical window")
            if isinstance(item, TemporalClip):
                ids.extend([item.clip_id, *[frame.evidence_id for frame in item.frames]])
                frames += len(item.frames)
            else:
                ids.append(item.evidence_id)
        if frames > 512:
            raise ValueError("physical window exceeds 512 sampled frames")
        if len(ids) != len(set(ids)):
            raise ValueError("physical evidence IDs, including clip IDs, must be unique within a window")
        return self

    def evidence(self) -> dict[str, EvidenceReference]:
        result = {}
        for clip in self.clips:
            # Sampled coverage comes from actual frames, not an inflated clip interval.
            result[clip.clip_id] = EvidenceReference(clip.clip_id, "clip", clip.camera_id,
                                                    clip.frames[0].timestamp, clip.frames[-1].timestamp)
            for frame in clip.frames:
                result[frame.evidence_id] = EvidenceReference(frame.evidence_id, "frame", clip.camera_id,
                                                             frame.timestamp, frame.timestamp, frame.path)
        for chunk in self.audio:
            result[chunk.evidence_id] = EvidenceReference(chunk.evidence_id, "audio", chunk.microphone_id,
                                                         chunk.started_at, chunk.ended_at, chunk.path)
        return result

    def evidence_ids(self) -> set[str]:
        return set(self.evidence())


class PhysicalLocation(PhysicalModel):
    camera_id: Identifier
    area: str | None = Field(default=None, max_length=160)


class GroundedPhysicalItem(PhysicalModel):
    confidence: Confidence
    uncertainty: str = Field(min_length=1, max_length=1000)


class PhysicalObject(GroundedPhysicalItem):
    detection_id: Identifier
    track_id: TrackID | None = None
    label: str = Field(min_length=1, max_length=160)
    box: NormalizedBox
    location: PhysicalLocation
    timestamp: Timestamp
    frame_evidence_id: Identifier
    visibility: Literal["visible", "partially_occluded", "uncertain"]
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=64)
    attributes: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def local_identifier(self):
        if self.detection_id.startswith("phys_"):
            raise ValueError("phys_ is reserved for persistent track IDs; use a local detection ID")
        return self


class PhysicalObservation(GroundedPhysicalItem):
    observation_id: Identifier
    subject_id: Identifier | None = None
    attribute: str = Field(min_length=1, max_length=160)
    value: JsonValue
    observed_at: Timestamp
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=64)


class TemporalEvent(GroundedPhysicalItem):
    event_id: Identifier
    kind: str = Field(min_length=1, max_length=160, validation_alias=AliasChoices("kind", "event_type"))
    object_label: str | None = Field(default=None, max_length=160)
    subject_ids: list[Identifier] = Field(default_factory=list, max_length=32)
    started_at: Timestamp
    ended_at: Timestamp
    description: str = Field(min_length=1, max_length=2000)
    pre_evidence_ids: list[Identifier] = Field(min_length=1, max_length=64)
    post_evidence_ids: list[Identifier] = Field(min_length=1, max_length=64)

    @property
    def evidence_ids(self) -> list[str]:
        return list(dict.fromkeys([*self.pre_evidence_ids, *self.post_evidence_ids]))

    @property
    def event_type(self) -> str:
        return self.kind

    @model_validator(mode="after")
    def ordered_time(self):
        if self.ended_at < self.started_at:
            raise ValueError("event ends before it starts")
        return self


class PhysicalDecision(PhysicalModel):
    summary: str = Field(max_length=3000)
    objects: list[PhysicalObject] = Field(default_factory=list, max_length=128)
    observations: list[PhysicalObservation] = Field(default_factory=list, max_length=128)
    events: list[TemporalEvent] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def unique_ids(self):
        for items, key in [(self.objects, "detection_id"), (self.observations, "observation_id"),
                           (self.events, "event_id")]:
            ids = [getattr(item, key) for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {key} in physical decision")
        return self


def validate_physical_decision(decision: PhysicalDecision | dict, window: PhysicalWindow | dict,
                               known_track_ids: set[str] | None = None) -> PhysicalDecision:
    """Validate only supplied evidence, causal times and local/known physical references."""
    decision = PhysicalDecision.model_validate(decision)
    window = PhysicalWindow.model_validate(window)
    known = set(known_track_ids or ())
    references = window.evidence()
    detections = {item.detection_id: item for item in decision.objects}

    def time_in_window(timestamp):
        if not window.started_at <= timestamp <= window.ended_at:
            raise ValueError("physical assertion is outside the input window")

    def cited(ids, at):
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence reference")
        missing = set(ids) - references.keys()
        if missing:
            raise ValueError("unknown physical evidence: " + ", ".join(sorted(missing)))
        items = [references[key] for key in ids]
        if any(item.ended_at > at for item in items):
            raise ValueError("physical assertion cites future evidence")
        return items

    def subject(identifier, at):
        if identifier in detections:
            if detections[identifier].timestamp > at:
                raise ValueError("physical assertion references a future detection")
        elif identifier not in known:
            raise ValueError("unknown physical subject: " + identifier)

    for item in decision.objects:
        time_in_window(item.timestamp)
        cited(item.evidence_ids, item.timestamp)
        frame = references.get(item.frame_evidence_id)
        if (frame is None or frame.kind != "frame" or item.frame_evidence_id not in item.evidence_ids
                or frame.source_id != item.location.camera_id or frame.started_at != item.timestamp):
            raise ValueError("object box must cite its exact camera frame and timestamp")
        if item.track_id is not None and item.track_id not in known:
            raise ValueError("unknown persistent physical track: " + item.track_id)
    for item in decision.observations:
        time_in_window(item.observed_at)
        cited(item.evidence_ids, item.observed_at)
        if item.subject_id is not None:
            subject(item.subject_id, item.observed_at)
    for item in decision.events:
        time_in_window(item.started_at)
        time_in_window(item.ended_at)
        pre = cited(item.pre_evidence_ids, item.started_at)
        post = cited(item.post_evidence_ids, item.ended_at)
        if set(item.pre_evidence_ids) & set(item.post_evidence_ids):
            raise ValueError("temporal event needs distinct pre/post evidence")
        if max(part.ended_at for part in pre) >= min(part.started_at for part in post):
            raise ValueError("temporal pre evidence must precede post evidence")
        if max(part.ended_at for part in post) < item.started_at:
            raise ValueError("post evidence precedes the claimed event interval")
        for identifier in item.subject_ids:
            subject(identifier, item.ended_at)
    return decision


def parse_physical_decision(text: str) -> PhysicalDecision:
    """One JSON object; never repair or invent physical claims."""
    return PhysicalDecision.model_validate_json(text)

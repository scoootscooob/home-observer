"""Structured local verification of physical outcomes from local evidence only.

The frontier asks a typed question about an opaque track. The verifier reads the
local physical journal, the tracking history and, optionally, a local learned
check over retained frames. It returns confirmed, contradicted or unknown with
a bounded reason code. Missing localization is never evidence of absence, and
two-dimensional displacement alone never confirms a lift; that requires
agreeing learned evidence. All frame references stay in the private audit part.
"""
from __future__ import annotations

import math
import re
import time
import uuid
from typing import Callable, Literal

from pydantic import Field

from .schema import StrictModel
from .watches import ID

QUESTIONS = ("moved_from_initial_location", "clear_of_support", "still_present", "pickup_completed")


class VerificationRequest(StrictModel):
    request_id: str = Field(pattern=ID)
    question: Literal[QUESTIONS]
    subject_id: str = Field(pattern=r"^phys_[A-Za-z0-9_-]{1,96}$")
    reference_event_id: str | None = Field(default=None, pattern=r"^pevt_[A-Za-z0-9_-]{1,96}$")
    lookback_s: float = Field(default=30, gt=0, le=300)


class VerifierPolicy(StrictModel):
    """Local thresholds. The frontier cannot change them through a request."""

    min_center_displacement: float = Field(default=0.15, gt=0, le=1)
    max_iou_for_moved: float = Field(default=0.3, ge=0, le=1)
    max_displacement_for_unmoved: float = Field(default=0.05, gt=0, le=1)
    min_iou_for_unmoved: float = Field(default=0.6, ge=0, le=1)
    min_visible_states: int = Field(default=3, ge=1, le=1000)
    stale_after_s: float = Field(default=5.0, gt=0, le=300)
    learned_positive_kinds: list[str] = Field(default_factory=lambda: [
        "pick-up", "pickup", "pick_up", "take", "lift", "picked_up", "grab"])
    learned_negative_kinds: list[str] = Field(default_factory=lambda: [
        "put-down", "put_down", "put", "put-on", "put-in", "put-into", "place", "placed", "release"])


def _center(box):
    if isinstance(box, dict):
        box = [box["x_min"], box["y_min"], box["x_max"], box["y_max"]]
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box


def box_iou(left, right) -> float:
    (_, a), (_, b) = _center(left), _center(right)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _normalized_kind(value):
    return str(value).strip().lower().replace(" ", "-") if isinstance(value, str) else ""


class LocalVerifier:
    """Reads local journals only. ``learned_check`` is an optional local callable.

    ``learned_check(subject_id, question, source_as_of, lookback_s)`` returns
    ``{"value": True | False | None, "confidence": float, "audit": {...}}`` using
    local media, or raises. The verifier never fetches media itself.
    """

    def __init__(self, journal, tracking_journal, *, clock_mapping: dict, policy: VerifierPolicy | None = None,
                 learned_check: Callable | None = None, clock=time.time, latest_source_time: Callable | None = None):
        self.journal, self.tracking = journal, tracking_journal
        self.policy = policy or VerifierPolicy()
        self.learned_check, self.clock = learned_check, clock
        # A question asked after the newest frame is answered as of the newest available
        # evidence, with the staleness recorded; it is never projected past the source.
        self.latest_source_time = latest_source_time
        self._verification_id = None
        self._staleness = 0.0
        origin, wall, scale = (clock_mapping.get("source_origin"), clock_mapping.get("wall_origin"),
                               clock_mapping.get("scale", 1.0))
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (origin, wall, scale)) or scale <= 0:
            raise ValueError("verifier requires a finite clock mapping with positive scale")
        self.source_origin, self.wall_origin, self.scale = float(origin), float(wall), float(scale)

    def to_source(self, wall_time):
        return self.source_origin + (wall_time - self.wall_origin) / self.scale

    def to_wall(self, source_time):
        return self.wall_origin + (source_time - self.source_origin) * self.scale

    def _geometry(self, request, source_as_of):
        detections = self.journal.track_detections(request.subject_id, limit=1)
        if not detections:
            return {"verdict": "unknown", "reason_code": "subject_unknown", "confidence": 0.0,
                    "frames_examined": 0, "span": None, "audit": {}}
        initial = detections[0]
        history = self.tracking.history(request.subject_id, since=source_as_of - request.lookback_s,
                                        until=source_as_of)
        audit = {"initial_box": initial["box"], "initial_timestamp": initial["timestamp"],
                 "initial_window_id": initial["window_id"], "history_states": len(history),
                 "history_frame_ids": [row["frame_evidence_id"] for row in history][-64:]}
        if not history:
            return {"verdict": "unknown", "reason_code": "insufficient_tracking_history", "confidence": 0.0,
                    "frames_examined": 0, "span": None, "audit": audit}
        span = {"started_at": history[0]["timestamp"], "ended_at": history[-1]["timestamp"]}
        latest = history[-1]
        visible = [row for row in history if row["visibility"] == "visible" and row["box"] is not None]
        audit.update(latest_visibility=latest["visibility"], visible_states=len(visible),
                     latest_timestamp=latest["timestamp"], latest_age_s=source_as_of - latest["timestamp"])
        if latest["visibility"] != "visible" or latest["box"] is None:
            return {"verdict": "unknown", "reason_code": "localization_uncertain", "confidence": 0.0,
                    "frames_examined": len(history), "span": span, "audit": audit}
        if len(visible) < self.policy.min_visible_states:
            return {"verdict": "unknown", "reason_code": "insufficient_tracking_history", "confidence": 0.0,
                    "frames_examined": len(history), "span": span, "audit": audit}
        (cx0, cy0), _ = _center(initial["box"])
        (cx1, cy1), latest_box = _center(latest["box"])
        displacement = math.hypot(cx1 - cx0, cy1 - cy0)
        iou = box_iou(initial["box"], latest["box"])
        score = latest.get("heuristic_confidence")
        score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else 0.0
        audit.update(latest_box=latest_box, center_displacement=displacement, iou_with_initial=iou,
                     latest_heuristic_confidence=score)
        if displacement >= self.policy.min_center_displacement and iou <= self.policy.max_iou_for_moved:
            return {"verdict": "confirmed", "reason_code": "geometric_displacement",
                    "confidence": round(min(1.0, max(0.0, score)) * 0.9, 3), "frames_examined": len(history),
                    "span": span, "audit": audit}
        if displacement <= self.policy.max_displacement_for_unmoved and iou >= self.policy.min_iou_for_unmoved:
            return {"verdict": "contradicted", "reason_code": "no_displacement_observed",
                    "confidence": round(min(1.0, max(0.0, score)) * 0.9, 3), "frames_examined": len(history),
                    "span": span, "audit": audit}
        return {"verdict": "unknown", "reason_code": "ambiguous_displacement", "confidence": 0.0,
                "frames_examined": len(history), "span": span, "audit": audit}

    def _learned_events(self, request, source_as_of):
        events = self.journal.recent_events(source_as_of, limit=200, since=source_as_of - request.lookback_s)
        positive, negative = [], []
        for item in events:
            if request.subject_id not in item.get("subject_ids", []):
                continue
            kind = _normalized_kind(item.get("kind"))
            if kind in self.policy.learned_positive_kinds:
                positive.append(item)
            elif kind in self.policy.learned_negative_kinds:
                negative.append(item)
        return positive, negative

    def verify(self, request: VerificationRequest | dict, *, now=None, verification_id: str | None = None) -> dict:
        request = request if isinstance(request, VerificationRequest) else VerificationRequest.model_validate(request)
        now = self.clock() if now is None else now
        if verification_id is not None and not re.fullmatch(r"pver_[a-f0-9]{32}", verification_id):
            raise ValueError("verification_id must be a pver_ identifier")
        self._verification_id = verification_id
        source_as_of = self.to_source(now)
        self._staleness = 0.0
        latest = self.latest_source_time() if self.latest_source_time is not None else None
        if isinstance(latest, (int, float)) and math.isfinite(latest) and latest < source_as_of:
            self._staleness = source_as_of - latest
            source_as_of = float(latest)
        if request.reference_event_id is not None:
            known = {item["event_id"] for item in self.journal.recent_events(source_as_of, limit=1000)}
            if request.reference_event_id not in known:
                return self._result(request, now, {"verdict": "unknown", "reason_code": "reference_event_unknown",
                                                   "confidence": 0.0, "frames_examined": 0, "span": None,
                                                   "audit": {}})
        if request.question == "still_present":
            history = self.tracking.history(request.subject_id, since=source_as_of - request.lookback_s,
                                            until=source_as_of)
            visible = [row for row in history if row["visibility"] == "visible"]
            span = {"started_at": history[0]["timestamp"], "ended_at": history[-1]["timestamp"]} if history else None
            if visible and source_as_of - visible[-1]["timestamp"] <= self.policy.stale_after_s:
                score = visible[-1].get("heuristic_confidence") or 0.0
                outcome = {"verdict": "confirmed", "reason_code": "visible_in_window",
                           "confidence": round(min(1.0, max(0.0, float(score))) * 0.9, 3)}
            else:
                # Unseen is not gone: absence of localization never contradicts presence.
                outcome = {"verdict": "unknown", "reason_code": "not_observed_in_window", "confidence": 0.0}
            return self._result(request, now, {**outcome, "frames_examined": len(history), "span": span,
                                               "audit": {"visible_states": len(visible)}})
        geometry = self._geometry(request, source_as_of)
        if request.question == "moved_from_initial_location":
            return self._result(request, now, geometry)
        # clear_of_support and pickup_completed need agreeing learned evidence.
        if geometry["verdict"] != "confirmed":
            return self._result(request, now, geometry)
        positive, negative = self._learned_events(request, source_as_of)
        audit = {**geometry["audit"], "learned_positive_events": [e["event_id"] for e in positive],
                 "learned_negative_events": [e["event_id"] for e in negative]}
        latest_learned = max([*positive, *negative], key=lambda e: (e["ended_at"], e["started_at"]), default=None)
        if latest_learned is not None and latest_learned in negative:
            return self._result(request, now, {**geometry, "verdict": "contradicted",
                                               "reason_code": "learned_evidence_contradicts", "audit": audit})
        if positive:
            confidence = min(geometry["confidence"], max(float(e.get("confidence", 0)) for e in positive))
            return self._result(request, now, {**geometry, "verdict": "confirmed",
                                               "reason_code": "geometric_and_learned_agreement",
                                               "confidence": round(confidence, 3), "audit": audit})
        if self.learned_check is None:
            return self._result(request, now, {**geometry, "verdict": "unknown",
                                               "reason_code": "learned_model_unavailable", "confidence": 0.0,
                                               "audit": audit})
        try:
            check = self.learned_check(request.subject_id, request.question, source_as_of, request.lookback_s)
            value, confidence = check.get("value"), float(check.get("confidence", 0.0))
            if value not in (True, False, None) or not 0 <= confidence <= 1:
                raise ValueError("learned check returned an invalid verdict")
        except Exception as exc:
            return self._result(request, now, {**geometry, "verdict": "unknown",
                                               "reason_code": "learned_output_invalid", "confidence": 0.0,
                                               "audit": {**audit, "learned_error_type": type(exc).__name__}})
        audit["learned_check"] = check.get("audit", {})
        if value is True:
            return self._result(request, now, {**geometry, "verdict": "confirmed",
                                               "reason_code": "geometric_and_learned_agreement",
                                               "confidence": round(min(geometry["confidence"], confidence), 3),
                                               "audit": audit})
        if value is False:
            return self._result(request, now, {**geometry, "verdict": "contradicted",
                                               "reason_code": "learned_evidence_contradicts",
                                               "confidence": round(confidence, 3), "audit": audit})
        return self._result(request, now, {**geometry, "verdict": "unknown",
                                           "reason_code": "requires_learned_evidence", "confidence": 0.0,
                                           "audit": audit})

    def _result(self, request, now, outcome):
        span = outcome.get("span")
        return {
            "verification_id": self._verification_id or "pver_" + uuid.uuid4().hex, "request_id": request.request_id,
            "question": request.question, "subject_id": request.subject_id,
            "reference_event_id": request.reference_event_id, "verdict": outcome["verdict"],
            "confidence": float(outcome["confidence"]), "reason_code": outcome["reason_code"],
            "evaluated_at": now, "frames_examined": int(outcome["frames_examined"]),
            "evidence_span": {"started_at": self.to_wall(span["started_at"]),
                              "ended_at": self.to_wall(span["ended_at"])} if span else {},
            "audit": {**outcome.get("audit", {}), "evidence_stale_s": round(self._staleness, 3)},
        }

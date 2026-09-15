"""Bounded CSRT propagation with visible, explicitly heuristic plausibility guards.

This tracks a supplied object box. It never discovers objects or infers recipe
actions. OpenCV's success flag is checked alongside appearance, scale, boundary
and frame-gap evidence; none of these checks proves persistent identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .physical_tracking import ContinuousTracker, FlowTrack, decode_frame
from .temporal_buffer import CapturedFrame


@dataclass
class AppearanceTrack(FlowTrack):
    backend: Any = None
    initial_histogram: Any = None
    initial_area: float = 0.0
    appearance_distance: float | None = None
    visible_fraction: float | None = None
    failure_reason: str | None = None


class CSRTContinuousTracker(ContinuousTracker):
    """One-camera CSRT boxes with conservative loss and no automatic reacquisition.

    `confidence` is an uncalibrated minimum of supplied seed confidence, histogram
    similarity and visible box fraction. It is not a probability of correctness.
    Motion uses approximate background translation compensation; camera rotation,
    parallax, occlusion and similar-looking objects remain unresolved limitations.
    """

    def __init__(
        self,
        *,
        maximum_tracks=8,
        min_points=4,
        motion_threshold=0.003,
        settle_seconds=0.3,
        lost_ttl_seconds=5.0,
        scale=0.5,
        max_frame_gap_s=0.25,
        max_appearance_distance=0.65,
        min_visible_fraction=0.65,
        min_area_ratio=0.25,
        max_area_ratio=4.0,
        max_step_area_ratio=1.7,
        max_center_step=0.12,
        motion_confirm_frames=3,
        tracker_factory=None,
    ):
        super().__init__(
            maximum_tracks=maximum_tracks,
            min_points=min_points,
            motion_threshold=motion_threshold,
            settle_seconds=settle_seconds,
            lost_ttl_seconds=lost_ttl_seconds,
            motion_confirm_frames=motion_confirm_frames,
        )
        values = [
            scale,
            max_frame_gap_s,
            max_appearance_distance,
            min_visible_fraction,
            min_area_ratio,
            max_area_ratio,
            max_step_area_ratio,
            max_center_step,
        ]
        if not np.isfinite(values).all() or not 0 < scale <= 1 or max_frame_gap_s <= 0:
            raise ValueError("finite tracker limits, a positive frame gap and scale in (0,1] are required")
        if not 0 < max_appearance_distance < 1 or not 0 < min_visible_fraction <= 1:
            raise ValueError("appearance and visibility guards must lie in their valid unit interval")
        if not 0 < min_area_ratio <= 1 <= max_area_ratio or max_step_area_ratio <= 1 or max_center_step <= 0:
            raise ValueError("invalid scale or center-displacement guard")
        self.scale, self.max_frame_gap_s = scale, max_frame_gap_s
        self.max_appearance_distance, self.min_visible_fraction = (
            max_appearance_distance,
            min_visible_fraction,
        )
        self.min_area_ratio, self.max_area_ratio = min_area_ratio, max_area_ratio
        self.max_step_area_ratio, self.max_center_step = max_step_area_ratio, max_center_step
        self._tracker_factory = tracker_factory
        self.loss_reasons = {}
        self.compensated_frames = 0
        self.backend_updates = 0
        self._config = dict(
            min_points=min_points,
            motion_threshold=motion_threshold,
            settle_seconds=settle_seconds,
            lost_ttl_seconds=lost_ttl_seconds,
            scale=scale,
            max_frame_gap_s=max_frame_gap_s,
            max_appearance_distance=max_appearance_distance,
            min_visible_fraction=min_visible_fraction,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_step_area_ratio=max_step_area_ratio,
            max_center_step=max_center_step,
            motion_confirm_frames=motion_confirm_frames,
            tracker_factory=tracker_factory,
        )

    def clone_single(self):
        """Fresh tracker for chronological replay of one delayed detection."""
        return type(self)(maximum_tracks=1, **self._config)

    def _new_backend(self):
        if self._tracker_factory is not None:
            return self._tracker_factory()
        import cv2

        factory = getattr(cv2, "TrackerCSRT_create", None)
        if factory is None:
            raise RuntimeError("CSRT requires the opencv-contrib-python-headless sensing dependency")
        return factory()

    def _scaled(self, image):
        import cv2

        return cv2.resize(image, None, fx=self.scale, fy=self.scale) if self.scale != 1 else image

    @staticmethod
    def _area(box):
        return float((box[2] - box[0]) * (box[3] - box[1]))

    @staticmethod
    def _histogram(image, box):
        import cv2

        height, width = image.shape[:2]
        left, top, right, bottom = np.rint(np.clip(box, 0, 1) * [width, height, width, height]).astype(int)
        if right - left < 4 or bottom - top < 4:
            return None
        crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([crop], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(histogram, histogram, alpha=1, norm_type=cv2.NORM_L1)

    def seed(self, track_id, label, box, frame: CapturedFrame, *, confidence=0.8):
        # Shared validation prevents historical detections being installed as if
        # current. The caller must replay a clone to the current capture frame.
        import cv2

        image = decode_frame(frame)
        coordinates = np.asarray(box, dtype=np.float32)
        backend = self._new_backend()
        prior_track = self.tracks.get(track_id)
        prior_gray, prior_frame = self.previous_gray, self.previous_frame
        try:
            super().seed(track_id, label, box, frame, confidence=confidence)
            base = self.tracks[track_id]
            histogram = self._histogram(image, coordinates)
            if histogram is None:
                raise ValueError("seed is too small for appearance evidence")
            scaled = self._scaled(image)
            height, width = scaled.shape[:2]
            left, top, right, bottom = np.rint(coordinates * [width, height, width, height]).astype(int)
            initialized = backend.init(scaled, (int(left), int(top), int(right - left), int(bottom - top)))
            if initialized is False:
                raise RuntimeError("CSRT backend did not initialize")
            self.tracks[track_id] = AppearanceTrack(
                **vars(base),
                backend=backend,
                initial_histogram=histogram,
                initial_area=self._area(coordinates),
                appearance_distance=0.0,
                visible_fraction=1.0,
            )
            self.tracks[track_id].visibility = "visible"
            self.previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except BaseException:
            if prior_track is None:
                self.tracks.pop(track_id, None)
            else:
                self.tracks[track_id] = prior_track
            self.previous_gray, self.previous_frame = prior_gray, prior_frame
            raise

    def _guard(self, track, proposal, image):
        import cv2

        if (
            proposal.shape != (4,)
            or not np.isfinite(proposal).all()
            or proposal[2] <= proposal[0]
            or proposal[3] <= proposal[1]
        ):
            return "invalid_box", None, 0.0
        clipped = np.clip(proposal, 0, 1)
        visible = self._area(clipped) / self._area(proposal)
        if visible < self.min_visible_fraction:
            return "mostly_outside_view", None, visible
        area = self._area(proposal)
        ratio, step_ratio = area / track.initial_area, area / self._area(track.box)
        if not self.min_area_ratio <= ratio <= self.max_area_ratio:
            return "implausible_total_scale", None, visible
        if not 1 / self.max_step_area_ratio <= step_ratio <= self.max_step_area_ratio:
            return "implausible_scale_step", None, visible
        old_center = (track.box[:2] + track.box[2:]) / 2
        center = (proposal[:2] + proposal[2:]) / 2
        if np.linalg.norm(center - old_center) > self.max_center_step:
            return "implausible_center_step", None, visible
        histogram = self._histogram(image, proposal)
        if histogram is None:
            return "insufficient_visible_patch", None, visible
        distance = float(cv2.compareHist(track.initial_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA))
        if distance > self.max_appearance_distance:
            return "appearance_changed", distance, visible
        return None, distance, visible

    def _event(self, track, kind, first, last, confidence, compensated):
        event = super()._event(track, kind, first, last, confidence, compensated)
        event["event_id"] = f"csrt_{track.track_id}_{kind}_{last.sequence}"
        event["uncertainty"] = (
            "CSRT with heuristic appearance/scale/visibility guards; confidence is an uncalibrated score, "
            "not a probability. Background compensation models translation only. Geometric change does not "
            "prove semantic pickup, placement, persistent identity, or physical absence."
        )
        if track.failure_reason:
            event["uncertainty"] += " Tracking became uncertain: " + track.failure_reason + "."
        return event

    def _lose(self, track, frame, reason, compensated, events, states):
        track.visibility, track.lost_since, track.failure_reason = "uncertain", frame.timestamp, reason
        track.stable_since = track.stable_anchor = None
        self.flow_failures += 1
        self.loss_reasons[reason] = self.loss_reasons.get(reason, 0) + 1
        # The zero score expresses absence of current target localization, not
        # confidence that the physical object disappeared.
        events.append(self._event(track, "visibility_lost", track.last_visible, frame, 0.0, compensated))
        states.append(
            {
                "track_id": track.track_id,
                "label": track.label,
                "visibility": "uncertain",
                "timestamp": frame.timestamp,
                "box": None,
                "evidence_id": frame.evidence_id,
                "confidence": 0.0,
                "confidence_kind": "uncalibrated_heuristic",
                "calibrated_confidence": None,
                "failure_reason": reason,
            }
        )

    def update(self, frame: CapturedFrame):
        import cv2

        image = decode_frame(frame)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.previous_frame is None:
            self.previous_gray, self.previous_frame = gray, frame
            self.frames_processed += 1
            return [], []
        if (
            frame.camera_id != self.previous_frame.camera_id
            or frame.timestamp <= self.previous_frame.timestamp
            or frame.sequence <= self.previous_frame.sequence
        ):
            raise ValueError("one chronological camera with increasing source frames is required")
        if gray.shape != self.previous_gray.shape:
            raise ValueError("camera resolution changed")
        gap = frame.timestamp - self.previous_frame.timestamp
        camera_motion, compensated = (
            self._camera_motion(gray) if gap <= self.max_frame_gap_s else (np.zeros(2), False)
        )
        self.compensated_frames += int(compensated)
        height, width = gray.shape
        scaled = self._scaled(image)
        scaled_h, scaled_w = scaled.shape[:2]
        states, events, expired = [], [], []
        for track in self.tracks.values():
            if track.lost_since is not None:
                if frame.timestamp - track.lost_since > self.lost_ttl_seconds:
                    expired.append(track.track_id)
                continue
            if gap > self.max_frame_gap_s:
                self._lose(track, frame, "source_frame_gap", False, events, states)
                continue
            try:
                self.backend_updates += 1
                success, box = track.backend.update(scaled)
            except cv2.error:
                success, box = False, None
            if not success:
                self._lose(track, frame, "backend_failed", compensated, events, states)
                continue
            x, y, w, h = box
            proposal = np.array([x / scaled_w, y / scaled_h, (x + w) / scaled_w, (y + h) / scaled_h])
            reason, distance, visible = self._guard(track, proposal, image)
            track.appearance_distance, track.visible_fraction = distance, visible
            if reason:
                self._lose(track, frame, reason, compensated, events, states)
                continue
            old_center = (track.box[:2] + track.box[2:]) / 2
            center = (proposal[:2] + proposal[2:]) / 2
            relative_motion = (
                float(np.linalg.norm(center - old_center - camera_motion / [width, height]))
                if compensated
                else None
            )
            score = float(min(track.confidence, 1 - distance, visible))
            track.box = np.clip(proposal, 0, 1)
            if not compensated:
                # Uncompensated camera motion cannot prove object motion/settling.
                track.stable_since = track.stable_anchor = None
            elif relative_motion >= self.motion_threshold:
                track.stable_since = track.stable_anchor = None
                if not track.moving:
                    if track.motion_streak == 0:
                        track.motion_streak_anchor = self.previous_frame
                    track.motion_streak += 1
                    if track.motion_streak >= self.motion_confirm_frames:
                        track.motion_anchor = track.motion_streak_anchor
                        events.append(self._event(track, "motion_started", track.motion_streak_anchor, frame,
                                                  score, True))
                        track.moving = True
                        track.motion_streak, track.motion_streak_anchor = 0, None
            elif track.moving:
                track.motion_streak, track.motion_streak_anchor = 0, None
                if track.stable_since is None:
                    track.stable_since, track.stable_anchor = frame.timestamp, frame
                if frame.timestamp - track.stable_since >= self.settle_seconds:
                    events.append(
                        self._event(track, "motion_settled", track.stable_anchor, frame, score, True)
                    )
                    track.moving = False
                    track.motion_anchor = None
            else:
                track.motion_streak, track.motion_streak_anchor = 0, None
            track.last_visible, track.visibility = frame, "visible"
            states.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "box": track.box.tolist(),
                    "timestamp": frame.timestamp,
                    "evidence_id": frame.evidence_id,
                    "visibility": "visible",
                    "confidence": score,
                    "confidence_kind": "uncalibrated_heuristic",
                    "calibrated_confidence": None,
                    "appearance_distance": distance,
                    "visible_fraction": visible,
                    "relative_motion": relative_motion,
                    "camera_compensated": compensated,
                    "total_area_ratio": self._area(proposal) / track.initial_area,
                }
            )
        for track_id in expired:
            del self.tracks[track_id]
        self.previous_gray, self.previous_frame = gray, frame
        self.frames_processed += 1
        return states, events

    def report(self):
        return {
            **super().report(),
            "backend": "OpenCV CSRT",
            "scale": self.scale,
            "loss_reasons": dict(self.loss_reasons),
            "camera_compensated_frames": self.compensated_frames,
            "backend_updates": self.backend_updates,
            "visible_tracks": sum(track.lost_since is None for track in self.tracks.values()),
            "uncertain_tracks": sum(track.lost_since is not None for track in self.tracks.values()),
            "guards": {key: value for key, value in self._config.items() if key != "tracker_factory"},
            "confidence_kind": "uncalibrated_heuristic",
            "calibrated_confidence": None,
            "scope": "Supplied-box propagation with appearance, scale, visibility and gap checks; "
            "no object discovery, semantic action recognition, or guaranteed identity.",
        }

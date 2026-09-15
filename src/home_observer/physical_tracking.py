"""Continuous classical tracking between slower learned observations.

This propagates observed image boxes; it does not discover or classify objects.
Geometric events are explicitly distinct from semantic pickup/placement events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .temporal_buffer import CapturedFrame


@dataclass
class FlowTrack:
    track_id: str
    label: str
    box: np.ndarray
    points: np.ndarray | None
    confidence: float
    anchor: CapturedFrame
    last_visible: CapturedFrame
    visibility: str = 'visible'
    moving: bool = False
    motion_anchor: CapturedFrame | None = None
    stable_since: float | None = None
    stable_anchor: CapturedFrame | None = None
    lost_since: float | None = None
    motion_streak: int = 0
    motion_streak_anchor: CapturedFrame | None = None


def decode_frame(frame: CapturedFrame):
    import cv2
    result = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError('captured frame is not a decodable image')
    return result


def activity_fraction(previous, current, pixel_threshold=20):
    """Scheduling signal only; lighting/camera motion can also cause activity."""
    import cv2
    if previous is None:
        return 0.0
    left = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if previous.ndim == 3 else previous
    right = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
    if left.shape != right.shape:
        raise ValueError('camera resolution changed within a stream')
    return float(np.mean(cv2.absdiff(left, right) > pixel_threshold))


class ContinuousTracker:
    """A bounded single-camera tracker with forward/backward flow checks."""
    def __init__(self, *, maximum_tracks=32, min_points=4, motion_threshold=0.003,
                 settle_seconds=0.3, lost_ttl_seconds=5.0, motion_confirm_frames=3):
        if maximum_tracks < 1 or min_points < 3 or min(motion_threshold, settle_seconds, lost_ttl_seconds) <= 0:
            raise ValueError('invalid tracking limits')
        if type(motion_confirm_frames) is not int or motion_confirm_frames < 1:
            raise ValueError('motion confirmation requires a positive whole number of consecutive frames')
        self.maximum_tracks, self.min_points = maximum_tracks, min_points
        self.motion_threshold, self.settle_seconds, self.lost_ttl_seconds = motion_threshold, settle_seconds, lost_ttl_seconds
        # A single above-threshold step (tracker re-centering right after a seed, jitter) is not
        # object motion; motion_started needs this many consecutive moving frames and is
        # anchored at the frame before the streak began.
        self.motion_confirm_frames = motion_confirm_frames
        self.tracks = {}
        self.previous_gray = self.previous_frame = None
        self.frames_processed = self.flow_failures = 0

    def _features(self, gray, box):
        import cv2
        height, width = gray.shape
        mask = np.zeros_like(gray)
        left, top, right, bottom = np.rint(box * [width, height, width, height]).astype(int)
        mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=3,
                                      blockSize=3, mask=mask)

    def seed(self, track_id, label, box, frame: CapturedFrame, *, confidence=0.8):
        import cv2
        if not track_id.startswith('phys_') or not label or not 0 <= confidence <= 1:
            raise ValueError('seed requires a physical track ID, label and confidence')
        coordinates = np.asarray(box, dtype=np.float32)
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
            raise ValueError('box must contain four normalized finite coordinates')
        if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            raise ValueError('box must have positive area')
        if track_id not in self.tracks and len(self.tracks) >= self.maximum_tracks:
            raise ValueError('physical tracker capacity reached')
        if self.previous_frame and (frame.camera_id != self.previous_frame.camera_id or
                                    frame.timestamp != self.previous_frame.timestamp):
            raise ValueError('seed on the current frame; replay buffered frames for delayed observations')
        gray = cv2.cvtColor(decode_frame(frame), cv2.COLOR_BGR2GRAY)
        self.previous_gray, self.previous_frame = gray, frame
        points = self._features(gray, coordinates)
        visibility = 'visible' if points is not None and len(points) >= self.min_points else 'uncertain'
        self.tracks[track_id] = FlowTrack(track_id, label, coordinates, points, confidence, frame, frame,
                                          visibility=visibility)

    @staticmethod
    def _flow(previous, current, points):
        import cv2
        if points is None or not len(points):
            return None, None
        moved, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
            winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if moved is None or status is None or error is None:
            return None, None
        returned, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None,
            winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if returned is None or back_status is None:
            return None, None
        good = ((status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
                & (np.linalg.norm(points.reshape(-1, 2) - returned.reshape(-1, 2), axis=1) < 1.5)
                & (error.reshape(-1) < 30))
        return moved, good

    def _camera_motion(self, current):
        import cv2
        mask = np.full_like(self.previous_gray, 255)
        height, width = mask.shape
        for track in self.tracks.values():
            left, top, right, bottom = np.rint(track.box * [width, height, width, height]).astype(int)
            mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 0
        points = cv2.goodFeaturesToTrack(self.previous_gray, 120, 0.02, 5, mask=mask)
        moved, good = self._flow(self.previous_gray, current, points)
        if good is None or good.sum() < 8:
            return np.zeros(2), False
        displacement = moved.reshape(-1, 2)[good] - points.reshape(-1, 2)[good]
        return np.median(displacement, axis=0), True

    def _event(self, track, kind, first, last, confidence, compensated):
        return {'event_id': f'flow_{track.track_id}_{kind}_{last.sequence}', 'kind': kind,
                'object_label': track.label, 'subject_ids': [track.track_id],
                'started_at': first.timestamp, 'ended_at': last.timestamp,
                'available_at': last.captured_at, 'confidence': float(confidence),
                'uncertainty': 'Classical image tracking; geometric change does not prove semantic pickup, placement or physical absence.',
                'description': f'{track.label}: {kind.replace("_", " ")} in the camera view',
                'pre_evidence_ids': [first.evidence_id], 'post_evidence_ids': [last.evidence_id],
                'camera_compensated': compensated, 'pre_frame': first, 'post_frame': last}

    def update(self, frame: CapturedFrame):
        import cv2
        gray = cv2.cvtColor(decode_frame(frame), cv2.COLOR_BGR2GRAY)
        if self.previous_frame is None:
            self.previous_gray, self.previous_frame = gray, frame
            self.frames_processed += 1
            return [], []
        if frame.camera_id != self.previous_frame.camera_id or frame.timestamp <= self.previous_frame.timestamp:
            raise ValueError('one chronological camera is required')
        if gray.shape != self.previous_gray.shape:
            raise ValueError('camera resolution changed')
        camera_motion, compensated = self._camera_motion(gray)
        height, width = gray.shape
        states, events, expired = [], [], []
        for track in self.tracks.values():
            if track.lost_since is not None:
                if frame.timestamp - track.lost_since > self.lost_ttl_seconds:
                    expired.append(track.track_id)
                continue  # Reacquisition requires a new learned observation, not fabricated flow.
            moved, good = self._flow(self.previous_gray, gray, track.points)
            if good is None or good.sum() < self.min_points or good.mean() < 0.4:
                self.flow_failures += 1
                track.visibility, track.lost_since = 'uncertain', frame.timestamp
                events.append(self._event(track, 'visibility_lost', track.last_visible, frame, 0.7, compensated))
                states.append({'track_id': track.track_id, 'label': track.label, 'visibility': 'uncertain',
                               'timestamp': frame.timestamp, 'box': None, 'evidence_id': frame.evidence_id})
                continue
            displacement = np.median(moved.reshape(-1, 2)[good] - track.points.reshape(-1, 2)[good], axis=0)
            shift = displacement / [width, height]
            proposed = track.box + np.array([shift[0], shift[1], shift[0], shift[1]])
            proposed = np.clip(proposed, 0, 1)
            if proposed[2] - proposed[0] < 0.005 or proposed[3] - proposed[1] < 0.005:
                track.visibility, track.lost_since = 'uncertain', frame.timestamp
                events.append(self._event(track, 'visibility_lost', track.last_visible, frame, 0.7, compensated))
                continue
            track.box = proposed
            relative_motion = float(np.linalg.norm((displacement - camera_motion) / [width, height]))
            confidence = float(track.confidence * good.mean() * (1 if compensated else 0.8))
            moving = relative_motion >= self.motion_threshold
            if moving:
                track.stable_since = None
                track.stable_anchor = None
                if not track.moving:
                    if track.motion_streak == 0:
                        track.motion_streak_anchor = self.previous_frame
                    track.motion_streak += 1
                    if track.motion_streak >= self.motion_confirm_frames:
                        track.motion_anchor = track.motion_streak_anchor
                        events.append(self._event(track, 'motion_started', track.motion_streak_anchor, frame,
                                                  confidence, compensated))
                        track.moving = True
                        track.motion_streak, track.motion_streak_anchor = 0, None
            elif track.moving:
                track.motion_streak, track.motion_streak_anchor = 0, None
                if track.stable_since is None:
                    track.stable_since = frame.timestamp
                    track.stable_anchor = frame
                if frame.timestamp - track.stable_since >= self.settle_seconds:
                    events.append(self._event(track, 'motion_settled', track.stable_anchor, frame, confidence, compensated))
                    track.moving = False
                    track.motion_anchor = None
            else:
                track.motion_streak, track.motion_streak_anchor = 0, None
            track.points = moved[good].reshape(-1, 1, 2)
            if len(track.points) < 20:
                refreshed = self._features(gray, track.box)
                if refreshed is not None and len(refreshed) >= self.min_points:
                    track.points = refreshed
            track.last_visible, track.visibility = frame, 'visible'
            states.append({'track_id': track.track_id, 'label': track.label, 'box': track.box.tolist(),
                           'timestamp': frame.timestamp, 'evidence_id': frame.evidence_id, 'visibility': 'visible',
                           'confidence': confidence, 'relative_motion': relative_motion,
                           'camera_compensated': compensated})
        for key in expired:
            del self.tracks[key]
        self.previous_gray, self.previous_frame = gray, frame
        self.frames_processed += 1
        return states, events

    def report(self):
        return {'frames_processed': self.frames_processed, 'active_tracks': len(self.tracks),
                'flow_failures': self.flow_failures, 'maximum_tracks': self.maximum_tracks,
                'scope': 'Translation tracking between observations; no semantic re-identification or 3D-world guarantee.'}

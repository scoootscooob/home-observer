import cv2
import numpy as np
import pytest

from home_observer.object_tracker import CSRTContinuousTracker
from home_observer.temporal_buffer import CapturedFrame


class Backend:
    def __init__(self, boxes=None):
        self.boxes = iter(boxes or [(30, 20, 40, 40)] * 20)
        self.calls = 0

    def init(self, image, box):
        self.initial = box

    def update(self, image):
        self.calls += 1
        return True, next(self.boxes)


def frame(index, *, color=(0, 0, 255), timestamp=None):
    rng = np.random.default_rng(14)
    image = rng.integers(0, 60, (100, 120, 3), dtype=np.uint8)
    image[20:60, 30:70] = color
    ok, jpeg = cv2.imencode(".png", image)
    assert ok
    return CapturedFrame(
        "ego", index + 1, index * 0.05 if timestamp is None else timestamp, 100 + index * 0.05, jpeg.tobytes()
    )


def tracker(backend=None, **kwargs):
    backend = backend or Backend()
    result = CSRTContinuousTracker(tracker_factory=lambda: backend, scale=1, **kwargs)
    result._camera_motion = lambda gray: (np.zeros(2), True)
    result.seed("phys_bowl", "bowl", [30 / 120, 20 / 100, 70 / 120, 60 / 100], frame(0), confidence=0.9)
    return result, backend


def test_api_success_with_changed_appearance_loses_identity_and_never_auto_reacquires():
    runtime, backend = tracker(lost_ttl_seconds=0.1)
    states, events = runtime.update(frame(1, color=(255, 0, 0)))
    assert states[0]["failure_reason"] == "appearance_changed"
    assert states[0]["box"] is None
    assert events[0]["kind"] == "visibility_lost"
    assert events[0]["confidence"] == 0
    assert "uncalibrated" in events[0]["uncertainty"]
    assert runtime.tracks["phys_bowl"].lost_since == 0.05
    runtime.update(frame(2))
    runtime.update(frame(4))
    assert backend.calls == 1
    assert not runtime.tracks


def test_large_source_gap_is_unknown_without_backend_update():
    runtime, backend = tracker()
    states, events = runtime.update(frame(1, timestamp=0.5))
    assert backend.calls == 0
    assert states[0]["failure_reason"] == "source_frame_gap"
    assert events[0]["kind"] == "visibility_lost"


def test_implausible_scale_and_outside_view_do_not_become_valid_boxes():
    for box, expected in [
        ((0, 0, 120, 100), "implausible_total_scale"),
        ((100, 20, 40, 40), "mostly_outside_view"),
    ]:
        runtime, _ = tracker(Backend([box]))
        states, _ = runtime.update(frame(1))
        assert states[0]["failure_reason"] == expected


def test_current_seed_only_and_clone_preserves_guards_without_track_state():
    runtime, _ = tracker(max_appearance_distance=0.5)
    runtime.update(frame(1))
    with pytest.raises(ValueError, match="current frame"):
        runtime.seed("phys_old", "bowl", [0.25, 0.2, 0.58, 0.6], frame(0))
    clone = runtime.clone_single()
    assert clone.maximum_tracks == 1
    assert clone.max_appearance_distance == 0.5
    assert not clone.tracks


def test_uncompensated_camera_motion_produces_no_object_motion_event():
    runtime, _ = tracker(Backend([(32, 20, 40, 40)]))
    runtime._camera_motion = lambda gray: (np.zeros(2), False)
    states, events = runtime.update(frame(1))
    assert states[0]["relative_motion"] is None
    assert states[0]["confidence_kind"] == "uncalibrated_heuristic"
    assert states[0]["calibrated_confidence"] is None
    assert not events


def test_failed_backend_seed_restores_previous_track_and_frame():
    runtime, _ = tracker()
    existing = runtime.tracks["phys_bowl"]

    class FailedBackend(Backend):
        def init(self, image, box):
            return False

    runtime._tracker_factory = FailedBackend
    with pytest.raises(RuntimeError, match="did not initialize"):
        runtime.seed("phys_bowl", "bowl", [0.25, 0.2, 0.58, 0.6], frame(0))
    assert runtime.tracks["phys_bowl"] is existing
    assert runtime.previous_frame.sequence == 1


def test_single_recentering_step_after_seed_is_not_motion_but_sustained_motion_is():
    boxes = [(34, 20, 40, 40)] + [(34, 20, 40, 40)] * 5 + [(38, 20, 40, 40), (42, 20, 40, 40), (46, 20, 40, 40),
                                                             (50, 20, 40, 40)] + [(50, 20, 40, 40)] * 12
    runtime, _ = tracker(Backend(boxes), motion_confirm_frames=3)
    kinds = []
    for index in range(1, 23):
        _, events = runtime.update(frame(index))
        kinds.extend((event["kind"], round(event["started_at"], 2), round(event["ended_at"], 2)) for event in events)
    assert kinds[0][0] == "motion_started"
    assert kinds[0][1] == pytest.approx(0.30)
    assert kinds[0][2] == pytest.approx(0.45)
    assert [kind for kind, *_ in kinds] == ["motion_started", "motion_settled"]

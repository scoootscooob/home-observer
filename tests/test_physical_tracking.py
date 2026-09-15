import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from home_observer.physical_tracking import ContinuousTracker  # noqa: E402
from home_observer.temporal_buffer import CapturedFrame  # noqa: E402


def scene(index, x, *, hidden=False, camera_shift=0):
    rng = np.random.default_rng(92)
    image = rng.integers(0, 60, (120, 180, 3), dtype=np.uint8)
    if not hidden:
        texture = np.random.default_rng(13).integers(50, 255, (30, 30, 3), dtype=np.uint8)
        image[45:75, x:x + 30] = texture
    if camera_shift:
        image = cv2.warpAffine(image, np.float32([[1, 0, camera_shift], [0, 1, 0]]), (180, 120))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert ok
    return CapturedFrame('counter', index + 1, index / 10, 100 + index / 10, encoded.tobytes())


def test_continuous_identity_and_motion_settlement_with_no_inference_calls():
    tracker = ContinuousTracker(settle_seconds=0.25)
    tracker.seed('phys_bowl', 'bowl', [40 / 180, 45 / 120, 70 / 180, 75 / 120], scene(0, 40))
    events, last = [], None
    for index in range(1, 14):
        states, changes = tracker.update(scene(index, min(40 + 4 * index, 60)))
        events.extend(changes)
        if states:
            last = states[0]
    assert [event['kind'] for event in events] == ['motion_started', 'motion_settled']
    assert all(event['subject_ids'] == ['phys_bowl'] for event in events)
    assert last['visibility'] == 'visible'
    assert abs(last['box'][0] * 180 - 60) < 3
    assert tracker.frames_processed == 13
    assert 'placement' in events[-1]['uncertainty']


def test_occlusion_is_uncertain_not_fabricated_object_departure():
    tracker = ContinuousTracker(lost_ttl_seconds=0.2)
    tracker.seed('phys_bowl', 'bowl', [40 / 180, 45 / 120, 70 / 180, 75 / 120], scene(0, 40))
    states, events = tracker.update(scene(1, 40, hidden=True))
    assert events[0]['kind'] == 'visibility_lost'
    assert states[0]['visibility'] == 'uncertain' and states[0]['box'] is None
    for index in range(2, 8):
        tracker.update(scene(index, 40, hidden=True))
    assert not tracker.tracks


def test_camera_motion_is_not_reported_as_relative_object_motion():
    tracker = ContinuousTracker()
    tracker.seed('phys_bowl', 'bowl', [40 / 180, 45 / 120, 70 / 180, 75 / 120], scene(0, 40))
    for index in range(1, 5):
        states, events = tracker.update(scene(index, 40, camera_shift=2 * index))
        assert not events
        assert states[0]['camera_compensated']


def test_stale_detection_cannot_be_seeded_as_if_it_were_current():
    tracker = ContinuousTracker()
    tracker.update(scene(2, 40))
    with pytest.raises(ValueError, match='current frame'):
        tracker.seed('phys_bowl', 'bowl', [0.2, 0.3, 0.4, 0.7], scene(0, 40))

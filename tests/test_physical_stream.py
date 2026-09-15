import json
import time

import cv2
import numpy as np

from home_observer.physical_stream import PhysicalStream
from home_observer.physical_tracking import ContinuousTracker
from home_observer.temporal_buffer import ActivityBuffer, CapturedFrame


def test_capture_and_brief_activity_continue_during_slow_inference(tmp_path):
    class SlowModel:
        def __init__(self):
            self.calls = []

        def observe(self, request):
            self.calls.append(time.time())
            assert set(request) == {'window', 'tracks', 'recent_events', 'focus'}
            time.sleep(0.3)
            return {'decision': {'summary': 'No semantic assertion in this plumbing test.'}, 'metrics': {}}

    start = time.time()
    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)

    def frames():
        for i in range(60):
            delay = start + i * 0.005 - time.time()
            if delay > 0:
                time.sleep(delay)
            image = base.copy()
            if 20 <= i <= 23:  # Brief 20ms object-like visual change while inference is busy.
                image[10:30, 10:30] = 255
            ok, encoded = cv2.imencode('.jpg', image)
            assert ok
            yield CapturedFrame('camera', i + 1, float(i * 0.005), time.time(), encoded.tobytes())

    model = SlowModel()
    stream = PhysicalStream(tmp_path, model, wall_origin=start,
        buffer=ActivityBuffer(pre_seconds=.02, post_seconds=.02, max_clip_seconds=.1,
                              heartbeat_seconds=.08, max_frames=100), keep_completed=16)
    try:
        report = stream.run(frames())
        assert report['captured_frames'] == 60
        assert report['tracker']['frames_processed'] == 60
        assert report['spool']['dropped_sequences'] == 0
        assert not report['errors']
        assert model.calls[0] < report['capture'][-1]['captured_at']
        retained = set()
        for path in (tmp_path / 'sequences').glob('*/manifest.json'):
            retained.update(x['sequence'] for x in json.loads(path.read_text())['frames'])
        assert {20, 21, 22, 23, 24, 25}.issubset(retained)
        rows = [json.loads(line) for line in stream.predictions.read_text().splitlines()]
        assert all(x['full_sequence_frames'] >= x['sampled_frames'] for x in rows)
        assert rows[0]['emitted_at'] > report['capture'][-1]['captured_at']
    finally:
        stream.close()


def test_inference_failure_does_not_stop_capture_and_is_recorded(tmp_path):
    class BrokenModel:
        def observe(self, request):
            raise ValueError('deliberate backend failure')

    _, encoded = cv2.imencode('.jpg', np.zeros((40, 40, 3), np.uint8))
    frames = [CapturedFrame('camera', i + 1, float(i), time.time(), encoded.tobytes()) for i in range(4)]
    stream = PhysicalStream(tmp_path, BrokenModel(), buffer=ActivityBuffer(heartbeat_seconds=2))
    try:
        report = stream.run(frames)
        assert report['captured_frames'] == 4
        assert report['spool']['clips']['failed'] >= 1
        assert 'deliberate backend failure' in stream.predictions.read_text()
        assert report['journal']['events'] == 0
    finally:
        stream.close()


def test_slow_tracking_does_not_block_capture(tmp_path):
    class SlowTracker(ContinuousTracker):
        def __init__(self):
            super().__init__()
            self.completed = []

        def update(self, frame):
            time.sleep(.012)
            result = super().update(frame)
            self.completed.append(time.time())
            return result

    class NoAssertions:
        def observe(self, request):
            return {'decision': {'summary': 'Capture isolation test'}}

    _, encoded = cv2.imencode('.jpg', np.zeros((40, 40, 3), np.uint8))

    def frames():
        for i in range(40):
            time.sleep(.002)
            yield CapturedFrame('camera', i + 1, float(i * .002), time.time(), encoded.tobytes())

    tracker = SlowTracker()
    stream = PhysicalStream(tmp_path, NoAssertions(), tracker=tracker)
    try:
        report = stream.run(frames())
        assert report['captured_frames'] == len(tracker.completed) == 40
        assert report['capture'][-1]['captured_at'] < tracker.completed[19]
        assert report['tracking_frame_losses'] == 0
        assert not report['errors']
    finally:
        stream.close()


def test_dropped_final_tracking_anchor_cannot_keep_worker_alive_after_eof(tmp_path):
    import threading

    class UnusedModel:
        def observe(self, request):
            raise AssertionError("This regression exercises the local tracking queue only")

    _, encoded = cv2.imencode('.jpg', np.zeros((40, 40, 3), np.uint8))
    first = CapturedFrame('camera', 1, 0.0, time.time(), encoded.tobytes())
    last = CapturedFrame('camera', 2, 0.1, time.time(), encoded.tobytes())
    stream = PhysicalStream(tmp_path, UnusedModel(), tracking_queue_frames=1, initial_context_frames=0)
    worker = threading.Thread(target=stream._track)
    try:
        # Capture outpaces tracking: the model's final anchor was captured and
        # retained for perception, but the size-one local tracking queue drops it.
        stream._capture([first, last])
        assert stream.capture_done.is_set()
        assert stream.tracking_frame_losses == 1
        stream.inference_done.set()
        stream.seeds.put({'track_id': 'phys_pending_anchor', 'label': 'bowl',
                          'timestamp': last.timestamp, 'frame_evidence_id': last.evidence_id,
                          'box': {'x_min': .1, 'y_min': .1, 'x_max': .5, 'y_max': .5},
                          'confidence': .8})
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive(), 'Dropped EOF anchor must be discarded, never requeued forever'
        assert stream.seeds.empty()
        assert stream.seed_log
        assert stream.seed_log[-1]['track_id'] == 'phys_pending_anchor'
        assert stream.seed_log[-1]['status'] == 'anchor_unavailable'
        assert stream.seed_log[-1]['frame_evidence_id'] == last.evidence_id
        assert not stream.errors
    finally:
        # The unfixed implementation fails safely instead of hanging pytest.
        stream.stopping.set()
        if worker.ident is not None:
            worker.join(timeout=1)
        stream.close()

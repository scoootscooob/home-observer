from pathlib import Path

import pytest

from home_observer.temporal_buffer import (
    ActivityBuffer,
    BufferedSequence,
    CapturedFrame,
    ClipSpool,
    select_evidence_frames,
)


def frame(index, activity=0.0):
    return CapturedFrame('kitchen', index + 1, index / 10, 100 + index / 10, b'image', activity)


def test_brief_activity_keeps_before_during_after_even_when_consumer_is_busy(tmp_path):
    buffer = ActivityBuffer(pre_seconds=0.3, post_seconds=0.3, max_clip_seconds=2,
                            heartbeat_seconds=20, max_frames=30)
    spool = ClipSpool(tmp_path, model_frames=4)
    try:
        for index in range(40):
            for sequence in buffer.push(frame(index, 0.5 if index == 12 else 0)):
                spool.enqueue(sequence)
        job = spool.claim()  # No consumer ran while all forty capture frames arrived.
        assert job is not None
        chosen = [item['sequence'] - 1 for item in job['frames'] if item['selected_for_model']]
        assert 12 in chosen and min(chosen) < 12 and max(chosen) > 12
        assert buffer.captured == 40
        assert job['full_sequence_frames'] > job['selected_model_frames']
        assert all(Path(item['path']).is_file() for item in job['frames'])
    finally:
        spool.close()


def test_quiet_periodic_coverage_spans_interval_and_uniform_evidence():
    buffer = ActivityBuffer(pre_seconds=0.3, post_seconds=0.3, max_clip_seconds=3,
                            heartbeat_seconds=2, max_frames=40)
    clips = [clip for index in range(22) for clip in buffer.push(frame(index))]
    assert len(clips) == 1
    assert clips[0].frames[-1].timestamp - clips[0].frames[0].timestamp >= 1.9
    chosen = select_evidence_frames(clips[0].frames, maximum=4)
    assert chosen[1].timestamp >= 0.5 and chosen[2].timestamp >= 1.2


def test_queue_bounds_do_not_delete_inflight_evidence_and_losses_are_explicit(tmp_path):
    spool = ClipSpool(tmp_path, max_pending=1, keep_completed=1)
    sequence = BufferedSequence((frame(0), frame(1)), 'test', 0, 0.1)
    try:
        first = spool.enqueue(sequence)
        job = spool.claim()
        assert spool.enqueue(sequence) is None
        assert Path(job['frames'][0]['path']).exists()
        assert spool.report()['dropped_sequences'] == 1
        spool.complete(first, 102)
        second = spool.enqueue(sequence)
        spool.claim()
        spool.complete(second, 103)
        assert not (tmp_path / first).exists()
        assert (tmp_path / second).exists()
        assert spool.report()['clips']['expired'] == 1
    finally:
        spool.close()


def test_single_owner_and_interrupted_perception_is_recoverable(tmp_path):
    first = ClipSpool(tmp_path)
    identifier = first.enqueue(BufferedSequence((frame(0), frame(1)), 'test', None, None))
    assert first.claim()['clip_id'] == identifier
    with pytest.raises(RuntimeError, match='active owner'):
        ClipSpool(tmp_path)
    first.close()
    reopened = ClipSpool(tmp_path)
    try:
        assert reopened.claim()['clip_id'] == identifier
    finally:
        reopened.close()


def test_buffer_rejects_time_reversal_and_bounds_long_activity():
    buffer = ActivityBuffer(pre_seconds=0.1, post_seconds=0.1, max_clip_seconds=0.6,
                            heartbeat_seconds=2, max_frames=8)
    emitted = [clip for index in range(300) for clip in buffer.push(frame(index, 0.5))]
    assert emitted and max(len(clip.frames) for clip in emitted) <= 8
    assert len(buffer.ring) <= 8 and len(buffer.active) <= 8
    with pytest.raises(ValueError, match='increasing'):
        buffer.push(frame(20))

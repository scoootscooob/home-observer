"""Continuous frame retention and an independent, bounded sequence work queue.

Activity is a scheduling signal, not a semantic event label. Capture never waits
for inference. The full retained sequence and the model's selected frames are
recorded separately; overload losses are explicit.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import secrets
import shutil
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    camera_id: str
    sequence: int
    timestamp: float
    captured_at: float
    jpeg: bytes
    activity: float = 0.0

    def __post_init__(self):
        if not self.camera_id or self.sequence < 1 or not self.jpeg:
            raise ValueError('frame identity and image bytes are required')
        if not all(math.isfinite(x) for x in (self.timestamp, self.captured_at, self.activity)):
            raise ValueError('frame times and activity must be finite')
        if not 0 <= self.activity <= 1:
            raise ValueError('activity must be in [0,1]')

    @property
    def evidence_id(self):
        return f'{self.camera_id}:{self.sequence}'


@dataclass(frozen=True)
class BufferedSequence:
    frames: tuple[CapturedFrame, ...]
    reason: str
    activity_started_at: float | None
    activity_ended_at: float | None


class ActivityBuffer:
    """One source's ring, pre-roll and post-roll, driven on every captured frame."""
    def __init__(self, *, pre_seconds=1.0, post_seconds=1.0, max_clip_seconds=6.0,
                 heartbeat_seconds=6.0, activity_threshold=0.002, max_frames=180):
        if min(pre_seconds, post_seconds) < 0 or max_clip_seconds <= pre_seconds + post_seconds:
            raise ValueError('clip duration must exceed its pre/post-roll')
        if heartbeat_seconds <= 0 or max_frames < 2 or not 0 <= activity_threshold <= 1:
            raise ValueError('invalid buffering limits')
        self.pre_seconds, self.post_seconds = pre_seconds, post_seconds
        self.max_clip_seconds, self.heartbeat_seconds = max_clip_seconds, heartbeat_seconds
        self.activity_threshold, self.max_frames = activity_threshold, max_frames
        self.ring = deque(maxlen=max_frames)
        self.active = []
        self.activity_start = self.activity_end = self.last_emission = None
        self.last_frame = None
        self.captured = self.capacity_splits = self.emitted = 0

    def _finish(self, reason):
        if not self.active:
            return None
        result = BufferedSequence(tuple(self.active), reason, self.activity_start, self.activity_end)
        self.last_emission = self.active[-1].timestamp
        self.active = []
        self.activity_start = self.activity_end = None
        self.emitted += 1
        return result

    def push(self, frame: CapturedFrame) -> list[BufferedSequence]:
        if self.last_frame and (frame.camera_id != self.last_frame.camera_id or
                                frame.timestamp <= self.last_frame.timestamp or
                                frame.sequence <= self.last_frame.sequence):
            raise ValueError('one camera with strictly increasing source times and frame numbers is required')
        self.last_frame = frame
        self.captured += 1
        self.ring.append(frame)
        retention = max(self.max_clip_seconds, self.heartbeat_seconds, self.pre_seconds)
        while len(self.ring) > 1 and self.ring[0].timestamp < frame.timestamp - retention:
            self.ring.popleft()
        emitted = []
        active_now = frame.activity >= self.activity_threshold
        if not self.active and active_now:
            self.active = [item for item in self.ring if item.timestamp >= frame.timestamp - self.pre_seconds]
            self.activity_start = self.activity_end = frame.timestamp
        elif self.active:
            self.active.append(frame)
            if active_now:
                self.activity_end = frame.timestamp
        if self.active:
            duration = frame.timestamp - self.active[0].timestamp
            if len(self.active) >= self.max_frames or duration >= self.max_clip_seconds:
                self.capacity_splits += 1
                emitted.append(self._finish('activity_segment_limit'))
                # Keep overlap when activity continues across a bounded segment.
                if active_now:
                    self.active = [item for item in self.ring if item.timestamp >= frame.timestamp - self.pre_seconds]
                    self.activity_start = self.activity_end = frame.timestamp
            elif not active_now and frame.timestamp - self.activity_end >= self.post_seconds:
                emitted.append(self._finish('activity_with_pre_post_roll'))
        elif self.last_emission is None:
            self.last_emission = frame.timestamp
        elif frame.timestamp - self.last_emission >= self.heartbeat_seconds:
            self.active = list(self.ring)
            emitted.append(self._finish('periodic_coverage'))
        return [item for item in emitted if item]

    def flush(self):
        return self._finish('source_end')


def select_evidence_frames(frames, maximum=8):
    """Keep endpoints and activity peaks with immediate temporal context."""
    if maximum < 2:
        raise ValueError('at least two frames are required for temporal evidence')
    if len(frames) <= maximum:
        return list(frames)
    chosen = {0, len(frames) - 1}
    # Adjacent frames around a brief peak have priority over a uniform grid.
    candidates = sorted((i for i in range(1, len(frames) - 1) if frames[i].activity > 0),
                        key=lambda i: (-frames[i].activity, i))
    for index in candidates:
        if len(chosen) >= maximum:
            break
        for neighbor in (index, index - 1, index + 1):
            if len(chosen) < maximum:
                chosen.add(neighbor)
    for index in (round(i * (len(frames) - 1) / (maximum - 1)) for i in range(maximum)):
        if len(chosen) < maximum:
            chosen.add(index)
    while len(chosen) < maximum:
        left, right = max(zip(sorted(chosen), sorted(chosen)[1:]), key=lambda pair: pair[1] - pair[0])
        chosen.add((left + right) // 2)
    return [frames[index] for index in sorted(chosen)]


class ClipSpool:
    """Durable independent inference jobs; immutable media until completion.

    On restart, an interrupted processing job returns to pending. This only
    replays perception, never an action. Event/action idempotency is separate.
    """
    def __init__(self, root, *, max_pending=16, keep_completed=16, model_frames=8):
        if min(max_pending, keep_completed) < 1 or model_frames < 2:
            raise ValueError('positive queue and retention limits are required')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_pending, self.keep_completed, self.model_frames = max_pending, keep_completed, model_frames
        self.ownership = (self.root / '.queue.lock').open('a')
        try:
            fcntl.flock(self.ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.ownership.close()
            raise RuntimeError('this sequence spool already has an active owner') from None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'queue.sqlite', check_same_thread=False, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS clips (id TEXT PRIMARY KEY, state TEXT, manifest TEXT, completed_at REAL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS losses (camera_id TEXT, started_at REAL, ended_at REAL, reason TEXT, frames INTEGER)')
        self.db.execute("UPDATE clips SET state='pending' WHERE state='processing'")
        self.db.commit()

    def enqueue(self, sequence: BufferedSequence):
        if not sequence.frames:
            return None
        frames = sequence.frames
        with self.lock:
            pending = self.db.execute("SELECT count(*) FROM clips WHERE state IN ('pending','processing')").fetchone()[0]
            if pending >= self.max_pending:
                self.db.execute('INSERT INTO losses VALUES(?,?,?,?,?)',
                                (frames[0].camera_id, frames[0].timestamp, frames[-1].timestamp, 'queue_capacity', len(frames)))
                self.db.commit()
                return None
            clip_id = 'clip_' + secrets.token_hex(12)
            folder = self.root / clip_id
            folder.mkdir()
            chosen = {frame.evidence_id for frame in select_evidence_frames(frames, self.model_frames)}
            items = []
            try:
                for index, frame in enumerate(frames):
                    filename = f'{index:06d}.jpg'
                    (folder / filename).write_bytes(frame.jpeg)
                    items.append({'camera_id': frame.camera_id, 'sequence': frame.sequence,
                                  'timestamp': frame.timestamp, 'captured_at': frame.captured_at,
                                  'evidence_id': frame.evidence_id, 'path': str(folder / filename),
                                  'sha256': hashlib.sha256(frame.jpeg).hexdigest(), 'activity': frame.activity,
                                  'selected_for_model': frame.evidence_id in chosen})
                manifest = {'clip_id': clip_id, 'reason': sequence.reason, 'frames': items,
                            'started_at': frames[0].timestamp, 'ended_at': frames[-1].timestamp,
                            'activity_started_at': sequence.activity_started_at,
                            'activity_ended_at': sequence.activity_ended_at,
                            'full_sequence_frames': len(items), 'selected_model_frames': len(chosen),
                            'semantic_event_label': None}
                (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
                self.db.execute('INSERT INTO clips VALUES(?,?,?,NULL)', (clip_id, 'pending', json.dumps(manifest)))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                shutil.rmtree(folder)
                raise
            return clip_id

    def claim(self):
        with self.lock:
            row = self.db.execute("SELECT id,manifest FROM clips WHERE state='pending' ORDER BY rowid LIMIT 1").fetchone()
            if not row:
                return None
            self.db.execute("UPDATE clips SET state='processing' WHERE id=?", (row[0],))
            self.db.commit()
            return json.loads(row[1])

    def complete(self, clip_id, completed_at, *, failed=False):
        with self.lock:
            row = self.db.execute('SELECT state FROM clips WHERE id=?', (clip_id,)).fetchone()
            if not row or row[0] != 'processing':
                raise ValueError('only a claimed clip can be completed')
            self.db.execute('UPDATE clips SET state=?,completed_at=? WHERE id=?',
                            ('failed' if failed else 'complete', completed_at, clip_id))
            self.db.commit()
            old = self.db.execute("SELECT id FROM clips WHERE state IN ('complete','failed') ORDER BY completed_at DESC LIMIT -1 OFFSET ?",
                                  (self.keep_completed,)).fetchall()
            for (expired,) in old:
                shutil.rmtree(self.root / expired)
                self.db.execute("UPDATE clips SET state='expired' WHERE id=?", (expired,))
            self.db.commit()

    def report(self):
        with self.lock:
            return {'clips': dict(self.db.execute('SELECT state,count(*) FROM clips GROUP BY state').fetchall()),
                    'dropped_sequences': self.db.execute('SELECT count(*) FROM losses').fetchone()[0],
                    'dropped_frame_references': self.db.execute('SELECT coalesce(sum(frames),0) FROM losses').fetchone()[0],
                    'limits': {'pending_plus_processing': self.max_pending, 'completed_media': self.keep_completed},
                    'scope': 'Activity scheduling and retained input coverage; semantic event recall requires labels.'}

    def close(self):
        with self.lock:
            self.db.close()
            fcntl.flock(self.ownership.fileno(), fcntl.LOCK_UN)
            self.ownership.close()

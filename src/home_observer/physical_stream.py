"""Continuous capture/tracking with independent buffered learned inference.

Source timestamps remain in the physical journal. Watches receive an explicit
source-to-wall-clock projection; inference completion never backdates an event.
"""
from __future__ import annotations

import hashlib
import json
import queue
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

from .physical_journal import PhysicalJournal
from .physical_schema import PhysicalDecision, PhysicalWindow
from .physical_tracking import ContinuousTracker, activity_fraction, decode_frame
from .temporal_buffer import ActivityBuffer, BufferedSequence, CapturedFrame, ClipSpool
from .tracking_journal import TrackingJournal


def manifest_window(manifest):
    frames = [{k: item[k] for k in ('evidence_id', 'timestamp', 'path')}
              for item in manifest['frames'] if item['selected_for_model']]
    return PhysicalWindow.model_validate({
        'window_id': manifest['clip_id'], 'started_at': manifest['started_at'],
        'ended_at': manifest['ended_at'], 'clips': [{
            'clip_id': manifest['clip_id'] + '_camera',
            'camera_id': manifest['frames'][0]['camera_id'],
            'started_at': manifest['started_at'], 'ended_at': manifest['ended_at'], 'frames': frames}]})


def video_frames(path, *, camera_id, source_origin=0.0, wall_origin=None, realtime=True):
    """Decode every file frame using container PTS, falling back to declared FPS.

    A replay is explicitly a replay. It has one wall/source mapping; it neither
    fabricates gaps between clips nor repeats a final frame after source EOF.
    """
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError('cannot open video source')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError('video needs a valid frame rate for timestamp fallback')
    wall_origin = time.time() if wall_origin is None else wall_origin
    previous_pts = None
    index = 0
    try:
        while True:
            ok, decoded = capture.read()
            if not ok:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if previous_pts is not None and pts <= previous_pts:
                pts = index / fps
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError('video timestamps are not monotonic')
            if realtime:
                delay = wall_origin + pts - time.time()
                if delay > 0:
                    time.sleep(delay)
            captured_at = time.time()
            ok, encoded = cv2.imencode('.jpg', decoded, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise ValueError('JPEG encoding failed')
            index += 1
            yield CapturedFrame(camera_id, index, source_origin + pts, captured_at, encoded.tobytes())
            previous_pts = pts
    finally:
        capture.release()


class PhysicalStream:
    """One camera; bounded producer queues, persistent journal, separate watches.

    Capture, tracking and model calls each have an independent worker. The
    caller's thread owns the watch database and verified executor.
    """
    def __init__(self, output, model, *, journal=None, buffer=None, tracker=None,
                 coordinator=None, source_origin=0.0, wall_origin=None,
                 max_pending=16, keep_completed=16, model_frames=8,
                 history_frames=600, event_queue_size=256, inference_delay_s=0,
                 post_source_watch_seconds=0, event_evidence_limit=256, initial_context_frames=8,
                 tracking_queue_frames=128):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.model, self.coordinator = model, coordinator
        self.journal = journal or PhysicalJournal(self.output / 'journal.sqlite3')
        self.owns_journal = journal is None
        self.tracking_journal = TrackingJournal(self.journal.path)
        self.buffer, self.tracker = buffer or ActivityBuffer(), tracker or ContinuousTracker()
        self.spool = ClipSpool(self.output / 'sequences', max_pending=max_pending,
                               keep_completed=keep_completed, model_frames=model_frames)
        self.history = deque(maxlen=history_frames)
        self.events = queue.Queue(maxsize=event_queue_size)
        self.seeds = queue.Queue(maxsize=128)
        self.results = queue.Queue(maxsize=max_pending + 1)
        if tracking_queue_frames < 1:
            raise ValueError('tracking queue must be bounded and positive')
        self.tracking_frames = queue.Queue(maxsize=tracking_queue_frames)
        self.capture_done, self.inference_done = threading.Event(), threading.Event()
        self.tracking_done, self.stopping = threading.Event(), threading.Event()
        self.source_origin = source_origin
        self.wall_origin = time.time() if wall_origin is None else wall_origin
        self.inference_delay_s = inference_delay_s
        if not 0 <= post_source_watch_seconds <= 3600:
            raise ValueError('post-source watch drain must be bounded from 0 to 3600 seconds')
        self.post_source_watch_seconds = post_source_watch_seconds
        if event_evidence_limit < 1:
            raise ValueError('event evidence retention must be positive')
        self.event_evidence_limit = event_evidence_limit
        self.expired_event_evidence = 0
        if not 0 <= initial_context_frames <= min(history_frames, 128):
            raise ValueError('initial context must fit retained history and frame schema')
        self.initial_context_frames = initial_context_frames
        self.losses, self.errors, self.capture_stats = deque(maxlen=128), deque(maxlen=128), deque(maxlen=128)
        self.seed_log = deque(maxlen=128)
        self.capture_count = self.event_loss_count = 0
        self.tracking_frame_losses = 0
        self.tracking_times = deque(maxlen=128)
        self.capture_lateness_sum = self.capture_lateness_max = 0.0
        self.predictions = self.output / 'predictions.jsonl'
        self.geometry = self.output / 'geometric-events.jsonl'
        self.track_log = self.output / 'tracks.jsonl'
        self.capture_log = self.output / 'capture.jsonl'

    @property
    def clock_mapping(self):
        return {'source_origin': self.source_origin, 'wall_origin': self.wall_origin, 'scale': 1.0}

    @staticmethod
    def _append(path, value):
        # Detailed traces rotate independently of the durable compact journal.
        if path.exists() and path.stat().st_size > 8 * 1024 * 1024:
            for index in range(3, 0, -1):
                older = path.with_name(path.name + f'.{index}')
                if older.exists():
                    if index == 3:
                        older.unlink()
                    else:
                        older.replace(path.with_name(path.name + f'.{index + 1}'))
            path.replace(path.with_name(path.name + '.1'))
        with path.open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')

    def _emit_geometry(self, event):
        event = {**event, 'produced_at': time.time()}
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.event_loss_count += 1
            self.losses.append({'kind': 'local_event_queue_full', 'event_id': event['event_id'],
                                'started_at': event['started_at'], 'ended_at': event['ended_at']})

    def _apply_seeds(self):
        while True:
            try:
                item = self.seeds.get_nowait()
            except queue.Empty:
                break
            frame = next((f for f in self.history if f.evidence_id == item['frame_evidence_id']), None)
            if frame is None:
                waiting_for_anchor = not self.history or item['timestamp'] > self.history[-1].timestamp
                tracking_input_finished = self.capture_done.is_set() and self.tracking_frames.empty()
                if waiting_for_anchor and not tracking_input_finished:
                    self.seeds.put_nowait(item)
                    return  # Tracking has not reached this captured anchor yet.
                self.seed_log.append({'track_id': item['track_id'],
                                      'status': 'anchor_unavailable' if waiting_for_anchor else 'anchor_expired',
                                      'frame_evidence_id': item['frame_evidence_id']})
                continue
            try:
                self._reconcile_seed(item, frame)
            except Exception as exc:
                self.seed_log.append({'track_id': item.get('track_id'), 'status': 'seed_rejected',
                                      'error_type': type(exc).__name__, 'error': str(exc)[:1000]})

    def _reconcile_seed(self, item, frame):
        replay = self.tracker.clone_single() if hasattr(self.tracker, 'clone_single') else ContinuousTracker(
            maximum_tracks=1, min_points=self.tracker.min_points,
            motion_threshold=self.tracker.motion_threshold, settle_seconds=self.tracker.settle_seconds,
            lost_ttl_seconds=self.tracker.lost_ttl_seconds)
        box = item['box']
        replay.seed(item['track_id'], item['label'], [box[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')],
                    frame, confidence=item['confidence'])
        replayed = 0
        recovered_events = []
        replay_states, replay_state_frame = [], None
        for later in self.history:
            if later.timestamp > frame.timestamp:
                states, found = replay.update(later)
                if states:
                    replay_states, replay_state_frame = states, later
                recovered_events.extend(found)
                replayed += 1
        track = replay.tracks.get(item['track_id'])
        first_local_tracking = item['track_id'] not in self.tracker.tracks and not self.tracking_journal.latest(
            self.history[-1].timestamp, [item['track_id']])
        if track and track.lost_since is None and (item['track_id'] in self.tracker.tracks
                                                   or len(self.tracker.tracks) < self.tracker.maximum_tracks):
            self.tracker.tracks[item['track_id']] = track
            status = 'replayed_to_current'
        else:
            status = 'uncertain_after_replay'
        if first_local_tracking:
            if replay_states:
                self.tracking_journal.record(replay_states, replay_state_frame, method=type(self.tracker).__name__)
            for event in recovered_events:
                self._emit_geometry({**event, 'processing_mode': 'buffer_replay'})
        self.seed_log.append({'track_id': item['track_id'], 'status': status, 'replayed_frames': replayed,
                              'recovered_events': len(recovered_events) if first_local_tracking else 0,
                              'anchor_timestamp': frame.timestamp})

    def _capture(self, frames):
        previous = None
        try:
            for frame in frames:
                if self.stopping.is_set():
                    break
                started = time.perf_counter()
                decoded = decode_frame(frame)
                frame = replace(frame, activity=activity_fraction(previous, decoded))
                previous = decoded
                try:
                    self.tracking_frames.put_nowait(frame)
                except queue.Full:
                    self.tracking_frame_losses += 1
                    self.losses.append({'kind': 'tracking_queue_full', 'source_time': frame.timestamp,
                                        'evidence_id': frame.evidence_id})
                for sequence in self.buffer.push(frame):
                    self.spool.enqueue(sequence)
                if self.initial_context_frames and self.buffer.captured == self.initial_context_frames:
                    self.spool.enqueue(BufferedSequence(tuple(self.buffer.ring), 'initial_context', None, None))
                stats = {'source_time': frame.timestamp, 'captured_at': frame.captured_at,
                    'processing_s': time.perf_counter() - started,
                    'capture_lateness_s': max(0, frame.captured_at - self.to_wall(frame.timestamp))}
                self.capture_count += 1
                self.capture_lateness_sum += stats['capture_lateness_s']
                self.capture_lateness_max = max(self.capture_lateness_max, stats['capture_lateness_s'])
                self.capture_stats.append(stats)
                self._append(self.capture_log, stats)
            final = self.buffer.flush()
            if final:
                self.spool.enqueue(final)
        except Exception as exc:
            self.errors.append({'stage': 'capture', 'type': type(exc).__name__, 'error': str(exc)[:2000]})
        finally:
            self.capture_done.set()

    def _track(self):
        try:
            while not self.stopping.is_set():
                try:
                    frame = self.tracking_frames.get(timeout=.02)
                except queue.Empty:
                    self._apply_seeds()
                    if (self.capture_done.is_set() and self.inference_done.is_set()
                            and self.results.unfinished_tasks == 0 and self.seeds.empty()):
                        break
                    continue
                started = time.perf_counter()
                self.history.append(frame)
                states, events = self.tracker.update(frame)
                for event in events:
                    self._emit_geometry(event)
                self.tracking_journal.record(states, frame, method=type(self.tracker).__name__)
                self._apply_seeds()
                self._append(self.track_log, {'timestamp': frame.timestamp, 'captured_at': frame.captured_at,
                                             'evidence_id': frame.evidence_id, 'tracks': states})
                self.tracking_times.append({'timestamp': frame.timestamp, 'processing_s': time.perf_counter() - started,
                    'capture_to_tracking_s': time.time() - frame.captured_at})
                self.tracking_frames.task_done()
        except Exception as exc:
            self.errors.append({'stage': 'tracking', 'type': type(exc).__name__, 'error': str(exc)[:2000]})
        finally:
            self.tracking_done.set()

    def _infer(self):
        try:
            while not self.stopping.is_set():
                manifest = self.spool.claim()
                if manifest is None:
                    if self.capture_done.wait(0.02):
                        # Flush is committed before capture_done, so no later clip can appear.
                        manifest = self.spool.claim()
                        if manifest is None:
                            break
                    else:
                        continue
                window = manifest_window(manifest)
                request = {'window': window.model_dump(), 'tracks': self.registry(window.ended_at),
                           'recent_events': self.journal.recent_events(window.ended_at), 'focus': []}
                started = time.time()
                try:
                    if self.inference_delay_s:
                        time.sleep(self.inference_delay_s)
                    result = self.model.observe(request)
                except Exception as exc:
                    result = {'decision': None, 'error': {'type': type(exc).__name__, 'message': str(exc)[:2000]}}
                completed = time.time()
                self.results.put({'id': window.window_id, 'window': window.model_dump(), 'result': result,
                    'emitted_at': completed, 'inference_started_at': started, 'clock_mapping': self.clock_mapping,
                    'full_sequence_frames': manifest['full_sequence_frames'],
                    'sampled_frames': manifest['selected_model_frames']})
                # The next learned window must see the preceding committed
                # physical result. Capture and tracking remain independent.
                while self.results.unfinished_tasks and not self.stopping.is_set():
                    time.sleep(.005)
        except Exception as exc:
            self.errors.append({'stage': 'inference_worker', 'type': type(exc).__name__, 'error': str(exc)[:2000]})
        finally:
            self.inference_done.set()

    def to_wall(self, source_time):
        return self.wall_origin + source_time - self.source_origin

    def registry(self, as_of):
        tracks = self.journal.tracks(as_of)
        updates = {item['track_id']: item for item in self.tracking_journal.latest(
            as_of, [track['track_id'] for track in tracks])}
        return [{**track, **({'current_tracking_state': updates[track['track_id']]}
                            if track['track_id'] in updates else {})} for track in tracks]

    def _deliver(self, event, origin, available_at):
        if self.coordinator is None:
            return
        mapped = {**event, 'started_at': self.to_wall(event['started_at']),
                  'ended_at': self.to_wall(event['ended_at']), 'origin': origin,
                  'available_at': max(available_at, self.to_wall(event['ended_at']))}
        self.coordinator.engine.consume(mapped)
        self.coordinator.publish()

    def _geometry_result(self, event):
        before, after = event['pre_frame'], event['post_frame']
        identity = 'geometry_' + hashlib.sha256(event['event_id'].encode()).hexdigest()[:24]
        folder = self.output / 'event-evidence' / identity
        folder.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, frame in enumerate((before, after)):
            path = folder / f'{index}.jpg'
            path.write_bytes(frame.jpeg)
            frames.append({'evidence_id': frame.evidence_id, 'timestamp': frame.timestamp, 'path': str(path)})
        window = PhysicalWindow.model_validate({'window_id': identity, 'started_at': before.timestamp,
            'ended_at': after.timestamp, 'clips': [{'clip_id': identity + '_clip', 'camera_id': before.camera_id,
                'started_at': before.timestamp, 'ended_at': after.timestamp, 'frames': frames}]})
        payload = {k: v for k, v in event.items() if k not in (
            'pre_frame', 'post_frame', 'camera_compensated', 'available_at', 'produced_at', 'processing_mode')}
        decision = PhysicalDecision.model_validate({'summary': 'Continuous geometric tracking', 'events': [payload]})
        self.journal.ingest(window, decision)
        stored = next(e for e in self.journal.recent_events(after.timestamp, limit=1000)
                      if e['window_id'] == identity)
        available = event['produced_at']
        self._append(self.geometry, {'event': stored, 'origin': 'geometric', 'emitted_at': available,
            'journaled_at': time.time(),
            'processing_mode': event.get('processing_mode', 'continuous'),
            'captured_at': after.captured_at, 'clock_mapping': self.clock_mapping,
            'camera_compensated': event['camera_compensated']})
        self._deliver(stored, 'geometric', available)
        retained = sorted((p for p in folder.parent.iterdir() if p.is_dir() and not p.is_symlink()
                           and re.fullmatch(r'geometry_[a-f0-9]{24}', p.name)), key=lambda p: p.stat().st_mtime)
        for expired in retained[:-self.event_evidence_limit]:
            shutil.rmtree(expired)
            self.expired_event_evidence += 1
            self._append(self.output / 'evidence-retention.jsonl', {'window_id': expired.name,
                'media_expired_at': time.time(), 'reason': 'event_evidence_retention_limit',
                'journal_metadata_retained': True})

    def _learned_result(self, row):
        result, window = row['result'], row['window']
        failed = not result.get('decision') or bool(result.get('error'))
        if not failed:
            try:
                decision = PhysicalDecision.model_validate(result['decision'])
                mapping = self.journal.ingest(window, decision)
                for detection in decision.objects:
                    item = {**detection.model_dump(), 'track_id': mapping[detection.detection_id]}
                    try:
                        self.seeds.put_nowait(item)
                    except queue.Full:
                        self.seed_log.append({'track_id': item['track_id'], 'status': 'seed_queue_full'})
                for event in self.journal.recent_events(window['ended_at'], limit=1000):
                    if event['window_id'] == window['window_id']:
                        self._deliver(event, 'learned', row['emitted_at'])
            except Exception as exc:
                failed = True
                result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        self._append(self.predictions, row)
        if self.coordinator:
            # The frontier receives failures and inspectable evidence too. This
            # transport record cannot trigger a semantic event watch by itself.
            self.coordinator.engine.store._emit('perception.failed' if failed else 'perception.completed',
                row['id'], {'window': window, 'result': result, 'emitted_at': row['emitted_at'],
                    'clock_mapping': self.clock_mapping,
                    'full_sequence_manifest': str(self.spool.root / row['id'] / 'manifest.json'),
                    'full_sequence_frames': row['full_sequence_frames'], 'sampled_frames': row['sampled_frames']},
                time.time())
            self.coordinator.publish()
        self.spool.complete(row['id'], time.time(), failed=failed)

    def run(self, frames):
        capture = threading.Thread(target=self._capture, args=(frames,), name='physical-capture')
        tracking = threading.Thread(target=self._track, name='physical-tracking')
        inference = threading.Thread(target=self._infer, name='physical-inference')
        started = time.time()
        capture.start()
        tracking.start()
        inference.start()
        try:
            while not (self.capture_done.is_set() and self.tracking_done.is_set() and self.inference_done.is_set()
                       and self.results.empty() and self.events.empty()):
                if self.coordinator:
                    self.coordinator.tick()
                for incoming, handler in ((self.events, self._geometry_result), (self.results, self._learned_result)):
                    try:
                        item = incoming.get_nowait()
                    except queue.Empty:
                        continue
                    try:
                        handler(item)
                    except Exception as exc:
                        self.errors.append({'stage': handler.__name__, 'type': type(exc).__name__, 'error': str(exc)[:2000]})
                    finally:
                        incoming.task_done()
                time.sleep(0.01)
        except BaseException:
            self.stopping.set()
            raise
        finally:
            capture.join()
            inference.join()
            tracking.join()
        drain_until = time.time() + self.post_source_watch_seconds
        while self.coordinator and time.time() < drain_until:
            self.coordinator.tick()
            time.sleep(0.02)
        watches = []
        if self.coordinator:
            self.coordinator.tick()
            watches = [self.coordinator.engine.store.get(row[0]) for row in
                       self.coordinator.engine.store.db.execute('SELECT watch_id FROM local_watches')]
        report = {'started_at': started, 'finished_at': time.time(), 'clock_mapping': self.clock_mapping,
            'captured_frames': self.buffer.captured, 'capture': list(self.capture_stats),
            'capture_trace_tail_limit': 128, 'capture_lateness_mean_s': self.capture_lateness_sum / max(1, self.capture_count),
            'capture_lateness_max_s': self.capture_lateness_max, 'tracker': self.tracker.report(),
            'tracking_queue_frame_limit': self.tracking_frames.maxsize,
            'tracking_frame_losses': self.tracking_frame_losses, 'tracking_timing_tail': list(self.tracking_times),
            'buffer_capacity_splits': self.buffer.capacity_splits, 'spool': self.spool.report(),
            'seed_reconciliation': list(self.seed_log), 'seeds_pending_at_eof': self.seeds.qsize(),
            'losses': list(self.losses), 'local_event_losses': self.event_loss_count, 'errors': list(self.errors),
            'journal': self.journal.stats(), 'artificial_inference_delay_s': self.inference_delay_s,
            'tracking_journal': self.tracking_journal.stats(),
            'physical_registry_at_eof': self.registry(self.buffer.last_frame.timestamp) if self.buffer.last_frame else [],
            'event_evidence_retention_limit': self.event_evidence_limit,
            'initial_context_frames': self.initial_context_frames,
            'expired_event_evidence': self.expired_event_evidence,
            'watches_at_shutdown': watches, 'post_source_watch_seconds': self.post_source_watch_seconds,
            'source_eof': True, 'watch_runtime_continues_after_return': False,
            'scope': 'One continuous video source; activity coverage and geometry are separate from learned event accuracy.'}
        (self.output / 'stream-report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report

    def close(self):
        self.spool.close()
        self.tracking_journal.close()
        if self.owns_journal:
            self.journal.close()

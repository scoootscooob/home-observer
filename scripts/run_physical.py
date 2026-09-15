#!/usr/bin/env python3
"""Replay one continuous real clip through Runpod perception and local watches.

Frontier agents exchange typed files through --bridge. Credentials are local
files, never part of the perception request, journal or command output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from home_observer.coordinator import FileCoordinator
from home_observer.ha import HomeAssistant
from home_observer.object_tracker import CSRTContinuousTracker
from home_observer.physical_journal import PhysicalJournal
from home_observer.physical_remote import PhysicalRemoteModel
from home_observer.physical_schema import PhysicalDecision, PhysicalWindow
from home_observer.physical_stream import PhysicalStream, video_frames
from home_observer.watches import ApprovedResponse, WatchEngine, WatchStore


def bootstrap_seed(stream, first, seed):
    """An explicit observed box bootstrap; provenance is retained, never called model detection."""
    allowed = {'label', 'box', 'confidence', 'provenance', 'uncertainty'}
    if set(seed) != allowed:
        raise ValueError('seed requires exactly label, box, confidence, provenance and uncertainty')
    folder = stream.output / 'bootstrap'
    folder.mkdir(exist_ok=True)
    path = folder / 'first-frame.jpg'
    path.write_bytes(first.jpeg)
    window = PhysicalWindow.model_validate({'window_id': 'bootstrap', 'started_at': first.timestamp,
        'ended_at': first.timestamp, 'clips': [{'clip_id': 'bootstrap_clip', 'camera_id': first.camera_id,
            'started_at': first.timestamp, 'ended_at': first.timestamp, 'frames': [
                {'evidence_id': first.evidence_id, 'timestamp': first.timestamp, 'path': str(path)}]}]})
    decision = PhysicalDecision.model_validate({'summary': 'Explicit box bootstrap from recorded provenance',
        'objects': [{'detection_id': 'bootstrap_object', 'label': seed['label'], 'box': seed['box'],
            'location': {'camera_id': first.camera_id}, 'timestamp': first.timestamp,
            'frame_evidence_id': first.evidence_id, 'visibility': 'partially_occluded',
            'confidence': seed['confidence'], 'uncertainty': seed['uncertainty'],
            'evidence_ids': [first.evidence_id], 'attributes': {'bootstrap_provenance': seed['provenance']}}]})
    mapping = stream.journal.ingest(window, decision)
    stream.seeds.put_nowait({**decision.objects[0].model_dump(), 'track_id': mapping['bootstrap_object']})
    (folder / 'seed.json').write_text(json.dumps({'seed': seed, 'window': window.model_dump(),
        'decision': decision.model_dump(), 'mapping': mapping}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--camera-id', default='recipe')
    parser.add_argument('--source-origin', type=float, default=0.0)
    parser.add_argument('--url', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--ha-credentials', type=Path)
    parser.add_argument('--ha-url', default='http://127.0.0.1:8124')
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--wait-for-watch-seconds', type=float, default=60)
    parser.add_argument('--post-source-watch-seconds', type=float, default=60)
    parser.add_argument('--inference-delay-s', type=float, default=0)
    parser.add_argument('--tracker', choices=['csrt', 'lk'], default='csrt')
    parser.add_argument('--opencv-threads', type=int, default=1)
    args = parser.parse_args()
    if args.opencv_threads < 1:
        parser.error('--opencv-threads must be positive')
    import cv2
    cv2.setNumThreads(args.opencv_threads)
    output = args.output.resolve()
    if (output / 'stream-report.json').exists() or (output / 'journal.sqlite3').exists():
        raise ValueError('use a fresh output directory for a new recorded run')
    output.mkdir(parents=True, exist_ok=True)
    approved = [ApprovedResponse.model_validate(x) for x in json.loads(args.responses.read_text())]
    executor = None
    if args.ha_credentials:
        credentials = json.loads(args.ha_credentials.read_text())
        token = credentials.get('long_lived_token') or credentials.get('token') or credentials.get('access_token')
        executor = HomeAssistant(credentials.get('base_url', args.ha_url), token, execute_real=True)
    store = WatchStore(output / 'journal.sqlite3')
    engine = WatchEngine(store, approved, executor=executor)
    bridge = FileCoordinator(engine, args.bridge)
    model = PhysicalRemoteModel(args.url, args.token_file.read_text().strip(), dataset_root=output)
    journal = PhysicalJournal(output / 'journal.sqlite3')
    (output / 'runtime-ready.json').write_text(json.dumps({'bridge': str(args.bridge.resolve()),
        'approved_response_ids': [x.response_id for x in approved], 'ready_at': time.time(),
        'source_kind': 'continuous_file_replay', 'source_sha256': hashlib.sha256(args.video.read_bytes()).hexdigest(),
        'source_origin': args.source_origin, 'camera_id': args.camera_id}, indent=2) + '\n')
    deadline = time.time() + args.wait_for_watch_seconds
    while not store.db.execute("SELECT count(*) FROM local_watches WHERE status='armed'").fetchone()[0]:
        bridge.tick()
        if time.time() >= deadline:
            raise TimeoutError('No frontier-created watch arrived before the configured start deadline')
        time.sleep(.05)
    wall_origin = time.time()
    stream = PhysicalStream(output, model, journal=journal, coordinator=bridge,
        tracker=CSRTContinuousTracker() if args.tracker == 'csrt' else None,
        source_origin=args.source_origin, wall_origin=wall_origin,
        inference_delay_s=args.inference_delay_s, post_source_watch_seconds=args.post_source_watch_seconds)
    frames = iter(video_frames(args.video, camera_id=args.camera_id, source_origin=args.source_origin,
                               wall_origin=wall_origin, realtime=True))
    try:
        if args.seed:
            first = next(frames)
            bootstrap_seed(stream, first, json.loads(args.seed.read_text()))
            from itertools import chain
            frames = chain([first], frames)
        report = stream.run(frames)
        print(json.dumps({'captured_frames': report['captured_frames'], 'journal': report['journal'],
                          'errors': report['errors'], 'output': str(output)}))
        if report['errors']:
            raise SystemExit(1)
    finally:
        stream.close()
        journal.close()
        store.close()
        model.close()
        if executor:
            executor.client.close()


if __name__ == '__main__':
    main()

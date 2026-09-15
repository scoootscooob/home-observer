import importlib.util
from pathlib import Path

from home_observer.model import ModelConfig
from home_observer.physical_model import physical_context, physical_request_from_row

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'scripts/probe_physical_temporal.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_only_pixel_assignment_changes_and_all_source_hashes_are_recorded(tmp_path):
    frames = []
    for index in range(8):
        (tmp_path / f'{index}.jpg').write_bytes(f'original-pixels-{index}'.encode())
        frames.append({'evidence_id': f'f{index}', 'timestamp': float(index), 'path': f'{index}.jpg'})
    row = {'window': {'window_id': 'source', 'started_at': 0., 'ended_at': 7., 'clips': [
        {'clip_id': 'clip', 'camera_id': 'ego', 'started_at': 0., 'ended_at': 7., 'frames': frames}]},
        'target': {'events': [{'kind': 'PRIVATE_LABEL'}]}}
    config = ModelConfig(dataset_root=str(tmp_path), max_frames=8)
    for mode in ('reverse_pixels', 'static_first_frame'):
        request, recorded = probe.derive(row, mode, config)
        assert physical_context(request) == physical_context(physical_request_from_row(row))
        assert 'PRIVATE_LABEL' not in physical_context(request)
        assert [f['evidence_id'] for f in recorded] == [f'f{i}' for i in range(8)]
        assert [f['timestamp'] for f in recorded] == list(range(8))
        expected = list(reversed(range(8))) if mode == 'reverse_pixels' else [0] * 8
        assert [f['pixel_source_evidence_id'] for f in recorded] == [f'f{i}' for i in expected]
        assert sum(f['pixel_bytes_changed'] for f in recorded) == (8 if mode == 'reverse_pixels' else 7)

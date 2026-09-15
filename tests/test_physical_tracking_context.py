import pytest
from test_physical_schema import physical_window

from home_observer.model import ModelConfig
from home_observer.physical_model import physical_context, prepare_physical_request


def request(tmp_path, state):
    for name in ('before.jpg', 'after.jpg'):
        (tmp_path / name).write_bytes(b'fixture')
    return {'window': physical_window(), 'tracks': [{'track_id': 'phys_bowl', 'current_tracking_state': state}]}


@pytest.mark.parametrize('state', [
    {'timestamp': 11.01}, {'timestamp': float('nan')}, {'timestamp': float('inf')},
    {'timestamp': True}, {'timestamp': '11.0'}, {}, [],
    {'timestamp': 11.0, 'track_id': 'phys_other'},
])
def test_reject_future_invalid_or_wrong_identity_tracker_context(tmp_path, state):
    with pytest.raises(ValueError, match='current_tracking_state'):
        prepare_physical_request(request(tmp_path, state), ModelConfig(dataset_root=str(tmp_path)))


def test_preserve_causal_tracking_state_without_promoting_it_to_new_detection(tmp_path):
    state = {'track_id': 'phys_bowl', 'timestamp': 10.75, 'visibility': 'uncertain', 'box': None,
             'details': {'confidence_kind': 'uncalibrated_heuristic'}}
    clean = prepare_physical_request(request(tmp_path, state), ModelConfig(dataset_root=str(tmp_path)))
    assert clean['tracks'][0]['current_tracking_state'] == state
    assert 'last_detection' not in clean['tracks'][0]
    assert 'current_tracking_state' in physical_context(clean)

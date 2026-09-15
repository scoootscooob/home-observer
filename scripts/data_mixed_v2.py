#!/usr/bin/env python3
"""Prepare balanced, authored TRAIN/VALIDATION decisions without reading test data.

Media paths remain relative to the existing data/ directory. Existing procedural
images are reused byte-for-byte; telemetry, capture times and prior snapshots
are explicitly authored augmentations, not new observations of a real home.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from home_observer.data import CAMERAS, fixture_policy, opaque_id, sha256_file, write_jsonl
from home_observer.policy import Policy
from home_observer.schema import Decision, ObservationWindow

# step, lux, leak, buzzer, expected actions; step selects unchanged source pixels:
# 0=empty/dark; 1=occupied/dark; 2=occupied/bright; 3=occupied/light already on.
CASES = [
    (1, 12, False, False, 1), (1, 39, False, False, 1),
    (1, 0, False, False, 1), (0, 12, True, False, 1),
    (0, 39, True, False, 1), (2, 40, True, False, 1),
    (2, 41, True, False, 1), (3, 12, True, False, 1),
    (1, 12, True, False, 2), (1, 39, True, False, 2),
    (0, 12, False, False, 0), (0, 39, False, False, 0),
    (2, 40, False, False, 0), (2, 41, False, False, 0),
    (3, 12, False, False, 0), (3, 39, False, False, 0),
    (0, 12, True, True, 0), (2, 40, True, True, 0),
    (3, 12, True, True, 0), (0, 39, False, True, 0),
]
FULL_MASK = dict(summary=True, observations=True, actions=True, noop=True)
SUMMARY_MASK = dict(summary=True, observations=False, actions=False, noop=False)


def number(seed: str) -> int:
    return int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16)


def unwrap(value):
    return value.get('state') if isinstance(value, dict) else value


def is_on(value):
    value = unwrap(value)
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('on', 'true')


def binary_value(value: bool, encoding: int):
    if encoding == 0:
        return 'on' if value else 'off'
    if encoding == 1:
        return value
    if encoding == 2:
        return 'true' if value else 'false'
    return {'state': 'on' if value else 'off', 'attributes': {}}


def configured_policy():
    policy = fixture_policy()
    policy['rules'].extend([
        'For binary sensors, on, true and JSON true mean active; off, false and JSON false mean inactive. Unwrap a Home Assistant object using its state field. Parse numeric lux strings as numbers.',
        'Current device states are authoritative. Prior state is historical context, never a replacement for current telemetry. Previously observing an active sensor does not imply that an action was executed. Use current sensor and actuator states to decide whether an action is still required.',
        'Record clear room occupancy and motion/leak sensor states when absent from prior memory, stale, or changed. Do not repeat an unchanged fact. Other telemetry supplies action preconditions. If both new observations and permitted actions are empty, use noop=true.',
    ])
    return policy


def retime(window, row_id, started_at, audio_source):
    window = copy.deepcopy(window)
    window['window_id'] = row_id
    window['started_at'], window['ended_at'] = started_at, started_at + 10
    for frame in window['frames']:
        frame['evidence_id'] = opaque_id(f'{row_id}:frame:{frame["camera_id"]}')
        frame['timestamp'] = started_at + 9
    # The sound is an independently selected, authored non-speech fixture asset.
    window['audio'] = copy.deepcopy(audio_source['window']['audio'])
    for index, audio in enumerate(window['audio']):
        audio['evidence_id'] = opaque_id(f'{row_id}:audio:{index}')
        audio['started_at'], audio['ended_at'] = started_at + 8, started_at + 10
    return window


def authored_states(parent, lux, leak, buzzer, encoding):
    states = copy.deepcopy(parent['window']['device_states'])
    for entity in states:
        if entity.startswith('binary_sensor.motion_'):
            states[entity] = binary_value(is_on(states[entity]), encoding)
        elif entity.startswith('light.') and encoding == 3:
            states[entity] = {'state': unwrap(states[entity]), 'attributes': {}}
    states['binary_sensor.water_leak'] = binary_value(leak, encoding)
    states['switch.buzzer'] = {'state': 'on' if buzzer else 'off'} if encoding == 3 else ('on' if buzzer else 'off')
    states['sensor.ambient_lux'] = lux if encoding % 2 == 0 else str(lux)
    if encoding == 3:
        states['sensor.ambient_lux'] = {'state': str(lux), 'attributes': {'unit_of_measurement': 'lx'}}
    return states


def geometry(parent):
    return {room: is_on(parent['window']['device_states'][f'binary_sensor.motion_{room}']) for room in CAMERAS}


def journal_state(window, occupied, stale=False):
    timestamp = window['ended_at']
    state = {}
    def fact(entity, attribute, value, evidence, source):
        state.setdefault(entity, {})[attribute] = {'value': None if stale else value,
            'confidence': 0.0 if stale else 1.0, 'observed_at': timestamp,
            'stale': stale, 'source': source, 'evidence_ids': evidence}
    for entity, value in window['device_states'].items():
        fact(entity, 'state', unwrap(value), [f'device:{entity}'], 'device')
    for frame in window['frames']:
        room = frame['camera_id']
        fact(f'room.{room}', 'occupied', occupied[room], [frame['evidence_id']], 'model')
    return state


def target_for(window, occupied, prior):
    """Authored rule labels; no current target is used to construct prior state."""
    observations, actions = [], []
    states = window['device_states']
    def changed(entity, attribute, value, binary=False):
        prior_fact = prior.get(entity, {}).get(attribute, {})
        if not prior_fact or prior_fact.get('stale'):
            return True
        previous = prior_fact.get('value')
        return is_on(previous) != is_on(value) if binary else previous != value
    for frame in window['frames']:
        room = frame['camera_id']
        if changed(f'room.{room}', 'occupied', occupied[room]):
            observations.append({'entity_id': f'room.{room}', 'attribute': 'occupied', 'value': occupied[room],
                'confidence': 1.0, 'evidence_ids': [frame['evidence_id']]})
    for entity, value in states.items():
        if entity.startswith('binary_sensor.') and changed(entity, 'state', value, binary=True):
            observations.append({'entity_id': entity, 'attribute': 'state', 'value': 'on' if is_on(value) else 'off',
                'confidence': 1.0, 'evidence_ids': [f'device:{entity}']})
    lux = float(unwrap(states['sensor.ambient_lux']))
    for room in CAMERAS:
        motion, light = f'binary_sensor.motion_{room}', f'light.{room}'
        if (occupied[room] or is_on(states.get(motion, False))) and lux < 40 and unwrap(states[light]) == 'off':
            motion_evidence = f'device:{motion}' if motion in states else next(
                f['evidence_id'] for f in window['frames'] if f['camera_id'] == room)
            actions.append({'domain': 'light', 'service': 'turn_on', 'entity_id': light, 'data': {},
                'reason': ('Current motion is active' if motion in states else 'The current camera shows an occupant') + ', ambient lux is below 40, and this light is off.',
                'evidence_ids': [motion_evidence, 'device:sensor.ambient_lux', f'device:{light}']})
    if is_on(states['binary_sensor.water_leak']) and unwrap(states['switch.buzzer']) == 'off':
        actions.append({'domain': 'switch', 'service': 'turn_on', 'entity_id': 'switch.buzzer', 'data': {},
            'reason': 'The current leak sensor is active and the buzzer is off.',
            'evidence_ids': ['device:binary_sensor.water_leak', 'device:switch.buzzer']})
    summary = ('Turn on ' + ' and '.join(action['entity_id'] for action in actions) + '; current policy conditions are met.') if actions else (
        'Recorded current occupancy and sensor changes; no device action is required.' if observations else
        'No new observation or device action is required.')
    return {'summary': summary, 'observations': observations, 'actions': actions, 'noop': not (observations or actions)}


def build(source_root: Path, output: Path):
    data_root = source_root.parent.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy_dict = configured_policy()
    policy = Policy(**policy_dict)
    all_rows, context_sources, inputs = {}, [], {}
    media_hashes = {}
    def hash_media(path):
        if path not in media_hashes:
            media_hashes[path] = sha256_file(data_root / path)
        return media_hashes[path]
    for split in ('train', 'validation'):
        source_path = source_root / f'{split}.jsonl'
        sources = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
        assert all(row['split'] == split for row in sources)
        inputs[split] = {'path': str(source_path.relative_to(data_root)), 'sha256': sha256_file(source_path)}
        recordings = defaultdict(list)
        for row in sources:
            if row.get('task_type') == 'device_control':
                recordings[row['recording_id']].append(row)
        selected = sorted(recordings)
        if split == 'validation':
            # Stable one-recording-per-family budget, never selected by model results.
            families = {}
            for recording in selected:
                families.setdefault(recordings[recording][0]['group_id'], recording)
            selected = sorted(families.values())
        rows = []
        for recording in selected:
            parents = sorted(recordings[recording], key=lambda r: r['window']['started_at'])
            assert len(parents) == 4
            visual_negative = 2 if number(f'visual-negative:{recording}') % 2 else 3
            cases = CASES + [(1, 12, False, False, 1), (1, 39, False, False, 1),
                             (0, 12, False, False, 0),
                             (visual_negative, 40 if visual_negative == 2 else 12, False, False, 0)]
            context_modes = ['none', 'same', 'different', 'stale'] * 6
            random.Random(f'context-modes:{recording}').shuffle(context_modes)
            for case_index, (step, lux, leak, buzzer, action_count) in enumerate(cases):
                parent = parents[step]
                row_id = opaque_id(f'mixed-v2-current:{recording}:{case_index}')
                encoding = number(f'encoding:{row_id}') % 4
                start = max(r['window']['ended_at'] for r in parents) + 1000 + number(f'clock:{row_id}') % 100000 / 10
                audio_parent = parents[number(f'audio:{row_id}') % len(parents)]
                current = retime(parent['window'], row_id, start, audio_parent)
                current['device_states'] = authored_states(parent, lux, leak, buzzer, encoding)
                visual_only = case_index >= len(CASES)
                if visual_only:
                    current['device_states'] = {entity: value for entity, value in current['device_states'].items()
                                                if not entity.startswith('binary_sensor.motion_')}
                occupied = geometry(parent)
                # Context generation precedes target generation. "same" is an
                # explicitly authored unchanged prior snapshot, not a target leak.
                context_mode = context_modes[case_index]
                prior = {}
                runtime_context = None
                prior_id = None
                if context_mode != 'none':
                    prior_parent = parent if context_mode in ('same', 'stale') else parents[(step + 1) % 4]
                    prior_id = opaque_id(f'mixed-v2-prior:{row_id}')
                    prior_start = start - (420 if context_mode == 'stale' else 30)
                    previous = retime(prior_parent['window'], prior_id, prior_start, audio_parent)
                    if context_mode in ('same', 'stale'):
                        previous['device_states'] = authored_states(prior_parent, lux, leak, buzzer, encoding)
                    else:
                        previous['device_states'] = authored_states(prior_parent,
                            unwrap(prior_parent['window']['device_states']['sensor.ambient_lux']), False, False, 0)
                    prior = journal_state(previous, geometry(prior_parent), stale=context_mode == 'stale')
                    runtime_context = {'version': 1, 'state': prior, 'recent_events': [],
                        'source_window_end': previous['ended_at'], 'source_group_id': parent['group_id']}
                    context_sources.append({'id': prior_id, 'split': split, 'group_id': parent['group_id'],
                        'source': 'synthetic_authored_prior_v2', 'window': previous,
                        'authored_occupancy': geometry(prior_parent), 'annotation_method': 'authored_history_snapshot_v2',
                        'provenance': {'kind': 'synthetic', 'parent_row_id': prior_parent['id'],
                            'parent_source': inputs[split], 'context_mode': context_mode,
                            'media_reuse': 'unchanged source image bytes; telemetry and timestamps explicitly authored'}})
                target = target_for(current, occupied, prior)
                assert len(target['actions']) == action_count
                row = {'id': row_id, 'split': split, 'source': 'synthetic_authored_v2',
                    'group_id': parent['group_id'], 'recording_id': recording,
                    'task_type': 'visual_occupancy' if visual_only else 'device_control',
                    'annotation_method': 'synthetic_geometry_telemetry_and_causal_rules_v2',
                    'supervision_mask': FULL_MASK.copy(), 'window': current, 'target': target,
                    'provenance': {'kind': 'synthetic', 'generator': 'scripts/data_mixed_v2.py',
                        'rule_version': 'motion-occupancy-lux-leak-prior-v2', 'parent_row_id': parent['id'],
                        'parent_source': inputs[split], 'parent_audio_row_id': audio_parent['id'],
                        'context_source_row_id': prior_id, 'context_mode': context_mode,
                        'scenario_index': case_index, 'sensor_encoding': encoding,
                        'image_geometry': {'occupied': occupied, 'source_step': step},
                        'media': [{'path': item['path'], 'sha256': hash_media(item['path'])}
                                  for item in current['frames'] + current['audio']],
                        'authored_changes': ['device telemetry', 'capture timestamps', 'opaque evidence IDs',
                                             'historical context', 'same-family audio selection'],
                        'image_changes': 'none; all four original source images retained byte-for-byte',
                        'audio': 'reused generated tone/silence; no speech or real alarm label'}}
                if runtime_context:
                    row['runtime_context'] = runtime_context
                validate_row(row, policy)
                rows.append(row)
        for original in sources:
            if original.get('task_type') != 'natural_image_summary':
                continue
            row = copy.deepcopy(original)
            assert set(row['target']) == {'summary'} and row['supervision_mask'] == SUMMARY_MASK
            assert not row.get('runtime_context')
            rows.append(row)
        random.Random(f'mixed-v2-shuffle:{split}').shuffle(rows)
        all_rows[split] = rows
        write_jsonl(output / f'{split}.jsonl', rows)
    train_groups = {r['group_id'] for r in all_rows['train']}
    validation_groups = {r['group_id'] for r in all_rows['validation']}
    assert train_groups.isdisjoint(validation_groups)
    assert {r['id'] for r in all_rows['train']}.isdisjoint(r['id'] for r in all_rows['validation'])
    # Check reused camera bytes too, not just row IDs or family names.
    camera_hashes = {split: {hash_media(f['path']) for row in rows for f in row['window']['frames']}
                     for split, rows in all_rows.items()}
    assert camera_hashes['train'].isdisjoint(camera_hashes['validation']), 'cross-split camera image duplicate'
    dev = [r for r in all_rows['validation'] if r.get('provenance', {}).get('scenario_index') in (0, 3, 8, 10, 12, 16, 20, 22)]
    captions = [r for r in all_rows['validation'] if r['task_type'] == 'natural_image_summary']
    dev.extend(sorted(captions, key=lambda r: r['id'])[:8])
    write_jsonl(output / 'validation-dev.jsonl', dev)
    generation_validation = []
    # Fixed, model-result-independent selection: two examples per task/action
    # bucket, preferring one cold start and one causal context per bucket.
    for task in ('device_control', 'visual_occupancy'):
        for has_action in (True, False):
            bucket = sorted([r for r in all_rows['validation'] if r['task_type'] == task
                             and bool(r['target']['actions']) == has_action], key=lambda r: r['id'])
            cold = next(r for r in bucket if not r.get('runtime_context'))
            history = next(r for r in bucket if r.get('runtime_context') and
                           (has_action or r['target']['noop']))
            generation_validation.extend([cold, history])
    assert len({r['id'] for r in generation_validation}) == 8
    write_jsonl(output / 'generation-validation.jsonl', generation_validation)
    (output / 'generation-validation-selection.json').write_text(json.dumps({
        'source': 'validation.jsonl', 'selection': 'fixed task/action/history strata ordered by opaque row ID; historical quiet rows require true noop; no model outputs used',
        'test_rows_used': 0, 'rows': [{'id': r['id'], 'group_id': r['group_id'], 'task_type': r['task_type'],
            'action': bool(r['target']['actions']), 'context_mode': r['provenance']['context_mode']}
            for r in generation_validation]}, indent=2) + '\n')
    write_jsonl(output / 'context-sources.jsonl', context_sources)
    (output / 'policy.json').write_text(json.dumps(policy_dict, indent=2) + '\n')
    def stats(rows):
        authored = [r for r in rows if r['source'] == 'synthetic_authored_v2']
        return {'rows': len(rows), 'synthetic_decisions': len(authored),
            'action_windows': sum(bool(r['target']['actions']) for r in authored),
            'no_action_windows': sum(not r['target']['actions'] for r in authored),
            'noop_windows': sum(r['target']['noop'] for r in authored),
            'summary_only_rows': sum(r['task_type'] == 'natural_image_summary' for r in rows),
            'task_types': dict(Counter(r['task_type'] for r in rows)),
            'families': len({r['group_id'] for r in authored}),
            'context_modes': dict(Counter(r['provenance']['context_mode'] for r in authored)),
            'sensor_encodings': dict(Counter(r['provenance']['sensor_encoding'] for r in authored)),
            'action_entities': dict(Counter(a['entity_id'] for r in authored for a in r['target']['actions']))}
    manifest = {'version': 2, 'kind': 'mixed_synthetic_decisions_and_human_captions',
        'source_inputs': inputs, 'generator_sha256': sha256_file(__file__),
        'dataset_root': '..', 'policy_path': 'policy.json',
        'counts': {split: stats(rows) for split, rows in all_rows.items()}, 'validation_dev': stats(dev),
        'generation_validation': stats(generation_validation),
        'context_source_rows': len(context_sources), 'split_unit': 'original scenario family / COCO image ID',
        'integrity_checks': ['Decision/ObservationWindow schema', 'Policy.validate', 'authored exact action count',
            'prior timestamps strictly before current window', 'group and row ID disjoint', 'camera-byte hashes disjoint',
            'all reused media exist and are hashed', 'natural summary mask preserved'],
        'test_data': 'No test file was read, copied, relabeled, or modified by this preparation.',
        'limits': ['Synthetic decisions are authored, not human household annotations.',
            'Reused procedural images are not new visual examples; sensor/time/history variants provide decision supervision.',
            'Action balance is an intentional training distribution, not an estimate of household event frequency.',
            'No real audio, alarm, speech or long-duration reliability claims.',
            'Natural captions supervise summary only; their missing decisions are not silence labels.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text('''# Mixed v2 training data

496 training rows: 432 authored synthetic decisions (216 with actions, 216 without) plus 64 existing human COCO captions. The synthetic rows include 72 visual-only examples with no current motion telemetry. Validation contains 160 rows from separate source families/images; `validation-dev.jsonl` contains 56 of those for bounded training-time checks. `generation-validation.jsonl` is a fixed 8-row validation-only generation check: four action/four no-action rows, both device and visual tasks, four cold starts/four historical contexts. The adjacent selection JSON records exact unchanged validation row IDs; no model result or test row was used to choose them.

Run from the project root:

```sh
.venv/bin/python scripts/data_mixed_v2.py
```

Use `--dataset-root data`, `--policy data/mixed-v2/policy.json`, and train/validation JSONL paths under this directory. The script reads **only** v1 train and validation files. It does not read or change test sets.

All four camera images are reused byte-for-byte from their recorded procedural source scenes. Occupied, empty, bright and already-lit geometry stays consistent with the authored telemetry. These are extra decision examples, not new natural visuals. Audio is same-family generated tone/silence selected independently of the action class. No audio alarm or speech labels are introduced.

`context-sources.jsonl` stores the authored earlier windows used for context. Rows identify their exact parent source IDs and input-file hashes. `runtime_context` is versioned and contains Journal-shaped history with timestamps strictly earlier than the current window. Targets are computed only after history is authored. Each source recording has exactly six cold-start, six unchanged-history, six changed-history and six stale-history examples. Cold-start rows have no runtime context and label all supported current facts. Other targets omit unchanged facts. Current evidence controls action preconditions.

The policy defines binary sensor equivalence (on/true/JSON true and off/false/JSON false), wrapped Home Assistant states, numeric lux strings, the strict below-40 threshold, light/buzzer already-on cases and changed-fact recording. The 50/50 action balance is a training choice, not a household prevalence claim. Natural captions keep summary-only masks and no runtime context.

`manifest.json` records counts, integrity checks and limitations. No v2 evaluation result is implied by preparation.
''')
    return manifest


def validate_row(row, policy):
    window = ObservationWindow.model_validate(row['window'])
    target = Decision.model_validate(row['target'])
    assert not policy.validate(target, window)
    context = row.get('runtime_context')
    if context:
        assert context['source_group_id'] == row['group_id']
        assert context['source_window_end'] < window.started_at
        for entity in context['state'].values():
            for fact in entity.values():
                assert fact['observed_at'] <= context['source_window_end']
        assert context['recent_events'] == []
    assert target.noop == (not target.actions and not target.observations)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=project / 'data' / 'mixed')
    parser.add_argument('--output', type=Path, default=project / 'data' / 'mixed-v2')
    args = parser.parse_args()
    result = build(args.source.resolve(), args.output.resolve())
    print(json.dumps({'output': str(args.output.resolve()), 'counts': result['counts'],
                      'validation_dev': result['validation_dev'], 'context_source_rows': result['context_source_rows']}, indent=2))


if __name__ == '__main__':
    main()

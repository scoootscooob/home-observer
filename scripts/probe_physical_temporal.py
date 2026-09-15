#!/usr/bin/env python3
"""Four preregistered pixel-only counterfactual calls; never manufacture event labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

from home_observer.model import ModelConfig, resolve_media_path
from home_observer.physical_model import physical_context, physical_request_from_row
from home_observer.physical_vllm import PhysicalVLLMModel

SELECTED_IDS = ('epic_030c003244925eb728dd', 'epic_01659d9083e47dac8d2f')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def derive(row, variant, config):
    request = physical_request_from_row(row)
    frames = [f for c in request['window']['clips'] for f in c['frames']]
    original = copy.deepcopy(frames)
    assert len(frames) == 8
    indices = list(reversed(range(8))) if variant == 'reverse_pixels' else [0] * 8
    correspondence = []
    for index, source_index in enumerate(indices):
        current, source = frames[index], original[source_index]
        original_path = resolve_media_path(original[index]['path'], config.dataset_root, config.max_media_bytes)
        pixel_path = resolve_media_path(source['path'], config.dataset_root, config.max_media_bytes)
        original_hash, pixel_hash = sha(original_path), sha(pixel_path)
        current['path'] = source['path']
        correspondence.append({'evidence_id': current['evidence_id'], 'timestamp': current['timestamp'],
            'pixel_source_evidence_id': source['evidence_id'], 'pixel_source_path': source['path'],
            'original_pixel_sha256': original_hash, 'input_pixel_sha256': pixel_hash,
            'pixel_bytes_changed': original_hash != pixel_hash})
    assert physical_context(request) == physical_context(physical_request_from_row(row))
    return request, correspondence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run', type=Path, default=Path('artifacts/physical-guided-adapted-test'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/physical-temporal-probe'))
    parser.add_argument('--budget-seconds', type=float, default=180)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 180:
        parser.error('budget must be positive and no greater than180seconds')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite temporal diagnostic')
    manifest = json.loads((args.original_run / 'manifest.json').read_text())
    report = json.loads((args.original_run / 'report.json').read_text())
    assert manifest['complete'] and manifest['dataset_complete'] and report['complete']
    assert manifest['backend_config']['schema_contract'] == 'physical_events_first_bounded_v3'
    assert manifest['backend_config']['served_model'] == 'physical-trained'
    assert sha(args.original_run / 'predictions.jsonl') == manifest['predictions_sha256']
    rows = [json.loads(line) for line in (args.original_run / 'predictions.jsonl').read_text().splitlines()]
    lookup = {row['id']: row for row in rows}
    assert next(row['id'] for row in rows if row['target']['events'][0]['kind'] == 'open'
                and row['target']['events'][0]['object_label'] == 'cupboard') == SELECTED_IDS[1]
    config = ModelConfig(**manifest['model_config'])
    source_root = Path(__file__).resolve().parents[1]
    for name, expected in manifest['implementation_sha256'].items():
        assert sha(source_root / name) == expected, name
    prepared = []
    for identifier in SELECTED_IDS:
        for variant in ('reverse_pixels', 'static_first_frame'):
            request, pixels = derive(lookup[identifier], variant, config)
            prepared.append({'source_id': identifier, 'variant': variant, 'request': request,
                             'frames': pixels, 'new_ground_truth': None})
    args.output.mkdir(parents=True)
    declaration = {'experiment': 'derived_temporal_sensitivity_diagnostic', 'primary_reports_unchanged': True,
        'new_human_labels': False, 'false_events_per_hour': None,
        'selection': 'Two clips named by root before probe: nested bowls and first source open-cupboard row',
        'selected_ids': list(SELECTED_IDS), 'expected_new_calls': 4, 'budget_seconds': args.budget_seconds,
        'original_run_manifest_sha256': sha(args.original_run / 'manifest.json'),
        'original_predictions_sha256': manifest['predictions_sha256'],
        'probe_script_sha256': sha(Path(__file__)), 'model_config': manifest['model_config'],
        'backend_config': manifest['backend_config'], 'prepared_counterfactuals': prepared,
        'source_annotations_apply_only_to_original_pixels': True, 'complete': False}
    save(args.output / 'manifest.json', declaration)
    save(args.output / 'originals.json', {identifier: lookup[identifier] for identifier in SELECTED_IDS})
    model = PhysicalVLLMModel(config, base_url=manifest['backend_config']['engine_url'],
                              served_model='physical-trained', schema_constraints=True)
    started = time.monotonic()
    results = []
    try:
        with (args.output / 'predictions.jsonl').open('w') as stream:
            for item in prepared:
                remaining = args.budget_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    break
                model.timeout = remaining
                model.last_generation = None
                result = {}
                begin = time.monotonic()
                try:
                    result = model.observe(item['request'])
                except Exception as exc:
                    result = {'error': f'{type(exc).__name__}: {exc}', **(model.last_generation or {})}
                original_metrics = lookup[item['source_id']]['result'].get('metrics', {})
                metrics = result.get('metrics', {})
                parity = {key: metrics.get(key) == original_metrics.get(key) for key in
                          ('context_sha256', 'schema_sha256', 'input_tokens', 'frame_evidence_ids')}
                row = {**item, 'result': result, 'metadata_parity_with_original': parity,
                       'elapsed_s': time.monotonic() - begin}
                results.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({'completed': len(results), 'source_id': item['source_id'],
                                  'variant': item['variant'], 'valid': 'error' not in result,
                                  'elapsed_s': row['elapsed_s']}), flush=True)
    finally:
        model.close()
    declaration.update(complete=len(results) == 4, completed_calls=len(results),
                       wall_s=time.monotonic() - started, predictions_sha256=sha(args.output / 'predictions.jsonl'))
    save(args.output / 'manifest.json', declaration)
    save(args.output / 'report.json', {'complete': declaration['complete'], 'completed_calls': len(results),
        'strictly_valid': sum('error' not in r['result'] for r in results),
        'wall_s': declaration['wall_s'], 'new_human_labels': False, 'interpretation':
        'Changing output shows sensitivity, not correctness. Static-frame movement claims need separate review; '
        'these are derived counterfactuals, not real continuous negatives or a false-events/hour estimate.'})


if __name__ == '__main__':
    main()

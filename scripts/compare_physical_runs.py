#!/usr/bin/env python3
"""Verify paired guided runs; preserve primary scores and write a separate label audit."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from home_observer.physical_evaluate import temporal_iou


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_at(root):
    manifest, report = read_json(root / 'manifest.json'), read_json(root / 'report.json')
    assert manifest['complete'] and report['complete'] and manifest['dataset_complete']
    assert sha(root / 'predictions.jsonl') == manifest['predictions_sha256']
    assert sha(root / manifest['raw_outputs_file']) == manifest['raw_outputs_sha256']
    rows = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == manifest['expected_rows'] == report['windows'] == 16
    return manifest, report, rows


def compare_pair(base_root, adapted_root, *, base_alias, adapted_alias):
    bm, br, base = rows_at(base_root)
    am, ar, adapted = rows_at(adapted_root)
    for key in ('dataset_sha256', 'selected_ids', 'prompt_sha256', 'media_sha256', 'implementation_sha256',
                'generation', 'versions', 'source_rows', 'selection_limit'):
        assert bm[key] == am[key], key
    for key, value in bm['model_config'].items():
        if key != 'adapter_path':
            assert value == am['model_config'][key], ('model_config', key)
    assert bm['model_config']['adapter_path'] is None and am['model_config']['adapter_path']
    assert not bm['adapter_sha256'] and am['adapter_sha256']
    for key, value in bm['backend_config'].items():
        if key != 'served_model':
            assert value == am['backend_config'][key], ('backend_config', key)
    assert bm['backend_config']['served_model'] == base_alias
    assert am['backend_config']['served_model'] == adapted_alias
    checks = []
    for b, a in zip(base, adapted, strict=True):
        for key in ('id', 'window', 'target', 'supervision_mask', 'annotation_coverage'):
            assert b.get(key) == a.get(key), (b['id'], key)
        bmetrics, ametrics = b['result'].get('metrics', {}), a['result'].get('metrics', {})
        # Fail comparison if a transport failure left no measurable model inputs.
        for key in ('system_prompt_sha256', 'context_sha256', 'schema_sha256', 'frame_evidence_ids',
                    'sampled_frames', 'input_tokens', 'schema_contract', 'decision_field_order'):
            assert key in bmetrics and bmetrics[key] == ametrics.get(key), (b['id'], key)
        assert bmetrics['engine_reported_model'] == base_alias
        assert ametrics['engine_reported_model'] == adapted_alias
        checks.append({'id': b['id'], 'model_visible_input_and_schema_match': True})
    return {'matched_configuration': True, 'rows': len(base), 'per_row_checks': checks,
            'base_primary_report_sha256': sha(base_root / 'report.json'),
            'adapted_primary_report_sha256': sha(adapted_root / 'report.json'),
            'base': br, 'adapted': ar, 'adapter_sha256': am['adapter_sha256']}, base, adapted


def class_map(path):
    result = {}
    for row in csv.DictReader(path.open()):
        for term in ast.literal_eval(row['instances']):
            key = ' '.join(term.strip().lower().split())
            result.setdefault(key, []).append({'id': int(row['id']), 'key': row['key']})
    return result


def mapped(term, classes):
    return classes.get(' '.join((term or '').strip().lower().split()), [])


def event_audit(row, verbs, nouns):
    target = row['target']['events'][0]
    strict = not row['result'].get('error')
    decision = row['result'].get('decision')
    raw_parseable = True
    if decision is None:
        try:
            decision = json.loads(row.get('raw_output') or '')
        except (ValueError, TypeError):
            decision, raw_parseable = {}, False
    if not isinstance(decision, dict):
        decision = {}
    items = []
    raw_events = decision.get('events', [])
    for event in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(event, dict):
            continue
        verb, noun = mapped(event.get('kind'), verbs), mapped(event.get('object_label'), nouns)
        target_verb, target_noun = mapped(target['kind'], verbs), mapped(target['object_label'], nouns)
        same_verb = bool({r['id'] for r in verb} & {r['id'] for r in target_verb})
        same_noun = bool({r['id'] for r in noun} & {r['id'] for r in target_noun})
        items.append({**event, 'temporal_iou_with_label': temporal_iou(event, target),
                      'official_verb_classes': verb, 'official_noun_classes': noun,
                      'official_class_pair_matches': same_verb and same_noun,
                      'exact_source_label_pair_matches': (
                          ' '.join(str(event.get('kind', '')).lower().split()) == ' '.join(target['kind'].lower().split())
                          and ' '.join(str(event.get('object_label', '')).lower().split()) == ' '.join(target['object_label'].lower().split())),
                      'eligible_as_accepted_output': strict})
    return {'strict_valid': strict, 'error': row['result'].get('error'),
            'raw_json_parseable': raw_parseable, 'rejected_raw_used_only_for_diagnosis': not strict,
            'summary': decision.get('summary'), 'events': items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--base-alias', default='physical-base')
    parser.add_argument('--adapted-alias', default='physical-trained')
    args = parser.parse_args()
    root = args.artifacts
    results = {'complete': True, 'primary_reports_unchanged': True, 'same_consumer_configuration': True,
               'allowed_differences': ['adapter_path', 'adapter_bytes', 'served_model_alias'],
               'expected_aliases': {'base': args.base_alias, 'adapted': args.adapted_alias}, 'splits': {}}
    all_test = None
    for split in ('validation', 'test'):
        result, base, adapted = compare_pair(root / f'physical-guided-base-{split}',
                                             root / f'physical-guided-adapted-{split}',
                                             base_alias=args.base_alias, adapted_alias=args.adapted_alias)
        results['splits'][split] = result
        if split == 'test':
            all_test = base, adapted
    (root / 'physical-guided-comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    directory = root / 'physical-semantic-audit'
    sources = directory / 'sources'
    verbs = class_map(sources / 'EPIC_100_verb_classes.csv')
    nouns = class_map(sources / 'EPIC_100_noun_classes.csv')
    entries = []
    for b, a in zip(*all_test, strict=True):
        entries.append({'id': b['id'], 'source_target': b['target'],
                        'base': event_audit(b, verbs, nouns), 'adapted': event_audit(a, verbs, nouns)})
    audit = {'reviewer': 'assistant', 'source_annotations': 'original human EPIC annotations',
             'scope': 'All 16 pilot test rows; secondary post-run label audit, not a replacement score',
             'primary_metric_unchanged': True,
             'official_mapping': 'Exact membership in official EPIC instances lists; no stemming or additional synonyms',
             'source_manifest': str(sources / 'manifest.json'),
             'rejected_raw_policy': 'Shown only for diagnosis; never treated as accepted production predictions',
             'interpretation': 'Class equivalence alone does not prove visual grounding or adequate interval localization',
             'rows': entries}
    audit['unvalidated_label_content_counts'] = {}
    for variant in ('base', 'adapted'):
        values = [entry[variant] for entry in entries]
        audit['unvalidated_label_content_counts'][variant] = {
            'source_rows': len(values),
            'raw_json_parseable_rows': sum(v['raw_json_parseable'] for v in values),
            'strictly_valid_rows': sum(v['strict_valid'] for v in values),
            'rows_with_exact_source_labels_and_iou_at_least_0_5': sum(
                any(e['exact_source_label_pair_matches'] and e['temporal_iou_with_label'] >= .5
                    for e in value['events']) for value in values),
            'rows_with_official_class_pair_and_iou_at_least_0_5': sum(
                any(e['official_class_pair_matches'] and e['temporal_iou_with_label'] >= .5
                    for e in value['events']) for value in values),
            'not_a_production_acceptance_metric': True,
        }
    (directory / 'paired-label-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# Paired physical interaction label audit', '',
             'Secondary analysis of all 16 test clips. Original primary scores are unchanged. '
             'IoU measures time-interval overlap; ≥0.5 is the primary localization threshold. '
             'Rejected raw output is shown for diagnosis only.', '',
             '| Row | Source interaction | Base events (IoU) | Adapted events (IoU) | Strict base/adapted |',
             '|---|---|---|---|---|']
    def events(value):
        return '<br>'.join(f"{e.get('kind')} / {e.get('object_label')} ({e['temporal_iou_with_label']:.3f})"
                          for e in value['events']) or 'None'
    for index, item in enumerate(entries, 1):
        target = item['source_target']['events'][0]
        cells = [str(index), f"{target['kind']} / {target['object_label']}", events(item['base']),
                 events(item['adapted']), f"{item['base']['strict_valid']} / {item['adapted']['strict_valid']}"]
        lines.append('| ' + ' | '.join(cell.replace('|', '\\|') for cell in cells) + ' |')
    lines += ['', 'Full row IDs, descriptions, timestamps, errors, and official class memberships '
              'are retained in `paired-label-audit.json`. Assistant semantic review, if added, '
              'are separate from both the primary metric and the official exact-membership mapping.', '']
    (directory / 'paired-label-audit.md').write_text('\n'.join(lines))
    print(json.dumps({'configuration_matches': True, 'paired_rows': 32, 'test_audit_rows': len(entries)}))


if __name__ == '__main__':
    main()

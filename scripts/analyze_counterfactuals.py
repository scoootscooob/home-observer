#!/usr/bin/env python3
"""Verify paired visual input changes and score the unchanged counterfactual labels."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from home_observer.cli import load_policy
from home_observer.evaluate import action_set, observation_set
from home_observer.prompts import build_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='artifacts/release-v1/counterfactual/predictions.jsonl')
    parser.add_argument('--output', default='reports/counterfactual-pairs.json')
    parser.add_argument('--policy', default='data/mixed-v2/policy.json')
    args = parser.parse_args()
    source = Path('data/counterfactuals/pairs.jsonl')
    predictions = {row['id']: row for row in map(json.loads, Path(args.predictions).read_text().splitlines())}
    groups = defaultdict(list)
    policy = load_policy(args.policy).as_dict()
    for row in map(json.loads, source.read_text().splitlines()):
        groups[row['pair_id']].append(row)
    reports = []
    def digest(path):
        return hashlib.sha256((source.parent / path).read_bytes()).hexdigest()
    for pair, rows in groups.items():
        assert len(rows) == 2
        requests = [dict(window=row['window'], state={}, recent_events=[], policy=policy) for row in rows]
        identical_text = build_context(requests[0]) == build_context(requests[1])
        identical_audio = [digest(a['path']) for a in rows[0]['window']['audio']] == [digest(a['path']) for a in rows[1]['window']['audio']]
        changed = [a['camera_id'] for a, b in zip(rows[0]['window']['frames'], rows[1]['window']['frames']) if digest(a['path']) != digest(b['path'])]
        members = []
        def occupancy(observations):
            return observation_set([item for item in observations
                                    if item['entity_id'].startswith('room.') and item['attribute'] == 'occupied'])
        for row in rows:
            result = predictions.get(row['id'], {}).get('result', {})
            decision = result.get('decision') or {}
            valid = bool(decision) and not result.get('error') and not result.get('rejections')
            members.append({'id': row['id'], 'member': row['counterfactual_member'], 'valid': valid,
                'exact_occupancy': valid and occupancy(decision.get('observations', [])) == occupancy(row['target']['observations']),
                'strict_all_observations_match': valid and observation_set(decision.get('observations', [])) == observation_set(row['target']['observations']),
                'exact_actions': valid and action_set(decision.get('actions', [])) == action_set(row['target']['actions'])})
        reports.append({'pair_id': pair, 'identical_model_visible_text': identical_text,
            'identical_audio': identical_audio, 'changed_cameras': changed, 'members': members,
            'passed': identical_text and identical_audio and len(changed) == 1 and all(m['exact_occupancy'] and m['exact_actions'] for m in members)})
    report = {'pairs': len(reports), 'passed_pairs': sum(r['passed'] for r in reports), 'results': reports,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'policy_sha256': hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
              'predictions_sha256': hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
              'scope': 'Authored visual counterfactuals, identical text/audio within each pair; no real-home generalization claim.'}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['pairs', 'passed_pairs']}))


if __name__ == '__main__':
    main()

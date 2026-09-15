#!/usr/bin/env python3
"""Score fixed labeled windows and explicit causal contexts through the cloud API."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from home_observer.cli import load_policy
from home_observer.engine import RemoteBackend
from home_observer.evaluate import summarize
from home_observer.schema import InferenceResponse, ObservationWindow
from home_observer.train import request_from_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/mixed-v2/generation-validation.jsonl')
    parser.add_argument('--policy', default='data/mixed-v2/policy.json')
    parser.add_argument('--remote-data-root', default='/workspace/home-observer/data')
    parser.add_argument('--endpoint', default='http://127.0.0.1:18080')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'predictions.jsonl').exists():
        raise ValueError('Use a new output directory')
    backend = RemoteBackend(args.endpoint, os.environ.get('HOME_OBSERVER_API_TOKEN', ''))
    policy = load_policy(args.policy)
    results = []
    health = backend.client.get(args.endpoint.rstrip('/') + '/health').json()
    for row in map(json.loads, Path(args.input).read_text().splitlines()):
        request = request_from_row(row, policy.as_dict())
        for media in request['window']['frames'] + request['window']['audio']:
            media['path'] = str(Path(args.remote_data_root) / media['path'])
        start = time.perf_counter()
        try:
            response = InferenceResponse.model_validate(backend.observe(request))
            result = {'decision': response.decision.model_dump(), 'metrics': response.metrics,
                      'rejections': policy.validate(response.decision, ObservationWindow.model_validate(request['window']))}
        except Exception as exc:
            result = {'error': f'{type(exc).__name__}: {exc}'}
        result['total_latency_s'] = time.perf_counter() - start
        item = {'id': row['id'], 'task_type': row.get('task_type'), 'target': row['target'],
                'supervision_mask': row.get('supervision_mask', {}), 'result': result, 'request': request}
        results.append(item)
        with (output / 'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n')
        print(json.dumps({'count': len(results), 'error': result.get('error'), 'latency_s': result['total_latency_s']}), flush=True)
    report = summarize(results)
    report['engine_health'] = health
    report['input_sha256'] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    report['policy_sha256'] = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    report['context_mode'] = 'fixed authored causal contexts; independent requests, no action execution'
    report['by_task_type'] = {kind: summarize([r for r in results if r['task_type'] == kind]) for kind in sorted({r['task_type'] for r in results})}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    backend.client.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

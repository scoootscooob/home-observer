#!/usr/bin/env python3
"""Exercise the Runpod model through this project's isolated, real HA demo API."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from home_observer.cli import load_policy
from home_observer.engine import Observer, RemoteBackend
from home_observer.evaluate import action_set
from home_observer.ha import HomeAssistant
from home_observer.journal import Journal
from home_observer.schema import Action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18080')
    parser.add_argument('--output', default='reports/learned-home-assistant')
    parser.add_argument('--remote-data-root', default='/workspace/home-observer/data')
    parser.add_argument('--policy', default='data/fixtures-expanded/policy.json')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('owned_ha_demo', project / 'scripts/ha_demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    credentials = demo.demo_credentials(demo.DEFAULT_WORK)  # Checks exact owned container label.
    executor = HomeAssistant(demo.BASE_URL, credentials['long_lived_token'], execute_real=True)
    backend = RemoteBackend(args.endpoint, os.environ.get('HOME_OBSERVER_API_TOKEN', ''))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise ValueError('Use a new output directory for a new verification run')
    rows = [json.loads(line) for line in (project / 'data/mixed/test.jsonl').read_text().splitlines()]
    light_case = next(row for row in rows if row['task_type'] == 'device_control' and row['target']['actions'])
    quiet_case = next(row for row in rows if row['task_type'] == 'device_control' and not row['target']['actions'])
    leak_case = json.loads(json.dumps(quiet_case))
    leak_case['id'] += '-integration-only-leak'
    leak_case['window']['device_states'].update({'binary_sensor.water_leak': 'on', 'switch.buzzer': 'off'})
    leak_case['target']['actions'] = [{'domain': 'switch', 'service': 'turn_on', 'entity_id': 'switch.buzzer',
        'data': {}, 'reason': 'Configured leak rule', 'evidence_ids': ['device:binary_sensor.water_leak', 'device:switch.buzzer']}]
    selected = [light_case, leak_case, quiet_case]
    records = []
    policy_path = Path(args.policy)
    policy = load_policy(policy_path)
    health = backend.client.get(args.endpoint.rstrip('/') + '/health').json()
    for index, original in enumerate(selected):
        row = json.loads(json.dumps(original))
        # Install authored test conditions in the owned demo, then read them back.
        for entity, value in row['window']['device_states'].items():
            state = value.get('state') if isinstance(value, dict) else value
            if entity.startswith(('light.', 'switch.')):
                executor.execute(Action(domain=entity.split('.')[0], service='turn_' + str(state),
                    entity_id=entity, reason='Reset isolated test condition', evidence_ids=['test_setup']), f'setup-{index}-{entity}')
            else:
                response = executor.client.post('/api/states/' + entity, json={'state': str(state)})
                response.raise_for_status()
        readback = {}
        for entity in row['window']['device_states']:
            response = executor.client.get('/api/states/' + entity)
            response.raise_for_status()
            readback[entity] = response.json()['state']
        row['window']['device_states'] = readback
        for item in row['window']['frames'] + row['window']['audio']:
            item['path'] = str(Path(args.remote_data_root) / item['path'])
        journal = Journal(output / f'case-{index}.sqlite')
        observer = Observer(backend, journal, policy, executor)
        result = observer.process(row['window'])
        duplicate = observer.process(row['window'])
        proposed = (result.get('decision') or {}).get('actions', [])
        exact = action_set(proposed) == action_set(row['target']['actions'])
        verified = all(a['status'] == 'complete' and a['result']['verified'] for a in result['actions'])
        passed = exact and verified and not result.get('error') and not result['rejections'] and bool(duplicate.get('skipped'))
        records.append({'id': row['id'], 'authored_test_conditions': True, 'device_readback': readback,
                        'expected_actions': row['target']['actions'], 'result': result,
                        'duplicate_replay_skipped': bool(duplicate.get('skipped')), 'passed': passed})
        journal.close()
        (output / 'report.json').write_text(json.dumps({'passed': all(r['passed'] for r in records),
            'engine_health': health, 'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            'scope': 'Learned cloud inference and actual isolated Home Assistant REST execution; synthetic test conditions.',
            'cases': records}, indent=2))
    executor.close()
    backend.client.close()
    print(json.dumps({'passed': all(r['passed'] for r in records), 'cases': len(records)}))
    return 0 if all(r['passed'] for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())

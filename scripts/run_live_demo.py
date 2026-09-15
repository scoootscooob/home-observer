#!/usr/bin/env python3
"""Run public-media capture against the project's isolated Home Assistant demo."""
import argparse
import importlib.util
import os
from pathlib import Path

from home_observer.live import main as live_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000')
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--sources', default='configs/live-public-demo.json')
    parser.add_argument('--audio', default='data/public-egolife/raw/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4')
    parser.add_argument('--api-token-file', help='Private local inference-token file; otherwise use environment')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if args.api_token_file:
        os.environ['HOME_OBSERVER_API_TOKEN'] = Path(args.api_token_file).read_text().strip()
    spec = importlib.util.spec_from_file_location('owned_ha_demo', project / 'scripts/ha_demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    credentials = demo.demo_credentials(demo.DEFAULT_WORK)
    os.environ['HOME_ASSISTANT_TOKEN'] = credentials['long_lived_token']
    os.chdir(project)
    return live_main([
        '--sources', args.sources,
        '--audio', args.audio,
        '--backend', 'remote', '--endpoint', args.endpoint,
        '--policy', 'data/mixed-v2/policy.json', '--execute-ha', '--ha-url', demo.BASE_URL,
        '--output', str(output), '--duration', str(args.duration),
        '--capture-fps', '2', '--interval', '1', '--retain-windows', '8', '--loop',
    ])


if __name__ == '__main__':
    raise SystemExit(main())

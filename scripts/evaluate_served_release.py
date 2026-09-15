#!/usr/bin/env python3
"""Run complete release suites against the already-running native multimodal vLLM engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from evaluate_release import prepare_suite, read_rows

from home_observer.cli import load_policy
from home_observer.evaluate import evaluate
from home_observer.model import ModelConfig
from home_observer.prompts import SYSTEM_PROMPT
from home_observer.vllm_backend import VLLMModel


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LoggedServedModel(VLLMModel):
    """Log each sequential request while retaining the production vLLM validator.

    The native release logger has a less strict parser. Calling its observe method
    would bypass this backend's required-field and configured-entity checks.
    """

    def __init__(self, config, *, raw_log, **kwargs):
        super().__init__(config, **kwargs)
        self.raw_log = Path(raw_log)
        self._pending_log = None

    def generate_text(self, request):
        raw, metrics = super().generate_text(request)
        if self._pending_log is not None:
            self._pending_log.update(raw_output=raw, metrics=metrics)
        if metrics.get('engine_reported_model') != self.served_model:
            raise ValueError('Completion model alias does not match the requested engine model')
        return raw, metrics

    def observe(self, request):
        if self._pending_log is not None:
            raise RuntimeError('Served release logging requires sequential requests')
        entry = {'window_id': request['window']['window_id'], 'valid_decision': False}
        self._pending_log = entry
        try:
            result = super().observe(request)
            entry['valid_decision'] = True
            return result
        except BaseException as exc:
            entry['error'] = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            self._pending_log = None
            with self.raw_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--policy', default='data/mixed-v2/policy.json')
    parser.add_argument('--adapter', help='Deployment provenance; alias must already be registered in the engine')
    parser.add_argument('--engine-endpoint', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--engine-model', default='observer')
    parser.add_argument('--decision-field-order', choices=['summary-first', 'actions-first'], default='actions-first')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_root).resolve(strict=True)
    policy = load_policy(args.policy)
    config = ModelConfig(dataset_root=str(data), adapter_path=args.adapter, quantization='none',
                         max_new_tokens=768, max_batch_size=1, merge_adapter=False)
    specifications = [
        ('heldout', data / 'mixed/test.jsonl', data, 'isolated_snapshot'),
        ('counterfactual', data / 'counterfactuals/pairs.jsonl', data / 'counterfactuals', 'isolated_snapshot'),
        ('public_video', data / 'public-egolife/replay.jsonl', data / 'public-egolife', 'chronological_unscored'),
    ]
    suites, prepared = [], []
    for name, source, media_root, mode in specifications:
        source_rows = read_rows(source)
        rows = prepare_suite(source_rows, media_root, data, policy, context_mode=mode, config=config)
        path = output / (name + '-inputs.jsonl')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        suites.append({'name': name, 'source': str(source), 'source_sha256': sha256(source),
                       'source_rows': len(source_rows), 'selected_rows': len(rows), 'context_mode': mode,
                       'targets_changed': False, 'window_ids_changed': False})
        prepared.append((name, path, mode))
    deployment = {'backend': 'vllm', 'served_model': args.engine_model,
                  'adapter_execution': 'vllm_native_lora' if args.adapter else 'base_model',
                  'peft_runtime': False, 'adapter_merged': False,
                  'decision_field_order': args.decision_field_order, 'schema_constraints': True,
                  'schema_contract': VLLMModel.schema_contract, 'brightness_enabled': False,
                  'sequential_requests': True, 'identity_source': 'deployment_configuration_and_completion_metadata'}
    manifest = {'model_config': asdict(config), 'suites': suites, 'deployment': deployment,
                'policy': policy.as_dict(), 'policy_file_sha256': sha256(args.policy),
                'policy_sha256': hashlib.sha256(json.dumps(policy.as_dict(), sort_keys=True).encode()).hexdigest(),
                'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                'adapter_sha256': sha256(Path(args.adapter) / 'adapter_model.safetensors') if args.adapter else None,
                'implementation_sha256': {str(path): sha256(path) for path in
                    [Path(__file__), Path('scripts/evaluate_release.py'), Path('src/home_observer/prompts.py'),
                     Path('src/home_observer/vllm_backend.py'), Path('src/home_observer/engine.py')]}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps({'prepared': True, 'gpu_executed': False, 'suites': suites}))
        return
    model = LoggedServedModel(config, raw_log=output / 'engine-outputs.jsonl',
                      base_url=args.engine_endpoint, served_model=args.engine_model,
                      decision_field_order=args.decision_field_order, allow_brightness=False)
    try:
        if not model.ready():
            raise RuntimeError('vLLM engine is not ready')
        reports = {}
        for name, path, mode in prepared:
            print(json.dumps({'starting_suite': name}), flush=True)
            reports[name] = evaluate(path, output / name, model, policy)
            reports[name]['context_mode'] = mode
            reports[name]['accuracy_interpretation'] = (
                'Independent annotated snapshots; empty prior observation history.' if mode == 'isolated_snapshot'
                else 'Chronological public replay without ground truth; no perception accuracy score.')
            (output / 'report.json').write_text(json.dumps(
                {'model_config': asdict(config), 'deployment': deployment, 'suites': reports,
                 'complete': len(reports) == len(specifications)}, indent=2) + '\n')
            print(json.dumps({'completed_suite': name, 'windows': reports[name]['windows'],
                              'invalid_or_failed_windows': reports[name]['invalid_or_failed_windows']}), flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()

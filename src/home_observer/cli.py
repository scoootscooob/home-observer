"""Local fixture validation, GPU-backed evaluation, and event-journal retrieval."""
import argparse
import json
import os
from pathlib import Path

from .engine import FixtureBackend, RemoteBackend
from .evaluate import evaluate
from .journal import Journal
from .policy import Policy, default_policy


def load_policy(path):
    if not path:
        return default_policy()
    data = json.loads(Path(path).read_text())
    allowed = set(Policy.__dataclass_fields__)
    return Policy(**{k: v for k, v in data.items() if k in allowed})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--data", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--policy")
    ev.add_argument("--backend", choices=["fixture", "native", "remote"], default="remote")
    ev.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ev.add_argument("--token-env", default="HOME_OBSERVER_API_TOKEN")
    ev.add_argument("--dataset-root", default=".")
    ev.add_argument("--preloaded-media", action="store_true",
                    help="Remote paths already exist on the GPU; default uploads local files under dataset-root")
    ev.add_argument("--model", default="google/gemma-4-E4B-it")
    ev.add_argument("--adapter")
    ev.add_argument('--merge-adapter', action='store_true', help='Explicit experimental BF16 merge')
    ev.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--task-type")
    ev.add_argument("--realtime", action="store_true")
    se = commands.add_parser("search")
    se.add_argument("--journal", required=True)
    se.add_argument("--query", required=True)
    se.add_argument("--before", required=True, type=float)
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    if args.command == "search":
        journal = Journal(args.journal)
        print(json.dumps(journal.search(args.query, args.before), indent=2))
        journal.close()
        return 0
    if args.backend == "fixture":
        backend = FixtureBackend()
    elif args.backend == "remote":
        backend = RemoteBackend(args.endpoint, os.environ.get(args.token_env, ""),
                                upload_media=not args.preloaded_media,
                                media_root=None if args.preloaded_media else args.dataset_root)
    else:
        from .model import ModelConfig, NativeModel
        backend = NativeModel(ModelConfig(model_id=args.model, adapter_path=args.adapter,
                                           merge_adapter=args.merge_adapter,
                                           dataset_root=args.dataset_root, quantization=args.quantization))
    try:
        result = evaluate(args.data, args.output, backend, load_policy(args.policy), args.limit,
                          task_type=args.task_type, realtime=args.realtime)
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    print(json.dumps(result, indent=2))
    return 0 if result["invalid_or_failed_windows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

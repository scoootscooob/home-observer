#!/usr/bin/env python3
"""Run a saved inference request on the native model, without starting HTTP."""
import argparse
import json
from pathlib import Path

from home_observer.model import DEFAULT_MODEL, ModelConfig, NativeModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="JSON InferenceRequest (server-local media paths)")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--output")
    args = parser.parse_args()
    backend = NativeModel(ModelConfig(model_id=args.model, adapter_path=args.adapter,
                                     dataset_root=args.dataset_root, quantization=args.quantization))
    result = backend.observe(json.loads(Path(args.request).read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

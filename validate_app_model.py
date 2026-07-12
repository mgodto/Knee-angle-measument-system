#!/usr/bin/env python3

"""Fail-fast validation for the model artifact used by the desktop app build."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from knee_model_runtime import create_model_adapter, load_app_config, model_spec_with_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Knee X-ray app checkpoint and print its identity.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    config = load_app_config(args.config)
    spec = config.model
    if args.checkpoint is not None:
        spec = model_spec_with_checkpoint(spec, args.checkpoint)
    adapter = create_model_adapter(spec)
    info = adapter.load()
    print(json.dumps(asdict(info), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Create a deterministic, non-clinical image for release smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20260720)
    height, width = 1024, 256
    base = rng.normal(110.0, 45.0, (height, width)).clip(0, 255).astype(np.uint8)
    gradient = np.linspace(35.0, -35.0, height, dtype=np.float32)[:, None]
    image = np.clip(base.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise SystemExit(f"failed to write smoke fixture: {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

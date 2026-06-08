#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from knee_dataset_utils import annotation_keypoints, load_manifest, read_json
from measure_angles import ANNOTATION_LINE_NAMES, ANNOTATION_POINT_NAMES, measure_from_named_points


def validate_manifest(manifest_path: Path) -> tuple[list[str], Counter]:
    rows = load_manifest(manifest_path)
    errors: list[str] = []
    counters: Counter = Counter()

    for row in rows:
        sample_id = row["sample_id"]
        annotation_path = Path(row["annotation_path"])
        raw_path = Path(row["raw_path"])
        counters["samples"] += 1
        counters[f"side_{row['side']}"] += 1
        counters[f"case_{row['case_id']}"] += 1

        if not annotation_path.exists():
            errors.append(f"{sample_id}: missing annotation file: {annotation_path}")
            continue
        if not raw_path.exists():
            errors.append(f"{sample_id}: missing raw image file: {raw_path}")
            continue

        annotation = read_json(annotation_path)
        raw_image = cv2.imread(str(raw_path))
        if raw_image is None:
            errors.append(f"{sample_id}: cannot read raw image: {raw_path}")
            continue

        image_h, image_w = raw_image.shape[:2]
        if int(annotation["image_width"]) != image_w or int(annotation["image_height"]) != image_h:
            errors.append(
                f"{sample_id}: annotation size {annotation['image_width']}x{annotation['image_height']} "
                f"does not match raw image {image_w}x{image_h}"
            )

        missing_points = [name for name in ANNOTATION_POINT_NAMES if name not in annotation.get("points", {})]
        missing_lines = [name for name in ANNOTATION_LINE_NAMES if name not in annotation.get("lines", {})]
        if missing_points:
            errors.append(f"{sample_id}: missing points: {', '.join(missing_points)}")
        if missing_lines:
            errors.append(f"{sample_id}: missing lines: {', '.join(missing_lines)}")
        if row["side"] not in {"L", "R"}:
            errors.append(f"{sample_id}: invalid side: {row['side']}")

        try:
            keypoints = annotation_keypoints(annotation)
        except Exception as exc:
            errors.append(f"{sample_id}: invalid keypoint schema: {exc}")
            continue
        for name, (x, y) in keypoints.items():
            if not (0 <= x < image_w and 0 <= y < image_h):
                errors.append(f"{sample_id}: {name} out of bounds: ({x:.1f}, {y:.1f}) for {image_w}x{image_h}")

        try:
            measure_from_named_points(
                raw_image,
                annotation["points"],
                raw_path=raw_path,
                named_lines=annotation.get("lines"),
                side=row["side"],
            )
        except Exception as exc:
            errors.append(f"{sample_id}: measurement failed: {exc}")

    return errors, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate resolved knee X-ray training samples.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/knee_dataset_manifest.csv"))
    args = parser.parse_args()

    errors, counters = validate_manifest(args.manifest)
    print(f"Samples: {counters['samples']}")
    print(f"Sides: L={counters['side_L']}, R={counters['side_R']}")
    print(f"Cases: {len([key for key in counters if key.startswith('case_')])}")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()

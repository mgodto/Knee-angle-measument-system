#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2

from knee_dataset_utils import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_RAW_ROOTS,
    annotation_sample_id,
    build_raw_image_index,
    extract_case_id,
    read_json,
    resolve_raw_candidate,
)
from measure_angles import measure_from_named_points


MANIFEST_FIELDS = [
    "sample_id",
    "case_id",
    "side",
    "annotation_path",
    "raw_path",
    "raw_filename",
    "image_width",
    "image_height",
    "raw_match_count",
    "mldfa",
    "mpta",
]


def build_manifest(annotation_dir: Path, raw_roots: list[Path]) -> list[dict[str, str]]:
    index = build_raw_image_index(raw_roots)
    rows: list[dict[str, str]] = []
    for annotation_path in sorted(annotation_dir.glob("*_annotation.json")):
        annotation = read_json(annotation_path)
        raw_candidate, raw_match_count = resolve_raw_candidate(annotation_path, annotation, index)
        raw_image = cv2.imread(str(raw_candidate.path))
        if raw_image is None:
            raise ValueError(f"Cannot read resolved raw image: {raw_candidate.path}")
        result, _debug = measure_from_named_points(
            raw_image,
            annotation["points"],
            raw_path=raw_candidate.path,
            named_lines=annotation.get("lines"),
            side=annotation.get("side"),
        )
        mldfa = float(result["mldfa_angle"])
        mpta = float(result["mpta_angle"])
        if not math.isfinite(mldfa) or not math.isfinite(mpta):
            raise ValueError(f"Non-finite angle in {annotation_path}")

        sample_id = annotation_sample_id(annotation_path)
        rows.append(
            {
                "sample_id": sample_id,
                "case_id": extract_case_id(sample_id),
                "side": str(annotation.get("side", "")),
                "annotation_path": str(annotation_path),
                "raw_path": str(raw_candidate.path),
                "raw_filename": str(annotation.get("raw_filename", raw_candidate.path.name)),
                "image_width": str(annotation["image_width"]),
                "image_height": str(annotation["image_height"]),
                "raw_match_count": str(raw_match_count),
                "mldfa": f"{mldfa:.6f}",
                "mpta": f"{mpta:.6f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leg-level training manifest from knee annotation JSON files.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--raw-root", type=Path, nargs="*", default=list(DEFAULT_RAW_ROOTS))
    parser.add_argument("--output", type=Path, default=Path("outputs/knee_dataset_manifest.csv"))
    args = parser.parse_args()

    rows = build_manifest(args.annotation_dir, args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ambiguous = sum(1 for row in rows if int(row["raw_match_count"]) > 1)
    print(f"Wrote {len(rows)} samples: {args.output}")
    print(f"Ambiguous raw matches resolved by heuristic: {ambiguous}")


if __name__ == "__main__":
    main()

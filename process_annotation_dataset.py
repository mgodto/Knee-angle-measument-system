#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from build_dataset_manifest import MANIFEST_FIELDS
from knee_dataset_utils import extract_case_id, load_manifest, read_json
from measure_angles import (
    render_annotation_line_image,
    render_annotation_point_image,
    measure_from_named_points,
)


DEFAULT_INPUT_MANIFEST = Path("outputs/knee_dataset_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("images/annotation_processed")
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int
    method: str

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def is_double_leg_sample(sample_id: str) -> bool:
    return "RL" in sample_id.upper()


def relative_to_cwd(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def collect_annotation_xy(annotation: dict) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for point in annotation.get("points", {}).values():
        xs.append(float(point["x"]))
        ys.append(float(point["y"]))
    for line in annotation.get("lines", {}).values():
        for point in line.values():
            xs.append(float(point["x"]))
            ys.append(float(point["y"]))
    if not xs or not ys:
        raise ValueError("Annotation has no points or lines to crop from")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def build_pair_stats(rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    paired: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        sample_id = row["sample_id"]
        if not is_double_leg_sample(sample_id):
            continue
        annotation = read_json(Path(row["annotation_path"]))
        xs, _ys = collect_annotation_xy(annotation)
        paired[row["raw_path"]].append(
            {
                "sample_id": sample_id,
                "side": row.get("side", ""),
                "median_x": float(np.median(xs)),
                "min_x": float(np.min(xs)),
                "max_x": float(np.max(xs)),
            }
        )
    return paired


def choose_crop_box(
    row: dict[str, str],
    annotation: dict,
    image_width: int,
    image_height: int,
    pair_stats: dict[str, list[dict[str, object]]],
) -> CropBox:
    sample_id = row["sample_id"]
    if not is_double_leg_sample(sample_id):
        return CropBox(0, 0, image_width, image_height, "already_single_leg")

    xs, _ys = collect_annotation_xy(annotation)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    target_median = float(np.median(xs))
    bbox_pad = max(180, int(round(image_width * 0.08)))
    boundary_pad = max(80, int(round(image_width * 0.035)))

    paired = pair_stats.get(row["raw_path"], [])
    other_medians = [
        float(item["median_x"])
        for item in paired
        if item.get("sample_id") != sample_id
    ]
    if other_medians:
        other_median = float(np.median(other_medians))
        separator = (target_median + other_median) / 2.0
        if target_median > other_median:
            desired_x0 = int(math.floor(separator - boundary_pad))
            required_x0 = int(math.floor(min_x - bbox_pad))
            x0 = max(0, min(desired_x0, required_x0))
            x1 = image_width
        else:
            x0 = 0
            desired_x1 = int(math.ceil(separator + boundary_pad))
            required_x1 = int(math.ceil(max_x + bbox_pad))
            x1 = min(image_width, max(desired_x1, required_x1))
        method = "paired_rl_horizontal_crop"
    else:
        x0 = max(0, int(math.floor(min_x - bbox_pad)))
        x1 = min(image_width, int(math.ceil(max_x + bbox_pad)))
        method = "bbox_rl_horizontal_crop"

    if x1 <= x0:
        raise ValueError(f"Invalid crop for {sample_id}: x0={x0}, x1={x1}")
    return CropBox(x0, 0, x1, image_height, method)


def shifted_point(point: dict, crop: CropBox) -> dict[str, float]:
    return {
        "x": float(point["x"]) - crop.x0,
        "y": float(point["y"]) - crop.y0,
    }


def adjust_annotation(
    annotation: dict,
    crop: CropBox,
    raw_output_path: Path,
    source_annotation_path: Path,
    source_raw_path: Path,
) -> dict:
    adjusted = copy.deepcopy(annotation)
    for name, point in adjusted.get("points", {}).items():
        adjusted["points"][name] = shifted_point(point, crop)
    for line_name, line in adjusted.get("lines", {}).items():
        for endpoint, point in line.items():
            adjusted["lines"][line_name][endpoint] = shifted_point(point, crop)

    adjusted["raw_path"] = relative_to_cwd(raw_output_path)
    adjusted["raw_filename"] = raw_output_path.name
    adjusted["image_width"] = crop.width
    adjusted["image_height"] = crop.height
    adjusted["processed_from"] = {
        "annotation_path": relative_to_cwd(source_annotation_path),
        "raw_path": relative_to_cwd(source_raw_path),
        "original_image_width": int(annotation["image_width"]),
        "original_image_height": int(annotation["image_height"]),
        "crop": {
            "x0": crop.x0,
            "y0": crop.y0,
            "x1": crop.x1,
            "y1": crop.y1,
            "width": crop.width,
            "height": crop.height,
            "method": crop.method,
        },
    }
    return adjusted


def assert_annotation_in_bounds(sample_id: str, annotation: dict) -> None:
    image_width = int(annotation["image_width"])
    image_height = int(annotation["image_height"])
    xs, ys = collect_annotation_xy(annotation)
    out_of_bounds = (
        np.any(xs < 0)
        or np.any(xs >= image_width)
        or np.any(ys < 0)
        or np.any(ys >= image_height)
    )
    if out_of_bounds:
        raise ValueError(
            f"{sample_id}: adjusted coordinates are out of bounds for "
            f"{image_width}x{image_height}"
        )


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, JPEG_PARAMS):
        raise ValueError(f"Failed to write image: {path}")


def render_processed_images(
    sample_id: str,
    raw_image: np.ndarray,
    annotation: dict,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    point_path = output_dir / f"{sample_id}_point.jpg"
    line_path = output_dir / f"{sample_id}_line.jpg"
    combined_path = output_dir / f"{sample_id}_combined.jpg"

    point_image = render_annotation_point_image(raw_image, annotation["points"])
    line_image = render_annotation_line_image(raw_image, annotation["points"], annotation.get("lines"))
    result, _debug = measure_from_named_points(
        raw_image,
        annotation["points"],
        raw_path=Path(annotation["raw_path"]),
        named_lines=annotation.get("lines"),
        side=annotation.get("side"),
    )

    write_image(point_path, point_image)
    write_image(line_path, line_image)
    write_image(combined_path, result["combined_image"])
    return point_path, line_path, combined_path


def process_dataset(input_manifest: Path, output_dir: Path, render_overlays: bool) -> list[dict[str, str]]:
    rows = load_manifest(input_manifest)
    pair_stats = build_pair_stats(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_rows: list[dict[str, str]] = []
    crop_rows: list[dict[str, str]] = []
    for row in rows:
        sample_id = row["sample_id"]
        annotation_path = Path(row["annotation_path"])
        raw_path = Path(row["raw_path"])
        annotation = read_json(annotation_path)
        raw_image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw_image is None:
            raise ValueError(f"{sample_id}: cannot read raw image: {raw_path}")
        image_height, image_width = raw_image.shape[:2]
        crop = choose_crop_box(row, annotation, image_width, image_height, pair_stats)

        raw_output_path = output_dir / f"{sample_id}_raw.jpg"
        annotation_output_path = output_dir / f"{sample_id}_annotation.json"
        if crop.x0 == 0 and crop.y0 == 0 and crop.x1 == image_width and crop.y1 == image_height:
            shutil.copy2(raw_path, raw_output_path)
            cropped_image = raw_image
        else:
            cropped_image = raw_image[crop.y0:crop.y1, crop.x0:crop.x1]
            write_image(raw_output_path, cropped_image)

        adjusted = adjust_annotation(
            annotation,
            crop,
            raw_output_path,
            annotation_path,
            raw_path,
        )
        assert_annotation_in_bounds(sample_id, adjusted)
        with annotation_output_path.open("w", encoding="utf-8") as handle:
            json.dump(adjusted, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        result, _debug = measure_from_named_points(
            cropped_image,
            adjusted["points"],
            raw_path=raw_output_path,
            named_lines=adjusted.get("lines"),
            side=adjusted.get("side"),
        )

        if render_overlays:
            render_processed_images(sample_id, cropped_image, adjusted, output_dir)

        processed_row = {
            "sample_id": sample_id,
            "case_id": extract_case_id(sample_id),
            "side": str(adjusted.get("side", "")),
            "annotation_path": relative_to_cwd(annotation_output_path),
            "raw_path": relative_to_cwd(raw_output_path),
            "raw_filename": raw_output_path.name,
            "image_width": str(crop.width),
            "image_height": str(crop.height),
            "raw_match_count": "1",
            "mldfa": f"{float(result['mldfa_angle']):.6f}",
            "mpta": f"{float(result['mpta_angle']):.6f}",
            "source_annotation_path": relative_to_cwd(annotation_path),
            "source_raw_path": relative_to_cwd(raw_path),
            "crop_x0": str(crop.x0),
            "crop_y0": str(crop.y0),
            "crop_x1": str(crop.x1),
            "crop_y1": str(crop.y1),
            "crop_width": str(crop.width),
            "crop_height": str(crop.height),
            "crop_method": crop.method,
            "is_cropped": str(crop.method != "already_single_leg"),
        }
        processed_rows.append(processed_row)
        crop_rows.append(
            {
                "sample_id": sample_id,
                "side": processed_row["side"],
                "source_raw_path": processed_row["source_raw_path"],
                "raw_path": processed_row["raw_path"],
                "source_width": str(image_width),
                "source_height": str(image_height),
                "crop_x0": str(crop.x0),
                "crop_y0": str(crop.y0),
                "crop_x1": str(crop.x1),
                "crop_y1": str(crop.y1),
                "crop_width": str(crop.width),
                "crop_height": str(crop.height),
                "crop_method": crop.method,
            }
        )

    manifest_fields = [
        *MANIFEST_FIELDS,
        "source_annotation_path",
        "source_raw_path",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "crop_width",
        "crop_height",
        "crop_method",
        "is_cropped",
    ]
    manifest_path = output_dir / "processed_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(processed_rows)

    summary_path = output_dir / "crop_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(crop_rows)

    return processed_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a single-leg processed knee annotation dataset from the current manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Only write raw crops, adjusted JSON, and manifests.",
    )
    args = parser.parse_args()

    rows = process_dataset(args.manifest, args.output_dir, render_overlays=not args.no_overlays)
    cropped = sum(1 for row in rows if row["is_cropped"] == "True")
    print(f"Wrote {len(rows)} processed samples to {args.output_dir}")
    print(f"Cropped double-leg samples: {cropped}")
    print(f"Processed manifest: {args.output_dir / 'processed_manifest.csv'}")
    print(f"Crop summary: {args.output_dir / 'crop_summary.csv'}")


if __name__ == "__main__":
    main()

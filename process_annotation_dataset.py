#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
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
INFERENCE_ROI_COORDINATE_SPACE = "source_image_pixels"
BROAD_IMAGE_MIN_WIDTH_HEIGHT_RATIO = 0.60
ANNOTATION_MAX_HORIZONTAL_SPAN_FRACTION = 0.45
ANNOTATION_SIDE_SEPARATION_FRACTION = 0.025
ANNOTATION_CENTERED_CROP_WIDTH_FRACTION = 0.60


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int
    method: str
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def is_double_leg_sample(sample_id: str) -> bool:
    return "RL" in sample_id.upper()


def paired_samples(row: dict[str, str], pair_stats: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    sample_id = row["sample_id"]
    return [
        item
        for item in pair_stats.get(row["raw_path"], [])
        if item.get("sample_id") != sample_id
    ]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_input_scope(annotation: dict) -> str:
    source = annotation.get("source")
    if not isinstance(source, dict):
        return ""
    return str(source.get("input_scope", "")).strip().lower()


def _exact_integer(value: object) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("ROI coordinates must be finite integers")
    return int(number)


def confirmed_inference_roi(
    annotation: dict,
    image_width: int,
    image_height: int,
) -> tuple[CropBox | None, str]:
    analysis = annotation.get("analysis")
    roi = analysis.get("inference_roi") if isinstance(analysis, dict) else None
    if roi is None:
        return None, "not_present"
    if not isinstance(roi, dict):
        return None, "not_object"
    if roi.get("confirmed") is not True:
        return None, "not_confirmed"
    if str(roi.get("coordinate_space", "")) != INFERENCE_ROI_COORDINATE_SPACE:
        return None, "invalid_coordinate_space"

    try:
        x0 = _exact_integer(roi["x0"])
        y0 = _exact_integer(roi["y0"])
        x1 = _exact_integer(roi["x1"])
        y1 = _exact_integer(roi["y1"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_coordinates"
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        return None, "out_of_bounds"
    try:
        if "width" in roi and _exact_integer(roi["width"]) != x1 - x0:
            return None, "width_mismatch"
        if "height" in roi and _exact_integer(roi["height"]) != y1 - y0:
            return None, "height_mismatch"
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_size"

    xs, ys = collect_annotation_xy(annotation)
    if not (
        np.all(np.isfinite(xs))
        and np.all(np.isfinite(ys))
        and np.all(xs >= x0)
        and np.all(xs < x1)
        and np.all(ys >= y0)
        and np.all(ys < y1)
    ):
        return None, "does_not_cover_annotation"
    if (
        source_input_scope(annotation) == "bilateral raster x-ray"
        and (x1 - x0) >= image_width * 0.80
    ):
        return None, "bilateral_roi_too_wide"

    return (
        CropBox(
            x0,
            y0,
            x1,
            y1,
            "measurement_inference_roi",
            {
                "provenance_source": "analysis.inference_roi",
                "coordinate_space": INFERENCE_ROI_COORDINATE_SPACE,
                "selection_method": str(roi.get("selection_method", "")),
                "confirmed": True,
                "inference_roi_status": "accepted",
            },
        ),
        "accepted",
    )


def build_pair_stats(rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    paired: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        annotation = read_json(Path(row["annotation_path"]))
        xs, _ys = collect_annotation_xy(annotation)
        paired[row["raw_path"]].append(
            {
                "sample_id": row["sample_id"],
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
    previous = _valid_previous_processing(annotation)
    previous_is_nonidentity = False
    if previous is not None:
        processed_from, previous_crop = previous
        previous_is_nonidentity = (
            _exact_integer(previous_crop["x0"]) != 0
            or _exact_integer(previous_crop["y0"]) != 0
            or _exact_integer(previous_crop["x1"])
            != _exact_integer(processed_from["original_image_width"])
            or _exact_integer(previous_crop["y1"])
            != _exact_integer(processed_from["original_image_height"])
        )
    if (
        previous is not None
        and previous_is_nonidentity
        and str(previous[1].get("method", "")) != "already_single_leg"
    ):
        return CropBox(
            0,
            0,
            image_width,
            image_height,
            "already_single_leg",
            {
                "provenance_source": "existing_processed_crop",
                "coordinate_space": "current_training_image_pixels",
                "selection_method": "preserve_existing_target_leg_crop",
                "confirmed": bool(previous[1].get("confirmed", False)),
                "inference_roi_status": "already_processed",
            },
        )
    inference_crop, inference_roi_status = confirmed_inference_roi(
        annotation,
        image_width,
        image_height,
    )
    if inference_crop is not None:
        return inference_crop

    xs, _ys = collect_annotation_xy(annotation)
    if not np.all(np.isfinite(xs)):
        raise ValueError(f"{sample_id}: annotation has non-finite horizontal coordinates")
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    target_median = float(np.median(xs))
    if min_x < 0 or max_x >= image_width:
        raise ValueError(f"{sample_id}: annotation horizontal coordinates are out of bounds")
    if max_x - min_x > image_width * ANNOTATION_MAX_HORIZONTAL_SPAN_FRACTION:
        raise ValueError(
            f"{sample_id}: annotation horizontal span is too wide for a safe target-leg crop"
        )
    bbox_pad = max(180, int(round(image_width * 0.08)))
    boundary_pad = max(80, int(round(image_width * 0.035)))

    other_medians = [float(item["median_x"]) for item in paired_samples(row, pair_stats)]
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
        method = "paired_horizontal_crop"
    else:
        declared_bilateral = source_input_scope(annotation) == "bilateral raster x-ray"
        broad_image = image_width / image_height >= BROAD_IMAGE_MIN_WIDTH_HEIGHT_RATIO
        if not (is_double_leg_sample(sample_id) or declared_bilateral or broad_image):
            return CropBox(
                0,
                0,
                image_width,
                image_height,
                "already_single_leg",
                {
                    "provenance_source": "source_image_geometry",
                    "coordinate_space": INFERENCE_ROI_COORDINATE_SPACE,
                    "selection_method": "narrow_image_no_crop",
                    "confirmed": False,
                    "inference_roi_status": inference_roi_status,
                },
            )

        image_center = image_width / 2.0
        side_separation = image_width * ANNOTATION_SIDE_SEPARATION_FRACTION
        if target_median < image_center - side_separation:
            x0 = 0
            desired_x1 = int(math.ceil(image_center + boundary_pad))
            required_x1 = int(math.ceil(max_x + bbox_pad))
            x1 = min(image_width, max(desired_x1, required_x1))
            placement = "image_left"
        elif target_median > image_center + side_separation:
            desired_x0 = int(math.floor(image_center - boundary_pad))
            required_x0 = int(math.floor(min_x - bbox_pad))
            x0 = max(0, min(desired_x0, required_x0))
            x1 = image_width
            placement = "image_right"
        else:
            crop_width = max(
                int(math.ceil(image_width * ANNOTATION_CENTERED_CROP_WIDTH_FRACTION)),
                int(math.ceil(max_x - min_x + 2 * bbox_pad)),
            )
            crop_width = min(image_width, crop_width)
            x0 = int(math.floor(target_median - crop_width / 2.0))
            x1 = x0 + crop_width
            x0 = min(x0, int(math.floor(min_x - bbox_pad)))
            x1 = max(x1, int(math.ceil(max_x + bbox_pad)))
            if x0 < 0:
                x0 = 0
            if x1 > image_width:
                x1 = image_width
            placement = "centered"
        method = (
            "bbox_horizontal_crop"
            if is_double_leg_sample(sample_id)
            else "annotation_bbox_horizontal_crop"
        )

        if min_x - x0 < min(bbox_pad, min_x) - 1 or x1 - max_x < min(
            bbox_pad,
            image_width - max_x,
        ) - 1:
            raise ValueError(f"{sample_id}: target-leg fallback crop does not preserve annotation padding")

    if x1 <= x0:
        raise ValueError(f"Invalid crop for {sample_id}: x0={x0}, x1={x1}")
    provenance: dict[str, object] = {
        "provenance_source": "paired_annotations" if other_medians else "annotation_bbox",
        "coordinate_space": INFERENCE_ROI_COORDINATE_SPACE,
        "selection_method": method,
        "confirmed": False,
        "inference_roi_status": inference_roi_status,
        "annotation_bbox_padding_pixels": bbox_pad,
    }
    if not other_medians:
        provenance["target_placement"] = placement
    return CropBox(x0, 0, x1, image_height, method, provenance)


def shifted_point(point: dict, crop: CropBox) -> dict[str, float]:
    return {
        "x": float(point["x"]) - crop.x0,
        "y": float(point["y"]) - crop.y0,
    }


def _valid_previous_processing(annotation: dict) -> tuple[dict, dict] | None:
    processed_from = annotation.get("processed_from")
    if processed_from is None:
        return None
    if not isinstance(processed_from, dict) or not isinstance(processed_from.get("crop"), dict):
        raise ValueError("Existing processed_from provenance is malformed")
    previous_crop = processed_from["crop"]
    try:
        x0 = _exact_integer(previous_crop["x0"])
        y0 = _exact_integer(previous_crop["y0"])
        x1 = _exact_integer(previous_crop["x1"])
        y1 = _exact_integer(previous_crop["y1"])
        crop_width = _exact_integer(previous_crop["width"])
        crop_height = _exact_integer(previous_crop["height"])
        original_width = _exact_integer(processed_from["original_image_width"])
        original_height = _exact_integer(processed_from["original_image_height"])
        current_width = _exact_integer(annotation["image_width"])
        current_height = _exact_integer(annotation["image_height"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Existing processed_from provenance is incomplete") from exc
    if not (
        0 <= x0 < x1 <= original_width
        and 0 <= y0 < y1 <= original_height
        and x1 - x0 == crop_width == current_width
        and y1 - y0 == crop_height == current_height
    ):
        raise ValueError("Existing processed_from crop does not match the current training image")
    return copy.deepcopy(processed_from), copy.deepcopy(previous_crop)


def _hash_if_file(value: object) -> str:
    path = Path(str(value or ""))
    return sha256_file(path) if path.is_file() else ""


def _declared_source_sha256(annotation: dict) -> str:
    source = annotation.get("source")
    value = str(source.get("sha256", "")).strip().lower() if isinstance(source, dict) else ""
    valid = len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
    return value if valid else ""


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
    local_crop = {
        "x0": crop.x0,
        "y0": crop.y0,
        "x1": crop.x1,
        "y1": crop.y1,
        "width": crop.width,
        "height": crop.height,
        "method": crop.method,
        **crop.provenance,
    }
    previous = _valid_previous_processing(annotation)
    local_is_identity = (
        crop.x0 == 0
        and crop.y0 == 0
        and crop.x1 == int(annotation["image_width"])
        and crop.y1 == int(annotation["image_height"])
    )
    if previous is None:
        adjusted["processed_from"] = {
            "annotation_path": relative_to_cwd(source_annotation_path),
            "raw_path": relative_to_cwd(source_raw_path),
            "source_annotation_sha256": sha256_file(source_annotation_path),
            "source_raw_sha256": sha256_file(source_raw_path),
            "original_image_width": int(annotation["image_width"]),
            "original_image_height": int(annotation["image_height"]),
            "crop": local_crop,
        }
        return adjusted

    processed_from, previous_crop = previous
    if local_is_identity:
        processed_from.setdefault(
            "source_annotation_sha256",
            _hash_if_file(processed_from.get("annotation_path")),
        )
        processed_from.setdefault(
            "source_raw_sha256",
            _declared_source_sha256(annotation)
            or _hash_if_file(processed_from.get("raw_path")),
        )
        adjusted["processed_from"] = processed_from
        return adjusted

    absolute_crop = {
        **local_crop,
        "x0": int(previous_crop["x0"]) + crop.x0,
        "y0": int(previous_crop["y0"]) + crop.y0,
        "x1": int(previous_crop["x0"]) + crop.x1,
        "y1": int(previous_crop["y0"]) + crop.y1,
        "provenance_source": "composed_crop",
        "previous_crop_method": str(previous_crop.get("method", "")),
    }
    processing_steps = list(processed_from.get("processing_steps", []))
    if not processing_steps:
        processing_steps.append({"crop": previous_crop})
    processing_steps.append(
        {
            "input_raw_path": relative_to_cwd(source_raw_path),
            "input_raw_sha256": sha256_file(source_raw_path),
            "crop": local_crop,
        }
    )
    processed_from["crop"] = absolute_crop
    processed_from["processing_steps"] = processing_steps
    processed_from.setdefault(
        "source_annotation_sha256",
        _hash_if_file(processed_from.get("annotation_path")),
    )
    processed_from.setdefault(
        "source_raw_sha256",
        _declared_source_sha256(annotation)
        or _hash_if_file(processed_from.get("raw_path")),
    )
    adjusted["processed_from"] = processed_from
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


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        processed_from = adjusted["processed_from"]
        final_crop = processed_from["crop"]
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
            "case_id": row.get("case_id") or extract_case_id(sample_id),
            "source_case_id": row.get("source_case_id", ""),
            "source_dataset": row.get("source_dataset", ""),
            "dataset_group": row.get("dataset_group", ""),
            "implant_status": row.get("implant_status", ""),
            "study_phase": row.get("study_phase", ""),
            "side": str(adjusted.get("side", "")),
            "annotation_path": relative_to_cwd(annotation_output_path),
            "raw_path": relative_to_cwd(raw_output_path),
            "raw_filename": raw_output_path.name,
            "image_width": str(crop.width),
            "image_height": str(crop.height),
            "raw_match_count": "1",
            "mldfa": f"{float(result['mldfa_angle']):.6f}",
            "mpta": f"{float(result['mpta_angle']):.6f}",
            "source_annotation_path": str(processed_from.get("annotation_path", "")),
            "source_raw_path": str(processed_from.get("raw_path", "")),
            "source_annotation_sha256": str(processed_from.get("source_annotation_sha256", "")),
            "source_raw_sha256": str(processed_from.get("source_raw_sha256", "")),
            "crop_x0": str(final_crop["x0"]),
            "crop_y0": str(final_crop["y0"]),
            "crop_x1": str(final_crop["x1"]),
            "crop_y1": str(final_crop["y1"]),
            "crop_width": str(final_crop["width"]),
            "crop_height": str(final_crop["height"]),
            "crop_method": str(final_crop.get("method", crop.method)),
            "crop_provenance_source": str(final_crop.get("provenance_source", "")),
            "crop_selection_method": str(final_crop.get("selection_method", "")),
            "crop_confirmed": str(bool(final_crop.get("confirmed", False))),
            "inference_roi_status": str(final_crop.get("inference_roi_status", "")),
            "is_cropped": str(
                int(final_crop["x0"]) != 0
                or int(final_crop["y0"]) != 0
                or int(final_crop["x1"]) != int(processed_from["original_image_width"])
                or int(final_crop["y1"]) != int(processed_from["original_image_height"])
            ),
        }
        processed_rows.append(processed_row)
        crop_rows.append(
            {
                "sample_id": sample_id,
                "side": processed_row["side"],
                "source_raw_path": processed_row["source_raw_path"],
                "raw_path": processed_row["raw_path"],
                "source_width": str(processed_from["original_image_width"]),
                "source_height": str(processed_from["original_image_height"]),
                "crop_x0": processed_row["crop_x0"],
                "crop_y0": processed_row["crop_y0"],
                "crop_x1": processed_row["crop_x1"],
                "crop_y1": processed_row["crop_y1"],
                "crop_width": processed_row["crop_width"],
                "crop_height": processed_row["crop_height"],
                "crop_method": processed_row["crop_method"],
                "crop_provenance_source": processed_row["crop_provenance_source"],
                "crop_selection_method": processed_row["crop_selection_method"],
                "crop_confirmed": processed_row["crop_confirmed"],
                "inference_roi_status": processed_row["inference_roi_status"],
            }
        )

    manifest_fields = [
        *MANIFEST_FIELDS,
        "source_case_id",
        "source_annotation_path",
        "source_raw_path",
        "source_annotation_sha256",
        "source_raw_sha256",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "crop_width",
        "crop_height",
        "crop_method",
        "crop_provenance_source",
        "crop_selection_method",
        "crop_confirmed",
        "inference_roi_status",
        "is_cropped",
    ]
    manifest_path = output_dir / "processed_manifest.csv"
    write_rows(manifest_path, manifest_fields, processed_rows)

    if any(row.get("implant_status") for row in processed_rows):
        confirmed_bone_rows = [
            row for row in processed_rows if row.get("implant_status") == "bone"
        ]
        confirmed_tka_rows = [
            row for row in processed_rows if row.get("implant_status") == "TKA"
        ]
        unknown_rows = [
            row for row in processed_rows if row.get("implant_status") == "unknown"
        ]
        bone_tka_rows = [
            row for row in processed_rows if row.get("implant_status") in {"bone", "TKA"}
        ]
        bone_or_legacy_unknown_rows = [
            row
            for row in processed_rows
            if row.get("implant_status") == "bone"
            or (
                row.get("implant_status") == "unknown"
                and row.get("dataset_group") == "legacy"
            )
        ]
        write_rows(
            output_dir / "processed_manifest_bone.csv",
            manifest_fields,
            confirmed_bone_rows,
        )
        write_rows(
            output_dir / "processed_manifest_tka.csv",
            manifest_fields,
            confirmed_tka_rows,
        )
        write_rows(
            output_dir / "processed_manifest_unknown.csv",
            manifest_fields,
            unknown_rows,
        )
        write_rows(
            output_dir / "processed_manifest_bone_tka.csv",
            manifest_fields,
            bone_tka_rows,
        )
        write_rows(
            output_dir / "processed_manifest_bone_or_legacy_unknown.csv",
            manifest_fields,
            bone_or_legacy_unknown_rows,
        )

    summary_path = output_dir / "crop_summary.csv"
    write_rows(summary_path, list(crop_rows[0].keys()), crop_rows)

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
    print(f"Cropped target-leg samples: {cropped}")
    print(f"Processed manifest: {args.output_dir / 'processed_manifest.csv'}")
    if any(row.get("implant_status") for row in rows):
        print(f"Confirmed bone manifest: {args.output_dir / 'processed_manifest_bone.csv'}")
        print(f"Confirmed TKA manifest: {args.output_dir / 'processed_manifest_tka.csv'}")
        print(f"Unknown implant-status manifest: {args.output_dir / 'processed_manifest_unknown.csv'}")
        print(f"Confirmed Bone + TKA manifest: {args.output_dir / 'processed_manifest_bone_tka.csv'}")
        print(f"Bone + legacy unknown manifest: {args.output_dir / 'processed_manifest_bone_or_legacy_unknown.csv'}")
    print(f"Crop summary: {args.output_dir / 'crop_summary.csv'}")


if __name__ == "__main__":
    main()

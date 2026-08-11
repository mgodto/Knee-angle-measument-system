#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2

from knee_xray.core.measure_angles import ANNOTATION_LINE_NAMES, ANNOTATION_POINT_NAMES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_ANNOTATION_DIR = Path("images/Knee_Xray_annotations")
DEFAULT_RAW_ROOTS = (
    Path("images/line_point"),
    Path("images/raw_line_point"),
    Path("images/point_only"),
)


@dataclass(frozen=True)
class RawCandidate:
    path: Path
    width: int
    height: int


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_case_id(name: str) -> str:
    match = re.search(r"\d+", Path(name).stem)
    if match:
        return match.group(0)
    return Path(name).stem


def annotation_sample_id(annotation_path: Path) -> str:
    stem = annotation_path.stem
    return stem.removesuffix("_annotation")


def build_raw_image_index(raw_roots: list[Path] | tuple[Path, ...]) -> dict[str, list[RawCandidate]]:
    index: dict[str, list[RawCandidate]] = {}
    for root in raw_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            height, width = image.shape[:2]
            index.setdefault(path.name, []).append(RawCandidate(path=path, width=width, height=height))
    return index


def common_prefix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    count = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        count += 1
    return count


def normalized_path_parts(value: object) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value)).replace("\\", "/").lower()
    return tuple(part for part in text.split("/") if part)


def reference_basename(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def common_suffix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    count = 0
    for left_part, right_part in zip(reversed(left), reversed(right)):
        if left_part != right_part:
            break
        count += 1
    return count


def raw_reference_score(candidate: RawCandidate, annotation: dict | None) -> int:
    if annotation is None:
        return 0
    candidate_parts = normalized_path_parts(candidate.path.as_posix())
    scores = [0]
    for reference in (annotation.get("raw_path"), annotation.get("source_raw_path")):
        if not reference:
            continue
        reference_parts = normalized_path_parts(reference)
        suffix_len = common_suffix_len(reference_parts, candidate_parts)
        scores.append(-1000 * suffix_len)
    return min(scores)


def raw_candidate_score(
    candidate: RawCandidate,
    case_id: str,
    annotation_path: Path | None = None,
    annotation: dict | None = None,
) -> tuple[int, str]:
    path_text = str(candidate.path)
    parts = set(candidate.path.parts)
    score = raw_reference_score(candidate, annotation)
    if annotation_path is not None:
        annotation_parent = annotation_path.parent
        if annotation_parent == candidate.path.parent or annotation_parent in candidate.path.parents:
            score -= 1000
        score -= common_prefix_len(annotation_parent.parts, candidate.path.parent.parts) * 3
    if "images/line_point" in path_text:
        score += 0
    elif "images/raw_line_point" in path_text:
        score += 10
    elif "images/point_only" in path_text:
        score += 50
    else:
        score += 100
    if case_id not in parts:
        score += 20
    return score, path_text


def resolve_raw_candidate(annotation_path: Path, annotation: dict, index: dict[str, list[RawCandidate]]) -> tuple[RawCandidate, int]:
    image_width = int(annotation["image_width"])
    image_height = int(annotation["image_height"])
    case_id = extract_case_id(annotation_sample_id(annotation_path))

    raw_filenames = dict.fromkeys(
        filename
        for filename in (
            reference_basename(annotation.get("source_raw_path")),
            str(annotation.get("source_raw_filename") or ""),
            str(annotation.get("raw_filename") or ""),
            reference_basename(annotation.get("raw_path")),
        )
        if filename
    )
    candidates: list[RawCandidate] = []
    for raw_filename in raw_filenames:
        candidates = [
            candidate
            for candidate in index.get(raw_filename, [])
            if candidate.width == image_width and candidate.height == image_height
        ]
        if candidates:
            break
    if not candidates:
        tried = ", ".join(raw_filenames) or "<missing filename>"
        raise FileNotFoundError(f"No local raw image matched [{tried}] at {image_width}x{image_height}")

    candidates.sort(key=lambda candidate: raw_candidate_score(candidate, case_id, annotation_path, annotation))
    return candidates[0], len(candidates)


def annotation_keypoints(
    annotation: dict,
    *,
    canonicalize_line_endpoints: bool = False,
) -> dict[str, tuple[float, float]]:
    points = annotation.get("points", {})
    lines = annotation.get("lines", {})
    keypoints: dict[str, tuple[float, float]] = {}
    for name in ANNOTATION_POINT_NAMES:
        point = points[name]
        keypoints[name] = (float(point["x"]), float(point["y"]))
    for line_name in ANNOTATION_LINE_NAMES:
        line = lines[line_name]
        endpoints = [line["p1"], line["p2"]]
        if canonicalize_line_endpoints and float(endpoints[0]["x"]) > float(endpoints[1]["x"]):
            endpoints.reverse()
        for endpoint, point in zip(("p1", "p2"), endpoints):
            keypoints[f"{line_name}_{endpoint}"] = (float(point["x"]), float(point["y"]))
    return keypoints


MANIFEST_PATH_FIELDS = (
    "annotation_path",
    "raw_path",
    "point_path",
    "line_path",
    "combined_path",
)


def resolve_manifest_path_value(value: str, manifest_dir: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute() or path.exists():
        return value
    manifest_relative = manifest_dir / path
    if manifest_relative.exists():
        return manifest_relative.as_posix()
    return value


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_dir = path.parent
    for row in rows:
        for field in MANIFEST_PATH_FIELDS:
            if field in row:
                row[field] = resolve_manifest_path_value(row[field], manifest_dir)
    return rows


def dataset_manifest_path(dataset_dir: Path | None, manifest_path: Path | None, default_manifest: Path) -> Path:
    if dataset_dir is not None:
        return dataset_dir / "manifest.csv"
    if manifest_path is not None:
        return manifest_path
    return default_manifest

#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2

from measure_angles import ANNOTATION_LINE_NAMES, ANNOTATION_POINT_NAMES


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


def raw_candidate_score(candidate: RawCandidate, case_id: str) -> tuple[int, str]:
    path_text = str(candidate.path)
    parts = set(candidate.path.parts)
    score = 0
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
    raw_filename = annotation.get("raw_filename") or Path(str(annotation.get("raw_path", ""))).name
    image_width = int(annotation["image_width"])
    image_height = int(annotation["image_height"])
    case_id = extract_case_id(annotation_sample_id(annotation_path))

    candidates = [
        candidate
        for candidate in index.get(raw_filename, [])
        if candidate.width == image_width and candidate.height == image_height
    ]
    if not candidates:
        raise FileNotFoundError(f"No local raw image matched {raw_filename} at {image_width}x{image_height}")

    candidates.sort(key=lambda candidate: raw_candidate_score(candidate, case_id))
    return candidates[0], len(candidates)


def annotation_keypoints(annotation: dict) -> dict[str, tuple[float, float]]:
    points = annotation.get("points", {})
    lines = annotation.get("lines", {})
    keypoints: dict[str, tuple[float, float]] = {}
    for name in ANNOTATION_POINT_NAMES:
        point = points[name]
        keypoints[name] = (float(point["x"]), float(point["y"]))
    for line_name in ANNOTATION_LINE_NAMES:
        line = lines[line_name]
        for endpoint in ("p1", "p2"):
            point = line[endpoint]
            keypoints[f"{line_name}_{endpoint}"] = (float(point["x"]), float(point["y"]))
    return keypoints


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

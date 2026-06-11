#!/usr/bin/env python3

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


UPPER_ANGLE_LABEL = "mLDFA"
LOWER_ANGLE_LABEL = "MPTA"
JLCA_ANGLE_LABEL = "JLCA"
HKA_ANGLE_LABEL = "HKA"
UPPER_ANGLE_FULL_NAME = "mechanical lateral distal femoral angle"
LOWER_ANGLE_FULL_NAME = "medial proximal tibial angle"
JLCA_ANGLE_FULL_NAME = "joint line convergence angle"
HKA_ANGLE_FULL_NAME = "hip-knee-ankle angle"
ANNOTATION_VERSION = 1
ANNOTATION_POINT_SPECS = [
    ("hip", "Femur axis point"),
    ("upper_left", "Upper line left"),
    ("upper_center", "Upper line center"),
    ("upper_right", "Upper line right"),
    ("lower_left", "Lower line left"),
    ("lower_center", "Lower line center"),
    ("lower_right", "Lower line right"),
    ("ankle", "Tibia axis point"),
]
ANNOTATION_POINT_NAMES = tuple(name for name, _label in ANNOTATION_POINT_SPECS)
ANNOTATION_LINE_SPECS = [
    ("upper_line", "Upper joint line"),
    ("lower_line", "Lower joint line"),
]
ANNOTATION_LINE_NAMES = tuple(name for name, _label in ANNOTATION_LINE_SPECS)
RENDER_STYLE_DEBUG = "debug"
RENDER_STYLE_CLINICAL = "clinical"
RENDER_STYLES = (RENDER_STYLE_DEBUG, RENDER_STYLE_CLINICAL)


@dataclass
class LineModel:
    vx: float
    vy: float
    x0: float
    y0: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x0, self.y0], dtype=np.float32)

    @property
    def direction(self) -> np.ndarray:
        vec = np.array([self.vx, self.vy], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    for ch in ["’", "′", "＇", "`", "´"]:
        name = name.replace(ch, "'")
    name = name.replace("\u3000", " ")
    return " ".join(name.split())


def infer_side(name: str) -> str | None:
    basename = normalize_name(Path(str(name)).name)
    stem = Path(basename).stem.upper()

    if re.search(r"\d+\s*RL(?:$|[^A-Z0-9])", stem) or re.search(r"(?:^|[^A-Z0-9])RL(?:$|[^A-Z0-9])", stem):
        return "RL"

    bracket_match = re.search(r"[\[\(\{]([LR])[\]\)\}]", stem)
    if bracket_match:
        return bracket_match.group(1)

    digit_match = re.search(r"\d+\s*([LR])(?:$|[^A-Z])", stem)
    if digit_match:
        return digit_match.group(1)

    start_match = re.search(r"^([LR])(?:$|[^A-Z])", stem)
    if start_match:
        return start_match.group(1)

    token_match = re.search(r"(?:^|[^A-Z0-9])([LR])(?:$|[^A-Z0-9])", stem)
    if token_match:
        return token_match.group(1)

    return None


def normalize_measurement_side(side: str | None) -> str | None:
    if side is None:
        return None
    value = normalize_name(str(side)).strip().upper()
    if value in {"L", "LEFT"}:
        return "L"
    if value in {"R", "RIGHT"}:
        return "R"
    return None


def normalize_render_style(style: str | None) -> str:
    if style is None:
        return RENDER_STYLE_DEBUG
    value = normalize_name(str(style)).strip().lower()
    if value in RENDER_STYLES:
        return value
    raise ValueError(f"Unknown render style: {style}. Expected one of: {', '.join(RENDER_STYLES)}")


def infer_knee_side_from_sources(*sources: object) -> str | None:
    for source in sources:
        if source is None:
            continue
        side = normalize_measurement_side(str(source))
        if side is None:
            side = infer_side(str(source))
        side = normalize_measurement_side(side)
        if side is not None:
            return side
    return None


def classify_file(path: Path) -> str:
    name = normalize_name(path.name)
    if "line" in name.lower():
        return "line"
    if "'" in name:
        return "point"
    if name.endswith(".jpg") and (name.endswith("RL.jpg") or name[-5].upper() in {"L", "R"}):
        return "raw"
    return "unknown"


def resolve_raw_path(point_path: Path, line_path: Path, raw_path: Path | None) -> Path:
    if raw_path is not None:
        return raw_path

    case_dir = point_path.parent
    side = infer_side(normalize_name(point_path.name)) or infer_side(normalize_name(line_path.name))

    candidates = []
    for path in case_dir.iterdir():
        if not path.is_file() or path.name == ".DS_Store":
            continue
        if classify_file(path) != "raw":
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No raw image found in {case_dir}")

    if side is not None:
        for candidate in candidates:
            candidate_side = infer_side(normalize_name(candidate.name))
            if candidate_side == side:
                return candidate
        for candidate in candidates:
            if infer_side(normalize_name(candidate.name)) == "RL":
                return candidate

    return sorted(candidates)[0]


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def point_to_array(point: dict | list | tuple | np.ndarray, name: str = "point") -> np.ndarray:
    if isinstance(point, dict):
        if "x" not in point or "y" not in point:
            raise ValueError(f"Annotation point '{name}' must contain x and y.")
        x_value = point["x"]
        y_value = point["y"]
    elif isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2:
        x_value, y_value = point
    else:
        raise ValueError(f"Annotation point '{name}' must be a 2D coordinate.")
    return np.array([float(x_value), float(y_value)], dtype=np.float32)


def named_points_to_arrays(named_points: dict) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    missing = [name for name in ANNOTATION_POINT_NAMES if name not in named_points]
    if missing:
        raise ValueError(f"Missing annotation points: {', '.join(missing)}")

    hip = point_to_array(named_points["hip"], "hip")
    ankle = point_to_array(named_points["ankle"], "ankle")
    upper_points = [point_to_array(named_points[name], name) for name in ("upper_left", "upper_center", "upper_right")]
    lower_points = [point_to_array(named_points[name], name) for name in ("lower_left", "lower_center", "lower_right")]
    return hip, ankle, upper_points, lower_points


def build_annotation_record(
    raw_path: Path,
    image_shape: tuple[int, int],
    named_points: dict,
    named_lines: dict | None = None,
    side: str | None = None,
) -> dict:
    image_h, image_w = image_shape[:2]
    record_points: dict[str, dict[str, float]] = {}
    for name in ANNOTATION_POINT_NAMES:
        point = point_to_array(named_points[name], name)
        record_points[name] = {
            "x": float(point[0]),
            "y": float(point[1]),
        }

    record = {
        "version": ANNOTATION_VERSION,
        "raw_path": str(raw_path),
        "raw_filename": raw_path.name,
        "image_width": int(image_w),
        "image_height": int(image_h),
        "points": record_points,
    }
    resolved_side = normalize_measurement_side(side) or infer_knee_side_from_sources(raw_path)
    if resolved_side is not None:
        record["side"] = resolved_side
    lines_record = build_line_record(named_lines)
    if lines_record:
        record["lines"] = lines_record
    return record


def line_endpoint_pair_to_arrays(line_data: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(line_data, dict) or "p1" not in line_data or "p2" not in line_data:
        raise ValueError(f"Annotation line '{name}' must contain p1 and p2.")
    return point_to_array(line_data["p1"], f"{name}.p1"), point_to_array(line_data["p2"], f"{name}.p2")


def named_lines_to_arrays(named_lines: dict | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not named_lines:
        return {}

    converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ANNOTATION_LINE_NAMES:
        if name not in named_lines:
            continue
        converted[name] = line_endpoint_pair_to_arrays(named_lines[name], name)
    return converted


def build_line_record(named_lines: dict | None) -> dict[str, dict[str, dict[str, float]]]:
    record: dict[str, dict[str, dict[str, float]]] = {}
    for name, (p1, p2) in named_lines_to_arrays(named_lines).items():
        record[name] = {
            "p1": {"x": float(p1[0]), "y": float(p1[1])},
            "p2": {"x": float(p2[0]), "y": float(p2[1])},
        }
    return record


def load_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "points" not in data or not isinstance(data["points"], dict):
        raise ValueError(f"Invalid annotation file: {path}")

    missing = [name for name in ANNOTATION_POINT_NAMES if name not in data["points"]]
    if missing:
        raise ValueError(f"Annotation file is missing points: {', '.join(missing)}")

    if "lines" in data and data["lines"] is not None:
        build_line_record(data["lines"])

    return data


def extract_line_mask(raw_gray: np.ndarray, line_gray: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(raw_gray, line_gray)
    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def knee_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("No line pixels found after raw/line subtraction.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def merge_close_points(points: list[dict], distance_thresh: float = 35.0) -> list[dict]:
    merged: list[dict] = []
    for point in sorted(points, key=lambda item: item["area"], reverse=True):
        chosen = None
        for group in merged:
            distance = math.hypot(point["x"] - group["x"], point["y"] - group["y"])
            if distance < distance_thresh:
                chosen = group
                break
        if chosen is None:
            merged.append(point.copy())
            continue

        total_area = chosen["area"] + point["area"]
        chosen["x"] = (chosen["x"] * chosen["area"] + point["x"] * point["area"]) / total_area
        chosen["y"] = (chosen["y"] * chosen["area"] + point["y"] * point["area"]) / total_area
        chosen["area"] = total_area

    return merged


def detect_blue_points(point_image: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(point_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([140, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        components.append(
            {
                "x": float(centroids[idx][0]),
                "y": float(centroids[idx][1]),
                "area": area,
            }
        )

    return merge_close_points(components)


def estimate_point_to_raw_transform(raw_image: np.ndarray, point_image: np.ndarray) -> np.ndarray:
    raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    point_gray = cv2.cvtColor(point_image, cv2.COLOR_BGR2GRAY)
    raw_h, raw_w = raw_gray.shape
    point_resized = cv2.resize(point_gray, (raw_w, raw_h), interpolation=cv2.INTER_AREA)

    raw_blur = cv2.GaussianBlur(raw_gray, (0, 0), 2.0)
    point_blur = cv2.GaussianBlur(point_resized, (0, 0), 2.0)
    template = raw_blur.astype(np.float32) / 255.0
    moving = point_blur.astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)
    return warp


def map_points_to_raw(
    points: list[dict],
    raw_shape: tuple[int, int],
    point_shape: tuple[int, int],
    point_to_raw_warp: np.ndarray | None = None,
) -> list[dict]:
    raw_h, raw_w = raw_shape
    point_h, point_w = point_shape
    sx = raw_w / point_w
    sy = raw_h / point_h

    mapped = []
    for point in points:
        scaled_x = point["x"] * sx
        scaled_y = point["y"] * sy
        if point_to_raw_warp is not None:
            raw_x = point_to_raw_warp[0, 0] * scaled_x + point_to_raw_warp[0, 1] * scaled_y + point_to_raw_warp[0, 2]
            raw_y = point_to_raw_warp[1, 0] * scaled_x + point_to_raw_warp[1, 1] * scaled_y + point_to_raw_warp[1, 2]
        else:
            raw_x = scaled_x
            raw_y = scaled_y

        mapped.append(
            {
                **point,
                "raw_x": raw_x,
                "raw_y": raw_y,
            }
        )
    return mapped


def select_measurement_points(points: list[dict], bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    x1, y1, x2, y2 = bbox
    knee_center_x = 0.5 * (x1 + x2)
    knee_center_y = 0.5 * (y1 + y2)

    knee_points = [
        point
        for point in points
        if x1 - 180 <= point["raw_x"] <= x2 + 180 and y1 - 120 <= point["raw_y"] <= y2 + 120
    ]
    if len(knee_points) != 6:
        knee_points = sorted(
            points,
            key=lambda point: (point["raw_y"] - knee_center_y) ** 2 + (point["raw_x"] - knee_center_x) ** 2,
        )[:6]

    upper_candidates = [point for point in points if point["raw_y"] < y1 - 80]
    lower_candidates = [point for point in points if point["raw_y"] > y2 + 80]
    if not upper_candidates or not lower_candidates:
        raise ValueError("Could not isolate the proximal and distal axis points from the point image.")

    hip_point = min(upper_candidates, key=lambda point: (abs(point["raw_x"] - knee_center_x), point["raw_y"]))
    ankle_point = min(lower_candidates, key=lambda point: (abs(point["raw_x"] - knee_center_x), -point["raw_y"]))

    knee_points = sorted(knee_points, key=lambda point: point["raw_x"])
    paired = [knee_points[0:2], knee_points[2:4], knee_points[4:6]]
    if any(len(pair) != 2 for pair in paired):
        raise ValueError("Failed to split the six knee points into three left/center/right pairs.")

    upper_points = []
    lower_points = []
    for pair in paired:
        ordered = sorted(pair, key=lambda point: point["raw_y"])
        upper_points.append(np.array([ordered[0]["raw_x"], ordered[0]["raw_y"]], dtype=np.float32))
        lower_points.append(np.array([ordered[1]["raw_x"], ordered[1]["raw_y"]], dtype=np.float32))

    hip = np.array([hip_point["raw_x"], hip_point["raw_y"]], dtype=np.float32)
    ankle = np.array([ankle_point["raw_x"], ankle_point["raw_y"]], dtype=np.float32)
    return hip, ankle, upper_points, lower_points


def fit_line_from_points(points: list[np.ndarray]) -> LineModel:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return LineModel(float(vx), float(vy), float(x0), float(y0))


def line_model_from_segment(start: np.ndarray, end: np.ndarray) -> LineModel:
    vec = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        raise ValueError("A joint line segment must have non-zero length.")
    direction = vec / norm
    return LineModel(float(direction[0]), float(direction[1]), float(start[0]), float(start[1]))


def line_distances(line: LineModel, points: np.ndarray) -> np.ndarray:
    v = line.direction
    p0 = line.point
    delta = points - p0
    return np.abs(delta[:, 0] * v[1] - delta[:, 1] * v[0])


def signed_line_offsets(line: LineModel, points: np.ndarray) -> np.ndarray:
    v = line.direction
    p0 = line.point
    delta = points - p0
    return delta[:, 0] * v[1] - delta[:, 1] * v[0]


def refine_line_with_mask(mask: np.ndarray, provisional: LineModel, other: LineModel) -> tuple[LineModel, tuple[np.ndarray, np.ndarray]]:
    ys, xs = np.where(mask > 0)
    pixels = np.column_stack([xs, ys]).astype(np.float32)
    if len(pixels) == 0:
        return provisional, (provisional.point, provisional.point)

    dist_self = line_distances(provisional, pixels)
    dist_other = line_distances(other, pixels)
    assigned = pixels[(dist_self <= dist_other) & (dist_self < 25.0)]
    if len(assigned) < 20:
        assigned = pixels[dist_self < 25.0]
    if len(assigned) < 2:
        return provisional, line_segment_from_group(provisional, pixels)

    line = fit_line_from_points([point for point in assigned])
    return line, line_segment_from_group(line, assigned)


def extract_joint_lines_from_mask(
    mask: np.ndarray,
    upper_points: list[np.ndarray],
    lower_points: list[np.ndarray],
) -> tuple[tuple[LineModel, tuple[np.ndarray, np.ndarray]], tuple[LineModel, tuple[np.ndarray, np.ndarray]]]:
    upper_guess = fit_line_from_points(upper_points)
    lower_guess = fit_line_from_points(lower_points)

    ys, xs = np.where(mask > 0)
    pixels = np.column_stack([xs, ys]).astype(np.float32)
    if len(pixels) == 0:
        raise ValueError("No line pixels found after raw/line subtraction.")

    midpoints = [(upper + lower) * 0.5 for upper, lower in zip(upper_points, lower_points)]
    separator = fit_line_from_points(midpoints)
    upper_sign = float(np.mean(signed_line_offsets(separator, np.asarray(upper_points, dtype=np.float32))))
    if abs(upper_sign) < 1e-6:
        upper_sign = -1.0

    split_values = signed_line_offsets(separator, pixels) * upper_sign
    upper_pixels = pixels[split_values >= 0]
    lower_pixels = pixels[split_values < 0]

    upper_pixels = upper_pixels[line_distances(upper_guess, upper_pixels) < 35.0] if len(upper_pixels) else upper_pixels
    lower_pixels = lower_pixels[line_distances(lower_guess, lower_pixels) < 35.0] if len(lower_pixels) else lower_pixels

    if len(upper_pixels) < 20 or len(lower_pixels) < 20:
        upper_line, upper_segment = refine_line_with_mask(mask, upper_guess, lower_guess)
        lower_line, lower_segment = refine_line_with_mask(mask, lower_guess, upper_guess)
        return (upper_line, upper_segment), (lower_line, lower_segment)

    upper_line = fit_line_from_points([point for point in upper_pixels])
    lower_line = fit_line_from_points([point for point in lower_pixels])
    upper_segment = line_segment_from_group(upper_line, upper_pixels)
    lower_segment = line_segment_from_group(lower_line, lower_pixels)
    return (upper_line, upper_segment), (lower_line, lower_segment)


def line_segment_from_group(line: LineModel, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = line.direction
    origin = line.point
    projections = (points - origin) @ direction
    start = origin + direction * (projections.min() - 10.0)
    end = origin + direction * (projections.max() + 10.0)
    return start, end


def line_segment_from_reference_points(line: LineModel, reference_points: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    direction = line.direction
    origin = line.point
    projections = (reference - origin) @ direction
    start = origin + direction * projections.min()
    end = origin + direction * projections.max()
    return start, end


def line_segment_from_model_extent(
    line: LineModel,
    reference_points: list[np.ndarray] | np.ndarray,
    margin: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    direction = line.direction
    origin = line.point
    projections = (reference - origin) @ direction
    start = origin + direction * (float(projections.min()) - margin)
    end = origin + direction * (float(projections.max()) + margin)
    return start, end


def extend_segment_to_include_point(
    start: np.ndarray,
    end: np.ndarray,
    point: np.ndarray,
    end_margin: float = 140.0,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return start, end
    unit = direction / length
    point_projection = float(np.dot(point - start, unit))
    end_projection = max(length, point_projection + end_margin)
    return start, start + unit * end_projection


def extend_segment_both_directions_to_include_point(
    start: np.ndarray,
    end: np.ndarray,
    point: np.ndarray,
    margin: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return start, end
    unit = direction / length
    point_projection = float(np.dot(point - start, unit))
    start_projection = min(0.0, point_projection - margin)
    end_projection = max(length, point_projection + margin)
    return start + unit * start_projection, start + unit * end_projection


def intersect_lines(line_a: LineModel, line_b: LineModel) -> np.ndarray:
    p = line_a.point
    r = line_a.direction
    q = line_b.point
    s = line_b.direction
    matrix = np.array([[r[0], -s[0]], [r[1], -s[1]]], dtype=np.float32)
    rhs = q - p
    try:
        t, _ = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The fitted lines are nearly parallel and cannot be intersected.") from exc
    return p + t * r


def acute_angle_degrees(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a / np.linalg.norm(vec_a)
    b = vec_b / np.linalg.norm(vec_b)
    dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def angle_degrees(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a / np.linalg.norm(vec_a)
    b = vec_b / np.linalg.norm(vec_b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def line_angle_degrees(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return acute_angle_degrees(vec_a, vec_b)


def direction_angle(vec: np.ndarray) -> float:
    return math.atan2(float(vec[1]), float(vec[0]))


def shortest_arc(start: float, end: float) -> tuple[float, float]:
    delta = (end - start + math.pi) % (2 * math.pi) - math.pi
    return start, start + delta


def draw_line(image: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    cv2.line(image, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), color, thickness, cv2.LINE_AA)


def ray_endpoint_from_reference(
    center: np.ndarray,
    direction: np.ndarray,
    reference_points: np.ndarray | list[np.ndarray] | tuple[np.ndarray, np.ndarray],
    margin: float = 28.0,
    min_length: float = 120.0,
) -> np.ndarray:
    reference = np.vstack(reference_points).astype(np.float32)
    unit = direction / np.linalg.norm(direction)
    projections = (reference - center) @ unit
    forward = projections[projections > 0]
    length = max(min_length, float(forward.max()) + margin) if len(forward) else min_length
    return center + unit * length


def sort_points_by_x(points: list[np.ndarray] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    order = np.argsort(arr[:, 0])
    return arr[order]


def draw_joint_points(
    image: np.ndarray,
    points: list[np.ndarray] | np.ndarray,
    color: tuple[int, int, int],
    radius: int = 4,
) -> None:
    for point in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        center = tuple(np.round(point).astype(int))
        cv2.circle(image, center, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def draw_arc(
    image: np.ndarray,
    center: np.ndarray,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    radius: int,
    color: tuple[int, int, int],
    thickness: int = 3,
    label_distance: float = 110.0,
) -> np.ndarray:
    angle_a = direction_angle(vec_a)
    angle_b = direction_angle(vec_b)
    start, end = shortest_arc(angle_a, angle_b)
    ts = np.linspace(start, end, 80)
    points = np.array(
        [
            [center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)]
            for theta in ts
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    mid_angle = 0.5 * (start + end)
    return np.array(
        [
            center[0] + (radius + label_distance) * math.cos(mid_angle),
            center[1] + (radius + label_distance) * math.sin(mid_angle),
        ],
        dtype=np.float32,
    )


def choose_closer_line_ray(axis_vec: np.ndarray, line_vec: np.ndarray) -> np.ndarray:
    if angle_degrees(axis_vec, line_vec) <= angle_degrees(axis_vec, -line_vec):
        return line_vec
    return -line_vec


def choose_farther_line_ray(axis_vec: np.ndarray, line_vec: np.ndarray) -> np.ndarray:
    if angle_degrees(axis_vec, line_vec) >= angle_degrees(axis_vec, -line_vec):
        return line_vec
    return -line_vec


def choose_line_ray_by_screen_side(line_vec: np.ndarray, screen_side: str) -> np.ndarray:
    desired = np.array([-1.0, 0.0], dtype=np.float32) if screen_side == "left" else np.array([1.0, 0.0], dtype=np.float32)
    return line_vec if float(np.dot(line_vec, desired)) >= float(np.dot(-line_vec, desired)) else -line_vec


def choose_acute_ray_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if angle_degrees(vec_a, vec_b) <= 90.0:
        return vec_a, vec_b
    return vec_a, -vec_b


def anatomical_angle_screen_sides(side: str | None) -> tuple[str, str] | None:
    normalized_side = normalize_measurement_side(side)
    if normalized_side == "R":
        return "left", "right"
    if normalized_side == "L":
        return "right", "left"
    return None


def draw_text_box(
    image: np.ndarray,
    text: str,
    center: np.ndarray,
    text_color: tuple[int, int, int] = (0, 255, 255),
    box_color: tuple[int, int, int] = (24, 24, 24),
    border_color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3
    pad_x = 14
    pad_y = 12

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    img_h, img_w = image.shape[:2]

    x = int(round(center[0] - text_w / 2))
    y = int(round(center[1] + text_h / 2))
    x = max(pad_x, min(x, img_w - text_w - pad_x))
    y = max(text_h + pad_y, min(y, img_h - baseline - pad_y))

    top_left = (x - pad_x, y - text_h - pad_y)
    bottom_right = (x + text_w + pad_x, y + baseline + pad_y)

    cv2.rectangle(image, top_left, bottom_right, box_color, -1)
    cv2.rectangle(image, top_left, bottom_right, border_color, 2)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_label_tag(
    image: np.ndarray,
    text: str,
    center: np.ndarray,
    color: tuple[int, int, int],
    font_scale: float = 0.68,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    img_h, img_w = image.shape[:2]
    pad = 8
    x = int(round(center[0] - text_w / 2))
    y = int(round(center[1] + text_h / 2))
    x = max(pad, min(x, img_w - text_w - pad))
    y = max(text_h + pad, min(y, img_h - baseline - pad))
    cv2.putText(image, text, (x, y), font, font_scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_measurement(
    raw_image: np.ndarray,
    joint_segment: tuple[np.ndarray, np.ndarray],
    joint_intersection: np.ndarray,
    axis_origin: np.ndarray,
    joint_ray: np.ndarray,
    angle_value: float,
    label: str,
) -> np.ndarray:
    canvas = raw_image.copy()
    annotate_measurement(canvas, joint_segment, joint_intersection, axis_origin, joint_ray, angle_value, label)
    return canvas


def safe_line_intersection(line_a: LineModel, line_b: LineModel, fallback: np.ndarray) -> np.ndarray:
    try:
        intersection = intersect_lines(line_a, line_b)
    except ValueError:
        return np.asarray(fallback, dtype=np.float32)
    if not np.all(np.isfinite(intersection)):
        return np.asarray(fallback, dtype=np.float32)
    return intersection


def annotate_two_line_angle(
    canvas: np.ndarray,
    center: np.ndarray,
    ray_a: np.ndarray,
    ray_b: np.ndarray,
    angle_value: float,
    label: str,
    color: tuple[int, int, int],
    radius: int,
    text_offset: np.ndarray | None = None,
    segment_a: tuple[np.ndarray, np.ndarray] | None = None,
    segment_b: tuple[np.ndarray, np.ndarray] | None = None,
    show_text: bool = True,
    label_only: bool = False,
    draw_segments: bool = True,
    line_thickness: int = 2,
    arc_thickness: int = 3,
) -> np.ndarray:
    if draw_segments:
        if segment_a is not None:
            draw_line(canvas, segment_a[0], segment_a[1], color, thickness=line_thickness)
        else:
            draw_line(
                canvas,
                center,
                ray_endpoint_from_reference(center, ray_a, [center + ray_a], min_length=160),
                color,
                thickness=line_thickness,
            )
        if segment_b is not None:
            draw_line(canvas, segment_b[0], segment_b[1], color, thickness=line_thickness)
        else:
            draw_line(
                canvas,
                center,
                ray_endpoint_from_reference(center, ray_b, [center + ray_b], min_length=160),
                color,
                thickness=line_thickness,
            )

    text_anchor = draw_arc(canvas, center, ray_a, ray_b, radius=radius, color=color, thickness=arc_thickness)
    if text_offset is not None:
        text_anchor = text_anchor + text_offset.astype(np.float32)
    if show_text:
        if label_only:
            draw_label_tag(canvas, label, text_anchor, color)
        else:
            draw_text_box(canvas, f"{label}={angle_value:.1f} deg", text_anchor, border_color=color)
    return text_anchor


def annotate_measurement(
    canvas: np.ndarray,
    joint_segment: tuple[np.ndarray, np.ndarray],
    joint_intersection: np.ndarray,
    axis_origin: np.ndarray,
    joint_ray: np.ndarray,
    angle_value: float,
    label: str,
    color: tuple[int, int, int] = (255, 0, 255),
    show_text: bool = True,
    label_only: bool = False,
    line_thickness: int = 2,
    arc_thickness: int = 3,
    draw_axis_point: bool = True,
) -> None:
    if draw_axis_point:
        draw_joint_points(canvas, [axis_origin], color, radius=4)
    draw_line(canvas, joint_segment[0], joint_segment[1], color, thickness=line_thickness)
    joint_ray_end = ray_endpoint_from_reference(joint_intersection, joint_ray, joint_segment)
    draw_line(canvas, joint_intersection, joint_ray_end, color, thickness=line_thickness)
    draw_line(canvas, axis_origin, joint_intersection, color, thickness=line_thickness)

    axis_vec = axis_origin - joint_intersection
    text_anchor = draw_arc(canvas, joint_intersection, axis_vec, joint_ray, radius=52, color=color, thickness=arc_thickness)
    if show_text:
        if label_only:
            draw_label_tag(canvas, label, text_anchor, color)
        else:
            text = f"{label}={angle_value:.1f} deg"
            draw_text_box(canvas, text, text_anchor)


def measure_from_named_points(
    raw_image: np.ndarray,
    named_points: dict,
    raw_path: Path | None = None,
    named_lines: dict | None = None,
    side: str | None = None,
    render_style: str = RENDER_STYLE_DEBUG,
) -> tuple[dict, dict]:
    style = normalize_render_style(render_style)
    hip, ankle, upper_points, lower_points = named_points_to_arrays(named_points)
    converted_lines = named_lines_to_arrays(named_lines)
    if "upper_line" in converted_lines and "lower_line" in converted_lines:
        upper_segment = converted_lines["upper_line"]
        lower_segment = converted_lines["lower_line"]
        upper_line = line_model_from_segment(*upper_segment)
        lower_line = line_model_from_segment(*lower_segment)
        line_source = "manual_lines"
    else:
        upper_line = fit_line_from_points(upper_points)
        lower_line = fit_line_from_points(lower_points)
        upper_segment = line_segment_from_reference_points(upper_line, upper_points)
        lower_segment = line_segment_from_reference_points(lower_line, lower_points)
        line_source = "points_fit"

    upper_center = upper_points[1]
    lower_center = lower_points[1]
    femur_axis = fit_line_from_points([hip, upper_center])
    tibia_axis = fit_line_from_points([lower_center, ankle])
    knee_center = (upper_center + lower_center) * 0.5

    upper_intersection = intersect_lines(femur_axis, upper_line)
    lower_intersection = intersect_lines(tibia_axis, lower_line)
    jlca_intersection = safe_line_intersection(upper_line, lower_line, knee_center)
    hka_intersection = safe_line_intersection(femur_axis, tibia_axis, knee_center)

    upper_axis_vec = hip - upper_intersection
    lower_axis_vec = ankle - lower_intersection
    femur_distal_vec = upper_center - hip
    tibia_distal_vec = ankle - lower_center
    measurement_side = normalize_measurement_side(side) or infer_knee_side_from_sources(raw_path)
    screen_sides = anatomical_angle_screen_sides(measurement_side)
    if screen_sides is None:
        upper_joint_ray = choose_closer_line_ray(upper_axis_vec, upper_line.direction)
        lower_joint_ray = choose_farther_line_ray(lower_axis_vec, lower_line.direction)
        jlca_screen_side = "left"
        angle_side_source = "geometric_fallback"
        upper_angle_screen_side = None
        lower_angle_screen_side = None
    else:
        upper_angle_screen_side, lower_angle_screen_side = screen_sides
        upper_joint_ray = choose_line_ray_by_screen_side(upper_line.direction, upper_angle_screen_side)
        lower_joint_ray = choose_line_ray_by_screen_side(lower_line.direction, lower_angle_screen_side)
        jlca_screen_side = upper_angle_screen_side
        angle_side_source = "anatomical_side"

    e_angle = angle_degrees(upper_axis_vec, upper_joint_ray)
    g_angle = angle_degrees(lower_axis_vec, lower_joint_ray)
    jlca_angle = line_angle_degrees(upper_line.direction, lower_line.direction)
    hka_angle = line_angle_degrees(femur_distal_vec, tibia_distal_vec)
    jlca_ray_a = choose_line_ray_by_screen_side(upper_line.direction, jlca_screen_side)
    jlca_ray_b = choose_line_ray_by_screen_side(lower_line.direction, jlca_screen_side)
    jlca_ray_a, jlca_ray_b = choose_acute_ray_pair(jlca_ray_a, jlca_ray_b)
    hka_ray_a, hka_ray_b = choose_acute_ray_pair(femur_distal_vec, tibia_distal_vec)
    upper_display_refs = np.vstack([np.asarray(upper_segment, dtype=np.float32), upper_intersection, jlca_intersection])
    lower_display_refs = np.vstack([np.asarray(lower_segment, dtype=np.float32), lower_intersection, jlca_intersection])
    upper_display_segment = line_segment_from_model_extent(upper_line, upper_display_refs, margin=90.0)
    lower_display_segment = line_segment_from_model_extent(lower_line, lower_display_refs, margin=90.0)
    hka_femur_segment = line_segment_from_model_extent(
        femur_axis,
        np.vstack([hip, upper_center, upper_intersection, hka_intersection]),
        margin=90.0,
    )
    hka_tibia_segment = line_segment_from_model_extent(
        tibia_axis,
        np.vstack([lower_center, ankle, lower_intersection, hka_intersection]),
        margin=90.0,
    )

    if style == RENDER_STYLE_CLINICAL:
        axis_color = (255, 180, 60)
        joint_color = (70, 220, 230)
        jlca_color = (80, 210, 180)
        hka_color = (180, 180, 255)
        clinical_line_thickness = 1

        e_image = raw_image.copy()
        annotate_measurement(
            e_image,
            upper_display_segment,
            upper_intersection,
            hip,
            upper_joint_ray,
            e_angle,
            UPPER_ANGLE_LABEL,
            color=joint_color,
            label_only=True,
            line_thickness=clinical_line_thickness,
            arc_thickness=clinical_line_thickness,
            draw_axis_point=False,
        )
        g_image = raw_image.copy()
        annotate_measurement(
            g_image,
            lower_display_segment,
            lower_intersection,
            ankle,
            lower_joint_ray,
            g_angle,
            LOWER_ANGLE_LABEL,
            color=joint_color,
            label_only=True,
            line_thickness=clinical_line_thickness,
            arc_thickness=clinical_line_thickness,
            draw_axis_point=False,
        )
        jlca_image = raw_image.copy()
        annotate_two_line_angle(
            jlca_image,
            jlca_intersection,
            jlca_ray_a,
            jlca_ray_b,
            jlca_angle,
            JLCA_ANGLE_LABEL,
            jlca_color,
            radius=42,
            segment_a=upper_display_segment,
            segment_b=lower_display_segment,
            label_only=True,
            line_thickness=clinical_line_thickness,
            arc_thickness=clinical_line_thickness,
        )
        hka_image = raw_image.copy()
        annotate_two_line_angle(
            hka_image,
            hka_intersection,
            hka_ray_a,
            hka_ray_b,
            hka_angle,
            HKA_ANGLE_LABEL,
            hka_color,
            radius=64,
            text_offset=np.array([0.0, 50.0], dtype=np.float32),
            segment_a=hka_femur_segment,
            segment_b=hka_tibia_segment,
            label_only=True,
            line_thickness=clinical_line_thickness,
            arc_thickness=clinical_line_thickness,
        )

        combined_image = raw_image.copy()
        draw_line(combined_image, upper_display_segment[0], upper_display_segment[1], joint_color, thickness=clinical_line_thickness)
        draw_line(combined_image, lower_display_segment[0], lower_display_segment[1], joint_color, thickness=clinical_line_thickness)
        draw_line(combined_image, hka_femur_segment[0], hka_femur_segment[1], axis_color, thickness=clinical_line_thickness)
        draw_line(combined_image, hka_tibia_segment[0], hka_tibia_segment[1], axis_color, thickness=clinical_line_thickness)

        upper_anchor = draw_arc(
            combined_image,
            upper_intersection,
            upper_axis_vec,
            upper_joint_ray,
            radius=50,
            color=joint_color,
            thickness=clinical_line_thickness,
            label_distance=50.0,
        )
        lower_anchor = draw_arc(
            combined_image,
            lower_intersection,
            lower_axis_vec,
            lower_joint_ray,
            radius=50,
            color=joint_color,
            thickness=clinical_line_thickness,
            label_distance=50.0,
        )
        jlca_anchor = draw_arc(
            combined_image,
            jlca_intersection,
            jlca_ray_a,
            jlca_ray_b,
            radius=38,
            color=jlca_color,
            thickness=clinical_line_thickness,
            label_distance=46.0,
        )
        hka_anchor = draw_arc(
            combined_image,
            hka_intersection,
            hka_ray_a,
            hka_ray_b,
            radius=72,
            color=hka_color,
            thickness=clinical_line_thickness,
            label_distance=48.0,
        )
        draw_label_tag(combined_image, UPPER_ANGLE_LABEL, upper_anchor, joint_color)
        draw_label_tag(combined_image, LOWER_ANGLE_LABEL, lower_anchor, joint_color)
        draw_label_tag(combined_image, JLCA_ANGLE_LABEL, jlca_anchor, jlca_color)
        draw_label_tag(combined_image, HKA_ANGLE_LABEL, hka_anchor, hka_color)
    else:
        e_image = draw_measurement(
            raw_image,
            upper_display_segment,
            upper_intersection,
            hip,
            upper_joint_ray,
            e_angle,
            UPPER_ANGLE_LABEL,
        )
        g_image = draw_measurement(
            raw_image,
            lower_display_segment,
            lower_intersection,
            ankle,
            lower_joint_ray,
            g_angle,
            LOWER_ANGLE_LABEL,
        )
        jlca_image = raw_image.copy()
        annotate_two_line_angle(
            jlca_image,
            jlca_intersection,
            jlca_ray_a,
            jlca_ray_b,
            jlca_angle,
            JLCA_ANGLE_LABEL,
            (0, 180, 255),
            radius=42,
            segment_a=upper_display_segment,
            segment_b=lower_display_segment,
        )
        hka_image = raw_image.copy()
        annotate_two_line_angle(
            hka_image,
            hka_intersection,
            hka_ray_a,
            hka_ray_b,
            hka_angle,
            HKA_ANGLE_LABEL,
            (255, 180, 0),
            radius=64,
            text_offset=np.array([0.0, 70.0], dtype=np.float32),
            segment_a=hka_femur_segment,
            segment_b=hka_tibia_segment,
        )
        combined_image = raw_image.copy()
        jlca_text_anchor = annotate_two_line_angle(
            combined_image,
            jlca_intersection,
            jlca_ray_a,
            jlca_ray_b,
            jlca_angle,
            JLCA_ANGLE_LABEL,
            (0, 180, 255),
            radius=36,
            text_offset=np.array([0.0, -55.0], dtype=np.float32),
            show_text=False,
            draw_segments=False,
        )
        hka_text_anchor = annotate_two_line_angle(
            combined_image,
            hka_intersection,
            hka_ray_a,
            hka_ray_b,
            hka_angle,
            HKA_ANGLE_LABEL,
            (255, 180, 0),
            radius=74,
            text_offset=np.array([0.0, 80.0], dtype=np.float32),
            show_text=False,
            draw_segments=False,
        )
        annotate_measurement(
            combined_image,
            upper_display_segment,
            upper_intersection,
            hip,
            upper_joint_ray,
            e_angle,
            UPPER_ANGLE_LABEL,
        )
        annotate_measurement(
            combined_image,
            lower_display_segment,
            lower_intersection,
            ankle,
            lower_joint_ray,
            g_angle,
            LOWER_ANGLE_LABEL,
        )
        draw_text_box(
            combined_image,
            f"{JLCA_ANGLE_LABEL}={jlca_angle:.1f} deg",
            jlca_text_anchor,
            border_color=(0, 180, 255),
        )
        draw_text_box(
            combined_image,
            f"{HKA_ANGLE_LABEL}={hka_angle:.1f} deg",
            hka_text_anchor,
            border_color=(255, 180, 0),
        )

    return (
        {
            "raw_path": raw_path,
            "e_angle": e_angle,
            "g_angle": g_angle,
            "jlca_angle": jlca_angle,
            "hka_angle": hka_angle,
            "mldfa_angle": e_angle,
            "mpta_angle": g_angle,
            "upper_angle_label": UPPER_ANGLE_LABEL,
            "lower_angle_label": LOWER_ANGLE_LABEL,
            "jlca_angle_label": JLCA_ANGLE_LABEL,
            "hka_angle_label": HKA_ANGLE_LABEL,
            "upper_angle_full_name": UPPER_ANGLE_FULL_NAME,
            "lower_angle_full_name": LOWER_ANGLE_FULL_NAME,
            "jlca_angle_full_name": JLCA_ANGLE_FULL_NAME,
            "hka_angle_full_name": HKA_ANGLE_FULL_NAME,
            "render_style": style,
            "side": measurement_side,
            "angle_side_source": angle_side_source,
            "upper_angle_screen_side": upper_angle_screen_side,
            "lower_angle_screen_side": lower_angle_screen_side,
            "e_image": e_image,
            "g_image": g_image,
            "jlca_image": jlca_image,
            "hka_image": hka_image,
            "combined_image": combined_image,
        },
        {
            "hip": hip,
            "ankle": ankle,
            "upper_center": upper_center,
            "lower_center": lower_center,
            "upper_points": np.asarray(upper_points, dtype=np.float32),
            "lower_points": np.asarray(lower_points, dtype=np.float32),
            "upper_segment": np.asarray(upper_segment, dtype=np.float32),
            "lower_segment": np.asarray(lower_segment, dtype=np.float32),
            "upper_display_segment": np.asarray(upper_display_segment, dtype=np.float32),
            "lower_display_segment": np.asarray(lower_display_segment, dtype=np.float32),
            "hka_femur_segment": np.asarray(hka_femur_segment, dtype=np.float32),
            "hka_tibia_segment": np.asarray(hka_tibia_segment, dtype=np.float32),
            "jlca_intersection": jlca_intersection,
            "hka_intersection": hka_intersection,
            "line_source": line_source,
            "side": measurement_side,
            "angle_side_source": angle_side_source,
        },
    )


def render_annotation_point_image(raw_image: np.ndarray, named_points: dict) -> np.ndarray:
    canvas = raw_image.copy()
    color = (255, 160, 0)
    for name in ANNOTATION_POINT_NAMES:
        point = point_to_array(named_points[name], name)
        draw_joint_points(canvas, [point], color, radius=5)
    return canvas


def render_annotation_line_image(
    raw_image: np.ndarray,
    named_points: dict,
    named_lines: dict | None = None,
) -> np.ndarray:
    canvas = raw_image.copy()
    converted_lines = named_lines_to_arrays(named_lines)
    if "upper_line" in converted_lines and "lower_line" in converted_lines:
        upper_segment = converted_lines["upper_line"]
        lower_segment = converted_lines["lower_line"]
    else:
        _hip, _ankle, upper_points, lower_points = named_points_to_arrays(named_points)
        upper_line = fit_line_from_points(upper_points)
        lower_line = fit_line_from_points(lower_points)
        upper_segment = line_segment_from_reference_points(upper_line, upper_points)
        lower_segment = line_segment_from_reference_points(lower_line, lower_points)
    color = (255, 0, 255)
    draw_line(canvas, upper_segment[0], upper_segment[1], color, thickness=2)
    draw_line(canvas, lower_segment[0], lower_segment[1], color, thickness=2)
    return canvas


def measure_from_annotation(
    annotation_path: Path,
    raw_path: Path | None = None,
    side: str | None = None,
    render_style: str = RENDER_STYLE_DEBUG,
) -> tuple[dict, dict]:
    annotation = load_annotation(annotation_path)
    if raw_path is None:
        raw_candidate = Path(annotation["raw_path"])
        raw_path = raw_candidate
    measurement_side = (
        normalize_measurement_side(side)
        or normalize_measurement_side(annotation.get("side"))
        or infer_knee_side_from_sources(annotation_path, raw_path, annotation.get("raw_filename"), annotation.get("raw_path"))
    )
    raw_image = read_color(raw_path)
    return measure_from_named_points(
        raw_image,
        annotation["points"],
        raw_path=raw_path,
        named_lines=annotation.get("lines"),
        side=measurement_side,
        render_style=render_style,
    )


def save_annotation_bundle(
    raw_path: Path,
    named_points: dict,
    out_dir: Path,
    prefix: str | None = None,
    named_lines: dict | None = None,
    side: str | None = None,
    render_style: str = RENDER_STYLE_DEBUG,
) -> tuple[dict[str, Path], dict]:
    raw_image = read_color(raw_path)
    measurement_side = normalize_measurement_side(side) or infer_knee_side_from_sources(raw_path)
    result, _debug = measure_from_named_points(
        raw_image,
        named_points,
        raw_path=raw_path,
        named_lines=named_lines,
        side=measurement_side,
        render_style=render_style,
    )
    point_image = render_annotation_point_image(raw_image, named_points)
    line_image = render_annotation_line_image(raw_image, named_points, named_lines=named_lines)
    annotation = build_annotation_record(raw_path, raw_image.shape, named_points, named_lines=named_lines, side=measurement_side)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_prefix = prefix or normalize_name(raw_path.stem).replace(" ", "_")
    annotation_path = out_dir / f"{base_prefix}_annotation.json"
    point_path = out_dir / f"{base_prefix}_point.jpg"
    line_path = out_dir / f"{base_prefix}_line.jpg"
    combined_path = out_dir / f"{base_prefix}_combined.jpg"

    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle, indent=2)
    cv2.imwrite(str(point_path), point_image)
    cv2.imwrite(str(line_path), line_image)
    cv2.imwrite(str(combined_path), result["combined_image"])

    return (
        {
            "annotation": annotation_path,
            "point_image": point_path,
            "line_image": line_path,
            "combined_image": combined_path,
        },
        result,
    )


def measure_case(
    point_path: Path,
    line_path: Path,
    raw_path: Path | None,
    side: str | None = None,
    render_style: str = RENDER_STYLE_DEBUG,
) -> tuple[dict, dict]:
    raw_path = resolve_raw_path(point_path, line_path, raw_path)
    measurement_side = normalize_measurement_side(side) or infer_knee_side_from_sources(point_path, line_path, raw_path)

    raw_image = read_color(raw_path)
    line_image = read_color(line_path)
    point_image = read_color(point_path)

    raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    line_mask = extract_line_mask(raw_gray, line_gray)
    bbox = knee_bbox_from_mask(line_mask)

    point_candidates = detect_blue_points(point_image)
    point_to_raw_warp = estimate_point_to_raw_transform(raw_image, point_image)
    mapped_points = map_points_to_raw(
        point_candidates,
        raw_gray.shape,
        point_image.shape[:2],
        point_to_raw_warp=point_to_raw_warp,
    )
    hip, ankle, upper_points, lower_points = select_measurement_points(mapped_points, bbox)
    named_points = {
        "hip": hip,
        "upper_left": upper_points[0],
        "upper_center": upper_points[1],
        "upper_right": upper_points[2],
        "lower_left": lower_points[0],
        "lower_center": lower_points[1],
        "lower_right": lower_points[2],
        "ankle": ankle,
    }
    result, debug = measure_from_named_points(
        raw_image,
        named_points,
        raw_path=raw_path,
        side=measurement_side,
        render_style=render_style,
    )
    debug["bbox"] = bbox
    debug["point_to_raw_warp"] = point_to_raw_warp
    debug["side"] = measurement_side
    return result, debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure mLDFA, MPTA, JLCA, and HKA from markup images or a structured annotation file."
    )
    parser.add_argument("--annotation", type=Path, help="Path to a JSON annotation file exported by annotate_gui.py.")
    parser.add_argument("--point", type=Path, help="Path to the point markup image.")
    parser.add_argument("--line", type=Path, help="Path to the line markup image.")
    parser.add_argument("--raw", type=Path, help="Optional raw radiograph path. Auto-detected if omitted.")
    parser.add_argument("--side", choices=["L", "R"], help="Patient knee side. Overrides side inferred from filenames.")
    parser.add_argument(
        "--render-style",
        choices=RENDER_STYLES,
        default=RENDER_STYLE_DEBUG,
        help="Overlay style for generated images.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for result images.")
    parser.add_argument("--prefix", type=str, help="Optional output filename prefix.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.annotation is not None:
        result, _ = measure_from_annotation(args.annotation, args.raw, side=args.side, render_style=args.render_style)
    else:
        if args.point is None or args.line is None:
            parser.error("--point and --line are required unless --annotation is provided.")
        result, _ = measure_case(args.point, args.line, args.raw, side=args.side, render_style=args.render_style)

    prefix = args.prefix
    if not prefix:
        if args.annotation is not None:
            prefix = normalize_name(args.annotation.stem).replace(" ", "_")
        else:
            prefix = normalize_name(args.point.stem).replace("'", "").replace(" ", "_")

    e_path = args.out_dir / f"{prefix}_{UPPER_ANGLE_LABEL}.jpg"
    g_path = args.out_dir / f"{prefix}_{LOWER_ANGLE_LABEL}.jpg"
    jlca_path = args.out_dir / f"{prefix}_{JLCA_ANGLE_LABEL}.jpg"
    hka_path = args.out_dir / f"{prefix}_{HKA_ANGLE_LABEL}.jpg"
    combined_path = args.out_dir / f"{prefix}_combined.jpg"
    cv2.imwrite(str(e_path), result["e_image"])
    cv2.imwrite(str(g_path), result["g_image"])
    cv2.imwrite(str(jlca_path), result["jlca_image"])
    cv2.imwrite(str(hka_path), result["hka_image"])
    cv2.imwrite(str(combined_path), result["combined_image"])

    print(f"raw image : {result['raw_path']}")
    print(f"side      : {result['side'] or 'unknown'} ({result['angle_side_source']})")
    print(f"style     : {result['render_style']}")
    print(f"{UPPER_ANGLE_LABEL:<10}: {result['mldfa_angle']:.2f} deg")
    print(f"{LOWER_ANGLE_LABEL:<10}: {result['mpta_angle']:.2f} deg")
    print(f"{JLCA_ANGLE_LABEL:<10}: {result['jlca_angle']:.2f} deg")
    print(f"{HKA_ANGLE_LABEL:<10}: {result['hka_angle']:.2f} deg")
    print(f"{UPPER_ANGLE_FULL_NAME}")
    print(f"{LOWER_ANGLE_FULL_NAME}")
    print(f"{JLCA_ANGLE_FULL_NAME}")
    print(f"{HKA_ANGLE_FULL_NAME}")
    print(f"{UPPER_ANGLE_LABEL} output : {e_path}")
    print(f"{LOWER_ANGLE_LABEL} output : {g_path}")
    print(f"{JLCA_ANGLE_LABEL} output : {jlca_path}")
    print(f"{HKA_ANGLE_LABEL} output : {hka_path}")
    print(f"Combined  : {combined_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


UPPER_ANGLE_LABEL = "mLDFA"
LOWER_ANGLE_LABEL = "MPTA"
UPPER_ANGLE_FULL_NAME = "mechanical lateral distal femoral angle"
LOWER_ANGLE_FULL_NAME = "medial proximal tibial angle"


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
    if "RL" in name:
        return "RL"
    if "L" in name:
        return "L"
    if "R" in name:
        return "R"
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


def draw_arc(image: np.ndarray, center: np.ndarray, vec_a: np.ndarray, vec_b: np.ndarray, radius: int, color: tuple[int, int, int]) -> np.ndarray:
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
    cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
    mid_angle = 0.5 * (start + end)
    return np.array(
        [
            center[0] + (radius + 110) * math.cos(mid_angle),
            center[1] + (radius + 110) * math.sin(mid_angle),
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


def annotate_measurement(
    canvas: np.ndarray,
    joint_segment: tuple[np.ndarray, np.ndarray],
    joint_intersection: np.ndarray,
    axis_origin: np.ndarray,
    joint_ray: np.ndarray,
    angle_value: float,
    label: str,
) -> None:
    color = (255, 0, 255)

    draw_joint_points(canvas, [axis_origin], color, radius=4)
    draw_line(canvas, joint_segment[0], joint_segment[1], color, thickness=2)
    joint_ray_end = ray_endpoint_from_reference(joint_intersection, joint_ray, joint_segment)
    draw_line(canvas, joint_intersection, joint_ray_end, color)
    draw_line(canvas, axis_origin, joint_intersection, color)

    axis_vec = axis_origin - joint_intersection
    text_anchor = draw_arc(canvas, joint_intersection, axis_vec, joint_ray, radius=52, color=color)
    text = f"{label}={angle_value:.1f} deg"
    draw_text_box(canvas, text, text_anchor)


def measure_case(point_path: Path, line_path: Path, raw_path: Path | None) -> tuple[dict, dict]:
    raw_path = resolve_raw_path(point_path, line_path, raw_path)

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

    upper_line = fit_line_from_points(upper_points)
    lower_line = fit_line_from_points(lower_points)
    upper_segment = line_segment_from_group(upper_line, np.asarray(upper_points, dtype=np.float32))
    lower_segment = line_segment_from_group(lower_line, np.asarray(lower_points, dtype=np.float32))

    upper_center = upper_points[1]
    lower_center = lower_points[1]
    femur_axis = fit_line_from_points([hip, upper_center])
    tibia_axis = fit_line_from_points([lower_center, ankle])

    upper_intersection = intersect_lines(femur_axis, upper_line)
    lower_intersection = intersect_lines(tibia_axis, lower_line)

    upper_axis_vec = hip - upper_intersection
    lower_axis_vec = ankle - lower_intersection
    upper_joint_ray = choose_closer_line_ray(upper_axis_vec, upper_line.direction)
    lower_joint_ray = choose_farther_line_ray(lower_axis_vec, lower_line.direction)

    e_angle = angle_degrees(upper_axis_vec, upper_joint_ray)
    g_angle = angle_degrees(lower_axis_vec, lower_joint_ray)

    e_image = draw_measurement(
        raw_image,
        upper_segment,
        upper_intersection,
        hip,
        upper_joint_ray,
        e_angle,
        UPPER_ANGLE_LABEL,
    )
    g_image = draw_measurement(
        raw_image,
        lower_segment,
        lower_intersection,
        ankle,
        lower_joint_ray,
        g_angle,
        LOWER_ANGLE_LABEL,
    )
    combined_image = raw_image.copy()
    annotate_measurement(
        combined_image,
        upper_segment,
        upper_intersection,
        hip,
        upper_joint_ray,
        e_angle,
        UPPER_ANGLE_LABEL,
    )
    annotate_measurement(
        combined_image,
        lower_segment,
        lower_intersection,
        ankle,
        lower_joint_ray,
        g_angle,
        LOWER_ANGLE_LABEL,
    )

    return (
        {
            "raw_path": raw_path,
            "e_angle": e_angle,
            "g_angle": g_angle,
            "mldfa_angle": e_angle,
            "mpta_angle": g_angle,
            "upper_angle_label": UPPER_ANGLE_LABEL,
            "lower_angle_label": LOWER_ANGLE_LABEL,
            "upper_angle_full_name": UPPER_ANGLE_FULL_NAME,
            "lower_angle_full_name": LOWER_ANGLE_FULL_NAME,
            "point_to_raw_warp": point_to_raw_warp,
            "e_image": e_image,
            "g_image": g_image,
            "combined_image": combined_image,
        },
        {
            "hip": hip,
            "ankle": ankle,
            "upper_center": upper_center,
            "lower_center": lower_center,
            "bbox": bbox,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the mLDFA and MPTA angles from point and line markup images."
    )
    parser.add_argument("--point", type=Path, required=True, help="Path to the point markup image.")
    parser.add_argument("--line", type=Path, required=True, help="Path to the line markup image.")
    parser.add_argument("--raw", type=Path, help="Optional raw radiograph path. Auto-detected if omitted.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for result images.")
    parser.add_argument("--prefix", type=str, help="Optional output filename prefix.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result, _ = measure_case(args.point, args.line, args.raw)

    prefix = args.prefix
    if not prefix:
        prefix = normalize_name(args.point.stem).replace("'", "").replace(" ", "_")

    e_path = args.out_dir / f"{prefix}_{UPPER_ANGLE_LABEL}.jpg"
    g_path = args.out_dir / f"{prefix}_{LOWER_ANGLE_LABEL}.jpg"
    combined_path = args.out_dir / f"{prefix}_combined.jpg"
    cv2.imwrite(str(e_path), result["e_image"])
    cv2.imwrite(str(g_path), result["g_image"])
    cv2.imwrite(str(combined_path), result["combined_image"])

    print(f"raw image : {result['raw_path']}")
    print(f"{UPPER_ANGLE_LABEL:<10}: {result['mldfa_angle']:.2f} deg")
    print(f"{LOWER_ANGLE_LABEL:<10}: {result['mpta_angle']:.2f} deg")
    print(f"{UPPER_ANGLE_FULL_NAME}")
    print(f"{LOWER_ANGLE_FULL_NAME}")
    print(f"{UPPER_ANGLE_LABEL} output : {e_path}")
    print(f"{LOWER_ANGLE_LABEL} output : {g_path}")
    print(f"Combined  : {combined_path}")


if __name__ == "__main__":
    main()

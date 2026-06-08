#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from knee_dataset_utils import annotation_keypoints, load_manifest, read_json
from measure_angles import measure_from_named_points
from train_keypoint_baseline import (
    KEYPOINT_NAMES,
    SmallHeatmapNet,
    coords_to_measurement_payload,
    decode_heatmaps,
    preprocess_xray,
    select_device,
    split_by_case,
)


POINT_COLOR_GT = (0, 220, 255)
POINT_COLOR_PRED = (0, 80, 255)
LINE_COLOR_GT = (0, 220, 255)
LINE_COLOR_PRED = (0, 80, 255)
TEXT_COLOR = (255, 255, 255)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SmallHeatmapNet, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    keypoint_names = tuple(checkpoint.get("keypoint_names", KEYPOINT_NAMES))
    if keypoint_names != KEYPOINT_NAMES:
        raise ValueError(f"Checkpoint keypoints do not match current model: {keypoint_names}")
    model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_keypoints(
    model: SmallHeatmapNet,
    row: dict[str, str],
    device: torch.device,
    image_width: int,
    image_height: int,
    stride: int,
) -> np.ndarray:
    image = preprocess_xray(Path(row["raw_path"]), image_width, image_height)
    tensor = torch.from_numpy(image[None, None, ...]).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
    return decode_heatmaps(logits, row, image_width, image_height, stride)


def line_pairs(coords: np.ndarray) -> list[tuple[int, int]]:
    name_to_index = {name: index for index, name in enumerate(KEYPOINT_NAMES)}
    return [
        (name_to_index["upper_line_p1"], name_to_index["upper_line_p2"]),
        (name_to_index["lower_line_p1"], name_to_index["lower_line_p2"]),
    ]


def draw_keypoints(
    image: np.ndarray,
    coords: np.ndarray,
    color: tuple[int, int, int],
    label_prefix: str,
    radius: int,
    line_width: int,
) -> None:
    for idx_a, idx_b in line_pairs(coords):
        a = tuple(np.round(coords[idx_a]).astype(int))
        b = tuple(np.round(coords[idx_b]).astype(int))
        cv2.line(image, a, b, color, line_width, cv2.LINE_AA)

    for index, (name, point) in enumerate(zip(KEYPOINT_NAMES, coords)):
        center = tuple(np.round(point).astype(int))
        cv2.circle(image, center, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{label_prefix}{index + 1}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_summary_box(image: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    line_height = 34
    pad = 14
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_w = max(widths) + pad * 2
    box_h = line_height * len(lines) + pad
    cv2.rectangle(image, (12, 12), (12 + box_w, 12 + box_h), (24, 24, 24), -1)
    cv2.rectangle(image, (12, 12), (12 + box_w, 12 + box_h), (240, 240, 240), 2)
    for idx, line in enumerate(lines):
        y = 12 + pad + 22 + idx * line_height
        cv2.putText(image, line, (12 + pad, y), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def safe_measure(
    raw_image: np.ndarray,
    coords: np.ndarray,
    row: dict[str, str],
) -> tuple[float, float]:
    points, lines = coords_to_measurement_payload(coords)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result, _debug = measure_from_named_points(
                raw_image,
                points,
                raw_path=Path(row["raw_path"]),
                named_lines=lines,
                side=row["side"],
            )
        return float(result["mldfa_angle"]), float(result["mpta_angle"])
    except Exception:
        return float("nan"), float("nan")


def finite_abs_error(pred: float, gt: float) -> float:
    error = abs(pred - gt)
    return error if math.isfinite(error) else float("nan")


def format_float(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def select_rows(
    rows: list[dict[str, str]],
    split: str,
    num_folds: int,
    fold: int,
    seed: int,
    sample_ids: list[str] | None,
    max_images: int,
) -> list[dict[str, str]]:
    if sample_ids:
        wanted = set(sample_ids)
        selected = [row for row in rows if row["sample_id"] in wanted]
    elif split == "all":
        selected = rows
    else:
        train_indices, val_indices = split_by_case(rows, num_folds, fold, seed)
        indices = train_indices if split == "train" else val_indices
        selected = [rows[index] for index in indices]
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize baseline keypoint predictions on raw knee X-rays.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/knee_dataset_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/knee_keypoint_baseline/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/knee_keypoint_visualizations"))
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-id", action="append", default=None, help="Visualize one sample id. Can be repeated.")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    image_width = int(checkpoint["image_width"])
    image_height = int(checkpoint["image_height"])
    stride = int(checkpoint["stride"])

    rows = load_manifest(args.manifest)
    selected_rows = select_rows(rows, args.split, args.num_folds, args.fold, args.seed, args.sample_id, args.max_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "prediction_metrics.csv"
    metric_rows: list[dict[str, str]] = []

    for row in selected_rows:
        raw_image = cv2.imread(row["raw_path"])
        if raw_image is None:
            raise ValueError(f"Cannot read raw image: {row['raw_path']}")
        annotation = read_json(Path(row["annotation_path"]))
        gt_map = annotation_keypoints(annotation)
        gt_coords = np.array([gt_map[name] for name in KEYPOINT_NAMES], dtype=np.float32)
        pred_coords = predict_keypoints(model, row, device, image_width, image_height, stride)
        point_errors = np.linalg.norm(pred_coords - gt_coords, axis=1)

        gt_mldfa = float(row["mldfa"])
        gt_mpta = float(row["mpta"])
        pred_mldfa, pred_mpta = safe_measure(raw_image, pred_coords, row)
        mldfa_error = finite_abs_error(pred_mldfa, gt_mldfa)
        mpta_error = finite_abs_error(pred_mpta, gt_mpta)

        overlay = raw_image.copy()
        draw_keypoints(overlay, gt_coords, POINT_COLOR_GT, "G", radius=5, line_width=2)
        draw_keypoints(overlay, pred_coords, POINT_COLOR_PRED, "P", radius=5, line_width=2)
        draw_summary_box(
            overlay,
            [
                f"{row['sample_id']} side={row['side']}",
                f"point MAE={point_errors.mean():.1f}px",
                f"mLDFA pred/GT/err={format_float(pred_mldfa)}/{format_float(gt_mldfa)}/{format_float(mldfa_error)}",
                f"MPTA pred/GT/err={format_float(pred_mpta)}/{format_float(gt_mpta)}/{format_float(mpta_error)}",
                "GT=yellow  Pred=red",
            ],
        )
        out_path = args.output_dir / f"{row['sample_id']}_prediction.jpg"
        cv2.imwrite(str(out_path), overlay)

        metric_rows.append(
            {
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "side": row["side"],
                "point_mae_px": f"{point_errors.mean():.4f}",
                "point_median_px": f"{np.median(point_errors):.4f}",
                "pred_mldfa": format_float(pred_mldfa, 6),
                "gt_mldfa": format_float(gt_mldfa, 6),
                "mldfa_abs_error": format_float(mldfa_error, 6),
                "pred_mpta": format_float(pred_mpta, 6),
                "gt_mpta": format_float(gt_mpta, 6),
                "mpta_abs_error": format_float(mpta_error, 6),
                "image_path": str(out_path),
            }
        )

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()) if metric_rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"Visualized {len(metric_rows)} samples into: {args.output_dir}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Shared keypoint-model architecture and tensor conversion helpers.

This module is intentionally free of GUI concerns.  Training code and runtime
adapters import the same architecture so a state-dict checkpoint cannot drift
away from the model used by the desktop application.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from measure_angles import ANNOTATION_POINT_NAMES, read_color


CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE_ID = "small_heatmap_v1"
ADAPTER_ID = "small_heatmap_v1"
PREPROCESSING_ID = "grayscale_resize_percentile_1_99_v1"
KEYPOINT_NAMES = (
    *ANNOTATION_POINT_NAMES,
    "upper_line_p1",
    "upper_line_p2",
    "lower_line_p1",
    "lower_line_p2",
)
POINT_NAME_SET = set(ANNOTATION_POINT_NAMES)
LINE_KEYPOINT_TO_LINE = {
    "upper_line_p1": ("upper_line", "p1"),
    "upper_line_p2": ("upper_line", "p2"),
    "lower_line_p1": ("lower_line", "p1"),
    "lower_line_p2": ("lower_line", "p2"),
}


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_xray_array(image: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Apply the exact grayscale/resize/percentile normalization used in training."""

    if image is None or image.size == 0:
        raise ValueError("入力されたX線画像が空です。")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"未対応の画像配列形状です：{image.shape}")

    resized = cv2.resize(gray, (image_width, image_height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32)
    low, high = np.percentile(normalized, [1.0, 99.0])
    if high <= low:
        low, high = float(normalized.min()), float(normalized.max())
    return np.clip((normalized - low) / max(high - low, 1e-6), 0.0, 1.0)


def preprocess_xray(path: Path, image_width: int, image_height: int) -> np.ndarray:
    """Unicode-safe path wrapper retained for the training/evaluation scripts."""

    return preprocess_xray_array(read_color(Path(path)), image_width, image_height)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallHeatmapNet(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
        )
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def decode_heatmaps_for_shape(
    logits: torch.Tensor,
    original_width: int,
    original_height: int,
    image_width: int,
    image_height: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps to original-image pixel coordinates and peak scores."""

    heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
    coords = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
    scores = np.zeros((len(KEYPOINT_NAMES),), dtype=np.float32)
    for idx, heatmap in enumerate(heatmaps):
        flat_index = int(np.argmax(heatmap))
        y, x = np.unravel_index(flat_index, heatmap.shape)
        image_x = float(x * stride)
        image_y = float(y * stride)
        coords[idx, 0] = image_x * float(original_width) / image_width
        coords[idx, 1] = image_y * float(original_height) / image_height
        scores[idx] = float(heatmap[y, x])
    return coords, scores


def decode_heatmaps(
    logits: torch.Tensor,
    row: dict[str, str],
    image_width: int,
    image_height: int,
    stride: int,
) -> np.ndarray:
    coords, _scores = decode_heatmaps_for_shape(
        logits,
        original_width=int(row["image_width"]),
        original_height=int(row["image_height"]),
        image_width=image_width,
        image_height=image_height,
        stride=stride,
    )
    return coords


def coords_to_measurement_payload(
    coords: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    if coords.shape != (len(KEYPOINT_NAMES), 2):
        raise ValueError(f"座標配列の形状は ({len(KEYPOINT_NAMES)}, 2) が必要ですが、{coords.shape} でした。")

    named_points: dict[str, dict[str, float]] = {}
    named_lines: dict[str, dict[str, dict[str, float]]] = {}
    for name, (x, y) in zip(KEYPOINT_NAMES, coords):
        point = {"x": float(x), "y": float(y)}
        if name in POINT_NAME_SET:
            named_points[name] = point
        else:
            line_name, endpoint = LINE_KEYPOINT_TO_LINE[name]
            named_lines.setdefault(line_name, {})[endpoint] = point
    return named_points, named_lines

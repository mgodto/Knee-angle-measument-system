#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from knee_dataset_utils import annotation_keypoints, load_manifest, read_json
from measure_angles import ANNOTATION_POINT_NAMES, measure_from_named_points


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_case(rows: list[dict[str, str]], num_folds: int, fold: int, seed: int) -> tuple[list[int], list[int]]:
    case_ids = sorted({row["case_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    val_cases = set(case_ids[fold::num_folds])
    train_indices = [idx for idx, row in enumerate(rows) if row["case_id"] not in val_cases]
    val_indices = [idx for idx, row in enumerate(rows) if row["case_id"] in val_cases]
    return train_indices, val_indices


def preprocess_xray(path: Path, image_width: int, image_height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    image = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
    return image


def make_heatmaps(
    keypoints: np.ndarray,
    orig_width: int,
    orig_height: int,
    image_width: int,
    image_height: int,
    stride: int,
    sigma: float,
) -> np.ndarray:
    heatmap_h = image_height // stride
    heatmap_w = image_width // stride
    scale_x = image_width / orig_width / stride
    scale_y = image_height / orig_height / stride
    xs = np.arange(heatmap_w, dtype=np.float32)[None, :]
    ys = np.arange(heatmap_h, dtype=np.float32)[:, None]
    heatmaps = np.zeros((len(KEYPOINT_NAMES), heatmap_h, heatmap_w), dtype=np.float32)
    for idx, (x, y) in enumerate(keypoints):
        cx = float(x) * scale_x
        cy = float(y) * scale_y
        heatmaps[idx] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
    return heatmaps


class KneeKeypointDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        indices: list[int],
        image_width: int,
        image_height: int,
        stride: int,
        sigma: float,
    ) -> None:
        self.rows = rows
        self.indices = indices
        self.image_width = image_width
        self.image_height = image_height
        self.stride = stride
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, dataset_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_index = self.indices[dataset_index]
        row = self.rows[row_index]
        annotation = read_json(Path(row["annotation_path"]))
        keypoint_map = annotation_keypoints(annotation)
        keypoints = np.array([keypoint_map[name] for name in KEYPOINT_NAMES], dtype=np.float32)
        image = preprocess_xray(Path(row["raw_path"]), self.image_width, self.image_height)
        heatmaps = make_heatmaps(
            keypoints,
            int(row["image_width"]),
            int(row["image_height"]),
            self.image_width,
            self.image_height,
            self.stride,
            self.sigma,
        )
        return (
            torch.from_numpy(image[None, ...]),
            torch.from_numpy(heatmaps),
            torch.tensor(row_index, dtype=torch.long),
        )


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


def decode_heatmaps(
    logits: torch.Tensor,
    row: dict[str, str],
    image_width: int,
    image_height: int,
    stride: int,
) -> np.ndarray:
    heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
    coords = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
    orig_w = float(row["image_width"])
    orig_h = float(row["image_height"])
    for idx, heatmap in enumerate(heatmaps):
        flat_index = int(np.argmax(heatmap))
        y, x = np.unravel_index(flat_index, heatmap.shape)
        image_x = float(x * stride)
        image_y = float(y * stride)
        coords[idx, 0] = image_x * orig_w / image_width
        coords[idx, 1] = image_y * orig_h / image_height
    return coords


def coords_to_measurement_payload(coords: np.ndarray) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    rows: list[dict[str, str]],
    device: torch.device,
    image_width: int,
    image_height: int,
    stride: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    point_errors: list[float] = []
    mldfa_errors: list[float] = []
    mpta_errors: list[float] = []
    with torch.no_grad():
        for images, targets, row_indices in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = F.mse_loss(torch.sigmoid(logits), targets)
            losses.append(float(loss.detach().cpu()))
            for batch_idx, row_index_tensor in enumerate(row_indices):
                row_index = int(row_index_tensor)
                row = rows[row_index]
                annotation = read_json(Path(row["annotation_path"]))
                target_keypoints = annotation_keypoints(annotation)
                target_coords = np.array([target_keypoints[name] for name in KEYPOINT_NAMES], dtype=np.float32)
                pred_coords = decode_heatmaps(logits[batch_idx], row, image_width, image_height, stride)
                point_errors.extend(np.linalg.norm(pred_coords - target_coords, axis=1).tolist())

                try:
                    raw_image = cv2.imread(row["raw_path"])
                    pred_points, pred_lines = coords_to_measurement_payload(pred_coords)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result, _debug = measure_from_named_points(
                            raw_image,
                            pred_points,
                            raw_path=Path(row["raw_path"]),
                            named_lines=pred_lines,
                            side=row["side"],
                        )
                    mldfa_error = abs(float(result["mldfa_angle"]) - float(row["mldfa"]))
                    mpta_error = abs(float(result["mpta_angle"]) - float(row["mpta"]))
                    if math.isfinite(mldfa_error):
                        mldfa_errors.append(mldfa_error)
                    if math.isfinite(mpta_error):
                        mpta_errors.append(mpta_error)
                except Exception:
                    continue

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "point_mae_px": float(np.mean(point_errors)) if point_errors else float("nan"),
        "mldfa_mae_deg": float(np.mean(mldfa_errors)) if mldfa_errors else float("nan"),
        "mpta_mae_deg": float(np.mean(mpta_errors)) if mpta_errors else float("nan"),
    }


def format_metric(value: float, precision: int, suffix: str) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{precision}f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small heatmap baseline for knee X-ray keypoint detection.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/knee_dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/knee_keypoint_baseline"))
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=320)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional smoke-test limit.")
    args = parser.parse_args()

    if args.image_width % args.stride != 0 or args.image_height % args.stride != 0:
        raise ValueError("--image-width and --image-height must be divisible by --stride")
    set_seed(args.seed)
    device = select_device(args.device)
    rows = load_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    train_indices, val_indices = split_by_case(rows, args.num_folds, args.fold, args.seed)
    if not train_indices or not val_indices:
        raise ValueError("Train/validation split is empty. Reduce --num-folds or remove --max-samples.")

    train_dataset = KneeKeypointDataset(rows, train_indices, args.image_width, args.image_height, args.stride, args.sigma)
    val_dataset = KneeKeypointDataset(rows, val_indices, args.image_width, args.image_height, args.stride, args.sigma)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    log_path = args.output_dir / "train_log.csv"
    best_val = float("inf")
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_point_mae_px", "val_mldfa_mae_deg", "val_mpta_mae_deg"],
        )
        log_writer.writeheader()
        print(f"Device: {device}")
        print(f"Train samples: {len(train_indices)}, val samples: {len(val_indices)}")
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for images, targets, _row_indices in train_loader:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = F.mse_loss(torch.sigmoid(logits), targets)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            val_metrics = evaluate(model, val_loader, rows, device, args.image_width, args.image_height, args.stride)
            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            log_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8f}",
                    "val_loss": f"{val_metrics['loss']:.8f}",
                    "val_point_mae_px": f"{val_metrics['point_mae_px']:.4f}",
                    "val_mldfa_mae_deg": f"{val_metrics['mldfa_mae_deg']:.4f}",
                    "val_mpta_mae_deg": f"{val_metrics['mpta_mae_deg']:.4f}",
                }
            )
            log_file.flush()
            print(
                f"epoch {epoch:03d} train={train_loss:.6f} val={val_metrics['loss']:.6f} "
                f"point={format_metric(val_metrics['point_mae_px'], 1, 'px')} "
                f"mLDFA={format_metric(val_metrics['mldfa_mae_deg'], 2, 'deg')} "
                f"MPTA={format_metric(val_metrics['mpta_mae_deg'], 2, 'deg')}"
            )
            if val_metrics["point_mae_px"] < best_val:
                best_val = val_metrics["point_mae_px"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "keypoint_names": KEYPOINT_NAMES,
                        "image_width": args.image_width,
                        "image_height": args.image_height,
                        "stride": args.stride,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    args.output_dir / "best.pt",
                )

    print(f"Saved log: {log_path}")
    print(f"Saved best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

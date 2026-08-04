#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from knee_dataset_utils import annotation_keypoints, dataset_manifest_path, load_manifest, read_json
from measure_angles import measure_from_named_points
from knee_keypoint_model import (
    ADAPTER_ID,
    ARCHITECTURE_ID,
    CHECKPOINT_SCHEMA_VERSION,
    HARD_ARGMAX_DECODER_ID,
    KEYPOINT_NAMES,
    SUPPORTED_DECODER_IDS,
    PREPROCESSING_ID,
    SmallHeatmapNet,
    coords_to_measurement_payload,
    decode_heatmaps,
    preprocess_xray,
    select_device,
)


CASE_ID_HASH_ALGORITHM = "sha256_json_sorted_unique_utf8_v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_ids_sha256(case_ids: Iterable[str]) -> str:
    normalized = sorted({str(case_id) for case_id in case_ids})
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def case_split_provenance(
    rows: list[dict[str, str]],
    train_indices: list[int],
    val_indices: list[int],
    num_folds: int,
    fold: int,
    seed: int,
) -> dict[str, int | str]:
    train_case_ids = {rows[index]["case_id"] for index in train_indices}
    val_case_ids = {rows[index]["case_id"] for index in val_indices}
    return {
        "num_folds": int(num_folds),
        "fold": int(fold),
        "seed": int(seed),
        "case_id_hash_algorithm": CASE_ID_HASH_ALGORITHM,
        "train_case_count": len(train_case_ids),
        "val_case_count": len(val_case_ids),
        "train_case_ids_sha256": case_ids_sha256(train_case_ids),
        "val_case_ids_sha256": case_ids_sha256(val_case_ids),
    }


def load_cross_validation_metrics(path: Path | None, manifest_sha256: str) -> tuple[dict[str, float], str | None]:
    if path is None:
        return {}, None
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary_manifest_sha = str(summary.get("manifest", {}).get("sha256", ""))
    if summary_manifest_sha != manifest_sha256:
        raise ValueError("--cv-summary was generated from a different training manifest")
    if not bool(summary.get("cross_validation", {}).get("complete_5_fold")):
        raise ValueError("--cv-summary must contain a complete 5-fold evaluation")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("verified") is not True:
        raise ValueError("--cv-summary must contain verified fold provenance")
    if provenance.get("complete_partition_verified") is not True:
        raise ValueError("--cv-summary must contain a verified complete case partition")
    metrics: dict[str, float] = {}
    for name, values in dict(summary.get("metrics") or {}).items():
        value = values.get("sample_weighted_mean") if isinstance(values, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[str(name)] = float(value)
    return metrics, sha256_path(path)


def split_by_case(rows: list[dict[str, str]], num_folds: int, fold: int, seed: int) -> tuple[list[int], list[int]]:
    case_ids = sorted({row["case_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    val_cases = set(case_ids[fold::num_folds])
    train_indices = [idx for idx, row in enumerate(rows) if row["case_id"] not in val_cases]
    val_indices = [idx for idx, row in enumerate(rows) if row["case_id"] in val_cases]
    return train_indices, val_indices


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


def heatmap_mse_loss(logits: torch.Tensor, targets: torch.Tensor, peak_weight: float = 1.0) -> torch.Tensor:
    predictions = torch.sigmoid(logits)
    if peak_weight == 1.0:
        return F.mse_loss(predictions, targets)
    weights = 1.0 + (peak_weight - 1.0) * targets
    per_heatmap = (weights * (predictions - targets).square()).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1))
    return per_heatmap.mean()


def complete_angle_checkpoint_score(
    metrics: dict[str, float | int],
    total_samples: int,
) -> float | None:
    """Return the mean angle MAE only when every validation sample is valid."""

    if total_samples < 1:
        return None
    values: list[float] = []
    for name in ("mldfa", "mpta"):
        value = metrics.get(f"{name}_mae_deg")
        valid_n = metrics.get(f"{name}_valid_n")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        if not isinstance(valid_n, (int, float)) or int(valid_n) != total_samples:
            return None
        values.append(float(value))
    return float(np.mean(values))


class KneeKeypointDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        indices: list[int],
        image_width: int,
        image_height: int,
        stride: int,
        sigma: float,
        cache: bool = False,
    ) -> None:
        self.rows = rows
        self.indices = indices
        self.image_width = image_width
        self.image_height = image_height
        self.stride = stride
        self.sigma = sigma
        self.cache = cache
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, dataset_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache and dataset_index in self._cache:
            return self._cache[dataset_index]

        row_index = self.indices[dataset_index]
        row = self.rows[row_index]
        annotation = read_json(Path(row["annotation_path"]))
        keypoint_map = annotation_keypoints(annotation, canonicalize_line_endpoints=True)
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
        sample = (
            torch.from_numpy(image[None, ...]),
            torch.from_numpy(heatmaps),
            torch.tensor(row_index, dtype=torch.long),
        )
        if self.cache:
            self._cache[dataset_index] = sample
        return sample


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    rows: list[dict[str, str]],
    device: torch.device,
    image_width: int,
    image_height: int,
    stride: int,
    heatmap_peak_weight: float = 1.0,
    decoder_id: str = HARD_ARGMAX_DECODER_ID,
) -> dict[str, float | int]:
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
            loss = heatmap_mse_loss(logits, targets, heatmap_peak_weight)
            losses.append(float(loss.detach().cpu()))
            for batch_idx, row_index_tensor in enumerate(row_indices):
                row_index = int(row_index_tensor)
                row = rows[row_index]
                annotation = read_json(Path(row["annotation_path"]))
                target_keypoints = annotation_keypoints(annotation, canonicalize_line_endpoints=True)
                target_coords = np.array([target_keypoints[name] for name in KEYPOINT_NAMES], dtype=np.float32)
                pred_coords = decode_heatmaps(
                    logits[batch_idx],
                    row,
                    image_width,
                    image_height,
                    stride,
                    decoder_id,
                )
                point_errors.extend(np.linalg.norm(pred_coords - target_coords, axis=1).tolist())

                try:
                    # Angle calculations depend only on coordinates and image
                    # dimensions. Avoid decoding the full-resolution X-ray for
                    # every validation sample and epoch.
                    raw_image = np.zeros((1, 1, 3), dtype=np.uint8)
                    pred_points, pred_lines = coords_to_measurement_payload(pred_coords)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result, _debug = measure_from_named_points(
                            raw_image,
                            pred_points,
                            raw_path=Path(row["raw_path"]),
                            named_lines=pred_lines,
                            side=row["side"],
                            render_component_images=False,
                        )
                    mldfa_error = abs(float(result["mldfa_angle"]) - float(row["mldfa"]))
                    mpta_error = abs(float(result["mpta_angle"]) - float(row["mpta"]))
                    if math.isfinite(mldfa_error):
                        mldfa_errors.append(mldfa_error)
                    if math.isfinite(mpta_error):
                        mpta_errors.append(mpta_error)
                except Exception:
                    continue

    total_samples = len(loader.dataset)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "point_mae_px": float(np.mean(point_errors)) if point_errors else float("nan"),
        "mldfa_mae_deg": float(np.mean(mldfa_errors)) if mldfa_errors else float("nan"),
        "mldfa_valid_n": len(mldfa_errors),
        "mldfa_failure_count": total_samples - len(mldfa_errors),
        "mpta_mae_deg": float(np.mean(mpta_errors)) if mpta_errors else float("nan"),
        "mpta_valid_n": len(mpta_errors),
        "mpta_failure_count": total_samples - len(mpta_errors),
    }


def format_metric(value: float, precision: int, suffix: str) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{precision}f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small heatmap baseline for knee X-ray keypoint detection.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, help="Dataset folder that contains manifest.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/knee_keypoint_baseline"))
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=320)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--heatmap-peak-weight",
        type=float,
        default=1.0,
        help="Relative MSE weight at the center of each target heatmap (1 disables weighting).",
    )
    parser.add_argument(
        "--decoder-id",
        choices=SUPPORTED_DECODER_IDS,
        default=HARD_ARGMAX_DECODER_ID,
        help="Versioned heatmap-to-coordinate decoder stored in the checkpoint.",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--model-version", default="", help="Optional deployment model version stored in checkpoints.")
    parser.add_argument("--model-scope", default="unspecified", help="Deployment cohort/scope stored in checkpoints.")
    parser.add_argument("--cache-dataset", action="store_true", help="Cache resized images and heatmaps in memory.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run full validation every N epochs.")
    parser.add_argument("--train-all", action="store_true", help="Train on every manifest row and save final.pt.")
    parser.add_argument("--cv-summary", type=Path, default=None, help="Complete CV summary stored with --train-all.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional smoke-test limit.")
    args = parser.parse_args()

    if args.image_width % args.stride != 0 or args.image_height % args.stride != 0:
        raise ValueError("--image-width and --image-height must be divisible by --stride")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be at least 1")
    if args.heatmap_peak_weight < 1.0:
        raise ValueError("--heatmap-peak-weight must be at least 1")
    set_seed(args.seed)
    device = select_device(args.device)
    args.manifest = dataset_manifest_path(args.dataset_dir, args.manifest, Path("outputs/knee_dataset_manifest.csv"))
    manifest_sha256 = sha256_path(args.manifest)
    if args.cv_summary is not None and not args.train_all:
        raise ValueError("--cv-summary requires --train-all")
    cv_metrics, cv_summary_sha256 = load_cross_validation_metrics(args.cv_summary, manifest_sha256)
    rows = load_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if args.train_all:
        train_indices, val_indices = list(range(len(rows))), []
    else:
        train_indices, val_indices = split_by_case(rows, args.num_folds, args.fold, args.seed)
    if not train_indices or (not args.train_all and not val_indices):
        raise ValueError("Train/validation split is empty. Reduce --num-folds or remove --max-samples.")
    split_provenance = None
    if not args.train_all:
        split_provenance = case_split_provenance(
            rows,
            train_indices,
            val_indices,
            args.num_folds,
            args.fold,
            args.seed,
        )

    train_dataset = KneeKeypointDataset(
        rows,
        train_indices,
        args.image_width,
        args.image_height,
        args.stride,
        args.sigma,
        cache=args.cache_dataset,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    if args.train_all:
        val_loader = None
    else:
        val_dataset = KneeKeypointDataset(
            rows,
            val_indices,
            args.image_width,
            args.image_height,
            args.stride,
            args.sigma,
            cache=args.cache_dataset,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    log_path = args.output_dir / "train_log.csv"
    best_val = float("inf")
    best_angle_val = float("inf")
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
                loss = heatmap_mse_loss(logits, targets, args.heatmap_peak_weight)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            should_evaluate = not args.train_all and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs)
            if not should_evaluate:
                log_writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": f"{train_loss:.8f}",
                        "val_loss": "",
                        "val_point_mae_px": "",
                        "val_mldfa_mae_deg": "",
                        "val_mpta_mae_deg": "",
                    }
                )
                log_file.flush()
                print(f"epoch {epoch:03d} train={train_loss:.6f}")
                continue

            assert val_loader is not None
            val_metrics = evaluate(
                model,
                val_loader,
                rows,
                device,
                args.image_width,
                args.image_height,
                args.stride,
                args.heatmap_peak_weight,
                args.decoder_id,
            )
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
            point_improved = val_metrics["point_mae_px"] < best_val
            angle_score = complete_angle_checkpoint_score(val_metrics, len(val_indices))
            angle_improved = angle_score is not None and angle_score < best_angle_val
            if point_improved or angle_improved:
                checkpoint = {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "adapter_id": ADAPTER_ID,
                    "architecture_id": ARCHITECTURE_ID,
                    "preprocessing_id": PREPROCESSING_ID,
                    "prediction_schema_version": 1,
                    "decoder_id": args.decoder_id,
                    "training_mode": "cross_validation_fold",
                    "model_version": args.model_version or None,
                    "model_scope": args.model_scope,
                    "training_manifest_sha256": manifest_sha256,
                    "split_provenance": split_provenance,
                    "model_state": model.state_dict(),
                    "keypoint_names": KEYPOINT_NAMES,
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "stride": args.stride,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }
                if point_improved:
                    best_val = val_metrics["point_mae_px"]
                    torch.save(checkpoint, args.output_dir / "best.pt")
                if angle_improved:
                    best_angle_val = angle_score
                    torch.save(checkpoint, args.output_dir / "best_angles.pt")

    print(f"Saved log: {log_path}")
    if args.train_all:
        final_path = args.output_dir / "final.pt"
        torch.save(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "adapter_id": ADAPTER_ID,
                "architecture_id": ARCHITECTURE_ID,
                "preprocessing_id": PREPROCESSING_ID,
                "prediction_schema_version": 1,
                "decoder_id": args.decoder_id,
                "training_mode": "all_samples",
                "model_version": args.model_version or None,
                "model_scope": args.model_scope,
                "training_manifest_sha256": manifest_sha256,
                "cross_validation_summary_sha256": cv_summary_sha256,
                "model_state": model.state_dict(),
                "keypoint_names": KEYPOINT_NAMES,
                "image_width": args.image_width,
                "image_height": args.image_height,
                "stride": args.stride,
                "epoch": args.epochs,
                "val_metrics": cv_metrics,
            },
            final_path,
        )
        print(f"Saved final checkpoint: {final_path}")
    else:
        print(f"Saved best checkpoint: {args.output_dir / 'best.pt'}")
        best_angle_path = args.output_dir / "best_angles.pt"
        if best_angle_path.exists():
            print(f"Saved best-angle checkpoint: {best_angle_path}")


if __name__ == "__main__":
    main()

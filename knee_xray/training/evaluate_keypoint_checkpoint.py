#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knee_xray.core.measure_angles import measure_from_named_points, read_color
from knee_xray.data.knee_dataset_utils import annotation_keypoints, load_manifest, read_json
from knee_xray.ml.knee_keypoint_model import (
    ADAPTER_ID,
    ARCHITECTURE_ID,
    HARD_ARGMAX_DECODER_ID,
    KEYPOINT_NAMES,
    PREPROCESSING_ID,
    SmallHeatmapNet,
    SUPPORTED_DECODER_IDS,
    coords_to_measurement_payload,
    decode_heatmaps,
    preprocess_xray_array,
)
from knee_xray.training.train_keypoint_baseline import case_split_provenance, split_by_case


ANGLE_NAMES = ("mldfa", "mpta", "jlca", "hka")
ANGLE_RESULT_KEYS = {
    "mldfa": "mldfa_angle",
    "mpta": "mpta_angle",
    "jlca": "jlca_angle",
    "hka": "hka_angle",
}
PER_SAMPLE_FIELDS = (
    "sample_id",
    "case_id",
    "side",
    "image_width",
    "image_height",
    "point_mae_px",
    "nme_height_pct",
    "mldfa_gt_deg",
    "mldfa_pred_deg",
    "mldfa_abs_error_deg",
    "mldfa_gt_source",
    "mpta_gt_deg",
    "mpta_pred_deg",
    "mpta_abs_error_deg",
    "mpta_gt_source",
    "jlca_gt_deg",
    "jlca_pred_deg",
    "jlca_abs_error_deg",
    "jlca_gt_source",
    "hka_gt_deg",
    "hka_pred_deg",
    "hka_abs_error_deg",
    "hka_gt_source",
    "prediction_measurement_error",
    "ground_truth_measurement_error",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available in this PyTorch environment.")
    return torch.device(value)


def load_small_heatmap_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SmallHeatmapNet, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch 2.0 or newer is required for safe checkpoint loading.") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary.")

    keypoint_names = tuple(checkpoint.get("keypoint_names", ()))
    if keypoint_names != KEYPOINT_NAMES:
        raise ValueError(f"Checkpoint keypoints do not match {KEYPOINT_NAMES}: {keypoint_names}")

    architecture_id = str(checkpoint.get("architecture_id", ARCHITECTURE_ID))
    adapter_id = str(checkpoint.get("adapter_id", ADAPTER_ID))
    preprocessing_id = str(checkpoint.get("preprocessing_id", PREPROCESSING_ID))
    if architecture_id != ARCHITECTURE_ID or adapter_id != ADAPTER_ID:
        raise ValueError(
            f"Checkpoint is not {ARCHITECTURE_ID}: architecture={architecture_id}, adapter={adapter_id}"
        )
    if preprocessing_id != PREPROCESSING_ID:
        raise ValueError(
            f"Checkpoint preprocessing is {preprocessing_id}, expected {PREPROCESSING_ID}."
        )
    decoder_id = str(checkpoint.get("decoder_id", HARD_ARGMAX_DECODER_ID))
    if decoder_id not in SUPPORTED_DECODER_IDS:
        raise ValueError(f"Unsupported checkpoint decoder: {decoder_id}")

    try:
        image_width = int(checkpoint["image_width"])
        image_height = int(checkpoint["image_height"])
        stride = int(checkpoint["stride"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint must contain integer image_width, image_height, and stride.") from exc
    if image_width <= 0 or image_height <= 0 or stride <= 0:
        raise ValueError("Checkpoint image dimensions and stride must be positive.")
    if image_width % stride or image_height % stride:
        raise ValueError("Checkpoint image dimensions must be divisible by stride.")

    model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES))
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError(f"Checkpoint model_state is incompatible with {ARCHITECTURE_ID}: {exc}") from exc
    model.to(device)
    model.eval()
    return model, checkpoint


def select_manifest_rows(
    rows: list[dict[str, str]],
    split: str,
    num_folds: int,
    fold: int,
    seed: int,
) -> list[dict[str, str]]:
    if split == "all":
        return list(rows)
    if split != "val":
        raise ValueError(f"Unsupported split: {split}")
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2 for a validation split.")
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"--fold must be between 0 and {num_folds - 1}.")
    _train_indices, val_indices = split_by_case(rows, num_folds, fold, seed)
    return [rows[index] for index in val_indices]


def select_intervention_rows(
    evaluation_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    num_folds: int,
    fold: int,
    seed: int,
) -> list[dict[str, str]]:
    """Use a checkpoint's original split while evaluating changed image preprocessing."""

    evaluation_by_id: dict[str, dict[str, str]] = {}
    for row in evaluation_rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in evaluation_by_id:
            raise ValueError("Evaluation manifest sample_id values must be non-empty and unique.")
        evaluation_by_id[sample_id] = row

    reference_by_id: dict[str, dict[str, str]] = {}
    for row in reference_rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in reference_by_id:
            raise ValueError("Split-reference manifest sample_id values must be non-empty and unique.")
        reference_by_id[sample_id] = row

    if set(evaluation_by_id) != set(reference_by_id):
        missing = sorted(set(reference_by_id).difference(evaluation_by_id))
        extra = sorted(set(evaluation_by_id).difference(reference_by_id))
        raise ValueError(
            "Evaluation and split-reference manifests must contain the same sample_id set: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for sample_id, reference in reference_by_id.items():
        evaluation = evaluation_by_id[sample_id]
        for field in ("case_id", "side"):
            if evaluation.get(field, "") != reference.get(field, ""):
                raise ValueError(
                    f"Evaluation manifest {field} differs from the split reference for {sample_id}."
                )

    selected_reference = select_manifest_rows(
        reference_rows,
        split="val",
        num_folds=num_folds,
        fold=fold,
        seed=seed,
    )
    return [evaluation_by_id[row["sample_id"]] for row in selected_reference]


def normalized_optional_text(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def resolve_config_output_dir(
    config_path: Path,
    configured_value: object,
    checkpoint_dir: Path,
) -> tuple[Path, Path | None]:
    if not isinstance(configured_value, str) or not configured_value.strip():
        raise ValueError(f"Sibling config is missing output_dir: {config_path}")
    configured_path = Path(configured_value)
    if configured_path.is_absolute():
        resolved = configured_path.resolve()
        if resolved != checkpoint_dir:
            raise ValueError(
                f"Sibling config output_dir {resolved} does not match checkpoint directory {checkpoint_dir}."
            )
        return resolved, None

    roots = [Path.cwd(), config_path.parent, *config_path.parent.parents]
    matches: list[Path] = []
    for root in roots:
        root = root.resolve()
        if (root / configured_path).resolve() == checkpoint_dir and root not in matches:
            matches.append(root)
    if not matches:
        raise ValueError(
            f"Sibling config output_dir {configured_value!r} cannot be resolved to checkpoint directory "
            f"{checkpoint_dir}."
        )
    return checkpoint_dir, matches[0]


def verify_sibling_config(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    expected_split: dict[str, int | str],
) -> dict[str, object]:
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "Validation provenance is missing from the checkpoint and no sibling config.json was found: "
            f"{checkpoint_path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid sibling config JSON {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Sibling config root must be an object: {config_path}")

    _output_dir, training_root = resolve_config_output_dir(
        config_path,
        config.get("output_dir"),
        checkpoint_path.parent,
    )
    configured_manifest_value = config.get("manifest")
    if not isinstance(configured_manifest_value, str) or not configured_manifest_value.strip():
        raise ValueError(f"Sibling config is missing manifest: {config_path}")
    configured_manifest = Path(configured_manifest_value)
    if not configured_manifest.is_absolute():
        if training_root is None:
            raise ValueError(
                f"Cannot resolve relative sibling-config manifest without a training root: {config_path}"
            )
        configured_manifest = training_root / configured_manifest
    configured_manifest = configured_manifest.resolve()
    if configured_manifest != manifest_path:
        raise ValueError(
            f"Sibling config manifest {configured_manifest} does not match evaluated manifest {manifest_path}."
        )
    if not configured_manifest.is_file() or sha256_path(configured_manifest) != manifest_sha256:
        raise ValueError(f"Sibling config manifest SHA-256 does not match evaluated manifest: {config_path}")

    for field in ("num_folds", "fold", "seed"):
        try:
            actual = int(config[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Sibling config has invalid {field}: {config_path}") from exc
        if actual != expected_split[field]:
            raise ValueError(
                f"Requested validation {field}={expected_split[field]} does not match sibling config "
                f"{field}={actual}."
            )

    configured_version = normalized_optional_text(config.get("model_version"))
    checkpoint_version = normalized_optional_text(checkpoint.get("model_version"))
    if configured_version != checkpoint_version:
        raise ValueError(
            f"Sibling config model_version {configured_version!r} does not match checkpoint "
            f"model_version {checkpoint_version!r}."
        )
    configured_decoder = str(config.get("decoder_id", HARD_ARGMAX_DECODER_ID))
    checkpoint_decoder = str(checkpoint.get("decoder_id", HARD_ARGMAX_DECODER_ID))
    if configured_decoder != checkpoint_decoder:
        raise ValueError(
            f"Sibling config decoder_id {configured_decoder!r} does not match checkpoint "
            f"decoder_id {checkpoint_decoder!r}."
        )
    for field in ("image_width", "image_height", "stride"):
        try:
            configured_value = int(config[field])
            checkpoint_value = int(checkpoint[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot verify sibling config {field}: {config_path}") from exc
        if configured_value != checkpoint_value:
            raise ValueError(
                f"Sibling config {field}={configured_value} does not match checkpoint "
                f"{field}={checkpoint_value}."
            )
    return {
        "source": "sibling_config",
        "config_path": str(config_path),
        "config_sha256": sha256_path(config_path),
    }


def verify_validation_provenance(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    rows: list[dict[str, str]],
    num_folds: int,
    fold: int,
    seed: int,
) -> dict[str, object]:
    checkpoint_manifest_sha = checkpoint.get("training_manifest_sha256")
    if checkpoint_manifest_sha != manifest_sha256:
        raise ValueError(
            "Checkpoint training_manifest_sha256 does not match the evaluated manifest SHA-256."
        )
    train_indices, val_indices = split_by_case(rows, num_folds, fold, seed)
    expected = case_split_provenance(
        rows,
        train_indices,
        val_indices,
        num_folds,
        fold,
        seed,
    )

    stored = checkpoint.get("split_provenance")
    source: dict[str, object]
    if stored is None:
        source = verify_sibling_config(
            checkpoint_path,
            checkpoint,
            manifest_path,
            manifest_sha256,
            expected,
        )
    else:
        if not isinstance(stored, dict):
            raise ValueError("Checkpoint split_provenance must be an object.")
        for field, expected_value in expected.items():
            if stored.get(field) != expected_value:
                raise ValueError(
                    f"Requested validation split does not match checkpoint split_provenance: "
                    f"{field}={stored.get(field)!r}, expected {expected_value!r}."
                )
        source = {"source": "checkpoint_split_provenance"}

    return {
        "verified": True,
        "manifest_sha256": manifest_sha256,
        **expected,
        **source,
    }


def predict_keypoints(
    model: SmallHeatmapNet,
    row: dict[str, str],
    raw_image: np.ndarray,
    device: torch.device,
    image_width: int,
    image_height: int,
    stride: int,
    decoder_id: str,
) -> np.ndarray:
    image = preprocess_xray_array(raw_image, image_width, image_height)
    tensor = torch.from_numpy(image[None, None, ...]).to(device)
    with torch.inference_mode():
        logits = model(tensor)[0]
    return decode_heatmaps(logits, row, image_width, image_height, stride, decoder_id)


def measure_angles_for_coords(
    raw_image: np.ndarray,
    coords: np.ndarray,
    row: dict[str, str],
) -> tuple[dict[str, float | None], str]:
    empty = {name: None for name in ANGLE_NAMES}
    try:
        points, lines = coords_to_measurement_payload(coords)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result, _debug = measure_from_named_points(
                raw_image,
                points,
                raw_path=Path(row["raw_path"]),
                named_lines=lines,
                side=row.get("side") or None,
                render_component_images=False,
            )
    except Exception as exc:
        return empty, f"{type(exc).__name__}: {exc}"

    angles = {
        name: finite_float(result.get(ANGLE_RESULT_KEYS[name]))
        for name in ANGLE_NAMES
    }
    invalid = [name for name, value in angles.items() if value is None]
    error = f"Non-finite angles: {', '.join(invalid)}" if invalid else ""
    return angles, error


def ground_truth_angles(
    row: dict[str, str],
    measured_angles: dict[str, float | None],
) -> tuple[dict[str, float | None], dict[str, str]]:
    angles: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    for name in ANGLE_NAMES:
        manifest_value = None
        for field in (name, f"{name}_angle"):
            manifest_value = finite_float(row.get(field))
            if manifest_value is not None:
                break
        if manifest_value is not None:
            angles[name] = manifest_value
            sources[name] = "manifest"
        else:
            angles[name] = measured_angles[name]
            sources[name] = "annotation_geometry"
    return angles, sources


def metric_summary(values: list[float], total_count: int) -> dict[str, int | float | None]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "valid_n": len(values),
        "failure_count": total_count - len(values),
    }


def evaluate_rows(
    model: SmallHeatmapNet,
    checkpoint: dict[str, Any],
    rows: list[dict[str, str]],
    device: torch.device,
) -> tuple[dict[str, dict[str, int | float | None]], list[dict[str, object]]]:
    image_width = int(checkpoint["image_width"])
    image_height = int(checkpoint["image_height"])
    stride = int(checkpoint["stride"])
    decoder_id = str(checkpoint.get("decoder_id", HARD_ARGMAX_DECODER_ID))
    point_maes: list[float] = []
    nmes: list[float] = []
    angle_errors: dict[str, list[float]] = {name: [] for name in ANGLE_NAMES}
    sample_metrics: list[dict[str, object]] = []

    for row in rows:
        sample_id = row.get("sample_id", "<unknown>")
        try:
            raw_image = read_color(Path(row["raw_path"]))
            annotation = read_json(Path(row["annotation_path"]))
            target_map = annotation_keypoints(annotation, canonicalize_line_endpoints=True)
            target_coords = np.asarray([target_map[name] for name in KEYPOINT_NAMES], dtype=np.float32)
            pred_coords = predict_keypoints(
                model,
                row,
                raw_image,
                device,
                image_width,
                image_height,
                stride,
                decoder_id,
            )
            if pred_coords.shape != target_coords.shape:
                raise ValueError(f"Prediction shape {pred_coords.shape} != target shape {target_coords.shape}")
            original_height = int(row["image_height"])
            if original_height <= 0:
                raise ValueError(f"Invalid image_height: {original_height}")
        except Exception as exc:
            raise RuntimeError(f"Failed to evaluate sample {sample_id}: {exc}") from exc

        point_errors = np.linalg.norm(pred_coords - target_coords, axis=1)
        point_mae = float(np.mean(point_errors))
        nme_height_pct = float(np.mean(point_errors / original_height) * 100.0)
        point_maes.append(point_mae)
        nmes.append(nme_height_pct)

        pred_angles, pred_measurement_error = measure_angles_for_coords(raw_image, pred_coords, row)
        measured_gt_angles, gt_measurement_error = measure_angles_for_coords(raw_image, target_coords, row)
        gt_angles, gt_sources = ground_truth_angles(row, measured_gt_angles)

        sample_row: dict[str, object] = {
            "sample_id": sample_id,
            "case_id": row.get("case_id", ""),
            "side": row.get("side", ""),
            "image_width": int(row["image_width"]),
            "image_height": original_height,
            "point_mae_px": point_mae,
            "nme_height_pct": nme_height_pct,
            "prediction_measurement_error": pred_measurement_error,
            "ground_truth_measurement_error": gt_measurement_error,
        }
        for name in ANGLE_NAMES:
            predicted = pred_angles[name]
            target = gt_angles[name]
            error = abs(predicted - target) if predicted is not None and target is not None else None
            if error is not None and math.isfinite(error):
                angle_errors[name].append(error)
            else:
                error = None
            sample_row[f"{name}_gt_deg"] = target
            sample_row[f"{name}_pred_deg"] = predicted
            sample_row[f"{name}_abs_error_deg"] = error
            sample_row[f"{name}_gt_source"] = gt_sources[name]
        sample_metrics.append(sample_row)

    total_count = len(rows)
    metrics = {
        "point_mae_px": {
            **metric_summary(point_maes, total_count),
            "keypoint_n": total_count * len(KEYPOINT_NAMES),
        },
        "nme_height_pct": metric_summary(nmes, total_count),
    }
    for name in ANGLE_NAMES:
        metrics[f"{name}_mae_deg"] = metric_summary(angle_errors[name], total_count)
    return metrics, sample_metrics


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_per_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in PER_SAMPLE_FIELDS})


def run_evaluation(
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    split: str = "val",
    num_folds: int = 5,
    fold: int = 0,
    seed: int = 42,
    device_name: str = "auto",
    split_reference_manifest_path: Path | None = None,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    device = resolve_device(device_name)
    model, checkpoint = load_small_heatmap_checkpoint(checkpoint_path, device)
    all_rows = load_manifest(manifest_path)
    manifest_sha256 = sha256_path(manifest_path)
    reference_manifest: dict[str, object] | None = None
    if split_reference_manifest_path is not None:
        if split != "val":
            raise ValueError("--split-reference-manifest is only valid with --split val.")
        reference_path = split_reference_manifest_path.resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Split-reference manifest not found: {reference_path}")
        reference_rows = load_manifest(reference_path)
        reference_sha256 = sha256_path(reference_path)
        selected_rows = select_intervention_rows(
            all_rows,
            reference_rows,
            num_folds,
            fold,
            seed,
        )
        reference_manifest = {
            "path": str(reference_path),
            "sha256": reference_sha256,
            "total_rows": len(reference_rows),
            "purpose": "checkpoint_split_and_provenance_reference",
        }
    else:
        reference_path = manifest_path
        reference_rows = all_rows
        reference_sha256 = manifest_sha256
        selected_rows = select_manifest_rows(all_rows, split, num_folds, fold, seed)
    if not selected_rows:
        raise ValueError("The selected manifest split is empty.")
    provenance = None
    if split == "val":
        provenance = verify_validation_provenance(
            checkpoint_path,
            checkpoint,
            reference_path,
            reference_sha256,
            reference_rows,
            num_folds,
            fold,
            seed,
        )

    metrics, sample_metrics = evaluate_rows(model, checkpoint, selected_rows, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    per_sample_path = output_dir / "per_sample_metrics.csv"
    write_per_sample_csv(per_sample_path, sample_metrics)

    summary: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_path(checkpoint_path),
            "checkpoint_schema_version": int(checkpoint.get("checkpoint_schema_version", 0)),
            "adapter_id": str(checkpoint.get("adapter_id", ADAPTER_ID)),
            "architecture_id": str(checkpoint.get("architecture_id", ARCHITECTURE_ID)),
            "preprocessing_id": str(checkpoint.get("preprocessing_id", PREPROCESSING_ID)),
            "decoder_id": str(checkpoint.get("decoder_id", HARD_ARGMAX_DECODER_ID)),
            "model_version": checkpoint.get("model_version"),
            "model_scope": checkpoint.get("model_scope"),
            "epoch": int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None,
            "image_width": int(checkpoint["image_width"]),
            "image_height": int(checkpoint["image_height"]),
            "stride": int(checkpoint["stride"]),
            "training_manifest_sha256": checkpoint.get("training_manifest_sha256"),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "total_rows": len(all_rows),
            "selected_rows": len(selected_rows),
        },
        "selection": {
            "split": split,
            "num_folds": num_folds if split == "val" else None,
            "fold": fold if split == "val" else None,
            "seed": seed if split == "val" else None,
        },
        "device": str(device),
        "keypoint_count": len(KEYPOINT_NAMES),
        "ground_truth_angles": {
            "mldfa_mpta": "manifest values when finite, otherwise annotation geometry",
            "jlca_hka": "manifest values when finite, otherwise annotation geometry",
        },
        "metrics": metrics,
        "outputs": {
            "summary_json": str(summary_path),
            "per_sample_csv": str(per_sample_path),
        },
    }
    if provenance is not None:
        summary["provenance"] = provenance
    if reference_manifest is not None:
        summary["split_reference_manifest"] = reference_manifest
        summary["evaluation_design"] = "fixed_checkpoint_split_with_preprocessing_intervention"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a small_heatmap_v1 knee-keypoint checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m knee_xray.training.evaluate_keypoint_checkpoint --checkpoint models/current.pt "
            "--manifest images/annotation_processed_combined/processed_manifest_bone.csv --split all\n"
            "  python -m knee_xray.training.evaluate_keypoint_checkpoint --checkpoint outputs/run/best.pt "
            "--manifest data/manifest.csv --split val --num-folds 5 --fold 0 --seed 42 --device mps"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a small_heatmap_v1 .pt checkpoint.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to a compatible dataset manifest CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint_evaluation"),
        help="Writes summary.json and per_sample_metrics.csv here.",
    )
    parser.add_argument("--split", choices=("all", "val"), default="val", help="Evaluate every row or fold validation rows.")
    parser.add_argument("--num-folds", type=int, default=5, help="Case-level fold count used by the trainer.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index used by the trainer.")
    parser.add_argument("--seed", type=int, default=42, help="Case-shuffle seed used by the trainer.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--split-reference-manifest",
        type=Path,
        default=None,
        help=(
            "For a preprocessing intervention, verify the checkpoint and choose the validation "
            "fold from its original training manifest while reading images/annotations from --manifest."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split=args.split,
        num_folds=args.num_folds,
        fold=args.fold,
        seed=args.seed,
        device_name=args.device,
        split_reference_manifest_path=args.split_reference_manifest,
    )
    print(f"Summary JSON: {summary['outputs']['summary_json']}")
    print(f"Per-sample CSV: {summary['outputs']['per_sample_csv']}")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

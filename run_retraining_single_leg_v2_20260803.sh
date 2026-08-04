#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin=".venv-training/bin/python"
output_root="outputs/retraining_single_leg_v2_20260803"
manifest_root="$output_root/manifests"
baseline_manifest_root="outputs/retraining_20260803/manifests"

num_folds=5
seed=42
fold_epochs=80
final_epochs=50
batch_size=4
heatmap_peak_weight=20
decoder_id="local_centroid_3x3_residual_v1"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || die "Required file is missing or empty: $1"
}

archive_incomplete() {
  local path="$1"
  local archive suffix
  [[ -e "$path" ]] || return 0
  archive="${path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  suffix=1
  while [[ -e "$archive" ]]; do
    archive="${path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)_${suffix}"
    suffix=$((suffix + 1))
  done
  mv "$path" "$archive"
  echo "Preserved incomplete step as: $archive"
}

validate_manifests() {
  "$python_bin" - \
    "$manifest_root/bone_confirmed.csv" \
    "$manifest_root/tka.csv" \
    "$manifest_root/bone_tka.csv" \
    "$baseline_manifest_root/bone_confirmed.csv" \
    "$baseline_manifest_root/tka.csv" \
    "$baseline_manifest_root/bone_tka.csv" <<'PY'
import csv
import random
import sys
from pathlib import Path


candidate_paths = [Path(value) for value in sys.argv[1:4]]
baseline_paths = [Path(value) for value in sys.argv[4:7]]
required = {
    "sample_id",
    "case_id",
    "side",
    "annotation_path",
    "raw_path",
    "image_width",
    "image_height",
    "mldfa",
    "mpta",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "crop_width",
    "crop_height",
    "crop_method",
    "is_cropped",
}


def read_manifest(path: Path, *, check_files: bool) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")

    by_id: dict[str, dict[str, str]] = {}
    case_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        case_id = row["case_id"].strip()
        if not sample_id or not case_id:
            raise ValueError(f"{path}:{line_number}: blank sample_id or case_id")
        if sample_id in by_id:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
        if row["side"].strip() not in {"L", "R"}:
            raise ValueError(f"{path}:{line_number}: invalid side {row['side']!r}")
        try:
            width = int(row["image_width"])
            height = int(row["image_height"])
            crop_width = int(row["crop_width"])
            crop_height = int(row["crop_height"])
            crop_x0 = int(row["crop_x0"])
            crop_y0 = int(row["crop_y0"])
            crop_x1 = int(row["crop_x1"])
            crop_y1 = int(row["crop_y1"])
            float(row["mldfa"])
            float(row["mpta"])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
        if width < 1 or height < 1 or crop_width != width or crop_height != height:
            raise ValueError(
                f"{path}:{line_number}: processed image/crop dimensions do not match"
            )
        if crop_x0 < 0 or crop_y0 < 0 or crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            raise ValueError(f"{path}:{line_number}: invalid crop box")
        if crop_x1 - crop_x0 != crop_width or crop_y1 - crop_y0 != crop_height:
            raise ValueError(f"{path}:{line_number}: crop box and crop dimensions disagree")
        if row["is_cropped"].strip().lower() not in {"true", "false"}:
            raise ValueError(f"{path}:{line_number}: invalid is_cropped value")
        if not row["crop_method"].strip():
            raise ValueError(f"{path}:{line_number}: blank crop_method")
        if check_files:
            for field in ("annotation_path", "raw_path"):
                value = Path(row[field])
                resolved = value if value.is_absolute() or value.exists() else path.parent / value
                if not resolved.is_file():
                    raise ValueError(
                        f"{path}:{line_number}: {field} does not exist: {row[field]}"
                    )
        by_id[sample_id] = row
        case_ids.add(case_id)

    if len(case_ids) < 5:
        raise ValueError(f"{path}: fewer than five patient groups")
    shuffled = sorted(case_ids)
    random.Random(42).shuffle(shuffled)
    for fold in range(5):
        validation_cases = set(shuffled[fold::5])
        if not validation_cases or validation_cases == case_ids:
            raise ValueError(f"{path}: fold {fold} has an empty train or validation partition")
    return by_id


candidates = [read_manifest(path, check_files=True) for path in candidate_paths]
baselines = [read_manifest(path, check_files=False) for path in baseline_paths]

for candidate_path, candidate, baseline in zip(candidate_paths, candidates, baselines):
    if set(candidate) != set(baseline):
        missing = sorted(set(baseline).difference(candidate))[:10]
        extra = sorted(set(candidate).difference(baseline))[:10]
        raise ValueError(
            f"{candidate_path}: sample set differs from 20260803 baseline; "
            f"missing={missing}, extra={extra}"
        )
    for sample_id in candidate:
        for field in ("case_id", "side"):
            if candidate[sample_id][field].strip() != baseline[sample_id][field].strip():
                raise ValueError(
                    f"{candidate_path}: {sample_id} changed {field}; paired OOF comparison is invalid"
                )

bone, tka, mixed = candidates
if set(bone).intersection(tka):
    raise ValueError("bone_confirmed.csv and tka.csv overlap")
if set(mixed) != set(bone).union(tka):
    raise ValueError("bone_tka.csv is not the exact Bone + TKA sample union")
for sample_id, row in mixed.items():
    source = bone.get(sample_id) or tka.get(sample_id)
    assert source is not None
    for field in required:
        if row[field].strip() != source[field].strip():
            raise ValueError(
                f"bone_tka.csv: {sample_id} differs from its cohort manifest in {field}"
            )

for path, candidate, baseline in zip(candidate_paths, candidates, baselines):
    changed = sum(
        any(
            candidate[sample_id][field].strip() != baseline[sample_id][field].strip()
            for field in (
                "crop_x0",
                "crop_y0",
                "crop_x1",
                "crop_y1",
                "crop_width",
                "crop_height",
                "crop_method",
                "is_cropped",
                "image_width",
                "image_height",
            )
        )
        for sample_id in candidate
    )
    if changed == 0:
        raise ValueError(f"{path}: no single-leg preprocessing differences from baseline")
    print(f"Manifest preflight: {path} rows={len(candidate)} changed={changed}")
PY
}

training_complete() {
  local mode="$1" manifest="$2" output_dir="$3" epochs="$4" fold="$5"
  local model_version="$6" model_scope="$7" cv_summary="${8:-}"
  [[ -d "$output_dir" ]] || return 1
  "$python_bin" - "$mode" "$manifest" "$output_dir" "$epochs" "$fold" \
    "$model_version" "$model_scope" "$cv_summary" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


mode, manifest_value, output_value, epochs_value, fold_value, version, scope, cv_value = sys.argv[1:]
manifest = Path(manifest_value)
output_dir = Path(output_value)
epochs = int(epochs_value)
fold = int(fold_value)
is_final = mode == "final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    config_path = output_dir / "config.json"
    log_path = output_dir / "train_log.csv"
    checkpoint_names = ["final.pt"] if is_final else ["best.pt", "best_angles.pt"]
    if not config_path.is_file() or not log_path.is_file():
        raise ValueError("missing config.json or train_log.csv")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "manifest": manifest_value,
        "dataset_dir": None,
        "output_dir": output_value,
        "image_width": 256,
        "image_height": 320,
        "stride": 4,
        "sigma": 2.0,
        "epochs": epochs,
        "batch_size": 4,
        "lr": 0.001,
        "weight_decay": 0.00001,
        "heatmap_peak_weight": 20.0,
        "decoder_id": "local_centroid_3x3_residual_v1",
        "num_folds": 5,
        "fold": fold,
        "seed": 42,
        "device": "mps",
        "model_version": version,
        "model_scope": scope,
        "cache_dataset": True,
        "eval_every": 5,
        "train_all": is_final,
        "cv_summary": cv_value or None,
        "max_samples": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config mismatch for {key}: {config.get(key)!r} != {value!r}")
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError("training log is incomplete")

    manifest_hash = sha256(manifest)
    for checkpoint_name in checkpoint_names:
        checkpoint_path = output_dir / checkpoint_name
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise ValueError(f"missing checkpoint {checkpoint_name}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("training_manifest_sha256") != manifest_hash:
            raise ValueError(f"{checkpoint_name} was trained from a different manifest")
        if checkpoint.get("decoder_id") != "local_centroid_3x3_residual_v1":
            raise ValueError(f"{checkpoint_name} decoder mismatch")
        if checkpoint.get("model_version") != version or checkpoint.get("model_scope") != scope:
            raise ValueError(f"{checkpoint_name} model identity mismatch")
        expected_mode = "all_samples" if is_final else "cross_validation_fold"
        if checkpoint.get("training_mode") != expected_mode:
            raise ValueError(f"{checkpoint_name} training mode mismatch")
        if not is_final:
            provenance = checkpoint.get("split_provenance") or {}
            if (
                provenance.get("num_folds") != 5
                or provenance.get("fold") != fold
                or provenance.get("seed") != 42
            ):
                raise ValueError(f"{checkpoint_name} split provenance mismatch")
    if is_final:
        cv_summary = Path(cv_value)
        checkpoint = torch.load(output_dir / "final.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("epoch") != epochs:
            raise ValueError("final checkpoint epoch mismatch")
        if checkpoint.get("cross_validation_summary_sha256") != sha256(cv_summary):
            raise ValueError("final checkpoint CV summary mismatch")
except Exception as exc:
    print(f"Incomplete training step {output_dir}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

evaluation_complete() {
  local manifest="$1" checkpoint="$2" output_dir="$3" fold="$4"
  [[ -d "$output_dir" ]] || return 1
  "$python_bin" - "$manifest" "$checkpoint" "$output_dir" "$fold" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path


manifest, checkpoint, output_dir, fold_value = map(Path, sys.argv[1:])
fold = int(str(fold_value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "per_sample_metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise ValueError("missing summary.json or per_sample_metrics.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("manifest", {}).get("sha256") != sha256(manifest):
        raise ValueError("manifest SHA mismatch")
    if summary.get("checkpoint", {}).get("sha256") != sha256(checkpoint):
        raise ValueError("checkpoint SHA mismatch")
    selection = summary.get("selection") or {}
    if selection != {"split": "val", "num_folds": 5, "fold": fold, "seed": 42}:
        raise ValueError("validation selection mismatch")
    if summary.get("device") != "mps":
        raise ValueError("evaluation device mismatch")
    provenance = summary.get("provenance") or {}
    if provenance.get("verified") is not True or provenance.get("fold") != fold:
        raise ValueError("evaluation provenance is incomplete")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(summary.get("manifest", {}).get("selected_rows", -1)):
        raise ValueError("per-sample metric row count mismatch")
except Exception as exc:
    print(f"Incomplete evaluation step {output_dir}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

cv_complete() {
  local manifest="$1" output_dir="$2"
  [[ -d "$output_dir" ]] || return 1
  "$python_bin" - "$manifest" "$output_dir" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path


manifest, output_dir = map(Path, sys.argv[1:])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    summary_path = output_dir / "cv_summary.json"
    oof_path = output_dir / "oof_metrics.csv"
    if not summary_path.is_file() or not oof_path.is_file():
        raise ValueError("missing cv_summary.json or oof_metrics.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("manifest", {}).get("sha256") != sha256(manifest):
        raise ValueError("manifest SHA mismatch")
    cv = summary.get("cross_validation") or {}
    if (
        cv.get("num_folds") != 5
        or cv.get("seed") != 42
        or cv.get("folds_present") != [0, 1, 2, 3, 4]
        or cv.get("fold_count") != 5
        or cv.get("complete_5_fold") is not True
    ):
        raise ValueError("cross-validation summary is incomplete")
    provenance = summary.get("provenance") or {}
    if (
        provenance.get("verified") is not True
        or provenance.get("complete_partition_verified") is not True
    ):
        raise ValueError("CV provenance is incomplete")
    with oof_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(cv.get("total_oof_samples", -1)):
        raise ValueError("OOF row count mismatch")
except Exception as exc:
    print(f"Incomplete CV summary step {output_dir}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_fold() {
  local manifest="$1" prefix="$2" scope="$3" version_prefix="$4" fold="$5"
  local train_dir="$output_root/${prefix}_fold${fold}"
  local evaluation_dir="$output_root/evaluations/${prefix}_fold${fold}_best"
  local version="20260803-single-leg-v2-${version_prefix}-fold${fold}"

  if training_complete fold "$manifest" "$train_dir" "$fold_epochs" "$fold" "$version" "$scope"; then
    echo "SKIP complete training: $train_dir"
  else
    archive_incomplete "$train_dir"
    "$python_bin" train_keypoint_baseline.py \
      --manifest "$manifest" \
      --output-dir "$train_dir" \
      --image-width 256 \
      --image-height 320 \
      --stride 4 \
      --sigma 2 \
      --epochs "$fold_epochs" \
      --batch-size "$batch_size" \
      --lr 0.001 \
      --weight-decay 0.00001 \
      --heatmap-peak-weight "$heatmap_peak_weight" \
      --decoder-id "$decoder_id" \
      --num-folds "$num_folds" \
      --fold "$fold" \
      --seed "$seed" \
      --device mps \
      --model-version "$version" \
      --model-scope "$scope" \
      --cache-dataset \
      --eval-every 5
    training_complete fold "$manifest" "$train_dir" "$fold_epochs" "$fold" "$version" "$scope" \
      || die "Training command returned but artifacts are incomplete: $train_dir"
  fi

  if evaluation_complete "$manifest" "$train_dir/best.pt" "$evaluation_dir" "$fold"; then
    echo "SKIP complete evaluation: $evaluation_dir"
  else
    archive_incomplete "$evaluation_dir"
    "$python_bin" evaluate_keypoint_checkpoint.py \
      --checkpoint "$train_dir/best.pt" \
      --manifest "$manifest" \
      --output-dir "$evaluation_dir" \
      --split val \
      --num-folds "$num_folds" \
      --fold "$fold" \
      --seed "$seed" \
      --device mps
    evaluation_complete "$manifest" "$train_dir/best.pt" "$evaluation_dir" "$fold" \
      || die "Evaluation command returned but artifacts are incomplete: $evaluation_dir"
  fi
}

run_model() {
  local manifest="$1" prefix="$2" scope="$3" version_prefix="$4" final_dir_name="$5"
  local final_version="$6"
  local cv_dir="$output_root/evaluations/${prefix}_cv"
  local final_dir="$output_root/$final_dir_name"
  local fold evaluation_inputs=()

  for fold in 0 1 2 3 4; do
    run_fold "$manifest" "$prefix" "$scope" "$version_prefix" "$fold"
    evaluation_inputs+=("$output_root/evaluations/${prefix}_fold${fold}_best")
  done

  if cv_complete "$manifest" "$cv_dir"; then
    echo "SKIP complete CV summary: $cv_dir"
  else
    archive_incomplete "$cv_dir"
    "$python_bin" summarize_keypoint_cv.py \
      "${evaluation_inputs[@]}" \
      --output-dir "$cv_dir"
    cv_complete "$manifest" "$cv_dir" \
      || die "CV summarizer returned but artifacts are incomplete: $cv_dir"
  fi

  if training_complete final "$manifest" "$final_dir" "$final_epochs" 0 \
    "$final_version" "$scope" "$cv_dir/cv_summary.json"; then
    echo "SKIP complete train-all: $final_dir"
  else
    archive_incomplete "$final_dir"
    "$python_bin" train_keypoint_baseline.py \
      --manifest "$manifest" \
      --output-dir "$final_dir" \
      --image-width 256 \
      --image-height 320 \
      --stride 4 \
      --sigma 2 \
      --epochs "$final_epochs" \
      --batch-size "$batch_size" \
      --lr 0.001 \
      --weight-decay 0.00001 \
      --heatmap-peak-weight "$heatmap_peak_weight" \
      --decoder-id "$decoder_id" \
      --num-folds "$num_folds" \
      --fold 0 \
      --seed "$seed" \
      --device mps \
      --model-version "$final_version" \
      --model-scope "$scope" \
      --cache-dataset \
      --eval-every 5 \
      --train-all \
      --cv-summary "$cv_dir/cv_summary.json"
    training_complete final "$manifest" "$final_dir" "$final_epochs" 0 \
      "$final_version" "$scope" "$cv_dir/cv_summary.json" \
      || die "Train-all command returned but artifacts are incomplete: $final_dir"
  fi
}

require_file "$python_bin"
require_file train_keypoint_baseline.py
require_file evaluate_keypoint_checkpoint.py
require_file summarize_keypoint_cv.py
require_file validate_knee_dataset.py
require_file "$manifest_root/bone_confirmed.csv"
require_file "$manifest_root/tka.csv"
require_file "$manifest_root/bone_tka.csv"
require_file "$baseline_manifest_root/bone_confirmed.csv"
require_file "$baseline_manifest_root/tka.csv"
require_file "$baseline_manifest_root/bone_tka.csv"

[[ ! -L "$output_root" ]] || die "Refusing to use a symlinked output root: $output_root"
mkdir -p "$output_root/evaluations"

validate_manifests
"$python_bin" validate_knee_dataset.py --manifest "$manifest_root/bone_confirmed.csv"
"$python_bin" validate_knee_dataset.py --manifest "$manifest_root/tka.csv"
"$python_bin" - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("Apple MPS is required for this reproducible retraining run")
print(f"Training runtime: torch={torch.__version__}, device=mps")
PY

run_model \
  "$manifest_root/bone_confirmed.csv" \
  bone_weighted_centroid \
  confirmed-bone \
  bone-weighted-centroid \
  final_bone_model \
  20260803-single-leg-v2-bone-final-v1

run_model \
  "$manifest_root/tka.csv" \
  tka_weighted_centroid \
  tka-cohort \
  tka-weighted-centroid \
  final_tka_model \
  20260803-single-leg-v2-tka-final-v1

run_model \
  "$manifest_root/bone_tka.csv" \
  bone_tka_mixed_weighted_centroid \
  bone-tka-mixed \
  bone-tka-mixed-weighted-centroid \
  final_bone_tka_mixed_model \
  20260803-single-leg-v2-bone-tka-mixed-final-v1

echo "Single-leg v2 Bone/TKA/Mixed retraining completed: $output_root"

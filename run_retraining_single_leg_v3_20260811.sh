#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="${KNEE_RETRAINING_PYTHON_BIN:-.venv-training/bin/python}"
output_root="${KNEE_RETRAINING_OUTPUT_ROOT:-outputs/retraining_single_leg_v3_20260811}"
model_version_namespace="${KNEE_RETRAINING_MODEL_VERSION_NAMESPACE-20260811-single-leg-v3-curated}"
manifest_root="${KNEE_RETRAINING_MANIFEST_ROOT:-$output_root/manifests_curated}"

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

[[ -n "$model_version_namespace" ]] \
  || die "KNEE_RETRAINING_MODEL_VERSION_NAMESPACE must not be empty"
[[ "$model_version_namespace" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "KNEE_RETRAINING_MODEL_VERSION_NAMESPACE contains unsafe characters"

require_file() {
  [[ -s "$1" ]] || die "Required file is missing or empty: $1"
}

preflight_only=false
case "${1:-}" in
  "") ;;
  --preflight-only) preflight_only=true ;;
  *) die "Usage: $0 [--preflight-only]" ;;
esac

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
    "$manifest_root/bone_tka.csv" <<'PY'
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path


candidate_paths = [Path(value) for value in sys.argv[1:4]]
cohort_names = ("Bone", "TKA", "Mixed")
minimum_previous_rows = {"Bone": 380, "TKA": 287}
expected_implant_status = {"Bone": "bone", "TKA": "TKA"}
patient_group_pattern = re.compile(r"patient_group:[0-9a-f]{64}")
sha256_pattern = re.compile(r"[0-9a-f]{64}")
annotation_hash_fields = ("image_width", "image_height", "side", "points", "lines")
accepted_crop_review_statuses = {
    "historical_original_pass",
    "historical_recrop_salvage",
    "new_batch_original_pass",
    "new_batch_recrop_salvage",
    "new_batch_aspect_cap_salvage",
    "tail_review_recrop_release_safe",
}
file_sha256_cache: dict[Path, str] = {}
required = {
    "sample_id",
    "case_id",
    "source_case_id",
    "source_dataset",
    "implant_status",
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
    "crop_provenance_source",
    "crop_selection_method",
    "crop_confirmed",
    "inference_roi_status",
    "is_cropped",
    "crop_review_status",
    "crop_review_reason",
    "evidence_sha256",
    "crop_coordinate_space",
    "crop_output_width",
    "crop_output_height",
    "crop_pad_method",
    "crop_pad_left",
    "crop_pad_right",
    "crop_pad_top",
    "crop_pad_bottom",
    "crop_pad_fill_value",
    "crop_rescaled",
    "crop_transform_sha256",
    "immediate_source_raw_path",
    "immediate_source_annotation_path",
    "immediate_source_raw_sha256",
    "immediate_source_annotation_sha256",
    "curated_raw_sha256",
    "curated_annotation_sha256",
    "training_included",
}


def resolve_file(path: Path, value: str) -> Path:
    configured = Path(value)
    candidates = [configured] if configured.is_absolute() else [Path.cwd() / configured, path.parent / configured]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise ValueError(f"{path}: missing or empty file: {value}")


def sha256_file(path: Path) -> str:
    cached = file_sha256_cache.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    file_sha256_cache[path] = value
    return value


def annotation_sha256(annotation: dict, path: Path) -> str:
    missing = [field for field in annotation_hash_fields if field not in annotation]
    if missing:
        raise ValueError(f"{path}: annotation is missing training fields {missing}")
    payload = {field: annotation[field] for field in annotation_hash_fields}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: str, path: Path, line_number: int, field: str) -> str:
    normalized = value.strip()
    if sha256_pattern.fullmatch(normalized) is None:
        raise ValueError(f"{path}:{line_number}: invalid {field}")
    return normalized


def require_boolean(
    value: str,
    path: Path,
    line_number: int,
    field: str,
    *,
    expected: bool | None = None,
) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{path}:{line_number}: invalid {field} value")
    result = normalized == "true"
    if expected is not None and result is not expected:
        raise ValueError(
            f"{path}:{line_number}: {field} must be {str(expected).lower()}"
        )
    return result


def require_training_coordinates(
    annotation: dict,
    width: int,
    height: int,
    path: Path,
    line_number: int,
) -> None:
    pairs: list[tuple[object, object]] = []
    points = annotation.get("points")
    lines = annotation.get("lines")
    point_names = (
        "hip",
        "upper_left",
        "upper_center",
        "upper_right",
        "lower_left",
        "lower_center",
        "lower_right",
        "ankle",
    )
    if not isinstance(points, dict) or not isinstance(lines, dict):
        raise ValueError(f"{path}:{line_number}: annotation points/lines are invalid")
    for name in point_names:
        value = points.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: missing point {name}")
        pairs.append((value.get("x"), value.get("y")))
    for line_name in ("upper_line", "lower_line"):
        line = lines.get(line_name)
        if not isinstance(line, dict):
            raise ValueError(f"{path}:{line_number}: missing line {line_name}")
        for endpoint in ("p1", "p2"):
            value = line.get(endpoint)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: missing {line_name}.{endpoint}"
                )
            pairs.append((value.get("x"), value.get("y")))
    if len(pairs) != 12:
        raise ValueError(f"{path}:{line_number}: expected 12 coordinate pairs")
    for x_value, y_value in pairs:
        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: non-numeric training coordinate"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{path}:{line_number}: non-finite training coordinate")
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"{path}:{line_number}: training coordinate is out of bounds")


def read_manifest(
    path: Path,
    cohort_name: str,
) -> tuple[dict[str, dict[str, str]], set[str], dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{path}: duplicate CSV columns")
        fields = set(fieldnames)
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")

    by_id: dict[str, dict[str, str]] = {}
    content_by_id: dict[str, str] = {}
    sample_by_content: dict[str, str] = {}
    case_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        case_id = row["case_id"].strip()
        source_case_id = row["source_case_id"].strip()
        if not sample_id or not case_id or not source_case_id:
            raise ValueError(
                f"{path}:{line_number}: blank sample_id, case_id, or source_case_id"
            )
        if sample_id in by_id:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
        if patient_group_pattern.fullmatch(case_id) is None:
            raise ValueError(f"{path}:{line_number}: invalid patient-group case_id")
        if source_case_id.startswith("patient_group:"):
            raise ValueError(f"{path}:{line_number}: source_case_id is already patient-grouped")
        if not row["source_dataset"].strip():
            raise ValueError(f"{path}:{line_number}: blank source_dataset")
        if row["side"].strip() not in {"L", "R"}:
            raise ValueError(f"{path}:{line_number}: invalid side {row['side']!r}")
        implant_status = row["implant_status"].strip()
        expected_status = expected_implant_status.get(cohort_name)
        if expected_status is not None and implant_status != expected_status:
            raise ValueError(
                f"{path}:{line_number}: {cohort_name} row has implant_status={implant_status!r}"
            )
        if cohort_name == "Mixed" and implant_status not in {"bone", "TKA"}:
            raise ValueError(
                f"{path}:{line_number}: Mixed row has implant_status={implant_status!r}"
            )
        try:
            width = int(row["image_width"])
            height = int(row["image_height"])
            crop_width = int(row["crop_width"])
            crop_height = int(row["crop_height"])
            crop_x0 = int(row["crop_x0"])
            crop_y0 = int(row["crop_y0"])
            crop_x1 = int(row["crop_x1"])
            crop_y1 = int(row["crop_y1"])
            crop_output_width = int(row["crop_output_width"])
            crop_output_height = int(row["crop_output_height"])
            crop_pad_left = int(row["crop_pad_left"])
            crop_pad_right = int(row["crop_pad_right"])
            crop_pad_top = int(row["crop_pad_top"])
            crop_pad_bottom = int(row["crop_pad_bottom"])
            crop_pad_fill_value = int(row["crop_pad_fill_value"])
            mldfa = float(row["mldfa"])
            mpta = float(row["mpta"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
        if not math.isfinite(mldfa) or not math.isfinite(mpta):
            raise ValueError(f"{path}:{line_number}: non-finite angle field")
        if width < 1 or height < 1 or crop_width < 1 or crop_height < 1:
            raise ValueError(
                f"{path}:{line_number}: invalid processed image/crop dimensions"
            )
        if (crop_output_width, crop_output_height) != (width, height):
            raise ValueError(
                f"{path}:{line_number}: crop output and processed image dimensions disagree"
            )
        if min(crop_pad_left, crop_pad_right, crop_pad_top, crop_pad_bottom) < 0:
            raise ValueError(f"{path}:{line_number}: negative crop padding")
        if not 0 <= crop_pad_fill_value <= 255:
            raise ValueError(f"{path}:{line_number}: invalid crop padding fill value")
        if crop_width + crop_pad_left + crop_pad_right != width:
            raise ValueError(
                f"{path}:{line_number}: crop width plus padding does not equal image width"
            )
        if crop_height + crop_pad_top + crop_pad_bottom != height:
            raise ValueError(
                f"{path}:{line_number}: crop height plus padding does not equal image height"
            )
        aspect_ratio = width / height
        if not 0.35 <= aspect_ratio <= 0.60:
            raise ValueError(
                f"{path}:{line_number}: single-leg image aspect ratio is outside "
                f"the reviewed 0.35-0.60 contract: {aspect_ratio:.6f}"
            )
        if crop_x0 < 0 or crop_y0 < 0 or crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            raise ValueError(f"{path}:{line_number}: invalid crop box")
        if crop_x1 - crop_x0 != crop_width or crop_y1 - crop_y0 != crop_height:
            raise ValueError(f"{path}:{line_number}: crop box and crop dimensions disagree")
        require_boolean(
            row["crop_confirmed"], path, line_number, "crop_confirmed", expected=True
        )
        require_boolean(row["is_cropped"], path, line_number, "is_cropped")
        require_boolean(
            row["crop_rescaled"], path, line_number, "crop_rescaled", expected=False
        )
        require_boolean(
            row["training_included"],
            path,
            line_number,
            "training_included",
            expected=True,
        )
        if not row["crop_method"].strip():
            raise ValueError(f"{path}:{line_number}: blank crop_method")
        if row["crop_provenance_source"].strip() != "training_crop_contract":
            raise ValueError(
                f"{path}:{line_number}: invalid crop provenance source"
            )
        if not row["crop_selection_method"].strip():
            raise ValueError(f"{path}:{line_number}: blank crop selection method")
        if not row["crop_coordinate_space"].strip():
            raise ValueError(f"{path}:{line_number}: blank crop coordinate space")
        if not row["crop_pad_method"].strip():
            raise ValueError(f"{path}:{line_number}: blank crop padding method")
        if row["inference_roi_status"].strip() != "crop_contract_reviewed":
            raise ValueError(
                f"{path}:{line_number}: inference ROI is not crop-contract reviewed"
            )
        crop_review_status = row["crop_review_status"].strip()
        if crop_review_status not in accepted_crop_review_statuses:
            raise ValueError(
                f"{path}:{line_number}: unapproved crop review status "
                f"{crop_review_status!r}"
            )
        if not row["crop_review_reason"].strip():
            raise ValueError(f"{path}:{line_number}: blank crop review reason")

        evidence_sha256 = require_sha256(
            row["evidence_sha256"], path, line_number, "evidence_sha256"
        )
        crop_transform_sha256 = require_sha256(
            row["crop_transform_sha256"],
            path,
            line_number,
            "crop_transform_sha256",
        )
        curated_raw_sha256 = require_sha256(
            row["curated_raw_sha256"], path, line_number, "curated_raw_sha256"
        )
        curated_annotation_sha256 = require_sha256(
            row["curated_annotation_sha256"],
            path,
            line_number,
            "curated_annotation_sha256",
        )
        immediate_source_raw_sha256 = require_sha256(
            row["immediate_source_raw_sha256"],
            path,
            line_number,
            "immediate_source_raw_sha256",
        )
        immediate_source_annotation_sha256 = require_sha256(
            row["immediate_source_annotation_sha256"],
            path,
            line_number,
            "immediate_source_annotation_sha256",
        )

        annotation_path = resolve_file(path, row["annotation_path"].strip())
        raw_path = resolve_file(path, row["raw_path"].strip())
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid annotation JSON: {annotation_path}"
            ) from exc
        if not isinstance(annotation, dict):
            raise ValueError(f"{path}:{line_number}: annotation root is not an object")
        try:
            annotation_width = int(annotation["image_width"])
            annotation_height = int(annotation["image_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: annotation dimensions are invalid"
            ) from exc
        if (annotation_width, annotation_height) != (width, height):
            raise ValueError(
                f"{path}:{line_number}: annotation and manifest dimensions disagree"
            )
        if str(annotation.get("side", "")).strip() != row["side"].strip():
            raise ValueError(f"{path}:{line_number}: annotation and manifest side disagree")

        require_training_coordinates(annotation, width, height, path, line_number)
        if sha256_file(raw_path) != curated_raw_sha256:
            raise ValueError(f"{path}:{line_number}: curated raw SHA mismatch")
        if sha256_file(annotation_path) != curated_annotation_sha256:
            raise ValueError(f"{path}:{line_number}: curated annotation SHA mismatch")
        annotation_raw_value = str(annotation.get("raw_path", "")).strip()
        if not annotation_raw_value:
            raise ValueError(f"{path}:{line_number}: annotation raw_path is blank")
        if resolve_file(path, annotation_raw_value) != raw_path:
            raise ValueError(f"{path}:{line_number}: annotation raw_path disagrees")

        immediate_raw_path = resolve_file(
            path, row["immediate_source_raw_path"].strip()
        )
        immediate_annotation_path = resolve_file(
            path, row["immediate_source_annotation_path"].strip()
        )
        if sha256_file(immediate_raw_path) != immediate_source_raw_sha256:
            raise ValueError(f"{path}:{line_number}: immediate-source raw SHA mismatch")
        if (
            sha256_file(immediate_annotation_path)
            != immediate_source_annotation_sha256
        ):
            raise ValueError(
                f"{path}:{line_number}: immediate-source annotation SHA mismatch"
            )

        contract = annotation.get("training_crop_contract")
        if not isinstance(contract, dict) or contract.get("schema_version") != 1:
            raise ValueError(
                f"{path}:{line_number}: missing training crop contract schema v1"
            )
        if contract.get("crop_confirmed") is not True or annotation.get(
            "crop_confirmed"
        ) is not True:
            raise ValueError(f"{path}:{line_number}: annotation crop is not confirmed")
        if contract.get("evidence_sha256") != evidence_sha256:
            raise ValueError(f"{path}:{line_number}: crop evidence SHA disagrees")
        evidence_path_value = str(contract.get("evidence_ledger_path", "")).strip()
        evidence_path = resolve_file(path, evidence_path_value)
        if sha256_file(evidence_path) != evidence_sha256:
            raise ValueError(f"{path}:{line_number}: crop evidence ledger SHA mismatch")
        if contract.get("crop_review_status") != crop_review_status:
            raise ValueError(f"{path}:{line_number}: crop review status disagrees")
        if contract.get("crop_review_reason") != row["crop_review_reason"].strip():
            raise ValueError(f"{path}:{line_number}: crop review reason disagrees")
        if contract.get("crop_transform_sha256") != crop_transform_sha256:
            raise ValueError(f"{path}:{line_number}: crop transform SHA disagrees")

        contract_immediate = contract.get("immediate_source")
        contract_transform = contract.get("transform")
        if not isinstance(contract_immediate, dict) or not isinstance(
            contract_transform, dict
        ):
            raise ValueError(f"{path}:{line_number}: incomplete crop contract")
        immediate_expected = {
            "raw_path": row["immediate_source_raw_path"].strip(),
            "annotation_path": row["immediate_source_annotation_path"].strip(),
            "raw_sha256": immediate_source_raw_sha256,
            "annotation_sha256": immediate_source_annotation_sha256,
        }
        for field, expected_value in immediate_expected.items():
            if contract_immediate.get(field) != expected_value:
                raise ValueError(
                    f"{path}:{line_number}: crop contract immediate-source {field} disagrees"
                )
        try:
            immediate_width = int(contract_immediate["width"])
            immediate_height = int(contract_immediate["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid immediate-source dimensions"
            ) from exc
        if not (0 <= crop_x0 < crop_x1 <= immediate_width):
            raise ValueError(f"{path}:{line_number}: crop exceeds immediate-source width")
        if not (0 <= crop_y0 < crop_y1 <= immediate_height):
            raise ValueError(f"{path}:{line_number}: crop exceeds immediate-source height")

        transform_crop = contract_transform.get("crop")
        transform_padding = contract_transform.get("padding")
        transform_output = contract_transform.get("output_canvas")
        if not all(
            isinstance(value, dict)
            for value in (transform_crop, transform_padding, transform_output)
        ):
            raise ValueError(f"{path}:{line_number}: incomplete crop transform")
        crop_expected = {
            "x0": crop_x0,
            "y0": crop_y0,
            "x1": crop_x1,
            "y1": crop_y1,
            "width": crop_width,
            "height": crop_height,
            "method": row["crop_method"].strip(),
            "coordinate_space": row["crop_coordinate_space"].strip(),
            "selection_method": row["crop_selection_method"].strip(),
            "confirmed": True,
        }
        for field, expected_value in crop_expected.items():
            if transform_crop.get(field) != expected_value:
                raise ValueError(
                    f"{path}:{line_number}: crop transform field {field} disagrees"
                )
        padding_expected = {
            "method": row["crop_pad_method"].strip(),
            "left": crop_pad_left,
            "right": crop_pad_right,
            "top": crop_pad_top,
            "bottom": crop_pad_bottom,
            "fill_value": crop_pad_fill_value,
            "rescaled": False,
        }
        for field, expected_value in padding_expected.items():
            if transform_padding.get(field) != expected_value:
                raise ValueError(
                    f"{path}:{line_number}: crop padding field {field} disagrees"
                )
        if transform_output != {"width": width, "height": height}:
            raise ValueError(f"{path}:{line_number}: crop output canvas disagrees")
        # The evidence hash was frozen before the finalizer promoted the crop to
        # confirmed=True. Historical prototype schemas either omitted that field
        # or carried their reviewed pre-promotion True/False value. The finalized
        # contract intentionally exposes only the promoted value, so verify the
        # frozen hash against the three lossless legacy encodings and require one
        # unambiguous match.
        reconstructed_hashes: set[str] = set()
        for original_confirmed in (None, True, False):
            original_crop = dict(transform_crop)
            if original_confirmed is None:
                original_crop.pop("confirmed", None)
            else:
                original_crop["confirmed"] = original_confirmed
            reconstructed_transform = {
                "schema_version": 1,
                "kind": contract_transform.get("kind"),
                "immediate_source": contract_immediate,
                "crop": original_crop,
                "padding": transform_padding,
                "output_canvas": transform_output,
            }
            reconstructed_hashes.add(canonical_json_sha256(reconstructed_transform))
        if crop_transform_sha256 not in reconstructed_hashes:
            raise ValueError(f"{path}:{line_number}: crop transform hash mismatch")

        validation = contract.get("validation")
        required_validation = {
            "coordinate_pair_count": 12,
            "all_coordinates_in_bounds": True,
            "angles_preserved_from_frozen_baseline": True,
            "padding_aware_geometry": True,
            "strip_rescaled": False,
        }
        if not isinstance(validation, dict) or any(
            validation.get(field) != expected_value
            for field, expected_value in required_validation.items()
        ):
            raise ValueError(f"{path}:{line_number}: crop contract validation disagrees")

        processed_from = annotation.get("processed_from")
        if not isinstance(processed_from, dict):
            raise ValueError(f"{path}:{line_number}: processed_from is missing")
        if processed_from.get("raw_path") != immediate_expected["raw_path"]:
            raise ValueError(f"{path}:{line_number}: processed_from raw source disagrees")
        if (
            processed_from.get("annotation_path")
            != immediate_expected["annotation_path"]
        ):
            raise ValueError(
                f"{path}:{line_number}: processed_from annotation source disagrees"
            )
        if processed_from.get("source_raw_sha256") != immediate_source_raw_sha256:
            raise ValueError(f"{path}:{line_number}: processed_from raw SHA disagrees")
        if (
            processed_from.get("source_annotation_sha256")
            != immediate_source_annotation_sha256
        ):
            raise ValueError(
                f"{path}:{line_number}: processed_from annotation SHA disagrees"
            )
        processed_crop = processed_from.get("crop")
        processed_padding = processed_from.get("padding")
        if not isinstance(processed_crop, dict) or not isinstance(
            processed_padding, dict
        ):
            raise ValueError(
                f"{path}:{line_number}: processed_from crop/padding is incomplete"
            )
        if any(
            processed_crop.get(field) != expected_value
            for field, expected_value in crop_expected.items()
        ):
            raise ValueError(f"{path}:{line_number}: processed_from crop disagrees")
        if any(
            processed_padding.get(field) != expected_value
            for field, expected_value in padding_expected.items()
        ):
            raise ValueError(f"{path}:{line_number}: processed_from padding disagrees")

        raw_sha256 = curated_raw_sha256
        training_annotation_sha256 = annotation_sha256(annotation, annotation_path)
        content_sha256 = hashlib.sha256(
            f"{raw_sha256}:{training_annotation_sha256}".encode("ascii")
        ).hexdigest()
        previous_sample = sample_by_content.setdefault(content_sha256, sample_id)
        if previous_sample != sample_id:
            raise ValueError(
                f"{path}:{line_number}: duplicate training content for "
                f"{previous_sample!r} and {sample_id!r}"
            )

        by_id[sample_id] = row
        content_by_id[sample_id] = content_sha256
        case_ids.add(case_id)

    if len(case_ids) < 5:
        raise ValueError(f"{path}: fewer than five patient groups")
    shuffled = sorted(case_ids)
    random.Random(42).shuffle(shuffled)
    for fold in range(5):
        validation_cases = set(shuffled[fold::5])
        if not validation_cases or validation_cases == case_ids:
            raise ValueError(f"{path}: fold {fold} has an empty train or validation partition")
    return by_id, fields, content_by_id


candidates = [
    read_manifest(path, cohort_name)
    for path, cohort_name in zip(candidate_paths, cohort_names)
]
(bone, bone_fields, bone_content), (tka, tka_fields, tka_content), (
    mixed,
    mixed_fields,
    mixed_content,
) = candidates

if set(bone).intersection(tka):
    raise ValueError("bone_confirmed.csv and tka.csv overlap")
if set(mixed) != set(bone).union(tka):
    raise ValueError("bone_tka.csv is not the exact Bone + TKA sample union")
if bone_fields != tka_fields or bone_fields != mixed_fields:
    raise ValueError("Bone, TKA, and Mixed manifests do not have identical columns")
if len(bone) <= minimum_previous_rows["Bone"]:
    raise ValueError(
        f"Bone manifest did not expand beyond {minimum_previous_rows['Bone']} rows"
    )
if len(tka) <= minimum_previous_rows["TKA"]:
    raise ValueError(
        f"TKA manifest did not expand beyond {minimum_previous_rows['TKA']} rows"
    )

content_owner: dict[str, tuple[str, str]] = {}
for cohort_name, content_rows in (("Bone", bone_content), ("TKA", tka_content)):
    for sample_id, content_sha256 in content_rows.items():
        previous = content_owner.setdefault(content_sha256, (cohort_name, sample_id))
        if previous != (cohort_name, sample_id):
            raise ValueError(
                "Bone/TKA manifests contain duplicate training content: "
                f"{previous[0]} {previous[1]!r} and {cohort_name} {sample_id!r}"
            )

for sample_id, row in mixed.items():
    source = bone.get(sample_id) or tka.get(sample_id)
    assert source is not None
    for field in bone_fields:
        if row[field].strip() != source[field].strip():
            raise ValueError(
                f"bone_tka.csv: {sample_id} differs from its cohort manifest in {field}"
            )
    expected_content = bone_content.get(sample_id) or tka_content.get(sample_id)
    if mixed_content[sample_id] != expected_content:
        raise ValueError(f"bone_tka.csv: {sample_id} content hash differs from its cohort")

for path, cohort_name, candidate in zip(candidate_paths, cohort_names, (bone, tka, mixed)):
    print(
        f"Manifest preflight: {path} cohort={cohort_name} rows={len(candidate)} "
        f"patient_groups={len({row['case_id'] for row in candidate.values()})}"
    )
print(
    "Expanded dataset preflight passed: "
    f"Bone +{len(bone) - minimum_previous_rows['Bone']}, "
    f"TKA +{len(tka) - minimum_previous_rows['TKA']}, Mixed={len(mixed)}"
)
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
  local version="${model_version_namespace}-${version_prefix}-fold${fold}"

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
  local final_version_suffix="$6"
  local final_version="${model_version_namespace}-${final_version_suffix}"
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

[[ ! -L "$output_root" ]] || die "Refusing to use a symlinked output root: $output_root"
validate_manifests
if [[ "$preflight_only" == true ]]; then
  echo "Expanded dataset manifest preflight completed: $manifest_root"
  exit 0
fi

mkdir -p "$output_root/evaluations"
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
  bone-final-v1

run_model \
  "$manifest_root/tka.csv" \
  tka_weighted_centroid \
  tka-cohort \
  tka-weighted-centroid \
  final_tka_model \
  tka-final-v1

run_model \
  "$manifest_root/bone_tka.csv" \
  bone_tka_mixed_weighted_centroid \
  bone-tka-mixed \
  bone-tka-mixed-weighted-centroid \
  final_bone_tka_mixed_model \
  bone-tka-mixed-final-v1

echo "Single-leg v3 Bone/TKA/Mixed retraining completed: $output_root"

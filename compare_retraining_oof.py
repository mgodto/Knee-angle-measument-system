#!/usr/bin/env python3
"""Compare paired Bone/TKA/Mixed OOF metrics from two retraining runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_CV_DIRS = {
    "Bone": "bone_weighted_centroid_cv",
    "TKA": "tka_weighted_centroid_cv",
    "Mixed": "bone_tka_mixed_weighted_centroid_cv",
}
METRIC_COLUMNS = (
    "point_mae_px",
    "nme_height_pct",
    "mldfa_abs_error_deg",
    "mpta_abs_error_deg",
    "jlca_abs_error_deg",
    "hka_abs_error_deg",
)
CROP_SIGNATURE_COLUMNS = (
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


class ComparisonError(ValueError):
    """Raised when two OOF runs cannot be compared fairly."""


@dataclass(frozen=True)
class OOFRun:
    root: Path
    model: str
    oof_path: Path
    summary_path: Path
    summary: dict[str, object]
    rows: dict[str, dict[str, str]]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ComparisonError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ComparisonError(f"Missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"JSON root must be an object: {path}")
    return value


def _integer(text: str, *, field: str, source: Path) -> int:
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"Invalid {field}={text!r} in {source}") from exc


def _validate_cv_summary(
    summary: dict[str, object], rows: dict[str, dict[str, str]], source: Path
) -> None:
    cv = summary.get("cross_validation")
    provenance = summary.get("provenance")
    if not isinstance(cv, dict) or not isinstance(provenance, dict):
        raise ComparisonError(f"Missing cross_validation/provenance object in {source}")
    try:
        num_folds = int(cv["num_folds"])
        seed = int(cv["seed"])
        declared_folds = sorted(int(value) for value in cv["folds_present"])
        declared_total = int(cv["total_oof_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"Invalid cross-validation metadata in {source}") from exc
    if num_folds <= 0 or seed < 0:
        raise ComparisonError(f"Invalid num_folds/seed in {source}")
    actual_folds = sorted({_integer(row["fold"], field="fold", source=source) for row in rows.values()})
    if declared_folds != actual_folds:
        raise ComparisonError(
            f"folds_present does not match OOF rows in {source}: "
            f"declared={declared_folds}, actual={actual_folds}"
        )
    if declared_total != len(rows):
        raise ComparisonError(
            f"total_oof_samples={declared_total}, but OOF has {len(rows)} rows in {source}"
        )
    expected_folds = list(range(num_folds))
    complete = actual_folds == expected_folds
    if not complete or cv.get("complete_5_fold") is not True:
        raise ComparisonError(
            f"Comparison requires a complete CV partition {expected_folds}; got {actual_folds} in {source}"
        )
    if provenance.get("verified") is not True:
        raise ComparisonError(f"Unverified CV provenance in {source}")
    if provenance.get("complete_partition_verified") is not True:
        raise ComparisonError(f"CV partition was not verified in {source}")


def _validate_patient_groups(rows: dict[str, dict[str, str]], source: Path) -> None:
    patient_folds: dict[str, int] = {}
    for sample_id, row in rows.items():
        patient_group = row["case_id"].strip()
        if not patient_group:
            raise ComparisonError(f"Empty case_id/patient_group for {sample_id!r} in {source}")
        fold = _integer(row["fold"], field="fold", source=source)
        previous = patient_folds.setdefault(patient_group, fold)
        if previous != fold:
            raise ComparisonError(
                f"Patient group {patient_group!r} crosses folds {previous} and {fold} in {source}"
            )


def load_oof_run(root: Path, model: str) -> OOFRun:
    cv_dir = root / "evaluations" / MODEL_CV_DIRS[model]
    oof_path = cv_dir / "oof_metrics.csv"
    summary_path = cv_dir / "cv_summary.json"
    fields, raw_rows = _read_csv(oof_path)
    required = {"fold", "sample_id", "case_id", "side", *METRIC_COLUMNS}
    missing = sorted(required.difference(fields))
    if missing:
        raise ComparisonError(f"Missing OOF columns {missing} in {oof_path}")
    rows: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ComparisonError(f"Empty sample_id in {oof_path}")
        if sample_id in rows:
            raise ComparisonError(f"Duplicate sample_id {sample_id!r} in {oof_path}")
        rows[sample_id] = row
    _validate_patient_groups(rows, oof_path)
    summary = _load_json(summary_path)
    _validate_cv_summary(summary, rows, summary_path)
    return OOFRun(root.resolve(), model, oof_path.resolve(), summary_path.resolve(), summary, rows)


def _validate_paired_runs(baseline: OOFRun, candidate: OOFRun) -> None:
    baseline_ids = set(baseline.rows)
    candidate_ids = set(candidate.rows)
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        raise ComparisonError(
            f"{baseline.model} sample sets differ; missing_in_candidate={missing[:10]}, "
            f"extra_in_candidate={extra[:10]}"
        )
    baseline_cv = baseline.summary["cross_validation"]
    candidate_cv = candidate.summary["cross_validation"]
    assert isinstance(baseline_cv, dict) and isinstance(candidate_cv, dict)
    for field in ("num_folds", "seed", "folds_present"):
        if baseline_cv.get(field) != candidate_cv.get(field):
            raise ComparisonError(
                f"{baseline.model} cross-validation {field} differs: "
                f"baseline={baseline_cv.get(field)!r}, candidate={candidate_cv.get(field)!r}"
            )
    for sample_id in sorted(baseline_ids):
        old = baseline.rows[sample_id]
        new = candidate.rows[sample_id]
        if old["case_id"].strip() != new["case_id"].strip():
            raise ComparisonError(
                f"{baseline.model} patient_group differs for {sample_id!r}: "
                f"baseline={old['case_id']!r}, candidate={new['case_id']!r}"
            )
        if _integer(old["fold"], field="fold", source=baseline.oof_path) != _integer(
            new["fold"], field="fold", source=candidate.oof_path
        ):
            raise ComparisonError(f"{baseline.model} fold differs for sample {sample_id!r}")
        if old["side"].strip() != new["side"].strip():
            raise ComparisonError(f"{baseline.model} side differs for sample {sample_id!r}")


def _metric(text: str, *, field: str, sample_id: str, source: Path) -> float | None:
    value = text.strip()
    if not value or value.lower() in {"na", "nan", "none"}:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ComparisonError(
            f"Invalid {field}={text!r} for {sample_id!r} in {source}"
        ) from exc
    return number if math.isfinite(number) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: list[float]) -> dict[str, int | float | None]:
    return {
        "valid_n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }


def _paired_metric_summary(
    baseline: OOFRun, candidate: OOFRun, sample_ids: Iterable[str], metric: str
) -> dict[str, object]:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    deltas: list[float] = []
    selected = list(sample_ids)
    for sample_id in selected:
        old = _metric(
            baseline.rows[sample_id][metric],
            field=metric,
            sample_id=sample_id,
            source=baseline.oof_path,
        )
        new = _metric(
            candidate.rows[sample_id][metric],
            field=metric,
            sample_id=sample_id,
            source=candidate.oof_path,
        )
        if old is None or new is None:
            continue
        baseline_values.append(old)
        candidate_values.append(new)
        deltas.append(new - old)
    old_mean = statistics.fmean(baseline_values) if baseline_values else None
    mean_delta = statistics.fmean(deltas) if deltas else None
    improvement_pct = None
    if old_mean not in (None, 0.0) and mean_delta is not None:
        improvement_pct = -mean_delta / old_mean * 100.0
    delta_summary: dict[str, object] = _statistics(deltas)
    delta_summary.update(
        {
            "definition": "candidate - baseline; negative is improvement",
            "mean_improvement": -mean_delta if mean_delta is not None else None,
            "mean_improvement_pct_vs_baseline": improvement_pct,
            "improved_count": sum(value < 0 for value in deltas),
            "unchanged_count": sum(value == 0 for value in deltas),
            "worsened_count": sum(value > 0 for value in deltas),
        }
    )
    return {
        "selected_n": len(selected),
        "paired_valid_n": len(deltas),
        "missing_pair_count": len(selected) - len(deltas),
        "baseline": _statistics(baseline_values),
        "candidate": _statistics(candidate_values),
        "paired_delta": delta_summary,
    }


def _extreme_tail(
    baseline: OOFRun,
    candidate: OOFRun,
    sample_ids: Iterable[str],
    threshold_px: float,
) -> dict[str, object]:
    paired: list[tuple[str, float, float]] = []
    for sample_id in sample_ids:
        old = _metric(
            baseline.rows[sample_id]["point_mae_px"],
            field="point_mae_px",
            sample_id=sample_id,
            source=baseline.oof_path,
        )
        new = _metric(
            candidate.rows[sample_id]["point_mae_px"],
            field="point_mae_px",
            sample_id=sample_id,
            source=candidate.oof_path,
        )
        if old is not None and new is not None:
            paired.append((sample_id, old, new))
    old_ids = [sample_id for sample_id, old, _ in paired if old > threshold_px]
    new_ids = [sample_id for sample_id, _, new in paired if new > threshold_px]
    old_set, new_set = set(old_ids), set(new_ids)
    return {
        "metric": "point_mae_px",
        "operator": ">",
        "threshold_px": threshold_px,
        "paired_valid_n": len(paired),
        "baseline_count": len(old_ids),
        "candidate_count": len(new_ids),
        "count_delta": len(new_ids) - len(old_ids),
        "resolved_count": len(old_set - new_set),
        "new_count": len(new_set - old_set),
        "persistent_count": len(old_set & new_set),
        "baseline_sample_ids": old_ids,
        "candidate_sample_ids": new_ids,
    }


def _subset_summary(
    baseline: OOFRun,
    candidate: OOFRun,
    sample_ids: Iterable[str],
    threshold_px: float,
) -> dict[str, object]:
    selected = sorted(sample_ids)
    return {
        "sample_count": len(selected),
        "patient_group_count": len({baseline.rows[sample_id]["case_id"] for sample_id in selected}),
        "metrics": {
            metric: _paired_metric_summary(baseline, candidate, selected, metric)
            for metric in METRIC_COLUMNS
        },
        "wrong_side_extreme_tail_proxy": _extreme_tail(
            baseline, candidate, selected, threshold_px
        ),
    }


def _resolve_manifest_path(run: OOFRun) -> Path:
    manifest = run.summary.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str):
        raise ComparisonError(f"Missing manifest.path in {run.summary_path}")
    configured = Path(manifest["path"])
    candidates = [configured]
    if not configured.is_absolute():
        candidates.extend([run.root / configured, run.summary_path.parent / configured])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ComparisonError(
        f"Cannot resolve manifest {configured!s} referenced by {run.summary_path}; "
        "pass --affected-samples explicitly"
    )


def _manifest_rows(run: OOFRun) -> tuple[Path, list[str], dict[str, dict[str, str]]]:
    path = _resolve_manifest_path(run)
    fields, raw_rows = _read_csv(path)
    if "sample_id" not in fields:
        raise ComparisonError(f"Manifest lacks sample_id: {path}")
    rows: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in rows:
            raise ComparisonError(f"Empty or duplicate sample_id in manifest {path}")
        rows[sample_id] = row
    if set(rows) != set(run.rows):
        raise ComparisonError(f"Manifest and OOF sample sets differ for {run.model}: {path}")
    return path, fields, rows


def detect_affected_samples(
    baseline: OOFRun, candidate: OOFRun
) -> tuple[set[str], dict[str, object]]:
    old_path, old_fields, old_rows = _manifest_rows(baseline)
    new_path, new_fields, new_rows = _manifest_rows(candidate)
    columns = [
        column
        for column in CROP_SIGNATURE_COLUMNS
        if column in old_fields and column in new_fields
    ]
    if not columns:
        raise ComparisonError(
            f"No shared crop signature columns in {old_path} and {new_path}; "
            "pass --affected-samples explicitly"
        )
    affected = {
        sample_id
        for sample_id in old_rows
        if any(
            old_rows[sample_id][column].strip() != new_rows[sample_id][column].strip()
            for column in columns
        )
    }
    return affected, {
        "mode": "manifest_crop_signature_diff",
        "baseline_manifest": str(old_path),
        "candidate_manifest": str(new_path),
        "compared_columns": columns,
    }


def _load_affected_file(path: Path) -> set[str]:
    if not path.is_file():
        raise ComparisonError(f"Affected-sample file does not exist: {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return set()
    first = next(csv.reader([lines[0]]), [])
    if "sample_id" in first:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {row["sample_id"].strip() for row in rows if row["sample_id"].strip()}
    return {line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _per_sample_rows(
    baseline: OOFRun, candidate: OOFRun, affected: set[str]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for sample_id in sorted(baseline.rows):
        old_row = baseline.rows[sample_id]
        row: dict[str, object] = {
            "model": baseline.model,
            "sample_id": sample_id,
            "patient_group": old_row["case_id"].strip(),
            "fold": _integer(old_row["fold"], field="fold", source=baseline.oof_path),
            "side": old_row["side"].strip(),
            "affected": str(sample_id in affected).lower(),
        }
        for metric in METRIC_COLUMNS:
            old = _metric(
                old_row[metric], field=metric, sample_id=sample_id, source=baseline.oof_path
            )
            new = _metric(
                candidate.rows[sample_id][metric],
                field=metric,
                sample_id=sample_id,
                source=candidate.oof_path,
            )
            row[f"baseline_{metric}"] = old
            row[f"candidate_{metric}"] = new
            row[f"delta_{metric}"] = new - old if old is not None and new is not None else None
        output.append(row)
    return output


def _write_outputs(
    output_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.json"
    csv_path = output_dir / "per_sample_comparison.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["model", "sample_id", "patient_group", "fold", "side", "affected"]
    for metric in METRIC_COLUMNS:
        fields.extend([f"baseline_{metric}", f"candidate_{metric}", f"delta_{metric}"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_retraining_runs(
    baseline_root: Path,
    candidate_root: Path,
    output_dir: Path,
    *,
    affected_samples_path: Path | None = None,
    extreme_threshold_px: float = 100.0,
) -> dict[str, object]:
    if extreme_threshold_px < 0 or not math.isfinite(extreme_threshold_px):
        raise ComparisonError("extreme_threshold_px must be a finite non-negative number")
    baseline_root = baseline_root.resolve()
    candidate_root = candidate_root.resolve()
    explicit_affected = (
        _load_affected_file(affected_samples_path.resolve())
        if affected_samples_path is not None
        else None
    )
    model_summaries: dict[str, object] = {}
    output_rows: list[dict[str, object]] = []
    known_sample_ids: set[str] = set()
    for model in MODEL_CV_DIRS:
        baseline = load_oof_run(baseline_root, model)
        candidate = load_oof_run(candidate_root, model)
        _validate_paired_runs(baseline, candidate)
        known_sample_ids.update(baseline.rows)
        if explicit_affected is None:
            affected, affected_source = detect_affected_samples(baseline, candidate)
        else:
            affected = set(baseline.rows).intersection(explicit_affected)
            assert affected_samples_path is not None
            affected_source = {
                "mode": "explicit_sample_list",
                "path": str(affected_samples_path.resolve()),
                "sha256": _sha256(affected_samples_path.resolve()),
            }
        all_ids = set(baseline.rows)
        model_summaries[model] = {
            "validation": {
                "same_sample_set": True,
                "same_patient_group_per_sample": True,
                "same_fold_per_sample": True,
                "patient_groups_do_not_cross_folds": True,
                "sample_count": len(all_ids),
                "patient_group_count": len(
                    {row["case_id"].strip() for row in baseline.rows.values()}
                ),
                "folds": sorted(
                    {_integer(row["fold"], field="fold", source=baseline.oof_path) for row in baseline.rows.values()}
                ),
                "baseline_oof_csv": str(baseline.oof_path),
                "candidate_oof_csv": str(candidate.oof_path),
            },
            "affected_detection": affected_source,
            "subsets": {
                "overall": _subset_summary(
                    baseline, candidate, all_ids, extreme_threshold_px
                ),
                "affected_recropped": _subset_summary(
                    baseline, candidate, affected, extreme_threshold_px
                ),
            },
        }
        output_rows.extend(_per_sample_rows(baseline, candidate, affected))
    if explicit_affected is not None:
        unknown = sorted(explicit_affected - known_sample_ids)
        if unknown:
            raise ComparisonError(
                f"Affected-sample list contains IDs absent from all three OOF cohorts: {unknown[:10]}"
            )
    summary: dict[str, object] = {
        "schema_version": 1,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "metric_distribution": "paired-valid samples; p95 uses linear interpolation",
        "paired_delta_definition": "candidate - baseline; negative is improvement",
        "models": model_summaries,
        "outputs": {
            "summary_json": str((output_dir / "comparison_summary.json").resolve()),
            "per_sample_csv": str((output_dir / "per_sample_comparison.csv").resolve()),
        },
    }
    _write_outputs(output_dir, summary, output_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--affected-samples",
        type=Path,
        help="Optional CSV with sample_id column or newline-delimited sample IDs. "
        "Without it, crop fields in the two manifests are compared.",
    )
    parser.add_argument("--extreme-threshold-px", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_retraining_runs(
        args.baseline_root,
        args.candidate_root,
        args.output_dir,
        affected_samples_path=args.affected_samples,
        extreme_threshold_px=args.extreme_threshold_px,
    )
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

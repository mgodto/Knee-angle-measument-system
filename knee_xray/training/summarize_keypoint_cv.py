#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


EXPECTED_NUM_FOLDS = 5
CASE_ID_HASH_ALGORITHM = "sha256_json_sorted_unique_utf8_v1"
METRIC_COLUMNS = {
    "point_mae_px": "point_mae_px",
    "nme_height_pct": "nme_height_pct",
    "mldfa_mae_deg": "mldfa_abs_error_deg",
    "mpta_mae_deg": "mpta_abs_error_deg",
    "jlca_mae_deg": "jlca_abs_error_deg",
    "hka_mae_deg": "hka_abs_error_deg",
}


class CVSummaryError(ValueError):
    """Raised when fold evaluation artifacts cannot be safely combined."""


@dataclass(frozen=True)
class FoldEvaluation:
    fold: int
    summary_path: Path
    per_sample_path: Path
    summary: dict[str, object]
    rows: list[dict[str, str]]
    fieldnames: list[str]


def resolve_summary_path(value: Path) -> Path:
    path = value.resolve()
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {path}")
    return path


def resolve_per_sample_path(summary_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CVSummaryError(f"Missing outputs.per_sample_csv in {summary_path}")
    path = Path(value)
    if path.is_absolute():
        candidate = path
    else:
        summary_relative = summary_path.parent / path
        candidate = summary_relative if summary_relative.exists() else path.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Per-sample CSV not found for {summary_path}: {candidate}")
    return candidate.resolve()


def nested_dict(value: object, name: str, source: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CVSummaryError(f"{name} must be an object in {source}")
    return value


def case_ids_sha256(case_ids: set[str]) -> str:
    payload = json.dumps(
        sorted(case_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_summary_provenance(
    summary: dict[str, object],
    summary_path: Path,
    manifest: dict[str, object],
    selection: dict[str, object],
    rows: list[dict[str, str]],
) -> None:
    provenance = nested_dict(summary.get("provenance"), "provenance", summary_path)
    if provenance.get("verified") is not True:
        raise CVSummaryError(f"Unverified validation provenance in {summary_path}")
    split_reference = summary.get("split_reference_manifest")
    if split_reference is None:
        provenance_manifest_sha256 = manifest.get("sha256")
    else:
        if summary.get("evaluation_design") != "fixed_checkpoint_split_with_preprocessing_intervention":
            raise CVSummaryError(f"Invalid preprocessing-intervention metadata in {summary_path}")
        reference = nested_dict(
            split_reference,
            "split_reference_manifest",
            summary_path,
        )
        provenance_manifest_sha256 = reference.get("sha256")
        if not isinstance(provenance_manifest_sha256, str) or not provenance_manifest_sha256:
            raise CVSummaryError(f"Missing split-reference manifest SHA-256 in {summary_path}")
    expected_metadata = {
        "manifest_sha256": provenance_manifest_sha256,
        "num_folds": int(selection["num_folds"]),
        "fold": int(selection["fold"]),
        "seed": int(selection["seed"]),
        "case_id_hash_algorithm": CASE_ID_HASH_ALGORITHM,
    }
    for field, expected in expected_metadata.items():
        if provenance.get(field) != expected:
            raise CVSummaryError(
                f"Provenance {field} does not match evaluation metadata in {summary_path}"
            )

    val_case_ids = {row.get("case_id", "").strip() for row in rows}
    if "" in val_case_ids:
        raise CVSummaryError(f"Empty case_id in {summary_path}")
    expected_val_hash = case_ids_sha256(val_case_ids)
    if provenance.get("val_case_count") != len(val_case_ids):
        raise CVSummaryError(f"Provenance val_case_count does not match CSV in {summary_path}")
    if provenance.get("val_case_ids_sha256") != expected_val_hash:
        raise CVSummaryError(f"Provenance val case hash does not match CSV in {summary_path}")
    train_count = provenance.get("train_case_count")
    train_hash = provenance.get("train_case_ids_sha256")
    if not isinstance(train_count, int) or train_count < 0:
        raise CVSummaryError(f"Invalid provenance train_case_count in {summary_path}")
    if not isinstance(train_hash, str) or len(train_hash) != 64:
        raise CVSummaryError(f"Invalid provenance train case hash in {summary_path}")


def load_fold_evaluation(summary_path: Path) -> FoldEvaluation:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CVSummaryError(f"Invalid JSON in {summary_path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise CVSummaryError(f"Summary root must be an object: {summary_path}")

    selection = nested_dict(summary.get("selection"), "selection", summary_path)
    manifest = nested_dict(summary.get("manifest"), "manifest", summary_path)
    outputs = nested_dict(summary.get("outputs"), "outputs", summary_path)
    if selection.get("split") != "val":
        raise CVSummaryError(f"Only validation-split evaluations can be combined: {summary_path}")
    try:
        num_folds = int(selection["num_folds"])
        fold = int(selection["fold"])
        int(selection["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CVSummaryError(f"Invalid fold metadata in {summary_path}") from exc
    if num_folds != EXPECTED_NUM_FOLDS:
        raise CVSummaryError(
            f"Expected num_folds={EXPECTED_NUM_FOLDS}, got {num_folds} in {summary_path}"
        )
    if fold < 0 or fold >= EXPECTED_NUM_FOLDS:
        raise CVSummaryError(f"Fold must be in 0..{EXPECTED_NUM_FOLDS - 1}: {summary_path}")
    manifest_sha256 = manifest.get("sha256")
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise CVSummaryError(f"Missing manifest.sha256 in {summary_path}")

    per_sample_path = resolve_per_sample_path(summary_path, outputs.get("per_sample_csv"))
    with per_sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required_fields = {"sample_id", "case_id", *METRIC_COLUMNS.values()}
    missing_fields = sorted(required_fields.difference(fieldnames))
    if missing_fields:
        raise CVSummaryError(
            f"Missing per-sample columns {missing_fields} in {per_sample_path}"
        )
    expected_rows = manifest.get("selected_rows")
    if expected_rows is not None:
        try:
            expected_rows = int(expected_rows)
        except (TypeError, ValueError) as exc:
            raise CVSummaryError(f"Invalid manifest.selected_rows in {summary_path}") from exc
        if expected_rows != len(rows):
            raise CVSummaryError(
                f"manifest.selected_rows={expected_rows}, but CSV has {len(rows)} rows: {summary_path}"
            )
    validate_summary_provenance(summary, summary_path, manifest, selection, rows)
    return FoldEvaluation(
        fold=fold,
        summary_path=summary_path,
        per_sample_path=per_sample_path,
        summary=summary,
        rows=rows,
        fieldnames=fieldnames,
    )


def finite_metric(value: str, *, column: str, sample_id: str, source: Path) -> float | None:
    text = value.strip()
    if not text or text.lower() in {"na", "nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise CVSummaryError(
            f"Invalid {column}={value!r} for sample {sample_id!r} in {source}"
        ) from exc
    return number if math.isfinite(number) else None


def validate_fold_set(evaluations: list[FoldEvaluation]) -> tuple[str, int]:
    folds: dict[int, Path] = {}
    manifest_hashes: dict[str, list[Path]] = {}
    seeds: dict[int, list[Path]] = {}
    for evaluation in evaluations:
        if evaluation.fold in folds:
            raise CVSummaryError(
                f"Duplicate fold {evaluation.fold}: {folds[evaluation.fold]} and {evaluation.summary_path}"
            )
        folds[evaluation.fold] = evaluation.summary_path
        manifest = nested_dict(
            evaluation.summary.get("manifest"), "manifest", evaluation.summary_path
        )
        selection = nested_dict(
            evaluation.summary.get("selection"), "selection", evaluation.summary_path
        )
        manifest_hashes.setdefault(str(manifest["sha256"]), []).append(evaluation.summary_path)
        seeds.setdefault(int(selection["seed"]), []).append(evaluation.summary_path)
    if len(manifest_hashes) != 1:
        details = {key: [str(path) for path in paths] for key, paths in manifest_hashes.items()}
        raise CVSummaryError(f"Evaluation manifest SHA-256 values differ: {details}")
    if len(seeds) != 1:
        details = {key: [str(path) for path in paths] for key, paths in seeds.items()}
        raise CVSummaryError(f"Evaluation seeds differ: {details}")
    return next(iter(manifest_hashes)), next(iter(seeds))


def validate_unique_samples(evaluations: list[FoldEvaluation]) -> None:
    seen: dict[str, tuple[int, Path]] = {}
    for evaluation in evaluations:
        for row in evaluation.rows:
            sample_id = row.get("sample_id", "").strip()
            if not sample_id:
                raise CVSummaryError(f"Empty sample_id in {evaluation.per_sample_path}")
            if sample_id in seen:
                previous_fold, previous_path = seen[sample_id]
                raise CVSummaryError(
                    f"Duplicate sample_id {sample_id!r}: fold {previous_fold} in {previous_path} "
                    f"and fold {evaluation.fold} in {evaluation.per_sample_path}"
                )
            seen[sample_id] = (evaluation.fold, evaluation.per_sample_path)


def validate_case_partitions(evaluations: list[FoldEvaluation]) -> bool:
    fold_case_ids: dict[int, set[str]] = {
        evaluation.fold: {row["case_id"].strip() for row in evaluation.rows}
        for evaluation in evaluations
    }
    seen_cases: dict[str, int] = {}
    for fold, case_ids in fold_case_ids.items():
        for case_id in case_ids:
            if case_id in seen_cases:
                raise CVSummaryError(
                    f"case_id {case_id!r} appears in validation folds {seen_cases[case_id]} and {fold}"
                )
            seen_cases[case_id] = fold

    complete = sorted(fold_case_ids) == list(range(EXPECTED_NUM_FOLDS))
    if not complete:
        return False
    all_case_ids = set().union(*fold_case_ids.values())
    for evaluation in evaluations:
        provenance = nested_dict(
            evaluation.summary.get("provenance"),
            "provenance",
            evaluation.summary_path,
        )
        expected_train_cases = all_case_ids.difference(fold_case_ids[evaluation.fold])
        if provenance.get("train_case_count") != len(expected_train_cases):
            raise CVSummaryError(
                f"Provenance train_case_count does not match the complete fold partition in "
                f"{evaluation.summary_path}"
            )
        if provenance.get("train_case_ids_sha256") != case_ids_sha256(expected_train_cases):
            raise CVSummaryError(
                f"Provenance train case hash does not match the complete fold partition in "
                f"{evaluation.summary_path}"
            )
    return True


def aggregate_metric(
    evaluations: list[FoldEvaluation],
    column: str,
) -> dict[str, object]:
    all_values: list[float] = []
    fold_rows: list[dict[str, int | float | None]] = []
    for evaluation in evaluations:
        values = [
            value
            for row in evaluation.rows
            if (
                value := finite_metric(
                    row[column],
                    column=column,
                    sample_id=row["sample_id"],
                    source=evaluation.per_sample_path,
                )
            )
            is not None
        ]
        all_values.extend(values)
        fold_rows.append(
            {
                "fold": evaluation.fold,
                "mean": statistics.fmean(values) if values else None,
                "valid_n": len(values),
                "failure_count": len(evaluation.rows) - len(values),
            }
        )

    fold_means = [float(row["mean"]) for row in fold_rows if row["mean"] is not None]
    total_rows = sum(len(evaluation.rows) for evaluation in evaluations)
    return {
        "sample_weighted_mean": statistics.fmean(all_values) if all_values else None,
        "valid_n": len(all_values),
        "failure_count": total_rows - len(all_values),
        "fold_mean": statistics.fmean(fold_means) if fold_means else None,
        "fold_sd": statistics.stdev(fold_means) if len(fold_means) >= 2 else None,
        "fold_mean_valid_n": len(fold_means),
        "folds": fold_rows,
    }


def combined_fieldnames(evaluations: list[FoldEvaluation]) -> list[str]:
    fields: list[str] = []
    for evaluation in evaluations:
        for field in evaluation.fieldnames:
            if field not in fields and field not in {"fold", "source_summary_json"}:
                fields.append(field)
    return ["fold", "source_summary_json", *fields]


def write_oof_csv(
    path: Path,
    evaluations: list[FoldEvaluation],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation in evaluations:
            for row in evaluation.rows:
                output_row = {field: row.get(field, "") for field in fieldnames}
                output_row["fold"] = evaluation.fold
                output_row["source_summary_json"] = str(evaluation.summary_path)
                writer.writerow(output_row)


def summarize_cv(
    inputs: list[Path],
    output_dir: Path,
) -> dict[str, object]:
    if not inputs:
        raise CVSummaryError("At least one evaluation summary or directory is required.")
    summary_paths = [resolve_summary_path(value) for value in inputs]
    if len(set(summary_paths)) != len(summary_paths):
        raise CVSummaryError("The same evaluation summary was provided more than once.")
    evaluations = [load_fold_evaluation(path) for path in summary_paths]
    evaluations.sort(key=lambda evaluation: evaluation.fold)
    manifest_sha256, seed = validate_fold_set(evaluations)
    validate_unique_samples(evaluations)
    complete_partition_verified = validate_case_partitions(evaluations)

    metrics = {
        name: aggregate_metric(evaluations, column)
        for name, column in METRIC_COLUMNS.items()
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cv_summary.json"
    oof_path = output_dir / "oof_metrics.csv"
    fields = combined_fieldnames(evaluations)
    write_oof_csv(oof_path, evaluations, fields)

    folds_present = [evaluation.fold for evaluation in evaluations]
    total_samples = sum(len(evaluation.rows) for evaluation in evaluations)
    first_manifest = nested_dict(
        evaluations[0].summary.get("manifest"), "manifest", evaluations[0].summary_path
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "manifest": {
            "sha256": manifest_sha256,
            "path": first_manifest.get("path"),
        },
        "cross_validation": {
            "num_folds": EXPECTED_NUM_FOLDS,
            "seed": seed,
            "folds_present": folds_present,
            "fold_count": len(folds_present),
            "complete_5_fold": folds_present == list(range(EXPECTED_NUM_FOLDS)),
            "total_oof_samples": total_samples,
            "fold_sd_ddof": 1,
        },
        "provenance": {
            "verified": True,
            "case_id_hash_algorithm": CASE_ID_HASH_ALGORITHM,
            "complete_partition_verified": complete_partition_verified,
        },
        "metrics": metrics,
        "sources": [
            {
                "fold": evaluation.fold,
                "summary_json": str(evaluation.summary_path),
                "per_sample_csv": str(evaluation.per_sample_path),
                "samples": len(evaluation.rows),
                "checkpoint": nested_dict(
                    evaluation.summary.get("checkpoint"),
                    "checkpoint",
                    evaluation.summary_path,
                ).get("path"),
                "checkpoint_sha256": nested_dict(
                    evaluation.summary.get("checkpoint"),
                    "checkpoint",
                    evaluation.summary_path,
                ).get("sha256"),
                "provenance": nested_dict(
                    evaluation.summary.get("provenance"),
                    "provenance",
                    evaluation.summary_path,
                ),
            }
            for evaluation in evaluations
        ],
        "outputs": {
            "cv_summary_json": str(summary_path),
            "oof_metrics_csv": str(oof_path),
        },
    }
    intervention_summaries = [
        evaluation.summary
        for evaluation in evaluations
        if evaluation.summary.get("split_reference_manifest") is not None
    ]
    if intervention_summaries:
        if len(intervention_summaries) != len(evaluations):
            raise CVSummaryError("Cannot mix ordinary and preprocessing-intervention evaluations.")
        reference_hashes = {
            str(
                nested_dict(
                    evaluation.summary.get("split_reference_manifest"),
                    "split_reference_manifest",
                    evaluation.summary_path,
                ).get("sha256")
            )
            for evaluation in evaluations
        }
        if len(reference_hashes) != 1:
            raise CVSummaryError(
                f"Split-reference manifest SHA-256 values differ: {sorted(reference_hashes)}"
            )
        first_reference = nested_dict(
            evaluations[0].summary.get("split_reference_manifest"),
            "split_reference_manifest",
            evaluations[0].summary_path,
        )
        summary["evaluation_design"] = "fixed_checkpoint_split_with_preprocessing_intervention"
        summary["split_reference_manifest"] = {
            "sha256": next(iter(reference_hashes)),
            "path": first_reference.get("path"),
        }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine compatible keypoint validation-fold evaluations into OOF CV metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m knee_xray.training.summarize_keypoint_cv "
            "outputs/eval/fold0 outputs/eval/fold1/summary.json "
            "--output-dir outputs/eval/cv"
        ),
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Evaluation summary.json files or directories containing summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/keypoint_cv_summary"),
        help="Writes cv_summary.json and oof_metrics.csv here.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_cv(args.inputs, args.output_dir)
    print(f"CV summary: {summary['outputs']['cv_summary_json']}")
    print(f"OOF metrics: {summary['outputs']['oof_metrics_csv']}")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

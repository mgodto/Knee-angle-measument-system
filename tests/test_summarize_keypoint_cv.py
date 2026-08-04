from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from summarize_keypoint_cv import (
    CASE_ID_HASH_ALGORITHM,
    CVSummaryError,
    case_ids_sha256,
    summarize_cv,
)


CSV_FIELDS = (
    "sample_id",
    "case_id",
    "point_mae_px",
    "nme_height_pct",
    "mldfa_abs_error_deg",
    "mpta_abs_error_deg",
    "jlca_abs_error_deg",
    "hka_abs_error_deg",
)


def write_evaluation(
    root: Path,
    name: str,
    fold: int,
    rows: list[dict[str, object]],
    *,
    manifest_sha256: str = "a" * 64,
    seed: int = 42,
    num_folds: int = 5,
    train_case_ids: set[str] | None = None,
    provenance_verified: bool = True,
    split_reference_sha256: str | None = None,
) -> Path:
    evaluation_dir = root / name
    evaluation_dir.mkdir()
    csv_path = evaluation_dir / "per_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    val_case_ids = {str(row["case_id"]) for row in rows}
    train_case_ids = train_case_ids or set()
    summary_path = evaluation_dir / "summary.json"
    payload = {
                "schema_version": 1,
                "checkpoint": {
                    "path": f"/checkpoints/fold{fold}.pt",
                    "sha256": str(fold) * 64,
                },
                "manifest": {
                    "path": "/data/manifest.csv",
                    "sha256": manifest_sha256,
                    "selected_rows": len(rows),
                },
                "selection": {
                    "split": "val",
                    "num_folds": num_folds,
                    "fold": fold,
                    "seed": seed,
                },
                "provenance": {
                    "verified": provenance_verified,
                    "manifest_sha256": split_reference_sha256 or manifest_sha256,
                    "num_folds": num_folds,
                    "fold": fold,
                    "seed": seed,
                    "case_id_hash_algorithm": CASE_ID_HASH_ALGORITHM,
                    "train_case_count": len(train_case_ids),
                    "val_case_count": len(val_case_ids),
                    "train_case_ids_sha256": case_ids_sha256(train_case_ids),
                    "val_case_ids_sha256": case_ids_sha256(val_case_ids),
                },
                "outputs": {"per_sample_csv": csv_path.name},
            }
    if split_reference_sha256 is not None:
        payload["evaluation_design"] = "fixed_checkpoint_split_with_preprocessing_intervention"
        payload["split_reference_manifest"] = {
            "path": "/data/original-training-manifest.csv",
            "sha256": split_reference_sha256,
        }
    summary_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return evaluation_dir


def metric_row(
    sample_id: str,
    point: object,
    nme: object,
    mldfa: object,
    mpta: object,
    jlca: object,
    hka: object,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "case_id": f"case-{sample_id}",
        "point_mae_px": point,
        "nme_height_pct": nme,
        "mldfa_abs_error_deg": mldfa,
        "mpta_abs_error_deg": mpta,
        "jlca_abs_error_deg": jlca,
        "hka_abs_error_deg": hka,
    }


class CVSummaryTests(unittest.TestCase):
    def test_accepts_fixed_split_preprocessing_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = write_evaluation(
                root,
                "intervention",
                0,
                [metric_row("a", 1, 1, 1, 1, 1, 1)],
                manifest_sha256="b" * 64,
                split_reference_sha256="a" * 64,
            )

            summary = summarize_cv([evaluation], root / "output")

            self.assertEqual(
                summary["evaluation_design"],
                "fixed_checkpoint_split_with_preprocessing_intervention",
            )
            self.assertEqual(summary["manifest"]["sha256"], "b" * 64)
            self.assertEqual(summary["split_reference_manifest"]["sha256"], "a" * 64)

    def test_marks_complete_when_all_five_folds_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            all_case_ids = {f"case-sample-{fold}" for fold in range(5)}
            inputs = [
                write_evaluation(
                    root,
                    f"fold{fold}",
                    fold,
                    [metric_row(f"sample-{fold}", fold + 1, 1, 1, 1, 1, 1)],
                    train_case_ids=all_case_ids - {f"case-sample-{fold}"},
                )
                for fold in range(5)
            ]

            summary = summarize_cv(inputs, root / "complete")

            self.assertTrue(summary["cross_validation"]["complete_5_fold"])
            self.assertTrue(summary["provenance"]["complete_partition_verified"])
            self.assertEqual(summary["cross_validation"]["folds_present"], [0, 1, 2, 3, 4])
            self.assertEqual(summary["cross_validation"]["total_oof_samples"], 5)

    def test_aggregates_sample_weighted_and_fold_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold0 = write_evaluation(
                root,
                "fold0",
                0,
                [
                    metric_row("a", 1.0, 0.1, 1.0, 2.0, 3.0, 4.0),
                    metric_row("b", 3.0, 0.3, "", 4.0, 5.0, 6.0),
                ],
            )
            fold1 = write_evaluation(
                root,
                "fold1",
                1,
                [metric_row("c", 9.0, 0.9, 5.0, 8.0, 9.0, 10.0)],
            )
            output_dir = root / "combined"

            summary = summarize_cv(
                [fold0, fold1 / "summary.json"],
                output_dir,
            )

            point = summary["metrics"]["point_mae_px"]
            self.assertAlmostEqual(point["sample_weighted_mean"], 13.0 / 3.0)
            self.assertAlmostEqual(point["fold_mean"], 5.5)
            self.assertAlmostEqual(point["fold_sd"], math.sqrt(24.5))
            self.assertEqual(point["valid_n"], 3)
            self.assertEqual(point["failure_count"], 0)

            mldfa = summary["metrics"]["mldfa_mae_deg"]
            self.assertAlmostEqual(mldfa["sample_weighted_mean"], 3.0)
            self.assertEqual(mldfa["valid_n"], 2)
            self.assertEqual(mldfa["failure_count"], 1)
            self.assertAlmostEqual(mldfa["fold_mean"], 3.0)
            self.assertAlmostEqual(mldfa["fold_sd"], math.sqrt(8.0))

            cv = summary["cross_validation"]
            self.assertEqual(cv["folds_present"], [0, 1])
            self.assertFalse(cv["complete_5_fold"])
            self.assertEqual(cv["total_oof_samples"], 3)
            self.assertTrue((output_dir / "cv_summary.json").is_file())
            with (output_dir / "oof_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                oof_rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in oof_rows], ["a", "b", "c"])
            self.assertEqual([row["fold"] for row in oof_rows], ["0", "0", "1"])
            self.assertTrue(all(row["source_summary_json"] for row in oof_rows))

    def test_rejects_different_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold0 = write_evaluation(root, "fold0", 0, [metric_row("a", 1, 1, 1, 1, 1, 1)])
            fold1 = write_evaluation(
                root,
                "fold1",
                1,
                [metric_row("b", 1, 1, 1, 1, 1, 1)],
                manifest_sha256="b" * 64,
            )
            with self.assertRaisesRegex(CVSummaryError, "manifest SHA-256"):
                summarize_cv([fold0, fold1], root / "output")

    def test_rejects_different_seeds_and_duplicate_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold0 = write_evaluation(root, "fold0", 0, [metric_row("a", 1, 1, 1, 1, 1, 1)])
            other_seed = write_evaluation(
                root,
                "other-seed",
                1,
                [metric_row("b", 1, 1, 1, 1, 1, 1)],
                seed=7,
            )
            with self.assertRaisesRegex(CVSummaryError, "seeds differ"):
                summarize_cv([fold0, other_seed], root / "seed-output")

            duplicate_fold = write_evaluation(
                root,
                "duplicate-fold",
                0,
                [metric_row("c", 1, 1, 1, 1, 1, 1)],
            )
            with self.assertRaisesRegex(CVSummaryError, "Duplicate fold 0"):
                summarize_cv([fold0, duplicate_fold], root / "fold-output")

    def test_rejects_duplicate_samples_and_non_five_fold_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold0 = write_evaluation(root, "fold0", 0, [metric_row("same", 1, 1, 1, 1, 1, 1)])
            fold1 = write_evaluation(root, "fold1", 1, [metric_row("same", 1, 1, 1, 1, 1, 1)])
            with self.assertRaisesRegex(CVSummaryError, "Duplicate sample_id"):
                summarize_cv([fold0, fold1], root / "sample-output")

            four_fold = write_evaluation(
                root,
                "four-fold",
                2,
                [metric_row("unique", 1, 1, 1, 1, 1, 1)],
                num_folds=4,
            )
            with self.assertRaisesRegex(CVSummaryError, "Expected num_folds=5"):
                summarize_cv([four_fold], root / "num-fold-output")

    def test_rejects_unverified_or_mismatched_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unverified = write_evaluation(
                root,
                "unverified",
                0,
                [metric_row("a", 1, 1, 1, 1, 1, 1)],
                provenance_verified=False,
            )
            with self.assertRaisesRegex(CVSummaryError, "Unverified validation provenance"):
                summarize_cv([unverified], root / "unverified-output")

            mismatched = write_evaluation(
                root,
                "mismatched",
                1,
                [metric_row("b", 1, 1, 1, 1, 1, 1)],
            )
            summary_path = mismatched / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["provenance"]["val_case_ids_sha256"] = "0" * 64
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(CVSummaryError, "val case hash"):
                summarize_cv([mismatched], root / "mismatched-output")


if __name__ == "__main__":
    unittest.main()

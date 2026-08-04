from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from compare_retraining_oof import (
    METRIC_COLUMNS,
    MODEL_CV_DIRS,
    ComparisonError,
    compare_retraining_runs,
)


SAMPLES = [f"sample-{index}" for index in range(5)]
BASELINE_POINTS = [10.0, 120.0, 30.0, 40.0, 50.0]
CANDIDATE_POINTS = [8.0, 80.0, 45.0, 35.0, 50.0]


def write_run(root: Path, points: list[float], *, changed_crop: set[str] | None = None) -> None:
    changed_crop = changed_crop or set()
    for model_index, cv_name in enumerate(MODEL_CV_DIRS.values()):
        cv_dir = root / "evaluations" / cv_name
        cv_dir.mkdir(parents=True)
        oof_path = cv_dir / "oof_metrics.csv"
        fields = ["fold", "sample_id", "case_id", "side", *METRIC_COLUMNS]
        with oof_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, sample_id in enumerate(SAMPLES):
                point = points[index] + model_index
                row = {
                    "fold": index,
                    "sample_id": sample_id,
                    "case_id": f"patient_group:{index}",
                    "side": "L" if index % 2 else "R",
                    "point_mae_px": point,
                    "nme_height_pct": point / 10.0,
                    "mldfa_abs_error_deg": point / 20.0,
                    "mpta_abs_error_deg": point / 25.0,
                    "jlca_abs_error_deg": point / 30.0,
                    "hka_abs_error_deg": point / 40.0,
                }
                writer.writerow(row)
        manifest_path = root / f"{cv_name}_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "case_id",
                    "crop_x0",
                    "crop_x1",
                    "crop_width",
                    "crop_method",
                    "is_cropped",
                    "image_width",
                    "image_height",
                ],
            )
            writer.writeheader()
            for index, sample_id in enumerate(SAMPLES):
                width = 500 if sample_id in changed_crop else 1000
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "case_id": f"patient_group:{index}",
                        "crop_x0": 0,
                        "crop_x1": width,
                        "crop_width": width,
                        "crop_method": "target_leg_crop" if width == 500 else "already_single_leg",
                        "is_cropped": str(width == 500),
                        "image_width": width,
                        "image_height": 2000,
                    }
                )
        (cv_dir / "cv_summary.json").write_text(
            json.dumps(
                {
                    "manifest": {"path": str(manifest_path), "sha256": "a" * 64},
                    "cross_validation": {
                        "num_folds": 5,
                        "seed": 42,
                        "folds_present": [0, 1, 2, 3, 4],
                        "fold_count": 5,
                        "complete_5_fold": True,
                        "total_oof_samples": 5,
                    },
                    "provenance": {
                        "verified": True,
                        "complete_partition_verified": True,
                    },
                }
            ),
            encoding="utf-8",
        )


def update_oof(root: Path, model: str, sample_id: str, **changes: object) -> None:
    path = root / "evaluations" / MODEL_CV_DIRS[model] / "oof_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row["sample_id"] == sample_id:
            row.update({key: str(value) for key, value in changes.items()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RetrainingOOFComparisonTests(unittest.TestCase):
    def test_writes_paired_statistics_tail_and_per_sample_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            output = root / "comparison"
            write_run(baseline, BASELINE_POINTS)
            write_run(candidate, CANDIDATE_POINTS)
            affected = root / "affected.csv"
            affected.write_text("sample_id\nsample-1\nsample-2\n", encoding="utf-8")

            summary = compare_retraining_runs(
                baseline, candidate, output, affected_samples_path=affected
            )

            bone = summary["models"]["Bone"]
            self.assertEqual(bone["subsets"]["overall"]["sample_count"], 5)
            self.assertEqual(bone["subsets"]["affected_recropped"]["sample_count"], 2)
            point = bone["subsets"]["overall"]["metrics"]["point_mae_px"]
            self.assertAlmostEqual(point["baseline"]["median"], 40.0)
            self.assertAlmostEqual(point["candidate"]["max"], 80.0)
            self.assertAlmostEqual(point["paired_delta"]["mean"], -6.4)
            self.assertEqual(point["paired_delta"]["improved_count"], 3)
            tail = bone["subsets"]["overall"]["wrong_side_extreme_tail_proxy"]
            self.assertEqual(tail["baseline_count"], 1)
            self.assertEqual(tail["candidate_count"], 0)
            self.assertEqual(tail["resolved_count"], 1)
            self.assertTrue((output / "comparison_summary.json").is_file())
            with (output / "per_sample_comparison.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 15)
            row = next(
                item for item in rows if item["model"] == "Bone" and item["sample_id"] == "sample-1"
            )
            self.assertEqual(row["affected"], "true")
            self.assertEqual(float(row["delta_point_mae_px"]), -40.0)

    def test_auto_detects_recropped_samples_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_run(baseline, BASELINE_POINTS)
            write_run(candidate, CANDIDATE_POINTS, changed_crop={"sample-1"})

            summary = compare_retraining_runs(baseline, candidate, root / "output")

            for model in MODEL_CV_DIRS:
                model_summary = summary["models"][model]
                self.assertEqual(
                    model_summary["affected_detection"]["mode"],
                    "manifest_crop_signature_diff",
                )
                self.assertEqual(
                    model_summary["subsets"]["affected_recropped"]["sample_count"], 1
                )

    def test_rejects_sample_patient_group_and_fold_mismatches(self) -> None:
        for mismatch, changes, message in (
            ("patient", {"case_id": "patient_group:different"}, "patient_group differs"),
            ("fold", {"fold": 1}, "fold differs"),
        ):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = root / "baseline"
                candidate = root / "candidate"
                write_run(baseline, BASELINE_POINTS)
                write_run(candidate, CANDIDATE_POINTS)
                update_oof(candidate, "Bone", "sample-0", **changes)
                if mismatch == "fold":
                    update_oof(candidate, "Bone", "sample-1", fold=0)
                with self.assertRaisesRegex(ComparisonError, message):
                    compare_retraining_runs(baseline, candidate, root / "output")

    def test_rejects_different_sample_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_run(baseline, BASELINE_POINTS)
            write_run(candidate, CANDIDATE_POINTS)
            path = candidate / "evaluations" / MODEL_CV_DIRS["TKA"] / "oof_metrics.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["sample_id"] = "candidate-only"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ComparisonError, "sample sets differ"):
                compare_retraining_runs(baseline, candidate, root / "output")


if __name__ == "__main__":
    unittest.main()

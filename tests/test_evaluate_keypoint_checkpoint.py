from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

from knee_xray.training.evaluate_keypoint_checkpoint import run_evaluation, select_manifest_rows, sha256_path
from knee_xray.ml.knee_keypoint_model import (
    ADAPTER_ID,
    ARCHITECTURE_ID,
    KEYPOINT_NAMES,
    PREPROCESSING_ID,
    SmallHeatmapNet,
)
from knee_xray.core.measure_angles import measure_from_named_points
from knee_xray.training.train_keypoint_baseline import case_split_provenance


def synthetic_geometry() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    points = {
        "hip": {"x": 180.0, "y": 70.0},
        "upper_left": {"x": 115.0, "y": 390.0},
        "upper_center": {"x": 200.0, "y": 380.0},
        "upper_right": {"x": 285.0, "y": 370.0},
        "lower_left": {"x": 115.0, "y": 410.0},
        "lower_center": {"x": 205.0, "y": 420.0},
        "lower_right": {"x": 285.0, "y": 430.0},
        "ankle": {"x": 230.0, "y": 730.0},
    }
    lines = {
        "upper_line": {
            "p1": {"x": 115.0, "y": 390.0},
            "p2": {"x": 285.0, "y": 370.0},
        },
        "lower_line": {
            "p1": {"x": 115.0, "y": 410.0},
            "p2": {"x": 285.0, "y": 430.0},
        },
    }
    return points, lines


def coords_from_geometry(
    points: dict[str, dict[str, float]],
    lines: dict[str, dict[str, dict[str, float]]],
) -> np.ndarray:
    values: list[list[float]] = []
    for name in KEYPOINT_NAMES:
        if name.endswith("_p1") or name.endswith("_p2"):
            line_name, endpoint = name.rsplit("_", 1)
            point = lines[line_name][endpoint]
        else:
            point = points[name]
        values.append([point["x"], point["y"]])
    return np.asarray(values, dtype=np.float32)


class RowSelectionTests(unittest.TestCase):
    def test_validation_rows_match_trainer_case_split(self) -> None:
        rows = [
            {"sample_id": f"sample-{index}", "case_id": f"case-{index // 2}"}
            for index in range(20)
        ]
        selected = select_manifest_rows(rows, split="val", num_folds=5, fold=0, seed=42)
        selected_case_ids = {row["case_id"] for row in selected}

        case_ids = sorted({row["case_id"] for row in rows})
        import random

        random.Random(42).shuffle(case_ids)
        self.assertEqual(selected_case_ids, set(case_ids[0::5]))
        self.assertTrue(selected)


class CheckpointEvaluationTests(unittest.TestCase):
    def create_fixture(self, root: Path) -> tuple[Path, Path, np.ndarray]:
        raw_path = root / "sample.png"
        raw_image = np.zeros((800, 400, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(raw_path), raw_image))

        points, lines = synthetic_geometry()
        annotation_path = root / "sample_annotation.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "raw_path": str(raw_path),
                    "raw_filename": raw_path.name,
                    "image_width": 400,
                    "image_height": 800,
                    "side": "L",
                    "points": points,
                    "lines": lines,
                }
            ),
            encoding="utf-8",
        )
        measurement, _debug = measure_from_named_points(
            raw_image,
            points,
            raw_path=raw_path,
            named_lines=lines,
            side="L",
            render_component_images=False,
        )

        manifest_path = root / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "case_id",
                    "side",
                    "annotation_path",
                    "raw_path",
                    "image_width",
                    "image_height",
                    "mldfa",
                    "mpta",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "sample",
                    "case_id": "case-1",
                    "side": "L",
                    "annotation_path": annotation_path.name,
                    "raw_path": raw_path.name,
                    "image_width": 400,
                    "image_height": 800,
                    "mldfa": measurement["mldfa_angle"],
                    "mpta": measurement["mpta_angle"],
                }
            )

        checkpoint_path = root / "best.pt"
        model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES))
        torch.save(
            {
                "checkpoint_schema_version": 1,
                "adapter_id": ADAPTER_ID,
                "architecture_id": ARCHITECTURE_ID,
                "preprocessing_id": PREPROCESSING_ID,
                "model_state": model.state_dict(),
                "keypoint_names": KEYPOINT_NAMES,
                "image_width": 32,
                "image_height": 64,
                "stride": 4,
                "epoch": 1,
            },
            checkpoint_path,
        )
        return checkpoint_path, manifest_path, coords_from_geometry(points, lines)

    def add_training_manifest_sha(self, checkpoint_path: Path, manifest_path: Path) -> dict[str, object]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["training_manifest_sha256"] = sha256_path(manifest_path)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint

    def test_writes_zero_error_json_and_csv_for_exact_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, manifest_path, target_coords = self.create_fixture(root)
            output_dir = root / "evaluation"

            with mock.patch(
                "knee_xray.training.evaluate_keypoint_checkpoint.predict_keypoints",
                return_value=target_coords,
            ):
                summary = run_evaluation(
                    checkpoint_path,
                    manifest_path,
                    output_dir,
                    split="all",
                    device_name="cpu",
                )

            self.assertEqual(summary["manifest"]["selected_rows"], 1)
            self.assertEqual(summary["metrics"]["point_mae_px"]["mean"], 0.0)
            self.assertEqual(summary["metrics"]["nme_height_pct"]["mean"], 0.0)
            for name in ("mldfa", "mpta", "jlca", "hka"):
                metric = summary["metrics"][f"{name}_mae_deg"]
                self.assertAlmostEqual(metric["mean"], 0.0)
                self.assertEqual(metric["valid_n"], 1)
                self.assertEqual(metric["failure_count"], 0)

            summary_path = output_dir / "summary.json"
            csv_path = output_dir / "per_sample_metrics.csv"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(csv_path.is_file())
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["keypoint_count"], 12)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["point_mae_px"]), 0.0)
            self.assertEqual(float(rows[0]["hka_abs_error_deg"]), 0.0)

    def test_counts_degenerate_angle_predictions_as_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, manifest_path, target_coords = self.create_fixture(root)
            output_dir = root / "evaluation-failure"

            with mock.patch(
                "knee_xray.training.evaluate_keypoint_checkpoint.predict_keypoints",
                return_value=np.zeros_like(target_coords),
            ):
                summary = run_evaluation(
                    checkpoint_path,
                    manifest_path,
                    output_dir,
                    split="all",
                    device_name="cpu",
                )

            self.assertGreater(summary["metrics"]["point_mae_px"]["mean"], 0.0)
            for name in ("mldfa", "mpta", "jlca", "hka"):
                metric = summary["metrics"][f"{name}_mae_deg"]
                self.assertIsNone(metric["mean"])
                self.assertEqual(metric["valid_n"], 0)
                self.assertEqual(metric["failure_count"], 1)

    def test_verifies_legacy_fold_checkpoint_from_sibling_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, manifest_path, target_coords = self.create_fixture(root)
            self.add_training_manifest_sha(checkpoint_path, manifest_path)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "manifest": str(manifest_path),
                        "output_dir": str(root),
                        "image_width": 32,
                        "image_height": 64,
                        "stride": 4,
                        "num_folds": 5,
                        "fold": 0,
                        "seed": 42,
                        "model_version": "",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "knee_xray.training.evaluate_keypoint_checkpoint.predict_keypoints",
                return_value=target_coords,
            ):
                summary = run_evaluation(
                    checkpoint_path,
                    manifest_path,
                    root / "evaluation-val",
                    split="val",
                    fold=0,
                    seed=42,
                    device_name="cpu",
                )

            provenance = summary["provenance"]
            self.assertTrue(provenance["verified"])
            self.assertEqual(provenance["source"], "sibling_config")
            self.assertEqual(provenance["val_case_count"], 1)
            self.assertEqual(provenance["train_case_count"], 0)

    def test_rejects_requested_split_that_differs_from_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, manifest_path, _target_coords = self.create_fixture(root)
            checkpoint = self.add_training_manifest_sha(checkpoint_path, manifest_path)
            checkpoint["split_provenance"] = case_split_provenance(
                [{"case_id": "case-1"}],
                [],
                [0],
                5,
                0,
                42,
            )
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "checkpoint split_provenance"):
                run_evaluation(
                    checkpoint_path,
                    manifest_path,
                    root / "wrong-split",
                    split="val",
                    fold=0,
                    seed=7,
                    device_name="cpu",
                )

    def test_preprocessing_intervention_uses_original_checkpoint_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, reference_manifest, target_coords = self.create_fixture(root)
            checkpoint = self.add_training_manifest_sha(checkpoint_path, reference_manifest)
            checkpoint["split_provenance"] = case_split_provenance(
                [{"case_id": "case-1"}],
                [],
                [0],
                5,
                0,
                42,
            )
            torch.save(checkpoint, checkpoint_path)

            evaluation_manifest = root / "single-leg-v2.csv"
            with reference_manifest.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["raw_path"] = str((root / "sample.png").resolve())
            rows[0]["annotation_path"] = str((root / "sample_annotation.json").resolve())
            with evaluation_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            with mock.patch(
                "knee_xray.training.evaluate_keypoint_checkpoint.predict_keypoints",
                return_value=target_coords,
            ):
                summary = run_evaluation(
                    checkpoint_path,
                    evaluation_manifest,
                    root / "preprocessing-intervention",
                    split="val",
                    fold=0,
                    seed=42,
                    device_name="cpu",
                    split_reference_manifest_path=reference_manifest,
                )

            self.assertEqual(summary["manifest"]["path"], str(evaluation_manifest.resolve()))
            self.assertEqual(
                summary["split_reference_manifest"]["path"],
                str(reference_manifest.resolve()),
            )
            self.assertEqual(
                summary["evaluation_design"],
                "fixed_checkpoint_split_with_preprocessing_intervention",
            )
            self.assertTrue(summary["provenance"]["verified"])

    def test_preprocessing_intervention_rejects_patient_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, reference_manifest, _target_coords = self.create_fixture(root)
            evaluation_manifest = root / "mismatch.csv"
            with reference_manifest.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["case_id"] = "different-patient"
            with evaluation_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "case_id differs"):
                run_evaluation(
                    checkpoint_path,
                    evaluation_manifest,
                    root / "mismatch-output",
                    split="val",
                    fold=0,
                    seed=42,
                    device_name="cpu",
                    split_reference_manifest_path=reference_manifest,
                )


if __name__ == "__main__":
    unittest.main()

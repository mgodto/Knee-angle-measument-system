from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from train_keypoint_baseline import (
    case_ids_sha256,
    case_split_provenance,
    complete_angle_checkpoint_score,
    heatmap_mse_loss,
    load_cross_validation_metrics,
)


class HeatmapLossTests(unittest.TestCase):
    def test_peak_weight_one_matches_original_loss(self) -> None:
        logits = torch.tensor([[[[-1.0, 0.5], [2.0, -0.25]]]])
        targets = torch.tensor([[[[0.0, 0.25], [1.0, 0.0]]]])

        actual = heatmap_mse_loss(logits, targets, peak_weight=1.0)
        expected = F.mse_loss(torch.sigmoid(logits), targets)

        torch.testing.assert_close(actual, expected)

    def test_peak_weight_emphasizes_target_center(self) -> None:
        targets = torch.tensor([[[[0.0, 1.0]]]])
        background_error = heatmap_mse_loss(
            torch.tensor([[[[10.0, 10.0]]]]), targets, peak_weight=20.0
        )
        peak_error = heatmap_mse_loss(
            torch.tensor([[[[-10.0, -10.0]]]]), targets, peak_weight=20.0
        )

        self.assertGreater(float(peak_error), float(background_error) * 10.0)


class CrossValidationMetadataTests(unittest.TestCase):
    def test_case_split_provenance_is_deterministic(self) -> None:
        rows = [
            {"case_id": "case-b"},
            {"case_id": "case-a"},
            {"case_id": "case-b"},
            {"case_id": "case-c"},
        ]

        provenance = case_split_provenance(rows, [0, 1, 2], [3], 5, 2, 42)

        self.assertEqual(provenance["fold"], 2)
        self.assertEqual(provenance["train_case_count"], 2)
        self.assertEqual(provenance["val_case_count"], 1)
        self.assertEqual(
            provenance["train_case_ids_sha256"],
            case_ids_sha256(["case-a", "case-b"]),
        )
        self.assertEqual(
            provenance["val_case_ids_sha256"],
            case_ids_sha256(["case-c"]),
        )

    def test_loads_complete_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {"sha256": "manifest-sha"},
                        "cross_validation": {"complete_5_fold": True},
                        "provenance": {
                            "verified": True,
                            "complete_partition_verified": True,
                        },
                        "metrics": {"point_mae_px": {"sample_weighted_mean": 12.5}},
                    }
                ),
                encoding="utf-8",
            )

            metrics, summary_sha = load_cross_validation_metrics(path, "manifest-sha")

            self.assertEqual(metrics, {"point_mae_px": 12.5})
            self.assertEqual(len(summary_sha or ""), 64)

    def test_rejects_summary_for_another_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {"sha256": "other"},
                        "cross_validation": {"complete_5_fold": True},
                        "provenance": {
                            "verified": True,
                            "complete_partition_verified": True,
                        },
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "different training manifest"):
                load_cross_validation_metrics(path, "expected")

    def test_rejects_unverified_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {"sha256": "manifest-sha"},
                        "cross_validation": {"complete_5_fold": True},
                        "provenance": {"verified": False},
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "verified fold provenance"):
                load_cross_validation_metrics(path, "manifest-sha")


class AngleCheckpointSelectionTests(unittest.TestCase):
    def test_requires_complete_angle_coverage(self) -> None:
        complete = {
            "mldfa_mae_deg": 2.0,
            "mldfa_valid_n": 10,
            "mpta_mae_deg": 4.0,
            "mpta_valid_n": 10,
        }
        incomplete = {**complete, "mpta_valid_n": 9}

        self.assertEqual(complete_angle_checkpoint_score(complete, 10), 3.0)
        self.assertIsNone(complete_angle_checkpoint_score(incomplete, 10))

    def test_rejects_non_finite_angle_metric(self) -> None:
        metrics = {
            "mldfa_mae_deg": float("nan"),
            "mldfa_valid_n": 3,
            "mpta_mae_deg": 2.0,
            "mpta_valid_n": 3,
        }

        self.assertIsNone(complete_angle_checkpoint_score(metrics, 3))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from build_dataset_manifest import MANIFEST_FIELDS
from measure_angles import ANNOTATION_LINE_NAMES, ANNOTATION_POINT_NAMES
from process_annotation_dataset import CropBox, adjust_annotation, choose_crop_box, process_dataset


def annotation_at(
    center_x: float,
    *,
    image_width: int = 1000,
    image_height: int = 1200,
    input_scope: str = "single-leg raster X-ray",
    inference_roi: dict | None = None,
) -> dict:
    point_offsets = (-30, -20, -10, 0, 10, 20, 25, 30)
    points = {
        name: {"x": center_x + point_offsets[index], "y": 100.0 + index * 120.0}
        for index, name in enumerate(ANNOTATION_POINT_NAMES)
    }
    lines = {
        name: {
            "p1": {"x": center_x - 40.0, "y": 610.0 + index * 20.0},
            "p2": {"x": center_x + 40.0, "y": 615.0 + index * 20.0},
        }
        for index, name in enumerate(ANNOTATION_LINE_NAMES)
    }
    analysis = {"side": "L"}
    if inference_roi is not None:
        analysis["inference_roi"] = inference_roi
    return {
        "schema_version": 1,
        "source": {
            "filename": "source.jpg",
            "sha256": "a" * 64,
            "image_width": image_width,
            "image_height": image_height,
            "input_scope": input_scope,
        },
        "analysis": analysis,
        "image_width": image_width,
        "image_height": image_height,
        "side": "L",
        "points": points,
        "lines": lines,
    }


class ProcessAnnotationDatasetTests(unittest.TestCase):
    def test_process_manifest_preserves_source_case_and_writes_bone_tka_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.jpg"
            self.assertTrue(
                cv2.imwrite(
                    str(raw_path),
                    np.full((1000, 400, 3), 127, dtype=np.uint8),
                )
            )
            annotation_path = root / "annotation.json"
            annotation_path.write_text(
                json.dumps(annotation_at(200, image_width=400, image_height=1000)),
                encoding="utf-8",
            )
            manifest_path = root / "input.csv"
            fields = [*MANIFEST_FIELDS, "source_case_id"]
            row = {field: "" for field in fields}
            row.update(
                {
                    "sample_id": "001L_unknown_bone",
                    "case_id": "patient_group:001",
                    "source_case_id": "20260803:001",
                    "source_dataset": "20260803",
                    "dataset_group": "20260803",
                    "implant_status": "bone",
                    "study_phase": "unknown",
                    "side": "L",
                    "annotation_path": annotation_path.as_posix(),
                    "raw_path": raw_path.as_posix(),
                    "raw_filename": raw_path.name,
                    "image_width": "400",
                    "image_height": "1000",
                    "raw_match_count": "1",
                    "mldfa": "90",
                    "mpta": "90",
                }
            )
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

            output_dir = root / "processed"
            with mock.patch(
                "process_annotation_dataset.measure_from_named_points",
                return_value=({"mldfa_angle": 90.0, "mpta_angle": 89.0}, {}),
            ):
                rows = process_dataset(manifest_path, output_dir, render_overlays=False)

            self.assertEqual(rows[0]["case_id"], "patient_group:001")
            self.assertEqual(rows[0]["source_case_id"], "20260803:001")
            with (output_dir / "processed_manifest_bone_tka.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                mixed_rows = list(csv.DictReader(handle))
            self.assertEqual(len(mixed_rows), 1)
            self.assertEqual(mixed_rows[0]["source_case_id"], "20260803:001")

    def test_confirmed_measurement_roi_is_used_and_audited(self) -> None:
        roi = {
            "x0": 550,
            "y0": 20,
            "x1": 1000,
            "y1": 1150,
            "width": 450,
            "height": 1130,
            "coordinate_space": "source_image_pixels",
            "selection_method": "doctor_confirmed",
            "confirmed": True,
        }
        annotation = annotation_at(
            750,
            input_scope="bilateral raster X-ray",
            inference_roi=roi,
        )

        crop = choose_crop_box(
            {"sample_id": "001L_unknown_bone", "raw_path": "raw.jpg"},
            annotation,
            1000,
            1200,
            {},
        )

        self.assertEqual((crop.x0, crop.y0, crop.x1, crop.y1), (550, 20, 1000, 1150))
        self.assertEqual(crop.method, "measurement_inference_roi")
        self.assertEqual(crop.provenance["selection_method"], "doctor_confirmed")
        self.assertEqual(crop.provenance["inference_roi_status"], "accepted")

    def test_invalid_roi_falls_back_to_wide_target_side_crop(self) -> None:
        annotation = annotation_at(
            750,
            input_scope="bilateral raster X-ray",
            inference_roi={
                "x0": 0,
                "y0": 0,
                "x1": 400,
                "y1": 1200,
                "coordinate_space": "source_image_pixels",
                "selection_method": "doctor_confirmed",
                "confirmed": True,
            },
        )

        crop = choose_crop_box(
            {"sample_id": "001L_unknown_bone", "raw_path": "raw.jpg"},
            annotation,
            1000,
            1200,
            {},
        )

        self.assertEqual((crop.x0, crop.x1), (420, 1000))
        self.assertGreaterEqual(crop.width, 0.58 * 1000)
        self.assertEqual(crop.method, "annotation_bbox_horizontal_crop")
        self.assertEqual(crop.provenance["inference_roi_status"], "does_not_cover_annotation")

    def test_existing_paired_separator_crop_is_unchanged(self) -> None:
        annotation = annotation_at(200)
        row = {"sample_id": "001L_pre_bone", "raw_path": "shared.jpg"}
        pair_stats = {
            "shared.jpg": [
                {"sample_id": "001L_pre_bone", "median_x": 200.0},
                {"sample_id": "001R_pre_bone", "median_x": 800.0},
            ]
        }

        crop = choose_crop_box(row, annotation, 1000, 1200, pair_stats)

        self.assertEqual((crop.x0, crop.x1), (0, 583))
        self.assertEqual(crop.method, "paired_horizontal_crop")

    def test_already_processed_pair_is_never_cropped_a_second_time(self) -> None:
        annotation = annotation_at(500, image_width=580)
        annotation["processed_from"] = {
            "annotation_path": "original.json",
            "raw_path": "original.jpg",
            "original_image_width": 1000,
            "original_image_height": 1200,
            "crop": {
                "x0": 0,
                "y0": 0,
                "x1": 580,
                "y1": 1200,
                "width": 580,
                "height": 1200,
                "method": "paired_horizontal_crop",
            },
        }

        crop = choose_crop_box(
            {"sample_id": "001RL_pre_bone", "raw_path": "cropped.jpg"},
            annotation,
            580,
            1200,
            {},
        )

        self.assertEqual((crop.x0, crop.x1), (0, 580))
        self.assertEqual(crop.method, "already_single_leg")
        self.assertEqual(crop.provenance["inference_roi_status"], "already_processed")

    def test_identity_crop_cannot_claim_that_a_broad_image_was_already_split(self) -> None:
        annotation = annotation_at(200)
        annotation["processed_from"] = {
            "annotation_path": "original.json",
            "raw_path": "original.jpg",
            "original_image_width": 1000,
            "original_image_height": 1200,
            "crop": {
                "x0": 0,
                "y0": 0,
                "x1": 1000,
                "y1": 1200,
                "width": 1000,
                "height": 1200,
                "method": "paired_horizontal_crop",
            },
        }

        crop = choose_crop_box(
            {"sample_id": "001L_pre_bone", "raw_path": "broad.jpg"},
            annotation,
            1000,
            1200,
            {},
        )

        self.assertEqual(crop.method, "annotation_bbox_horizontal_crop")
        self.assertLess(crop.width, 1000)

    def test_existing_crop_provenance_requires_exact_integer_consistent_dimensions(self) -> None:
        annotation = annotation_at(300, image_width=580)
        annotation["processed_from"] = {
            "annotation_path": "original.json",
            "raw_path": "original.jpg",
            "original_image_width": 1000,
            "original_image_height": 1200,
            "crop": {
                "x0": 0.5,
                "y0": 0,
                "x1": 580.5,
                "y1": 1200,
                "width": 580,
                "height": 1200,
                "method": "paired_horizontal_crop",
            },
        }
        with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
            choose_crop_box(
                {"sample_id": "001L_pre_bone", "raw_path": "cropped.jpg"},
                annotation,
                580,
                1200,
                {},
            )

        annotation["processed_from"]["crop"].update(
            {"x0": 0, "x1": 580, "width": 579}
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            choose_crop_box(
                {"sample_id": "001L_pre_bone", "raw_path": "cropped.jpg"},
                annotation,
                580,
                1200,
                {},
            )

    def test_annotation_spanning_both_legs_is_rejected(self) -> None:
        annotation = annotation_at(500, input_scope="bilateral raster X-ray")
        annotation["lines"][ANNOTATION_LINE_NAMES[0]]["p1"]["x"] = 100.0
        annotation["lines"][ANNOTATION_LINE_NAMES[0]]["p2"]["x"] = 900.0

        with self.assertRaisesRegex(ValueError, "span is too wide"):
            choose_crop_box(
                {"sample_id": "001L_unknown_bone", "raw_path": "raw.jpg"},
                annotation,
                1000,
                1200,
                {},
            )

    def test_second_crop_composes_absolute_provenance_and_identity_preserves_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_annotation_path = root / "current.json"
            current_raw_path = root / "current.jpg"
            output_raw_path = root / "output.jpg"
            original_annotation_path = root / "original.json"
            original_raw_path = root / "original.jpg"
            for path, content in (
                (current_annotation_path, b"current annotation"),
                (current_raw_path, b"current raw"),
                (original_annotation_path, b"original annotation"),
                (original_raw_path, b"original raw"),
            ):
                path.write_bytes(content)

            annotation = annotation_at(600, image_width=1100)
            annotation["processed_from"] = {
                "annotation_path": original_annotation_path.as_posix(),
                "raw_path": original_raw_path.as_posix(),
                "original_image_width": 2000,
                "original_image_height": 1200,
                "crop": {
                    "x0": 900,
                    "y0": 0,
                    "x1": 2000,
                    "y1": 1200,
                    "width": 1100,
                    "height": 1200,
                    "method": "paired_horizontal_crop",
                },
            }

            identity = adjust_annotation(
                copy.deepcopy(annotation),
                CropBox(0, 0, 1100, 1200, "already_single_leg"),
                output_raw_path,
                current_annotation_path,
                current_raw_path,
            )
            self.assertEqual(identity["processed_from"]["crop"]["x0"], 900)
            self.assertEqual(
                identity["processed_from"]["crop"]["method"],
                "paired_horizontal_crop",
            )

            composed = adjust_annotation(
                copy.deepcopy(annotation),
                CropBox(
                    100,
                    0,
                    1000,
                    1200,
                    "annotation_bbox_horizontal_crop",
                    {"selection_method": "annotation_bbox_horizontal_crop"},
                ),
                output_raw_path,
                current_annotation_path,
                current_raw_path,
            )
            final_crop = composed["processed_from"]["crop"]
            self.assertEqual((final_crop["x0"], final_crop["x1"]), (1000, 1900))
            self.assertEqual(final_crop["width"], 900)
            self.assertEqual(final_crop["provenance_source"], "composed_crop")
            self.assertEqual(composed["processed_from"]["raw_path"], original_raw_path.as_posix())
            self.assertEqual(len(composed["processed_from"]["processing_steps"]), 2)


if __name__ == "__main__":
    unittest.main()

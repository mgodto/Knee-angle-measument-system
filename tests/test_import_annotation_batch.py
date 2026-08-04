from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from import_annotation_batch import (
    Candidate,
    ReplacementPlan,
    adapt_annotation_schema,
    collect_candidates,
    deduplicate_candidates,
    filter_existing_collisions,
    load_overrides,
    merge_stage,
    parse_canonical,
    rewrite_processed_provenance,
    sha256_file,
    validate_dataset_root,
)
from measure_angles import ANNOTATION_LINE_NAMES, ANNOTATION_POINT_NAMES
from organize_implant_dataset import DATASET_FOLDERS


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sample_dir"])
        writer.writeheader()
        writer.writerows(rows)


def make_sample(root: Path, relative_dir: str, marker: str) -> Path:
    sample_dir = root / relative_dir
    sample_dir.mkdir(parents=True)
    (sample_dir / "marker.txt").write_text(marker, encoding="utf-8")
    return sample_dir


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def training_annotation(*, side: str = "R", offset: float = 0.0) -> dict:
    points = {
        name: {"x": 5.0 + index + offset, "y": 6.0 + index}
        for index, name in enumerate(ANNOTATION_POINT_NAMES)
    }
    lines = {
        name: {
            "p1": {"x": 4.0 + index + offset, "y": 14.0 + index},
            "p2": {"x": 14.0 + index + offset, "y": 15.0 + index},
        }
        for index, name in enumerate(ANNOTATION_LINE_NAMES)
    }
    return {
        "image_width": 32,
        "image_height": 32,
        "side": side,
        "points": points,
        "lines": lines,
    }


def measurement_annotation(raw_path: Path, *, side: str = "R", sha256: str | None = None) -> dict:
    annotation = training_annotation(side=side)
    return {
        "schema_version": 1,
        "source": {
            "filename": raw_path.name,
            "sha256": sha256 or sha256_file(raw_path),
            "image_width": annotation["image_width"],
            "image_height": annotation["image_height"],
        },
        "analysis": {"side": side},
        "points": annotation["points"],
        "lines": annotation["lines"],
    }


def bilateral_annotation(
    *,
    side: str,
    screen_side: str,
    confirmed: bool = True,
    include_roi: bool = True,
) -> dict:
    image_width = 1000
    image_height = 1200
    center_x = 200.0 if screen_side == "left" else 800.0
    points = {
        name: {"x": center_x + index - 4.0, "y": 100.0 + index * 100.0}
        for index, name in enumerate(ANNOTATION_POINT_NAMES)
    }
    lines = {
        name: {
            "p1": {"x": center_x - 40.0, "y": 500.0 + index * 40.0},
            "p2": {"x": center_x + 40.0, "y": 505.0 + index * 40.0},
        }
        for index, name in enumerate(ANNOTATION_LINE_NAMES)
    }
    analysis: dict[str, object] = {"side": side}
    if include_roi:
        x0, x1 = (0, 580) if screen_side == "left" else (420, image_width)
        analysis["inference_roi"] = {
            "x0": x0,
            "y0": 0,
            "x1": x1,
            "y1": image_height,
            "width": x1 - x0,
            "height": image_height,
            "coordinate_space": "source_image_pixels",
            "selection_method": "doctor_confirmed",
            "confirmed": confirmed,
        }
    return {
        "source": {
            "filename": "bilateral.jpg",
            "sha256": "a" * 64,
            "image_width": image_width,
            "image_height": image_height,
            "input_scope": "bilateral raster X-ray",
        },
        "analysis": analysis,
        "image_width": image_width,
        "image_height": image_height,
        "side": side,
        "points": points,
        "lines": lines,
    }


def make_candidate(
    batch_dir: Path,
    sample_id: str,
    raw_path: Path,
    *,
    relative_path: str | None = None,
    replace_existing_sample_id: str = "",
    replace_existing_sample_dir: str = "",
    allow_raw_replacement: bool = False,
) -> Candidate:
    relative_path = relative_path or f"{sample_id}.json"
    return Candidate(
        annotation_path=batch_dir / relative_path,
        relative_path=relative_path,
        sample_id=sample_id,
        case_number=sample_id.split("_", 1)[0],
        side="R",
        phase="unknown",
        implant_status="bone",
        case_id=f"batch:{sample_id}",
        annotation=training_annotation(),
        raw_path=raw_path,
        raw_match_count=1,
        raw_sha256=sha256_file(raw_path),
        source_json_sha256=hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
        mldfa=90.0,
        mpta=90.0,
        inference_rank=0,
        replace_existing_sample_id=replace_existing_sample_id,
        replace_existing_sample_dir=replace_existing_sample_dir,
        allow_raw_replacement=allow_raw_replacement,
    )


def make_bilateral_candidate(
    batch_dir: Path,
    raw_path: Path,
    *,
    side: str,
    screen_side: str,
    case_number: str = "001",
    case_id: str = "batch:001",
    phase: str = "pre",
    confirmed: bool = True,
    include_roi: bool = True,
    suffix: str = "",
) -> Candidate:
    sample_id = f"{case_number}{side}_{phase}_bone{suffix}"
    candidate = make_candidate(
        batch_dir,
        sample_id,
        raw_path,
        relative_path=f"{sample_id}.json",
    )
    candidate.case_number = case_number
    candidate.case_id = case_id
    candidate.side = side
    candidate.phase = phase
    candidate.annotation = bilateral_annotation(
        side=side,
        screen_side=screen_side,
        confirmed=confirmed,
        include_roi=include_roi,
    )
    return candidate


def candidate_decision(candidate: Candidate) -> dict[str, str]:
    return {
        "annotation_path": candidate.annotation_path.as_posix(),
        "sample_id": candidate.sample_id,
        "case_id": candidate.case_id,
        "implant_status": candidate.implant_status,
        "side": candidate.side,
        "raw_path": candidate.raw_path.as_posix(),
        "action": "candidate",
        "reason": "test",
    }


def write_dataset_rows(root: Path, folder_name: str, rows: list[dict[str, str]]) -> None:
    manifest_path = root / folder_name / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "sample_dir", "annotation_path", "raw_path", "source_raw_path"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_existing_row(
    root: Path,
    folder_name: str,
    sample_id: str,
    sample_dir: str,
    raw_bytes: bytes,
    annotation: dict,
) -> dict[str, str]:
    directory = root / folder_name / sample_dir
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "raw.jpg"
    raw_path.write_bytes(raw_bytes)
    write_json(directory / "annotation.json", annotation)
    return {
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "annotation_path": f"{sample_dir}/annotation.json",
        "raw_path": f"{sample_dir}/raw.jpg",
        "source_raw_path": "",
    }


class ImportAnnotationBatchTests(unittest.TestCase):
    def test_parse_canonical_normalizes_spaces_and_annotation_suffixes(self) -> None:
        self.assertEqual(
            parse_canonical(" 074R _pre _bone_annotation"),
            ("074R_pre_bone", "074", "R", "pre", "bone"),
        )
        self.assertEqual(
            parse_canonical("130L_post_TKA_boneannotation"),
            ("130L_post_TKA", "130", "L", "post", "TKA"),
        )

    def test_measurement_app_schema_adapter_maps_source_and_analysis(self) -> None:
        payload = {
            "source": {
                "filename": "scan.jpg",
                "sha256": "a" * 64,
                "image_width": 120,
                "image_height": 240,
            },
            "analysis": {
                "side": "l",
                "inference_roi": {
                    "x0": 60,
                    "y0": 0,
                    "x1": 120,
                    "y1": 240,
                    "coordinate_space": "source_image_pixels",
                    "selection_method": "doctor_confirmed",
                    "confirmed": True,
                },
            },
            "points": {"hip": {"x": 1, "y": 2}},
            "lines": {"upper_line": {"p1": {"x": 3, "y": 4}}},
        }

        adapted, expected_sha256 = adapt_annotation_schema(payload) or ({}, "")

        self.assertEqual(adapted["image_width"], 120)
        self.assertEqual(adapted["image_height"], 240)
        self.assertEqual(adapted["raw_filename"], "scan.jpg")
        self.assertEqual(adapted["source_raw_filename"], "scan.jpg")
        self.assertEqual(adapted["side"], "L")
        self.assertEqual(adapted["analysis"]["inference_roi"]["x0"], 60)
        self.assertTrue(adapted["analysis"]["inference_roi"]["confirmed"])
        self.assertEqual(expected_sha256, "a" * 64)

    def test_rewrite_processed_provenance_keeps_authoritative_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            batch.mkdir()
            raw_path = batch / "raw.jpg"
            raw_path.write_bytes(b"authoritative raw")
            candidate = make_candidate(batch, "001R_unknown_bone", raw_path)
            candidate.annotation_path.write_bytes(b"authoritative annotation")
            candidate.source_json_sha256 = sha256_file(candidate.annotation_path)

            processed_annotation = root / "processed.json"
            write_json(
                processed_annotation,
                {
                    **training_annotation(),
                    "processed_from": {
                        "annotation_path": "staged.json",
                        "raw_path": "staged.jpg",
                        "source_annotation_sha256": "0" * 64,
                        "source_raw_sha256": "0" * 64,
                        "original_image_width": 32,
                        "original_image_height": 32,
                        "crop": {
                            "x0": 0,
                            "y0": 0,
                            "x1": 32,
                            "y1": 32,
                            "width": 32,
                            "height": 32,
                            "method": "already_single_leg",
                        },
                    },
                },
            )
            manifest = root / "processed_manifest.csv"
            fields = [
                "sample_id",
                "annotation_path",
                "source_dataset",
                "source_annotation_path",
                "source_raw_path",
                "source_annotation_sha256",
                "source_raw_sha256",
                "raw_match_count",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": candidate.sample_id,
                        "annotation_path": processed_annotation.as_posix(),
                        "source_dataset": "batch",
                        "source_annotation_path": "staged.json",
                        "source_raw_path": "staged.jpg",
                        "source_annotation_sha256": "0" * 64,
                        "source_raw_sha256": "0" * 64,
                        "raw_match_count": "1",
                    }
                )

            rows = rewrite_processed_provenance(manifest, {candidate.sample_id: candidate})

            rewritten = json.loads(processed_annotation.read_text(encoding="utf-8"))
            provenance = rewritten["processed_from"]
            self.assertEqual(provenance["annotation_path"], candidate.annotation_path.as_posix())
            self.assertEqual(provenance["raw_path"], raw_path.as_posix())
            self.assertEqual(provenance["source_annotation_sha256"], candidate.source_json_sha256)
            self.assertEqual(provenance["source_raw_sha256"], candidate.raw_sha256)
            self.assertEqual(rows[0]["source_annotation_sha256"], candidate.source_json_sha256)
            self.assertEqual(rows[0]["source_raw_sha256"], candidate.raw_sha256)

    def test_collect_candidates_accepts_arbitrary_safe_sample_id_and_validates_raw_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            source_dir = batch / "unknown study"
            source_dir.mkdir(parents=True)
            raw_path = source_dir / "scan.jpg"
            self.assertTrue(cv2.imwrite(str(raw_path), np.full((32, 32, 3), 127, dtype=np.uint8)))
            annotation_path = source_dir / "scan_measurement.json"
            write_json(annotation_path, measurement_annotation(raw_path, side="L"))
            relative_path = annotation_path.relative_to(batch).as_posix()
            overrides = {
                relative_path: {
                    "sample_id": "5143229L_unknown_bone_timepoint2",
                    "case_id": "batch:5143229",
                    "side": "L",
                    "implant_status": "bone",
                    "study_phase": "unknown",
                }
            }

            with mock.patch(
                "import_annotation_batch.measure_from_named_points",
                return_value=({"mldfa_angle": 90.0, "mpta_angle": 89.0}, {}),
            ):
                candidates, decisions = collect_candidates(batch, root / "dataset", overrides)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].sample_id, "5143229L_unknown_bone_timepoint2")
            self.assertEqual(candidates[0].case_number, "5143229")
            self.assertEqual(candidates[0].phase, "unknown")
            self.assertEqual(decisions[0]["action"], "candidate")

            payload = measurement_annotation(raw_path, side="L", sha256="0" * 64)
            write_json(annotation_path, payload)
            with mock.patch(
                "import_annotation_batch.measure_from_named_points",
                return_value=({"mldfa_angle": 90.0, "mpta_angle": 89.0}, {}),
            ):
                bad_candidates, bad_decisions = collect_candidates(batch, root / "dataset", overrides)
            self.assertEqual(bad_candidates, [])
            self.assertEqual(bad_decisions[0]["action"], "quarantine")
            self.assertIn("raw_sha256_mismatch", bad_decisions[0]["reason"])

    def test_load_overrides_rejects_unsafe_identifiers_and_unscoped_raw_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsafe_id = root / "unsafe_id.json"
            write_json(unsafe_id, {"overrides": {"a.json": {"sample_id": "../escape"}}})
            with self.assertRaisesRegex(ValueError, "Unsafe sample_id"):
                load_overrides(unsafe_id)

            unsafe_dir = root / "unsafe_dir.json"
            write_json(
                unsafe_dir,
                {"overrides": {"a.json": {"replace_existing_sample_dir": "../samples/a"}}},
            )
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                load_overrides(unsafe_dir)

            unscoped = root / "unscoped.json"
            write_json(unscoped, {"overrides": {"a.json": {"allow_raw_replacement": True}}})
            with self.assertRaisesRegex(ValueError, "requires an explicit replacement target"):
                load_overrides(unscoped)

    def test_batch_raw_collision_requires_an_explicit_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            raw_path = batch / "raw.jpg"
            raw_path.write_bytes(b"same raw")
            first = make_candidate(batch, "001R_unknown_bone_tp1", raw_path)
            second = make_candidate(batch, "002R_unknown_bone_tp1", raw_path)
            decisions = [candidate_decision(first), candidate_decision(second)]

            accepted = deduplicate_candidates([first, second], decisions, batch)

            self.assertEqual(accepted, [])
            self.assertTrue(all(row["action"] == "quarantine" for row in decisions))
            self.assertTrue(
                all("batch_raw_collision_requires_explicit_replacement" in row["reason"] for row in decisions)
            )

    def test_same_raw_confirmed_bilateral_lr_pair_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            raw_path = batch / "bilateral.jpg"
            raw_path.write_bytes(b"same bilateral raw")
            left = make_bilateral_candidate(
                batch,
                raw_path,
                side="L",
                screen_side="right",
            )
            right = make_bilateral_candidate(
                batch,
                raw_path,
                side="R",
                screen_side="left",
            )
            decisions = [candidate_decision(left), candidate_decision(right)]

            accepted = deduplicate_candidates([left, right], decisions, batch)

            self.assertEqual([candidate.sample_id for candidate in accepted], sorted([left.sample_id, right.sample_id]))
            self.assertTrue(all(row["action"] == "candidate" for row in decisions))

    def test_same_raw_is_quarantined_unless_it_is_exactly_a_safe_bilateral_lr_pair(self) -> None:
        scenarios: dict[str, list[dict[str, object]]] = {
            "same_anatomical_side": [
                {"side": "L", "screen_side": "left", "suffix": "_a"},
                {"side": "L", "screen_side": "right", "suffix": "_b"},
            ],
            "same_screen_side": [
                {"side": "L", "screen_side": "left"},
                {"side": "R", "screen_side": "left"},
            ],
            "unconfirmed": [
                {"side": "L", "screen_side": "right"},
                {"side": "R", "screen_side": "left", "confirmed": False},
            ],
            "different_case": [
                {"side": "L", "screen_side": "right"},
                {
                    "side": "R",
                    "screen_side": "left",
                    "case_number": "002",
                    "case_id": "batch:002",
                },
            ],
            "different_timepoint": [
                {"side": "L", "screen_side": "right", "phase": "pre"},
                {"side": "R", "screen_side": "left", "phase": "post"},
            ],
            "missing_roi": [
                {"side": "L", "screen_side": "right"},
                {"side": "R", "screen_side": "left", "include_roi": False},
            ],
            "three_annotations": [
                {"side": "L", "screen_side": "right", "suffix": "_a"},
                {"side": "R", "screen_side": "left"},
                {"side": "L", "screen_side": "right", "suffix": "_b"},
            ],
        }
        for name, definitions in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                batch = Path(temporary)
                raw_path = batch / "bilateral.jpg"
                raw_path.write_bytes(b"same bilateral raw")
                candidates = [
                    make_bilateral_candidate(batch, raw_path, **definition)
                    for definition in definitions
                ]
                decisions = [candidate_decision(candidate) for candidate in candidates]

                accepted = deduplicate_candidates(candidates, decisions, batch)

                self.assertEqual(accepted, [])
                self.assertTrue(all(row["action"] == "quarantine" for row in decisions))
                self.assertTrue(
                    all(
                        "batch_raw_collision_requires_explicit_replacement" in row["reason"]
                        for row in decisions
                    )
                )

    def test_filter_uses_sample_dir_to_disambiguate_duplicate_ids_as_one_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            batch = root / "batch"
            batch.mkdir()
            bone_folder = DATASET_FOLDERS["non_tka"]
            first_existing = add_existing_row(
                dataset,
                bone_folder,
                "018R_unknown_bone",
                "samples/018R_unknown_bone",
                b"raw for 017 repair",
                training_annotation(offset=0.0),
            )
            second_existing = add_existing_row(
                dataset,
                bone_folder,
                "018R_unknown_bone",
                "samples/018R_unknown_bone_2",
                b"raw for 018 correction",
                training_annotation(offset=0.5),
            )
            write_dataset_rows(dataset, bone_folder, [first_existing, second_existing])

            raw_017 = batch / "017.jpg"
            raw_017.write_bytes(b"raw for 017 repair")
            raw_018 = batch / "018.jpg"
            raw_018.write_bytes(b"raw for 018 correction")
            candidate_017 = make_candidate(
                batch,
                "017R_unknown_bone",
                raw_017,
                replace_existing_sample_id="018R_unknown_bone",
                replace_existing_sample_dir="samples/018R_unknown_bone",
            )
            candidate_018 = make_candidate(
                batch,
                "018R_unknown_bone",
                raw_018,
                replace_existing_sample_id="018R_unknown_bone",
                replace_existing_sample_dir="samples/018R_unknown_bone_2",
            )
            rows: list[dict[str, str]] = []
            for candidate, offset in ((candidate_017, 2.0), (candidate_018, 3.0)):
                annotation_path = batch / f"{candidate.sample_id}_processed.json"
                write_json(annotation_path, training_annotation(offset=offset))
                rows.append(
                    {
                        "sample_id": candidate.sample_id,
                        "annotation_path": annotation_path.as_posix(),
                        "raw_path": candidate.raw_path.as_posix(),
                    }
                )
            by_sample = {
                candidate_017.sample_id: candidate_017,
                candidate_018.sample_id: candidate_018,
            }
            decisions = [candidate_decision(candidate_017), candidate_decision(candidate_018)]

            accepted, plans = filter_existing_collisions(
                rows,
                by_sample,
                dataset,
                decisions,
                batch,
            )

            self.assertEqual({row["sample_id"] for row in accepted}, set(by_sample))
            self.assertEqual(
                {(plan.old_sample_dir, plan.new_sample_id) for plan in plans},
                {
                    ("samples/018R_unknown_bone", "017R_unknown_bone"),
                    ("samples/018R_unknown_bone_2", "018R_unknown_bone"),
                },
            )
            self.assertTrue(all(row["action"] == "ready_to_import" for row in decisions))

    def test_filter_requires_allow_raw_replacement_for_changed_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            batch = root / "batch"
            batch.mkdir()
            bone_folder = DATASET_FOLDERS["non_tka"]
            existing = add_existing_row(
                dataset,
                bone_folder,
                "001R_unknown_bone",
                "samples/001R_unknown_bone",
                b"old raw",
                training_annotation(),
            )
            write_dataset_rows(dataset, bone_folder, [existing])
            new_raw = batch / "new.jpg"
            new_raw.write_bytes(b"new raw")
            candidate = make_candidate(
                batch,
                "001R_unknown_bone",
                new_raw,
                replace_existing_sample_dir="samples/001R_unknown_bone",
            )
            annotation_path = batch / "processed.json"
            write_json(annotation_path, training_annotation(offset=2.0))
            rows = [
                {
                    "sample_id": candidate.sample_id,
                    "annotation_path": annotation_path.as_posix(),
                    "raw_path": new_raw.as_posix(),
                }
            ]

            decisions = [candidate_decision(candidate)]
            accepted, plans = filter_existing_collisions(
                rows, {candidate.sample_id: candidate}, dataset, decisions, batch
            )
            self.assertEqual(accepted, [])
            self.assertEqual(plans, [])
            self.assertIn("requires_allow_raw_replacement", decisions[0]["reason"])

            candidate.allow_raw_replacement = True
            decisions = [candidate_decision(candidate)]
            accepted, plans = filter_existing_collisions(
                rows, {candidate.sample_id: candidate}, dataset, decisions, batch
            )
            self.assertEqual(accepted, rows)
            self.assertEqual(len(plans), 1)

    def test_existing_raw_collision_without_replacement_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            batch = root / "batch"
            batch.mkdir()
            bone_folder = DATASET_FOLDERS["non_tka"]
            existing = add_existing_row(
                dataset,
                bone_folder,
                "001R_unknown_bone",
                "samples/001R_unknown_bone",
                b"shared raw",
                training_annotation(),
            )
            write_dataset_rows(dataset, bone_folder, [existing])
            raw_path = batch / "shared.jpg"
            raw_path.write_bytes(b"shared raw")
            candidate = make_candidate(batch, "002R_unknown_bone", raw_path)
            annotation_path = batch / "processed.json"
            write_json(annotation_path, training_annotation(offset=3.0))
            rows = [
                {
                    "sample_id": candidate.sample_id,
                    "annotation_path": annotation_path.as_posix(),
                    "raw_path": raw_path.as_posix(),
                }
            ]
            decisions = [candidate_decision(candidate)]

            accepted, plans = filter_existing_collisions(
                rows, {candidate.sample_id: candidate}, dataset, decisions, batch
            )

            self.assertEqual(accepted, [])
            self.assertEqual(plans, [])
            self.assertIn("existing_raw_collision_requires_explicit_replacement", decisions[0]["reason"])

    def test_merge_stage_backs_up_and_replaces_an_existing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            stage = root / "stage"
            backup = root / "report" / "replaced_existing"
            for folder_name in DATASET_FOLDERS.values():
                write_manifest(target / folder_name / "manifest.csv", [])
                write_manifest(stage / folder_name / "manifest.csv", [])

            tka_folder = DATASET_FOLDERS["TKA"]
            write_manifest(
                target / tka_folder / "manifest.csv",
                [{"sample_id": "058L_pre_TKA", "sample_dir": "samples/058L_pre_TKA"}],
            )
            make_sample(target / tka_folder, "samples/058L_pre_TKA", "old")
            write_manifest(
                stage / tka_folder / "manifest.csv",
                [{"sample_id": "058R_pre_TKA", "sample_dir": "samples/058R_pre_TKA"}],
            )
            make_sample(stage / tka_folder, "samples/058R_pre_TKA", "new")

            with mock.patch("import_annotation_batch.validate_dataset_root", return_value=[]):
                counts = merge_stage(
                    stage,
                    target,
                    replacements={"058L_pre_TKA": "058R_pre_TKA"},
                    replacement_backup_dir=backup,
                )

            self.assertEqual(counts[tka_folder], 1)
            self.assertFalse((target / tka_folder / "samples/058L_pre_TKA").exists())
            self.assertTrue((target / tka_folder / "samples/058R_pre_TKA").exists())
            self.assertTrue((backup / tka_folder / "samples/058L_pre_TKA").exists())
            with (target / tka_folder / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["058R_pre_TKA"])

    def test_merge_stage_restores_replacement_when_a_destination_collides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            stage = root / "stage"
            backup = root / "report" / "replaced_existing"
            for folder_name in DATASET_FOLDERS.values():
                write_manifest(target / folder_name / "manifest.csv", [])
                write_manifest(stage / folder_name / "manifest.csv", [])

            tka_folder = DATASET_FOLDERS["TKA"]
            old_row = {"sample_id": "058L_pre_TKA", "sample_dir": "samples/058L_pre_TKA"}
            new_row = {"sample_id": "058R_pre_TKA", "sample_dir": "samples/058R_pre_TKA"}
            write_manifest(target / tka_folder / "manifest.csv", [old_row])
            make_sample(target / tka_folder, old_row["sample_dir"], "old")
            make_sample(target / tka_folder, new_row["sample_dir"], "collision")
            write_manifest(stage / tka_folder / "manifest.csv", [new_row])
            make_sample(stage / tka_folder, new_row["sample_dir"], "new")

            with self.assertRaises(FileExistsError):
                merge_stage(
                    stage,
                    target,
                    replacements={"058L_pre_TKA": "058R_pre_TKA"},
                    replacement_backup_dir=backup,
                )

            self.assertTrue((target / tka_folder / old_row["sample_dir"]).exists())
            self.assertFalse((backup / tka_folder / old_row["sample_dir"]).exists())
            with (target / tka_folder / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [old_row])

    def test_merge_stage_replaces_across_bone_and_tka_folders_by_exact_sample_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            stage = root / "stage"
            backup = root / "report" / "replaced_existing"
            for folder_name in DATASET_FOLDERS.values():
                write_manifest(target / folder_name / "manifest.csv", [])
                write_manifest(stage / folder_name / "manifest.csv", [])

            bone_folder = DATASET_FOLDERS["non_tka"]
            tka_folder = DATASET_FOLDERS["TKA"]
            old_row = {"sample_id": "018R_unknown_bone", "sample_dir": "samples/018R_unknown_bone_2"}
            new_row = {"sample_id": "018R_unknown_TKA", "sample_dir": "samples/018R_unknown_TKA"}
            write_manifest(target / bone_folder / "manifest.csv", [old_row])
            make_sample(target / bone_folder, old_row["sample_dir"], "old bone")
            write_manifest(stage / tka_folder / "manifest.csv", [new_row])
            make_sample(stage / tka_folder, new_row["sample_dir"], "new TKA")
            plan = ReplacementPlan(
                old_folder_name=bone_folder,
                old_sample_id=old_row["sample_id"],
                old_sample_dir=old_row["sample_dir"],
                new_sample_id=new_row["sample_id"],
            )

            with mock.patch("import_annotation_batch.validate_dataset_root", return_value=[]):
                merge_stage(
                    stage,
                    target,
                    replacements=[plan],
                    replacement_backup_dir=backup,
                )

            self.assertFalse((target / bone_folder / old_row["sample_dir"]).exists())
            self.assertTrue((target / tka_folder / new_row["sample_dir"]).exists())
            self.assertTrue((backup / bone_folder / old_row["sample_dir"]).exists())
            replacement_records = json.loads((backup / "replacements.json").read_text(encoding="utf-8"))
            self.assertEqual(replacement_records[0]["old_sample_dir_relative"], old_row["sample_dir"])

    def test_merge_stage_rolls_back_cross_folder_replacement_after_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            stage = root / "stage"
            backup = root / "report" / "replaced_existing"
            for folder_name in DATASET_FOLDERS.values():
                write_manifest(target / folder_name / "manifest.csv", [])
                write_manifest(stage / folder_name / "manifest.csv", [])

            bone_folder = DATASET_FOLDERS["non_tka"]
            tka_folder = DATASET_FOLDERS["TKA"]
            old_row = {"sample_id": "018R_unknown_bone", "sample_dir": "samples/018R_unknown_bone_2"}
            new_row = {"sample_id": "018R_unknown_TKA", "sample_dir": "samples/018R_unknown_TKA"}
            write_manifest(target / bone_folder / "manifest.csv", [old_row])
            make_sample(target / bone_folder, old_row["sample_dir"], "old bone")
            write_manifest(stage / tka_folder / "manifest.csv", [new_row])
            make_sample(stage / tka_folder, new_row["sample_dir"], "new TKA")
            plan = ReplacementPlan(
                old_folder_name=bone_folder,
                old_sample_id=old_row["sample_id"],
                old_sample_dir=old_row["sample_dir"],
                new_sample_id=new_row["sample_id"],
            )

            with mock.patch(
                "import_annotation_batch.validate_dataset_root",
                return_value=["duplicate_raw_sha256:test"],
            ):
                with self.assertRaisesRegex(ValueError, "Merged dataset validation failed"):
                    merge_stage(
                        stage,
                        target,
                        replacements=[plan],
                        replacement_backup_dir=backup,
                    )

            self.assertTrue((target / bone_folder / old_row["sample_dir"]).exists())
            self.assertFalse((target / tka_folder / new_row["sample_dir"]).exists())
            self.assertFalse((backup / bone_folder / old_row["sample_dir"]).exists())
            self.assertFalse((backup / "replacements.json").exists())
            with (target / bone_folder / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [old_row])

    def test_validate_dataset_root_reports_global_sample_id_and_raw_sha_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_raw = b"identical raw bytes"
            for index, folder_name in enumerate(DATASET_FOLDERS.values()):
                sample_dir = f"samples/duplicate_{index}"
                directory = root / folder_name / sample_dir
                directory.mkdir(parents=True)
                for filename in ("annotation.json", "raw.jpg", "point.jpg", "line.jpg", "combined.jpg"):
                    (directory / filename).write_bytes(
                        duplicate_raw if filename == "raw.jpg" else b"nonempty"
                    )
                manifest_path = root / folder_name / "manifest.csv"
                with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["sample_id", "sample_dir", "raw_path"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sample_id": "duplicate_id",
                            "sample_dir": sample_dir,
                            "raw_path": f"{sample_dir}/raw.jpg",
                        }
                    )

            with mock.patch("import_annotation_batch.validate_manifest", return_value=([], {})):
                errors = validate_dataset_root(root)

            self.assertTrue(any(error.startswith("duplicate_sample_id:duplicate_id") for error in errors))
            self.assertTrue(any(error.startswith("duplicate_raw_sha256:") for error in errors))


if __name__ == "__main__":
    unittest.main()

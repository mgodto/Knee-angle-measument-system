from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from knee_xray.training.prepare_retraining_manifests import (
    ManifestAuditError,
    prepare_retraining_manifests,
)


FIELDS = [
    "sample_id",
    "case_id",
    "source_dataset",
    "dataset_group",
    "implant_status",
    "study_phase",
    "side",
    "annotation_path",
    "raw_path",
    "source_annotation_path",
    "source_raw_path",
    "sample_dir",
    "point_path",
    "line_path",
    "combined_path",
]


def _annotation(side: str, note: str) -> dict[str, object]:
    return {
        "image_width": 100,
        "image_height": 200,
        "side": side,
        "points": {
            "hip": {"x": 50, "y": 10},
            "upper_left": {"x": 20, "y": 80},
            "upper_center": {"x": 50, "y": 80},
            "upper_right": {"x": 80, "y": 80},
            "lower_left": {"x": 20, "y": 100},
            "lower_center": {"x": 50, "y": 100},
            "lower_right": {"x": 80, "y": 100},
            "ankle": {"x": 50, "y": 190},
        },
        "lines": {
            "upper_line": {"p1": {"x": 20, "y": 80}, "p2": {"x": 80, "y": 80}},
            "lower_line": {"p1": {"x": 20, "y": 100}, "p2": {"x": 80, "y": 100}},
        },
        "provenance_note": note,
    }


def _make_row(
    dataset_dir: Path,
    *,
    directory: str,
    sample_id: str,
    case_id: str,
    source_dataset: str,
    implant_status: str,
    side: str,
    raw_bytes: bytes,
    annotation_note: str,
    source_raw_basename: str | None = None,
) -> dict[str, str]:
    sample_dir = dataset_dir / "samples" / directory
    sample_dir.mkdir(parents=True)
    (sample_dir / "raw.jpg").write_bytes(raw_bytes)
    (sample_dir / "annotation.json").write_text(
        json.dumps(_annotation(side, annotation_note)), encoding="utf-8"
    )
    for filename in ("point.jpg", "line.jpg", "combined.jpg"):
        (sample_dir / filename).write_bytes(b"preview")
    relative_dir = f"samples/{directory}"
    return {
        "sample_id": sample_id,
        "case_id": case_id,
        "source_dataset": source_dataset,
        "dataset_group": case_id.split(":", 1)[0],
        "implant_status": implant_status,
        "study_phase": "post" if implant_status == "TKA" else "pre",
        "side": side,
        "annotation_path": f"{relative_dir}/annotation.json",
        "raw_path": f"{relative_dir}/raw.jpg",
        "source_annotation_path": f"images/original/{directory}_annotation.json",
        "source_raw_path": f"images/original/{source_raw_basename or f'{directory}.jpg'}",
        "sample_dir": relative_dir,
        "point_path": f"{relative_dir}/point.jpg",
        "line_path": f"{relative_dir}/line.jpg",
        "combined_path": f"{relative_dir}/combined.jpg",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PrepareRetrainingManifestsTests(unittest.TestCase):
    def test_removes_misfiled_row_and_writes_repo_relative_clean_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            non_tka_dir = root / "images" / "organized" / "non_tka"
            tka_dir = root / "images" / "organized" / "tka"
            duplicate_raw = b"same-xray"
            wrong = _make_row(
                non_tka_dir,
                directory="wrong-002R",
                sample_id="002R_pre_bone",
                case_id="new:001",
                source_dataset="new",
                implant_status="bone",
                side="R",
                raw_bytes=duplicate_raw,
                annotation_note="wrong source metadata",
            )
            bone = _make_row(
                non_tka_dir,
                directory="valid-002R",
                sample_id="002R_pre_bone",
                case_id="new:002",
                source_dataset="new",
                implant_status="bone",
                side="R",
                raw_bytes=duplicate_raw,
                annotation_note="different source metadata",
            )
            legacy = _make_row(
                non_tka_dir,
                directory="003L",
                sample_id="003L",
                case_id="legacy:003",
                source_dataset="legacy",
                implant_status="unknown",
                side="L",
                raw_bytes=b"legacy-xray",
                annotation_note="legacy",
            )
            tka = _make_row(
                tka_dir,
                directory="004R-post",
                sample_id="004R_post_TKA",
                case_id="new:004",
                source_dataset="new",
                implant_status="TKA",
                side="R",
                raw_bytes=b"tka-xray",
                annotation_note="tka",
            )
            non_tka_manifest = non_tka_dir / "manifest.csv"
            tka_manifest = tka_dir / "manifest.csv"
            _write_manifest(non_tka_manifest, [wrong, bone, legacy])
            _write_manifest(tka_manifest, [tka])
            output_dir = root / "outputs" / "clean"

            audit = prepare_retraining_manifests(
                repo_root=root,
                non_tka_manifest=non_tka_manifest,
                tka_manifest=tka_manifest,
                output_dir=output_dir,
            )

            self.assertEqual(audit["counts"]["removed_case_number_mismatches"], 1)
            self.assertEqual(audit["counts"]["output_all_clean"], 3)
            self.assertEqual(audit["removed_rows"][0]["case_id"], "new:001")
            self.assertTrue(audit["validation_passed"])
            self.assertNotIn("group_by_patient", audit["parameters"])
            self.assertEqual(len(_read_manifest(output_dir / "bone_confirmed.csv")), 1)
            self.assertEqual(len(_read_manifest(output_dir / "legacy_unknown.csv")), 1)
            self.assertEqual(len(_read_manifest(output_dir / "tka.csv")), 1)
            bone_tka_rows = _read_manifest(output_dir / "bone_tka.csv")
            self.assertEqual(len(bone_tka_rows), 2)
            self.assertEqual(
                {row["implant_status"] for row in bone_tka_rows},
                {"bone", "TKA"},
            )
            all_rows = _read_manifest(output_dir / "all_clean.csv")
            self.assertEqual(len(all_rows), 3)
            self.assertNotIn("source_case_id", all_rows[0])
            for row in all_rows:
                self.assertFalse(Path(row["raw_path"]).is_absolute())
                self.assertTrue((root / row["raw_path"]).is_file())
                self.assertTrue((root / row["annotation_path"]).is_file())
                self.assertTrue(row["raw_path"].startswith("images/organized/"))
            disk_audit = json.loads((output_dir / "audit_summary.json").read_text(encoding="utf-8"))
            for dataset in disk_audit["datasets"].values():
                self.assertEqual(dataset["exact_duplicate_groups"], 0)
                self.assertEqual(dataset["sample_hash_cross_fold_groups"], [])
                self.assertEqual(dataset["case_hash_cross_fold_groups"], [])

    def test_exact_duplicate_content_aborts_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            non_tka_dir = root / "images" / "organized" / "non_tka"
            tka_dir = root / "images" / "organized" / "tka"
            first = _make_row(
                non_tka_dir,
                directory="001L",
                sample_id="001L_pre_bone",
                case_id="new:001",
                source_dataset="new",
                implant_status="bone",
                side="L",
                raw_bytes=b"duplicate-xray",
                annotation_note="source one",
            )
            second = _make_row(
                non_tka_dir,
                directory="002L",
                sample_id="002L_pre_bone",
                case_id="new:002",
                source_dataset="new",
                implant_status="bone",
                side="L",
                raw_bytes=b"duplicate-xray",
                annotation_note="source two",
            )
            tka = _make_row(
                tka_dir,
                directory="003R",
                sample_id="003R_post_TKA",
                case_id="new:003",
                source_dataset="new",
                implant_status="TKA",
                side="R",
                raw_bytes=b"unique-tka",
                annotation_note="tka",
            )
            non_tka_manifest = non_tka_dir / "manifest.csv"
            tka_manifest = tka_dir / "manifest.csv"
            _write_manifest(non_tka_manifest, [first, second])
            _write_manifest(tka_manifest, [tka])
            output_dir = root / "outputs" / "clean"

            with self.assertRaisesRegex(ManifestAuditError, "Exact duplicate content"):
                prepare_retraining_manifests(
                    repo_root=root,
                    non_tka_manifest=non_tka_manifest,
                    tka_manifest=tka_manifest,
                    output_dir=output_dir,
                )

            self.assertFalse(output_dir.exists())

    def test_patient_grouping_uses_transitive_case_and_token_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            non_tka_dir = root / "images" / "organized" / "non_tka"
            tka_dir = root / "images" / "organized" / "tka"
            first_case_token_a = _make_row(
                non_tka_dir,
                directory="001L",
                sample_id="001L_pre_bone",
                case_id="new:001",
                source_dataset="new",
                implant_status="bone",
                side="L",
                raw_bytes=b"patient-a-left",
                annotation_note="patient a left",
                source_raw_basename="PATIENT_A_CR_20260101_001_0001.jpg",
            )
            first_case_token_b = _make_row(
                tka_dir,
                directory="001R",
                sample_id="001R_post_TKA",
                case_id="new:001",
                source_dataset="new",
                implant_status="TKA",
                side="R",
                raw_bytes=b"patient-b-right",
                annotation_note="patient b right",
                source_raw_basename="PATIENT_B_CR_20260202_001_0001.jpg",
            )
            second_case_token_b = _make_row(
                non_tka_dir,
                directory="002L",
                sample_id="002L_pre_bone",
                case_id="new:002",
                source_dataset="new",
                implant_status="bone",
                side="L",
                raw_bytes=b"patient-b-left",
                annotation_note="patient b left",
                source_raw_basename="PATIENT_B_CR_20260303_001_0001.jpg",
            )
            old_first = _make_row(
                non_tka_dir,
                directory="003L",
                sample_id="003L_pre_bone",
                case_id="old:003",
                source_dataset="20260803",
                implant_status="bone",
                side="L",
                raw_bytes=b"old-patient-first",
                annotation_note="old patient first",
                source_raw_basename="163956L_bone_001_0001.jpg",
            )
            old_second = _make_row(
                non_tka_dir,
                directory="004L",
                sample_id="004L_pre_bone",
                case_id="old:004",
                source_dataset="20260803",
                implant_status="bone",
                side="L",
                raw_bytes=b"old-patient-second",
                annotation_note="old patient second",
                source_raw_basename="163956R_bone_001_0005.jpg",
            )
            misfiled_bridge = _make_row(
                non_tka_dir,
                directory="misfiled-099R",
                sample_id="099R_pre_bone",
                case_id="old:003",
                source_dataset="20260803",
                implant_status="bone",
                side="R",
                raw_bytes=b"misfiled-bridge",
                annotation_note="must be removed before grouping",
                source_raw_basename="PATIENT_B_CR_20260404_001_0001.jpg",
            )
            non_tka_rows = [
                first_case_token_a,
                second_case_token_b,
                old_first,
                old_second,
                misfiled_bridge,
            ]
            tka_rows = [first_case_token_b]
            non_tka_manifest = non_tka_dir / "manifest.csv"
            tka_manifest = tka_dir / "manifest.csv"
            _write_manifest(non_tka_manifest, non_tka_rows)
            _write_manifest(tka_manifest, tka_rows)

            first_output = root / "outputs" / "first"
            audit = prepare_retraining_manifests(
                repo_root=root,
                non_tka_manifest=non_tka_manifest,
                tka_manifest=tka_manifest,
                output_dir=first_output,
                group_by_patient=True,
            )

            rows = _read_manifest(first_output / "all_clean.csv")
            groups_by_source: dict[str, set[str]] = {}
            for row in rows:
                groups_by_source.setdefault(row["source_case_id"], set()).add(row["case_id"])
                self.assertRegex(row["case_id"], r"^patient_group:[0-9a-f]{64}$")
            self.assertEqual(groups_by_source["new:001"], groups_by_source["new:002"])
            self.assertEqual(groups_by_source["old:003"], groups_by_source["old:004"])
            self.assertNotEqual(groups_by_source["new:001"], groups_by_source["old:003"])
            self.assertEqual(audit["datasets"]["all_clean"]["cases"], 2)
            self.assertEqual(audit["counts"]["removed_case_number_mismatches"], 1)
            self.assertEqual(audit["removed_rows"][0]["sample_id"], "099R_pre_bone")
            self.assertTrue(audit["parameters"]["group_by_patient"])

            _write_manifest(non_tka_manifest, list(reversed(non_tka_rows)))
            _write_manifest(tka_manifest, list(reversed(tka_rows)))
            second_output = root / "outputs" / "second"
            prepare_retraining_manifests(
                repo_root=root,
                non_tka_manifest=non_tka_manifest,
                tka_manifest=tka_manifest,
                output_dir=second_output,
                group_by_patient=True,
            )
            reordered = _read_manifest(second_output / "all_clean.csv")
            first_mapping = {
                (row["sample_id"], row["source_case_id"]): row["case_id"]
                for row in rows
            }
            second_mapping = {
                (row["sample_id"], row["source_case_id"]): row["case_id"]
                for row in reordered
            }
            self.assertEqual(first_mapping, second_mapping)


if __name__ == "__main__":
    unittest.main()

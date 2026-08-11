from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "training" / "run_retraining_single_leg_v3_20260811.sh"
V2_RUNNER = REPO_ROOT / "scripts" / "training" / "run_retraining_single_leg_v2_20260803.sh"
MANIFEST_FIELDS = (
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
)


def _patient_group(index: int) -> str:
    digest = hashlib.sha256(f"fixture-patient-{index}".encode("ascii")).hexdigest()
    return f"patient_group:{digest}"


def _annotation(side: str) -> dict[str, object]:
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
    points = {
        name: {"x": float(index + 1), "y": float(index + 2)}
        for index, name in enumerate(point_names)
    }
    lines = {
        "upper_line": {"p1": {"x": 1.0, "y": 5.0}, "p2": {"x": 9.0, "y": 5.0}},
        "lower_line": {"p1": {"x": 1.0, "y": 15.0}, "p2": {"x": 9.0, "y": 15.0}},
    }
    return {
        "image_width": 10,
        "image_height": 20,
        "side": side,
        "points": points,
        "lines": lines,
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_rows(root: Path, cohort: str, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    data_root = root / "files"
    data_root.mkdir(parents=True, exist_ok=True)
    evidence_path = data_root / "fixture_crop_contract_ledger.csv"
    if not evidence_path.exists():
        evidence_path.write_text("fixture crop review evidence\n", encoding="utf-8")
    evidence_sha256 = _sha256_file(evidence_path)
    implant_status = "bone" if cohort == "bone" else "TKA"
    for index in range(count):
        sample_id = f"{cohort}_{index:04d}"
        side = "L" if index % 2 == 0 else "R"
        raw_path = data_root / f"{sample_id}.jpg"
        annotation_path = data_root / f"{sample_id}.json"
        immediate_raw_path = data_root / f"{sample_id}.source.jpg"
        immediate_annotation_path = data_root / f"{sample_id}.source.json"
        raw_path.write_bytes(f"fixture-raw:{sample_id}".encode("ascii"))
        immediate_raw_path.write_bytes(
            f"fixture-source-raw:{sample_id}".encode("ascii")
        )
        immediate_annotation_path.write_text(
            json.dumps(_annotation(side), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        immediate = {
            "raw_path": str(immediate_raw_path),
            "annotation_path": str(immediate_annotation_path),
            "raw_sha256": _sha256_file(immediate_raw_path),
            "annotation_sha256": _sha256_file(immediate_annotation_path),
            "width": 10,
            "height": 20,
        }
        crop = {
            "x0": 0,
            "y0": 0,
            "x1": 10,
            "y1": 20,
            "width": 10,
            "height": 20,
            "method": "identity_confirmed_direct_pass",
            "coordinate_space": "current_processed_image_pixels",
            "selection_method": "reviewed_current_representation_identity",
        }
        padding = {
            "method": "none",
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0,
            "fill_value": 0,
            "rescaled": False,
        }
        transform = {
            "schema_version": 1,
            "kind": "reviewed_current_processed_identity",
            "immediate_source": immediate,
            "crop": crop,
            "padding": padding,
            "output_canvas": {"width": 10, "height": 20},
        }
        transform_sha256 = _canonical_json_sha256(transform)
        annotation = _annotation(side)
        annotation.update(
            {
                "raw_path": str(raw_path),
                "crop_confirmed": True,
                "processed_from": {
                    "annotation_path": immediate["annotation_path"],
                    "raw_path": immediate["raw_path"],
                    "original_image_width": 10,
                    "original_image_height": 20,
                    "source_annotation_sha256": immediate["annotation_sha256"],
                    "source_raw_sha256": immediate["raw_sha256"],
                    "crop": {**crop, "confirmed": True},
                    "padding": padding,
                },
                "training_crop_contract": {
                    "schema_version": 1,
                    "evidence_ledger_path": str(evidence_path),
                    "evidence_sha256": evidence_sha256,
                    "crop_review_status": "historical_original_pass",
                    "crop_review_reason": "FIXTURE_REVIEW_PASS",
                    "decision_source_path": "",
                    "decision_source_sha256": "",
                    "selected_representation": "reviewed_current_processed_identity",
                    "crop_transform_sha256": transform_sha256,
                    "immediate_source": immediate,
                    "transform": {
                        "kind": transform["kind"],
                        "crop": {**crop, "confirmed": True},
                        "padding": padding,
                        "output_canvas": transform["output_canvas"],
                    },
                    "crop_confirmed": True,
                    "validation": {
                        "coordinate_pair_count": 12,
                        "all_coordinates_in_bounds": True,
                        "angles_preserved_from_frozen_baseline": True,
                        "padding_aware_geometry": True,
                        "strip_rescaled": False,
                    },
                },
            }
        )
        annotation_path.write_text(
            json.dumps(annotation, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "case_id": _patient_group(index % 50),
                "source_case_id": f"fixture:{index % 50:03d}",
                "source_dataset": "fixture",
                "implant_status": implant_status,
                "side": side,
                "annotation_path": str(annotation_path),
                "raw_path": str(raw_path),
                "image_width": "10",
                "image_height": "20",
                "mldfa": "88.0",
                "mpta": "87.0",
                "crop_x0": "0",
                "crop_y0": "0",
                "crop_x1": "10",
                "crop_y1": "20",
                "crop_width": "10",
                "crop_height": "20",
                "crop_method": crop["method"],
                "crop_provenance_source": "training_crop_contract",
                "crop_selection_method": crop["selection_method"],
                "crop_confirmed": "true",
                "inference_roi_status": "crop_contract_reviewed",
                "is_cropped": "false",
                "crop_review_status": "historical_original_pass",
                "crop_review_reason": "FIXTURE_REVIEW_PASS",
                "evidence_sha256": evidence_sha256,
                "crop_coordinate_space": crop["coordinate_space"],
                "crop_output_width": "10",
                "crop_output_height": "20",
                "crop_pad_method": "none",
                "crop_pad_left": "0",
                "crop_pad_right": "0",
                "crop_pad_top": "0",
                "crop_pad_bottom": "0",
                "crop_pad_fill_value": "0",
                "crop_rescaled": "false",
                "crop_transform_sha256": transform_sha256,
                "immediate_source_raw_path": immediate["raw_path"],
                "immediate_source_annotation_path": immediate["annotation_path"],
                "immediate_source_raw_sha256": immediate["raw_sha256"],
                "immediate_source_annotation_sha256": immediate[
                    "annotation_sha256"
                ],
                "curated_raw_sha256": _sha256_file(raw_path),
                "curated_annotation_sha256": _sha256_file(annotation_path),
                "training_included": "true",
            }
        )
    return rows


class SingleLegV3RunnerTests(unittest.TestCase):
    def test_runner_uses_v3_paths_and_versions_without_legacy_pairing(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'output_root="${KNEE_RETRAINING_OUTPUT_ROOT:-outputs/'
            'retraining_single_leg_v3_20260811}"',
            text,
        )
        self.assertIn(
            'model_version_namespace="${KNEE_RETRAINING_MODEL_VERSION_NAMESPACE-'
            '20260811-single-leg-v3-curated}"',
            text,
        )
        self.assertIn('$output_root/manifests_curated', text)
        self.assertIn('"tail_review_recrop_release_safe"', text)
        self.assertIn(
            'local version="${model_version_namespace}-${version_prefix}-fold${fold}"',
            text,
        )
        self.assertIn(
            'local final_version="${model_version_namespace}-${final_version_suffix}"',
            text,
        )
        for stale_version in (
            "20260811-single-leg-v3-curated-bone-final-v1",
            "20260811-single-leg-v3-curated-tka-final-v1",
            "20260811-single-leg-v3-curated-bone-tka-mixed-final-v1",
        ):
            self.assertNotIn(stale_version, text)
        self.assertNotIn("baseline_manifest_root", text)
        self.assertNotIn("sample set differs from 20260803 baseline", text)

    @unittest.skipIf(
        os.name == "nt",
        "Bash runner integration requires POSIX path and executable semantics",
    )
    def test_output_root_and_model_namespace_overrides_reach_every_training_call(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "candidate-output"
            manifest_root = output_root / "manifests_curated"
            manifest_root.mkdir(parents=True)
            for name in ("bone_confirmed.csv", "tka.csv", "bone_tka.csv"):
                (manifest_root / name).write_text("fixture\n", encoding="utf-8")

            command_log = root / "commands.jsonl"
            fake_python = root / "fake-python"
            fake_python.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args and args[0] != "-":
    with Path(os.environ["KNEE_RUNNER_TEST_COMMAND_LOG"]).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(args) + "\\n")
    if "--output-dir" in args:
        Path(args[args.index("--output-dir") + 1]).mkdir(parents=True, exist_ok=True)
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            namespace = "fixture-20260811.safe_v4"
            environment = os.environ.copy()
            environment["KNEE_RETRAINING_OUTPUT_ROOT"] = str(output_root)
            environment["KNEE_RETRAINING_MODEL_VERSION_NAMESPACE"] = namespace
            environment["KNEE_RETRAINING_PYTHON_BIN"] = str(fake_python)
            environment["KNEE_RUNNER_TEST_COMMAND_LOG"] = str(command_log)
            environment.pop("KNEE_RETRAINING_MANIFEST_ROOT", None)
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [
                json.loads(line)
                for line in command_log.read_text(encoding="utf-8").splitlines()
            ]
            training_commands = [
                command
                for command in commands
                if command[:2] == ["-m", "knee_xray.training.train_keypoint_baseline"]
            ]
            self.assertEqual(len(training_commands), 18)

            expected_versions = {
                *(
                    f"{namespace}-{prefix}-fold{fold}"
                    for prefix in (
                        "bone-weighted-centroid",
                        "tka-weighted-centroid",
                        "bone-tka-mixed-weighted-centroid",
                    )
                    for fold in range(5)
                ),
                f"{namespace}-bone-final-v1",
                f"{namespace}-tka-final-v1",
                f"{namespace}-bone-tka-mixed-final-v1",
            }
            actual_versions = {
                command[command.index("--model-version") + 1]
                for command in training_commands
            }
            self.assertEqual(actual_versions, expected_versions)

            expected_manifests = {
                str(manifest_root / "bone_confirmed.csv"),
                str(manifest_root / "tka.csv"),
                str(manifest_root / "bone_tka.csv"),
            }
            actual_manifests = {
                command[command.index("--manifest") + 1]
                for command in training_commands
            }
            self.assertEqual(actual_manifests, expected_manifests)
            for command in training_commands:
                command_output = Path(command[command.index("--output-dir") + 1])
                self.assertIn(output_root.resolve(), command_output.resolve().parents)

    @unittest.skipIf(
        os.name == "nt",
        "Bash runner integration requires POSIX path and executable semantics",
    )
    def test_model_namespace_rejects_empty_or_unsafe_values_before_preflight(
        self,
    ) -> None:
        for namespace, expected_error in (
            ("", "must not be empty"),
            ("unsafe namespace", "contains unsafe characters"),
            ("../unsafe", "contains unsafe characters"),
        ):
            with self.subTest(namespace=namespace):
                environment = os.environ.copy()
                environment["KNEE_RETRAINING_MODEL_VERSION_NAMESPACE"] = namespace
                result = subprocess.run(
                    ["bash", str(RUNNER), "--preflight-only"],
                    cwd=REPO_ROOT,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_resume_oof_and_train_all_functions_match_v2_contract(self) -> None:
        old = V2_RUNNER.read_text(encoding="utf-8")
        new = RUNNER.read_text(encoding="utf-8").replace(
            '  local final_version_suffix="$6"\n'
            '  local final_version="${model_version_namespace}-${final_version_suffix}"\n',
            '  local final_version="$6"\n',
        ).replace(
            "${model_version_namespace}", "20260803-single-leg-v2"
        )
        boundaries = (
            ("training_complete() {", "evaluation_complete() {"),
            ("evaluation_complete() {", "cv_complete() {"),
            ("cv_complete() {", "run_fold() {"),
            ("run_fold() {", "run_model() {"),
            ("run_model() {", 'require_file "$python_bin"'),
        )

        for start, end in boundaries:
            with self.subTest(section=start):
                self.assertEqual(
                    new[new.index(start) : new.index(end)],
                    old[old.index(start) : old.index(end)],
                )

    @unittest.skipIf(
        os.name == "nt",
        "Bash runner integration requires POSIX path and executable semantics",
    )
    def test_expanded_manifest_preflight_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_root = Path(directory) / "manifests"
            manifest_root.mkdir()
            bone_rows = _build_rows(Path(directory), "bone", 381)
            tka_rows = _build_rows(Path(directory), "tka", 288)
            _write_manifest(manifest_root / "bone_confirmed.csv", bone_rows)
            _write_manifest(manifest_root / "tka.csv", tka_rows)
            _write_manifest(manifest_root / "bone_tka.csv", [*bone_rows, *tka_rows])

            environment = os.environ.copy()
            environment["KNEE_RETRAINING_MANIFEST_ROOT"] = str(manifest_root)
            environment["KNEE_RETRAINING_PYTHON_BIN"] = sys.executable
            command = ["bash", str(RUNNER), "--preflight-only"]
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Bone +1, TKA +1, Mixed=669", result.stdout)

            first_raw = Path(bone_rows[0]["raw_path"])
            duplicate_raw = Path(bone_rows[2]["raw_path"])
            duplicate_raw_original = duplicate_raw.read_bytes()
            duplicate_raw_sha_original = bone_rows[2]["curated_raw_sha256"]
            duplicate_raw.write_bytes(first_raw.read_bytes())
            bone_rows[2]["curated_raw_sha256"] = bone_rows[0]["curated_raw_sha256"]
            _write_manifest(manifest_root / "bone_confirmed.csv", bone_rows)
            _write_manifest(manifest_root / "bone_tka.csv", [*bone_rows, *tka_rows])
            duplicate_result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(duplicate_result.returncode, 0)
            self.assertIn("duplicate training content", duplicate_result.stderr)
            duplicate_raw.write_bytes(duplicate_raw_original)
            bone_rows[2]["curated_raw_sha256"] = duplicate_raw_sha_original

            bone_rows[0]["training_included"] = "false"
            _write_manifest(manifest_root / "bone_confirmed.csv", bone_rows)
            _write_manifest(manifest_root / "bone_tka.csv", [*bone_rows, *tka_rows])
            contract_result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(contract_result.returncode, 0)
            self.assertIn("training_included must be true", contract_result.stderr)
            bone_rows[0]["training_included"] = "true"

            bone_rows[0]["image_width"] = "13"
            bone_rows[0]["crop_width"] = "13"
            bone_rows[0]["crop_x1"] = "13"
            bone_rows[0]["crop_output_width"] = "13"
            _write_manifest(manifest_root / "bone_confirmed.csv", bone_rows)
            _write_manifest(manifest_root / "bone_tka.csv", [*bone_rows, *tka_rows])
            aspect_result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(aspect_result.returncode, 0)
            self.assertIn("single-leg image aspect ratio", aspect_result.stderr)


if __name__ == "__main__":
    unittest.main()

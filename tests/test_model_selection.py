from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from knee_measurement_app import (
    model_version_expectation,
    validate_expected_model_versions,
)
from knee_model_runtime import ModelLoadError, load_app_config, resolve_model_selection


def _model_payload(checkpoint: str, display_name: str, cohort: str) -> dict[str, object]:
    return {
        "adapter": "small_heatmap_v1",
        "checkpoint": checkpoint,
        "display_name": display_name,
        "version": "auto",
        "cohort": cohort,
        "device": "cpu",
        "options": {
            "low_peak_threshold": 0.35,
            "allow_legacy_checkpoint": False,
        },
    }


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class ReleaseModelVersionExpectationTests(unittest.TestCase):
    def test_accepts_exact_builtin_model_key_and_nonempty_version(self) -> None:
        self.assertEqual(
            model_version_expectation(
                "bone=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1"
            ),
            (
                "bone",
                "20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1",
            ),
        )

    def test_rejects_unknown_blank_or_unscoped_expectation(self) -> None:
        for value in ("external=v1", "bone=", "v1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                model_version_expectation(value)

    def test_release_gate_rejects_config_version_override(self) -> None:
        config = SimpleNamespace(
            models={
                "bone": SimpleNamespace(version="forged-version"),
                "tka": SimpleNamespace(version="auto"),
                "mixed": SimpleNamespace(version="auto"),
            }
        )
        infos = {
            key: SimpleNamespace(version="expected-version")
            for key in config.models
        }
        with self.assertRaisesRegex(RuntimeError, "must use config version 'auto'"):
            validate_expected_model_versions(
                config,
                infos,
                [(key, "expected-version") for key in config.models],
            )

    def test_release_gate_accepts_only_exact_checkpoint_versions(self) -> None:
        config = SimpleNamespace(
            models={
                key: SimpleNamespace(version="auto")
                for key in ("bone", "tka", "mixed")
            }
        )
        infos = {
            key: SimpleNamespace(version=f"{key}-checkpoint-version")
            for key in config.models
        }
        expectations = [
            (key, f"{key}-checkpoint-version") for key in config.models
        ]
        validate_expected_model_versions(config, infos, expectations)
        with self.assertRaisesRegex(RuntimeError, "does not match expected"):
            validate_expected_model_versions(
                config,
                infos,
                [(key, "wrong-version") for key in config.models],
            )

    def test_release_gate_rejects_incomplete_or_duplicate_expectations(self) -> None:
        config = SimpleNamespace(
            models={
                key: SimpleNamespace(version="auto")
                for key in ("bone", "tka", "mixed")
            }
        )
        infos = {
            key: SimpleNamespace(version=f"{key}-checkpoint-version")
            for key in config.models
        }
        with self.assertRaisesRegex(RuntimeError, "cover exactly"):
            validate_expected_model_versions(
                config,
                infos,
                [("bone", "bone-checkpoint-version")],
            )
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            validate_expected_model_versions(
                config,
                infos,
                [
                    ("bone", "bone-checkpoint-version"),
                    ("bone", "bone-checkpoint-version"),
                    ("tka", "tka-checkpoint-version"),
                    ("mixed", "mixed-checkpoint-version"),
                ],
            )


class MultiModelConfigTests(unittest.TestCase):
    def test_three_model_config_resolves_every_checkpoint_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "app.json"
            mixed = _model_payload("models/mixed.pt", "Mixed", "bone-tka-mixed")
            _write_config(
                config_path,
                {
                    "schema_version": 1,
                    "model": mixed,
                    "models": {
                        "bone": _model_payload("models/bone.pt", "Bone", "confirmed-bone"),
                        "tka": _model_payload("models/tka.pt", "TKA", "tka-cohort"),
                        "mixed": mixed,
                    },
                    "default_model_key": "mixed",
                    "auto_fallback_model_key": "mixed",
                },
            )

            config = load_app_config(config_path)

            self.assertEqual(set(config.models), {"bone", "tka", "mixed"})
            self.assertEqual(config.default_model_key, "mixed")
            self.assertEqual(config.auto_fallback_model_key, "mixed")
            self.assertEqual(config.model, config.models["mixed"])
            for key in ("bone", "tka", "mixed"):
                self.assertEqual(config.models[key].checkpoint, (root / "models" / f"{key}.pt").resolve())
                self.assertEqual(config.models[key].device, "cpu")

    def test_legacy_single_model_schema_remains_supported_as_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "legacy.json"
            _write_config(
                config_path,
                {
                    "schema_version": 1,
                    "model": _model_payload("weights/legacy.pt", "Legacy", "legacy-mixed"),
                },
            )

            config = load_app_config(config_path)

            self.assertEqual(set(config.models), {"mixed"})
            self.assertEqual(config.model, config.models["mixed"])
            self.assertEqual(config.model.checkpoint, (root / "weights" / "legacy.pt").resolve())
            self.assertEqual(config.default_model_key, "mixed")
            self.assertEqual(config.auto_fallback_model_key, "mixed")

            selection = resolve_model_selection(
                "auto",
                config.models,
                Path("001L_pre_bone_raw.jpg"),
                fallback_model_key=config.auto_fallback_model_key,
            )
            self.assertEqual(selection.model_key, "mixed")
            self.assertEqual(selection.source, "auto_fallback_unknown")

    def test_declared_multi_model_config_requires_all_builtin_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "missing-tka.json"
            mixed = _model_payload("models/mixed.pt", "Mixed", "bone-tka-mixed")
            _write_config(
                config_path,
                {
                    "schema_version": 1,
                    "model": mixed,
                    "models": {
                        "bone": _model_payload("models/bone.pt", "Bone", "confirmed-bone"),
                        "mixed": mixed,
                    },
                },
            )

            with self.assertRaisesRegex(ModelLoadError, "tka"):
                load_app_config(config_path)

    def test_legacy_model_must_match_declared_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "mismatched-default.json"
            _write_config(
                config_path,
                {
                    "schema_version": 1,
                    "model": _model_payload("models/not-mixed.pt", "Wrong", "wrong"),
                    "models": {
                        "bone": _model_payload("models/bone.pt", "Bone", "confirmed-bone"),
                        "tka": _model_payload("models/tka.pt", "TKA", "tka-cohort"),
                        "mixed": _model_payload("models/mixed.pt", "Mixed", "bone-tka-mixed"),
                    },
                    "default_model_key": "mixed",
                },
            )

            with self.assertRaisesRegex(ModelLoadError, "一致"):
                load_app_config(config_path)


class ModelSelectionTests(unittest.TestCase):
    AVAILABLE = ("bone", "tka", "mixed")

    def test_auto_selects_bone_from_exact_filename_token(self) -> None:
        selection = resolve_model_selection("auto", self.AVAILABLE, Path("015R_pre_bone_raw.jpg"))
        self.assertEqual(selection.requested_mode, "auto")
        self.assertEqual(selection.model_key, "bone")
        self.assertEqual(selection.source, "filename")

    def test_auto_selects_tka_case_insensitively_and_after_nfkc_normalization(self) -> None:
        for filename in ("103L_post_TKA_raw.jpg", "103L_post_tka_raw.jpg", "103L_post_ＴＫＡ_raw.jpg"):
            with self.subTest(filename=filename):
                selection = resolve_model_selection("auto", self.AVAILABLE, Path(filename))
                self.assertEqual(selection.model_key, "tka")
                self.assertEqual(selection.source, "filename")

    def test_auto_recognizes_non_implant_phrase_as_bone_without_tka_conflict(self) -> None:
        for description in ("未加入人工關節", "未加入人工関節", "人工關節なし", "non-TKA"):
            with self.subTest(description=description):
                selection = resolve_model_selection("auto", self.AVAILABLE, description)
                self.assertEqual(selection.model_key, "bone")
                self.assertEqual(selection.source, "filename")

    def test_auto_selects_mixed_when_filename_explicitly_names_mixed(self) -> None:
        selection = resolve_model_selection("auto", self.AVAILABLE, Path("patient_MIXED_raw.jpg"))
        self.assertEqual(selection.model_key, "mixed")
        self.assertEqual(selection.source, "filename")

    def test_auto_falls_back_to_mixed_for_conflicting_cohort_tokens(self) -> None:
        for sources in (
            (Path("001L_bone_TKA_raw.jpg"),),
            (Path("001L_bone_raw.jpg"), "TKA"),
            ("未加入人工關節", "人工膝關節"),
        ):
            with self.subTest(sources=sources):
                selection = resolve_model_selection("auto", self.AVAILABLE, *sources)
                self.assertEqual(selection.model_key, "mixed")
                self.assertEqual(selection.source, "auto_fallback_conflict")

    def test_auto_falls_back_to_mixed_for_unknown_or_non_exact_token(self) -> None:
        for filename in ("patient_001L_raw.jpg", "patient_boneannotation_raw.jpg", "patient_TKA2_raw.jpg"):
            with self.subTest(filename=filename):
                selection = resolve_model_selection("auto", self.AVAILABLE, Path(filename))
                self.assertEqual(selection.model_key, "mixed")
                self.assertEqual(selection.source, "auto_fallback_unknown")

    def test_manual_model_choice_overrides_filename_and_conflicts(self) -> None:
        for mode in self.AVAILABLE:
            with self.subTest(mode=mode):
                selection = resolve_model_selection(mode.upper(), self.AVAILABLE, Path("001L_bone_TKA.jpg"))
                self.assertEqual(selection.requested_mode, mode)
                self.assertEqual(selection.model_key, mode)
                self.assertEqual(selection.source, "manual_override")

    def test_unavailable_manual_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelLoadError, "tka"):
            resolve_model_selection("tka", ("mixed",), Path("001L_post_TKA.jpg"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "audit_measurement_archive.py"
RELEASE_ROOT = "KneeXrayMeasurement-Windows-x64-v0.3.0-20260720"
MAC_RELEASE_ROOT = "KneeXrayMeasurement-macOS-arm64-v0.3.0-20260720"
MODEL_PAYLOADS = {
    "bone.pt": b"approved-bone-model",
    "tka.pt": b"approved-tka-model",
    "mixed.pt": b"approved-mixed-model",
}
MODEL_HASHES = {
    filename: hashlib.sha256(payload).hexdigest()
    for filename, payload in MODEL_PAYLOADS.items()
}


def model_config(checkpoint: str, display_name: str) -> dict[str, object]:
    return {
        "adapter": "small_heatmap_v1",
        "checkpoint": checkpoint,
        "display_name": display_name,
        "version": "auto",
        "cohort": display_name,
        "device": "cpu",
        "options": {"low_peak_threshold": 0.35, "allow_legacy_checkpoint": False},
    }


def release_config() -> dict[str, object]:
    models = {
        "bone": model_config("models/bone.pt", "bone"),
        "tka": model_config("models/tka.pt", "tka"),
        "mixed": model_config("models/mixed.pt", "mixed"),
    }
    return {
        "schema_version": 1,
        "default_model_key": "mixed",
        "auto_fallback_model_key": "mixed",
        "model": models["mixed"],
        "models": models,
    }


class MeasurementArchiveAuditTests(unittest.TestCase):
    @staticmethod
    def _doctor_readme() -> bytes:
        text = "Full-Length Leg X-ray Automated Measurement v0.3.0\n" + "\n".join(MODEL_HASHES.values())
        return text.encode("utf-8")

    def _write_archive(
        self,
        archive: Path,
        *,
        config: dict[str, object] | None = None,
        extra_members: dict[str, bytes] | None = None,
        include_readme: bool = True,
        model_payloads: dict[str, bytes] | None = None,
    ) -> None:
        payloads = dict(MODEL_PAYLOADS if model_payloads is None else model_payloads)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"{RELEASE_ROOT}/KneeXrayMeasurement.exe", b"test executable")
            handle.writestr(
                f"{RELEASE_ROOT}/_internal/knee_measurement_app.json",
                json.dumps(config or release_config(), ensure_ascii=False).encode("utf-8"),
            )
            for filename, payload in payloads.items():
                handle.writestr(f"{RELEASE_ROOT}/_internal/models/{filename}", payload)
            if include_readme:
                handle.writestr(f"{RELEASE_ROOT}/README_DOCTOR_EN.txt", self._doctor_readme())
            for member, payload in (extra_members or {}).items():
                handle.writestr(member, payload)

    def _write_macos_archive(self, archive: Path) -> None:
        app_root = f"{MAC_RELEASE_ROOT}/KneeXrayMeasurement.app/Contents"
        resource_root = f"{app_root}/Resources"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"{app_root}/MacOS/KneeXrayMeasurement", b"test executable")
            handle.writestr(
                f"{app_root}/Info.plist",
                plistlib.dumps({"CFBundleShortVersionString": "0.3.0"}),
            )
            handle.writestr(
                f"{resource_root}/knee_measurement_app.json",
                json.dumps(release_config(), ensure_ascii=False).encode("utf-8"),
            )
            for filename, payload in MODEL_PAYLOADS.items():
                handle.writestr(f"{resource_root}/models/{filename}", payload)
            handle.writestr(f"{MAC_RELEASE_ROOT}/README_DOCTOR_JA.txt", self._doctor_readme())

    def _audit(
        self,
        archive: Path,
        *,
        platform: str = "windows",
        release_root: str = RELEASE_ROOT,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(archive),
            "--platform",
            platform,
            "--expected-root",
            release_root,
        ]
        for filename in ("bone.pt", "tka.pt", "mixed.pt"):
            command.extend(("--expected-model", f"{filename}={MODEL_HASHES[filename]}"))
        command.extend(("--expected-app-version", "0.3.0"))
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_clean_three_model_archive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(archive)
            result = self._audit(archive)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "ok"', result.stdout)

    def test_clean_macos_three_model_archive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_macos_archive(archive)
            result = self._audit(archive, platform="macos", release_root=MAC_RELEASE_ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "ok"', result.stdout)

    def test_old_current_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(
                archive,
                extra_members={f"{RELEASE_ROOT}/_internal/models/current.pt": b"old mixed model"},
            )
            result = self._audit(archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bundled model set", result.stderr)

    def test_wrong_model_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            payloads = dict(MODEL_PAYLOADS)
            payloads["tka.pt"] = b"tampered"
            self._write_archive(archive, model_payloads=payloads)
            result = self._audit(archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 mismatch", result.stderr)

    def test_metadata_junk_and_unlisted_sibling_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(
                archive,
                extra_members={
                    f"{RELEASE_ROOT}/.DS_Store": b"junk",
                    f"{RELEASE_ROOT}/developer-notes.txt": b"not for doctors",
                    f"{RELEASE_ROOT}/_internal/images/patient.png": b"not for release",
                },
            )
            result = self._audit(archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("banned metadata file", result.stderr)
        self.assertIn("forbidden resource directory", result.stderr)
        self.assertIn("outside the release whitelist", result.stderr)

    def test_missing_doctor_readme_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(archive, include_readme=False)
            result = self._audit(archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required member", result.stderr)

    def test_wrong_config_model_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            config = release_config()
            config["models"]["bone"]["checkpoint"] = "models/tka.pt"  # type: ignore[index]
            self._write_archive(archive, config=config)
            result = self._audit(archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model 'bone' must use checkpoint models/bone.pt", result.stderr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import marshal
from pathlib import Path
import plistlib
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "audit_measurement_archive.py"
RELEASE_ROOT = "KneeXrayMeasurement-Windows-x64-v0.5.1-20260803"
RELABELLED_RELEASE_ROOT = "KneeXrayMeasurement-Windows-x64-v0.6.0-20260811"
CANDIDATE_RELEASE_ROOT = "KneeXrayMeasurement-ResearchCandidate-Windows-x64-v0.6.0-20260811"
MAC_RELEASE_ROOT = "KneeXrayMeasurement-macOS-arm64-v0.5.1-20260803"
MAC_CANDIDATE_RELEASE_ROOT = "KneeXrayMeasurement-ResearchCandidate-macOS-arm64-v0.6.0-20260811"
CANDIDATE_MARKER = "INTERNAL RESEARCH CANDIDATE - NOT FOR CLINICAL USE"
MODEL_PAYLOADS = {
    "bone.pt": b"approved-bone-model",
    "tka.pt": b"approved-tka-model",
    "mixed.pt": b"approved-mixed-model",
}
MODEL_HASHES = {
    filename: hashlib.sha256(payload).hexdigest()
    for filename, payload in MODEL_PAYLOADS.items()
}


def pyinstaller_executable(
    entrypoint: str,
    app_version: str,
    *,
    candidate_channel: bool = False,
) -> bytes:
    entrypoint_name = entrypoint.encode("ascii")
    source = f"APP_VERSION = {app_version!r}\n"
    if candidate_channel:
        source += f"APP_RELEASE_CHANNEL = {CANDIDATE_MARKER!r}\n"
    entrypoint_payload = marshal.dumps(
        compile(source, f"{entrypoint}.py", "exec")
    )
    compressed_payload = zlib.compress(entrypoint_payload)
    toc_entry_format = "!IIIIBc"
    toc_entry_length = struct.calcsize(toc_entry_format)
    name_length = len(entrypoint_name) + 1
    name_length += (-toc_entry_length - name_length) % 16
    toc = struct.pack(
        toc_entry_format + f"{name_length}s",
        toc_entry_length + name_length,
        0,
        len(compressed_payload),
        len(entrypoint_payload),
        1,
        b"s",
        entrypoint_name,
    )
    cookie_format = "!8sIIII64s"
    cookie_length = struct.calcsize(cookie_format)
    archive_length = len(compressed_payload) + len(toc) + cookie_length
    cookie = struct.pack(
        cookie_format,
        b"MEI\014\013\012\013\016",
        archive_length,
        len(compressed_payload),
        len(toc),
        sys.version_info.major * 100 + sys.version_info.minor,
        f"python{sys.version_info.major}{sys.version_info.minor}.dll".encode("ascii"),
    )
    return b"test PyInstaller bootloader" + compressed_payload + toc + cookie


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
    def _doctor_readme(app_version: str = "0.5.1", *, candidate_marker: bool = False) -> bytes:
        text = f"Full-Length Leg X-ray Automated Measurement v{app_version}\n" + "\n".join(
            MODEL_HASHES.values()
        )
        if candidate_marker:
            text = f"{CANDIDATE_MARKER}\n{text}"
        return text.encode("utf-8")

    def _write_archive(
        self,
        archive: Path,
        *,
        config: dict[str, object] | None = None,
        extra_members: dict[str, bytes] | None = None,
        include_readme: bool = True,
        model_payloads: dict[str, bytes] | None = None,
        backslash_members: bool = False,
        release_root: str = RELEASE_ROOT,
        readme_app_version: str = "0.5.1",
        readme_candidate_marker: bool = False,
        executable_app_version: str = "0.5.1",
        executable_candidate_channel: bool = False,
    ) -> None:
        def member_name(name: str) -> str:
            return name.replace("/", "\\") if backslash_members else name

        payloads = dict(MODEL_PAYLOADS if model_payloads is None else model_payloads)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(
                member_name(f"{release_root}/KneeXrayMeasurement.exe"),
                pyinstaller_executable(
                    "knee_measurement_app_windows",
                    executable_app_version,
                    candidate_channel=executable_candidate_channel,
                ),
            )
            handle.writestr(
                member_name(f"{release_root}/_internal/knee_measurement_app.json"),
                json.dumps(config or release_config(), ensure_ascii=False).encode("utf-8"),
            )
            for filename, payload in payloads.items():
                handle.writestr(
                    member_name(f"{release_root}/_internal/models/{filename}"),
                    payload,
                )
            if include_readme:
                handle.writestr(
                    member_name(f"{release_root}/README_DOCTOR_EN.txt"),
                    self._doctor_readme(
                        readme_app_version,
                        candidate_marker=readme_candidate_marker,
                    ),
                )
            for member, payload in (extra_members or {}).items():
                handle.writestr(member_name(member), payload)

    def _write_macos_archive(
        self,
        archive: Path,
        *,
        release_root: str = MAC_RELEASE_ROOT,
        app_version: str = "0.5.1",
        readme_candidate_marker: bool = False,
        executable_app_version: str | None = None,
        executable_candidate_channel: bool = False,
    ) -> None:
        executable_app_version = executable_app_version or app_version
        app_root = f"{release_root}/KneeXrayMeasurement.app/Contents"
        resource_root = f"{app_root}/Resources"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(
                f"{app_root}/MacOS/KneeXrayMeasurement",
                pyinstaller_executable(
                    "knee_measurement_app",
                    executable_app_version,
                    candidate_channel=executable_candidate_channel,
                ),
            )
            handle.writestr(
                f"{app_root}/Info.plist",
                plistlib.dumps({"CFBundleShortVersionString": app_version}),
            )
            handle.writestr(
                f"{resource_root}/knee_measurement_app.json",
                json.dumps(release_config(), ensure_ascii=False).encode("utf-8"),
            )
            for filename, payload in MODEL_PAYLOADS.items():
                handle.writestr(f"{resource_root}/models/{filename}", payload)
            handle.writestr(
                f"{release_root}/README_DOCTOR_JA.txt",
                self._doctor_readme(
                    app_version,
                    candidate_marker=readme_candidate_marker,
                ),
            )

    def _audit(
        self,
        archive: Path,
        *,
        platform: str = "windows",
        release_root: str = RELEASE_ROOT,
        expected_app_version: str = "0.5.1",
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
        command.extend(("--expected-app-version", expected_app_version))
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_clean_three_model_archive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(archive)
            result = self._audit(archive)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "ok"', result.stdout)

    def test_clean_windows_backslash_member_names_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(archive, backslash_members=True)
            result = self._audit(archive)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "ok"', result.stdout)

    def test_relabelled_old_windows_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(
                archive,
                release_root=RELABELLED_RELEASE_ROOT,
                readme_app_version="0.6.0",
                executable_app_version="0.5.1",
            )
            result = self._audit(
                archive,
                release_root=RELABELLED_RELEASE_ROOT,
                expected_app_version="0.6.0",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "unexpected Windows executable APP_VERSION: 0.5.1 (expected 0.6.0)",
            result.stderr,
        )

    def test_research_candidate_requires_nonclinical_readme_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(
                archive,
                release_root=CANDIDATE_RELEASE_ROOT,
                readme_app_version="0.6.0",
                executable_app_version="0.6.0",
            )
            missing = self._audit(
                archive,
                release_root=CANDIDATE_RELEASE_ROOT,
                expected_app_version="0.6.0",
            )
            self._write_archive(
                archive,
                release_root=CANDIDATE_RELEASE_ROOT,
                readme_app_version="0.6.0",
                readme_candidate_marker=True,
                executable_app_version="0.6.0",
                executable_candidate_channel=True,
            )
            marked = self._audit(
                archive,
                release_root=CANDIDATE_RELEASE_ROOT,
                expected_app_version="0.6.0",
            )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("does not contain the non-clinical marker", missing.stderr)
        self.assertEqual(marked.returncode, 0, marked.stdout + marked.stderr)

    def test_research_candidate_windows_executable_requires_nonclinical_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_archive(
                archive,
                release_root=CANDIDATE_RELEASE_ROOT,
                readme_app_version="0.6.0",
                readme_candidate_marker=True,
                executable_app_version="0.6.0",
            )
            result = self._audit(
                archive,
                release_root=CANDIDATE_RELEASE_ROOT,
                expected_app_version="0.6.0",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Windows executable does not contain the non-clinical channel",
            result.stderr,
        )

    def test_clean_macos_three_model_archive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_macos_archive(archive)
            result = self._audit(archive, platform="macos", release_root=MAC_RELEASE_ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "ok"', result.stdout)

    def test_relabelled_old_macos_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_macos_archive(
                archive,
                release_root=MAC_CANDIDATE_RELEASE_ROOT,
                app_version="0.6.0",
                readme_candidate_marker=True,
                executable_app_version="0.5.1",
                executable_candidate_channel=True,
            )
            result = self._audit(
                archive,
                platform="macos",
                release_root=MAC_CANDIDATE_RELEASE_ROOT,
                expected_app_version="0.6.0",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "unexpected macOS executable APP_VERSION: 0.5.1 (expected 0.6.0)",
            result.stderr,
        )

    def test_research_candidate_macos_executable_requires_nonclinical_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            self._write_macos_archive(
                archive,
                release_root=MAC_CANDIDATE_RELEASE_ROOT,
                app_version="0.6.0",
                readme_candidate_marker=True,
            )
            result = self._audit(
                archive,
                platform="macos",
                release_root=MAC_CANDIDATE_RELEASE_ROOT,
                expected_app_version="0.6.0",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "macOS executable does not contain the non-clinical channel",
            result.stderr,
        )

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

    def test_configured_model_version_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "release.zip"
            config = release_config()
            config["models"]["bone"]["version"] = "forged-version"  # type: ignore[index]
            self._write_archive(archive, config=config)
            result = self._audit(archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model 'bone' must use version auto", result.stderr)


if __name__ == "__main__":
    unittest.main()

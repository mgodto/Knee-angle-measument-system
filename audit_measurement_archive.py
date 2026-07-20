#!/usr/bin/env python3

"""Fail closed when a measurement release ZIP contains unexpected content."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import plistlib
import re
import zipfile
from pathlib import Path, PurePosixPath


APPROVED_MODEL_FILENAMES = ("bone.pt", "tka.pt", "mixed.pt")
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt"}
BANNED_COMPONENTS = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__MACOSX",
    "__pycache__",
}
BANNED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
BANNED_SUFFIXES = {".dcm", ".dicom", ".log", ".nii", ".nrrd", ".pyc", ".pyo", ".tmp"}
FORBIDDEN_RESOURCE_ROOTS = {
    "annotations",
    "build",
    "checkpoints",
    "dist",
    "images",
    "outputs",
    "runs",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expected_model_arg(value: str) -> tuple[str, str]:
    filename, separator, digest = value.partition("=")
    filename = filename.strip()
    digest = digest.strip().lower()
    path = PurePosixPath(filename)
    if not separator or path.name != filename or filename not in APPROVED_MODEL_FILENAMES:
        raise argparse.ArgumentTypeError(
            "expected model must be bone.pt=SHA256, tka.pt=SHA256, or mixed.pt=SHA256"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise argparse.ArgumentTypeError("model SHA-256 must contain 64 lowercase hex characters")
    return filename, digest


def _is_allowed_member(name: str, expected_root: str, platform: str) -> bool:
    readme_filename = "README_DOCTOR_JA.txt" if platform == "macos" else "README_DOCTOR_EN.txt"
    readme_member = f"{expected_root}/{readme_filename}"
    if platform == "macos":
        app_member = f"{expected_root}/KneeXrayMeasurement.app"
        return name in {expected_root, readme_member, app_member} or name.startswith(f"{app_member}/")
    internal_member = f"{expected_root}/_internal"
    executable_member = f"{expected_root}/KneeXrayMeasurement.exe"
    return name in {expected_root, readme_member, internal_member, executable_member} or name.startswith(
        f"{internal_member}/"
    )


def _validate_config(
    handle: zipfile.ZipFile,
    config_member: str,
    member_infos: dict[str, zipfile.ZipInfo],
    expected_models: dict[str, str],
    errors: list[str],
) -> None:
    try:
        payload = json.loads(handle.read(member_infos[config_member]).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid bundled app configuration: {exc}")
        return

    if payload.get("schema_version") != 1:
        errors.append("bundled app configuration must use schema_version 1")
    if payload.get("default_model_key") != "mixed":
        errors.append("bundled default_model_key must be mixed")
    if payload.get("auto_fallback_model_key") != "mixed":
        errors.append("bundled auto_fallback_model_key must be mixed")

    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != {"bone", "tka", "mixed"}:
        errors.append("bundled configuration must define exactly bone, tka, and mixed models")
        return
    for key in ("bone", "tka", "mixed"):
        model = models.get(key)
        if not isinstance(model, dict):
            errors.append(f"bundled configuration model '{key}' is not an object")
            continue
        expected_checkpoint = f"models/{key}.pt"
        if model.get("checkpoint") != expected_checkpoint:
            errors.append(
                f"bundled configuration model '{key}' must use checkpoint {expected_checkpoint}"
            )
        if model.get("device") != "cpu":
            errors.append(f"bundled configuration model '{key}' must use CPU")

    legacy_model = payload.get("model")
    if legacy_model != models.get("mixed"):
        errors.append("legacy model configuration must exactly match the mixed model")

    configured_filenames = {
        PurePosixPath(str(model.get("checkpoint", ""))).name
        for model in models.values()
        if isinstance(model, dict)
    }
    if configured_filenames != set(expected_models):
        errors.append(
            "bundled configuration checkpoint set does not match the approved model files"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--platform", choices=("macos", "windows"), required=True)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument(
        "--expected-model",
        action="append",
        required=True,
        type=expected_model_arg,
        metavar="FILENAME=SHA256",
    )
    parser.add_argument("--expected-app-version", default="0.3.0")
    args = parser.parse_args()

    expected_models: dict[str, str] = {}
    for filename, digest in args.expected_model:
        if filename in expected_models:
            parser.error(f"duplicate expected model: {filename}")
        expected_models[filename] = digest
    if set(expected_models) != set(APPROVED_MODEL_FILENAMES):
        parser.error("expected models must be exactly bone.pt, tka.pt, and mixed.pt")
    if f"-v{args.expected_app_version}-" not in args.expected_root:
        parser.error("expected root must contain the expected app version")

    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"archive does not exist: {archive}")

    errors: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member:
            errors.append(f"CRC failure: {bad_member}")
        infos = handle.infolist()
        if not infos:
            errors.append("archive is empty")

        normalized_names: list[str] = []
        member_infos: dict[str, zipfile.ZipInfo] = {}
        casefolded: dict[str, str] = {}
        for info in infos:
            raw_name = info.filename.replace("\\", "/")
            path = PurePosixPath(raw_name)
            normalized = path.as_posix().rstrip("/")
            normalized_names.append(normalized)
            member_infos.setdefault(normalized, info)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"unsafe path: {raw_name}")
            if not path.parts or path.parts[0] != args.expected_root:
                errors.append(f"unexpected top-level path: {raw_name}")
            if not _is_allowed_member(normalized, args.expected_root, args.platform):
                errors.append(f"member is outside the release whitelist: {raw_name}")
            if any(part in BANNED_COMPONENTS or part.startswith(".venv") for part in path.parts):
                errors.append(f"banned directory: {raw_name}")
            if path.name in BANNED_NAMES or path.name.startswith("._"):
                errors.append(f"banned metadata file: {raw_name}")
            if path.suffix.lower() in BANNED_SUFFIXES or path.name.lower().endswith(".nii.gz"):
                errors.append(f"banned generated or medical-data file: {raw_name}")
            folded = normalized.casefold()
            previous = casefolded.get(folded)
            if previous is not None and previous != normalized:
                errors.append(f"case-insensitive path collision: {previous} / {normalized}")
            casefolded[folded] = normalized

        duplicate_names = sorted(name for name, count in Counter(normalized_names).items() if count > 1)
        errors.extend(f"duplicate archive path: {name}" for name in duplicate_names)

        readme_filename = (
            "README_DOCTOR_JA.txt" if args.platform == "macos" else "README_DOCTOR_EN.txt"
        )
        readme_member = f"{args.expected_root}/{readme_filename}"
        if args.platform == "macos":
            resource_root = f"{args.expected_root}/KneeXrayMeasurement.app/Contents/Resources"
            config_member = f"{resource_root}/knee_measurement_app.json"
            plist_member = f"{args.expected_root}/KneeXrayMeasurement.app/Contents/Info.plist"
            executable_member = f"{args.expected_root}/KneeXrayMeasurement.app/Contents/MacOS/KneeXrayMeasurement"
            for required in (readme_member, config_member, plist_member, executable_member):
                if required not in normalized_names:
                    errors.append(f"missing required member: {required}")
            if plist_member in normalized_names:
                try:
                    plist = plistlib.loads(handle.read(member_infos[plist_member]))
                    version = str(plist.get("CFBundleShortVersionString", ""))
                    if version != args.expected_app_version:
                        errors.append(f"unexpected app version: {version}")
                except Exception as exc:
                    errors.append(f"invalid app Info.plist: {exc}")
        else:
            resource_root = f"{args.expected_root}/_internal"
            config_member = f"{resource_root}/knee_measurement_app.json"
            executable_member = f"{args.expected_root}/KneeXrayMeasurement.exe"
            for required in (readme_member, config_member, executable_member):
                if required not in normalized_names:
                    errors.append(f"missing required member: {required}")

        for forbidden_root in FORBIDDEN_RESOURCE_ROOTS:
            prefix = f"{resource_root}/{forbidden_root}"
            if any(name == prefix or name.startswith(f"{prefix}/") for name in normalized_names):
                errors.append(f"forbidden resource directory: {prefix}")

        expected_model_members = {
            f"{resource_root}/models/{filename}": digest
            for filename, digest in expected_models.items()
        }
        model_members = {
            name for name in normalized_names if PurePosixPath(name).suffix.lower() in MODEL_SUFFIXES
        }
        if model_members != set(expected_model_members):
            errors.append(
                "bundled model set does not match the approved three files; "
                f"expected {sorted(expected_model_members)}, found {sorted(model_members)}"
            )
        for member, expected_digest in expected_model_members.items():
            if (
                member in normalized_names
                and sha256_bytes(handle.read(member_infos[member])) != expected_digest
            ):
                errors.append(f"bundled model SHA-256 mismatch: {member}")

        if config_member in normalized_names:
            _validate_config(handle, config_member, member_infos, expected_models, errors)
        if readme_member in normalized_names:
            try:
                readme = handle.read(member_infos[readme_member]).decode("utf-8")
            except (KeyError, UnicodeDecodeError) as exc:
                errors.append(f"invalid doctor README: {exc}")
            else:
                if f"v{args.expected_app_version}" not in readme:
                    errors.append("doctor README does not contain the expected app version")
                for filename, digest in expected_models.items():
                    if digest not in readme:
                        errors.append(f"doctor README does not contain SHA-256 for {filename}")

    if errors:
        raise SystemExit("release archive audit failed:\n- " + "\n- ".join(sorted(set(errors))))

    summary = {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "entry_count": len(infos),
        "expected_root": args.expected_root,
        "models": expected_models,
        "platform": args.platform,
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

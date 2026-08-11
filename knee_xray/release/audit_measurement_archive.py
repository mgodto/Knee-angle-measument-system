#!/usr/bin/env python3

"""Fail closed when a measurement release ZIP contains unexpected content."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import plistlib
import re
import struct
import zipfile
import zlib
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
PYINSTALLER_COOKIE_MAGIC = b"MEI\014\013\012\013\016"
PYINSTALLER_COOKIE_FORMAT = "!8sIIII64s"
PYINSTALLER_TOC_ENTRY_FORMAT = "!IIIIBc"
WINDOWS_ENTRYPOINT_NAME = "knee_measurement_app_windows"
MACOS_ENTRYPOINT_NAME = "knee_measurement_app"
SEMANTIC_VERSION_PATTERN = re.compile(rb"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+)(?![0-9])")
RESEARCH_CANDIDATE_ROOT_TOKEN = "ResearchCandidate"
RESEARCH_CANDIDATE_MARKER = "INTERNAL RESEARCH CANDIDATE - NOT FOR CLINICAL USE"


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


def _pyinstaller_entrypoint_identity(
    payload: bytes,
    entrypoint_name: str,
) -> tuple[str, bool]:
    cookie_length = struct.calcsize(PYINSTALLER_COOKIE_FORMAT)
    cookie_offset = payload.rfind(PYINSTALLER_COOKIE_MAGIC)
    if cookie_offset < 0 or cookie_offset + cookie_length > len(payload):
        raise ValueError("PyInstaller archive cookie is missing")

    try:
        _magic, archive_length, toc_offset, toc_length, _python_version, _python_library = struct.unpack(
            PYINSTALLER_COOKIE_FORMAT,
            payload[cookie_offset : cookie_offset + cookie_length],
        )
    except struct.error as exc:
        raise ValueError(f"invalid PyInstaller archive cookie: {exc}") from exc

    archive_start = cookie_offset + cookie_length - archive_length
    toc_start = archive_start + toc_offset
    toc_end = toc_start + toc_length
    if archive_start < 0 or not archive_start <= toc_start <= toc_end <= cookie_offset:
        raise ValueError("invalid PyInstaller archive offsets")

    toc_entry_length = struct.calcsize(PYINSTALLER_TOC_ENTRY_FORMAT)
    cursor = toc_start
    entrypoint: tuple[int, int, int, int] | None = None
    while cursor < toc_end:
        if cursor + toc_entry_length > toc_end:
            raise ValueError("truncated PyInstaller table of contents")
        try:
            entry_length, offset, data_length, uncompressed_length, compressed, typecode = struct.unpack(
                PYINSTALLER_TOC_ENTRY_FORMAT,
                payload[cursor : cursor + toc_entry_length],
            )
        except struct.error as exc:
            raise ValueError(f"invalid PyInstaller table of contents: {exc}") from exc
        if entry_length < toc_entry_length or cursor + entry_length > toc_end:
            raise ValueError("invalid PyInstaller table-of-contents entry length")
        try:
            name = payload[cursor + toc_entry_length : cursor + entry_length].rstrip(b"\0").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid PyInstaller entry name: {exc}") from exc
        if name == entrypoint_name:
            if entrypoint is not None or typecode != b"s":
                raise ValueError("invalid or duplicate PyInstaller GUI entrypoint")
            entrypoint = (offset, data_length, uncompressed_length, compressed)
        cursor += entry_length

    if entrypoint is None:
        raise ValueError(f"missing PyInstaller entrypoint: {entrypoint_name}")
    offset, data_length, uncompressed_length, compressed = entrypoint
    if offset < 0 or data_length < 0 or offset + data_length > toc_offset:
        raise ValueError("invalid PyInstaller GUI entrypoint offsets")
    entrypoint_payload = payload[
        archive_start + offset : archive_start + offset + data_length
    ]
    if compressed not in {0, 1}:
        raise ValueError("invalid PyInstaller GUI entrypoint compression flag")
    if compressed:
        try:
            entrypoint_payload = zlib.decompress(entrypoint_payload)
        except zlib.error as exc:
            raise ValueError(f"invalid compressed Windows GUI entrypoint: {exc}") from exc
    if len(entrypoint_payload) != uncompressed_length:
        raise ValueError("PyInstaller GUI entrypoint size does not match its archive metadata")
    if b"APP_VERSION" not in entrypoint_payload:
        raise ValueError("PyInstaller GUI entrypoint does not define APP_VERSION")

    # Do not unmarshal the code object: an archive built with Python 3.11 must
    # still be auditable from Python 3.10 or 3.12. The generated entrypoint has
    # exactly one semantic-version string constant, APP_VERSION; fail closed if
    # that invariant becomes ambiguous.
    versions = {
        match.group(1).decode("ascii")
        for match in SEMANTIC_VERSION_PATTERN.finditer(entrypoint_payload)
    }
    if len(versions) != 1:
        raise ValueError(
            "PyInstaller GUI entrypoint must contain exactly one semantic APP_VERSION literal"
        )
    candidate_marker = RESEARCH_CANDIDATE_MARKER.encode("ascii")
    has_candidate_channel = (
        b"APP_RELEASE_CHANNEL" in entrypoint_payload
        and candidate_marker in entrypoint_payload
    )
    return versions.pop(), has_candidate_channel


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
        if model.get("version") != "auto":
            errors.append(f"bundled configuration model '{key}' must use version auto")

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
    parser.add_argument("--expected-app-version", default="0.5.1")
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
            if executable_member in normalized_names:
                try:
                    executable_version, has_candidate_channel = _pyinstaller_entrypoint_identity(
                        handle.read(member_infos[executable_member]),
                        MACOS_ENTRYPOINT_NAME,
                    )
                    if executable_version != args.expected_app_version:
                        errors.append(
                            "unexpected macOS executable APP_VERSION: "
                            f"{executable_version} (expected {args.expected_app_version})"
                        )
                    if (
                        RESEARCH_CANDIDATE_ROOT_TOKEN in args.expected_root
                        and not has_candidate_channel
                    ):
                        errors.append(
                            "research-candidate macOS executable does not contain the non-clinical channel"
                        )
                except (KeyError, ValueError) as exc:
                    errors.append(f"invalid macOS executable identity: {exc}")
        else:
            resource_root = f"{args.expected_root}/_internal"
            config_member = f"{resource_root}/knee_measurement_app.json"
            executable_member = f"{args.expected_root}/KneeXrayMeasurement.exe"
            for required in (readme_member, config_member, executable_member):
                if required not in normalized_names:
                    errors.append(f"missing required member: {required}")
            if executable_member in normalized_names:
                try:
                    executable_version, has_candidate_channel = _pyinstaller_entrypoint_identity(
                        handle.read(member_infos[executable_member]),
                        WINDOWS_ENTRYPOINT_NAME,
                    )
                    if executable_version != args.expected_app_version:
                        errors.append(
                            "unexpected Windows executable APP_VERSION: "
                            f"{executable_version} (expected {args.expected_app_version})"
                        )
                    if (
                        RESEARCH_CANDIDATE_ROOT_TOKEN in args.expected_root
                        and not has_candidate_channel
                    ):
                        errors.append(
                            "research-candidate Windows executable does not contain the non-clinical channel"
                        )
                except (KeyError, ValueError) as exc:
                    errors.append(f"invalid Windows executable APP_VERSION: {exc}")

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
                if (
                    RESEARCH_CANDIDATE_ROOT_TOKEN in args.expected_root
                    and RESEARCH_CANDIDATE_MARKER not in readme
                ):
                    errors.append(
                        "research-candidate doctor README does not contain the non-clinical marker"
                    )
                for filename, digest in expected_models.items():
                    if digest not in readme:
                        errors.append(f"doctor README does not contain SHA-256 for {filename}")

    if errors:
        raise SystemExit("release archive audit failed:\n- " + "\n- ".join(sorted(set(errors))))

    summary = {
        "app_version": args.expected_app_version,
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "entry_count": len(infos),
        "executable_identity_verified": True,
        "expected_root": args.expected_root,
        "models": expected_models,
        "platform": args.platform,
        "release_channel": (
            "internal_research_candidate"
            if RESEARCH_CANDIDATE_ROOT_TOKEN in args.expected_root
            else "standard"
        ),
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Fail closed when a measurement release ZIP contains unexpected content."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import re
import zipfile
from pathlib import Path, PurePosixPath


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
BANNED_SUFFIXES = {".log", ".pyc", ".pyo", ".tmp"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--platform", choices=("macos", "windows"), required=True)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-app-version", default="0.2.4")
    args = parser.parse_args()

    expected_model_sha = args.expected_model_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_model_sha):
        raise SystemExit("expected model SHA-256 must contain 64 lowercase hex characters")

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
        casefolded: dict[str, str] = {}
        for info in infos:
            raw_name = info.filename.replace("\\", "/")
            path = PurePosixPath(raw_name)
            normalized = path.as_posix().rstrip("/")
            normalized_names.append(normalized)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"unsafe path: {raw_name}")
            if not path.parts or path.parts[0] != args.expected_root:
                errors.append(f"unexpected top-level path: {raw_name}")
            if any(part in BANNED_COMPONENTS or part.startswith(".venv") for part in path.parts):
                errors.append(f"banned directory: {raw_name}")
            if path.name in BANNED_NAMES or path.name.startswith("._"):
                errors.append(f"banned metadata file: {raw_name}")
            if path.suffix.lower() in BANNED_SUFFIXES:
                errors.append(f"banned generated file: {raw_name}")
            folded = normalized.casefold()
            previous = casefolded.get(folded)
            if previous is not None and previous != normalized:
                errors.append(f"case-insensitive path collision: {previous} / {normalized}")
            casefolded[folded] = normalized

        duplicate_names = sorted({name for name in normalized_names if normalized_names.count(name) > 1})
        errors.extend(f"duplicate archive path: {name}" for name in duplicate_names)

        if args.platform == "macos":
            model_member = f"{args.expected_root}/KneeXrayMeasurement.app/Contents/Resources/models/current.pt"
            plist_member = f"{args.expected_root}/KneeXrayMeasurement.app/Contents/Info.plist"
            executable_member = f"{args.expected_root}/KneeXrayMeasurement.app/Contents/MacOS/KneeXrayMeasurement"
            for required in (model_member, plist_member, executable_member):
                if required not in normalized_names:
                    errors.append(f"missing required member: {required}")
            if plist_member in normalized_names:
                plist = plistlib.loads(handle.read(plist_member))
                version = str(plist.get("CFBundleShortVersionString", ""))
                if version != args.expected_app_version:
                    errors.append(f"unexpected app version: {version}")
        else:
            model_member = f"{args.expected_root}/_internal/models/current.pt"
            executable_member = f"{args.expected_root}/KneeXrayMeasurement.exe"
            for required in (model_member, executable_member):
                if required not in normalized_names:
                    errors.append(f"missing required member: {required}")

        model_members = [name for name in normalized_names if Path(name).suffix.lower() in {".pt", ".pth", ".ckpt"}]
        if model_members != [model_member]:
            errors.append(f"expected exactly one bundled model at {model_member}; found {model_members}")
        elif sha256_bytes(handle.read(model_member)) != expected_model_sha:
            errors.append("bundled model SHA-256 does not match the approved checkpoint")

    if errors:
        raise SystemExit("release archive audit failed:\n- " + "\n- ".join(sorted(set(errors))))

    summary = {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "entry_count": len(infos),
        "expected_root": args.expected_root,
        "model_sha256": expected_model_sha,
        "platform": args.platform,
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2

from knee_xray.data.knee_dataset_utils import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_RAW_ROOTS,
    annotation_sample_id,
    build_raw_image_index,
    extract_case_id,
    read_json,
    resolve_raw_candidate,
)
from knee_xray.core.measure_angles import measure_from_named_points


DEFAULT_ANNOTATION_DIRS = (DEFAULT_ANNOTATION_DIR,)
MANIFEST_FIELDS = [
    "sample_id",
    "case_id",
    "source_dataset",
    "dataset_group",
    "implant_status",
    "study_phase",
    "side",
    "annotation_path",
    "raw_path",
    "raw_filename",
    "image_width",
    "image_height",
    "raw_match_count",
    "mldfa",
    "mpta",
]

INVENTORY_FIELDS = [
    "sample_id",
    "case_id",
    "source_dataset",
    "dataset_group",
    "implant_status",
    "study_phase",
    "side",
    "annotation_path",
    "raw_filename",
    "raw_status",
    "raw_path",
    "raw_match_count",
    "image_width",
    "image_height",
    "issue",
]


def iter_annotation_paths(annotation_dirs: list[Path] | tuple[Path, ...]) -> list[Path]:
    paths: dict[str, Path] = {}
    for annotation_dir in annotation_dirs:
        if annotation_dir.is_file():
            if annotation_dir.name.endswith("_annotation.json"):
                paths[annotation_dir.as_posix()] = annotation_dir
            continue
        for annotation_path in annotation_dir.rglob("*_annotation.json"):
            paths[annotation_path.as_posix()] = annotation_path
    return [paths[key] for key in sorted(paths)]


def classify_source(annotation_path: Path) -> tuple[str, str]:
    parts = annotation_path.parts
    if "20260706new" in parts:
        source_dataset = "20260706new"
        if "001-027" in parts:
            return source_dataset, "20260706new_001-027"
        if "051-072" in parts:
            return source_dataset, "20260706new_051-072"
        return source_dataset, "20260706new"
    return "legacy", "legacy"


def metadata_text(sample_id: str, annotation_path: Path, annotation: dict) -> str:
    return " ".join(
        [
            sample_id,
            annotation_path.as_posix(),
            str(annotation.get("raw_filename", "")),
            str(annotation.get("raw_path", "")),
        ]
    ).lower()


def has_token(text: str, token: str) -> bool:
    return bool(re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:[^a-z0-9]|$)", text))


def classify_implant_status(text: str) -> str:
    if has_token(text, "tka"):
        return "TKA"
    if has_token(text, "bone"):
        return "bone"
    return "unknown"


def classify_study_phase(text: str) -> str:
    if has_token(text, "pre"):
        return "pre"
    if has_token(text, "post"):
        return "post"
    return "unknown"


def extract_dataset_case_id(annotation_path: Path, dataset_group: str, sample_id: str) -> str:
    parts = annotation_path.parts
    for dataset_dir in ("001-027", "051-072"):
        if dataset_group != f"20260706new_{dataset_dir}" or dataset_dir not in parts:
            continue
        dataset_index = parts.index(dataset_dir)
        if dataset_index + 1 >= len(parts):
            break
        match = re.search(r"\d+", parts[dataset_index + 1])
        if match:
            return match.group(0)
    return extract_case_id(sample_id)


def annotation_metadata(annotation_path: Path, annotation: dict) -> dict[str, str]:
    sample_id = annotation_sample_id(annotation_path)
    source_dataset, dataset_group = classify_source(annotation_path)
    local_case_id = extract_dataset_case_id(annotation_path, dataset_group, sample_id)
    text = metadata_text(sample_id, annotation_path, annotation)
    return {
        "sample_id": sample_id,
        "case_id": f"{dataset_group}:{local_case_id}",
        "source_dataset": source_dataset,
        "dataset_group": dataset_group,
        "implant_status": classify_implant_status(text),
        "study_phase": classify_study_phase(text),
        "side": str(annotation.get("side", "")),
    }


def passes_filters(metadata: dict[str, str], implant_statuses: set[str] | None, study_phases: set[str] | None) -> bool:
    if implant_statuses is not None and metadata["implant_status"] not in implant_statuses:
        return False
    if study_phases is not None and metadata["study_phase"] not in study_phases:
        return False
    return True


def inventory_row(
    metadata: dict[str, str],
    annotation_path: Path,
    annotation: dict,
    raw_status: str,
    raw_path: Path | None = None,
    raw_match_count: int = 0,
    issue: str = "",
) -> dict[str, str]:
    return {
        **metadata,
        "annotation_path": str(annotation_path),
        "raw_filename": str(annotation.get("raw_filename") or Path(str(annotation.get("raw_path", ""))).name),
        "raw_status": raw_status,
        "raw_path": "" if raw_path is None else str(raw_path),
        "raw_match_count": str(raw_match_count),
        "image_width": str(annotation.get("image_width", "")),
        "image_height": str(annotation.get("image_height", "")),
        "issue": issue,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    annotation_dirs: list[Path] | tuple[Path, ...],
    raw_roots: list[Path],
    *,
    skip_missing_raw: bool = False,
    skip_invalid: bool = False,
    implant_statuses: set[str] | None = None,
    study_phases: set[str] | None = None,
    inventory_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    index = build_raw_image_index(raw_roots)
    rows: list[dict[str, str]] = []
    for annotation_path in iter_annotation_paths(annotation_dirs):
        annotation = read_json(annotation_path)
        metadata = annotation_metadata(annotation_path, annotation)
        if not passes_filters(metadata, implant_statuses, study_phases):
            continue
        try:
            raw_candidate, raw_match_count = resolve_raw_candidate(annotation_path, annotation, index)
        except Exception as exc:
            if inventory_rows is not None:
                inventory_rows.append(
                    inventory_row(metadata, annotation_path, annotation, "missing", issue=str(exc))
                )
            if skip_missing_raw:
                continue
            raise

        raw_image = cv2.imread(str(raw_candidate.path))
        if raw_image is None:
            issue = f"Cannot read resolved raw image: {raw_candidate.path}"
            if inventory_rows is not None:
                inventory_rows.append(
                    inventory_row(metadata, annotation_path, annotation, "invalid", raw_candidate.path, raw_match_count, issue)
                )
            if skip_invalid:
                continue
            raise ValueError(issue)

        try:
            result, _debug = measure_from_named_points(
                raw_image,
                annotation["points"],
                raw_path=raw_candidate.path,
                named_lines=annotation.get("lines"),
                side=annotation.get("side"),
            )
            mldfa = float(result["mldfa_angle"])
            mpta = float(result["mpta_angle"])
            if not math.isfinite(mldfa) or not math.isfinite(mpta):
                raise ValueError(f"Non-finite angle in {annotation_path}")
        except Exception as exc:
            if inventory_rows is not None:
                inventory_rows.append(
                    inventory_row(metadata, annotation_path, annotation, "invalid", raw_candidate.path, raw_match_count, str(exc))
                )
            if skip_invalid:
                continue
            raise

        if inventory_rows is not None:
            inventory_rows.append(
                inventory_row(metadata, annotation_path, annotation, "ok", raw_candidate.path, raw_match_count)
            )
        rows.append(
            {
                **metadata,
                "side": str(annotation.get("side", "")),
                "annotation_path": str(annotation_path),
                "raw_path": str(raw_candidate.path),
                "raw_filename": str(annotation.get("raw_filename", raw_candidate.path.name)),
                "image_width": str(annotation["image_width"]),
                "image_height": str(annotation["image_height"]),
                "raw_match_count": str(raw_match_count),
                "mldfa": f"{mldfa:.6f}",
                "mpta": f"{mpta:.6f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leg-level training manifest from knee annotation JSON files.")
    parser.add_argument("--annotation-dir", type=Path, nargs="*", default=list(DEFAULT_ANNOTATION_DIRS))
    parser.add_argument("--raw-root", type=Path, nargs="*", default=list(DEFAULT_RAW_ROOTS))
    parser.add_argument("--output", type=Path, default=Path("outputs/knee_dataset_manifest.csv"))
    parser.add_argument("--inventory-output", type=Path, help="Optional CSV with all scanned annotations and raw-match status.")
    parser.add_argument("--missing-output", type=Path, help="Optional CSV with skipped/missing/invalid annotations.")
    parser.add_argument("--skip-missing-raw", action="store_true", help="Skip annotations whose raw image is not present locally.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip annotations that cannot be read or measured.")
    parser.add_argument("--implant-status", choices=["bone", "TKA", "unknown"], nargs="*", help="Optional implant-status filter.")
    parser.add_argument("--study-phase", choices=["pre", "post", "unknown"], nargs="*", help="Optional pre/post filter.")
    args = parser.parse_args()

    inventory_rows: list[dict[str, str]] = []
    rows = build_manifest(
        args.annotation_dir,
        args.raw_root,
        skip_missing_raw=args.skip_missing_raw,
        skip_invalid=args.skip_invalid,
        implant_statuses=set(args.implant_status) if args.implant_status else None,
        study_phases=set(args.study_phase) if args.study_phase else None,
        inventory_rows=inventory_rows if args.inventory_output or args.missing_output else None,
    )
    write_csv(args.output, rows, MANIFEST_FIELDS)
    if args.inventory_output:
        write_csv(args.inventory_output, inventory_rows, INVENTORY_FIELDS)
    if args.missing_output:
        missing_rows = [row for row in inventory_rows if row["raw_status"] != "ok"]
        write_csv(args.missing_output, missing_rows, INVENTORY_FIELDS)

    ambiguous = sum(1 for row in rows if int(row["raw_match_count"]) > 1)
    print(f"Wrote {len(rows)} samples: {args.output}")
    print(f"Ambiguous raw matches resolved by heuristic: {ambiguous}")
    if args.inventory_output:
        print(f"Inventory: {args.inventory_output}")
    if args.missing_output:
        print(f"Missing/invalid annotations: {args.missing_output}")


if __name__ == "__main__":
    main()

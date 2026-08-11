#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from knee_xray.core.measure_angles import (
    measure_from_named_points,
    read_color,
    render_annotation_line_image,
    render_annotation_point_image,
    write_image,
)
from knee_xray.data.knee_dataset_utils import load_manifest, read_json


DEFAULT_INPUT_MANIFEST = Path("images/annotation_processed_combined/processed_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("images/annotation_dataset_by_implant")
DATASET_FOLDERS = {
    "non_tka": "未加入人工關節",
    "TKA": "加入人工關節",
}
EXTRA_FIELDS = ["sample_dir", "point_path", "line_path", "combined_path"]


def cleanup_macos_metadata(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.name == ".DS_Store" or path.name.startswith("._"):
            path.unlink()
        elif path.is_dir() and path.name == "__MACOSX":
            shutil.rmtree(path)


def safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "sample"


def relative_to(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_dir_name(row: dict[str, str], used_names: set[str]) -> str:
    base = safe_folder_name(row["sample_id"])
    name = base
    suffix = 2
    while name in used_names:
        name = f"{base}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def write_sample(row: dict[str, str], dataset_dir: Path, used_names: set[str]) -> dict[str, str]:
    sample_dir = dataset_dir / "samples" / sample_dir_name(row, used_names)
    sample_dir.mkdir(parents=True, exist_ok=True)

    source_raw_path = Path(row["raw_path"])
    source_annotation_path = Path(row["annotation_path"])
    raw_suffix = source_raw_path.suffix.lower() or ".jpg"
    raw_path = sample_dir / f"raw{raw_suffix}"
    annotation_path = sample_dir / "annotation.json"
    point_path = sample_dir / "point.jpg"
    line_path = sample_dir / "line.jpg"
    combined_path = sample_dir / "combined.jpg"

    annotation = read_json(source_annotation_path)
    raw_image = read_color(source_raw_path)
    annotation["raw_path"] = raw_path.name
    annotation["raw_filename"] = raw_path.name
    annotation["image_width"] = int(raw_image.shape[1])
    annotation["image_height"] = int(raw_image.shape[0])
    annotation.setdefault("source_annotation_path", row.get("source_annotation_path", str(source_annotation_path)))
    annotation.setdefault("source_raw_path", row.get("source_raw_path", str(source_raw_path)))

    shutil.copy2(source_raw_path, raw_path)
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    point_image = render_annotation_point_image(raw_image, annotation["points"])
    line_image = render_annotation_line_image(raw_image, annotation["points"], annotation.get("lines"))
    result, _debug = measure_from_named_points(
        raw_image,
        annotation["points"],
        raw_path=raw_path,
        named_lines=annotation.get("lines"),
        side=annotation.get("side"),
    )
    write_image(point_path, point_image)
    write_image(line_path, line_image)
    write_image(combined_path, result["combined_image"])

    output_row = dict(row)
    output_row.update(
        {
            "annotation_path": relative_to(annotation_path, dataset_dir),
            "raw_path": relative_to(raw_path, dataset_dir),
            "raw_filename": raw_path.name,
            "image_width": str(raw_image.shape[1]),
            "image_height": str(raw_image.shape[0]),
            "sample_dir": relative_to(sample_dir, dataset_dir),
            "point_path": relative_to(point_path, dataset_dir),
            "line_path": relative_to(line_path, dataset_dir),
            "combined_path": relative_to(combined_path, dataset_dir),
        }
    )
    return output_row


def dataset_key(row: dict[str, str]) -> str | None:
    implant_status = row.get("implant_status", "")
    if implant_status == "TKA":
        return "TKA"
    if implant_status == "bone":
        return "non_tka"
    if implant_status == "unknown" and row.get("dataset_group") == "legacy":
        return "non_tka"
    return None


def organize_dataset(input_manifest: Path, output_root: Path, clean: bool = False) -> dict[str, list[dict[str, str]]]:
    rows = load_manifest(input_manifest)
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cleanup_macos_metadata(output_root)
    grouped_rows: dict[str, list[dict[str, str]]] = {key: [] for key in DATASET_FOLDERS}
    skipped_rows: list[dict[str, str]] = []
    used_names: dict[str, set[str]] = {key: set() for key in DATASET_FOLDERS}

    for row in rows:
        key = dataset_key(row)
        if key is None:
            skipped_rows.append(row)
            continue
        dataset_dir = output_root / DATASET_FOLDERS[key]
        grouped_rows[key].append(write_sample(row, dataset_dir, used_names[key]))

    fieldnames = list(rows[0].keys()) if rows else []
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    for implant_status, folder_name in DATASET_FOLDERS.items():
        dataset_dir = output_root / folder_name
        rows_for_dataset = grouped_rows[implant_status]
        write_rows(dataset_dir / "manifest.csv", rows_for_dataset, fieldnames)

    if skipped_rows:
        write_rows(output_root / "skipped_unknown.csv", skipped_rows, list(skipped_rows[0].keys()))

    cleanup_macos_metadata(output_root)
    return grouped_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create review/training folders split by implant status."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean", action="store_true", help="Remove the output root before rebuilding it.")
    args = parser.parse_args()

    grouped_rows = organize_dataset(args.manifest, args.output_root, clean=args.clean)
    print(f"Output root: {args.output_root}")
    for implant_status, folder_name in DATASET_FOLDERS.items():
        dataset_dir = args.output_root / folder_name
        print(f"{folder_name}: {len(grouped_rows[implant_status])} samples")
        print(f"  manifest: {dataset_dir / 'manifest.csv'}")
        print(f"  samples: {dataset_dir / 'samples'}")


if __name__ == "__main__":
    main()

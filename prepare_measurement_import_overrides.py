#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DATASET_FOLDERS = ("未加入人工關節", "加入人工關節")
MAIN_FOLDER_RE = re.compile(
    r"^(?P<case>\d{3})\s+(?P<side>[LR])\s+(?P<status>bone|TKA)(?P<variant>\d*)$",
    re.IGNORECASE,
)
ARCHIVE_FOLDER_RE = re.compile(
    r"^(?P<patient>\d+)(?P<side>[LR])_(?P<status>bone|TKA)(?P<variant>\d*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExistingRow:
    dataset_folder: str
    manifest_dir: Path
    row: dict[str, str]
    raw_sha256: str
    patient_token: str


@dataclass(frozen=True)
class MeasurementSample:
    annotation_path: Path
    relative_path: str
    raw_path: Path
    raw_sha256: str
    patient_token: str
    source_case: str
    side: str
    folder_status: str
    source_name_status: str
    variant: str
    layout: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def patient_token_from_name(value: str) -> str:
    basename = Path(str(value).replace("\\", "/")).name
    if "_CR_" in basename:
        return basename.split("_CR_", 1)[0]
    match = re.match(r"(?P<patient>\d+)[LR]_(?:bone|TKA)", basename, re.IGNORECASE)
    return match.group("patient") if match else ""


def leading_case_number(sample_id: str) -> str:
    match = re.match(r"\d+", sample_id)
    return "" if match is None else str(int(match.group(0)))


def trailing_case_number(case_id: str) -> str:
    match = re.search(r"\d+$", case_id)
    return "" if match is None else str(int(match.group(0)))


def normalized_status(value: str) -> str:
    return "TKA" if value.lower() == "tka" else "bone"


def status_from_source_name(value: str) -> str:
    match = re.search(r"_(bone|TKA)(?:\d*)?(?:_|\.)", value, re.IGNORECASE)
    return "" if match is None else normalized_status(match.group(1))


def resolve_manifest_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def load_existing_rows(dataset_root: Path) -> list[ExistingRow]:
    rows: list[ExistingRow] = []
    for dataset_folder in DATASET_FOLDERS:
        manifest_path = dataset_root / dataset_folder / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        for row in manifest_rows:
            raw_path = resolve_manifest_path(manifest_path.parent, row["raw_path"])
            rows.append(
                ExistingRow(
                    dataset_folder=dataset_folder,
                    manifest_dir=manifest_path.parent,
                    row=row,
                    raw_sha256=sha256_file(raw_path),
                    patient_token=patient_token_from_name(row.get("source_raw_path", "")),
                )
            )
    return rows


def build_image_index(batch_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in batch_dir.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        index[path.name].append(path)
    return index


def resolve_measurement_raw(
    annotation_path: Path,
    payload: dict,
    image_index: dict[str, list[Path]],
) -> tuple[Path, str]:
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Missing source object: {annotation_path}")
    filename = str(source.get("filename", ""))
    expected_sha256 = str(source.get("sha256", "")).lower()
    if not filename or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"Invalid source filename/SHA: {annotation_path}")
    matches = [
        path
        for path in image_index.get(filename, [])
        if sha256_file(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one raw match for {annotation_path}, found {len(matches)}: {matches}"
        )
    return matches[0], expected_sha256


def parse_folder_context(annotation_path: Path, batch_dir: Path) -> tuple[str, str, str, str, str]:
    relative = annotation_path.relative_to(batch_dir)
    for part in relative.parts[:-1]:
        match = MAIN_FOLDER_RE.fullmatch(part)
        if match:
            return (
                "main",
                str(int(match.group("case"))),
                match.group("side").upper(),
                normalized_status(match.group("status")),
                match.group("variant"),
            )
        match = ARCHIVE_FOLDER_RE.fullmatch(part)
        if match:
            return (
                "archive",
                match.group("patient"),
                match.group("side").upper(),
                normalized_status(match.group("status")),
                match.group("variant"),
            )
    raise ValueError(f"Unrecognized measurement folder layout: {annotation_path}")


def collect_measurements(batch_dir: Path) -> list[MeasurementSample]:
    image_index = build_image_index(batch_dir)
    samples: list[MeasurementSample] = []
    for annotation_path in sorted(batch_dir.rglob("*_measurement.json")):
        if "__MACOSX" in annotation_path.parts or annotation_path.name.startswith("._"):
            continue
        payload = read_json(annotation_path)
        source = payload.get("source")
        analysis = payload.get("analysis")
        if not isinstance(source, dict) or not isinstance(analysis, dict):
            raise ValueError(f"Not a measurement export: {annotation_path}")
        layout, source_case, folder_side, folder_status, variant = parse_folder_context(
            annotation_path, batch_dir
        )
        analysis_side = str(analysis.get("side", "")).upper()
        if analysis_side != folder_side:
            raise ValueError(
                f"Side mismatch for {annotation_path}: folder={folder_side},json={analysis_side}"
            )
        raw_path, raw_sha256 = resolve_measurement_raw(annotation_path, payload, image_index)
        token = patient_token_from_name(str(source.get("filename", "")))
        if not token and layout == "archive":
            token = source_case
        if not token:
            raise ValueError(f"Cannot derive patient token: {annotation_path}")
        samples.append(
            MeasurementSample(
                annotation_path=annotation_path,
                relative_path=annotation_path.relative_to(batch_dir).as_posix(),
                raw_path=raw_path,
                raw_sha256=raw_sha256,
                patient_token=token,
                source_case=source_case,
                side=analysis_side,
                folder_status=folder_status,
                source_name_status=status_from_source_name(str(source.get("filename", ""))),
                variant=variant,
                layout=layout,
            )
        )
    return samples


def existing_case_number(row: ExistingRow) -> str:
    return trailing_case_number(row.row.get("case_id", "")) or leading_case_number(
        row.row.get("sample_id", "")
    )


def select_duplicate_winners(
    samples: list[MeasurementSample],
    existing_by_raw: dict[str, list[ExistingRow]],
    existing_by_token: dict[str, list[ExistingRow]],
) -> tuple[set[str], dict[str, str]]:
    by_raw: dict[str, list[MeasurementSample]] = defaultdict(list)
    for sample in samples:
        by_raw[sample.raw_sha256].append(sample)
    winners: set[str] = set()
    skipped: dict[str, str] = {}
    for raw_sha256, group in by_raw.items():
        if len(group) == 1:
            winners.add(group[0].relative_path)
            continue
        preferred_cases = {
            existing_case_number(row) for row in existing_by_raw.get(raw_sha256, [])
        }
        if not preferred_cases:
            for sample in group:
                preferred_cases.update(
                    existing_case_number(row)
                    for row in existing_by_token.get(sample.patient_token, [])
                )
        preferred = [sample for sample in group if sample.source_case in preferred_cases]
        if len(preferred) != 1:
            details = [(sample.relative_path, sample.source_case) for sample in group]
            raise ValueError(
                "Raw duplicate requires explicit review; could not select one evidence-backed case: "
                f"sha={raw_sha256},preferred_cases={sorted(preferred_cases)},samples={details}"
            )
        kept = preferred[0]
        winners.add(kept.relative_path)
        for sample in group:
            if sample is kept:
                continue
            skipped[sample.relative_path] = f"raw_duplicate_wrong_case_keep:{kept.relative_path}"
    return winners, skipped


def select_existing_target(
    sample: MeasurementSample,
    rows: list[ExistingRow],
) -> ExistingRow | None:
    if not rows:
        return None
    matching_case = [row for row in rows if existing_case_number(row) == sample.source_case]
    if len(matching_case) == 1:
        return matching_case[0]
    if len(rows) == 1:
        return rows[0]
    raise ValueError(
        f"Ambiguous existing raw target for {sample.relative_path}: "
        f"{[(row.row.get('sample_id'), row.row.get('case_id'), row.row.get('sample_dir')) for row in rows]}"
    )


def select_known_misassigned_target(
    sample: MeasurementSample,
    existing_by_token: dict[str, list[ExistingRow]],
) -> ExistingRow | None:
    if sample.layout != "main":
        return None
    candidates = [
        row
        for row in existing_by_token.get(sample.patient_token, [])
        if row.row.get("side") == sample.side
        and existing_case_number(row) == sample.source_case
        and leading_case_number(row.row.get("sample_id", "")) != sample.source_case
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous known-misassignment target for {sample.relative_path}: "
            f"{[(row.row.get('sample_id'), row.row.get('case_id'), row.row.get('sample_dir')) for row in candidates]}"
        )
    return candidates[0] if candidates else None


def phase_from_row(row: ExistingRow) -> str:
    phase = row.row.get("study_phase", "")
    return phase if phase in {"pre", "post", "unknown"} else "unknown"


def replacement_sample_id(sample: MeasurementSample, target: ExistingRow) -> str:
    target_case = leading_case_number(target.row.get("sample_id", ""))
    if target_case == sample.source_case:
        return target.row["sample_id"]
    return f"{int(sample.source_case):03d}{sample.side}_{phase_from_row(target)}_{target.row['implant_status']}"


def raw_instance(sample: MeasurementSample) -> str:
    numbers = re.findall(r"\d+", sample.raw_path.stem)
    instance = numbers[-1][-4:] if numbers else sample.raw_sha256[:8]
    variant = f"{sample.variant}_" if sample.variant else ""
    return f"{variant}{instance}"


def main_case_id_map(existing_rows: list[ExistingRow]) -> dict[str, str]:
    by_case: dict[str, set[str]] = defaultdict(set)
    for row in existing_rows:
        if row.row.get("source_dataset") == "legacy":
            continue
        sample_case = leading_case_number(row.row.get("sample_id", ""))
        case_id = row.row.get("case_id", "")
        if sample_case and case_id:
            by_case[sample_case].add(case_id)
    return {
        case_number: next(iter(case_ids))
        for case_number, case_ids in by_case.items()
        if len(case_ids) == 1
    }


def archive_case_map(samples: list[MeasurementSample], first_case_number: int) -> dict[str, str]:
    patient_ids = sorted(
        {sample.source_case for sample in samples if sample.layout == "archive"},
        key=int,
    )
    return {
        patient_id: str(first_case_number + index)
        for index, patient_id in enumerate(patient_ids)
    }


def build_overrides(
    batch_dir: Path,
    dataset_root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    existing_rows = load_existing_rows(dataset_root)
    samples = collect_measurements(batch_dir)
    existing_by_raw: dict[str, list[ExistingRow]] = defaultdict(list)
    existing_by_token: dict[str, list[ExistingRow]] = defaultdict(list)
    for row in existing_rows:
        existing_by_raw[row.raw_sha256].append(row)
        if row.patient_token:
            existing_by_token[row.patient_token].append(row)

    winners, skipped = select_duplicate_winners(samples, existing_by_raw, existing_by_token)
    main_cases = [int(sample.source_case) for sample in samples if sample.layout == "main"]
    existing_cases = [
        int(case_number)
        for row in existing_rows
        if (case_number := existing_case_number(row))
    ]
    first_archive_case = max([*main_cases, *existing_cases], default=0) + 1
    archive_cases = archive_case_map(samples, first_archive_case)
    existing_main_cases = main_case_id_map(existing_rows)

    overrides: dict[str, dict[str, object]] = {}
    accepted_sample_ids: set[str] = set()
    replacement_targets: set[tuple[str, str]] = set()
    actions: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    sides: Counter[str] = Counter()

    for sample in samples:
        if sample.relative_path in skipped:
            overrides[sample.relative_path] = {"skip_reason": skipped[sample.relative_path]}
            actions["skipped_raw_duplicate"] += 1
            continue
        if sample.relative_path not in winners:
            raise AssertionError(f"Unaccounted duplicate decision: {sample.relative_path}")

        target = select_existing_target(sample, existing_by_raw.get(sample.raw_sha256, []))
        allow_raw_replacement = False
        if target is None:
            target = select_known_misassigned_target(sample, existing_by_token)
            allow_raw_replacement = target is not None

        if target is not None:
            sample_id = replacement_sample_id(sample, target)
            case_id = target.row["case_id"]
            implant_status = target.row["implant_status"]
            study_phase = phase_from_row(target)
            target_key = (target.dataset_folder, target.row["sample_dir"])
            if target_key in replacement_targets:
                raise ValueError(f"Existing target selected twice: {target_key}")
            replacement_targets.add(target_key)
            entry: dict[str, object] = {
                "sample_id": sample_id,
                "case_id": case_id,
                "side": sample.side,
                "implant_status": implant_status,
                "study_phase": study_phase,
                "replace_existing_sample_id": target.row["sample_id"],
                "replace_existing_sample_dir": target.row["sample_dir"],
                "reason": (
                    "reviewed_measurement_correction_raw_repair"
                    if allow_raw_replacement
                    else "reviewed_measurement_correction_same_raw"
                ),
            }
            if allow_raw_replacement:
                entry["allow_raw_replacement"] = True
            if implant_status != sample.folder_status:
                entry["reason"] = (
                    f"{entry['reason']};retain_existing_implant_status_over_folder_label:"
                    f"{sample.folder_status}->{implant_status}"
                )
            overrides[sample.relative_path] = entry
            actions["replacement_raw_repair" if allow_raw_replacement else "replacement"] += 1
        else:
            if sample.layout == "archive":
                case_number = archive_cases[sample.source_case]
                case_id = f"{batch_dir.name}-archive:{case_number}"
            else:
                case_number = str(int(sample.source_case))
                case_id = existing_main_cases.get(case_number, f"{batch_dir.name}:{int(case_number):03d}")
            sample_id = (
                f"{int(case_number):03d}{sample.side}_unknown_{sample.folder_status}_"
                f"{raw_instance(sample)}"
            )
            if sample_id in accepted_sample_ids:
                sample_id = f"{sample_id}_{sample.raw_sha256[:8]}"
            reason = "reviewed_new_measurement_sample"
            if sample.source_name_status and sample.source_name_status != sample.folder_status:
                reason += (
                    ";retain_folder_implant_status_over_source_name:"
                    f"{sample.source_name_status}->{sample.folder_status}"
                )
            overrides[sample.relative_path] = {
                "sample_id": sample_id,
                "case_id": case_id,
                "side": sample.side,
                "implant_status": sample.folder_status,
                "study_phase": "unknown",
                "reason": reason,
            }
            implant_status = sample.folder_status
            actions["addition"] += 1

        if sample_id in accepted_sample_ids:
            raise ValueError(f"Duplicate accepted sample_id: {sample_id}")
        accepted_sample_ids.add(sample_id)
        statuses[implant_status] += 1
        sides[sample.side] += 1

    if len(overrides) != len(samples):
        raise AssertionError(f"Override accounting mismatch: {len(overrides)} != {len(samples)}")
    summary: dict[str, object] = {
        "schema_version": 1,
        "batch_dir": batch_dir.resolve().as_posix(),
        "dataset_root": dataset_root.resolve().as_posix(),
        "source_measurements": len(samples),
        "actions": dict(sorted(actions.items())),
        "accepted": len(accepted_sample_ids),
        "unique_raw_accepted": len(
            {
                sample.raw_sha256
                for sample in samples
                if sample.relative_path not in skipped
            }
        ),
        "implant_status": dict(sorted(statuses.items())),
        "sides": dict(sorted(sides.items())),
        "replacement_targets": len(replacement_targets),
        "archive_case_mapping": archive_cases,
    }
    return overrides, summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build explicit, reviewable overrides for a Knee Measurement App export before "
            "running import_annotation_batch.py."
        )
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("images/annotation_dataset_by_implant"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    overrides, summary = build_overrides(args.batch_dir, args.dataset_root)
    write_json(args.output, {"schema_version": 1, "overrides": overrides})
    if args.summary_output is not None:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

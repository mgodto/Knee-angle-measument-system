#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NON_TKA_MANIFEST = Path(
    "images/annotation_dataset_by_implant/未加入人工關節/manifest.csv"
)
DEFAULT_TKA_MANIFEST = Path(
    "images/annotation_dataset_by_implant/加入人工關節/manifest.csv"
)
OUTPUT_FILENAMES = {
    "bone_confirmed": "bone_confirmed.csv",
    "legacy_unknown": "legacy_unknown.csv",
    "tka": "tka.csv",
    "bone_tka": "bone_tka.csv",
    "all_clean": "all_clean.csv",
}
AUDIT_FILENAME = "audit_summary.json"
PATH_FIELDS = (
    "annotation_path",
    "raw_path",
    "point_path",
    "line_path",
    "combined_path",
    "sample_dir",
    "source_annotation_path",
    "source_raw_path",
)
ANNOTATION_HASH_FIELDS = ("image_width", "image_height", "side", "points", "lines")
CR_PATIENT_TOKEN_RE = re.compile(r"(?P<token>.+?)_CR_", re.IGNORECASE)
OLD_20260803_PATIENT_TOKEN_RE = re.compile(
    r"(?P<token>\d+)(?=[LR]_(?:bone|TKA)(?:\d*)?(?:_|$))",
    re.IGNORECASE,
)
PATIENT_GROUP_NAMESPACE = "knee-xray-patient-group-v1"


class ManifestAuditError(ValueError):
    """Raised when a manifest cannot safely be used for retraining."""


@dataclass(frozen=True)
class HashedRow:
    row: dict[str, str]
    content_sha256: str


def _absolute_path(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ManifestAuditError(f"Path is outside the repository: {path}") from exc


def _normalize_manifest_path(value: str, manifest_path: Path, repo_root: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return _repo_relative(path, repo_root)

    repo_candidate = repo_root / path
    manifest_candidate = manifest_path.parent / path
    if repo_candidate.exists():
        candidate = repo_candidate
    elif manifest_candidate.exists():
        candidate = manifest_candidate
    elif path.parts and (repo_root / path.parts[0]).exists():
        candidate = repo_candidate
    else:
        candidate = manifest_candidate
    return _repo_relative(candidate, repo_root)


def load_manifest(path: Path, repo_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    path = _absolute_path(path, repo_root)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ManifestAuditError(f"Manifest is empty: {path}")
    for required in ("sample_id", "case_id", "implant_status", "source_dataset"):
        if required not in fieldnames:
            raise ManifestAuditError(f"Manifest is missing required field {required!r}: {path}")
    for row in rows:
        for field in PATH_FIELDS:
            if field in row:
                row[field] = _normalize_manifest_path(row[field], path, repo_root)
    return rows, fieldnames


def _case_number(value: str, *, field: str) -> int:
    match = re.match(r"\d+", value) if field == "sample_id" else re.search(r"\d+$", value)
    if not match:
        raise ManifestAuditError(f"Cannot extract a case number from {field}={value!r}")
    return int(match.group(0))


def remove_case_number_mismatches(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    clean: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []
    for row in rows:
        sample_number = _case_number(row["sample_id"], field="sample_id")
        case_number = _case_number(row["case_id"], field="case_id")
        if sample_number == case_number:
            clean.append(row)
            continue
        removed.append(
            {
                "reason": "sample_case_number_mismatch",
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "source_annotation_path": row.get("source_annotation_path", ""),
                "source_raw_path": row.get("source_raw_path", ""),
            }
        )
    return clean, removed


def _patient_token(source_raw_path: str) -> str | None:
    basename = str(source_raw_path).replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(basename).stem
    match = CR_PATIENT_TOKEN_RE.match(stem)
    if match is None:
        match = OLD_20260803_PATIENT_TOKEN_RE.match(stem)
    if match is None:
        return None
    return match.group("token").casefold()


def group_rows_by_patient(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    grouped_rows = [dict(row) for row in rows]
    case_nodes: list[str] = []
    for row in grouped_rows:
        source_case_id = row["case_id"]
        case_node = f"case:{source_case_id}"
        find(case_node)
        token = _patient_token(row.get("source_raw_path", ""))
        if token is not None:
            union(case_node, f"token:{token}")
        case_nodes.append(case_node)

    component_nodes: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        component_nodes[find(node)].append(node)
    group_ids = {
        root: "patient_group:"
        + hashlib.sha256(
            (
                PATIENT_GROUP_NAMESPACE
                + "\n"
                + json.dumps(sorted(nodes), ensure_ascii=False, separators=(",", ":"))
            ).encode("utf-8")
        ).hexdigest()
        for root, nodes in component_nodes.items()
    }

    for row, case_node in zip(grouped_rows, case_nodes):
        row["source_case_id"] = row["case_id"]
        row["case_id"] = group_ids[find(case_node)]
    return grouped_rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_sha256(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    missing = [field for field in ANNOTATION_HASH_FIELDS if field not in annotation]
    if missing:
        raise ManifestAuditError(f"Annotation is missing hash fields {missing}: {path}")
    payload = {field: annotation[field] for field in ANNOTATION_HASH_FIELDS}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_rows(rows: list[dict[str, str]], repo_root: Path) -> list[HashedRow]:
    hashed: list[HashedRow] = []
    for row in rows:
        raw_path = repo_root / row.get("raw_path", "")
        annotation_path = repo_root / row.get("annotation_path", "")
        if not raw_path.is_file():
            raise ManifestAuditError(f"Missing raw image for {row['sample_id']}: {raw_path}")
        if not annotation_path.is_file():
            raise ManifestAuditError(
                f"Missing annotation for {row['sample_id']}: {annotation_path}"
            )
        raw_sha256 = _sha256_file(raw_path)
        annotation_sha256 = _annotation_sha256(annotation_path)
        content_sha256 = hashlib.sha256(
            f"{raw_sha256}:{annotation_sha256}".encode("ascii")
        ).hexdigest()
        hashed.append(HashedRow(row=row, content_sha256=content_sha256))
    return hashed


def reject_exact_duplicates(records: list[HashedRow], dataset_name: str) -> None:
    groups: dict[str, list[HashedRow]] = defaultdict(list)
    for record in records:
        groups[record.content_sha256].append(record)
    duplicates = [group for group in groups.values() if len(group) > 1]
    if not duplicates:
        return
    details = [
        [(record.row["sample_id"], record.row["case_id"]) for record in group]
        for group in duplicates
    ]
    raise ManifestAuditError(f"Exact duplicate content in {dataset_name}: {details}")


def _fold_assignment(case_ids: set[str], num_folds: int, seed: int) -> dict[str, int]:
    if num_folds < 2:
        raise ManifestAuditError("num_folds must be at least 2")
    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)
    return {case_id: index % num_folds for index, case_id in enumerate(shuffled)}


def audit_dataset(
    records: list[HashedRow],
    *,
    num_folds: int,
    seed: int,
) -> dict[str, object]:
    reject_exact_duplicates(records, "audit dataset")
    case_ids = {record.row["case_id"] for record in records}
    assignments = _fold_assignment(case_ids, num_folds, seed)

    sample_hash_cases: dict[str, set[str]] = defaultdict(set)
    case_samples: dict[str, list[str]] = defaultdict(list)
    for record in records:
        case_id = record.row["case_id"]
        sample_hash_cases[record.content_sha256].add(case_id)
        case_samples[case_id].append(record.content_sha256)

    sample_cross_fold: list[dict[str, object]] = []
    for content_hash, cases in sample_hash_cases.items():
        folds = {assignments[case_id] for case_id in cases}
        if len(folds) > 1:
            sample_cross_fold.append(
                {"content_sha256": content_hash, "case_ids": sorted(cases), "folds": sorted(folds)}
            )

    case_hash_cases: dict[str, set[str]] = defaultdict(set)
    for case_id, hashes in case_samples.items():
        encoded = "\n".join(sorted(hashes)).encode("ascii")
        case_hash_cases[hashlib.sha256(encoded).hexdigest()].add(case_id)
    case_cross_fold: list[dict[str, object]] = []
    for case_hash, cases in case_hash_cases.items():
        folds = {assignments[case_id] for case_id in cases}
        if len(folds) > 1:
            case_cross_fold.append(
                {"case_sha256": case_hash, "case_ids": sorted(cases), "folds": sorted(folds)}
            )

    if sample_cross_fold or case_cross_fold:
        raise ManifestAuditError(
            "Content hashes cross folds: "
            f"sample={sample_cross_fold}, case={case_cross_fold}"
        )

    fold_rows = []
    for fold in range(num_folds):
        fold_cases = {case_id for case_id, value in assignments.items() if value == fold}
        fold_rows.append(
            {
                "fold": fold,
                "cases": len(fold_cases),
                "samples": sum(record.row["case_id"] in fold_cases for record in records),
            }
        )
    return {
        "rows": len(records),
        "cases": len(case_ids),
        "sides": dict(sorted(Counter(record.row.get("side", "") for record in records).items())),
        "implant_status": dict(
            sorted(Counter(record.row["implant_status"] for record in records).items())
        ),
        "source_dataset": dict(
            sorted(Counter(record.row["source_dataset"] for record in records).items())
        ),
        "folds": fold_rows,
        "exact_duplicate_groups": 0,
        "sample_hash_cross_fold_groups": sample_cross_fold,
        "case_hash_cross_fold_groups": case_cross_fold,
    }


def _combined_fieldnames(first: list[str], second: list[str]) -> list[str]:
    return list(dict.fromkeys([*first, *second]))


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_retraining_manifests(
    *,
    repo_root: Path,
    non_tka_manifest: Path,
    tka_manifest: Path,
    output_dir: Path,
    num_folds: int = 5,
    seed: int = 42,
    group_by_patient: bool = False,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    non_tka_rows, non_tka_fields = load_manifest(non_tka_manifest, repo_root)
    tka_rows, tka_fields = load_manifest(tka_manifest, repo_root)
    non_tka_clean, non_tka_removed = remove_case_number_mismatches(non_tka_rows)
    tka_clean, tka_removed = remove_case_number_mismatches(tka_rows)
    if group_by_patient:
        non_tka_count = len(non_tka_clean)
        grouped_rows = group_rows_by_patient([*non_tka_clean, *tka_clean])
        non_tka_clean = grouped_rows[:non_tka_count]
        tka_clean = grouped_rows[non_tka_count:]

    bone_rows = [row for row in non_tka_clean if row["implant_status"] == "bone"]
    legacy_rows = [
        row
        for row in non_tka_clean
        if row["implant_status"] == "unknown" and row["source_dataset"] == "legacy"
    ]
    unexpected_non_tka = [row for row in non_tka_clean if row not in bone_rows + legacy_rows]
    if unexpected_non_tka:
        details = [(row["sample_id"], row["implant_status"]) for row in unexpected_non_tka]
        raise ManifestAuditError(f"Unexpected non-TKA rows: {details}")
    unexpected_tka = [row for row in tka_clean if row["implant_status"] != "TKA"]
    if unexpected_tka:
        details = [(row["sample_id"], row["implant_status"]) for row in unexpected_tka]
        raise ManifestAuditError(f"Unexpected TKA rows: {details}")

    bone_records = hash_rows(bone_rows, repo_root)
    legacy_records = hash_rows(legacy_rows, repo_root)
    tka_records = hash_rows(tka_clean, repo_root)
    bone_tka_records = [*bone_records, *tka_records]
    all_records = [*bone_records, *legacy_records, *tka_records]
    reject_exact_duplicates(all_records, "all_clean")

    datasets = {
        "bone_confirmed": bone_records,
        "legacy_unknown": legacy_records,
        "tka": tka_records,
        "bone_tka": bone_tka_records,
        "all_clean": all_records,
    }
    dataset_audits = {
        name: audit_dataset(records, num_folds=num_folds, seed=seed)
        for name, records in datasets.items()
    }

    output_dir = _absolute_path(output_dir, repo_root)
    _repo_relative(output_dir, repo_root)
    output_paths = {
        name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()
    }
    audit_path = output_dir / AUDIT_FILENAME
    parameters: dict[str, object] = {"num_folds": num_folds, "seed": seed}
    hash_definition: dict[str, object] = {
        "sample": "sha256(raw file sha256 + canonical annotation training fields sha256)",
        "annotation_fields": list(ANNOTATION_HASH_FIELDS),
        "case": "sha256(sorted sample hashes for the case)",
    }
    if group_by_patient:
        parameters["group_by_patient"] = True
        hash_definition["patient_group"] = (
            "sha256(sorted original-case and parsed-patient-token connected-component nodes)"
        )

    audit = {
        "schema_version": 1,
        "inputs": {
            "non_tka_manifest": _repo_relative(_absolute_path(non_tka_manifest, repo_root), repo_root),
            "tka_manifest": _repo_relative(_absolute_path(tka_manifest, repo_root), repo_root),
        },
        "outputs": {
            **{name: _repo_relative(path, repo_root) for name, path in output_paths.items()},
            "audit_summary": _repo_relative(audit_path, repo_root),
        },
        "parameters": parameters,
        "hash_definition": hash_definition,
        "counts": {
            "input_non_tka": len(non_tka_rows),
            "input_tka": len(tka_rows),
            "removed_case_number_mismatches": len(non_tka_removed) + len(tka_removed),
            "output_all_clean": len(all_records),
        },
        "removed_rows": [*non_tka_removed, *tka_removed],
        "datasets": dataset_audits,
        "validation_passed": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _combined_fieldnames(non_tka_fields, tka_fields)
    if group_by_patient and "source_case_id" not in fieldnames:
        fieldnames.append("source_case_id")
    for name, records in datasets.items():
        _write_manifest(output_paths[name], [record.row for record in records], fieldnames)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare duplicate-safe, repository-relative manifests for retraining."
    )
    parser.add_argument("--non-tka-manifest", type=Path, default=DEFAULT_NON_TKA_MANIFEST)
    parser.add_argument("--tka-manifest", type=Path, default=DEFAULT_TKA_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-by-patient",
        action="store_true",
        help=(
            "Replace case_id with a stable anonymous connected-component ID built from "
            "original case IDs and reliable source-image patient tokens."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    audit = prepare_retraining_manifests(
        repo_root=repo_root,
        non_tka_manifest=args.non_tka_manifest,
        tka_manifest=args.tka_manifest,
        output_dir=args.output_dir,
        num_folds=args.num_folds,
        seed=args.seed,
        group_by_patient=args.group_by_patient,
    )
    print(f"Clean samples: {audit['counts']['output_all_clean']}")
    print(f"Removed mismatched rows: {audit['counts']['removed_case_number_mismatches']}")
    print(f"Audit: {audit['outputs']['audit_summary']}")


if __name__ == "__main__":
    main()

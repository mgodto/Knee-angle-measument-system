#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from build_dataset_manifest import MANIFEST_FIELDS
from knee_dataset_utils import (
    annotation_keypoints,
    build_raw_image_index,
    load_manifest,
    read_json,
    resolve_raw_candidate,
)
from measure_angles import measure_from_named_points
from organize_implant_dataset import DATASET_FOLDERS, organize_dataset
from process_annotation_dataset import confirmed_inference_roi, process_dataset, source_input_scope
from validate_knee_dataset import validate_manifest


DEFAULT_DATASET_ROOT = Path("images/annotation_dataset_by_implant")
CANONICAL_RE = re.compile(
    r"(?<!\d)(?P<case>\d{3})(?P<side>[LR])?_(?P<phase>pre|post)_(?P<status>bone|TKA|UKA)",
    re.IGNORECASE,
)
STUDY_RE = re.compile(
    r"(?<!\d)(?P<case>\d{3})(?:\s*[LRP]{1,2})?[\s_]+(?P<phase>pre|post)\b",
    re.IGNORECASE,
)
SAFE_SAMPLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SAFE_METADATA_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DECISION_FIELDS = [
    "annotation_path",
    "sample_id",
    "case_id",
    "implant_status",
    "side",
    "raw_path",
    "action",
    "reason",
]


@dataclass
class Candidate:
    annotation_path: Path
    relative_path: str
    sample_id: str
    case_number: str
    side: str
    phase: str
    implant_status: str
    case_id: str
    annotation: dict
    raw_path: Path
    raw_match_count: int
    raw_sha256: str
    source_json_sha256: str
    mldfa: float
    mpta: float
    inference_rank: int
    replace_existing_sample_id: str = ""
    replace_existing_sample_dir: str = ""
    allow_raw_replacement: bool = False
    corrections: dict[str, object] = field(default_factory=dict)

    @property
    def has_explicit_replacement(self) -> bool:
        return bool(self.replace_existing_sample_id or self.replace_existing_sample_dir)


@dataclass(frozen=True)
class ExistingSample:
    folder_name: str
    row: dict[str, str]

    @property
    def locator(self) -> str:
        return f"{self.folder_name}/{self.row['sample_dir']}"


@dataclass(frozen=True)
class ReplacementPlan:
    old_folder_name: str
    old_sample_id: str
    old_sample_dir: str
    new_sample_id: str

    @property
    def locator(self) -> str:
        return f"{self.old_folder_name}/{self.old_sample_dir}"


def compact_text(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    return re.sub(r"_+", "_", compact)


def parse_canonical(value: str, fallback_side: str = "") -> tuple[str, str, str, str, str] | None:
    match = CANONICAL_RE.search(compact_text(value))
    if match is None:
        return None
    case_number = match.group("case")
    side = (match.group("side") or fallback_side).upper()
    if side not in {"L", "R"}:
        return None
    phase = match.group("phase").lower()
    status_token = match.group("status").lower()
    implant_status = "TKA" if status_token == "tka" else status_token
    sample_id = f"{case_number}{side}_{phase}_{implant_status}"
    return sample_id, case_number, side, phase, implant_status


def study_context(value: str) -> tuple[str, str] | None:
    normalized = str(value).replace("\\", "/")
    for part in reversed(normalized.split("/")):
        match = STUDY_RE.search(part)
        if match:
            return match.group("case"), match.group("phase").lower()
    return None


def nearest_local_context(annotation_path: Path, batch_dir: Path) -> tuple[str, str] | None:
    current = annotation_path.parent
    while current != batch_dir and batch_dir in current.parents:
        context = study_context(current.name)
        if context:
            return context
        current = current.parent
    return study_context(annotation_path.parent.name)


def is_safe_sample_id(value: object) -> bool:
    text = str(value)
    return bool(text and len(text) <= 200 and SAFE_SAMPLE_ID_RE.fullmatch(text))


def normalize_safe_relative_path(value: object, *, field_name: str) -> str:
    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in text.split("/")):
        raise ValueError(f"{field_name} must be a safe relative path")
    return path.as_posix()


def case_number_from_sample_id(sample_id: str) -> str:
    match = re.match(r"\d+", sample_id)
    if match:
        return match.group(0)
    return sample_id.split("_", 1)[0]


def adapt_annotation_schema(annotation: object) -> tuple[dict, str] | None:
    """Return the importer schema plus an optional authoritative raw SHA-256."""
    if not isinstance(annotation, dict):
        return None

    legacy_fields = ("image_width", "image_height", "points", "lines", "side")
    if all(key in annotation for key in legacy_fields):
        return copy.deepcopy(annotation), ""

    source = annotation.get("source")
    analysis = annotation.get("analysis")
    if not (
        isinstance(source, dict)
        and isinstance(analysis, dict)
        and "points" in annotation
        and "lines" in annotation
    ):
        return None

    missing_source = [
        key
        for key in ("filename", "sha256", "image_width", "image_height")
        if key not in source
    ]
    if missing_source or "side" not in analysis:
        missing = [*(f"source.{key}" for key in missing_source)]
        if "side" not in analysis:
            missing.append("analysis.side")
        raise ValueError("measurement_app_schema_missing:" + ",".join(missing))

    expected_raw_sha256 = str(source["sha256"]).strip().lower()
    if not SHA256_RE.fullmatch(expected_raw_sha256):
        raise ValueError("measurement_app_schema_invalid:source.sha256")

    adapted = copy.deepcopy(annotation)
    adapted["image_width"] = int(source["image_width"])
    adapted["image_height"] = int(source["image_height"])
    adapted["raw_filename"] = str(source["filename"])
    adapted["source_raw_filename"] = str(source["filename"])
    adapted["side"] = str(analysis["side"]).upper()
    return adapted, expected_raw_sha256


def is_annotation(annotation: object) -> bool:
    try:
        return adapt_annotation_schema(annotation) is not None
    except (TypeError, ValueError):
        return False


def explicit_json_for_side(study_dir: Path, side: str) -> bool:
    for path in study_dir.rglob("*.json"):
        parsed = parse_canonical(path.stem)
        if parsed is not None and parsed[2] == side:
            return True
    return False


def infer_sample(
    annotation_path: Path,
    annotation: dict,
    batch_dir: Path,
) -> tuple[tuple[str, str, str, str, str] | None, int, str]:
    json_side = str(annotation.get("side", "")).upper()
    parsed = parse_canonical(annotation_path.stem, json_side)
    if parsed is not None:
        return parsed, 1, "filename"

    current = annotation_path.parent
    study_dir: Path | None = None
    while current != batch_dir and batch_dir in current.parents:
        parsed = parse_canonical(current.name, json_side)
        if parsed is not None:
            return parsed, 2, "parent_folder"
        if study_dir is None and study_context(current.name) is not None:
            study_dir = current
        current = current.parent

    if study_dir is None or explicit_json_for_side(study_dir, json_side):
        return None, 99, "unclassified"
    sibling_candidates = {
        candidate
        for image_path in study_dir.glob("*.jpg")
        if (candidate := parse_canonical(image_path.stem, json_side)) is not None
        and candidate[2] == json_side
    }
    if len(sibling_candidates) == 1:
        return next(iter(sibling_candidates)), 3, "sibling_derivative"
    return None, 99, "unclassified"


def load_overrides(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = read_json(path)
    if "overrides" in payload:
        payload = payload["overrides"]
    if not isinstance(payload, dict):
        raise ValueError("Overrides JSON must contain an object or an 'overrides' object")
    overrides: dict[str, dict[str, object]] = {}
    allowed_fields = {
        "sample_id",
        "side",
        "case_id",
        "implant_status",
        "study_phase",
        "skip_reason",
        "reason",
        "replace_existing_sample_id",
        "replace_existing_sample_dir",
        "allow_raw_replacement",
    }
    for relative_path, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Override for {relative_path} must be an object")
        unknown_fields = set(value) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unsupported override fields for {relative_path}: {sorted(unknown_fields)}")
        normalized_path = Path(str(relative_path)).as_posix().lstrip("./")
        normalized = dict(value)
        if "sample_id" in normalized and not is_safe_sample_id(normalized["sample_id"]):
            raise ValueError(f"Unsafe sample_id override for {relative_path}")
        if "replace_existing_sample_id" in normalized and not is_safe_sample_id(
            normalized["replace_existing_sample_id"]
        ):
            raise ValueError(f"Unsafe replace_existing_sample_id override for {relative_path}")
        if "replace_existing_sample_dir" in normalized:
            normalized["replace_existing_sample_dir"] = normalize_safe_relative_path(
                normalized["replace_existing_sample_dir"],
                field_name="replace_existing_sample_dir",
            )
        if "side" in normalized:
            normalized["side"] = str(normalized["side"]).upper()
            if normalized["side"] not in {"L", "R"}:
                raise ValueError(f"Invalid side override for {relative_path}")
        if "implant_status" in normalized:
            status = str(normalized["implant_status"])
            normalized["implant_status"] = "TKA" if status.lower() == "tka" else status.lower()
            if normalized["implant_status"] not in {"bone", "TKA"}:
                raise ValueError(f"Invalid implant_status override for {relative_path}")
        if "study_phase" in normalized:
            normalized["study_phase"] = str(normalized["study_phase"])
            if not SAFE_METADATA_RE.fullmatch(normalized["study_phase"]):
                raise ValueError(f"Invalid study_phase override for {relative_path}")
        if "allow_raw_replacement" in normalized:
            if not isinstance(normalized["allow_raw_replacement"], bool):
                raise ValueError(f"allow_raw_replacement must be boolean for {relative_path}")
            if normalized["allow_raw_replacement"] and not (
                normalized.get("replace_existing_sample_id")
                or normalized.get("replace_existing_sample_dir")
            ):
                raise ValueError(
                    f"allow_raw_replacement requires an explicit replacement target for {relative_path}"
                )
        overrides[normalized_path] = normalized
    return overrides


def existing_case_ids(dataset_root: Path) -> dict[str, str]:
    by_case: dict[str, set[str]] = defaultdict(set)
    for folder_name in DATASET_FOLDERS.values():
        manifest_path = dataset_root / folder_name / "manifest.csv"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_dataset") == "legacy":
                    continue
                match = re.match(r"(?P<case>\d+)", row.get("sample_id", ""))
                if match and row.get("case_id"):
                    by_case[match.group("case")].add(row["case_id"])
    return {
        case_number: next(iter(case_ids))
        for case_number, case_ids in by_case.items()
        if len(case_ids) == 1
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_training_payload(annotation: dict, *, restore_crop: bool = False) -> dict:
    x_offset = 0.0
    y_offset = 0.0
    image_width = int(annotation["image_width"])
    image_height = int(annotation["image_height"])
    if restore_crop:
        processed_from = annotation.get("processed_from", {})
        crop = processed_from.get("crop", {}) if isinstance(processed_from, dict) else {}
        x_offset = float(crop.get("x0", 0.0))
        y_offset = float(crop.get("y0", 0.0))
        image_width = int(processed_from.get("original_image_width", image_width))
        image_height = int(processed_from.get("original_image_height", image_height))

    def shifted(point: dict) -> dict[str, float]:
        return {
            "x": round(float(point["x"]) + x_offset, 8),
            "y": round(float(point["y"]) + y_offset, 8),
        }

    points = {
        name: shifted(point)
        for name, point in sorted(annotation.get("points", {}).items())
    }
    lines = {
        line_name: {
            endpoint: shifted(point)
            for endpoint, point in sorted(line.items())
        }
        for line_name, line in sorted(annotation.get("lines", {}).items())
    }
    return {
        "image_width": image_width,
        "image_height": image_height,
        "side": str(annotation.get("side", "")),
        "points": points,
        "lines": lines,
    }


def payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decision(
    annotation_path: Path,
    *,
    sample_id: str = "",
    case_id: str = "",
    implant_status: str = "",
    side: str = "",
    raw_path: Path | None = None,
    action: str,
    reason: str,
) -> dict[str, str]:
    return {
        "annotation_path": annotation_path.as_posix(),
        "sample_id": sample_id,
        "case_id": case_id,
        "implant_status": implant_status,
        "side": side,
        "raw_path": "" if raw_path is None else raw_path.as_posix(),
        "action": action,
        "reason": reason,
    }


def collect_candidates(
    batch_dir: Path,
    dataset_root: Path,
    overrides: dict[str, dict[str, object]],
) -> tuple[list[Candidate], list[dict[str, str]]]:
    raw_index = build_raw_image_index([batch_dir])
    case_id_map = existing_case_ids(dataset_root)
    candidates: list[Candidate] = []
    decisions: list[dict[str, str]] = []
    seen_override_paths: set[str] = set()

    for annotation_path in sorted(batch_dir.rglob("*.json")):
        relative_path = annotation_path.relative_to(batch_dir).as_posix()
        try:
            original_annotation = read_json(annotation_path)
        except Exception as exc:
            decisions.append(decision(annotation_path, action="quarantine", reason=f"invalid_json:{exc}"))
            continue

        override = overrides.get(relative_path, {})
        if override:
            seen_override_paths.add(relative_path)
        try:
            adapted = adapt_annotation_schema(original_annotation)
        except Exception as exc:
            decisions.append(
                decision(annotation_path, action="quarantine", reason=f"invalid_annotation_schema:{exc}")
            )
            continue
        if adapted is None:
            decisions.append(decision(annotation_path, action="ignored", reason="not_annotation_schema"))
            continue
        if override.get("skip_reason"):
            decisions.append(
                decision(annotation_path, action="skipped", reason=f"override_skip:{override['skip_reason']}")
            )
            continue

        annotation, expected_raw_sha256 = adapted
        corrections: dict[str, object] = {}
        if override.get("side"):
            corrections["side"] = {"from": annotation.get("side"), "to": str(override["side"]).upper()}
            annotation["side"] = str(override["side"]).upper()

        inferred, inference_rank, inference_reason = infer_sample(annotation_path, annotation, batch_dir)
        if override.get("sample_id"):
            sample_id = str(override["sample_id"])
            parsed_override = parse_canonical(sample_id, str(annotation.get("side", "")))
            fallback = parsed_override or inferred
            case_number = case_number_from_sample_id(sample_id)
            sample_side = fallback[2] if fallback is not None else str(annotation.get("side", "")).upper()
            phase = str(override.get("study_phase") or (fallback[3] if fallback is not None else ""))
            implant_status = str(
                override.get("implant_status") or (fallback[4] if fallback is not None else "")
            )
            inference_rank = 0
            inference_reason = "override"
            corrections["sample_id"] = {"from": annotation_path.stem, "to": sample_id}
        else:
            if inferred is None:
                decisions.append(decision(annotation_path, action="skipped", reason="unknown_implant_or_sample_id"))
                continue
            sample_id, case_number, sample_side, phase, implant_status = inferred
            if override.get("study_phase"):
                corrections["study_phase"] = {"from": phase, "to": str(override["study_phase"])}
                phase = str(override["study_phase"])
            if override.get("implant_status"):
                corrections["implant_status"] = {
                    "from": implant_status,
                    "to": str(override["implant_status"]),
                }
                implant_status = str(override["implant_status"])

        if not sample_side or not phase or not implant_status:
            decisions.append(
                decision(
                    annotation_path,
                    sample_id=sample_id,
                    implant_status=implant_status,
                    side=str(annotation.get("side", "")),
                    action="quarantine",
                    reason="incomplete_sample_metadata_override",
                )
            )
            continue
        if implant_status not in {"bone", "TKA"}:
            decisions.append(
                decision(
                    annotation_path,
                    sample_id=sample_id,
                    implant_status=implant_status,
                    side=str(annotation.get("side", "")),
                    action="skipped",
                    reason=f"unsupported_implant_status:{implant_status}",
                )
            )
            continue
        annotation_side = str(annotation.get("side", "")).upper()
        if annotation_side != sample_side:
            decisions.append(
                decision(
                    annotation_path,
                    sample_id=sample_id,
                    implant_status=implant_status,
                    side=annotation_side,
                    action="quarantine",
                    reason=f"side_mismatch:sample={sample_side},annotation={annotation_side}",
                )
            )
            continue

        source_context = study_context(str(annotation.get("source_raw_path", ""))) or study_context(
            str(annotation.get("raw_path", ""))
        )
        local_context = nearest_local_context(annotation_path, batch_dir)
        authoritative_context = source_context or local_context
        if phase in {"pre", "post"} and authoritative_context and authoritative_context != (case_number, phase):
            decisions.append(
                decision(
                    annotation_path,
                    sample_id=sample_id,
                    implant_status=implant_status,
                    side=annotation_side,
                    action="quarantine",
                    reason=(
                        f"case_phase_mismatch:sample={case_number}/{phase},"
                        f"source={authoritative_context[0]}/{authoritative_context[1]}"
                    ),
                )
            )
            continue

        try:
            raw_candidate, raw_match_count = resolve_raw_candidate(annotation_path, annotation, raw_index)
            resolved_context = study_context(raw_candidate.path.as_posix())
            if (
                phase in {"pre", "post"}
                and source_context is None
                and resolved_context
                and resolved_context != (case_number, phase)
            ):
                raise ValueError(
                    "resolved_raw_case_phase_mismatch:"
                    f"sample={case_number}/{phase},raw={resolved_context[0]}/{resolved_context[1]}"
                )
            raw_image = cv2.imread(str(raw_candidate.path), cv2.IMREAD_COLOR)
            if raw_image is None:
                raise ValueError(f"cannot_read_raw:{raw_candidate.path}")
            image_height, image_width = raw_image.shape[:2]
            if (image_width, image_height) != (
                int(annotation["image_width"]),
                int(annotation["image_height"]),
            ):
                raise ValueError("annotation_raw_size_mismatch")
            raw_sha = sha256_file(raw_candidate.path)
            if expected_raw_sha256 and raw_sha != expected_raw_sha256:
                raise ValueError(
                    f"raw_sha256_mismatch:expected={expected_raw_sha256},actual={raw_sha}"
                )
            keypoints = annotation_keypoints(annotation)
            if any(
                not (math.isfinite(x) and math.isfinite(y) and 0 <= x < image_width and 0 <= y < image_height)
                for x, y in keypoints.values()
            ):
                raise ValueError("invalid_or_out_of_bounds_keypoint")
            measured, _debug = measure_from_named_points(
                raw_image,
                annotation["points"],
                raw_path=raw_candidate.path,
                named_lines=annotation.get("lines"),
                side=annotation_side,
            )
            mldfa = float(measured["mldfa_angle"])
            mpta = float(measured["mpta_angle"])
            if not math.isfinite(mldfa) or not math.isfinite(mpta):
                raise ValueError("non_finite_measurement")
        except Exception as exc:
            decisions.append(
                decision(
                    annotation_path,
                    sample_id=sample_id,
                    implant_status=implant_status,
                    side=annotation_side,
                    action="quarantine",
                    reason=f"invalid_annotation_or_raw:{exc}",
                )
            )
            continue

        case_id = str(override.get("case_id") or case_id_map.get(case_number) or f"{batch_dir.name}:{case_number}")
        if override.get("reason"):
            corrections["review_reason"] = str(override["reason"])
        candidates.append(
            Candidate(
                annotation_path=annotation_path,
                relative_path=relative_path,
                sample_id=sample_id,
                case_number=case_number,
                side=annotation_side,
                phase=phase,
                implant_status=implant_status,
                case_id=case_id,
                annotation=annotation,
                raw_path=raw_candidate.path,
                raw_match_count=raw_match_count,
                raw_sha256=raw_sha,
                source_json_sha256=sha256_file(annotation_path),
                mldfa=mldfa,
                mpta=mpta,
                inference_rank=inference_rank,
                replace_existing_sample_id=str(override.get("replace_existing_sample_id") or ""),
                replace_existing_sample_dir=str(override.get("replace_existing_sample_dir") or ""),
                allow_raw_replacement=bool(override.get("allow_raw_replacement", False)),
                corrections=corrections,
            )
        )
        decisions.append(
            decision(
                annotation_path,
                sample_id=sample_id,
                case_id=case_id,
                implant_status=implant_status,
                side=annotation_side,
                raw_path=raw_candidate.path,
                action="candidate",
                reason=inference_reason,
            )
        )
    unused_overrides = sorted(set(overrides) - seen_override_paths)
    if unused_overrides:
        raise ValueError(f"Overrides did not match annotation files: {unused_overrides}")
    return candidates, decisions


def update_decision(
    decisions: list[dict[str, str]],
    relative_or_absolute_path: str,
    *,
    batch_dir: Path,
    action: str,
    reason: str,
) -> None:
    path = Path(relative_or_absolute_path)
    absolute = path if path.is_absolute() else batch_dir / path
    target = absolute.as_posix()
    for row in decisions:
        if Path(row["annotation_path"]).as_posix() == target:
            row["action"] = action
            row["reason"] = reason
            return


def confirmed_bilateral_screen_side(candidate: Candidate) -> str | None:
    annotation = candidate.annotation
    if source_input_scope(annotation) != "bilateral raster x-ray":
        return None
    if str(annotation.get("side", "")).upper() != candidate.side:
        return None
    try:
        image_width = int(annotation["image_width"])
        image_height = int(annotation["image_height"])
        crop, status = confirmed_inference_roi(annotation, image_width, image_height)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if crop is None or status != "accepted":
        return None
    if crop.x0 == 0 and crop.x1 < image_width:
        return "left"
    if crop.x0 > 0 and crop.x1 == image_width:
        return "right"
    return None


def valid_bilateral_lr_pair(candidates: list[Candidate]) -> bool:
    if len(candidates) != 2:
        return False
    if len({candidate.raw_sha256 for candidate in candidates}) != 1:
        return False
    if len({candidate.case_number for candidate in candidates}) != 1:
        return False
    if len({candidate.case_id for candidate in candidates}) != 1:
        return False
    if len({candidate.phase for candidate in candidates}) != 1:
        return False
    if {candidate.side for candidate in candidates} != {"L", "R"}:
        return False
    return {
        confirmed_bilateral_screen_side(candidate)
        for candidate in candidates
    } == {"left", "right"}


def deduplicate_candidates(
    candidates: list[Candidate],
    decisions: list[dict[str, str]],
    batch_dir: Path,
) -> list[Candidate]:
    by_content: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_content[(candidate.raw_sha256, candidate.source_json_sha256)].append(candidate)

    unique: list[Candidate] = []
    for group in by_content.values():
        ordered = sorted(
            group,
            key=lambda item: (
                not item.has_explicit_replacement,
                item.inference_rank,
                item.relative_path,
            ),
        )
        sample_ids = {item.sample_id for item in ordered}
        if len(sample_ids) > 1:
            paths = ";".join(item.relative_path for item in ordered)
            for conflict in ordered:
                update_decision(
                    decisions,
                    conflict.relative_path,
                    batch_dir=batch_dir,
                    action="quarantine",
                    reason=f"exact_content_has_conflicting_sample_ids:{paths}",
                )
            continue
        kept = ordered[0]
        unique.append(kept)
        for duplicate in ordered[1:]:
            update_decision(
                decisions,
                duplicate.relative_path,
                batch_dir=batch_dir,
                action="skip_batch_duplicate",
                reason=f"same_raw_and_annotation_as:{kept.relative_path}",
            )

    by_raw: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in unique:
        by_raw[candidate.raw_sha256].append(candidate)
    raw_unique: list[Candidate] = []
    for raw_sha256, group in by_raw.items():
        if len(group) == 1:
            raw_unique.extend(group)
            continue
        if valid_bilateral_lr_pair(group):
            raw_unique.extend(group)
            continue
        paths = ";".join(sorted(item.relative_path for item in group))
        for candidate in group:
            if candidate.has_explicit_replacement:
                raw_unique.append(candidate)
                continue
            update_decision(
                decisions,
                candidate.relative_path,
                batch_dir=batch_dir,
                action="quarantine",
                reason=f"batch_raw_collision_requires_explicit_replacement:{raw_sha256}:{paths}",
            )

    by_sample: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in raw_unique:
        by_sample[candidate.sample_id].append(candidate)
    accepted: list[Candidate] = []
    for sample_id, group in by_sample.items():
        if len(group) == 1:
            accepted.extend(group)
            continue
        paths = ";".join(sorted(item.relative_path for item in group))
        for conflict in group:
            update_decision(
                decisions,
                conflict.relative_path,
                batch_dir=batch_dir,
                action="quarantine",
                reason=f"sample_id_content_conflict:{sample_id}:{paths}",
            )
    return sorted(accepted, key=lambda item: item.sample_id)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare_processing_manifest(
    candidates: list[Candidate],
    batch_dir: Path,
    work_dir: Path,
) -> tuple[Path, dict[str, Candidate]]:
    annotations_dir = work_dir / "annotations"
    raw_groups_dir = work_dir / "raw_groups"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    raw_groups_dir.mkdir(parents=True, exist_ok=True)

    raw_group_paths: dict[tuple[str, str, str], Path] = {}
    rows: list[dict[str, str]] = []
    by_sample: dict[str, Candidate] = {}
    for candidate in candidates:
        group_key = (candidate.case_number, candidate.phase, candidate.raw_sha256)
        raw_group_path = raw_group_paths.get(group_key)
        if raw_group_path is None:
            suffix = candidate.raw_path.suffix.lower() or ".jpg"
            group_digest = hashlib.sha256("|".join(group_key).encode("utf-8")).hexdigest()[:16]
            raw_group_path = raw_groups_dir / f"{candidate.case_number}_{candidate.phase}_{group_digest}{suffix}"
            shutil.copy2(candidate.raw_path, raw_group_path)
            raw_group_paths[group_key] = raw_group_path

        normalized_annotation = copy.deepcopy(candidate.annotation)
        if normalized_annotation.get("source_raw_path"):
            normalized_annotation["submitted_source_raw_path"] = normalized_annotation["source_raw_path"]
        if normalized_annotation.get("raw_path"):
            normalized_annotation["submitted_raw_path"] = normalized_annotation["raw_path"]
        normalized_annotation["source_annotation_path"] = candidate.annotation_path.as_posix()
        normalized_annotation["source_raw_path"] = candidate.raw_path.as_posix()
        normalized_annotation["source_dataset"] = batch_dir.name
        if candidate.corrections:
            normalized_annotation["import_corrections"] = candidate.corrections
        annotation_copy_path = annotations_dir / f"{candidate.sample_id}_annotation.json"
        write_json(annotation_copy_path, normalized_annotation)

        raw_image = cv2.imread(str(raw_group_path), cv2.IMREAD_COLOR)
        if raw_image is None:
            raise ValueError(f"Cannot read staged raw image: {raw_group_path}")
        image_height, image_width = raw_image.shape[:2]
        rows.append(
            {
                "sample_id": candidate.sample_id,
                "case_id": candidate.case_id,
                "source_dataset": batch_dir.name,
                "dataset_group": batch_dir.name,
                "implant_status": candidate.implant_status,
                "study_phase": candidate.phase,
                "side": candidate.side,
                "annotation_path": annotation_copy_path.as_posix(),
                "raw_path": raw_group_path.as_posix(),
                "raw_filename": raw_group_path.name,
                "image_width": str(image_width),
                "image_height": str(image_height),
                "raw_match_count": str(candidate.raw_match_count),
                "mldfa": f"{candidate.mldfa:.6f}",
                "mpta": f"{candidate.mpta:.6f}",
            }
        )
        by_sample[candidate.sample_id] = candidate

    manifest_path = work_dir / "reviewed_input_manifest.csv"
    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    return manifest_path, by_sample


def rewrite_processed_provenance(processed_manifest: Path, by_sample: dict[str, Candidate]) -> list[dict[str, str]]:
    fieldnames, _raw_rows = raw_manifest(processed_manifest)
    rows = load_manifest(processed_manifest)
    for row in rows:
        candidate = by_sample[row["sample_id"]]
        annotation_path = Path(row["annotation_path"])
        annotation = read_json(annotation_path)
        annotation["source_annotation_path"] = candidate.annotation_path.as_posix()
        annotation["source_raw_path"] = candidate.raw_path.as_posix()
        annotation["source_dataset"] = row["source_dataset"]
        if candidate.corrections:
            annotation["import_corrections"] = candidate.corrections
        processed_from = annotation.get("processed_from")
        if isinstance(processed_from, dict):
            processed_from["annotation_path"] = candidate.annotation_path.as_posix()
            processed_from["raw_path"] = candidate.raw_path.as_posix()
            processed_from["source_annotation_sha256"] = candidate.source_json_sha256
            processed_from["source_raw_sha256"] = candidate.raw_sha256
        write_json(annotation_path, annotation)
        row["source_annotation_path"] = candidate.annotation_path.as_posix()
        row["source_raw_path"] = candidate.raw_path.as_posix()
        if "source_annotation_sha256" in row:
            row["source_annotation_sha256"] = candidate.source_json_sha256
        if "source_raw_sha256" in row:
            row["source_raw_sha256"] = candidate.raw_sha256
        row["raw_match_count"] = str(candidate.raw_match_count)

    write_csv(processed_manifest, rows, fieldnames)
    return rows


def existing_samples(dataset_root: Path) -> list[ExistingSample]:
    result: list[ExistingSample] = []
    for folder_name in DATASET_FOLDERS.values():
        manifest_path = dataset_root / folder_name / "manifest.csv"
        if not manifest_path.exists():
            continue
        for row in load_manifest(manifest_path):
            result.append(ExistingSample(folder_name=folder_name, row=row))
    return result


def existing_rows_by_sample(dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in existing_samples(dataset_root):
        result[sample.row["sample_id"]].append(sample.row)
    return result


def replacement_target(candidate: Candidate, samples: list[ExistingSample]) -> list[ExistingSample]:
    if not candidate.has_explicit_replacement:
        return []
    matches: list[ExistingSample] = []
    for sample in samples:
        if (
            candidate.replace_existing_sample_id
            and sample.row["sample_id"] != candidate.replace_existing_sample_id
        ):
            continue
        if candidate.replace_existing_sample_dir:
            requested_dir = candidate.replace_existing_sample_dir
            if requested_dir not in {sample.row["sample_dir"], sample.locator}:
                continue
        matches.append(sample)
    return matches


def existing_sample_raw_hashes(sample: ExistingSample) -> set[str]:
    hashes: set[str] = set()
    for field_name in ("raw_path", "source_raw_path"):
        value = sample.row.get(field_name, "")
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            hashes.add(sha256_file(path))
    return hashes


def processed_candidate_raw_hashes(row: dict[str, str], candidate: Candidate) -> set[str]:
    return {candidate.raw_sha256, sha256_file(Path(row["raw_path"]))}


def filter_existing_collisions(
    processed_rows: list[dict[str, str]],
    by_sample: dict[str, Candidate],
    dataset_root: Path,
    decisions: list[dict[str, str]],
    batch_dir: Path,
) -> tuple[list[dict[str, str]], list[ReplacementPlan]]:
    candidates_by_sample = by_sample
    samples = existing_samples(dataset_root)
    existing_by_sample: dict[str, list[ExistingSample]] = defaultdict(list)
    raw_hashes_by_locator: dict[str, set[str]] = {}
    for sample in samples:
        existing_by_sample[sample.row["sample_id"]].append(sample)
        raw_hashes_by_locator[sample.locator] = existing_sample_raw_hashes(sample)

    candidate_raw_hashes: dict[str, set[str]] = {}
    candidate_raw_collisions: dict[str, list[ExistingSample]] = {}
    replacement_targets: dict[str, ExistingSample] = {}
    replacement_errors: dict[str, str] = {}
    replacements_by_target: dict[str, list[str]] = defaultdict(list)
    for row in processed_rows:
        sample_id = row["sample_id"]
        candidate = candidates_by_sample[sample_id]
        new_raw_hashes = processed_candidate_raw_hashes(row, candidate)
        candidate_raw_hashes[sample_id] = new_raw_hashes
        candidate_raw_collisions[sample_id] = [
            sample
            for sample in samples
            if new_raw_hashes & raw_hashes_by_locator[sample.locator]
        ]
        if not candidate.has_explicit_replacement:
            continue
        targets = replacement_target(candidate, samples)
        if len(targets) != 1:
            replacement_errors[sample_id] = (
                "replacement_target_not_unique:"
                f"sample_id={candidate.replace_existing_sample_id or '<any>'},"
                f"sample_dir={candidate.replace_existing_sample_dir or '<any>'},"
                f"matches={len(targets)}"
            )
            continue
        target = targets[0]
        replacement_targets[sample_id] = target
        replacements_by_target[target.locator].append(sample_id)

    for locator, sample_ids in replacements_by_target.items():
        if len(sample_ids) > 1:
            for sample_id in sample_ids:
                replacement_errors[sample_id] = (
                    f"replacement_target_claimed_multiple_times:{locator}:"
                    + ";".join(sorted(sample_ids))
                )
    for sample_id, target in replacement_targets.items():
        if sample_id in replacement_errors:
            continue
        candidate = candidates_by_sample[sample_id]
        same_raw = bool(candidate_raw_hashes[sample_id] & raw_hashes_by_locator[target.locator])
        if not same_raw and not candidate.allow_raw_replacement:
            replacement_errors[sample_id] = (
                f"replacement_raw_mismatch_requires_allow_raw_replacement:{target.locator}"
            )

    active_replacements = set(replacement_targets) - set(replacement_errors)
    while True:
        active_target_locators = {
            replacement_targets[sample_id].locator for sample_id in active_replacements
        }
        newly_invalid: dict[str, str] = {}
        for sample_id in active_replacements:
            target = replacement_targets[sample_id]
            id_conflicts = [
                sample
                for sample in existing_by_sample.get(sample_id, [])
                if sample.locator != target.locator
                and sample.locator not in active_target_locators
            ]
            if id_conflicts:
                newly_invalid[sample_id] = (
                    "replacement_would_leave_duplicate_sample_id:"
                    + ";".join(sorted(sample.locator for sample in id_conflicts))
                )
                continue
            outside_raw_collisions = [
                sample
                for sample in candidate_raw_collisions[sample_id]
                if sample.locator != target.locator
                and sample.locator not in active_target_locators
            ]
            if outside_raw_collisions:
                newly_invalid[sample_id] = (
                    "replacement_would_leave_existing_raw_collision:"
                    + ";".join(sorted(sample.locator for sample in outside_raw_collisions))
                )
        if not newly_invalid:
            break
        replacement_errors.update(newly_invalid)
        active_replacements.difference_update(newly_invalid)

    accepted: list[dict[str, str]] = []
    replacements: list[ReplacementPlan] = []
    for row in processed_rows:
        sample_id = row["sample_id"]
        candidate = candidates_by_sample[sample_id]
        new_annotation = read_json(Path(row["annotation_path"]))
        new_signature = payload_sha256(normalized_training_payload(new_annotation, restore_crop=True))
        new_raw_hashes = candidate_raw_hashes[sample_id]
        raw_collisions = candidate_raw_collisions[sample_id]

        if candidate.has_explicit_replacement:
            if sample_id in replacement_errors:
                update_decision(
                    decisions,
                    candidate.relative_path,
                    batch_dir=batch_dir,
                    action="quarantine",
                    reason=replacement_errors[sample_id],
                )
                continue
            target = replacement_targets[sample_id]
            same_raw = bool(new_raw_hashes & raw_hashes_by_locator[target.locator])
            accepted.append(row)
            replacements.append(
                ReplacementPlan(
                    old_folder_name=target.folder_name,
                    old_sample_id=target.row["sample_id"],
                    old_sample_dir=target.row["sample_dir"],
                    new_sample_id=sample_id,
                )
            )
            update_decision(
                decisions,
                candidate.relative_path,
                batch_dir=batch_dir,
                action="ready_to_import",
                reason=(
                    f"validated_replacement_for:{target.locator}:"
                    + ("raw_replacement_allowed" if not same_raw else "same_raw")
                ),
            )
            continue

        sample_id_collisions = existing_by_sample.get(sample_id, [])
        if len(sample_id_collisions) == 1:
            existing_sample = sample_id_collisions[0]
            existing_annotation = read_json(Path(existing_sample.row["annotation_path"]))
            existing_signature = payload_sha256(
                normalized_training_payload(existing_annotation, restore_crop=True)
            )
            if (
                new_signature == existing_signature
                and new_raw_hashes & raw_hashes_by_locator[existing_sample.locator]
            ):
                update_decision(
                    decisions,
                    candidate.relative_path,
                    batch_dir=batch_dir,
                    action="skip_existing_duplicate",
                    reason=f"semantic_annotation_already_exists:{sample_id}",
                )
                continue
        if sample_id_collisions:
            update_decision(
                decisions,
                candidate.relative_path,
                batch_dir=batch_dir,
                action="quarantine",
                reason=(
                    f"existing_sample_id_content_conflict:{sample_id}:"
                    + ";".join(sorted(sample.locator for sample in sample_id_collisions))
                ),
            )
            continue
        if raw_collisions:
            update_decision(
                decisions,
                candidate.relative_path,
                batch_dir=batch_dir,
                action="quarantine",
                reason=(
                    "existing_raw_collision_requires_explicit_replacement:"
                    + ";".join(sorted(sample.locator for sample in raw_collisions))
                ),
            )
            continue
        accepted.append(row)
        update_decision(
            decisions,
            candidate.relative_path,
            batch_dir=batch_dir,
            action="ready_to_import",
            reason="validated_new_sample",
        )
    return accepted, replacements


def validate_dataset_root(dataset_root: Path) -> list[str]:
    errors: list[str] = []
    expected_files = {"annotation.json", "raw.jpg", "point.jpg", "line.jpg", "combined.jpg"}
    sample_id_locations: dict[str, list[str]] = defaultdict(list)
    raw_sha_locations: dict[str, list[str]] = defaultdict(list)
    for folder_name in DATASET_FOLDERS.values():
        manifest_path = dataset_root / folder_name / "manifest.csv"
        manifest_errors, _counters = validate_manifest(manifest_path)
        errors.extend(f"{folder_name}:{error}" for error in manifest_errors)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            locator = f"{folder_name}/{row['sample_dir']}"
            sample_id_locations[row["sample_id"]].append(locator)
            sample_dir = dataset_root / folder_name / row["sample_dir"]
            actual_files = {path.name for path in sample_dir.iterdir() if path.is_file()} if sample_dir.exists() else set()
            if actual_files != expected_files:
                errors.append(
                    f"{folder_name}:{row['sample_id']}:expected_files={sorted(expected_files)},"
                    f"actual_files={sorted(actual_files)}"
                )
                continue
            for filename in expected_files:
                if (sample_dir / filename).stat().st_size == 0:
                    errors.append(f"{folder_name}:{row['sample_id']}:empty_file={filename}")
            raw_path = Path(row.get("raw_path", ""))
            if not raw_path.is_absolute():
                raw_path = dataset_root / folder_name / raw_path
            if raw_path.is_file():
                raw_sha_locations[sha256_file(raw_path)].append(locator)
    for sample_id, locations in sorted(sample_id_locations.items()):
        if len(locations) > 1:
            errors.append(f"duplicate_sample_id:{sample_id}:{';'.join(sorted(locations))}")
    for raw_sha256, locations in sorted(raw_sha_locations.items()):
        if len(locations) > 1:
            errors.append(f"duplicate_raw_sha256:{raw_sha256}:{';'.join(sorted(locations))}")
    return errors


def raw_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def merged_fieldnames(left: list[str], right: list[str]) -> list[str]:
    result = list(left)
    result.extend(name for name in right if name not in result)
    return result


def normalize_replacement_plans(
    dataset_root: Path,
    replacements: dict[str, str] | list[ReplacementPlan],
) -> list[ReplacementPlan]:
    if isinstance(replacements, list):
        plans = list(replacements)
    else:
        samples = existing_samples(dataset_root)
        plans = []
        for old_sample_id, new_sample_id in replacements.items():
            matches = [sample for sample in samples if sample.row["sample_id"] == old_sample_id]
            if len(matches) != 1:
                raise ValueError(
                    f"Legacy replacement target must be unique: {old_sample_id}:matches={len(matches)}"
                )
            target = matches[0]
            plans.append(
                ReplacementPlan(
                    old_folder_name=target.folder_name,
                    old_sample_id=old_sample_id,
                    old_sample_dir=target.row["sample_dir"],
                    new_sample_id=new_sample_id,
                )
            )
    locators = [plan.locator for plan in plans]
    if len(locators) != len(set(locators)):
        raise ValueError(f"Duplicate replacement targets: {sorted(locators)}")
    return plans


def merge_stage(
    stage_root: Path,
    dataset_root: Path,
    *,
    replacements: dict[str, str] | list[ReplacementPlan],
    replacement_backup_dir: Path,
) -> dict[str, int]:
    replacement_plans = normalize_replacement_plans(dataset_root, replacements)
    plans_by_locator = {plan.locator: plan for plan in replacement_plans}
    replacement_record_path = replacement_backup_dir / "replacements.json"
    if replacement_plans and replacement_record_path.exists():
        raise FileExistsError(f"Replacement record already exists: {replacement_record_path}")

    copied_dirs: list[Path] = []
    moved_replacements: list[tuple[Path, Path]] = []
    replacement_records: list[dict[str, str]] = []
    replacement_record_written = False
    original_manifests: dict[Path, bytes] = {}
    pending: list[tuple[Path, Path, list[dict[str, str]], list[str]]] = []
    added_counts: dict[str, int] = {}

    try:
        for folder_name in DATASET_FOLDERS.values():
            source_dir = stage_root / folder_name
            target_dir = dataset_root / folder_name
            source_fields, source_rows = raw_manifest(source_dir / "manifest.csv")
            target_manifest = target_dir / "manifest.csv"
            target_fields, target_rows = raw_manifest(target_manifest)
            original_manifests[target_manifest] = target_manifest.read_bytes()
            rows_to_replace = [
                row
                for row in target_rows
                if f"{folder_name}/{row['sample_dir']}" in plans_by_locator
            ]
            for old_row in rows_to_replace:
                locator = f"{folder_name}/{old_row['sample_dir']}"
                plan = plans_by_locator[locator]
                if old_row["sample_id"] != plan.old_sample_id:
                    raise ValueError(
                        f"Replacement target changed before merge: {locator}:"
                        f"expected={plan.old_sample_id},found={old_row['sample_id']}"
                    )
                old_sample_dir = target_dir / old_row["sample_dir"]
                backup_sample_dir = replacement_backup_dir / folder_name / old_row["sample_dir"]
                if not old_sample_dir.exists():
                    raise FileNotFoundError(f"Replacement source directory is missing: {old_sample_dir}")
                if backup_sample_dir.exists():
                    raise FileExistsError(f"Replacement backup already exists: {backup_sample_dir}")
                backup_sample_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old_sample_dir, backup_sample_dir)
                moved_replacements.append((backup_sample_dir, old_sample_dir))
                replacement_records.append(
                    {
                        "old_sample_id": old_row["sample_id"],
                        "new_sample_id": plan.new_sample_id,
                        "old_folder_name": folder_name,
                        "old_sample_dir_relative": old_row["sample_dir"],
                        "old_sample_dir": old_sample_dir.as_posix(),
                        "backup_sample_dir": backup_sample_dir.as_posix(),
                    }
                )
            target_rows = [
                row
                for row in target_rows
                if f"{folder_name}/{row['sample_dir']}" not in plans_by_locator
            ]
            for row in source_rows:
                source_sample_dir = source_dir / row["sample_dir"]
                target_sample_dir = target_dir / row["sample_dir"]
                if target_sample_dir.exists():
                    raise FileExistsError(f"Refusing to overwrite existing sample directory: {target_sample_dir}")
                copied_dirs.append(target_sample_dir)
                shutil.copytree(source_sample_dir, target_sample_dir)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".manifest.csv.importing-",
                dir=target_dir,
            )
            os.close(descriptor)
            temporary_manifest = Path(temporary_name)
            pending.append(
                (
                    target_manifest,
                    temporary_manifest,
                    [*target_rows, *source_rows],
                    merged_fieldnames(target_fields, source_fields),
                )
            )
            added_counts[folder_name] = len(source_rows)

        for _target_manifest, temporary_manifest, rows, fieldnames in pending:
            write_csv(temporary_manifest, rows, fieldnames)
        for target_manifest, temporary_manifest, _rows, _fieldnames in pending:
            os.replace(temporary_manifest, target_manifest)
        errors = validate_dataset_root(dataset_root)
        if errors:
            raise ValueError("Merged dataset validation failed: " + " | ".join(errors[:20]))
        found_replacements = {
            f"{row['old_folder_name']}/{row['old_sample_dir_relative']}"
            for row in replacement_records
        }
        if found_replacements != set(plans_by_locator):
            raise ValueError(
                "Replacement targets mismatch: "
                f"expected={sorted(plans_by_locator)},found={sorted(found_replacements)}"
            )
        if replacement_records:
            replacement_record_written = True
            write_json(replacement_record_path, replacement_records)
    except Exception:
        for target_manifest, content in original_manifests.items():
            descriptor, restore_name = tempfile.mkstemp(
                prefix=".manifest.csv.restore-",
                dir=target_manifest.parent,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
            os.replace(restore_name, target_manifest)
        for copied_dir in reversed(copied_dirs):
            if copied_dir.exists():
                shutil.rmtree(copied_dir)
        for backup_sample_dir, old_sample_dir in reversed(moved_replacements):
            if backup_sample_dir.exists():
                old_sample_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup_sample_dir, old_sample_dir)
        for _target_manifest, temporary_manifest, _rows, _fieldnames in pending:
            if temporary_manifest.exists():
                temporary_manifest.unlink()
        if replacement_record_written and replacement_record_path.exists():
            replacement_record_path.unlink()
        raise
    return added_counts


def write_reports(
    report_dir: Path,
    decisions: list[dict[str, str]],
    summary: dict,
) -> None:
    ordered = sorted(decisions, key=lambda row: row["annotation_path"])
    write_csv(report_dir / "decisions.csv", ordered, DECISION_FIELDS)
    write_json(report_dir / "summary.json", summary)


def import_batch(
    batch_dir: Path,
    dataset_root: Path,
    report_dir: Path,
    overrides_path: Path | None,
    *,
    apply: bool,
) -> dict:
    batch_dir = batch_dir.resolve()
    dataset_root = dataset_root.resolve()
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(overrides_path)
    candidates, decisions = collect_candidates(batch_dir, dataset_root, overrides)
    candidates = deduplicate_candidates(candidates, decisions, batch_dir)
    added_counts = {folder_name: 0 for folder_name in DATASET_FOLDERS.values()}
    replacements: list[ReplacementPlan] = []
    fatal_error = ""
    if candidates:
        try:
            with tempfile.TemporaryDirectory(prefix=f"knee-import-{batch_dir.name}-") as temporary:
                work_dir = Path(temporary)
                input_manifest, by_sample = prepare_processing_manifest(candidates, batch_dir, work_dir)
                processed_dir = work_dir / "processed"
                process_dataset(input_manifest, processed_dir, render_overlays=False)
                processed_manifest = processed_dir / "processed_manifest.csv"
                processed_rows = rewrite_processed_provenance(processed_manifest, by_sample)
                new_rows, replacements = filter_existing_collisions(
                    processed_rows,
                    by_sample,
                    dataset_root,
                    decisions,
                    batch_dir,
                )
                filtered_manifest = work_dir / "new_processed_manifest.csv"
                fieldnames, _raw_processed_rows = raw_manifest(processed_manifest)
                write_csv(filtered_manifest, new_rows, fieldnames)

                stage_root = work_dir / "organized"
                organize_dataset(filtered_manifest, stage_root, clean=True)
                stage_errors = validate_dataset_root(stage_root)
                if stage_errors:
                    raise ValueError("Staged dataset validation failed: " + " | ".join(stage_errors[:20]))

                if apply:
                    added_counts = merge_stage(
                        stage_root,
                        dataset_root,
                        replacements=replacements,
                        replacement_backup_dir=report_dir / "replaced_existing",
                    )
                    for row in decisions:
                        if row["action"] == "ready_to_import":
                            row["action"] = "imported"
        except Exception as exc:
            fatal_error = f"{type(exc).__name__}:{exc}"

    counts = Counter(row["action"] for row in decisions)
    summary = {
        "schema_version": 1,
        "batch_dir": batch_dir.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "mode": "apply" if apply else "dry_run",
        "source_json_files": len(decisions),
        "actions": dict(sorted(counts.items())),
        "added_samples": added_counts,
        "replaced_existing_samples": {
            plan.locator: plan.new_sample_id
            for plan in sorted(replacements, key=lambda item: item.locator)
        },
        "classification_rule": {
            "TKA": DATASET_FOLDERS["TKA"],
            "bone": DATASET_FOLDERS["non_tka"],
        },
    }
    if fatal_error:
        summary["fatal_error"] = fatal_error
    write_reports(report_dir, decisions, summary)
    if fatal_error:
        raise RuntimeError(fatal_error)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely stage, deduplicate, validate, and incrementally import a physician annotation batch. "
            "The default mode is dry-run; pass --apply to update the target dataset."
        )
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--overrides-json",
        type=Path,
        help=(
            "Optional JSON mapping batch-relative annotation paths to sample_id, side, case_id, "
            "implant_status, study_phase, explicit replacement selectors/permission, skip_reason, "
            "and/or a review reason."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Update the target only after staging passes validation.")
    args = parser.parse_args()
    report_dir = args.report_dir or Path("outputs") / f"annotation_import_{args.batch_dir.name}"
    summary = import_batch(
        args.batch_dir,
        args.dataset_root,
        report_dir,
        args.overrides_json,
        apply=args.apply,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

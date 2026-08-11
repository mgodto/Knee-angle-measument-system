#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "point_mae_px": "point_mae_px",
    "nme_height_pct": "nme_height_pct",
    "mldfa_mae_deg": "mldfa_abs_error_deg",
    "mpta_mae_deg": "mpta_abs_error_deg",
    "jlca_mae_deg": "jlca_abs_error_deg",
    "hka_mae_deg": "hka_abs_error_deg",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_value(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "").strip()
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    outliers: dict[str, list[dict[str, object]]] = {}
    for metric_name, field in METRICS.items():
        values = [
            (row, value)
            for row in rows
            if (value := finite_value(row, field)) is not None
        ]
        array = np.asarray([value for _row, value in values], dtype=np.float64)
        metrics[metric_name] = {
            "mean": float(array.mean()) if array.size else None,
            "median": float(np.median(array)) if array.size else None,
            "p90": float(np.percentile(array, 90)) if array.size else None,
            "p95": float(np.percentile(array, 95)) if array.size else None,
            "max": float(array.max()) if array.size else None,
            "valid_n": int(array.size),
            "failure_count": len(rows) - int(array.size),
        }
        ordered = sorted(values, key=lambda item: item[1], reverse=True)[:10]
        outliers[metric_name] = [
            {
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "fold": int(row["fold"]),
                "value": value,
            }
            for row, value in ordered
        ]
    return {
        "rows": len(rows),
        "cases": len({row["case_id"] for row in rows}),
        "metrics": metrics,
        "outliers": outliers,
    }


def summarize_oof(oof_path: Path, manifest_path: Path) -> dict[str, object]:
    oof_rows = load_csv(oof_path)
    manifest_rows = load_csv(manifest_path)
    status_by_sample: dict[str, str] = {}
    for row in manifest_rows:
        sample_id = row["sample_id"]
        if sample_id in status_by_sample:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        status_by_sample[sample_id] = row["implant_status"]

    seen: set[str] = set()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in oof_rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in OOF metrics: {sample_id}")
        seen.add(sample_id)
        try:
            status = status_by_sample[sample_id]
        except KeyError as exc:
            raise ValueError(f"OOF sample is absent from manifest: {sample_id}") from exc
        grouped[status].append(row)

    expected = set(status_by_sample)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Manifest samples are absent from OOF metrics: {missing[:20]}")

    return {
        "schema_version": 1,
        "oof_metrics": oof_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "overall": summarize_rows(oof_rows),
        "cohorts": {
            status: summarize_rows(rows)
            for status, rows in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize OOF keypoint and angle errors overall and by implant cohort."
    )
    parser.add_argument("--oof-metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_oof(args.oof_metrics, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Subgroup summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()

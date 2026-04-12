#!/usr/bin/env python3

import argparse
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    for ch in ["’", "′", "＇", "`", "´"]:
        name = name.replace(ch, "'")
    name = name.replace("\u3000", " ")
    return " ".join(name.split())


def infer_side(name: str) -> str | None:
    if "RL" in name:
        return "RL"
    if "L" in name:
        return "L"
    if "R" in name:
        return "R"
    return None


def classify_file(path: Path) -> str:
    name = normalize_name(path.name)
    if "line" in name.lower():
        return "line"
    if "'" in name:
        return "point"
    if name.endswith(".jpg") and (name.endswith("RL.jpg") or name[-5].upper() in {"L", "R"}):
        return "raw"
    return "unknown"


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def extract_point_metrics(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"read_ok": False}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([140, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 20]

    return {
        "read_ok": True,
        "component_count": len(areas),
        "areas": sorted(areas, reverse=True),
    }


def extract_line_metrics(raw_path: Path, line_path: Path) -> dict:
    raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    line = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
    if raw is None or line is None:
        return {"read_ok": False}

    diff = cv2.absdiff(raw, line)
    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    hough = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=25, minLineLength=50, maxLineGap=10)

    lengths = []
    if hough is not None:
        for seg in hough:
            x1, y1, x2, y2 = seg[0]
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))

    return {
        "read_ok": True,
        "diff_pixels": int((mask > 0).sum()),
        "hough_count": 0 if hough is None else len(hough),
        "top_lengths": sorted(lengths, reverse=True)[:5],
    }


def build_case_record(case_dir: Path) -> dict:
    record = {
        "case": f"{case_dir.parent.name}/{case_dir.name}",
        "raw": [],
        "point": [],
        "line": [],
        "unknown": [],
    }

    for path in sorted(case_dir.iterdir()):
        if not path.is_file() or path.name == ".DS_Store":
            continue

        kind = classify_file(path)
        record[kind].append(
            {
                "path": path,
                "name": path.name,
                "normalized_name": normalize_name(path.name),
                "side": infer_side(normalize_name(path.name)),
                "size": image_size(path),
            }
        )

    return record


def index_by_side(items: list[dict]) -> dict[str, dict]:
    indexed = {}
    for item in items:
        side = item["side"]
        if side:
            indexed[side] = item
    return indexed


def print_case_table(records: list[dict]) -> None:
    for rec in records:
        print(
            f"{rec['case']}: raw={len(rec['raw'])} "
            f"point={len(rec['point'])} line={len(rec['line'])} "
            f"unknown={len(rec['unknown'])}"
        )


def summarize(records: list[dict]) -> None:
    raw_sizes = Counter()
    point_sizes = Counter()
    line_sizes = Counter()
    point_counts = Counter()
    line_hough_counts = Counter()
    missing_cases = []
    point_anomalies = []
    point_ratios_x = []
    point_ratios_y = []

    for rec in records:
        if not rec["raw"] or not rec["point"] or not rec["line"]:
            missing_cases.append(rec["case"])

        raw_by_side = index_by_side(rec["raw"])
        point_by_side = index_by_side(rec["point"])
        line_by_side = index_by_side(rec["line"])

        for item in rec["raw"]:
            if item["size"]:
                raw_sizes[item["size"]] += 1
        for item in rec["point"]:
            if item["size"]:
                point_sizes[item["size"]] += 1
        for item in rec["line"]:
            if item["size"]:
                line_sizes[item["size"]] += 1

        for side in ("L", "R"):
            point_item = point_by_side.get(side)
            raw_item = raw_by_side.get(side) or raw_by_side.get("RL")
            if point_item and raw_item and point_item["size"] and raw_item["size"]:
                pw, ph = point_item["size"]
                rw, rh = raw_item["size"]
                point_ratios_x.append(pw / rw)
                point_ratios_y.append(ph / rh)

        for point_item in rec["point"]:
            metrics = extract_point_metrics(point_item["path"])
            if not metrics.get("read_ok"):
                point_anomalies.append((rec["case"], point_item["name"], "read_fail"))
                continue

            point_counts[metrics["component_count"]] += 1
            if metrics["component_count"] != 8:
                point_anomalies.append(
                    (rec["case"], point_item["name"], metrics["component_count"], metrics["areas"])
                )

        for side, line_item in line_by_side.items():
            raw_item = raw_by_side.get(side) or raw_by_side.get("RL")
            if raw_item is None:
                continue

            metrics = extract_line_metrics(raw_item["path"], line_item["path"])
            if not metrics.get("read_ok"):
                continue
            line_hough_counts[metrics["hough_count"]] += 1

    print("Dataset summary")
    print(f"  cases: {len(records)}")
    print(f"  missing raw/point/line groups: {len(missing_cases)}")
    print(f"  raw size distribution: {dict(raw_sizes)}")
    print(f"  line size distribution: {dict(line_sizes)}")
    print(f"  point size variants: {len(point_sizes)}")

    if point_ratios_x and point_ratios_y:
        print(
            "  point/raw width ratio min/mean/max: "
            f"{min(point_ratios_x):.4f} / {statistics.mean(point_ratios_x):.4f} / {max(point_ratios_x):.4f}"
        )
        print(
            "  point/raw height ratio min/mean/max: "
            f"{min(point_ratios_y):.4f} / {statistics.mean(point_ratios_y):.4f} / {max(point_ratios_y):.4f}"
        )

    print("  blue point component counts:", dict(sorted(point_counts.items())))
    print("  line diff Hough counts:", dict(sorted(line_hough_counts.items())))

    if point_anomalies:
        print("\nPoint anomalies")
        for item in point_anomalies:
            print(f"  {item}")
    else:
        print("\nPoint anomalies")
        print("  none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the knee X-ray line/point dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("images/line_point"),
        help="Dataset root that contains grouped case folders.",
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print per-case raw/point/line counts before the aggregate summary.",
    )
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.root.glob("*/*") if path.is_dir())
    records = [build_case_record(case_dir) for case_dir in case_dirs]

    if args.show_cases:
        print_case_table(records)
        print()

    summarize(records)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


PRODUCTS = ("measurement", "annotation")
PLATFORMS = ("macos", "windows")
RETAIN_COUNT = 3
DELIVERY_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)(?P<date>\d{8})(?:T(?P<time>\d{6})Z)?(?=\.zip$)",
    re.IGNORECASE,
)


def delivery_sort_key(path: Path) -> tuple[int, str, str, str]:
    """Sort dated releases after legacy files without a filename timestamp."""

    match = DELIVERY_TIMESTAMP_PATTERN.search(path.name)
    if match:
        return (1, match.group("date"), match.group("time") or "000000", path.name)
    return (0, f"{path.stat().st_mtime_ns:020d}", "", path.name)


def retain_deliverables(
    repo_root: Path,
    product: str,
    platform: str,
    *,
    current: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Return stale ZIPs and remove them unless this is a dry run."""

    if product not in PRODUCTS:
        raise ValueError(f"unsupported product: {product}")
    if platform not in PLATFORMS:
        raise ValueError(f"unsupported platform: {platform}")

    repo_root = Path(repo_root)
    directory = repo_root / "deliverables" / product / platform
    if not directory.is_dir():
        raise FileNotFoundError(f"deliverable directory does not exist: {directory}")
    try:
        directory.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"deliverable directory resolves outside repository: {directory}"
        ) from exc

    current_path: Path | None = None
    if current is not None:
        candidate = Path(current)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"current deliverable is not a regular file: {candidate}")
        if candidate.suffix.lower() != ".zip":
            raise ValueError(f"current deliverable is not a ZIP: {candidate}")
        if candidate.resolve().parent != directory.resolve():
            raise ValueError(f"current deliverable is outside target directory: {candidate}")
        current_path = candidate.resolve()

    zip_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() == ".zip"
        ),
        key=delivery_sort_key,
    )
    stale_count = max(0, len(zip_paths) - RETAIN_COUNT)
    removable_paths = [
        path
        for path in zip_paths
        if current_path is None or path.resolve() != current_path
    ]
    stale_paths = removable_paths[:stale_count]
    if not dry_run:
        for path in stale_paths:
            path.unlink()
    return stale_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the latest three ZIPs in one Knee X-ray deliverable directory."
    )
    parser.add_argument("product", choices=PRODUCTS)
    parser.add_argument("platform", choices=PLATFORMS)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--current",
        type=Path,
        help="Protect the ZIP created by the current build during retention.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stale_paths = retain_deliverables(
        args.repo_root,
        args.product,
        args.platform,
        current=args.current,
        dry_run=args.dry_run,
    )
    action = "WOULD REMOVE" if args.dry_run else "REMOVED"
    for path in stale_paths:
        print(f"{action}: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from knee_xray.release.retain_deliverables import main, retain_deliverables


class DeliverableRetentionTests(unittest.TestCase):
    @staticmethod
    def write_zip(directory: Path, name: str, mtime_ns: int) -> Path:
        path = directory / name
        path.write_bytes(b"test ZIP payload")
        os.utime(path, ns=(mtime_ns, mtime_ns))
        return path

    def test_dry_run_and_prune_only_target_directory_zips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            target = repo_root / "deliverables" / "measurement" / "macos"
            target.mkdir(parents=True)

            old = self.write_zip(target, "old.zip", 100)
            tied_old = self.write_zip(target, "tie-a.zip", 200)
            tied_new = self.write_zip(target, "tie-b.zip", 200)
            new = self.write_zip(target, "new.zip", 300)
            newest = self.write_zip(target, "newest.ZIP", 400)
            note = target / "notes.txt"
            note.write_text("keep", encoding="utf-8")
            zip_directory = target / "directory.zip"
            zip_directory.mkdir()
            nested_zip = self.write_zip(zip_directory, "nested.zip", 1)
            sibling_directory = repo_root / "deliverables" / "annotation" / "macos"
            sibling_directory.mkdir(parents=True)
            sibling_zip = self.write_zip(sibling_directory, "sibling.zip", 1)

            dry_run_paths = retain_deliverables(
                repo_root,
                "measurement",
                "macos",
                dry_run=True,
            )
            self.assertEqual(dry_run_paths, [old, tied_old])
            self.assertTrue(old.exists())
            self.assertTrue(tied_old.exists())

            removed_paths = retain_deliverables(repo_root, "measurement", "macos")
            self.assertEqual(removed_paths, [old, tied_old])
            self.assertFalse(old.exists())
            self.assertFalse(tied_old.exists())
            for retained in (tied_new, new, newest, note, nested_zip, sibling_zip):
                self.assertTrue(retained.exists())

    def test_cli_dry_run_reports_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            target = repo_root / "deliverables" / "annotation" / "windows"
            target.mkdir(parents=True)
            oldest = self.write_zip(target, "oldest.zip", 100)
            for index in range(3):
                self.write_zip(target, f"new-{index}.zip", 200 + index)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "annotation",
                        "windows",
                        "--repo-root",
                        str(repo_root),
                        "--dry-run",
                    ]
                )

            self.assertTrue(oldest.exists())
            self.assertEqual(stdout.getvalue().strip(), f"WOULD REMOVE: {oldest}")

    def test_current_build_is_protected_even_with_an_old_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            target = repo_root / "deliverables" / "measurement" / "windows"
            target.mkdir(parents=True)
            current = self.write_zip(target, "current.zip", 50)
            oldest_other = self.write_zip(target, "old.zip", 100)
            retained_one = self.write_zip(target, "new.zip", 200)
            retained_two = self.write_zip(target, "newest.zip", 300)

            removed_paths = retain_deliverables(
                repo_root,
                "measurement",
                "windows",
                current=current,
            )

            self.assertEqual(removed_paths, [oldest_other])
            self.assertTrue(current.exists())
            self.assertFalse(oldest_other.exists())
            self.assertTrue(retained_one.exists())
            self.assertTrue(retained_two.exists())

    def test_filename_release_timestamp_takes_priority_over_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            target = repo_root / "deliverables" / "measurement" / "macos"
            target.mkdir(parents=True)
            oldest_release = self.write_zip(
                target,
                "KneeXrayMeasurement-macOS-v0.3.0-20260720.zip",
                900,
            )
            for version, date, mtime_ns in (
                ("0.4.0", "20260727", 300),
                ("0.5.1", "20260803", 200),
                ("0.6.0", "20260811", 100),
            ):
                self.write_zip(
                    target,
                    f"KneeXrayMeasurement-macOS-v{version}-{date}.zip",
                    mtime_ns,
                )

            removed_paths = retain_deliverables(repo_root, "measurement", "macos")

            self.assertEqual(removed_paths, [oldest_release])
            self.assertFalse(oldest_release.exists())

    def test_utc_build_timestamp_orders_same_day_annotation_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            target = repo_root / "deliverables" / "annotation" / "macos"
            target.mkdir(parents=True)
            oldest_build = self.write_zip(
                target,
                "KneeAnnotationTool-macOS-v1.2-build3-20260811T010000Z.zip",
                900,
            )
            for build_time, mtime_ns in (
                ("020000", 300),
                ("030000", 200),
                ("040000", 100),
            ):
                self.write_zip(
                    target,
                    f"KneeAnnotationTool-macOS-v1.2-build3-20260811T{build_time}Z.zip",
                    mtime_ns,
                )

            removed_paths = retain_deliverables(repo_root, "annotation", "macos")

            self.assertEqual(removed_paths, [oldest_build])
            self.assertFalse(oldest_build.exists())

    def test_current_build_must_be_a_regular_zip_in_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            target = repo_root / "deliverables" / "annotation" / "macos"
            target.mkdir(parents=True)
            outside = repo_root / "outside.zip"
            outside.write_bytes(b"outside")

            with self.assertRaisesRegex(ValueError, "outside target directory"):
                retain_deliverables(
                    repo_root,
                    "annotation",
                    "macos",
                    current=outside,
                )

    def test_rejects_delivery_tree_symlinked_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            repo_root = temporary_root / "repo"
            outside = temporary_root / "outside"
            (outside / "measurement" / "macos").mkdir(parents=True)
            repo_root.mkdir()
            try:
                (repo_root / "deliverables").symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "resolves outside repository"):
                retain_deliverables(repo_root, "measurement", "macos")

    def test_rejects_paths_outside_known_product_platform_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "unsupported product"):
                retain_deliverables(repo_root, "../other", "macos")
            with self.assertRaisesRegex(ValueError, "unsupported platform"):
                retain_deliverables(repo_root, "measurement", "linux")


if __name__ == "__main__":
    unittest.main()

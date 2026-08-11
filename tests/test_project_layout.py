from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_PY_ALLOWLIST = {"annotate_gui.py", "knee_measurement_app.py"}
ROOT_FORBIDDEN_SUFFIXES = {".bat", ".sh", ".spec"}
REQUIRED_PACKAGE_DIRS = {
    "core",
    "data",
    "inference",
    "ml",
    "release",
    "training",
    "ui",
}
REQUIRED_PROJECT_DIRS = {
    PurePosixPath("config"),
    PurePosixPath("docs"),
    PurePosixPath("packaging"),
    PurePosixPath("requirements"),
    PurePosixPath("scripts/build"),
    PurePosixPath("scripts/training"),
}
REQUIRED_INFRA_FILES = {
    PurePosixPath("config/knee_measurement_app.json"),
    PurePosixPath("docs/PROJECT_STRUCTURE.md"),
    PurePosixPath("docs/README_DOCTOR_EN.txt"),
    PurePosixPath("docs/README_DOCTOR_JA.txt"),
    PurePosixPath("docs/README_training.md"),
    PurePosixPath("packaging/knee_annotation_tool.spec"),
    PurePosixPath("packaging/knee_measurement_app.spec"),
    PurePosixPath("requirements/requirements-app.txt"),
    PurePosixPath("requirements/requirements-packaging.txt"),
    PurePosixPath("requirements/requirements-training.txt"),
    PurePosixPath("knee_xray/release/retain_deliverables.py"),
    PurePosixPath("scripts/build/build_mac.sh"),
    PurePosixPath("scripts/build/build_measurement_mac.sh"),
    PurePosixPath("scripts/build/build_measurement_windows.bat"),
    PurePosixPath("scripts/build/build_windows.bat"),
    PurePosixPath("scripts/training/run_retraining_single_leg_v2_20260803.sh"),
    PurePosixPath("scripts/training/run_retraining_single_leg_v3_20260811.sh"),
}
ARTIFACT_TREE_NAMES = {
    "automation_monthly_reports",
    "build",
    "deliverables",
    "dist",
    "images",
    "outputs",
    "private",
    "projects",
    "releases",
}
CODE_SUFFIXES = {".bat", ".py", ".sh", ".spec"}
LAUNCHER_TARGETS = {
    "annotate_gui.py": "knee_xray.ui.annotate_gui",
    "knee_measurement_app.py": "knee_xray.ui.knee_measurement_app",
}
BUILD_DELIVERY_TARGETS = {
    "build_mac.sh": "deliverables/annotation/macos",
    "build_measurement_mac.sh": "deliverables/measurement/macos",
    "build_windows.bat": "deliverables/annotation/windows",
    "build_measurement_windows.bat": "deliverables/measurement/windows",
}
WORKFLOW_DELIVERY_TARGETS = {
    "build-windows.yml": "deliverables/annotation/windows/*.zip",
    "build-measurement-windows.yml": "deliverables/measurement/windows/*.zip",
}


def versioned_worktree_paths() -> set[PurePosixPath]:
    """Return tracked and untracked, non-ignored files that still exist."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest(f"Git worktree metadata is required: {exc}") from exc

    paths = set()
    for raw_path in result.stdout.decode("utf-8").split("\0"):
        if not raw_path:
            continue
        path = PurePosixPath(raw_path)
        if (PROJECT_ROOT / Path(*path.parts)).exists():
            paths.add(path)
    return paths


def is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


class ProjectLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worktree_paths = versioned_worktree_paths()

    def test_only_compatibility_python_launchers_are_at_root(self) -> None:
        root_python = {
            path.name
            for path in self.worktree_paths
            if len(path.parts) == 1 and path.suffix == ".py"
        }
        self.assertEqual(root_python, ROOT_PY_ALLOWLIST)

    def test_no_build_or_training_entrypoints_are_at_root(self) -> None:
        misplaced = sorted(
            path.as_posix()
            for path in self.worktree_paths
            if len(path.parts) == 1 and path.suffix in ROOT_FORBIDDEN_SUFFIXES
        )
        self.assertEqual(misplaced, [])

    def test_canonical_package_shape(self) -> None:
        package_root = PROJECT_ROOT / "knee_xray"
        self.assertTrue((package_root / "__init__.py").is_file())
        for directory in sorted(REQUIRED_PACKAGE_DIRS):
            with self.subTest(directory=directory):
                self.assertTrue((package_root / directory / "__init__.py").is_file())
        for relative_path in sorted(REQUIRED_PROJECT_DIRS):
            with self.subTest(directory=relative_path.as_posix()):
                self.assertTrue((PROJECT_ROOT / Path(*relative_path.parts)).is_dir())

    def test_organized_infrastructure_files_are_present(self) -> None:
        for relative_path in sorted(REQUIRED_INFRA_FILES):
            with self.subTest(path=relative_path.as_posix()):
                self.assertTrue((PROJECT_ROOT / Path(*relative_path.parts)).is_file())

    def test_measurement_packaging_uses_the_canonical_macos_entrypoint(self) -> None:
        spec = (PROJECT_ROOT / "packaging" / "knee_measurement_app.spec").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'project_root / "knee_xray" / "ui" / "knee_measurement_app.py"',
            spec,
        )
        self.assertIn('project_root / "knee_measurement_app_windows.py"', spec)

    def test_no_canonical_code_is_versioned_in_artifact_trees(self) -> None:
        misplaced = sorted(
            path.as_posix()
            for path in self.worktree_paths
            if path.parts
            and path.parts[0] in ARTIFACT_TREE_NAMES
            and path.suffix in CODE_SUFFIXES
        )
        self.assertEqual(misplaced, [])

    def test_repository_dist_and_root_zip_outputs_are_retired(self) -> None:
        misplaced = sorted(
            path.as_posix()
            for path in self.worktree_paths
            if (path.parts and path.parts[0] == "dist")
            or (len(path.parts) == 1 and path.suffix.lower() == ".zip")
        )
        self.assertEqual(misplaced, [])

    def test_builds_stage_and_deliver_in_canonical_directories(self) -> None:
        repository_dist_reference = re.compile(
            r"(?:\$PWD|\$\{?project_root\}?|%CD%)[/\\]dist|"
            r"(?<![A-Za-z0-9_-])dist[/\\](?:KneeAnnotationTool|KneeXrayMeasurement)",
            re.IGNORECASE,
        )
        for filename, delivery_target in BUILD_DELIVERY_TARGETS.items():
            with self.subTest(filename=filename):
                source = (PROJECT_ROOT / "scripts" / "build" / filename).read_text(
                    encoding="utf-8"
                )
                normalized = source.replace("\\", "/")
                self.assertIn("build/pyinstaller-dist", normalized)
                self.assertIn("build/deliverable-stage", normalized)
                self.assertIn(delivery_target, normalized)
                self.assertIn("knee_xray.release.retain_deliverables", source)
                self.assertIn("--current", source)
                self.assertIsNone(repository_dist_reference.search(source))

    def test_workflows_upload_only_canonical_delivery_zips(self) -> None:
        for filename, delivery_target in WORKFLOW_DELIVERY_TARGETS.items():
            with self.subTest(filename=filename):
                source = (
                    PROJECT_ROOT / ".github" / "workflows" / filename
                ).read_text(encoding="utf-8")
                normalized = source.replace("\\", "/")
                self.assertIn(f"path: {delivery_target}", normalized)
                self.assertNotRegex(normalized, r"(?m)^\s*path:\s*dist/")

    def test_ignore_rules_expose_misplaced_dist_and_zip_outputs(self) -> None:
        ignore_lines = {
            line.strip()
            for line in (PROJECT_ROOT / ".gitignore").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("/deliverables/", ignore_lines)
        self.assertNotIn("dist/", ignore_lines)
        self.assertNotIn("/dist/", ignore_lines)
        self.assertNotIn("*.zip", ignore_lines)

    def test_compatibility_launchers_remain_thin(self) -> None:
        allowed_nodes = (ast.AnnAssign, ast.Assign, ast.Expr, ast.If, ast.Import, ast.ImportFrom)
        for filename, target_module in LAUNCHER_TARGETS.items():
            with self.subTest(filename=filename):
                source = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
                nonblank_lines = [line for line in source.splitlines() if line.strip()]
                self.assertLessEqual(len(nonblank_lines), 80)

                tree = ast.parse(source, filename=filename)
                unexpected = [
                    type(node).__name__
                    for node in tree.body
                    if not isinstance(node, allowed_nodes)
                ]
                self.assertEqual(unexpected, [])
                self.assertEqual(sum(is_main_guard(node) for node in tree.body), 1)

                imported_modules = {
                    node.module
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module is not None
                }
                self.assertIn(target_module, imported_modules)


if __name__ == "__main__":
    unittest.main()

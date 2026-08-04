from __future__ import annotations

import ast
from pathlib import Path
import re
import tempfile
import unittest

from generate_windows_english_entrypoint import generate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_APP = PROJECT_ROOT / "knee_measurement_app.py"
CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


class WindowsEnglishEntrypointTests(unittest.TestCase):
    def test_generation_is_deterministic_ascii_and_compilable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.py"
            second_path = Path(directory) / "second.py"
            first = generate(CANONICAL_APP, first_path)
            second = generate(CANONICAL_APP, second_path)

        self.assertEqual(first, second)
        self.assertFalse(CJK_PATTERN.search(first))
        first.encode("ascii")
        self.assertIn("WINDOWS_ENGLISH_BUILD = True", first)
        self.assertIn('APP_TITLE = "Full-Length Leg X-ray Automated Measurement"', first)
        self.assertIn('"auto": "Auto"', first)
        self.assertIn('text="Crop bilateral image"', first)
        self.assertIn('"Return to single-leg"', first)
        self.assertIn('"hip": "Point 1 - Hip center"', first)

    def test_generated_entrypoint_preserves_canonical_class_and_function_names(self) -> None:
        canonical_tree = ast.parse(CANONICAL_APP.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            generated = generate(CANONICAL_APP, Path(directory) / "windows_app.py")
        generated_tree = ast.parse(generated)

        def definitions(tree: ast.Module) -> set[tuple[str, str]]:
            return {
                (type(node).__name__, node.name)
                for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            }

        self.assertLessEqual(definitions(canonical_tree), definitions(generated_tree))

    def test_new_untranslated_ui_text_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            output = Path(directory) / "output.py"
            source.write_text(
                CANONICAL_APP.read_text(encoding="utf-8") + '\nNEW_UI_TEXT = "追加表示"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "untranslated CJK text remains"):
                generate(source, output)


if __name__ == "__main__":
    unittest.main()

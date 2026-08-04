from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from create_release_smoke_fixture import FIXTURE_SIZE, create_fixture


class ReleaseSmokeFixtureTests(unittest.TestCase):
    def test_fixture_is_deterministic_decodable_and_single_leg_shaped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            create_fixture(first)
            create_fixture(second)

            self.assertEqual(hashlib.sha256(first.read_bytes()).digest(), hashlib.sha256(second.read_bytes()).digest())
            with Image.open(first) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, FIXTURE_SIZE)
                self.assertLess(image.width / image.height, 0.60)
                self.assertNotEqual(image.getextrema()[0], image.getextrema()[1])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

from knee_dataset_utils import RawCandidate, resolve_raw_candidate


class RawCandidateResolutionTests(unittest.TestCase):
    def test_source_raw_filename_is_used_when_exported_raw_is_missing(self) -> None:
        source = RawCandidate(
            path=Path("images/20260720/074 RL post/001/source.jpg"),
            width=2372,
            height=2880,
        )
        annotation = {
            "raw_filename": "source_L_raw.jpg",
            "raw_path": "source_L_raw.jpg",
            "source_raw_filename": "source.jpg",
            "source_raw_path": "/doctor/data/074 RL post/001/source.jpg",
            "image_width": 2372,
            "image_height": 2880,
        }

        resolved, match_count = resolve_raw_candidate(
            Path("images/20260720/074 RL post/074L_post_bone_annotation.json"),
            annotation,
            {"source.jpg": [source]},
        )

        self.assertEqual(resolved, source)
        self.assertEqual(match_count, 1)

    def test_source_reference_takes_priority_over_an_existing_exported_raw(self) -> None:
        exported = RawCandidate(path=Path("wrong/exported_raw.jpg"), width=100, height=200)
        source = RawCandidate(path=Path("correct/study/source.jpg"), width=100, height=200)
        annotation = {
            "raw_filename": "exported_raw.jpg",
            "raw_path": "wrong/exported_raw.jpg",
            "source_raw_path": r"C:\doctor\correct\study\source.jpg",
            "image_width": 100,
            "image_height": 200,
        }

        resolved, match_count = resolve_raw_candidate(
            Path("batch/case/sample_annotation.json"),
            annotation,
            {"exported_raw.jpg": [exported], "source.jpg": [source]},
        )

        self.assertEqual(resolved, source)
        self.assertEqual(match_count, 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from knee_dataset_utils import annotation_keypoints
from measure_angles import ANNOTATION_POINT_NAMES


class AnnotationKeypointTests(unittest.TestCase):
    def test_line_endpoint_canonicalization_orders_points_left_to_right(self) -> None:
        annotation = {
            "points": {
                name: {"x": index, "y": index + 1}
                for index, name in enumerate(ANNOTATION_POINT_NAMES)
            },
            "lines": {
                "upper_line": {"p1": {"x": 80, "y": 10}, "p2": {"x": 20, "y": 12}},
                "lower_line": {"p1": {"x": 15, "y": 30}, "p2": {"x": 75, "y": 32}},
            },
        }

        original = annotation_keypoints(annotation)
        canonical = annotation_keypoints(annotation, canonicalize_line_endpoints=True)

        self.assertEqual(original["upper_line_p1"], (80.0, 10.0))
        self.assertEqual(canonical["upper_line_p1"], (20.0, 12.0))
        self.assertEqual(canonical["upper_line_p2"], (80.0, 10.0))
        self.assertEqual(canonical["lower_line_p1"], (15.0, 30.0))
        self.assertEqual(canonical["lower_line_p2"], (75.0, 32.0))


if __name__ == "__main__":
    unittest.main()

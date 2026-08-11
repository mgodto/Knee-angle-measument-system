from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from knee_xray.ml.knee_keypoint_model import (
    ADAPTER_ID,
    ARCHITECTURE_ID,
    CHECKPOINT_SCHEMA_VERSION,
    HARD_ARGMAX_DECODER_ID,
    KEYPOINT_NAMES,
    LOCAL_CENTROID_DECODER_ID,
    PREPROCESSING_ID,
    SmallHeatmapNet,
    decode_heatmaps_for_shape,
)
from knee_xray.inference.knee_model_runtime import ModelLoadError, ModelSpec, SmallHeatmapV1Adapter


class DecoderTests(unittest.TestCase):
    def test_local_centroid_refines_peak_without_changing_peak_score(self) -> None:
        logits = torch.full((1, 5, 5), -4.0)
        logits[0, 2, 2] = 4.0
        logits[0, 2, 1] = 0.0
        logits[0, 2, 3] = 2.0

        hard_coords, hard_scores = decode_heatmaps_for_shape(
            logits, 20, 20, 20, 20, 4, HARD_ARGMAX_DECODER_ID
        )
        refined_coords, refined_scores = decode_heatmaps_for_shape(
            logits, 20, 20, 20, 20, 4, LOCAL_CENTROID_DECODER_ID
        )

        self.assertEqual(tuple(hard_coords[0]), (8.0, 8.0))
        self.assertGreater(float(refined_coords[0, 0]), float(hard_coords[0, 0]))
        self.assertAlmostEqual(float(refined_coords[0, 1]), float(hard_coords[0, 1]), places=5)
        torch.testing.assert_close(torch.from_numpy(refined_scores), torch.from_numpy(hard_scores))

    def test_runtime_rejects_unknown_checkpoint_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "unsupported_decoder.pt"
            model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES))
            torch.save(
                {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "adapter_id": ADAPTER_ID,
                    "architecture_id": ARCHITECTURE_ID,
                    "preprocessing_id": PREPROCESSING_ID,
                    "decoder_id": "unknown_decoder",
                    "model_state": model.state_dict(),
                    "keypoint_names": KEYPOINT_NAMES,
                    "image_width": 32,
                    "image_height": 64,
                    "stride": 4,
                },
                checkpoint_path,
            )

            adapter = SmallHeatmapV1Adapter(
                ModelSpec(adapter=ADAPTER_ID, checkpoint=checkpoint_path, device="cpu")
            )
            with self.assertRaisesRegex(ModelLoadError, "デコーダー"):
                adapter.load()

    def test_runtime_loads_versioned_local_centroid_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "local_centroid.pt"
            model = SmallHeatmapNet(out_channels=len(KEYPOINT_NAMES))
            torch.save(
                {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "adapter_id": ADAPTER_ID,
                    "architecture_id": ARCHITECTURE_ID,
                    "preprocessing_id": PREPROCESSING_ID,
                    "decoder_id": LOCAL_CENTROID_DECODER_ID,
                    "model_state": model.state_dict(),
                    "keypoint_names": KEYPOINT_NAMES,
                    "image_width": 32,
                    "image_height": 64,
                    "stride": 4,
                },
                checkpoint_path,
            )

            adapter = SmallHeatmapV1Adapter(
                ModelSpec(adapter=ADAPTER_ID, checkpoint=checkpoint_path, device="cpu")
            )
            info = adapter.load()

            self.assertEqual(info.decoder_id, LOCAL_CENTROID_DECODER_ID)


if __name__ == "__main__":
    unittest.main()

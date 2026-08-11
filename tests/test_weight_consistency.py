from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knee_xray.inference.knee_model_runtime import (
    EXPECTED_KEYPOINT_NAMES,
    ModelLoadError,
    ModelSpec,
    SmallHeatmapV1Adapter,
    save_model_preference,
    sha256_file,
    user_preferences_path,
)


class WeightConsistencyTests(unittest.TestCase):
    @staticmethod
    def _spec(checkpoint: Path) -> ModelSpec:
        return ModelSpec(
            adapter="small_heatmap_v1",
            checkpoint=checkpoint,
            display_name="External",
            version="auto",
            cohort="external",
            device="cpu",
            options={"allow_legacy_checkpoint": True},
        )

    def test_loaded_model_info_hashes_the_exact_bytes_given_to_torch(self) -> None:
        try:
            import torch
            from knee_xray.ml.knee_keypoint_model import SmallHeatmapNet
        except ImportError:
            self.skipTest("PyTorch is not installed")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "external.pt"
            model = SmallHeatmapNet(out_channels=len(EXPECTED_KEYPOINT_NAMES))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "keypoint_names": EXPECTED_KEYPOINT_NAMES,
                    "image_width": 32,
                    "image_height": 32,
                    "stride": 4,
                },
                checkpoint_path,
            )
            loaded_bytes_sha256 = sha256_file(checkpoint_path)
            real_torch_load = torch.load

            def load_then_replace_source(file_object, *args, **kwargs):
                checkpoint = real_torch_load(file_object, *args, **kwargs)
                checkpoint_path.write_bytes(b"changed-after-torch-load")
                return checkpoint

            with mock.patch.object(torch, "load", side_effect=load_then_replace_source):
                info = SmallHeatmapV1Adapter(self._spec(checkpoint_path)).load()

            self.assertNotEqual(sha256_file(checkpoint_path), loaded_bytes_sha256)
            self.assertEqual(info.checkpoint_sha256, loaded_bytes_sha256)

    def test_save_rejects_source_changed_after_model_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "external.pt"
            checkpoint_path.write_bytes(b"validated-model-bytes")
            expected_sha256 = sha256_file(checkpoint_path)
            checkpoint_path.write_bytes(b"new-model-bytes")

            with mock.patch.dict(os.environ, {"KNEE_XRAY_APP_DATA": str(root / "app-data")}):
                with self.assertRaisesRegex(ModelLoadError, "\u8aad\u307f\u8fbc\u307f\u5f8c\u306b\u5909\u66f4"):
                    save_model_preference(
                        self._spec(checkpoint_path),
                        expected_sha256=expected_sha256,
                    )
                self.assertFalse(user_preferences_path().exists())

    def test_save_rejects_source_changed_during_managed_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "external.pt"
            checkpoint_path.write_bytes(b"validated-model-bytes")
            expected_sha256 = sha256_file(checkpoint_path)
            real_copy2 = shutil.copy2

            def change_source_then_copy(source, destination, *args, **kwargs):
                Path(source).write_bytes(b"changed-during-copy")
                return real_copy2(source, destination, *args, **kwargs)

            with mock.patch.dict(os.environ, {"KNEE_XRAY_APP_DATA": str(root / "app-data")}):
                with mock.patch(
                    "knee_xray.inference.knee_model_runtime.shutil.copy2",
                    side_effect=change_source_then_copy,
                ):
                    with self.assertRaisesRegex(ModelLoadError, "\u4fdd\u5b58\u4e2d\u306b\u5909\u66f4"):
                        save_model_preference(
                            self._spec(checkpoint_path),
                            expected_sha256=expected_sha256,
                        )

                self.assertFalse(user_preferences_path().exists())
                managed_dir = user_preferences_path().parent / "models"
                self.assertEqual(list(managed_dir.iterdir()), [])

    def test_save_with_matching_expected_hash_persists_verified_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "external.pt"
            checkpoint_path.write_bytes(b"validated-model-bytes")
            expected_sha256 = sha256_file(checkpoint_path)

            with mock.patch.dict(os.environ, {"KNEE_XRAY_APP_DATA": str(root / "app-data")}):
                preference_path = save_model_preference(
                    self._spec(checkpoint_path),
                    expected_sha256=expected_sha256.upper(),
                )
                preference = json.loads(preference_path.read_text(encoding="utf-8"))
                managed_checkpoint = Path(preference["checkpoint"])

                self.assertEqual(preference["checkpoint_sha256"], expected_sha256)
                self.assertEqual(sha256_file(managed_checkpoint), expected_sha256)
                self.assertEqual(managed_checkpoint.read_bytes(), checkpoint_path.read_bytes())


if __name__ == "__main__":
    unittest.main()

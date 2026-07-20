from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from knee_measurement_app import KneeMeasurementApp, safe_export_stem
from measure_angles import RENDER_STYLE_CLINICAL, measure_from_named_points
from knee_model_runtime import (
    EXPECTED_KEYPOINT_NAMES,
    AnalysisResult,
    KneeAnalysisService,
    InferenceError,
    LandmarkPrediction,
    ModelInfo,
    ModelLoadError,
    ModelSpec,
    SideRequiredError,
    SmallHeatmapV1Adapter,
    export_record,
    clear_model_preference,
    coordinate_display_name,
    coordinate_geometry_warnings,
    load_app_config,
    load_model_preference,
    measurement_from_coordinates,
    model_quality_warnings,
    resolve_side,
    sha256_file,
    save_model_preference,
    write_json,
    write_overlay,
    write_result_bundle,
)


class FakeAdapter:
    def __init__(self) -> None:
        self._info = ModelInfo(
            adapter="fake",
            display_name="Synthetic test model",
            version="test-1",
            cohort="test",
            checkpoint_path="/tmp/fake.pt",
            checkpoint_sha256="a" * 64,
            device="cpu",
            input_width=128,
            input_height=256,
            stride=4,
            epoch=1,
            val_metrics={},
            checkpoint_schema_version=1,
            architecture_id="fake",
            preprocessing_id="fake",
            metadata_source="test",
            training_manifest_sha256=None,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> ModelInfo:
        return self._info

    def predict(self, image_bgr: np.ndarray) -> LandmarkPrediction:
        points = {
            "hip": np.array([180.0, 70.0], dtype=np.float32),
            "upper_left": np.array([115.0, 390.0], dtype=np.float32),
            "upper_center": np.array([200.0, 380.0], dtype=np.float32),
            "upper_right": np.array([285.0, 370.0], dtype=np.float32),
            "lower_left": np.array([115.0, 410.0], dtype=np.float32),
            "lower_center": np.array([205.0, 420.0], dtype=np.float32),
            "lower_right": np.array([285.0, 430.0], dtype=np.float32),
            "ankle": np.array([230.0, 730.0], dtype=np.float32),
        }
        lines = {
            "upper_line": {
                "p1": np.array([115.0, 390.0], dtype=np.float32),
                "p2": np.array([285.0, 370.0], dtype=np.float32),
            },
            "lower_line": {
                "p1": np.array([115.0, 410.0], dtype=np.float32),
                "p2": np.array([285.0, 430.0], dtype=np.float32),
            },
        }
        return LandmarkPrediction(
            schema_version=1,
            points=points,
            lines=lines,
            peak_scores={name: 0.9 for name in EXPECTED_KEYPOINT_NAMES},
            model_info=self.info,
            elapsed_ms=2.5,
        )


def write_test_image(path: Path) -> None:
    image = np.zeros((800, 400, 3), dtype=np.uint8)
    cv2.line(image, (200, 40), (200, 760), (180, 180, 180), 24)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("Could not encode test image")
    path.write_bytes(encoded.tobytes())


class ConfigTests(unittest.TestCase):
    def test_relative_checkpoint_resolves_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "models" / "best.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"placeholder")
            config_path = root / "app.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": {"adapter": "small_heatmap_v1", "checkpoint": "models/best.pt"},
                    }
                ),
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            self.assertEqual(config.model.checkpoint, checkpoint.resolve())

    def test_invalid_config_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema_version": 999, "model": {}}', encoding="utf-8")
            with self.assertRaises(ModelLoadError):
                load_app_config(path)

    def test_invalid_peak_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": {
                            "adapter": "small_heatmap_v1",
                            "checkpoint": "best.pt",
                            "options": {"low_peak_threshold": "bad"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelLoadError, "low_peak_threshold"):
                load_app_config(path)

    def test_external_model_preference_persists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "external.pt"
            checkpoint.write_bytes(b"test")
            base = ModelSpec(adapter="small_heatmap_v1", checkpoint=root / "built-in.pt")
            external = ModelSpec(
                adapter="small_heatmap_v1",
                checkpoint=checkpoint,
                display_name="External",
                version="auto",
                cohort="external",
                device="auto",
            )
            previous = os.environ.get("KNEE_XRAY_APP_DATA")
            os.environ["KNEE_XRAY_APP_DATA"] = str(root / "preferences")
            try:
                save_model_preference(external)
                loaded, warning = load_model_preference(base)
                self.assertIsNone(warning)
                self.assertNotEqual(loaded.checkpoint, checkpoint.resolve())
                self.assertTrue(loaded.checkpoint.is_file())
                self.assertEqual(loaded.checkpoint.read_bytes(), checkpoint.read_bytes())
                self.assertEqual(loaded.device, "cpu")
                managed_checkpoint = loaded.checkpoint
                managed_checkpoint.write_bytes(b"tampered")
                fallback, warning = load_model_preference(base)
                self.assertEqual(fallback, base)
                self.assertIsNotNone(warning)
                clear_model_preference()
                self.assertFalse(managed_checkpoint.exists())
                restored, warning = load_model_preference(base)
                self.assertIsNone(warning)
                self.assertEqual(restored, base)
            finally:
                if previous is None:
                    os.environ.pop("KNEE_XRAY_APP_DATA", None)
                else:
                    os.environ["KNEE_XRAY_APP_DATA"] = previous

    def test_export_filename_is_windows_safe(self) -> None:
        self.assertEqual(safe_export_stem('patient:01?L'), "patient_01_L")

    def test_release_config_forces_cpu(self) -> None:
        self.assertEqual(load_app_config().model.device, "cpu")


class GuiModelSwitchTests(unittest.TestCase):
    def test_successful_model_switch_invalidates_old_result_before_rerun(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        old_service = object()
        new_service = object()
        app.service = old_service
        app.model_spec = None
        app.preference_warning = None
        app.raw_path = Path("unknown-side.jpg")
        app.analysis = object()
        app.points = {"hip": np.array([1.0, 2.0], dtype=np.float32)}
        calls: list[str] = []

        def clear_analysis(*, keep_raw: bool) -> None:
            self.assertTrue(keep_raw)
            calls.append("clear")
            app.analysis = None
            app.points = {}

        def start_inference() -> None:
            self.assertIsNone(app.analysis)
            self.assertFalse(app.points)
            calls.append("infer")

        app._clear_analysis = clear_analysis
        app._update_model_badge = lambda: None
        app._update_model_details = lambda: None
        app._set_busy = lambda *_args, **_kwargs: None
        app._start_inference = start_inference
        spec = ModelSpec(adapter="fake", checkpoint=Path("new.pt"))

        app._finish_model_event(
            {
                "ok": True,
                "value": {
                    "service": new_service,
                    "spec": spec,
                    "preference_action": None,
                },
            }
        )

        self.assertIs(app.service, new_service)
        self.assertEqual(calls, ["clear", "infer"])

    def test_open_path_defers_image_decode_to_inference_worker(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.service = object()
        app.raw_image = None
        app.path_var = mock.Mock()
        app.status_var = mock.Mock()
        app.side_var = mock.Mock()
        app._clear_analysis = mock.Mock()
        app._start_inference = mock.Mock()

        path = Path("001L.png").resolve()
        with mock.patch("knee_measurement_app.infer_knee_side_from_sources", return_value="L"):
            app.open_path(path)

        self.assertEqual(app.raw_path, path)
        self.assertIsNone(app.raw_image)
        self.assertEqual(app.side_source_hint, "filename")
        app._start_inference.assert_called_once_with()

    def test_finished_inference_reuses_raw_bitmap_until_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "001L.png"
            write_test_image(image_path)
            result = KneeAnalysisService(
                FakeAdapter(),
                render_component_images=False,
            ).analyze_path(image_path)

        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.displayed_source_sha256 = None
        app.input_view = mock.Mock()
        app.input_view.has_image = False
        app.side_var = mock.Mock()
        app.status_var = mock.Mock()
        app.result_state_var = mock.Mock()
        app._set_busy = mock.Mock()
        app._display_measurement = mock.Mock()
        app._refresh_coordinate_table = mock.Mock()
        app._update_quality_display = mock.Mock()
        app._refresh_action_states = mock.Mock()
        app.edited_keys = {"hip"}
        app.history = [{}]

        app._finish_inference_event({"ok": True, "value": result})
        app.input_view.has_image = True
        app._finish_inference_event({"ok": True, "value": result})
        self.assertEqual(app.input_view.set_image.call_count, 1)

        changed = replace(result, source_sha256="b" * 64)
        app._finish_inference_event({"ok": True, "value": changed})
        self.assertEqual(app.input_view.set_image.call_count, 2)
        self.assertEqual(app.displayed_source_sha256, "b" * 64)


class AnalysisServiceTests(unittest.TestCase):
    def test_point_display_names_use_ids_without_screen_left_right_claims(self) -> None:
        for point_id, name in enumerate(EXPECTED_KEYPOINT_NAMES[:8], start=1):
            label = coordinate_display_name(name)
            self.assertTrue(label.startswith(f"点{point_id}・"))
            self.assertNotIn("画像左", label)
            self.assertNotIn("画像右", label)

    def test_center_landmarks_outside_outer_points_warn_for_affected_angles(self) -> None:
        prediction = FakeAdapter().predict(np.zeros((800, 400, 3), dtype=np.uint8))
        points = {name: point.copy() for name, point in prediction.points.items()}
        points["upper_center"][0] = 350.0
        before = {name: point.copy() for name, point in points.items()}

        warnings = coordinate_geometry_warnings(points, prediction.lines, (800, 400, 3))

        self.assertTrue(any("点3" in warning and "mLDFA" in warning and "HKA" in warning for warning in warnings))
        self.assertFalse(any("点6" in warning and "MPTA" in warning for warning in warnings))
        for name in points:
            np.testing.assert_array_equal(points[name], before[name])

        points = {name: point.copy() for name, point in prediction.points.items()}
        points["lower_center"][0] = 350.0
        warnings = coordinate_geometry_warnings(points, prediction.lines, (800, 400, 3))
        self.assertTrue(any("点6" in warning and "MPTA" in warning and "HKA" in warning for warning in warnings))

    def test_valid_center_geometry_has_no_center_position_warning(self) -> None:
        prediction = FakeAdapter().predict(np.zeros((800, 400, 3), dtype=np.uint8))

        warnings = coordinate_geometry_warnings(
            prediction.points,
            prediction.lines,
            (800, 400, 3),
        )

        self.assertFalse(any("水平方向の間にありません" in warning for warning in warnings))

    def test_real_right_to_left_outer_point_orientation_has_no_center_warning(self) -> None:
        prediction = FakeAdapter().predict(np.zeros((800, 400, 3), dtype=np.uint8))
        points = {name: point.copy() for name, point in prediction.points.items()}
        points["upper_left"][0], points["upper_right"][0] = (
            points["upper_right"][0],
            points["upper_left"][0],
        )
        points["lower_left"][0], points["lower_right"][0] = (
            points["lower_right"][0],
            points["lower_left"][0],
        )

        warnings = coordinate_geometry_warnings(points, prediction.lines, (800, 400, 3))

        self.assertFalse(any("水平方向の間にありません" in warning for warning in warnings))

    def test_reversed_center_vertical_order_names_points_three_and_six(self) -> None:
        prediction = FakeAdapter().predict(np.zeros((800, 400, 3), dtype=np.uint8))
        for upper_y in (420.0, 430.0):
            with self.subTest(upper_y=upper_y):
                points = {name: point.copy() for name, point in prediction.points.items()}
                points["upper_center"][1] = upper_y

                warnings = coordinate_geometry_warnings(points, prediction.lines, (800, 400, 3))

                self.assertTrue(any("点3" in warning and "点6" in warning and "上下方向" in warning for warning in warnings))

    def test_side_is_required_when_filename_has_no_laterality(self) -> None:
        with self.assertRaises(SideRequiredError):
            resolve_side(Path("patient.png"), None)
        self.assertEqual(resolve_side(Path("patient.png"), "R"), "R")

    def test_partial_validation_metrics_are_reported_as_incomplete(self) -> None:
        adapter = FakeAdapter()
        adapter._info = ModelInfo(
            **{
                **adapter.info.__dict__,
                "val_metrics": {"loss": 0.1},
            }
        )
        prediction = adapter.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        warnings = model_quality_warnings(prediction, low_peak_threshold=0.35)
        self.assertTrue(any("mLDFAの平均絶対誤差" in warning for warning in warnings))
        self.assertTrue(any("MPTAの平均絶対誤差" in warning for warning in warnings))

    def test_raw_to_coordinates_angles_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "001L.png"
            write_test_image(image_path)
            result = KneeAnalysisService(
                FakeAdapter(),
                render_component_images=False,
            ).analyze_path(image_path)

            self.assertEqual(result.side, "L")
            self.assertEqual(set(result.prediction.points), {
                "hip", "upper_left", "upper_center", "upper_right",
                "lower_left", "lower_center", "lower_right", "ankle",
            })
            self.assertEqual(set(result.prediction.lines), {"upper_line", "lower_line"})
            for key in ("mldfa_angle", "mpta_angle", "jlca_angle", "hka_angle"):
                self.assertTrue(math.isfinite(float(result.measurement[key])))
            self.assertEqual(result.measurement["combined_image"].shape, result.raw_image.shape)
            for key in ("e_image", "g_image", "jlca_image", "hka_image"):
                self.assertIsNone(result.measurement[key])

            full_measurement, _debug = measure_from_named_points(
                result.raw_image,
                result.prediction.named_points_payload(),
                raw_path=result.raw_path,
                named_lines=result.prediction.named_lines_payload(),
                side=result.side,
                render_style=RENDER_STYLE_CLINICAL,
            )
            for key in ("e_image", "g_image", "jlca_image", "hka_image"):
                self.assertEqual(full_measurement[key].shape, result.raw_image.shape)
            np.testing.assert_array_equal(
                full_measurement["combined_image"],
                result.measurement["combined_image"],
            )
            for key in ("mldfa_angle", "mpta_angle", "jlca_angle", "hka_angle"):
                self.assertEqual(full_measurement[key], result.measurement[key])

            record = export_record(
                result,
                result.prediction.points,
                result.prediction.lines,
                result.measurement,
                app_version="test",
                manually_modified=False,
            )
            self.assertEqual(record["model"]["checkpoint_sha256"], "a" * 64)
            self.assertFalse(record["analysis"]["manually_modified"])
            self.assertEqual(set(record["angles_deg"]), {"mLDFA", "MPTA", "JLCA", "HKA"})
            self.assertNotIn("path", record["source"])

            json_path = root / "result.json"
            overlay_path = root / "result.png"
            write_json(json_path, record)
            write_overlay(overlay_path, result.measurement["combined_image"])
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["source"]["filename"], image_path.name)
            self.assertGreater(overlay_path.stat().st_size, 0)

            bundle_json = root / "bundle.json"
            bundle_overlay = root / "bundle.png"
            write_result_bundle(bundle_json, record, bundle_overlay, result.measurement["combined_image"])
            self.assertTrue(bundle_json.is_file())
            self.assertTrue(bundle_overlay.is_file())

            edited_points = {name: point.copy() for name, point in result.prediction.points.items()}
            edited_points["hip"][0] += 5.0
            edited_measurement = measurement_from_coordinates(
                result.raw_image,
                result.raw_path,
                result.side,
                edited_points,
                result.prediction.lines,
            )
            edited = export_record(
                result,
                edited_points,
                result.prediction.lines,
                edited_measurement,
                app_version="test",
                manually_modified=True,
                edited_keys={"hip"},
            )
            self.assertEqual(edited["points"]["hip"]["source"], "manual")
            self.assertEqual(edited["analysis"]["edited_keys"], ["hip"])
            self.assertNotEqual(edited["points"]["hip"]["x"], edited["points"]["hip"]["model_prediction"]["x"])

    def test_manual_degenerate_axis_is_rejected(self) -> None:
        adapter = FakeAdapter()
        image = np.zeros((800, 400, 3), dtype=np.uint8)
        prediction = adapter.predict(image)
        points = {name: point.copy() for name, point in prediction.points.items()}
        points["hip"] = points["upper_center"].copy()
        with self.assertRaises(InferenceError):
            measurement_from_coordinates(
                image,
                Path("001L.png"),
                "L",
                points,
                prediction.lines,
            )


class CheckpointTests(unittest.TestCase):
    def test_external_legacy_checkpoint_requires_manifest(self) -> None:
        try:
            import torch
            from knee_keypoint_model import SmallHeatmapNet
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            model = SmallHeatmapNet(out_channels=len(EXPECTED_KEYPOINT_NAMES))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "keypoint_names": EXPECTED_KEYPOINT_NAMES,
                    "image_width": 128,
                    "image_height": 160,
                    "stride": 4,
                },
                path,
            )
            adapter = SmallHeatmapV1Adapter(
                ModelSpec(adapter="small_heatmap_v1", checkpoint=path, device="cpu")
            )
            with self.assertRaisesRegex(ModelLoadError, "スキーマバージョン1"):
                adapter.load()

    def test_incompatible_keypoint_schema_is_rejected_before_activation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save(
                {
                    "model_state": {},
                    "keypoint_names": ("wrong",),
                    "image_width": 128,
                    "image_height": 160,
                    "stride": 4,
                },
                path,
            )
            adapter = SmallHeatmapV1Adapter(ModelSpec(adapter="small_heatmap_v1", checkpoint=path, device="cpu"))
            with self.assertRaisesRegex(ModelLoadError, "ランドマーク定義"):
                adapter.load()

    def test_non_finite_checkpoint_is_rejected(self) -> None:
        try:
            import torch
            from knee_keypoint_model import SmallHeatmapNet
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.pt"
            model = SmallHeatmapNet(out_channels=len(EXPECTED_KEYPOINT_NAMES))
            with torch.no_grad():
                model.head.bias[0] = float("nan")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "keypoint_names": EXPECTED_KEYPOINT_NAMES,
                    "image_width": 128,
                    "image_height": 160,
                    "stride": 4,
                },
                path,
            )
            adapter = SmallHeatmapV1Adapter(
                ModelSpec(
                    adapter="small_heatmap_v1",
                    checkpoint=path,
                    device="cpu",
                    options={"allow_legacy_checkpoint": True},
                )
            )
            with self.assertRaisesRegex(ModelLoadError, "NaNまたは無限大"):
                adapter.load()

    def test_current_checkpoint_end_to_end_when_available(self) -> None:
        checkpoint = load_app_config().model.checkpoint
        image_path = Path("images/annotation_processed_combined/015R_pre_bone_raw.jpg")
        if not checkpoint.is_file() or not image_path.is_file():
            self.skipTest("Local ignored checkpoint/test image is not available")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        adapter = SmallHeatmapV1Adapter(load_app_config().model)
        adapter.load()
        result: AnalysisResult = KneeAnalysisService(adapter).analyze_path(image_path, requested_side="R")
        self.assertEqual(len(result.prediction.peak_scores), 12)
        self.assertEqual(result.side, "R")
        self.assertEqual(result.prediction.model_info.checkpoint_sha256, sha256_file(checkpoint))
        self.assertEqual(result.prediction.model_info.checkpoint_schema_version, 1)
        self.assertEqual(result.prediction.model_info.decoder_id, "local_centroid_3x3_residual_v1")
        for key in ("mldfa_angle", "mpta_angle", "jlca_angle", "hka_angle"):
            self.assertTrue(math.isfinite(float(result.measurement[key])))


if __name__ == "__main__":
    unittest.main()

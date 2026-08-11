from __future__ import annotations

import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from knee_xray.ui.knee_measurement_app import (
    INPUT_SCOPE_LABELS,
    SCREEN_SIDE_LABELS,
    SCREEN_SIDE_UNSELECTED,
    KneeMeasurementApp,
    measurement_export_stem,
)
from knee_xray.inference.knee_model_runtime import ModelSelection


class _Variable:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _ImmediateThread:
    def __init__(self, *, target: object, **_kwargs: object) -> None:
        self.target = target

    def start(self) -> None:
        self.target()


class BilateralGuiTests(unittest.TestCase):
    def _bilateral_app(self) -> KneeMeasurementApp:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.input_scope_var = _Variable(INPUT_SCOPE_LABELS["bilateral"])
        app.screen_side_var = _Variable(SCREEN_SIDE_LABELS["left"])
        app.roi_split_var = _Variable(50.0)
        app.raw_image = np.zeros((800, 1000, 3), dtype=np.uint8)
        app.raw_path = Path("001L_bilateral.png")
        app.roi_confirmed = False
        app.confirmed_crop_box = None
        app.roi_selection_method = ""
        return app

    def test_candidate_roi_uses_selected_screen_side_and_adjustable_divider(self) -> None:
        app = self._bilateral_app()

        self.assertEqual(app._candidate_crop_box(), (0, 0, 580, 800))

        app.screen_side_var.set(SCREEN_SIDE_LABELS["right"])
        app.roi_split_var.set(43.0)
        self.assertEqual(app._candidate_crop_box(), (350, 0, 1000, 800))

    def test_bilateral_export_stem_adds_anatomical_side_only_for_bilateral_input(self) -> None:
        self.assertEqual(
            measurement_export_stem(
                "patient:01",
                input_scope="bilateral raster X-ray",
                side="L",
            ),
            "patient_01_L",
        )
        self.assertEqual(
            measurement_export_stem(
                "patient:01",
                input_scope="single-leg raster X-ray",
                side="L",
            ),
            "patient_01",
        )
        with self.assertRaises(ValueError):
            measurement_export_stem(
                "patient:01",
                input_scope="bilateral raster X-ray",
                side="Auto",
            )

    def test_wide_image_stays_in_single_leg_flow_until_doctor_requests_crop(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.task_id = 0
        app.input_scope_var = _Variable(INPUT_SCOPE_LABELS["single"])
        app.raw_path = None
        app.raw_image = None
        app.roi_confirmed = False
        app.confirmed_crop_box = None
        app.roi_selection_method = ""
        app.path_var = _Variable()
        app.side_var = _Variable()
        app.screen_side_var = _Variable()
        app.roi_status_var = _Variable()
        app.result_state_var = _Variable()
        app.status_var = _Variable()
        app.warning_title_var = _Variable()
        app.side_source_hint = "unknown"
        app.input_view = mock.Mock()
        app.input_view.has_image = False
        app.model_specs = {"mixed": object()}
        selection = ModelSelection("auto", "mixed", "auto_fallback_unknown")
        app._resolve_model_selection = mock.Mock(return_value=selection)
        app._activate_model_selection = mock.Mock()
        app._clear_analysis = mock.Mock()
        app._update_roi_panel_visibility = mock.Mock()
        app._set_warning_text = mock.Mock()
        app._set_warning_banner = mock.Mock()
        app.notebook = mock.Mock()
        app.quality_tab = object()
        app._refresh_action_states = mock.Mock()

        with mock.patch(
            "knee_xray.ui.knee_measurement_app.read_color",
            return_value=np.zeros((1000, 700, 3), dtype=np.uint8),
        ):
            app.open_path(Path("001L_wide.png"))

        self.assertEqual(app._input_scope(), "single")
        app._activate_model_selection.assert_called_once_with(selection, run_inference=True)
        app._update_roi_panel_visibility.assert_not_called()

    def test_crop_button_explicitly_enters_and_leaves_bilateral_mode(self) -> None:
        app = self._bilateral_app()
        app.input_scope_var.set(INPUT_SCOPE_LABELS["single"])
        app.busy = False
        app._on_input_scope_changed = mock.Mock()

        app._toggle_bilateral_mode()

        self.assertEqual(app._input_scope(), "bilateral")
        app._on_input_scope_changed.assert_called_once_with()

        app._on_input_scope_changed.reset_mock()
        app._toggle_bilateral_mode()

        self.assertEqual(app._input_scope(), "single")
        app._on_input_scope_changed.assert_called_once_with()

    def test_entering_bilateral_mode_clears_result_and_waits_for_roi(self) -> None:
        app = self._bilateral_app()
        app.input_scope_var.set(INPUT_SCOPE_LABELS["single"])
        app.busy = False
        app.task_id = 3
        app.analysis = object()
        app.points = {"hip": np.array([1.0, 2.0])}
        app.measurement = {"mldfa_angle": 90.0}
        app.result_state_var = _Variable()
        app.roi_status_var = _Variable()
        app.status_var = _Variable()
        app.input_view = mock.Mock()
        app._clear_analysis = mock.Mock()
        app._update_roi_panel_visibility = mock.Mock()
        app._suggest_screen_side = mock.Mock()
        app._ensure_input_preview = mock.Mock(return_value=True)
        app._start_inference = mock.Mock()
        app._refresh_action_states = mock.Mock()

        app._toggle_bilateral_mode()

        self.assertEqual(app._input_scope(), "bilateral")
        app._clear_analysis.assert_called_once_with(keep_raw=True)
        app._update_roi_panel_visibility.assert_called_once_with()
        app._start_inference.assert_not_called()
        self.assertFalse(app.roi_confirmed)
        self.assertIsNone(app.confirmed_crop_box)
        self.assertEqual(app.result_state_var.get(), "ROI確認待ち")
        self.assertEqual(app.task_id, 4)

    def test_returning_to_single_clears_roi_and_runs_full_image_once(self) -> None:
        app = self._bilateral_app()
        app.busy = False
        app.task_id = 8
        app.analysis = object()
        app.points = {"hip": np.array([1.0, 2.0])}
        app.measurement = {"mldfa_angle": 90.0}
        app.roi_confirmed = True
        app.confirmed_crop_box = (0, 0, 580, 800)
        app.roi_selection_method = "patient_side_convention_overlap_confirmed"
        app.result_state_var = _Variable()
        app.roi_status_var = _Variable()
        app.status_var = _Variable()
        app.input_view = mock.Mock()
        app._clear_analysis = mock.Mock()
        app._update_roi_panel_visibility = mock.Mock()
        app._model_selection_is_active = mock.Mock(return_value=True)
        app._start_inference = mock.Mock()
        app._refresh_action_states = mock.Mock()

        app._toggle_bilateral_mode()

        self.assertEqual(app._input_scope(), "single")
        app._clear_analysis.assert_called_once_with(keep_raw=True)
        app._update_roi_panel_visibility.assert_called_once_with()
        app._start_inference.assert_called_once_with()
        self.assertFalse(app.roi_confirmed)
        self.assertIsNone(app.confirmed_crop_box)
        self.assertEqual(app.roi_selection_method, "")
        self.assertEqual(app.screen_side_var.get(), SCREEN_SIDE_UNSELECTED)
        self.assertEqual(app.task_id, 9)

    def test_roi_controls_are_visible_only_after_bilateral_button_mode(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.input_scope_var = _Variable(INPUT_SCOPE_LABELS["single"])
        app.roi_panel = mock.Mock()
        app.roi_panel.winfo_manager.return_value = ""
        app.bilateral_crop_button = mock.Mock()

        app._update_roi_panel_visibility()

        app.roi_panel.pack_forget.assert_called_once_with()
        self.assertEqual(
            app.bilateral_crop_button.configure.call_args.kwargs["text"],
            "両側画像を切り出す",
        )

        app.roi_panel.reset_mock()
        app.input_scope_var.set(INPUT_SCOPE_LABELS["bilateral"])
        app._update_roi_panel_visibility()

        app.roi_panel.pack.assert_called_once_with(fill="x", pady=(6, 0))
        self.assertEqual(
            app.bilateral_crop_button.configure.call_args.kwargs["text"],
            "片側画像に戻す",
        )

    def test_unconfirmed_bilateral_image_cannot_start_inference(self) -> None:
        app = self._bilateral_app()
        app._model_selection_is_active = mock.Mock(return_value=True)
        app._show_roi_required_warning = mock.Mock()

        app._start_inference()

        app._show_roi_required_warning.assert_called_once_with()

    def test_landmark_outside_confirmed_roi_blocks_export_safety_check(self) -> None:
        app = self._bilateral_app()
        app.roi_confirmed = True
        app.confirmed_crop_box = (0, 0, 580, 800)
        app.points = {"hip": np.array([600.0, 100.0], dtype=np.float32)}
        app.lines = {
            "upper_line": {
                "p1": np.array([100.0, 400.0], dtype=np.float32),
                "p2": np.array([200.0, 400.0], dtype=np.float32),
            }
        }

        self.assertFalse(app._coordinates_within_confirmed_roi())
        self.assertTrue(app._roi_coordinate_warnings())

    def test_confirmed_roi_is_forwarded_to_runtime_with_provenance(self) -> None:
        app = self._bilateral_app()
        app.roi_confirmed = True
        app.confirmed_crop_box = (0, 0, 580, 800)
        app.roi_selection_method = "patient_side_convention_overlap_confirmed"
        app.side_source_hint = "manual_override"
        app.side_var = _Variable("L")
        app.task_id = 4
        app.task_events = queue.Queue()
        app.active_model_key = "bone"
        app.model_selection = ModelSelection("bone", "bone", "manual_override")
        analysis = SimpleNamespace()
        analyze_path = mock.Mock(return_value=analysis)
        app.service = SimpleNamespace(
            adapter=SimpleNamespace(info=SimpleNamespace(checkpoint_sha256="a" * 64)),
            analyze_path=analyze_path,
        )
        app._model_selection_is_active = mock.Mock(return_value=True)
        app._effective_side = mock.Mock(return_value="L")
        app._clear_analysis = mock.Mock()
        app._set_busy = mock.Mock()
        app.result_state_var = _Variable()

        with (
            mock.patch("knee_xray.ui.knee_measurement_app.threading.Thread", _ImmediateThread),
            mock.patch("knee_xray.ui.knee_measurement_app.replace", side_effect=lambda value, **_kwargs: value),
        ):
            app._start_inference()

        analyze_path.assert_called_once_with(
            app.raw_path,
            requested_side="L",
            crop_box=(0, 0, 580, 800),
            roi_selection_method="patient_side_convention_overlap_confirmed",
            roi_confirmed=True,
        )
        event = app.task_events.get_nowait()
        self.assertEqual(event["crop_box"], (0, 0, 580, 800))
        self.assertEqual(event["input_scope"], "bilateral")

    def test_doctor_confirmation_records_conventional_overlap_policy(self) -> None:
        app = self._bilateral_app()
        app.screen_side_var.set(SCREEN_SIDE_LABELS["right"])
        app.busy = False
        app.roi_status_var = _Variable()
        app.input_view = mock.Mock()
        app._effective_side = mock.Mock(return_value="L")
        app._clear_analysis = mock.Mock()
        app._start_inference = mock.Mock()

        app._confirm_bilateral_roi()

        self.assertTrue(app.roi_confirmed)
        self.assertEqual(app.confirmed_crop_box, (420, 0, 1000, 800))
        self.assertEqual(
            app.roi_selection_method,
            "patient_side_convention_overlap_confirmed",
        )
        app._start_inference.assert_called_once_with()

    def test_roi_change_invalidates_result_and_requires_confirmation(self) -> None:
        app = self._bilateral_app()
        app.analysis = object()
        app.points = {"hip": np.array([1.0, 2.0])}
        app.measurement = {"mldfa_angle": 90.0}
        app.roi_confirmed = True
        app.confirmed_crop_box = (0, 0, 500, 800)
        app.task_id = 2
        app.busy = False
        app.roi_split_label_var = _Variable()
        app.result_state_var = _Variable()
        app.roi_status_var = _Variable()
        app.status_var = _Variable()
        app.input_view = mock.Mock()
        app._clear_analysis = mock.Mock()
        app._refresh_action_states = mock.Mock()

        app._on_roi_split_changed("55")

        app._clear_analysis.assert_called_once_with(keep_raw=True)
        self.assertFalse(app.roi_confirmed)
        self.assertIsNone(app.confirmed_crop_box)
        self.assertEqual(app.result_state_var.get(), "ROI確認待ち")
        self.assertIn("55%", str(app.roi_split_label_var.get()))
        self.assertEqual(app.task_id, 3)

    def test_stale_result_for_previous_roi_is_ignored(self) -> None:
        app = self._bilateral_app()
        app.roi_confirmed = True
        app.confirmed_crop_box = (0, 0, 500, 800)
        app.active_model_key = "bone"
        app.analysis = "current"
        app._set_busy = mock.Mock()

        app._finish_inference_event(
            {
                "ok": True,
                "value": object(),
                "raw_path": app.raw_path,
                "model_key": "bone",
                "input_scope": "bilateral",
                "crop_box": (500, 0, 1000, 800),
            }
        )

        self.assertEqual(app.analysis, "current")
        app._set_busy.assert_not_called()


if __name__ == "__main__":
    unittest.main()

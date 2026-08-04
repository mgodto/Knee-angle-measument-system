from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from knee_measurement_app import (
    INPUT_SCOPE_LABELS,
    SCREEN_SIDE_LABELS,
    SCREEN_SIDE_UNSELECTED,
    KneeMeasurementApp,
)
from knee_model_runtime import (
    AnalysisResult,
    LandmarkPrediction,
    ModelInfo,
    SideRequiredError,
    measurement_from_coordinates,
    measurement_out_of_range_angles,
)


class _Variable:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _geometry() -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
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
    return points, lines


def _analysis(side: str = "L", model_warnings: tuple[str, ...] = ()) -> AnalysisResult:
    raw_image = np.zeros((800, 400, 3), dtype=np.uint8)
    points, lines = _geometry()
    raw_path = Path("001L.png")
    measurement = measurement_from_coordinates(
        raw_image,
        raw_path,
        side,
        points,
        lines,
        render_component_images=False,
    )
    info = ModelInfo(
        adapter="test",
        display_name="Test",
        version="1",
        cohort="test",
        checkpoint_path="/tmp/test.pt",
        checkpoint_sha256="a" * 64,
        device="cpu",
        input_width=128,
        input_height=256,
        stride=4,
        epoch=1,
        val_metrics={},
        checkpoint_schema_version=1,
        architecture_id="test",
        preprocessing_id="test",
        metadata_source="test",
        training_manifest_sha256=None,
    )
    prediction = LandmarkPrediction(
        schema_version=1,
        points={name: point.copy() for name, point in points.items()},
        lines={name: {key: point.copy() for key, point in value.items()} for name, value in lines.items()},
        peak_scores={},
        model_info=info,
        elapsed_ms=1.0,
    )
    return AnalysisResult(
        raw_path=raw_path,
        raw_image=raw_image,
        side=side,
        prediction=prediction,
        measurement=measurement,
        model_warnings=model_warnings,
        warnings=model_warnings,
        source_sha256="b" * 64,
        total_elapsed_ms=1.0,
        side_source="filename",
    )


class P1GuiSafetyTests(unittest.TestCase):
    def test_model_preference_save_receives_loaded_checkpoint_sha(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.raw_path = None
        app.preference_warning = "old"
        app.analysis = None
        app.warning_title_var = _Variable("AIモデル設定エラー")
        app._set_warning_text = mock.Mock()
        app._set_warning_banner = mock.Mock()
        app.notebook = mock.Mock()
        app.quality_tab = object()
        app._update_model_badge = mock.Mock()
        app._update_model_details = mock.Mock()
        app._set_busy = mock.Mock()
        spec = object()
        service = object()
        info = SimpleNamespace(checkpoint_sha256="c" * 64)
        event = {
            "ok": True,
            "value": {
                "spec": spec,
                "service": service,
                "info": info,
                "preference_action": "save",
            },
        }

        with mock.patch("knee_measurement_app.save_model_preference") as save_preference:
            app._finish_model_event(event)

        save_preference.assert_called_once_with(spec, expected_sha256="c" * 64)
        self.assertIs(app.service, service)
        self.assertIsNone(app.preference_warning)
        app._set_warning_banner.assert_called_with(
            "確認事項：解析後に表示します",
            "neutral",
            False,
        )

    def test_missing_side_sets_visible_warning_and_tab_count(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.raw_path = Path("patient.png")
        app.service = object()
        app._effective_side = mock.Mock(side_effect=SideRequiredError("side required"))
        app._clear_measurement_display = mock.Mock()
        app.result_state_var = _Variable()
        app.status_var = _Variable()
        app.warning_title_var = _Variable()
        app._set_warning_text = mock.Mock()
        app._set_warning_banner = mock.Mock()
        app.notebook = mock.Mock()
        app.quality_tab = object()

        app._start_inference()

        app._set_warning_banner.assert_called_once_with(
            "⚠ 左右を選択してください",
            "warning",
            True,
        )
        app.notebook.tab.assert_called_once_with(app.quality_tab, text="確認事項（1）・AIモデル")

    def test_side_change_invalidates_old_result_and_reruns_inference(self) -> None:
        analysis = _analysis("L")
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.analysis = analysis
        app.raw_path = analysis.raw_path
        app.raw_image = analysis.raw_image
        app.side_var = _Variable("R")
        app.side_source_hint = "filename"
        app.busy = False
        app.service = object()
        app.task_id = 1
        app._clear_analysis = mock.Mock()
        app._model_selection_is_active = mock.Mock(return_value=True)
        app._start_inference = mock.Mock()

        app._on_side_changed()

        app._clear_analysis.assert_called_once_with(keep_raw=True)
        app._start_inference.assert_called_once_with()
        self.assertEqual(app.side_source_hint, "manual_override")
        self.assertEqual(app.task_id, 2)

    def test_case_warning_selects_details_but_model_only_warning_does_not(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.analysis = _analysis("L", model_warnings=("モデル確認事項",))
        app.points = app.analysis.prediction.points
        app.lines = app.analysis.prediction.lines
        app.measurement = dict(app.analysis.measurement)
        app.measurement["mpta_angle"] = 143.5
        app.edited_keys = set()
        app.preference_warning = None
        app.warning_title_var = _Variable()
        app.quality_tab = object()
        app.notebook = mock.Mock()
        app._set_warning_text = mock.Mock()
        app._set_warning_banner = mock.Mock()
        app._show_quality_tab = mock.Mock()
        app._update_model_details = mock.Mock()
        app._update_angle_alerts = mock.Mock()

        app._update_quality_display(select_details=True)
        app._show_quality_tab.assert_called_once_with()
        self.assertIn("確認事項：", app.warning_title_var.get())
        self.assertIn("必ず詳細を確認", app._set_warning_banner.call_args.args[0])

        app._show_quality_tab.reset_mock()
        app.measurement.update(
            mldfa_angle=90.0,
            mpta_angle=90.0,
            jlca_angle=0.0,
            hka_angle=0.0,
        )
        app._update_quality_display(select_details=True)
        app._show_quality_tab.assert_not_called()

        app.points = {name: point.copy() for name, point in app.points.items()}
        app.points["upper_center"][0] = 350.0
        app._update_quality_display(select_details=True)
        warning_lines = app._set_warning_text.call_args.args[0]
        self.assertTrue(any("点3" in warning and "mLDFA" in warning for warning in warning_lines))
        app._show_quality_tab.assert_called_once_with()

    def test_clear_analysis_resets_visible_warning_state(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.analysis = _analysis()
        app.points = app.analysis.prediction.points
        app.lines = app.analysis.prediction.lines
        app.measurement = app.analysis.measurement
        app.edited_keys = {"hip"}
        app.history = [{}]
        app.drag_target = ("point", "hip")
        app.coordinate_tree = mock.Mock()
        app.coordinate_tree.get_children.return_value = ()
        app._clear_measurement_display = mock.Mock()
        app.warning_title_var = _Variable("確認事項：5件")
        app.result_state_var = _Variable("手動修正あり・要確認")
        app._set_warning_text = mock.Mock()
        app._set_warning_banner = mock.Mock()
        app.quality_tab = object()
        app.notebook = mock.Mock()
        app.raw_path = app.analysis.raw_path
        app.raw_image = app.analysis.raw_image
        app.displayed_source_sha256 = app.analysis.source_sha256
        app.side_source_hint = "filename"
        app.side_var = _Variable("L")
        app.input_scope_var = _Variable(INPUT_SCOPE_LABELS["bilateral"])
        app.screen_side_var = _Variable(SCREEN_SIDE_LABELS["right"])
        app.roi_split_var = _Variable("61")
        app.roi_split_label_var = _Variable()
        app.roi_confirmed = True
        app.confirmed_crop_box = (100, 0, 400, 800)
        app.roi_selection_method = "manual_screen_side_divider_overlap_confirmed"
        app.roi_panel = mock.Mock()
        app.bilateral_crop_button = mock.Mock()
        app.input_view = mock.Mock()
        app._refresh_action_states = mock.Mock()

        app._clear_analysis(keep_raw=False)

        self.assertEqual(app.warning_title_var.get(), "解析結果はありません")
        self.assertEqual(app.result_state_var.get(), "入力待ち")
        self.assertEqual(app.input_scope_var.get(), INPUT_SCOPE_LABELS["single"])
        self.assertEqual(app.screen_side_var.get(), SCREEN_SIDE_UNSELECTED)
        self.assertEqual(app.roi_split_var.get(), 50.0)
        self.assertFalse(app.roi_confirmed)
        self.assertIsNone(app.confirmed_crop_box)
        self.assertEqual(app.roi_selection_method, "")
        app.roi_panel.pack_forget.assert_called_once_with()
        app._set_warning_banner.assert_called_once_with(
            "確認事項：解析後に表示します",
            "neutral",
            False,
        )
        app.notebook.tab.assert_called_once_with(app.quality_tab, text="確認事項（0）・AIモデル")
        app.notebook.select.assert_called_once_with(0)

    def test_reset_to_prediction_does_not_overwrite_recalculation_error_state(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.busy = False
        app.analysis = _analysis()
        app.history = []
        app.points = {}
        app.lines = {}
        app.edited_keys = {"hip"}
        app.result_state_var = _Variable("手動修正あり・要確認")
        app._snapshot = mock.Mock(return_value={})
        app.input_view = mock.Mock()
        app._refresh_coordinate_table = mock.Mock()
        app._refresh_action_states = mock.Mock()

        def fail_recalculation() -> None:
            app.result_state_var.set("手動修正・計算エラー")

        app._recalculate_after_edit = fail_recalculation
        app.reset_to_prediction()

        self.assertEqual(app.result_state_var.get(), "手動修正・計算エラー")

    def test_angle_range_helper_includes_only_values_outside_inclusive_boundaries(self) -> None:
        boundary_values = {
            "mldfa_angle": 45.0,
            "mpta_angle": 135.0,
            "jlca_angle": -30.0,
            "hka_angle": 45.0,
        }
        self.assertEqual(measurement_out_of_range_angles(boundary_values), ())

        outside_values = {
            "mldfa_angle": 44.9,
            "mpta_angle": 135.1,
            "jlca_angle": 30.1,
            "hka_angle": -45.1,
        }
        self.assertEqual(
            measurement_out_of_range_angles(outside_values),
            ("mLDFA", "MPTA", "JLCA", "HKA"),
        )

    def test_angle_cards_use_text_and_color_for_out_of_range_values(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.measurement = {
            "mldfa_angle": 90.0,
            "mpta_angle": 143.5,
            "jlca_angle": 62.8,
            "hka_angle": 0.0,
        }
        app.angle_cards = {key: mock.Mock() for key in ("mLDFA", "MPTA", "JLCA", "HKA")}
        app.angle_name_labels = {key: mock.Mock() for key in app.angle_cards}
        app.angle_value_labels = {key: mock.Mock() for key in app.angle_cards}

        app._update_angle_alerts()

        self.assertEqual(app.angle_name_labels["MPTA"].configure.call_args.kwargs["text"], "MPTA ⚠")
        self.assertEqual(app.angle_name_labels["JLCA"].configure.call_args.kwargs["text"], "JLCA ⚠")
        self.assertEqual(app.angle_name_labels["mLDFA"].configure.call_args.kwargs["text"], "mLDFA")
        self.assertEqual(app.angle_name_labels["HKA"].configure.call_args.kwargs["text"], "HKA")


if __name__ == "__main__":
    unittest.main()

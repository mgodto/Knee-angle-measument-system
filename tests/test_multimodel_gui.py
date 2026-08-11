from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from knee_xray.ui.knee_measurement_app import KneeMeasurementApp, MODEL_MODE_LABELS
from knee_xray.inference.knee_model_runtime import ModelSelection, ModelSpec


class _Variable:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _spec(name: str) -> ModelSpec:
    return ModelSpec(adapter="fake", checkpoint=Path(f"/{name}.pt"), display_name=name)


def _service(digest: str = "a" * 64) -> SimpleNamespace:
    info = SimpleNamespace(
        checkpoint_sha256=digest,
        short_hash=digest[:12],
        device="cpu",
        display_name="Test model",
        version="1",
        input_width=320,
        input_height=256,
        cohort="test",
        metadata_source="checkpoint_manifest",
        val_metrics={},
    )
    return SimpleNamespace(adapter=SimpleNamespace(info=info))


class MultiModelGuiTests(unittest.TestCase):
    def test_opening_new_image_resets_manual_choice_to_auto_route(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.task_id = 4
        app.model_specs = {key: _spec(key) for key in ("bone", "tka", "mixed")}
        app.config = SimpleNamespace(auto_fallback_model_key="mixed")
        app.raw_path = None
        app.service = _service()
        app.path_var = _Variable()
        app.side_var = _Variable()
        app._clear_analysis = mock.Mock()
        app._activate_model_selection = mock.Mock()

        path = Path("001L_post_TKA.png").resolve()
        app.open_path(path)

        selection = app._activate_model_selection.call_args.args[0]
        self.assertEqual(selection, ModelSelection("auto", "tka", "filename"))
        self.assertTrue(app._activate_model_selection.call_args.kwargs["run_inference"])
        self.assertEqual(app.raw_path, path)
        self.assertEqual(app.task_id, 5)

    def test_cached_manual_model_switch_reuses_service_and_reruns(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        mixed_spec = _spec("mixed")
        tka_spec = _spec("tka")
        mixed_service = _service("a" * 64)
        tka_service = _service("b" * 64)
        app.config = SimpleNamespace(auto_fallback_model_key="mixed")
        app.model_specs = {"mixed": mixed_spec, "tka": tka_spec}
        app.model_cache = {"tka": (tka_spec, tka_service)}
        app.model_selection = ModelSelection("auto", "mixed", "auto_fallback_unknown")
        app.model_mode_var = _Variable(MODEL_MODE_LABELS["auto"])
        app.model_source_var = _Variable()
        app.service = mixed_service
        app.model_spec = mixed_spec
        app.active_model_key = "mixed"
        app.pending_model_key = None
        app.preference_warning = None
        app.raw_path = Path("001L_post_TKA.png")
        app.task_id = 9
        app._clear_analysis = mock.Mock()
        app._update_model_badge = mock.Mock()
        app._update_model_details = mock.Mock()
        app._set_busy = mock.Mock()
        app._start_inference = mock.Mock()
        app._start_model_load = mock.Mock()

        app._activate_model_selection(ModelSelection("tka", "tka", "manual_override"))

        self.assertIs(app.service, tka_service)
        self.assertEqual(app.active_model_key, "tka")
        app._start_model_load.assert_not_called()
        app._clear_analysis.assert_called_once_with(keep_raw=True)
        app._start_inference.assert_called_once_with()

    def test_stale_inference_for_previous_image_is_ignored(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.raw_path = Path("new_001L.png")
        app.active_model_key = "tka"
        app.analysis = "current"
        app._set_busy = mock.Mock()

        app._finish_inference_event(
            {
                "ok": True,
                "value": object(),
                "raw_path": Path("old_001L.png"),
                "model_key": "bone",
                "checkpoint_sha256": "a" * 64,
            }
        )

        self.assertEqual(app.analysis, "current")
        app._set_busy.assert_not_called()

    def test_model_details_show_route_and_full_weight_sha(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        digest = "c" * 64
        app.service = _service(digest)
        app.active_model_key = "tka"
        app.model_selection = ModelSelection("auto", "tka", "filename")
        app.model_detail_var = _Variable()

        app._update_model_details()

        detail = app.model_detail_var.get()
        self.assertIn("自動判定 → TKA", detail)
        self.assertIn("ファイル名／フォルダから自動判定", detail)
        self.assertIn(digest, detail)

    def test_external_model_picker_keeps_compatibility(self) -> None:
        app = KneeMeasurementApp.__new__(KneeMeasurementApp)
        app.busy = False
        app.default_model_spec = _spec("mixed")
        app._start_model_load = mock.Mock()

        with mock.patch("knee_xray.ui.knee_measurement_app.filedialog.askopenfilename", return_value="/tmp/custom.pt"):
            app.browse_weight()

        call = app._start_model_load.call_args
        self.assertEqual(call.kwargs["model_key"], "external")
        self.assertEqual(call.kwargs["selection"].source, "external_model")
        self.assertEqual(call.kwargs["preference_action"], "save")
        self.assertEqual(call.args[0].checkpoint, Path("/tmp/custom.pt").resolve())


if __name__ == "__main__":
    unittest.main()

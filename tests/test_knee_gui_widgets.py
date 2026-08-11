from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from knee_xray.ui.knee_gui_widgets import InteractiveImageCanvas, RESAMPLING


class _FakeCanvas:
    def __init__(self) -> None:
        self.idle_callbacks: list[object] = []

    def after_idle(self, callback: object) -> str:
        self.idle_callbacks.append(callback)
        return "after-1"

    def canvasx(self, value: int) -> float:
        return float(value)

    def canvasy(self, value: int) -> float:
        return float(value)


class _RenderCanvas:
    def winfo_width(self) -> int:
        return 800

    def winfo_height(self) -> int:
        return 600

    def itemconfigure(self, *_args: object, **_kwargs: object) -> None:
        pass

    def coords(self, *_args: object) -> None:
        pass

    def configure(self, **_kwargs: object) -> None:
        pass

    def delete(self, *_args: object) -> None:
        pass

    def tag_raise(self, *_args: object) -> None:
        pass


class _FakeImage:
    width = 1_000
    height = 1_000
    size = (width, height)

    def __init__(self) -> None:
        self.resize_calls: list[tuple[tuple[int, int], object]] = []

    def resize(self, size: tuple[int, int], resampling: object) -> object:
        self.resize_calls.append((size, resampling))
        return object()


def _view_with_image(width: int, height: int) -> InteractiveImageCanvas:
    view = InteractiveImageCanvas.__new__(InteractiveImageCanvas)
    view.min_zoom = 0.05
    view.max_zoom = 8.0
    view.zoom = 1.0
    view._base_image = Image.new("RGB", (width, height))
    return view


class InteractiveImageCanvasTests(unittest.TestCase):
    def test_render_zoom_is_limited_by_pixel_budget(self) -> None:
        view = _view_with_image(2_372, 2_881)

        view.zoom = view._clamp_zoom(8.0)
        display_w, display_h = view._display_size()

        self.assertLessEqual(display_w * display_h, view.MAX_RENDER_PIXELS)
        self.assertGreater(view.zoom, 1.0)
        self.assertLess(view.zoom, view.max_zoom)

    def test_normal_fit_zoom_is_not_changed_by_pixel_budget(self) -> None:
        view = _view_with_image(2_372, 2_881)

        self.assertAlmostEqual(view._clamp_zoom(0.24), 0.24)

    def test_multiple_wheel_events_are_coalesced_into_one_render(self) -> None:
        view = _view_with_image(1_000, 1_000)
        view.zoom = 0.5
        view._offset_x = 0.0
        view._offset_y = 0.0
        view._pending_zoom_factor = 1.0
        view._pending_zoom_anchor = None
        view._pending_zoom_after_id = None
        view.canvas = _FakeCanvas()
        render_calls: list[None] = []
        move_calls: list[tuple[float, float, int, int]] = []
        view._render = lambda: render_calls.append(None)
        view._move_view = lambda *args: move_calls.append(args)
        event = types.SimpleNamespace(num=None, delta=120, x=100, y=120)

        view._on_mousewheel(event)
        view._on_mousewheel(event)
        view._on_mousewheel(event)

        self.assertEqual(len(view.canvas.idle_callbacks), 1)
        self.assertEqual(render_calls, [])
        self.assertEqual(view.zoom, 0.5)
        view.canvas.idle_callbacks[0]()
        self.assertEqual(len(render_calls), 1)
        self.assertEqual(len(move_calls), 1)
        self.assertAlmostEqual(view.zoom, 0.5 * 1.15**3)

    def test_coordinate_mapping_is_preserved(self) -> None:
        view = _view_with_image(1_000, 1_000)
        view.zoom = 0.625
        view._offset_x = 17.0
        view._offset_y = 23.0
        view.canvas = _FakeCanvas()
        point = np.array([320.0, 640.0])

        canvas_x, canvas_y = view.image_to_canvas(point)
        mapped = view._event_to_image(types.SimpleNamespace(x=canvas_x, y=canvas_y))

        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped[0], point[0])
        self.assertAlmostEqual(mapped[1], point[1])

    def test_same_display_size_reuses_existing_photo(self) -> None:
        view = InteractiveImageCanvas.__new__(InteractiveImageCanvas)
        view.min_zoom = 0.05
        view.max_zoom = 8.0
        view.zoom = 0.5
        view._base_image = _FakeImage()
        view._photo = None
        view._rendered_size = None
        view._offset_x = 0.0
        view._offset_y = 0.0
        overlay_calls: list[None] = []
        view._overlay_drawer = lambda _view: overlay_calls.append(None)
        view._overlay_tag = "image_overlay"
        view._image_id = 1
        view._text_id = 2
        view.canvas = _RenderCanvas()

        with mock.patch("knee_xray.ui.knee_gui_widgets.ImageTk.PhotoImage", return_value=object()) as photo:
            view._render()
            view._render()

        self.assertEqual(view._base_image.resize_calls, [((500, 500), RESAMPLING.LANCZOS)])
        photo.assert_called_once()
        self.assertEqual(len(overlay_calls), 2)


if __name__ == "__main__":
    unittest.main()

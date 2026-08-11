#!/usr/bin/env python3

"""Reusable zoomable image canvas used by the measurement GUI."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk


RESAMPLING = getattr(Image, "Resampling", Image)
PointerCallback = Callable[[float, float], None]
OverlayDrawer = Callable[["InteractiveImageCanvas"], None]


class InteractiveImageCanvas(ttk.Frame):
    MAX_RENDER_PIXELS = 12_000_000

    def __init__(self, master: tk.Misc, empty_text: str, bg: str = "#0b1220") -> None:
        super().__init__(master)
        self.empty_text = empty_text
        self.bg = bg
        self.zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 8.0
        self._base_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._rendered_size: tuple[int, int] | None = None
        self._auto_fit_pending = False
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._on_press: PointerCallback | None = None
        self._on_drag: PointerCallback | None = None
        self._on_release: PointerCallback | None = None
        self._overlay_drawer: OverlayDrawer | None = None
        self._overlay_tag = "image_overlay"
        self._pending_zoom_factor = 1.0
        self._pending_zoom_anchor: tuple[int, int] | None = None
        self._pending_zoom_after_id: str | None = None

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self._image_id = self.canvas.create_image(0, 0, anchor="nw")
        self._text_id = self.canvas.create_text(
            0,
            0,
            text=empty_text,
            fill="#94a3b8",
            font=("TkDefaultFont", 15),
            justify="center",
            width=420,
        )

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Enter>", lambda _event: self.canvas.focus_set())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.reset_view())
        self.canvas.bind("<ButtonPress-1>", self._handle_press)
        self.canvas.bind("<B1-Motion>", self._handle_drag)
        self.canvas.bind("<ButtonRelease-1>", self._handle_release)
        self.clear()

    @property
    def overlay_tag(self) -> str:
        return self._overlay_tag

    @property
    def has_image(self) -> bool:
        return self._base_image is not None

    def set_pointer_callbacks(
        self,
        on_press: PointerCallback | None = None,
        on_drag: PointerCallback | None = None,
        on_release: PointerCallback | None = None,
    ) -> None:
        self._on_press = on_press
        self._on_drag = on_drag
        self._on_release = on_release

    def set_overlay_drawer(self, overlay_drawer: OverlayDrawer | None = None) -> None:
        self._overlay_drawer = overlay_drawer
        self.redraw_overlay()

    def clear(self, text: str | None = None) -> None:
        self._cancel_pending_zoom()
        self._base_image = None
        self._photo = None
        self._rendered_size = None
        self.zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._auto_fit_pending = False
        self.canvas.delete(self._overlay_tag)
        self.canvas.itemconfigure(self._image_id, image="", state="hidden")
        self.canvas.itemconfigure(self._text_id, text=text or self.empty_text, state="normal")
        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        self._center_text()

    def set_image(self, image_bgr: np.ndarray, reset_view: bool = True) -> None:
        self._cancel_pending_zoom()
        if image_bgr.ndim == 2:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        elif image_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._base_image = Image.fromarray(rgb)
        self._photo = None
        self._rendered_size = None
        x_view = self.canvas.xview()
        y_view = self.canvas.yview()
        if self.canvas.winfo_width() > 1 and self.canvas.winfo_height() > 1:
            self._auto_fit_pending = False
            if reset_view:
                self.reset_view()
            else:
                self._render()
                if x_view and y_view:
                    self.canvas.xview_moveto(x_view[0])
                    self.canvas.yview_moveto(y_view[0])
        else:
            self._auto_fit_pending = True
            self._render()

    def reset_view(self) -> None:
        if self._base_image is None:
            return
        self._cancel_pending_zoom()
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        fit_zoom = min(canvas_w / self._base_image.width, canvas_h / self._base_image.height)
        self.zoom = self._clamp_zoom(fit_zoom)
        self._render()
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def redraw_overlay(self) -> None:
        self.canvas.delete(self._overlay_tag)
        if self._base_image is None or self._overlay_drawer is None:
            return
        self._overlay_drawer(self)
        self.canvas.tag_raise(self._overlay_tag)

    def image_to_canvas(self, point: np.ndarray) -> tuple[float, float]:
        return (
            float(self._offset_x + point[0] * self.zoom),
            float(self._offset_y + point[1] * self.zoom),
        )

    def _on_configure(self, _event: object) -> None:
        if self._base_image is None:
            self._center_text()
        elif self._auto_fit_pending:
            self._auto_fit_pending = False
            self.reset_view()
        else:
            self._render()

    def _center_text(self) -> None:
        self.canvas.coords(self._text_id, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def _render(self) -> None:
        if self._base_image is None:
            return
        self.zoom = self._clamp_zoom(self.zoom)
        display_w = max(1, int(round(self._base_image.width * self.zoom)))
        display_h = max(1, int(round(self._base_image.height * self.zoom)))
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        self._offset_x = max((canvas_w - display_w) / 2.0, 0.0)
        self._offset_y = max((canvas_h - display_h) / 2.0, 0.0)
        display_size = (display_w, display_h)
        if self._photo is None or self._rendered_size != display_size:
            if display_size == self._base_image.size:
                resized = self._base_image
            else:
                resampling = (
                    RESAMPLING.LANCZOS
                    if display_w < self._base_image.width or display_h < self._base_image.height
                    else RESAMPLING.BILINEAR
                )
                resized = self._base_image.resize(display_size, resampling)
            self._photo = ImageTk.PhotoImage(resized)
            self._rendered_size = display_size
            self.canvas.itemconfigure(self._image_id, image=self._photo, state="normal")
        self.canvas.coords(self._image_id, self._offset_x, self._offset_y)
        self.canvas.itemconfigure(self._text_id, state="hidden")
        self.canvas.configure(scrollregion=(0, 0, max(display_w, canvas_w), max(display_h, canvas_h)))
        self.redraw_overlay()

    def _display_size(self) -> tuple[int, int]:
        if self._base_image is None:
            return 1, 1
        return (
            max(1, int(round(self._base_image.width * self.zoom))),
            max(1, int(round(self._base_image.height * self.zoom))),
        )

    def _maximum_render_zoom(self) -> float:
        if self._base_image is None:
            return self.max_zoom
        image_pixels = self._base_image.width * self._base_image.height
        rounding_margin = 0.5 / max(min(self._base_image.width, self._base_image.height), 1)
        pixel_limited_zoom = max(
            math.sqrt(self.MAX_RENDER_PIXELS / max(image_pixels, 1)) - rounding_margin,
            1e-12,
        )
        return min(self.max_zoom, pixel_limited_zoom)

    def _clamp_zoom(self, zoom: float) -> float:
        maximum = self._maximum_render_zoom()
        minimum = min(self.min_zoom, maximum)
        return min(max(zoom, minimum), maximum)

    def _event_to_image(self, event: tk.Event) -> tuple[float, float] | None:
        if self._base_image is None:
            return None
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        return (
            float(np.clip((canvas_x - self._offset_x) / self.zoom, 0, self._base_image.width - 1)),
            float(np.clip((canvas_y - self._offset_y) / self.zoom, 0, self._base_image.height - 1)),
        )

    def _handle_press(self, event: tk.Event) -> None:
        coords = self._event_to_image(event)
        if coords is not None and self._on_press is not None:
            self._on_press(*coords)

    def _handle_drag(self, event: tk.Event) -> None:
        coords = self._event_to_image(event)
        if coords is not None and self._on_drag is not None:
            self._on_drag(*coords)

    def _handle_release(self, event: tk.Event) -> None:
        coords = self._event_to_image(event)
        if coords is not None and self._on_release is not None:
            self._on_release(*coords)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._base_image is None:
            return
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            scale = 1.15
        elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            scale = 1 / 1.15
        else:
            return

        self._pending_zoom_factor *= scale
        self._pending_zoom_anchor = (event.x, event.y)
        if self._pending_zoom_after_id is None:
            self._pending_zoom_after_id = self.canvas.after_idle(self._apply_pending_zoom)

    def _apply_pending_zoom(self) -> None:
        self._pending_zoom_after_id = None
        scale = self._pending_zoom_factor
        anchor = self._pending_zoom_anchor
        self._pending_zoom_factor = 1.0
        self._pending_zoom_anchor = None
        if self._base_image is None or anchor is None:
            return

        old_w, old_h = self._display_size()
        event_x, event_y = anchor
        canvas_x = self.canvas.canvasx(event_x)
        canvas_y = self.canvas.canvasy(event_y)
        rel_x = float(np.clip((canvas_x - self._offset_x) / max(old_w, 1), 0.0, 1.0))
        rel_y = float(np.clip((canvas_y - self._offset_y) / max(old_h, 1), 0.0, 1.0))
        new_zoom = self._clamp_zoom(self.zoom * scale)
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        self.zoom = new_zoom
        self._render()
        new_w, new_h = self._display_size()
        left = self._offset_x + rel_x * new_w - event_x
        top = self._offset_y + rel_y * new_h - event_y
        self._move_view(left, top, new_w, new_h)

    def _cancel_pending_zoom(self) -> None:
        if self._pending_zoom_after_id is not None:
            self.canvas.after_cancel(self._pending_zoom_after_id)
        self._pending_zoom_after_id = None
        self._pending_zoom_factor = 1.0
        self._pending_zoom_anchor = None

    def _move_view(self, left: float, top: float, image_w: int, image_h: int) -> None:
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        max_left = max(image_w - canvas_w, 0)
        max_top = max(image_h - canvas_h, 0)
        left = min(max(left, 0.0), float(max_left))
        top = min(max(top, 0.0), float(max_top))
        self.canvas.xview_moveto(0.0 if image_w <= 0 else left / max(image_w, 1))
        self.canvas.yview_moveto(0.0 if image_h <= 0 else top / max(image_h, 1))

    def _on_pan_start(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

#!/usr/bin/env python3

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from measure_angles import (
    ANNOTATION_LINE_SPECS,
    ANNOTATION_POINT_SPECS,
    LOWER_ANGLE_LABEL,
    UPPER_ANGLE_LABEL,
    infer_knee_side_from_sources,
    load_annotation,
    measure_from_named_points,
    normalize_name,
    normalize_measurement_side,
    save_annotation_bundle,
)


RESAMPLING = getattr(Image, "Resampling", Image)
POINT_HANDLE_RADIUS = 18.0
LINE_HANDLE_RADIUS = 18.0


def default_output_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path.home() / "Documents" / "Knee_Xray_annotations"
    return Path("annotations")


def clone_points(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: point.copy() for name, point in points.items()}


def clone_lines(lines: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        line_name: {endpoint: value.copy() for endpoint, value in endpoints.items()}
        for line_name, endpoints in lines.items()
    }


class InteractiveImageCanvas(ttk.Frame):
    def __init__(self, master: tk.Misc, empty_text: str, bg: str = "#111111") -> None:
        super().__init__(master)
        self.empty_text = empty_text
        self.bg = bg
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 8.0
        self._base_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._auto_fit_pending = False
        self._on_press = None
        self._on_drag = None
        self._on_release = None
        self._overlay_drawer = None
        self._overlay_tag = "annotation_overlay"

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self._image_id = self.canvas.create_image(0, 0, anchor="nw")
        self._text_id = self.canvas.create_text(0, 0, text=empty_text, fill="#dddddd", font=("Helvetica", 16))

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Enter>", lambda _event: self.canvas.focus_set())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonPress-3>", self._on_drag_start)
        self.canvas.bind("<B3-Motion>", self._on_drag_move)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<ButtonPress-1>", self._handle_press)
        self.canvas.bind("<B1-Motion>", self._handle_drag)
        self.canvas.bind("<ButtonRelease-1>", self._handle_release)

        self.clear()

    def set_pointer_callbacks(self, on_press=None, on_drag=None, on_release=None) -> None:
        self._on_press = on_press
        self._on_drag = on_drag
        self._on_release = on_release

    def set_overlay_drawer(self, overlay_drawer=None) -> None:
        self._overlay_drawer = overlay_drawer
        self._redraw_overlay()

    @property
    def overlay_tag(self) -> str:
        return self._overlay_tag

    def clear(self, text: str | None = None) -> None:
        self._base_image = None
        self._photo = None
        self.zoom = 1.0
        self._auto_fit_pending = False
        self.canvas.delete(self._overlay_tag)
        self.canvas.itemconfigure(self._image_id, image="", state="hidden")
        self.canvas.itemconfigure(self._text_id, text=text or self.empty_text, state="normal")
        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        self._center_text()

    def set_image(self, image_bgr: np.ndarray, reset_view: bool = True) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._base_image = Image.fromarray(rgb)
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
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        fit_zoom = min(canvas_w / self._base_image.width, canvas_h / self._base_image.height)
        self.zoom = min(max(fit_zoom, self.min_zoom), self.max_zoom)
        self._render()
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def _on_configure(self, _event: object) -> None:
        if self._base_image is None:
            self._center_text()
            return
        if self._auto_fit_pending:
            self._auto_fit_pending = False
            self.reset_view()

    def _center_text(self) -> None:
        self.canvas.coords(self._text_id, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def _render(self) -> None:
        if self._base_image is None:
            self.clear()
            return

        display_w = max(1, int(round(self._base_image.width * self.zoom)))
        display_h = max(1, int(round(self._base_image.height * self.zoom)))
        resized = self._base_image.resize((display_w, display_h), RESAMPLING.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.itemconfigure(self._image_id, image=self._photo, state="normal")
        self.canvas.coords(self._image_id, 0, 0)
        self.canvas.itemconfigure(self._text_id, state="hidden")
        self.canvas.configure(scrollregion=(0, 0, display_w, display_h))
        self._redraw_overlay()

    def _redraw_overlay(self) -> None:
        self.canvas.delete(self._overlay_tag)
        if self._base_image is None or self._overlay_drawer is None:
            return
        self._overlay_drawer(self)
        self.canvas.tag_raise(self._overlay_tag)

    def image_to_canvas(self, point: np.ndarray) -> tuple[float, float]:
        return float(point[0] * self.zoom), float(point[1] * self.zoom)

    def _current_display_size(self) -> tuple[int, int]:
        if self._base_image is None:
            return 1, 1
        return (
            max(1, int(round(self._base_image.width * self.zoom))),
            max(1, int(round(self._base_image.height * self.zoom))),
        )

    def _canvas_event_to_image(self, event: tk.Event) -> tuple[float, float] | None:
        if self._base_image is None:
            return None
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = float(np.clip(canvas_x / self.zoom, 0, self._base_image.width - 1))
        image_y = float(np.clip(canvas_y / self.zoom, 0, self._base_image.height - 1))
        return image_x, image_y

    def _handle_press(self, event: tk.Event) -> None:
        coords = self._canvas_event_to_image(event)
        if coords is not None and self._on_press is not None:
            self._on_press(*coords)

    def _handle_drag(self, event: tk.Event) -> None:
        coords = self._canvas_event_to_image(event)
        if coords is not None and self._on_drag is not None:
            self._on_drag(*coords)

    def _handle_release(self, event: tk.Event) -> None:
        coords = self._canvas_event_to_image(event)
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

        old_w, old_h = self._current_display_size()
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        rel_x = canvas_x / max(old_w, 1)
        rel_y = canvas_y / max(old_h, 1)

        new_zoom = min(max(self.zoom * scale, self.min_zoom), self.max_zoom)
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        self.zoom = new_zoom
        self._render()

        new_w, new_h = self._current_display_size()
        left = rel_x * new_w - event.x
        top = rel_y * new_h - event.y
        self._move_view(left, top, new_w, new_h)

    def _move_view(self, left: float, top: float, image_w: int, image_h: int) -> None:
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        max_left = max(image_w - canvas_w, 0)
        max_top = max(image_h - canvas_h, 0)
        clamped_left = min(max(left, 0.0), float(max_left))
        clamped_top = min(max(top, 0.0), float(max_top))
        self.canvas.xview_moveto(0.0 if image_w <= 0 else clamped_left / max(image_w, 1))
        self.canvas.yview_moveto(0.0 if image_h <= 0 else clamped_top / max(image_h, 1))

    def _on_drag_start(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _on_drag_move(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_double_click(self, _event: tk.Event) -> None:
        self.reset_view()


class AnnotationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Knee Annotation Tool")
        self.root.geometry("1760x1020")
        self.root.minsize(1460, 900)

        self.output_dir = default_output_dir()
        self.raw_path: Path | None = None
        self.annotation_path: Path | None = None
        self.raw_image: np.ndarray | None = None
        self.points: dict[str, np.ndarray] = {}
        self.lines: dict[str, dict[str, np.ndarray]] = {}
        self.history: list[dict] = []
        self.current_index = 0
        self.current_line_name = ANNOTATION_LINE_SPECS[0][0]
        self.drag_target: tuple | None = None
        self.drag_changed = False
        self.blank_press = False

        self.tool_mode = tk.StringVar(value="point")
        self.raw_var = tk.StringVar(value="Raw image: -")
        self.annotation_var = tk.StringVar(value="Annotation: -")
        self.output_var = tk.StringVar(value=f"Output dir: {self.output_dir}")
        self.side_var = tk.StringVar(value="Auto")
        self.side_info_var = tk.StringVar(value="Measurement side: unknown")
        self.status_var = tk.StringVar(value="Open a raw image, then place the 8 points and 2 lines.")
        self.current_item_var = tk.StringVar(value="")
        self.preview_mode_var = tk.StringVar(value="Preview mode: waiting for annotations")
        self.upper_angle_var = tk.StringVar(value=f"{UPPER_ANGLE_LABEL}: -")
        self.lower_angle_var = tk.StringVar(value=f"{LOWER_ANGLE_LABEL}: -")
        self.summary_var = tk.StringVar(value="Points 0 / 8, lines 0 / 2")

        self._build_ui()
        self._refresh_lists()
        self._refresh_views(reset_view=True)
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Button(top, text="Open Raw...", command=self.open_raw_image).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(top, text="Open Annotation...", command=self.open_annotation).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Save Bundle...", command=self.save_bundle).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Browse Output...", command=self.browse_output_dir).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Undo", command=self.undo_last).grid(row=0, column=4, padx=(16, 4))
        ttk.Button(top, text="Delete Point", command=self.delete_selected_point).grid(row=0, column=5, padx=4)
        ttk.Button(top, text="Reset Line", command=self.reset_selected_line).grid(row=0, column=6, padx=4)
        ttk.Button(top, text="Clear All", command=self.clear_all_annotations).grid(row=0, column=7, padx=4)
        ttk.Label(top, text="Side:").grid(row=0, column=8, padx=(16, 4))
        side_combo = ttk.Combobox(
            top,
            textvariable=self.side_var,
            values=("Auto", "L", "R"),
            width=7,
            state="readonly",
        )
        side_combo.grid(row=0, column=9, padx=4)
        side_combo.bind("<<ComboboxSelected>>", self._on_side_changed)

        path_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        path_frame.pack(fill="x")
        ttk.Label(path_frame, textvariable=self.raw_var, wraplength=1680).pack(anchor="w")
        ttk.Label(path_frame, textvariable=self.annotation_var, wraplength=1680).pack(anchor="w")
        ttk.Label(path_frame, textvariable=self.output_var, wraplength=1680).pack(anchor="w")
        ttk.Label(path_frame, textvariable=self.side_info_var, wraplength=1680).pack(anchor="w")

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(body, text="Raw Annotation", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.annotation_view = InteractiveImageCanvas(left, empty_text="Open a raw image")
        self.annotation_view.grid(row=0, column=0, sticky="nsew")
        self.annotation_view.set_overlay_drawer(self._draw_editor_overlay)
        self.annotation_view.set_pointer_callbacks(
            on_press=self._on_canvas_press,
            on_drag=self._on_canvas_drag,
            on_release=self._on_canvas_release,
        )
        ttk.Label(
            left,
            text="Left click: place selected item or drag existing handles, wheel: zoom, right/middle drag: pan, double-click: fit",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        preview = ttk.LabelFrame(right, text="Measurement Preview", padding=8)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)

        self.preview_view = InteractiveImageCanvas(preview, empty_text="Preview will appear after the required annotations are placed")
        self.preview_view.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            preview,
            text="Wheel: zoom, right/middle drag: pan, double-click: fit",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.rowconfigure(2, weight=1)

        tool_box = ttk.LabelFrame(controls, text="Tool", padding=8)
        tool_box.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Radiobutton(tool_box, text="Points", variable=self.tool_mode, value="point", command=self._on_tool_changed).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Radiobutton(tool_box, text="Lines", variable=self.tool_mode, value="line", command=self._on_tool_changed).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(12, 0),
        )
        ttk.Label(tool_box, textvariable=self.current_item_var, font=("Helvetica", 14, "bold")).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        point_box = ttk.LabelFrame(controls, text="Points", padding=8)
        point_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(10, 0))
        point_box.rowconfigure(0, weight=1)
        point_box.columnconfigure(0, weight=1)
        self.point_list = tk.Listbox(point_box, height=10, exportselection=False, font=("Menlo", 11))
        self.point_list.grid(row=0, column=0, sticky="nsew")
        self.point_list.bind("<<ListboxSelect>>", self._on_point_selected)

        line_box = ttk.LabelFrame(controls, text="Lines", padding=8)
        line_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(10, 0))
        line_box.rowconfigure(0, weight=1)
        line_box.columnconfigure(0, weight=1)
        self.line_list = tk.Listbox(line_box, height=4, exportselection=False, font=("Menlo", 11))
        self.line_list.grid(row=0, column=0, sticky="nsew")
        self.line_list.bind("<<ListboxSelect>>", self._on_line_selected)

        stats = ttk.LabelFrame(controls, text="Status", padding=8)
        stats.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)
        ttk.Label(stats, textvariable=self.summary_var).grid(row=0, column=0, sticky="w")
        ttk.Label(stats, textvariable=self.preview_mode_var).grid(row=0, column=1, sticky="w")
        ttk.Label(stats, textvariable=self.status_var, wraplength=760, justify="left").grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        ttk.Label(stats, textvariable=self.upper_angle_var, font=("Helvetica", 18, "bold")).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        ttk.Label(stats, textvariable=self.lower_angle_var, font=("Helvetica", 18, "bold")).grid(
            row=2,
            column=1,
            sticky="w",
            pady=(10, 0),
        )

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self.open_raw_image())
        self.root.bind("<Control-s>", lambda _event: self.save_bundle())
        self.root.bind("<Up>", lambda _event: self._step_point_selection(-1))
        self.root.bind("<Down>", lambda _event: self._step_point_selection(1))
        self.root.bind("<BackSpace>", lambda _event: self.delete_selected_point())
        self.root.bind("<Delete>", lambda _event: self.delete_selected_point())

    def _effective_side(self) -> str | None:
        explicit_side = normalize_measurement_side(self.side_var.get())
        if explicit_side is not None:
            return explicit_side
        return infer_knee_side_from_sources(self.annotation_path, self.raw_path)

    def _set_side_from_sources(self, *sources: object) -> None:
        side = infer_knee_side_from_sources(*sources)
        self.side_var.set(side if side is not None else "Auto")

    def _on_side_changed(self, _event: object | None = None) -> None:
        self._refresh_metadata()
        self._refresh_views(reset_view=False)

    def _snapshot_state(self) -> dict:
        return {
            "points": clone_points(self.points),
            "lines": clone_lines(self.lines),
            "current_index": self.current_index,
            "current_line_name": self.current_line_name,
            "tool_mode": self.tool_mode.get(),
        }

    def _restore_state(self, snapshot: dict) -> None:
        self.points = clone_points(snapshot["points"])
        self.lines = clone_lines(snapshot["lines"])
        self.current_index = snapshot["current_index"]
        self.current_line_name = snapshot["current_line_name"]
        self.tool_mode.set(snapshot["tool_mode"])
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def _push_snapshot(self) -> None:
        self.history.append(self._snapshot_state())

    def _step_point_selection(self, delta: int) -> None:
        target = min(max(self.current_index + delta, 0), len(ANNOTATION_POINT_SPECS) - 1)
        self.current_index = target
        self.tool_mode.set("point")
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def open_raw_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Raw Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not filename:
            return

        raw_path = Path(filename)
        raw_image = cv2.imread(str(raw_path))
        if raw_image is None:
            messagebox.showerror("Open Error", f"Cannot read image:\n{raw_path}")
            return

        self.raw_path = raw_path
        self.raw_image = raw_image
        self.annotation_path = None
        self.points.clear()
        self.lines.clear()
        self.history.clear()
        self.current_index = 0
        self.current_line_name = ANNOTATION_LINE_SPECS[0][0]
        self.tool_mode.set("point")
        self._set_side_from_sources(raw_path)
        self._refresh_metadata()
        self._refresh_lists()
        self._refresh_views(reset_view=True)
        self.status_var.set("Raw image loaded. Place the 8 points and then draw the 2 joint lines.")

    def open_annotation(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Annotation JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        annotation_path = Path(filename)
        try:
            annotation = load_annotation(annotation_path)
        except Exception as exc:
            messagebox.showerror("Open Error", str(exc))
            return

        raw_path = Path(annotation["raw_path"])
        raw_image = cv2.imread(str(raw_path))
        if raw_image is None:
            messagebox.showerror(
                "Open Error",
                f"Cannot read the raw image recorded in the annotation file:\n{raw_path}",
            )
            return

        self.raw_path = raw_path
        self.raw_image = raw_image
        self.annotation_path = annotation_path
        self.points = {
            name: np.array([annotation["points"][name]["x"], annotation["points"][name]["y"]], dtype=np.float32)
            for name, _label in ANNOTATION_POINT_SPECS
        }
        self.lines = {}
        for line_name, _label in ANNOTATION_LINE_SPECS:
            line_info = annotation.get("lines", {}).get(line_name)
            if not line_info:
                continue
            self.lines[line_name] = {
                "p1": np.array([line_info["p1"]["x"], line_info["p1"]["y"]], dtype=np.float32),
                "p2": np.array([line_info["p2"]["x"], line_info["p2"]["y"]], dtype=np.float32),
            }

        self.history.clear()
        self.current_index = 0
        self.current_line_name = ANNOTATION_LINE_SPECS[0][0]
        self.tool_mode.set("point")
        annotation_side = normalize_measurement_side(annotation.get("side"))
        if annotation_side is not None:
            self.side_var.set(annotation_side)
        else:
            self._set_side_from_sources(annotation_path, raw_path)
        self._refresh_metadata()
        self._refresh_lists()
        self._refresh_views(reset_view=True)
        self.status_var.set("Annotation loaded. You can drag points and line endpoints to refine it.")

    def browse_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select Output Directory", initialdir=str(self.output_dir))
        if not directory:
            return
        self.output_dir = Path(directory)
        self._refresh_metadata()

    def undo_last(self) -> None:
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return
        snapshot = self.history.pop()
        self._restore_state(snapshot)
        self.status_var.set("Reverted the last edit.")

    def delete_selected_point(self) -> None:
        name = ANNOTATION_POINT_SPECS[self.current_index][0]
        if name not in self.points:
            self.status_var.set(f"Point not set: {name}")
            return
        self._push_snapshot()
        self.points.pop(name, None)
        self._refresh_lists()
        self._refresh_views(reset_view=False)
        self.status_var.set(f"Deleted point: {name}")

    def reset_selected_line(self) -> None:
        if self.current_line_name not in self.lines:
            self.status_var.set(f"Line not set: {self.current_line_name}")
            return
        self._push_snapshot()
        self.lines.pop(self.current_line_name, None)
        self.tool_mode.set("line")
        self._refresh_lists()
        self._refresh_views(reset_view=False)
        self.status_var.set(f"Reset line: {self._line_label(self.current_line_name)}")

    def clear_all_annotations(self) -> None:
        if not self.points and not self.lines:
            return
        self._push_snapshot()
        self.points.clear()
        self.lines.clear()
        self.current_index = 0
        self.current_line_name = ANNOTATION_LINE_SPECS[0][0]
        self.tool_mode.set("point")
        self._refresh_lists()
        self._refresh_views(reset_view=False)
        self.status_var.set("Cleared all points and lines.")

    def save_bundle(self) -> None:
        if self.raw_path is None or self.raw_image is None:
            messagebox.showerror("Save Error", "Open a raw image before saving.")
            return

        missing_points = [label for name, label in ANNOTATION_POINT_SPECS if name not in self.points]
        if missing_points:
            messagebox.showerror("Save Error", "Missing points:\n" + "\n".join(missing_points))
            return

        missing_lines = [label for name, label in ANNOTATION_LINE_SPECS if not self._line_is_complete(name)]
        if missing_lines:
            messagebox.showerror("Save Error", "Missing lines:\n" + "\n".join(missing_lines))
            return

        side = self._effective_side()
        if side is None:
            messagebox.showerror("Save Error", "Cannot determine knee side. Choose L or R in Side before saving.")
            return

        prefix = normalize_name(self.raw_path.stem).replace(" ", "_")
        if infer_knee_side_from_sources(self.raw_path) != side:
            prefix = f"{prefix}_{side}"
        try:
            paths, result = save_annotation_bundle(
                self.raw_path,
                self._named_points_payload(),
                self.output_dir,
                prefix=prefix,
                named_lines=self._named_lines_payload(),
                side=side,
            )
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            return

        self.annotation_path = paths["annotation"]
        self._refresh_metadata()
        self.upper_angle_var.set(f"{UPPER_ANGLE_LABEL}: {result['mldfa_angle']:.2f} deg")
        self.lower_angle_var.set(f"{LOWER_ANGLE_LABEL}: {result['mpta_angle']:.2f} deg")
        self.status_var.set(
            "Saved annotation bundle: "
            f"{paths['annotation'].name}, {paths['point_image'].name}, {paths['line_image'].name}, {paths['combined_image'].name}"
        )

    def _on_canvas_press(self, image_x: float, image_y: float) -> None:
        hit = self._hit_test(
            np.array([image_x, image_y], dtype=np.float32),
            zoom=self.annotation_view.zoom,
        )
        self.drag_target = hit
        self.drag_changed = False
        self.blank_press = hit is None

        if hit is None:
            return

        hit_type = hit[0]
        if hit_type == "point":
            self.current_index = self._index_for_name(hit[1])
            self.tool_mode.set("point")
        elif hit_type == "line":
            self.current_line_name = hit[1]
            self.tool_mode.set("line")
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def _on_canvas_drag(self, image_x: float, image_y: float) -> None:
        if self.drag_target is None:
            return
        if not self.drag_changed:
            self._push_snapshot()
            self.drag_changed = True
        target_point = np.array([image_x, image_y], dtype=np.float32)
        self._move_handle(self.drag_target, target_point)
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def _on_canvas_release(self, image_x: float, image_y: float) -> None:
        target_point = np.array([image_x, image_y], dtype=np.float32)
        if self.drag_target is not None:
            if self.drag_changed:
                self._move_handle(self.drag_target, target_point)
                self.status_var.set("Moved annotation handle.")
            self.drag_target = None
            self.drag_changed = False
            self.blank_press = False
            self._refresh_lists()
            self._refresh_views(reset_view=False)
            return

        if not self.blank_press:
            return

        if self.tool_mode.get() == "point":
            self._place_current_point(target_point)
        else:
            self._place_current_line_endpoint(target_point)
        self.blank_press = False

    def _move_handle(self, hit: tuple, target_point: np.ndarray) -> None:
        hit_type = hit[0]
        if hit_type == "point":
            point_name = hit[1]
            self.points[point_name] = target_point
            self.current_index = self._index_for_name(point_name)
        elif hit_type == "line":
            line_name, endpoint_name = hit[1], hit[2]
            line = self.lines.setdefault(line_name, {})
            line[endpoint_name] = target_point
            self.current_line_name = line_name

    def _place_current_point(self, target_point: np.ndarray) -> None:
        point_name, point_label = ANNOTATION_POINT_SPECS[self.current_index]
        self._push_snapshot()
        self.points[point_name] = target_point
        self.current_index = self._next_missing_point_index(self.current_index)
        self._refresh_lists()
        self._refresh_views(reset_view=False)
        self.status_var.set(f"Placed {point_label} at ({target_point[0]:.1f}, {target_point[1]:.1f})")

    def _place_current_line_endpoint(self, target_point: np.ndarray) -> None:
        line_name = self.current_line_name
        line_label = self._line_label(line_name)
        endpoint_name = self._next_line_endpoint_name(line_name, target_point)
        self._push_snapshot()
        line = self.lines.setdefault(line_name, {})
        line[endpoint_name] = target_point
        self._refresh_lists()
        self._refresh_views(reset_view=False)
        self.status_var.set(
            f"Set {line_label} {endpoint_name} at ({target_point[0]:.1f}, {target_point[1]:.1f})"
        )

    def _on_point_selected(self, _event: object) -> None:
        selection = self.point_list.curselection()
        if not selection:
            return
        self.current_index = int(selection[0])
        self.tool_mode.set("point")
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def _on_line_selected(self, _event: object) -> None:
        selection = self.line_list.curselection()
        if not selection:
            return
        self.current_line_name = ANNOTATION_LINE_SPECS[int(selection[0])][0]
        self.tool_mode.set("line")
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def _on_tool_changed(self) -> None:
        self._refresh_lists()
        self._refresh_views(reset_view=False)

    def _line_label(self, line_name: str) -> str:
        for name, label in ANNOTATION_LINE_SPECS:
            if name == line_name:
                return label
        return line_name

    def _index_for_name(self, target_name: str) -> int:
        for index, (name, _label) in enumerate(ANNOTATION_POINT_SPECS):
            if name == target_name:
                return index
        return 0

    def _next_missing_point_index(self, current_index: int) -> int:
        for offset in range(1, len(ANNOTATION_POINT_SPECS) + 1):
            candidate = min(current_index + offset, len(ANNOTATION_POINT_SPECS) - 1)
            point_name = ANNOTATION_POINT_SPECS[candidate][0]
            if point_name not in self.points:
                return candidate
        return min(current_index + 1, len(ANNOTATION_POINT_SPECS) - 1)

    def _next_line_endpoint_name(self, line_name: str, target_point: np.ndarray) -> str:
        line = self.lines.get(line_name, {})
        if "p1" not in line:
            return "p1"
        if "p2" not in line:
            return "p2"
        dist_p1 = float(np.linalg.norm(line["p1"] - target_point))
        dist_p2 = float(np.linalg.norm(line["p2"] - target_point))
        return "p1" if dist_p1 <= dist_p2 else "p2"

    def _line_is_complete(self, line_name: str) -> bool:
        line = self.lines.get(line_name)
        return line is not None and "p1" in line and "p2" in line

    def _complete_lines_count(self) -> int:
        return sum(1 for name, _label in ANNOTATION_LINE_SPECS if self._line_is_complete(name))

    def _named_points_payload(self) -> dict[str, dict[str, float]]:
        return {
            name: {"x": float(self.points[name][0]), "y": float(self.points[name][1])}
            for name, _label in ANNOTATION_POINT_SPECS
            if name in self.points
        }

    def _named_lines_payload(self) -> dict[str, dict[str, dict[str, float]]]:
        payload: dict[str, dict[str, dict[str, float]]] = {}
        for line_name, _label in ANNOTATION_LINE_SPECS:
            if not self._line_is_complete(line_name):
                continue
            line = self.lines[line_name]
            payload[line_name] = {
                "p1": {"x": float(line["p1"][0]), "y": float(line["p1"][1])},
                "p2": {"x": float(line["p2"][0]), "y": float(line["p2"][1])},
            }
        return payload

    def _hit_test(self, target_point: np.ndarray, zoom: float = 1.0) -> tuple | None:
        hits: list[tuple[float, tuple]] = []
        image_point_radius = POINT_HANDLE_RADIUS / max(zoom, 1e-6)
        image_line_radius = LINE_HANDLE_RADIUS / max(zoom, 1e-6)

        for name, point in self.points.items():
            distance = float(np.linalg.norm(point - target_point))
            if distance <= image_point_radius:
                hits.append((distance, ("point", name)))

        for line_name, endpoints in self.lines.items():
            for endpoint_name, point in endpoints.items():
                distance = float(np.linalg.norm(point - target_point))
                if distance <= image_line_radius:
                    hits.append((distance, ("line", line_name, endpoint_name)))

        if not hits:
            return None
        hits.sort(key=lambda item: item[0])
        return hits[0][1]

    def _refresh_metadata(self) -> None:
        self.raw_var.set(f"Raw image: {self.raw_path}" if self.raw_path else "Raw image: -")
        self.annotation_var.set(f"Annotation: {self.annotation_path}" if self.annotation_path else "Annotation: -")
        self.output_var.set(f"Output dir: {self.output_dir}")
        side = self._effective_side()
        if side is None:
            self.side_info_var.set("Measurement side: unknown. Choose L or R before measuring/saving.")
        else:
            self.side_info_var.set(f"Measurement side: {side} ({'left knee' if side == 'L' else 'right knee'})")

    def _refresh_lists(self) -> None:
        self.point_list.delete(0, tk.END)
        for index, (name, label) in enumerate(ANNOTATION_POINT_SPECS):
            point = self.points.get(name)
            prefix = "->" if index == self.current_index and self.tool_mode.get() == "point" else "  "
            coords = "-" if point is None else f"({point[0]:7.1f}, {point[1]:7.1f})"
            self.point_list.insert(tk.END, f"{prefix} {index + 1}. {label:<18} {coords}")

        self.point_list.selection_clear(0, tk.END)
        self.point_list.selection_set(self.current_index)
        self.point_list.activate(self.current_index)

        self.line_list.delete(0, tk.END)
        selected_line_index = 0
        for index, (name, label) in enumerate(ANNOTATION_LINE_SPECS):
            if name == self.current_line_name:
                selected_line_index = index
            line = self.lines.get(name, {})
            p1 = "-" if "p1" not in line else f"({line['p1'][0]:7.1f}, {line['p1'][1]:7.1f})"
            p2 = "-" if "p2" not in line else f"({line['p2'][0]:7.1f}, {line['p2'][1]:7.1f})"
            prefix = "->" if name == self.current_line_name and self.tool_mode.get() == "line" else "  "
            self.line_list.insert(tk.END, f"{prefix} {label:<16} p1 {p1}  p2 {p2}")

        self.line_list.selection_clear(0, tk.END)
        self.line_list.selection_set(selected_line_index)
        self.line_list.activate(selected_line_index)

        if self.tool_mode.get() == "point":
            current_label = ANNOTATION_POINT_SPECS[self.current_index][1]
            self.current_item_var.set(f"Point mode: {self.current_index + 1} / 8  {current_label}")
        else:
            line = self.lines.get(self.current_line_name, {})
            if "p1" not in line:
                hint = "next endpoint p1"
            elif "p2" not in line:
                hint = "next endpoint p2"
            else:
                hint = "drag p1/p2 or click to replace the nearer endpoint"
            self.current_item_var.set(f"Line mode: {self._line_label(self.current_line_name)}  {hint}")

        self.summary_var.set(
            f"Points {len(self.points)} / {len(ANNOTATION_POINT_SPECS)}, "
            f"lines {self._complete_lines_count()} / {len(ANNOTATION_LINE_SPECS)}"
        )

    def _refresh_views(self, reset_view: bool) -> None:
        if self.raw_image is None:
            self.annotation_view.clear("Open a raw image")
            self.preview_view.clear("Preview will appear after the required annotations are placed")
            self.upper_angle_var.set(f"{UPPER_ANGLE_LABEL}: -")
            self.lower_angle_var.set(f"{LOWER_ANGLE_LABEL}: -")
            self.preview_mode_var.set("Preview mode: waiting for annotations")
            return

        self.annotation_view.set_image(self.raw_image, reset_view=reset_view)

        if len(self.points) != len(ANNOTATION_POINT_SPECS):
            self.preview_view.clear("Place all 8 points to enable the measurement preview")
            self.upper_angle_var.set(f"{UPPER_ANGLE_LABEL}: -")
            self.lower_angle_var.set(f"{LOWER_ANGLE_LABEL}: -")
            self.preview_mode_var.set("Preview mode: waiting for all points")
            return

        side = self._effective_side()
        if side is None:
            self.preview_view.clear("Choose Side L or R to enable anatomical angle measurement")
            self.upper_angle_var.set(f"{UPPER_ANGLE_LABEL}: -")
            self.lower_angle_var.set(f"{LOWER_ANGLE_LABEL}: -")
            self.preview_mode_var.set("Preview mode: waiting for side")
            return

        named_lines = self._named_lines_payload()
        preview_uses_manual_lines = len(named_lines) == len(ANNOTATION_LINE_SPECS)
        try:
            result, _debug = measure_from_named_points(
                self.raw_image,
                self._named_points_payload(),
                raw_path=self.raw_path,
                named_lines=named_lines if preview_uses_manual_lines else None,
                side=side,
            )
        except Exception as exc:
            self.preview_view.clear(f"Measurement failed:\n{exc}")
            self.upper_angle_var.set(f"{UPPER_ANGLE_LABEL}: -")
            self.lower_angle_var.set(f"{LOWER_ANGLE_LABEL}: -")
            self.preview_mode_var.set("Preview mode: error")
            self.status_var.set(f"Measurement preview failed: {exc}")
            return

        self.preview_view.set_image(result["combined_image"], reset_view=reset_view)
        self.upper_angle_var.set(f"{UPPER_ANGLE_LABEL}: {result['mldfa_angle']:.2f} deg")
        self.lower_angle_var.set(f"{LOWER_ANGLE_LABEL}: {result['mpta_angle']:.2f} deg")
        if preview_uses_manual_lines:
            self.preview_mode_var.set(f"Preview mode: using manual joint lines, side {side}")
        else:
            self.preview_mode_var.set(f"Preview mode: provisional line fits, side {side}")

    def _draw_editor_overlay(self, view: InteractiveImageCanvas) -> None:
        if self.raw_image is None:
            return

        tag = view.overlay_tag
        point_radius_outer = 11
        point_radius_inner = 8
        line_handle_outer = 10
        line_handle_inner = 7
        line_width = 2
        point_font = ("Helvetica", 13, "bold")
        endpoint_font = ("Helvetica", 11, "bold")
        line_font = ("Helvetica", 12, "bold")
        selected_color = "#ffff00"
        line_default_color = "#00dc78"
        point_inner_color = "#00a0ff"
        white = "#ffffff"

        def canvas_point(point: np.ndarray) -> tuple[float, float]:
            return view.image_to_canvas(point)

        def draw_handle(point: np.ndarray, outer_radius: int, inner_radius: int, inner_color: str, outer_color: str) -> None:
            x, y = canvas_point(point)
            view.canvas.create_oval(
                x - outer_radius,
                y - outer_radius,
                x + outer_radius,
                y + outer_radius,
                fill=outer_color,
                outline="",
                tags=(tag,),
            )
            view.canvas.create_oval(
                x - inner_radius,
                y - inner_radius,
                x + inner_radius,
                y + inner_radius,
                fill=inner_color,
                outline="",
                tags=(tag,),
            )

        for line_name, line_label in ANNOTATION_LINE_SPECS:
            line = self.lines.get(line_name, {})
            p1 = line.get("p1")
            p2 = line.get("p2")
            is_selected = self.tool_mode.get() == "line" and line_name == self.current_line_name
            line_color = selected_color if is_selected else line_default_color

            if p1 is not None and p2 is not None:
                x1, y1 = canvas_point(p1)
                x2, y2 = canvas_point(p2)
                view.canvas.create_line(x1, y1, x2, y2, fill=line_color, width=line_width, tags=(tag,))
                mid = ((p1 + p2) * 0.5).astype(np.float32)
                mx, my = canvas_point(mid)
                view.canvas.create_text(
                    mx + 8,
                    my - 8,
                    text=line_label,
                    fill=line_color,
                    font=line_font,
                    anchor="sw",
                    tags=(tag,),
                )

            for endpoint_name in ("p1", "p2"):
                point = line.get(endpoint_name)
                if point is None:
                    continue
                draw_handle(point, line_handle_outer, line_handle_inner, line_color, white)
                x, y = canvas_point(point)
                view.canvas.create_text(
                    x + 10,
                    y - 8,
                    text=endpoint_name,
                    fill=line_color,
                    font=endpoint_font,
                    anchor="sw",
                    tags=(tag,),
                )

        for index, (name, _label) in enumerate(ANNOTATION_POINT_SPECS):
            point = self.points.get(name)
            if point is None:
                continue

            is_current = self.tool_mode.get() == "point" and index == self.current_index
            outer_color = selected_color if is_current else white
            draw_handle(point, point_radius_outer, point_radius_inner, point_inner_color, outer_color)
            x, y = canvas_point(point)
            view.canvas.create_text(
                x + 10,
                y - 10,
                text=str(index + 1),
                fill=selected_color,
                font=point_font,
                anchor="sw",
                tags=(tag,),
            )

    def _render_editor_overlay(self) -> np.ndarray:
        assert self.raw_image is not None
        canvas = self.raw_image.copy()

        for line_name, line_label in ANNOTATION_LINE_SPECS:
            line = self.lines.get(line_name, {})
            p1 = line.get("p1")
            p2 = line.get("p2")
            is_selected = self.tool_mode.get() == "line" and line_name == self.current_line_name
            line_color = (0, 255, 255) if is_selected else (0, 220, 120)
            handle_color = (0, 255, 255) if is_selected else (0, 220, 120)

            if p1 is not None and p2 is not None:
                cv2.line(
                    canvas,
                    tuple(np.round(p1).astype(int)),
                    tuple(np.round(p2).astype(int)),
                    line_color,
                    2,
                    cv2.LINE_AA,
                )
                mid = ((p1 + p2) * 0.5).astype(np.float32)
                cv2.putText(
                    canvas,
                    line_label,
                    (int(mid[0] + 8), int(mid[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

            for endpoint_name in ("p1", "p2"):
                point = line.get(endpoint_name)
                if point is None:
                    continue
                center = tuple(np.round(point).astype(int))
                cv2.circle(canvas, center, 10, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, center, 7, handle_color, -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    endpoint_name,
                    (center[0] + 10, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

        for index, (name, _label) in enumerate(ANNOTATION_POINT_SPECS):
            point = self.points.get(name)
            if point is None:
                continue

            center = tuple(np.round(point).astype(int))
            is_current = self.tool_mode.get() == "point" and index == self.current_index
            outer_color = (0, 255, 255) if is_current else (255, 255, 255)
            inner_color = (255, 160, 0)
            cv2.circle(canvas, center, 11, outer_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 8, inner_color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index + 1),
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas


def main() -> None:
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

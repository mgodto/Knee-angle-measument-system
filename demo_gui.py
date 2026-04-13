#!/usr/bin/env python3

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from measure_angles import (
    LOWER_ANGLE_FULL_NAME,
    LOWER_ANGLE_LABEL,
    UPPER_ANGLE_FULL_NAME,
    UPPER_ANGLE_LABEL,
    classify_file,
    infer_side,
    measure_case,
    normalize_name,
)


RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass
class MeasurementPair:
    label: str
    point_path: Path
    line_path: Path
    raw_path: Path | None


def discover_measurement_pairs(root: Path) -> list[MeasurementPair]:
    pairs: list[MeasurementPair] = []
    for case_dir in sorted(path for path in root.glob("*/*") if path.is_dir()):
        points: dict[str, Path] = {}
        lines: dict[str, Path] = {}
        raws: dict[str, Path] = {}

        for path in sorted(case_dir.iterdir()):
            if not path.is_file() or path.name == ".DS_Store":
                continue
            kind = classify_file(path)
            side = infer_side(normalize_name(path.name))
            if side is None:
                continue
            if kind == "point":
                points[side] = path
            elif kind == "line":
                lines[side] = path
            elif kind == "raw":
                raws[side] = path

        for side, point_path in sorted(points.items()):
            line_path = lines.get(side)
            if line_path is None:
                continue
            raw_path = raws.get(side) or raws.get("RL")
            label = f"{case_dir.parent.name}/{case_dir.name} [{side}]"
            pairs.append(MeasurementPair(label, point_path, line_path, raw_path))

    return pairs


def read_preview_image(path: Path) -> cv2.typing.MatLike | None:
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if image is None:
        image = cv2.imread(str(path))
    return image


class ZoomableImageCanvas(ttk.Frame):
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
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

        self.clear()

    def clear(self, text: str | None = None) -> None:
        self._base_image = None
        self._photo = None
        self.zoom = 1.0
        self._auto_fit_pending = False
        self.canvas.itemconfigure(self._image_id, image="", state="hidden")
        self.canvas.itemconfigure(self._text_id, text=text or self.empty_text, state="normal")
        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        self._center_text()

    def set_image(self, image_bgr) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._base_image = Image.fromarray(rgb)
        if self.canvas.winfo_width() > 1 and self.canvas.winfo_height() > 1:
            self._auto_fit_pending = False
            self.reset_view()
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
        self.canvas.coords(
            self._text_id,
            self.canvas.winfo_width() / 2,
            self.canvas.winfo_height() / 2,
        )

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

    def _current_display_size(self) -> tuple[int, int]:
        if self._base_image is None:
            return 1, 1
        return (
            max(1, int(round(self._base_image.width * self.zoom))),
            max(1, int(round(self._base_image.height * self.zoom))),
        )

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


class MeasurementDemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Knee Angle Demo")
        self.root.geometry("1640x980")
        self.root.minsize(1320, 860)

        self.dataset_root = Path("images/line_point")
        self.pairs = discover_measurement_pairs(self.dataset_root)
        self.pair_lookup = {pair.label: pair for pair in self.pairs}
        self.result_data: dict | None = None
        self.current_point_path: Path | None = None
        self.current_line_path: Path | None = None
        self.current_raw_path: Path | None = None

        self.case_var = tk.StringVar()
        self.point_var = tk.StringVar(value="Point image: -")
        self.line_var = tk.StringVar(value="Line image: -")
        self.raw_var = tk.StringVar(value="Raw image: -")
        self.status_var = tk.StringVar(value="Select a case or browse files, then click Run.")
        self.upper_angle_var = tk.StringVar(value=f"{UPPER_ANGLE_LABEL}: -")
        self.lower_angle_var = tk.StringVar(value=f"{LOWER_ANGLE_LABEL}: -")
        self.angle_info_var = tk.StringVar(
            value=(
                f"{UPPER_ANGLE_LABEL} = {UPPER_ANGLE_FULL_NAME}\n"
                f"{LOWER_ANGLE_LABEL} = {LOWER_ANGLE_FULL_NAME}"
            )
        )

        self._image_refs: dict[str, ImageTk.PhotoImage] = {}

        self._build_ui()
        if self.pairs:
            first_label = self.pairs[0].label
            self.case_var.set(first_label)
            self._load_pair(self.pair_lookup[first_label])
            self.root.after(100, self.run_measurement)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Case").grid(row=0, column=0, sticky="w")
        case_values = [pair.label for pair in self.pairs]
        self.case_combo = ttk.Combobox(
            top,
            textvariable=self.case_var,
            values=case_values,
            state="readonly" if case_values else "disabled",
            width=36,
        )
        self.case_combo.grid(row=0, column=1, sticky="we", padx=(8, 12))
        self.case_combo.bind("<<ComboboxSelected>>", self._on_case_selected)

        ttk.Button(top, text="Browse Point...", command=self.browse_point).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Browse Line...", command=self.browse_line).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Run", command=self.run_measurement).grid(row=0, column=4, padx=(12, 4))
        ttk.Button(top, text="Save Result...", command=self.save_result).grid(row=0, column=5, padx=4)
        top.columnconfigure(1, weight=1)

        path_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        path_frame.pack(fill="x")
        ttk.Label(path_frame, textvariable=self.point_var, wraplength=1500).pack(anchor="w")
        ttk.Label(path_frame, textvariable=self.line_var, wraplength=1500).pack(anchor="w")
        ttk.Label(path_frame, textvariable=self.raw_var, wraplength=1500).pack(anchor="w")

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        point_frame = ttk.LabelFrame(left, text="Point Input", padding=8)
        point_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        point_frame.rowconfigure(0, weight=1)
        point_frame.columnconfigure(0, weight=1)

        line_frame = ttk.LabelFrame(left, text="Line Input", padding=8)
        line_frame.grid(row=1, column=0, sticky="nsew")
        line_frame.rowconfigure(0, weight=1)
        line_frame.columnconfigure(0, weight=1)

        self.point_view = ZoomableImageCanvas(point_frame, empty_text="No point image")
        self.point_view.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            point_frame,
            text="Mouse wheel: zoom, drag: pan, double-click: fit to window",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.line_view = ZoomableImageCanvas(line_frame, empty_text="No line image")
        self.line_view.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            line_frame,
            text="Mouse wheel: zoom, drag: pan, double-click: fit to window",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        result_frame = ttk.LabelFrame(right, text="Measurement Result", padding=8)
        result_frame.grid(row=0, column=0, sticky="nsew")
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.result_view = ZoomableImageCanvas(result_frame, empty_text="Run the measurement to see the result")
        self.result_view.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            result_frame,
            text="Mouse wheel: zoom, drag: pan, double-click: fit to window",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        stats = ttk.Frame(right, padding=(0, 10, 0, 0))
        stats.grid(row=1, column=0, sticky="ew")
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)

        self.upper_label = tk.Label(
            stats,
            textvariable=self.upper_angle_var,
            font=("Helvetica", 24, "bold"),
            anchor="w",
        )
        self.upper_label.grid(row=0, column=0, sticky="w")
        self.lower_label = tk.Label(
            stats,
            textvariable=self.lower_angle_var,
            font=("Helvetica", 24, "bold"),
            anchor="w",
        )
        self.lower_label.grid(row=0, column=1, sticky="w")
        ttk.Label(stats, textvariable=self.angle_info_var, wraplength=900, justify="left").grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        ttk.Label(stats, textvariable=self.status_var, wraplength=900).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )

    def _on_case_selected(self, _event: object) -> None:
        pair = self.pair_lookup.get(self.case_var.get())
        if pair is None:
            return
        self._load_pair(pair)

    def _load_pair(self, pair: MeasurementPair) -> None:
        self.current_point_path = pair.point_path
        self.current_line_path = pair.line_path
        self.current_raw_path = pair.raw_path
        self._refresh_paths()
        self._refresh_input_previews()
        self.status_var.set("Inputs loaded. Click Run to generate the combined measurement result.")

    def _refresh_paths(self) -> None:
        self.point_var.set(f"Point image: {self.current_point_path}" if self.current_point_path else "Point image: -")
        self.line_var.set(f"Line image: {self.current_line_path}" if self.current_line_path else "Line image: -")
        self.raw_var.set(f"Raw image: {self.current_raw_path}" if self.current_raw_path else "Raw image: auto-detect")

    def _refresh_input_previews(self) -> None:
        self._set_zoomable_from_path(self.point_view, self.current_point_path, "No point image")
        self._set_zoomable_from_path(self.line_view, self.current_line_path, "No line image")

    def _set_zoomable_from_path(
        self,
        viewer: ZoomableImageCanvas,
        path: Path | None,
        empty_text: str,
    ) -> None:
        if path is None or not path.exists():
            viewer.clear(empty_text)
            return

        image = cv2.imread(str(path))
        if image is None:
            viewer.clear(f"Cannot read:\n{path.name}")
            return
        viewer.set_image(image)

    def _set_label_image(
        self,
        widget: tk.Label,
        image_bgr,
        key: str,
        max_size: tuple[int, int],
    ) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(max_size, RESAMPLING.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._image_refs[key] = photo
        widget.configure(image=photo, text="")

    def browse_point(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Point Image",
            filetypes=[("JPEG images", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not filename:
            return
        point_path = Path(filename)
        self.current_point_path = point_path
        self.current_line_path = None
        self.current_raw_path = None
        self._guess_related_paths(from_path=point_path, target_kind="point")
        self._refresh_paths()
        self._refresh_input_previews()
        self.status_var.set("Custom point image loaded. Click Run to update the result.")

    def browse_line(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Line Image",
            filetypes=[("JPEG images", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not filename:
            return
        line_path = Path(filename)
        self.current_line_path = line_path
        self.current_point_path = None
        self.current_raw_path = None
        self._guess_related_paths(from_path=line_path, target_kind="line")
        self._refresh_paths()
        self._refresh_input_previews()
        self.status_var.set("Custom line image loaded. Click Run to update the result.")

    def _guess_related_paths(self, from_path: Path, target_kind: str) -> None:
        case_dir = from_path.parent
        side = infer_side(normalize_name(from_path.name))
        if side is None:
            return

        for candidate in case_dir.iterdir():
            if not candidate.is_file() or candidate.name == ".DS_Store":
                continue
            kind = classify_file(candidate)
            candidate_side = infer_side(normalize_name(candidate.name))
            if target_kind == "point" and kind == "line" and candidate_side == side:
                self.current_line_path = candidate
            if target_kind == "line" and kind == "point" and candidate_side == side:
                self.current_point_path = candidate
            if kind == "raw" and candidate_side in {side, "RL"}:
                self.current_raw_path = candidate

    def run_measurement(self) -> None:
        if self.current_point_path is None or self.current_line_path is None:
            messagebox.showerror("Missing Input", "Please select both the point image and the line image.")
            return

        self.status_var.set("Running measurement...")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            result, _ = measure_case(self.current_point_path, self.current_line_path, self.current_raw_path)
        except Exception as exc:
            self.root.config(cursor="")
            self.status_var.set("Measurement failed.")
            messagebox.showerror("Measurement Error", str(exc))
            return
        finally:
            self.root.config(cursor="")

        self.result_data = result
        self.current_raw_path = Path(result["raw_path"])
        self._refresh_paths()
        self.result_view.set_image(result["combined_image"])
        self.upper_angle_var.set(f"{result['upper_angle_label']}: {result['mldfa_angle']:.2f} deg")
        self.lower_angle_var.set(f"{result['lower_angle_label']}: {result['mpta_angle']:.2f} deg")
        self.status_var.set("Measurement complete. You can save the combined result image if needed.")

    def save_result(self) -> None:
        if self.result_data is None:
            messagebox.showinfo("No Result", "Run a measurement before saving the result image.")
            return

        default_name = "measurement_result.jpg"
        if self.current_point_path is not None:
            stem = normalize_name(self.current_point_path.stem).replace("'", "").replace(" ", "_")
            default_name = f"{stem}_combined.jpg"

        filename = filedialog.asksaveasfilename(
            title="Save Combined Result",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not filename:
            return

        out_path = Path(filename)
        ok = cv2.imwrite(str(out_path), self.result_data["combined_image"])
        if not ok:
            messagebox.showerror("Save Error", f"Could not save result to {out_path}")
            return
        self.status_var.set(f"Saved result image to {out_path}")


def main() -> None:
    root = tk.Tk()
    app = MeasurementDemoApp(root)
    if not app.pairs:
        messagebox.showwarning(
            "No Dataset Pairs",
            "No valid point/line pairs were found under images/line_point. You can still browse files manually.",
        )
    root.mainloop()


if __name__ == "__main__":
    main()

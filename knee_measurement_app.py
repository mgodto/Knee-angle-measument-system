#!/usr/bin/env python3

"""Doctor-facing raw X-ray to automated knee measurement desktop app."""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import tempfile
import threading
import tkinter as tk
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np

from knee_gui_widgets import InteractiveImageCanvas
from knee_model_runtime import (
    AnalysisResult,
    AppConfig,
    KneeAnalysisService,
    ModelSelection,
    ModelSpec,
    SideRequiredError,
    clear_model_preference,
    combine_warnings,
    coordinate_display_name,
    coordinate_geometry_warnings,
    create_model_adapter,
    export_record,
    load_app_config,
    load_model_preference,
    measurement_out_of_range_angles,
    measurement_warnings,
    measurement_from_coordinates,
    model_spec_with_checkpoint,
    resolve_model_selection,
    resolve_side,
    save_model_preference,
    write_result_bundle,
)
from measure_angles import (
    ANNOTATION_LINE_NAMES,
    ANNOTATION_POINT_NAMES,
    infer_knee_side_from_sources,
    normalize_measurement_side,
    read_color,
)


APP_VERSION = "0.6.0"
APP_TITLE = "下肢全長X線 自動計測"
APP_RELEASE_CHANNEL = "INTERNAL RESEARCH CANDIDATE - NOT FOR CLINICAL USE"

INPUT_SCOPE_LABELS = {
    "single": "片側画像",
    "bilateral": "両側画像",
}
INPUT_SCOPE_BY_LABEL = {label: scope for scope, label in INPUT_SCOPE_LABELS.items()}
BILATERAL_ROI_OVERLAP_FRACTION = 0.08
SCREEN_SIDE_LABELS = {
    "left": "画面左側の脚",
    "right": "画面右側の脚",
}
SCREEN_SIDE_BY_LABEL = {label: side for side, label in SCREEN_SIDE_LABELS.items()}
SCREEN_SIDE_UNSELECTED = "選択してください"

MODEL_MODE_LABELS = {
    "auto": "自動判定",
    "bone": "Bone（人工関節なし）",
    "tka": "TKA（人工関節あり）",
    "mixed": "Mixed（判定不明）",
}
MODEL_LABEL_TO_MODE = {label: mode for mode, label in MODEL_MODE_LABELS.items()}
MODEL_SHORT_LABELS = {
    "bone": "Bone",
    "tka": "TKA",
    "mixed": "Mixed",
    "external": "外部",
}
MODEL_SOURCE_LABELS = {
    "filename": "ファイル名／フォルダから自動判定",
    "auto_fallback_unknown": "種類不明のため Mixed を使用",
    "auto_fallback_conflict": "種類の候補が競合したため Mixed を使用",
    "manual_override": "手動指定",
    "external_model": "外部モデルを手動指定",
}

COLORS = {
    "page": "#eef2f7",
    "header": "#0f172a",
    "header_muted": "#a8b4c7",
    "card": "#ffffff",
    "border": "#d8e0ea",
    "text": "#172033",
    "muted": "#65748b",
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "success": "#15803d",
    "warning": "#b45309",
    "danger": "#b91c1c",
    "image_bg": "#09111f",
    "point": "#22d3ee",
    "line": "#34d399",
    "edited": "#f59e0b",
    "selected": "#fde047",
}

POINT_LABELS = {name: coordinate_display_name(name) for name in ANNOTATION_POINT_NAMES}
LINE_LABELS = {
    "upper_line_p1": "大腿骨関節線端点 1",
    "upper_line_p2": "大腿骨関節線端点 2",
    "lower_line_p1": "脛骨関節線端点 1",
    "lower_line_p2": "脛骨関節線端点 2",
}
TABLE_ROWS = [
    *((name, POINT_LABELS[name]) for name in ANNOTATION_POINT_NAMES),
    *((name, LINE_LABELS[name]) for name in LINE_LABELS),
]


def clone_points(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(point, dtype=np.float32).copy() for name, point in points.items()}


def clone_lines(lines: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        line_name: {
            endpoint: np.asarray(point, dtype=np.float32).copy()
            for endpoint, point in endpoints.items()
        }
        for line_name, endpoints in lines.items()
    }


def safe_export_stem(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" .")
    return value or "knee_xray"


def measurement_export_stem(value: str, *, input_scope: str, side: str) -> str:
    stem = safe_export_stem(value)
    if input_scope != "bilateral raster X-ray":
        return stem
    normalized_side = normalize_measurement_side(side)
    if normalized_side is None:
        raise ValueError("両側画像の書き出しには解剖学的なL/Rが必要です。")
    return f"{stem}_{normalized_side}"


def model_version_expectation(value: str) -> tuple[str, str]:
    model_key, separator, version = value.partition("=")
    model_key = model_key.strip()
    version = version.strip()
    if not separator or model_key not in {"bone", "tka", "mixed"} or not version:
        raise argparse.ArgumentTypeError(
            "expected model version must be bone=VERSION, tka=VERSION, or mixed=VERSION"
        )
    return model_key, version


def validate_expected_model_versions(
    config: AppConfig,
    infos: dict[str, Any],
    expectations: list[tuple[str, str]],
) -> None:
    expected_versions = dict(expectations)
    if len(expected_versions) != len(expectations):
        raise RuntimeError("Duplicate --expected-model-version key.")
    if expected_versions and set(expected_versions) != set(config.models):
        raise RuntimeError(
            "Expected model versions must cover exactly bone, tka, and mixed."
        )
    for key, expected_version in expected_versions.items():
        if config.models[key].version.strip().lower() != "auto":
            raise RuntimeError(
                f"Model {key} must use config version 'auto' when an exact "
                "checkpoint version is required."
            )
        actual_version = infos[key].version
        if actual_version != expected_version:
            raise RuntimeError(
                f"Model {key} version {actual_version!r} does not match "
                f"expected {expected_version!r}."
            )


class KneeMeasurementApp:
    def __init__(self, root: tk.Tk, config_path: Path | None = None) -> None:
        self.root = root
        self.root.title(f"{APP_TITLE} · v{APP_VERSION} · {APP_RELEASE_CHANNEL}")
        self.root.geometry("1500x930")
        self.root.minsize(1080, 720)
        self.root.configure(bg=COLORS["page"])

        self.config: AppConfig | None = None
        self.default_model_spec: ModelSpec | None = None
        self.model_spec: ModelSpec | None = None
        self.model_specs: dict[str, ModelSpec] = {}
        self.model_cache: dict[str, tuple[ModelSpec, KneeAnalysisService]] = {}
        self.model_selection = ModelSelection("auto", "mixed", "auto_fallback_unknown")
        self.active_model_key: str | None = None
        self.pending_model_key: str | None = None
        self.preference_warning: str | None = None
        self.service: KneeAnalysisService | None = None
        self.analysis: AnalysisResult | None = None
        self.raw_path: Path | None = None
        self.raw_image: np.ndarray | None = None
        self.displayed_source_sha256: str | None = None
        self.points: dict[str, np.ndarray] = {}
        self.lines: dict[str, dict[str, np.ndarray]] = {}
        self.measurement: dict[str, Any] | None = None
        self.edited_keys: set[str] = set()
        self.side_source_hint = "unknown"
        self.roi_confirmed = False
        self.confirmed_crop_box: tuple[int, int, int, int] | None = None
        self.roi_selection_method = ""
        self.history: list[dict[str, Any]] = []
        self.drag_target: tuple[str, ...] | None = None
        self.drag_snapshot: dict[str, Any] | None = None
        self.drag_changed = False

        self.task_events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.task_id = 0
        self.busy = False
        self.closing = False

        self.side_var = tk.StringVar(value="自動判定")
        self.input_scope_var = tk.StringVar(value=INPUT_SCOPE_LABELS["single"])
        self.screen_side_var = tk.StringVar(value=SCREEN_SIDE_UNSELECTED)
        self.roi_split_var = tk.DoubleVar(value=50.0)
        self.roi_split_label_var = tk.StringVar(value="分割位置：50%（中央8%重複）")
        self.roi_status_var = tk.StringVar(value="両側画像では、対象脚のROI確認後にAI解析を行います。")
        self.model_mode_var = tk.StringVar(value=MODEL_MODE_LABELS["auto"])
        self.model_source_var = tk.StringVar(value="モデル選択：自動判定（画像未選択）")
        self.path_var = tk.StringVar(value="画像が選択されていません")
        self.status_var = tk.StringVar(value="AIモデルを準備しています…")
        self.model_badge_var = tk.StringVar(value="モデル：未読み込み")
        self.result_state_var = tk.StringVar(value="入力待ち")
        self.model_detail_var = tk.StringVar(value="モデル情報は読み込み後に表示されます。")
        self.warning_title_var = tk.StringVar(value="解析結果はありません")
        self.warning_banner_var = tk.StringVar(value="確認事項：解析後に表示します")
        self.angle_vars = {
            "mLDFA": tk.StringVar(value="—"),
            "MPTA": tk.StringVar(value="—"),
            "JLCA": tk.StringVar(value="—"),
            "HKA": tk.StringVar(value="—"),
        }
        self.angle_cards: dict[str, tk.Frame] = {}
        self.angle_name_labels: dict[str, tk.Label] = {}
        self.angle_value_labels: dict[str, tk.Label] = {}

        self._configure_styles()
        self._build_ui()
        self._build_menu()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self._poll_task_events)

        try:
            self.config = load_app_config(config_path)
            self.default_model_spec = self.config.model
            self.model_specs = dict(self.config.models or {self.config.default_model_key: self.config.model})
            preferred_spec, self.preference_warning = load_model_preference(self.default_model_spec)
        except Exception as exc:
            self._set_model_unavailable(str(exc))
        else:
            if preferred_spec.checkpoint != self.default_model_spec.checkpoint:
                self.model_mode_var.set("外部モデル")
                self.model_selection = ModelSelection("external", "external", "external_model")
                self.model_spec = preferred_spec
                external_selection = self.model_selection
                self.root.after(
                    80,
                    lambda spec=preferred_spec, selection=external_selection: self._start_model_load(
                        spec,
                        startup=True,
                        model_key="external",
                        selection=selection,
                    ),
                )
            else:
                self.model_selection = self._resolve_model_selection("auto")
                self.model_spec = self.model_specs[self.model_selection.model_key]
                startup_selection = self.model_selection
                self.root.after(
                    80,
                    lambda selection=startup_selection: self._activate_model_selection(selection, startup=True),
                )

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["page"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["card"], foreground=COLORS["muted"])
        style.configure("Section.TLabel", background=COLORS["card"], foreground=COLORS["text"], font=("TkDefaultFont", 13, "bold"))
        style.configure("TButton", padding=(10, 7))
        style.configure("TCombobox", padding=5)
        style.configure("Treeview", rowheight=27, font=("TkDefaultFont", 10), fieldbackground=COLORS["card"])
        style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"), padding=6)
        style.configure("TNotebook", background=COLORS["card"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 7))

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=COLORS["header"], height=78)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)
        title_box = tk.Frame(header, bg=COLORS["header"])
        title_box.grid(row=0, column=0, sticky="w", padx=20, pady=13)
        tk.Label(
            title_box,
            text=APP_TITLE,
            bg=COLORS["header"],
            fg="#ffffff",
            font=("TkDefaultFont", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="片側／両側下肢全長X線・ランドマーク自動推定／角度計測",
            bg=COLORS["header"],
            fg=COLORS["header_muted"],
            font=("TkDefaultFont", 10),
        ).pack(anchor="w", pady=(3, 0))

        model_box = tk.Frame(header, bg=COLORS["header"])
        model_box.grid(row=0, column=1, sticky="e", padx=(10, 12))
        tk.Label(
            model_box,
            textvariable=self.model_badge_var,
            bg="#172554",
            fg="#dbeafe",
            padx=12,
            pady=6,
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left", padx=6)
        tk.Label(
            model_box,
            text=APP_RELEASE_CHANNEL,
            bg="#451a03",
            fg="#fed7aa",
            padx=12,
            pady=6,
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left", padx=6)

        toolbar_shell = tk.Frame(self.root, bg=COLORS["page"])
        toolbar_shell.grid(row=1, column=0, sticky="ew", padx=14, pady=(12, 8))
        toolbar = tk.Frame(
            toolbar_shell,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        toolbar.pack(fill="x")
        toolbar.grid_columnconfigure(8, weight=1)

        self.open_button = tk.Button(
            toolbar,
            text="X線画像を開く…",
            command=self.open_image,
            bg=COLORS["primary"],
            fg="#ffffff",
            activebackground=COLORS["primary_dark"],
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=9,
            font=("TkDefaultFont", 10, "bold"),
            cursor="hand2",
        )
        self.open_button.grid(row=0, column=0, rowspan=2, padx=(12, 8), pady=10)
        self.rerun_button = ttk.Button(toolbar, text="再解析", command=self.rerun_inference, state="disabled")
        self.rerun_button.grid(row=0, column=1, rowspan=2, padx=4, pady=10)

        ttk.Label(toolbar, text="左右", style="Card.TLabel").grid(row=0, column=2, sticky="sw", padx=(12, 4), pady=(8, 0))
        self.side_combo = ttk.Combobox(
            toolbar,
            textvariable=self.side_var,
            values=("自動判定", "L", "R"),
            state="readonly",
            width=7,
        )
        self.side_combo.grid(row=1, column=2, sticky="nw", padx=(12, 4), pady=(0, 8))
        self.side_combo.bind("<<ComboboxSelected>>", self._on_side_changed)

        ttk.Label(toolbar, text="入力画像", style="Card.TLabel").grid(
            row=0,
            column=3,
            sticky="sw",
            padx=(12, 4),
            pady=(8, 0),
        )
        self.bilateral_crop_button = ttk.Button(
            toolbar,
            text="両側画像を切り出す",
            command=self._toggle_bilateral_mode,
            state="disabled",
            width=16,
        )
        self.bilateral_crop_button.grid(row=1, column=3, sticky="nw", padx=(12, 4), pady=(0, 8))

        ttk.Label(toolbar, text="画像種類 / AIモデル", style="Card.TLabel").grid(
            row=0,
            column=4,
            sticky="sw",
            padx=(12, 4),
            pady=(8, 0),
        )
        self.model_combo = ttk.Combobox(
            toolbar,
            textvariable=self.model_mode_var,
            values=tuple(MODEL_MODE_LABELS.values()),
            state="readonly",
            width=19,
        )
        self.model_combo.grid(row=1, column=4, sticky="nw", padx=(12, 4), pady=(0, 8))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_mode_changed)

        self.model_button = ttk.Menubutton(toolbar, text="AIモデル…")
        self.model_menu = tk.Menu(self.model_button, tearoff=False)
        self.model_menu.add_command(label="AIモデルファイルを選択…", command=self.browse_weight)
        self.model_menu.add_command(label="自動判定に戻す", command=self.restore_builtin_model)
        self.model_button.configure(menu=self.model_menu)
        self.model_button.grid(row=0, column=5, rowspan=2, padx=(8, 4), pady=10)
        self.undo_button = ttk.Button(toolbar, text="修正を元に戻す", command=self.undo_edit, state="disabled")
        self.undo_button.grid(row=0, column=6, rowspan=2, padx=4, pady=10)
        self.reset_button = ttk.Button(toolbar, text="AI推定位置に戻す", command=self.reset_to_prediction, state="disabled")
        self.reset_button.grid(row=0, column=7, rowspan=2, padx=4, pady=10)

        path_box = tk.Frame(toolbar, bg=COLORS["card"])
        path_box.grid(row=0, column=8, rowspan=2, sticky="ew", padx=(14, 10), pady=8)
        tk.Label(
            path_box,
            textvariable=self.path_var,
            bg=COLORS["card"],
            fg=COLORS["text"],
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(fill="x")
        tk.Label(
            path_box,
            text="JPG / PNG / BMP / TIFF（両側画像は上部ボタンからROIを切り出します）",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            anchor="w",
            font=("TkDefaultFont", 9),
        ).pack(fill="x", pady=(3, 0))
        tk.Label(
            path_box,
            textvariable=self.model_source_var,
            bg=COLORS["card"],
            fg=COLORS["primary_dark"],
            anchor="w",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(fill="x", pady=(2, 0))

        self.export_button = ttk.Button(toolbar, text="結果を書き出す…", command=self.export_result, state="disabled")
        self.export_button.grid(row=0, column=9, rowspan=2, padx=(4, 12), pady=10)

        self.roi_panel = tk.Frame(
            toolbar_shell,
            bg="#eff6ff",
            highlightbackground="#93c5fd",
            highlightthickness=1,
        )
        tk.Label(
            self.roi_panel,
            text="両側画像：解剖学的なL/Rとは別に、画面上の対象脚を確認してください。",
            bg="#eff6ff",
            fg=COLORS["primary_dark"],
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left", padx=(10, 6), pady=7)
        self.screen_side_combo = ttk.Combobox(
            self.roi_panel,
            textvariable=self.screen_side_var,
            values=(SCREEN_SIDE_UNSELECTED, *SCREEN_SIDE_LABELS.values()),
            state="readonly",
            width=15,
        )
        self.screen_side_combo.pack(side="left", padx=4, pady=5)
        self.screen_side_combo.bind("<<ComboboxSelected>>", self._on_screen_side_changed)
        self.swap_screen_side_button = ttk.Button(
            self.roi_panel,
            text="対象を入れ替え",
            command=self._swap_screen_side,
        )
        self.swap_screen_side_button.pack(side="left", padx=4, pady=5)
        self.roi_split_scale = ttk.Scale(
            self.roi_panel,
            from_=30.0,
            to=70.0,
            variable=self.roi_split_var,
            command=self._on_roi_split_changed,
            length=130,
        )
        self.roi_split_scale.pack(side="left", padx=(10, 3), pady=5)
        tk.Label(
            self.roi_panel,
            textvariable=self.roi_split_label_var,
            bg="#eff6ff",
            fg=COLORS["muted"],
            width=22,
            anchor="w",
        ).pack(side="left", padx=(0, 5), pady=7)
        self.confirm_roi_button = ttk.Button(
            self.roi_panel,
            text="ROIを確認して解析",
            command=self._confirm_bilateral_roi,
            state="disabled",
        )
        self.confirm_roi_button.pack(side="left", padx=5, pady=5)
        tk.Label(
            self.roi_panel,
            textvariable=self.roi_status_var,
            bg="#eff6ff",
            fg=COLORS["warning"],
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=(5, 10), pady=7)

        content = tk.Frame(self.root, bg=COLORS["page"])
        content.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 10))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=46, uniform="panels")
        content.grid_columnconfigure(1, weight=54, uniform="panels")

        left = tk.Frame(content, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self._section_header(
            left,
            "1  元画像とAI推定点",
            "マーカーをドラッグして修正できます。離すと角度を再計算します。",
        ).grid(row=0, column=0, sticky="ew")
        input_shell = tk.Frame(left, bg=COLORS["card"])
        input_shell.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        input_shell.grid_rowconfigure(0, weight=1)
        input_shell.grid_columnconfigure(0, weight=1)
        self.input_view = InteractiveImageCanvas(
            input_shell,
            empty_text="下肢全長X線画像を開いてください\n両側画像の場合は上部の切り出しボタンを使用します",
            bg=COLORS["image_bg"],
        )
        self.input_view.grid(row=0, column=0, sticky="nsew")
        self.input_view.set_overlay_drawer(self._draw_prediction_overlay)
        self.input_view.set_pointer_callbacks(self._on_canvas_press, self._on_canvas_drag, self._on_canvas_release)
        legend = tk.Frame(left, bg=COLORS["card"])
        legend.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        self._legend_item(legend, COLORS["point"], "AIランドマーク").pack(side="left", padx=(0, 14))
        self._legend_item(legend, COLORS["line"], "AI関節線").pack(side="left", padx=(0, 14))
        self._legend_item(legend, COLORS["edited"], "手動修正").pack(side="left", padx=(0, 14))
        tk.Label(
            legend,
            text="ホイール：拡縮／右ドラッグ：移動／ダブルクリック：全体",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=("TkDefaultFont", 9),
        ).pack(side="right")

        right = tk.Frame(content, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(1, weight=5)
        right.grid_rowconfigure(4, weight=3)
        right.grid_columnconfigure(0, weight=1)
        right_header = self._section_header(right, "2  自動計測結果", "計測結果画像・各角度・全ランドマーク座標")
        right_header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            right_header,
            textvariable=self.result_state_var,
            bg="#e2e8f0",
            fg=COLORS["muted"],
            padx=10,
            pady=4,
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="right", padx=(8, 12), pady=12)

        result_shell = tk.Frame(right, bg=COLORS["card"])
        result_shell.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        result_shell.grid_rowconfigure(0, weight=1)
        result_shell.grid_columnconfigure(0, weight=1)
        self.result_view = InteractiveImageCanvas(
            result_shell,
            empty_text="解析後、計測結果を重ねた画像がここに表示されます",
            bg=COLORS["image_bg"],
        )
        self.result_view.grid(row=0, column=0, sticky="nsew")
        self.result_view.canvas.configure(cursor="arrow")

        angle_strip = tk.Frame(right, bg=COLORS["card"])
        angle_strip.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        for index, (key, subtitle) in enumerate(
            (
                ("mLDFA", "機械的外側遠位大腿骨角"),
                ("MPTA", "内側近位脛骨角"),
                ("JLCA", "関節裂隙収束角"),
                ("HKA", "股関節–膝関節–足関節角"),
            )
        ):
            angle_strip.grid_columnconfigure(index, weight=1, uniform="angles")
            card = tk.Frame(
                angle_strip,
                bg="#f8fafc",
                highlightbackground=COLORS["border"],
                highlightthickness=1,
            )
            card.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0 if index == 3 else 4))
            name_label = tk.Label(
                card,
                text=key,
                bg="#f8fafc",
                fg=COLORS["muted"],
                font=("TkDefaultFont", 9, "bold"),
            )
            name_label.pack(anchor="w", padx=10, pady=(8, 0))
            value_label = tk.Label(
                card,
                textvariable=self.angle_vars[key],
                bg="#f8fafc",
                fg=COLORS["text"],
                font=("TkDefaultFont", 18, "bold"),
            )
            value_label.pack(anchor="w", padx=10, pady=(0, 1))
            tk.Label(card, text=subtitle, bg="#f8fafc", fg=COLORS["muted"], font=("TkDefaultFont", 8)).pack(
                anchor="w", padx=10, pady=(0, 8)
            )
            self.angle_cards[key] = card
            self.angle_name_labels[key] = name_label
            self.angle_value_labels[key] = value_label

        self.warning_banner = tk.Frame(
            right,
            bg="#f1f5f9",
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        self.warning_banner.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.warning_banner_label = tk.Label(
            self.warning_banner,
            textvariable=self.warning_banner_var,
            bg="#f1f5f9",
            fg=COLORS["muted"],
            anchor="w",
            font=("TkDefaultFont", 9, "bold"),
        )
        self.warning_banner_label.pack(side="left", fill="x", expand=True, padx=10, pady=7)
        self.warning_detail_button = ttk.Button(
            self.warning_banner,
            text="詳細を確認",
            command=self._show_quality_tab,
            state="disabled",
        )
        self.warning_detail_button.pack(side="right", padx=(4, 8), pady=4)

        notebook_shell = tk.Frame(right, bg=COLORS["card"])
        notebook_shell.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        notebook_shell.grid_rowconfigure(0, weight=1)
        notebook_shell.grid_columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(notebook_shell)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self._build_coordinate_tab()
        self._build_quality_tab()

        footer = tk.Frame(self.root, bg=COLORS["header"], height=34)
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_propagate(False)
        footer.grid_columnconfigure(0, weight=1)
        tk.Label(
            footer,
            textvariable=self.status_var,
            bg=COLORS["header"],
            fg="#dbe7f6",
            anchor="w",
            font=("TkDefaultFont", 9),
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=7)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=150)
        self.progress.grid(row=0, column=1, padx=14, pady=7)

    def _section_header(self, master: tk.Misc, title: str, subtitle: str) -> tk.Frame:
        frame = tk.Frame(master, bg=COLORS["card"])
        text_box = tk.Frame(frame, bg=COLORS["card"])
        text_box.pack(side="left", padx=12, pady=10)
        tk.Label(
            text_box,
            text=title,
            bg=COLORS["card"],
            fg=COLORS["text"],
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            text_box,
            text=subtitle,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=("TkDefaultFont", 9),
        ).pack(anchor="w", pady=(2, 0))
        return frame

    def _legend_item(self, master: tk.Misc, color: str, text: str) -> tk.Frame:
        frame = tk.Frame(master, bg=COLORS["card"])
        tk.Label(frame, text="●", bg=COLORS["card"], fg=color, font=("TkDefaultFont", 12, "bold")).pack(side="left")
        tk.Label(frame, text=text, bg=COLORS["card"], fg=COLORS["muted"], font=("TkDefaultFont", 9)).pack(
            side="left", padx=(3, 0)
        )
        return frame

    def _build_coordinate_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="Card.TFrame")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        self.coordinate_tree = ttk.Treeview(
            tab,
            columns=("landmark", "x", "y", "score", "source"),
            show="headings",
            height=5,
        )
        for column, text, width, anchor in (
            ("landmark", "ランドマーク", 190, "w"),
            ("x", "X (px)", 78, "e"),
            ("y", "Y (px)", 78, "e"),
            ("score", "AIスコア", 78, "e"),
            ("source", "修正区分", 76, "center"),
        ):
            self.coordinate_tree.heading(column, text=text)
            self.coordinate_tree.column(column, width=width, minwidth=55, anchor=anchor, stretch=column == "landmark")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.coordinate_tree.yview)
        self.coordinate_tree.configure(yscrollcommand=scrollbar.set)
        self.coordinate_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        self.notebook.add(tab, text="座標一覧（12）")

    def _build_quality_tab(self) -> None:
        tab = tk.Frame(self.notebook, bg=COLORS["card"])
        self.quality_tab = tab
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        tk.Label(
            tab,
            textvariable=self.warning_title_var,
            bg=COLORS["card"],
            fg=COLORS["warning"],
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        tk.Label(
            tab,
            textvariable=self.model_detail_var,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            anchor="w",
            justify="left",
            font=("TkDefaultFont", 9),
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.warning_text = tk.Text(
            tab,
            height=4,
            wrap="word",
            relief="flat",
            bg="#fffaf0",
            fg=COLORS["text"],
            padx=9,
            pady=7,
            font=("TkDefaultFont", 9),
            state="disabled",
        )
        warning_scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.warning_text.yview)
        self.warning_text.configure(yscrollcommand=warning_scrollbar.set)
        self.warning_text.grid(row=2, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        warning_scrollbar.grid(row=2, column=1, sticky="ns", padx=(0, 8), pady=(0, 8))
        self.notebook.add(tab, text="確認事項（0）・AIモデル")
        self.notebook.select(0)

    def _show_quality_tab(self) -> None:
        self.notebook.select(self.quality_tab)

    def _bind_shortcuts(self) -> None:
        for sequence in ("<Control-KeyPress-o>", "<Command-KeyPress-o>"):
            self.root.bind_all(sequence, lambda _event: self.open_image())
        for sequence in ("<Control-KeyPress-e>", "<Command-KeyPress-e>"):
            self.root.bind_all(sequence, lambda _event: self.export_result())
        for sequence in ("<Control-KeyPress-z>", "<Command-KeyPress-z>"):
            self.root.bind_all(sequence, lambda _event: self.undo_edit())

    def _build_menu(self) -> None:
        accelerator_prefix = "Command" if sys.platform == "darwin" else "Ctrl"
        menubar = tk.Menu(self.root, tearoff=False)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="X線画像を開く…", accelerator=f"{accelerator_prefix}+O", command=self.open_image)
        file_menu.add_command(label="結果を書き出す…", accelerator=f"{accelerator_prefix}+E", command=self.export_result)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.close)
        menubar.add_cascade(label="ファイル", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="修正を元に戻す", accelerator=f"{accelerator_prefix}+Z", command=self.undo_edit)
        edit_menu.add_command(label="AI推定位置に戻す", command=self.reset_to_prediction)
        menubar.add_cascade(label="編集", menu=edit_menu)

        model_menu = tk.Menu(menubar, tearoff=False)
        model_menu.add_command(label="AIモデルファイルを選択…", command=self.browse_weight)
        model_menu.add_command(label="自動判定に戻す", command=self.restore_builtin_model)
        menubar.add_cascade(label="AIモデル", menu=model_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="このアプリについて", command=self._show_about)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        self.root.configure(menu=menubar)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "このアプリについて",
            f"{APP_TITLE}  v{APP_VERSION}\n"
            f"{APP_RELEASE_CHANNEL}\n\n"
            "研究用ソフトウェアです。すべてのランドマークと角度を医師が確認してください。\n"
            "GPUは不要です。CPUで動作し、画像や結果を外部へ送信しません。",
        )

    def _requested_input_scope(self) -> str:
        variable = getattr(self, "input_scope_var", None)
        label = variable.get() if variable is not None else INPUT_SCOPE_LABELS["single"]
        return INPUT_SCOPE_BY_LABEL.get(label, "single")

    def _input_scope(self) -> str:
        return self._requested_input_scope()

    def _screen_side(self) -> str | None:
        variable = getattr(self, "screen_side_var", None)
        return SCREEN_SIDE_BY_LABEL.get(variable.get()) if variable is not None else None

    @staticmethod
    def _conventional_screen_side(anatomical_side: str) -> str:
        # Standard radiographic display places the patient's left on the
        # viewer's right.  This is only a suggestion; bilateral inference is
        # blocked until the doctor confirms the visible target leg.
        return "right" if anatomical_side == "L" else "left"

    def _candidate_crop_box(self) -> tuple[int, int, int, int] | None:
        if self._input_scope() != "bilateral" or self.raw_image is None:
            return None
        screen_side = self._screen_side()
        if screen_side is None:
            return None
        height, width = self.raw_image.shape[:2]
        split_percent = float(self.roi_split_var.get())
        split_x = int(round(width * split_percent / 100.0))
        split_x = min(max(split_x, 1), width - 1)
        overlap = int(round(width * BILATERAL_ROI_OVERLAP_FRACTION))
        if screen_side == "left":
            return (0, 0, min(split_x + overlap, width), height)
        return (max(split_x - overlap, 0), 0, width, height)

    def _analysis_matches_current_input(self) -> bool:
        if self.analysis is None:
            return False
        if self._input_scope() == "single":
            return getattr(self.analysis, "inference_roi", None) is None
        return bool(
            getattr(self, "roi_confirmed", False)
            and self.confirmed_crop_box is not None
            and getattr(self.analysis, "inference_roi", None) == self.confirmed_crop_box
            and getattr(self.analysis, "roi_confirmed", False)
        )

    def _coordinates_within_confirmed_roi(self) -> bool:
        if self._input_scope() != "bilateral":
            return True
        if self.confirmed_crop_box is None:
            return False
        x0, y0, x1, y1 = self.confirmed_crop_box
        coordinates = list(self.points.values())
        coordinates.extend(
            point
            for endpoints in self.lines.values()
            for point in endpoints.values()
        )
        if not coordinates:
            return False
        return all(
            x0 <= float(point[0]) < x1 and y0 <= float(point[1]) < y1
            for point in coordinates
        )

    def _roi_coordinate_warnings(self) -> tuple[str, ...]:
        if self._input_scope() != "bilateral" or self._coordinates_within_confirmed_roi():
            return ()
        return (
            "確認済みROIの外にランドマークがあるため、書き出しを無効にしました。"
            "対象脚またはROIを再確認してください。",
        )

    def _input_ready_for_inference(self) -> bool:
        if self._input_scope() == "single":
            return True
        return bool(getattr(self, "roi_confirmed", False) and self.confirmed_crop_box is not None)

    def _update_roi_panel_visibility(self) -> None:
        panel = getattr(self, "roi_panel", None)
        if panel is None:
            return
        if self._input_scope() == "bilateral":
            if not panel.winfo_manager():
                panel.pack(fill="x", pady=(6, 0))
        else:
            panel.pack_forget()
        button = getattr(self, "bilateral_crop_button", None)
        if button is not None:
            button.configure(
                text=(
                    "片側画像に戻す"
                    if self._input_scope() == "bilateral"
                    else "両側画像を切り出す"
                )
            )

    def _ensure_input_preview(self) -> bool:
        if self.raw_path is None:
            return False
        if self.raw_image is None:
            try:
                self.raw_image = read_color(self.raw_path)
            except Exception as exc:
                self.status_var.set(f"画像のプレビューを読み込めません：{exc}")
                messagebox.showerror("画像を読み込めません", str(exc))
                return False
        if not self.input_view.has_image:
            self.input_view.set_image(self.raw_image, reset_view=True)
        self.input_view.redraw_overlay()
        return True

    def _suggest_screen_side(self) -> None:
        try:
            side = self._effective_side()
        except SideRequiredError:
            self.screen_side_var.set(SCREEN_SIDE_UNSELECTED)
            return
        self.screen_side_var.set(SCREEN_SIDE_LABELS[self._conventional_screen_side(side)])

    def _invalidate_roi(self, status: str) -> None:
        self.task_id = getattr(self, "task_id", 0) + 1
        self.roi_confirmed = False
        self.confirmed_crop_box = None
        self.roi_selection_method = ""
        if self.analysis is not None or self.points or self.measurement is not None:
            self._clear_analysis(keep_raw=True)
        self.result_state_var.set("ROI確認待ち")
        self.roi_status_var.set(status)
        self.status_var.set(status)
        if hasattr(self, "input_view"):
            self.input_view.redraw_overlay()
        self._refresh_action_states()

    def _toggle_bilateral_mode(self) -> None:
        if self.busy or self.raw_path is None:
            return
        target_scope = "single" if self._input_scope() == "bilateral" else "bilateral"
        self.input_scope_var.set(INPUT_SCOPE_LABELS[target_scope])
        self._on_input_scope_changed()

    def _on_input_scope_changed(self, _event: object | None = None) -> None:
        if self.busy:
            return
        self.task_id = getattr(self, "task_id", 0) + 1
        self._clear_analysis(keep_raw=True)
        self.roi_confirmed = False
        self.confirmed_crop_box = None
        self.roi_selection_method = ""
        self._update_roi_panel_visibility()
        if self._input_scope() == "bilateral":
            self._suggest_screen_side()
            if self.raw_path is not None and not self._ensure_input_preview():
                return
            if self._screen_side() is None:
                self.result_state_var.set("左右の選択待ち")
                self.roi_status_var.set("先に上部で解剖学的なLまたはRを選択してください。")
                self.status_var.set("両側画像モード：L/Rを選択してから対象脚のROIを確認してください。")
            else:
                self.result_state_var.set("ROI確認待ち")
                self.roi_status_var.set("画面上の対象脚と分割線を確認してください。")
                self.status_var.set("両側画像モード：対象脚のROIを確認するまでAI解析と書き出しは行いません。")
        elif self._input_scope() == "single" and self.raw_path is not None:
            self.screen_side_var.set(SCREEN_SIDE_UNSELECTED)
            if self._model_selection_is_active():
                self._start_inference()
        self._refresh_action_states()

    def _on_screen_side_changed(self, _event: object | None = None) -> None:
        if self.busy or self._input_scope() != "bilateral":
            return
        self._invalidate_roi("対象脚を変更しました。ROIを確認して再解析してください。")

    def _swap_screen_side(self) -> None:
        if self.busy or self._input_scope() != "bilateral":
            return
        current = self._screen_side()
        if current is None:
            self._suggest_screen_side()
        else:
            self.screen_side_var.set(SCREEN_SIDE_LABELS["right" if current == "left" else "left"])
        self._invalidate_roi("画面上の対象脚を入れ替えました。ROIを確認してください。")

    def _on_roi_split_changed(self, value: str | None = None) -> None:
        if value is not None:
            self.roi_split_var.set(float(value))
        split = int(round(float(self.roi_split_var.get())))
        self.roi_split_label_var.set(f"分割位置：{split}%（中央8%重複）")
        if self.busy or self._input_scope() != "bilateral":
            return
        self._invalidate_roi("分割線を変更しました。対象脚のROIを再確認してください。")

    def _confirm_bilateral_roi(self) -> None:
        if self.busy or self.raw_path is None or self._input_scope() != "bilateral":
            return
        try:
            anatomical_side = self._effective_side()
        except SideRequiredError:
            self._show_side_required_warning()
            return
        crop_box = self._candidate_crop_box()
        if crop_box is None:
            self.roi_status_var.set("画面左側または右側の対象脚を選択してください。")
            self._refresh_action_states()
            return
        self._clear_analysis(keep_raw=True)
        self.confirmed_crop_box = crop_box
        self.roi_confirmed = True
        conventional = self._screen_side() == self._conventional_screen_side(anatomical_side)
        adjusted = abs(float(self.roi_split_var.get()) - 50.0) >= 0.5
        if conventional and not adjusted:
            self.roi_selection_method = "patient_side_convention_overlap_confirmed"
        else:
            self.roi_selection_method = "manual_screen_side_divider_overlap_confirmed"
        self.roi_status_var.set("確認済みROIでAI解析中です。")
        self.input_view.redraw_overlay()
        self._start_inference()

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        if status:
            self.status_var.set(status)
        if busy:
            self.progress.start(12)
            self.open_button.configure(state="disabled")
            self.model_button.configure(state="disabled")
            self.rerun_button.configure(state="disabled")
            self.side_combo.configure(state="disabled")
            self.bilateral_crop_button.configure(state="disabled")
            self.model_combo.configure(state="disabled")
            self.screen_side_combo.configure(state="disabled")
            self.swap_screen_side_button.configure(state="disabled")
            self.roi_split_scale.configure(state="disabled")
            self.confirm_roi_button.configure(state="disabled")
        else:
            self.progress.stop()
            self.open_button.configure(state="normal")
            self.model_button.configure(state="normal")
            self.side_combo.configure(state="readonly")
            self.model_combo.configure(state="readonly")
            self.screen_side_combo.configure(state="readonly")
            self.swap_screen_side_button.configure(state="normal")
            self.roi_split_scale.configure(state="normal")
            self.rerun_button.configure(
                state=(
                    "normal"
                    if self.raw_path and self._model_selection_is_active() and self._input_ready_for_inference()
                    else "disabled"
                )
            )
        self._refresh_action_states()

    def _refresh_action_states(self) -> None:
        model_ready = self._model_selection_is_active()
        valid_result = (
            self.analysis is not None
            and self.measurement is not None
            and not self.busy
            and model_ready
            and self._analysis_matches_current_input()
        )
        export_ready = valid_result and self._coordinates_within_confirmed_roi()
        self.export_button.configure(state="normal" if export_ready else "disabled")
        self.reset_button.configure(state="normal" if valid_result else "disabled")
        self.undo_button.configure(state="normal" if self.history and valid_result else "disabled")
        if hasattr(self, "confirm_roi_button"):
            can_confirm_roi = bool(
                not self.busy
                and model_ready
                and self._input_scope() == "bilateral"
                and self.raw_path is not None
                and self.raw_image is not None
                and self._screen_side() is not None
                and not self.roi_confirmed
            )
            self.confirm_roi_button.configure(state="normal" if can_confirm_roi else "disabled")
        if hasattr(self, "bilateral_crop_button"):
            self.bilateral_crop_button.configure(
                state="normal" if not self.busy and self.raw_path is not None else "disabled",
                text=(
                    "片側画像に戻す"
                    if self._input_scope() == "bilateral"
                    else "両側画像を切り出す"
                ),
            )

    def _set_model_unavailable(self, error: str) -> None:
        self.model_badge_var.set("モデル：読み込み不可")
        if hasattr(self, "model_source_var"):
            self.model_source_var.set("モデル選択：読み込み不可")
        self.model_detail_var.set(error)
        self.warning_title_var.set("モデル設定エラー")
        self._set_warning_text([error])
        self._set_warning_banner("⚠ AIモデル設定エラー — 詳細を確認してください", "warning", True)
        self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
        self.status_var.set(f"モデルを使用できません：{error}")

    def _resolve_model_selection(self, requested_mode: str) -> ModelSelection:
        fallback = self.config.auto_fallback_model_key if self.config is not None else "mixed"
        sources: tuple[object, ...] = (self.raw_path,) if self.raw_path is not None else ()
        return resolve_model_selection(
            requested_mode,
            self.model_specs,
            *sources,
            fallback_model_key=fallback,
        )

    @staticmethod
    def _model_cache_key(model_key: str, spec: ModelSpec) -> str:
        if model_key == "external":
            return f"external:{spec.checkpoint}"
        return model_key

    def _set_model_selection(self, selection: ModelSelection) -> None:
        self.model_selection = selection
        if selection.requested_mode in MODEL_MODE_LABELS:
            self.model_mode_var.set(MODEL_MODE_LABELS[selection.requested_mode])
        elif selection.model_key == "external":
            self.model_mode_var.set("外部モデル")
        self._update_model_source_display()

    def _update_model_source_display(self) -> None:
        if not hasattr(self, "model_source_var"):
            return
        selection = getattr(
            self,
            "model_selection",
            ModelSelection("auto", "mixed", "auto_fallback_unknown"),
        )
        requested = (
            "自動判定"
            if selection.requested_mode == "auto"
            else MODEL_SHORT_LABELS.get(selection.model_key, selection.model_key)
        )
        source = MODEL_SOURCE_LABELS.get(selection.source, selection.source)
        target = MODEL_SHORT_LABELS.get(selection.model_key, selection.model_key)
        pending = getattr(self, "pending_model_key", None)
        if pending is not None:
            self.model_source_var.set(f"モデル選択：{requested} → {target}（{source}・読み込み中）")
            return
        service = getattr(self, "service", None)
        if service is None:
            self.model_source_var.set(f"モデル選択：{requested} → {target}（{source}）")
            return
        info = service.adapter.info
        active = MODEL_SHORT_LABELS.get(getattr(self, "active_model_key", None), "外部")
        self.model_source_var.set(
            f"モデル選択：{requested} → 使用中 {active}・{info.short_hash}（{source}）"
        )

    def _persist_model_choice(
        self,
        action: str | None,
        spec: ModelSpec,
        checkpoint_sha256: str,
    ) -> None:
        if action is None:
            return
        try:
            if action == "save":
                save_model_preference(spec, expected_sha256=checkpoint_sha256)
            elif action == "clear":
                clear_model_preference()
            self.preference_warning = None
        except Exception as exc:
            self.preference_warning = f"AIモデルは読み込まれましたが、選択内容を保存できません：{exc}"

    def _activate_model_selection(
        self,
        selection: ModelSelection,
        *,
        startup: bool = False,
        preference_action: str | None = None,
        run_inference: bool = True,
    ) -> None:
        spec = self.model_specs[selection.model_key]
        self._set_model_selection(selection)
        cache_key = self._model_cache_key(selection.model_key, spec)
        cached = self.model_cache.get(cache_key)
        if cached is None or cached[0] != spec:
            self._start_model_load(
                spec,
                startup=startup,
                preference_action=preference_action,
                model_key=selection.model_key,
                selection=selection,
                run_inference=run_inference,
            )
            return

        self.task_id += 1
        self.pending_model_key = None
        self.model_spec, self.service = cached
        self.active_model_key = selection.model_key
        info = self.service.adapter.info
        self._persist_model_choice(preference_action, self.model_spec, info.checkpoint_sha256)
        if self.raw_path is not None:
            self._clear_analysis(keep_raw=True)
        self._update_model_badge()
        self._update_model_details()
        self._set_busy(False, "検証済みのAIモデルを切り替えました。")
        if run_inference and self.raw_path is not None:
            self._start_inference()

    def _model_selection_is_active(self) -> bool:
        if not hasattr(self, "service"):
            return True
        if self.service is None or getattr(self, "pending_model_key", None) is not None:
            return False
        selection = getattr(self, "model_selection", None)
        active_model_key = getattr(self, "active_model_key", None)
        if selection is None or active_model_key is None:
            return True
        return active_model_key == selection.model_key

    def _start_model_load(
        self,
        spec: ModelSpec,
        startup: bool = False,
        preference_action: str | None = None,
        model_key: str | None = None,
        selection: ModelSelection | None = None,
        run_inference: bool = True,
    ) -> None:
        model_key = model_key or "external"
        selection = selection or ModelSelection(model_key, model_key, "manual_override")
        self._set_model_selection(selection)
        self.pending_model_key = model_key
        if self.raw_path is not None:
            # A result produced by the previous weight must never remain
            # exportable while a different model is being validated.
            self._clear_analysis(keep_raw=True)
        self.task_id += 1
        task_id = self.task_id
        self._set_busy(True, "AIモデルを検証して読み込んでいます…")
        if self.service is None:
            self.model_badge_var.set(f"モデル：{MODEL_SHORT_LABELS.get(model_key, model_key)} を読み込み中…")
        else:
            active_info = self.service.adapter.info
            active_label = MODEL_SHORT_LABELS.get(self.active_model_key, "外部")
            self.model_badge_var.set(
                f"読込中：{MODEL_SHORT_LABELS.get(model_key, model_key)}・現在 {active_label} {active_info.short_hash}"
            )
        self._update_model_source_display()

        def worker() -> None:
            try:
                adapter = create_model_adapter(spec)
                info = adapter.load()
                threshold = float(spec.options.get("low_peak_threshold", 0.35))
                service = KneeAnalysisService(
                    adapter,
                    low_peak_threshold=threshold,
                    render_component_images=False,
                )
                payload = {
                    "adapter": adapter,
                    "service": service,
                    "info": info,
                    "spec": spec,
                    "startup": startup,
                    "preference_action": preference_action,
                    "model_key": model_key,
                    "selection": selection,
                    "run_inference": run_inference,
                }
                self.task_events.put({"id": task_id, "kind": "model", "ok": True, "value": payload})
            except Exception as exc:
                self.task_events.put(
                    {
                        "id": task_id,
                        "kind": "model",
                        "ok": False,
                        "error": str(exc),
                        "startup": startup,
                        "model_key": model_key,
                        "selection": selection,
                    }
                )

        threading.Thread(target=worker, name="knee-model-loader", daemon=True).start()

    def _show_side_required_warning(self) -> None:
        self._clear_measurement_display("LまたはRを選択してください。以前の角度は消去されました。")
        self.result_state_var.set("左右の選択待ち")
        self.status_var.set("ファイル名から左右を判定できません。上部でLまたはRを選択してください。")
        self.warning_title_var.set("左右を選択してください")
        self._set_warning_text(["mLDFA、MPTA、JLCA、HKAの解剖学的方向には明確なL/R情報が必要です。"])
        self._set_warning_banner("⚠ 左右を選択してください", "warning", True)
        self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")

    def _show_roi_required_warning(self) -> None:
        self._clear_analysis(keep_raw=True)
        self.result_state_var.set("ROI確認待ち")
        self.status_var.set("両側画像の対象脚を選び、ROIを確認してから解析してください。")
        self.roi_status_var.set("未確認：対象脚と分割線を確認してください。")
        self.warning_title_var.set("両側画像のROIを確認してください")
        self._set_warning_text(["反対側の脚への誤推定を防ぐため、未確認の両側画像は解析・書き出しできません。"])
        self._set_warning_banner("⚠ 両側画像のROI確認が必要です", "warning", True)
        self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")

    def _start_inference(self) -> None:
        if self.raw_path is None or not self._model_selection_is_active():
            return
        if self._input_scope() == "bilateral" and not self._input_ready_for_inference():
            self._show_roi_required_warning()
            return
        try:
            side = self._effective_side()
        except SideRequiredError:
            self._show_side_required_warning()
            return
        explicit_side = (
            None
            if self.side_source_hint == "filename"
            else normalize_measurement_side(self.side_var.get())
        )

        self.task_id += 1
        task_id = self.task_id
        raw_path = self.raw_path
        service = self.service
        model_key = getattr(self, "active_model_key", None)
        checkpoint_sha256 = service.adapter.info.checkpoint_sha256
        model_selection = self.model_selection
        input_scope = self._input_scope()
        crop_box = self.confirmed_crop_box if input_scope == "bilateral" else None
        roi_selection_method = self.roi_selection_method if crop_box is not None else ""
        self._clear_analysis(keep_raw=True)
        self._set_busy(True, f"{raw_path.name}を解析しています（{side}側）…")
        self.result_state_var.set("AI解析中")

        def worker() -> None:
            try:
                if crop_box is None:
                    analysis = service.analyze_path(raw_path, requested_side=explicit_side)
                else:
                    analysis = service.analyze_path(
                        raw_path,
                        requested_side=explicit_side,
                        crop_box=crop_box,
                        roi_selection_method=roi_selection_method,
                        roi_confirmed=True,
                    )
                value = replace(
                    analysis,
                    model_selection=model_selection,
                )
                self.task_events.put(
                    {
                        "id": task_id,
                        "kind": "inference",
                        "ok": True,
                        "value": value,
                        "raw_path": raw_path,
                        "model_key": model_key,
                        "checkpoint_sha256": checkpoint_sha256,
                        "input_scope": input_scope,
                        "crop_box": crop_box,
                    }
                )
            except Exception as exc:
                self.task_events.put(
                    {
                        "id": task_id,
                        "kind": "inference",
                        "ok": False,
                        "error": str(exc),
                        "raw_path": raw_path,
                        "model_key": model_key,
                        "checkpoint_sha256": checkpoint_sha256,
                        "input_scope": input_scope,
                        "crop_box": crop_box,
                    }
                )

        threading.Thread(target=worker, name="knee-inference", daemon=True).start()

    def _poll_task_events(self) -> None:
        if self.closing:
            return
        try:
            while True:
                event = self.task_events.get_nowait()
                if event["id"] != self.task_id:
                    continue
                if event["kind"] == "model":
                    self._finish_model_event(event)
                elif event["kind"] == "inference":
                    self._finish_inference_event(event)
        except queue.Empty:
            pass
        self.root.after(20 if self.busy else 80, self._poll_task_events)

    def _finish_model_event(self, event: dict[str, Any]) -> None:
        if not event["ok"]:
            error = event["error"]
            self.pending_model_key = None
            if (
                self.service is None
                and event.get("startup")
                and self.default_model_spec is not None
                and event.get("model_key") == "external"
            ):
                self.preference_warning = (
                    "前回選択した外部AIモデルを読み込めなかったため、標準モデルに戻しました："
                    f"{error}"
                )
                try:
                    clear_model_preference()
                except Exception as exc:
                    self.preference_warning += f"（設定の消去にも失敗しました：{exc}）"
                self.status_var.set(self.preference_warning)
                fallback = self._resolve_model_selection("auto")
                self._activate_model_selection(fallback, startup=False, run_inference=False)
                return
            if self.service is None:
                self._set_model_unavailable(error)
            else:
                self._update_model_badge()
                self._update_model_details()
                self._update_model_source_display()
                self.status_var.set("新しいAIモデルを読み込めませんでした。以前の解析結果は消去しました。")
                messagebox.showerror(
                    "AIモデルの読み込みに失敗しました",
                    f"新しいAIモデルは適用されていません。"
                    "以前の結果は消去しました。使用するモデルを再度選択してください。"
                    f"\n\n{error}",
                )
            self._set_busy(False)
            return

        payload = event["value"]
        spec = payload["spec"]
        service = payload["service"]
        model_key = payload.get("model_key") or getattr(self, "pending_model_key", None) or "external"
        selection = payload.get("selection")
        if selection is not None:
            self._set_model_selection(selection)
        self.pending_model_key = None
        if isinstance(spec, ModelSpec) and model_key != "external":
            cache_key = self._model_cache_key(model_key, spec)
            if not hasattr(self, "model_cache"):
                self.model_cache = {}
            self.model_cache[cache_key] = (spec, service)
        self.service = service
        self.model_spec = spec
        self.active_model_key = model_key
        # A successful model switch invalidates every result produced by the
        # previous service.  Clear it before attempting the rerun so that an
        # unresolved side cannot leave old coordinates under the new badge.
        if self.raw_path is not None:
            self._clear_analysis(keep_raw=True)
        preference_action = payload.get("preference_action")
        if preference_action is not None:
            self._persist_model_choice(
                preference_action,
                spec,
                payload["info"].checkpoint_sha256,
            )
        self._update_model_badge()
        self._update_model_details()
        self._update_model_source_display()
        ready_status = "AIモデルの準備が完了しました。下肢全長X線画像を開いてください。"
        if self.raw_path is not None and not payload.get("run_inference", True):
            if self._input_scope() == "bilateral":
                ready_status = "AIモデルの準備が完了しました。対象脚のROIを確認してください。"
            else:
                ready_status = "AIモデルの準備が完了しました。LまたはRを選択してください。"
        if self.preference_warning:
            ready_status = self.preference_warning
            self.warning_title_var.set("AIモデル設定の警告")
            self._set_warning_text([self.preference_warning])
            self._set_warning_banner("⚠ AIモデル設定の確認事項：1件", "warning", True)
            self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
        elif self.analysis is None and self.raw_path is None:
            self.warning_title_var.set("解析結果はありません")
            self._set_warning_text(["解析後、AIモデルと計測結果の確認事項を表示します。"])
            self._set_warning_banner("確認事項：解析後に表示します", "neutral", False)
            self.notebook.tab(self.quality_tab, text="確認事項（0）・AIモデル")
        self._set_busy(False, ready_status)
        if payload.get("run_inference", True) and self.raw_path is not None:
            self._start_inference()

    def _finish_inference_event(self, event: dict[str, Any]) -> None:
        event_path = event.get("raw_path")
        if event_path is not None and Path(event_path) != self.raw_path:
            return
        event_scope = event.get("input_scope")
        if event_scope is not None and event_scope != self._input_scope():
            return
        if "crop_box" in event:
            expected_crop = self.confirmed_crop_box if self._input_scope() == "bilateral" else None
            if event["crop_box"] != expected_crop:
                return
        event_model_key = event.get("model_key")
        if event_model_key is not None and event_model_key != getattr(self, "active_model_key", None):
            return
        if event["ok"] and event.get("checkpoint_sha256"):
            result_info = event["value"].prediction.model_info
            if result_info.checkpoint_sha256 != event["checkpoint_sha256"]:
                return
        self._set_busy(False)
        if not event["ok"]:
            error = event["error"]
            self._clear_analysis(keep_raw=True)
            self.result_state_var.set("解析エラー")
            self.status_var.set(f"解析に失敗しました：{error}")
            self.warning_title_var.set("今回の解析に失敗しました")
            self._set_warning_text([error, "前回の結果を消去しました。画像、左右、AIモデルを確認して再実行してください。"])
            self._set_warning_banner("⚠ 解析に失敗しました — 詳細を確認してください", "warning", True)
            self.notebook.tab(self.quality_tab, text="解析エラー・AIモデル")
            self._show_quality_tab()
            messagebox.showerror("自動計測に失敗しました", error)
            return

        analysis: AnalysisResult = event["value"]
        if self._input_scope() == "bilateral" and (
            not self.roi_confirmed
            or analysis.inference_roi != self.confirmed_crop_box
            or not analysis.roi_confirmed
        ):
            return
        self.analysis = analysis
        self.raw_path = analysis.raw_path
        self.raw_image = analysis.raw_image
        self.points = clone_points(analysis.prediction.points)
        self.lines = clone_lines(analysis.prediction.lines)
        self.measurement = analysis.measurement
        self.edited_keys.clear()
        self.history.clear()
        self.side_var.set(analysis.side)
        self.side_source_hint = analysis.side_source
        if not self.input_view.has_image or self.displayed_source_sha256 != analysis.source_sha256:
            self.input_view.set_image(self.raw_image, reset_view=True)
            self.displayed_source_sha256 = analysis.source_sha256
        self.input_view.redraw_overlay()
        self._display_measurement(reset_view=True)
        self._refresh_coordinate_table()
        self._update_quality_display(select_details=True)
        self.result_state_var.set("AI推定結果・要確認")
        if self._input_scope() == "bilateral":
            self.roi_status_var.set("確認済みROIで解析完了。切り替えると現在の結果は消去されます。")
        model_label = MODEL_SHORT_LABELS.get(getattr(self, "active_model_key", None), "外部")
        model_id = analysis.prediction.model_info.short_hash
        self.status_var.set(
            f"解析完了：{analysis.side}側・{model_label} {model_id}・{analysis.total_elapsed_ms:.0f} ms。"
            "左側画像のマーカーをドラッグして修正できます。"
        )
        self._refresh_action_states()

    def _update_model_badge(self) -> None:
        if self.service is None:
            self.model_badge_var.set("モデル：未読み込み")
            return
        info = self.service.adapter.info
        model_label = MODEL_SHORT_LABELS.get(getattr(self, "active_model_key", None), "外部")
        self.model_badge_var.set(f"使用モデル：{model_label}・{info.short_hash}・{info.device.upper()}")
        self._update_model_source_display()

    def _update_model_details(self) -> None:
        if self.service is None:
            return
        info = self.service.adapter.info
        metrics = info.val_metrics
        metric_parts: list[str] = []
        if "point_mae_px" in metrics:
            metric_parts.append(f"検証時ランドマークMAE {metrics['point_mae_px']:.1f}px")
        if "mldfa_mae_deg" in metrics:
            metric_parts.append(f"mLDFA MAE {metrics['mldfa_mae_deg']:.1f}°")
        if "mpta_mae_deg" in metrics:
            metric_parts.append(f"MPTA MAE {metrics['mpta_mae_deg']:.1f}°")
        metric_text = f" · {' · '.join(metric_parts)}" if metric_parts else ""
        metadata_label = {
            "legacy_assumed_contract": "旧形式（互換読み込み）",
            "checkpoint_manifest": "埋め込みマニフェスト",
        }.get(info.metadata_source, info.metadata_source)
        selection = getattr(
            self,
            "model_selection",
            ModelSelection("auto", "mixed", "auto_fallback_unknown"),
        )
        requested_label = (
            "自動判定"
            if selection.requested_mode == "auto"
            else MODEL_SHORT_LABELS.get(selection.model_key, selection.model_key)
        )
        selected_label = MODEL_SHORT_LABELS.get(selection.model_key, selection.model_key)
        source_label = MODEL_SOURCE_LABELS.get(selection.source, selection.source)
        self.model_detail_var.set(
            f"{info.display_name}・{info.version}・入力 {info.input_width}×{info.input_height}\n"
            f"選択：{requested_label} → {selected_label}（{source_label}）\n"
            f"実行環境：{info.device.upper()}・weight SHA-256：{info.checkpoint_sha256}\n"
            f"対象：{info.cohort}・メタデータ：{metadata_label}{metric_text}"
        )

    def open_image(self) -> None:
        if self.busy:
            return
        filename = filedialog.askopenfilename(
            title="下肢全長X線画像を選択",
            filetypes=[
                ("X線画像", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path: Path, requested_side: str | None = None) -> None:
        path = Path(path).expanduser().resolve()
        self.task_id = getattr(self, "task_id", 0) + 1
        self._clear_analysis(keep_raw=False)
        self.raw_path = path
        self.path_var.set(path.name)
        explicit = normalize_measurement_side(requested_side)
        inferred = infer_knee_side_from_sources(path)
        self.side_var.set(explicit or inferred or "自動判定")
        self.side_source_hint = "explicit" if explicit else ("filename" if inferred else "unknown")
        side_is_resolved = explicit is not None or inferred is not None
        if not side_is_resolved and not self._ensure_input_preview():
            return

        if self._input_scope() == "bilateral":
            self.roi_confirmed = False
            self.confirmed_crop_box = None
            self.roi_selection_method = ""
            self._suggest_screen_side()
            self._update_roi_panel_visibility()
            if getattr(self, "model_specs", None):
                selection = self._resolve_model_selection("auto")
                self._activate_model_selection(selection, run_inference=False)
            self.result_state_var.set("ROI確認待ち")
            if side_is_resolved:
                self.roi_status_var.set("画面上の対象脚と分割線を確認してください。")
                self.status_var.set("両側画像を読み込みました。ROIを確認するまでAI解析は行いません。")
                self.warning_title_var.set("両側画像のROIを確認してください")
                self._set_warning_text(["反対側の脚への誤推定を防ぐため、画面上の対象脚を医師が確認する必要があります。"])
                self._set_warning_banner("⚠ 両側画像のROI確認が必要です", "warning", True)
                self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
            else:
                self._show_side_required_warning()
            self._refresh_action_states()
            return

        if getattr(self, "model_specs", None):
            selection = self._resolve_model_selection("auto")
            self._activate_model_selection(selection, run_inference=side_is_resolved)
        elif side_is_resolved and self.service is not None:
            # Backward-compatible path for programmatic callers using the
            # legacy single-model configuration.
            self._start_inference()

        if explicit is None and inferred is None:
            self.result_state_var.set("左右の選択待ち")
            self.status_var.set("画像を選択しました。ファイル名から左右を判定できないため、LまたはRを選択してください。")
            self.warning_title_var.set("左右を選択してください")
            self._set_warning_text(["左右はAIモデルの出力ではありません。元の検査情報に基づいてLまたはRを選択してください。"])
            self._set_warning_banner("⚠ 左右を選択してください", "warning", True)
            self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
            return
        if getattr(self, "model_specs", None):
            return
        if self.service is None:
            self.status_var.set("画像を選択しました。AIモデルの準備完了後に自動で解析します。")

    def rerun_inference(self) -> None:
        if not self.busy and self.raw_path is not None and self._model_selection_is_active():
            self._start_inference()

    def _on_model_mode_changed(self, _event: object | None = None) -> None:
        if self.busy:
            return
        mode = MODEL_LABEL_TO_MODE.get(self.model_mode_var.get())
        if mode is None:
            return
        try:
            selection = self._resolve_model_selection(mode)
        except Exception as exc:
            messagebox.showerror("AIモデルを選択できません", str(exc))
            return
        self._activate_model_selection(selection, run_inference=self.raw_path is not None)

    def browse_weight(self) -> None:
        if self.busy or self.default_model_spec is None:
            return
        filename = filedialog.askopenfilename(
            title="互換性のあるAIモデルファイルを選択",
            filetypes=[("PyTorchモデルファイル", "*.pt *.pth"), ("すべてのファイル", "*.*")],
        )
        if not filename:
            return
        path = Path(filename).expanduser().resolve()
        spec = model_spec_with_checkpoint(self.default_model_spec, path)
        spec = replace(
            spec,
            display_name=f"外部モデル：{path.stem}",
            version="auto",
            cohort="対象データ未指定（外部モデル）",
            options={**spec.options, "allow_legacy_checkpoint": False},
        )
        selection = ModelSelection("external", "external", "external_model")
        self._start_model_load(
            spec,
            startup=False,
            preference_action="save",
            model_key="external",
            selection=selection,
        )

    def restore_builtin_model(self) -> None:
        if self.busy or not self.model_specs:
            return
        selection = self._resolve_model_selection("auto")
        self._activate_model_selection(
            selection,
            preference_action="clear",
            run_inference=self.raw_path is not None,
        )

    def _effective_side(self) -> str:
        if self.raw_path is None:
            raise SideRequiredError("画像が読み込まれていません。")
        explicit = normalize_measurement_side(self.side_var.get())
        return resolve_side(self.raw_path, explicit)

    def _on_side_changed(self, _event: object | None = None) -> None:
        if self.busy or self.raw_path is None:
            return
        explicit_side = normalize_measurement_side(self.side_var.get())
        self.side_source_hint = "manual_override" if explicit_side else "unknown"
        self.task_id = getattr(self, "task_id", 0) + 1
        self._clear_analysis(keep_raw=True)
        try:
            side = self._effective_side()
        except SideRequiredError:
            self.roi_confirmed = False
            self.confirmed_crop_box = None
            self._show_side_required_warning()
            return
        side_source = "manual_override" if explicit_side else "filename"
        self.side_source_hint = side_source
        if self._input_scope() == "bilateral":
            self.roi_confirmed = False
            self.confirmed_crop_box = None
            self.roi_selection_method = ""
            self.screen_side_var.set(SCREEN_SIDE_LABELS[self._conventional_screen_side(side)])
            self.result_state_var.set("ROI確認待ち")
            self.roi_status_var.set("左右を変更したため対象脚候補を更新しました。ROIを再確認してください。")
            self.status_var.set(f"{side}側に変更しました。以前の結果は消去済みです。ROI確認後に再解析します。")
            self.input_view.redraw_overlay()
            self._refresh_action_states()
            return
        if self.service is not None and self._model_selection_is_active():
            self._start_inference()

    def _clear_analysis(self, keep_raw: bool) -> None:
        self.analysis = None
        self.points = {}
        self.lines = {}
        self.measurement = None
        self.edited_keys.clear()
        self.history.clear()
        self.drag_target = None
        self.coordinate_tree.delete(*self.coordinate_tree.get_children())
        self._clear_measurement_display("解析完了後に結果を表示します")
        self.result_state_var.set("入力待ち")
        self.warning_title_var.set("解析結果はありません")
        self._set_warning_text(["解析後、AIモデルと計測結果の確認事項を表示します。"])
        self._set_warning_banner("確認事項：解析後に表示します", "neutral", False)
        self.notebook.tab(self.quality_tab, text="確認事項（0）・AIモデル")
        self.notebook.select(0)
        if not keep_raw:
            self.raw_path = None
            self.raw_image = None
            self.displayed_source_sha256 = None
            self.side_source_hint = "unknown"
            if hasattr(self, "input_scope_var"):
                self.input_scope_var.set(INPUT_SCOPE_LABELS["single"])
            self.side_var.set("自動判定")
            self.roi_confirmed = False
            self.confirmed_crop_box = None
            self.roi_selection_method = ""
            if hasattr(self, "screen_side_var"):
                self.screen_side_var.set(SCREEN_SIDE_UNSELECTED)
            if hasattr(self, "roi_split_var"):
                self.roi_split_var.set(50.0)
                self.roi_split_label_var.set("分割位置：50%（中央8%重複）")
            self._update_roi_panel_visibility()
            self.input_view.clear()
        self.input_view.redraw_overlay()
        self._refresh_action_states()

    def _clear_measurement_display(self, message: str) -> None:
        self.measurement = None
        self.result_view.clear(message)
        for variable in self.angle_vars.values():
            variable.set("—")
        self._update_angle_alerts()
        self.export_button.configure(state="disabled")

    def _display_measurement(self, reset_view: bool) -> None:
        if self.measurement is None:
            return
        self.result_view.set_image(self.measurement["combined_image"], reset_view=reset_view)
        self.angle_vars["mLDFA"].set(f"{float(self.measurement['mldfa_angle']):.2f}°")
        self.angle_vars["MPTA"].set(f"{float(self.measurement['mpta_angle']):.2f}°")
        self.angle_vars["JLCA"].set(f"{float(self.measurement['jlca_angle']):+.2f}°")
        self.angle_vars["HKA"].set(f"{float(self.measurement['hka_angle']):+.2f}°")
        self._refresh_action_states()

    def _update_angle_alerts(self) -> None:
        out_of_range = (
            set(measurement_out_of_range_angles(self.measurement))
            if self.measurement is not None
            else set()
        )
        for key, card in self.angle_cards.items():
            is_alert = key in out_of_range
            self.angle_name_labels[key].configure(
                text=f"{key} ⚠" if is_alert else key,
                fg=COLORS["danger"] if is_alert else COLORS["muted"],
            )
            self.angle_value_labels[key].configure(
                fg=COLORS["danger"] if is_alert else COLORS["text"],
            )
            card.configure(
                highlightbackground=COLORS["danger"] if is_alert else COLORS["border"],
                highlightthickness=2 if is_alert else 1,
            )

    def _prediction_coordinate(self, key: str) -> np.ndarray | None:
        if key in self.points:
            return self.points[key]
        match = re.fullmatch(r"(upper_line|lower_line)_(p1|p2)", key)
        if match:
            return self.lines.get(match.group(1), {}).get(match.group(2))
        return None

    def _draw_inference_roi_overlay(self, view: InteractiveImageCanvas) -> None:
        if self._input_scope() != "bilateral" or self.raw_image is None:
            return
        height, width = self.raw_image.shape[:2]
        split_x = int(round(width * float(self.roi_split_var.get()) / 100.0))
        split_x = min(max(split_x, 1), width - 1)
        divider_top = view.image_to_canvas(np.array([split_x, 0], dtype=np.float32))
        divider_bottom = view.image_to_canvas(np.array([split_x, height], dtype=np.float32))
        tag = view.overlay_tag
        view.canvas.create_line(
            divider_top[0],
            divider_top[1],
            divider_bottom[0],
            divider_bottom[1],
            fill="#facc15",
            width=3,
            dash=(7, 4),
            tags=(tag,),
        )
        crop_box = self.confirmed_crop_box if self.roi_confirmed else self._candidate_crop_box()
        if crop_box is None:
            return
        x0, y0, x1, y1 = crop_box
        selected_top_left = view.image_to_canvas(np.array([x0, y0], dtype=np.float32))
        selected_bottom_right = view.image_to_canvas(np.array([x1, y1], dtype=np.float32))
        if x0 == 0:
            excluded = (x1, 0, width, height)
        else:
            excluded = (0, 0, x0, height)
        excluded_top_left = view.image_to_canvas(np.array(excluded[:2], dtype=np.float32))
        excluded_bottom_right = view.image_to_canvas(np.array(excluded[2:], dtype=np.float32))
        view.canvas.create_rectangle(
            excluded_top_left[0],
            excluded_top_left[1],
            excluded_bottom_right[0],
            excluded_bottom_right[1],
            fill="#111827",
            stipple="gray50",
            outline="",
            tags=(tag,),
        )
        color = COLORS["success"] if self.roi_confirmed else "#facc15"
        view.canvas.create_rectangle(
            selected_top_left[0],
            selected_top_left[1],
            selected_bottom_right[0],
            selected_bottom_right[1],
            outline=color,
            width=4,
            tags=(tag,),
        )
        view.canvas.create_text(
            selected_top_left[0] + 8,
            selected_top_left[1] + 8,
            text="推論対象ROI（確認済み）" if self.roi_confirmed else "推論対象ROI（要確認）",
            fill=color,
            font=("TkDefaultFont", 10, "bold"),
            anchor="nw",
            tags=(tag,),
        )

    def _draw_prediction_overlay(self, view: InteractiveImageCanvas) -> None:
        self._draw_inference_roi_overlay(view)
        if not self.points:
            return
        tag = view.overlay_tag

        for line_index, line_name in enumerate(ANNOTATION_LINE_NAMES):
            endpoints = self.lines.get(line_name, {})
            p1 = endpoints.get("p1")
            p2 = endpoints.get("p2")
            if p1 is None or p2 is None:
                continue
            key1 = f"{line_name}_p1"
            key2 = f"{line_name}_p2"
            line_color = COLORS["edited"] if key1 in self.edited_keys or key2 in self.edited_keys else COLORS["line"]
            x1, y1 = view.image_to_canvas(p1)
            x2, y2 = view.image_to_canvas(p2)
            view.canvas.create_line(x1, y1, x2, y2, fill=line_color, width=2, tags=(tag,))
            for endpoint_index, (endpoint, point) in enumerate((("p1", p1), ("p2", p2)), start=1):
                key = f"{line_name}_{endpoint}"
                color = COLORS["selected"] if self._drag_key() == key else (COLORS["edited"] if key in self.edited_keys else COLORS["line"])
                x, y = view.image_to_canvas(point)
                self._canvas_handle(view, x, y, color, 8)
                label = f"{'U' if line_index == 0 else 'L'}{endpoint_index}"
                view.canvas.create_text(
                    x + 10,
                    y - 9,
                    text=label,
                    fill=color,
                    font=("TkDefaultFont", 9, "bold"),
                    anchor="sw",
                    tags=(tag,),
                )

        for index, name in enumerate(ANNOTATION_POINT_NAMES, start=1):
            point = self.points.get(name)
            if point is None:
                continue
            color = COLORS["selected"] if self._drag_key() == name else (COLORS["edited"] if name in self.edited_keys else COLORS["point"])
            x, y = view.image_to_canvas(point)
            self._canvas_handle(view, x, y, color, 6)
            view.canvas.create_text(
                x + 9,
                y - 8,
                text=str(index),
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
                anchor="sw",
                tags=(tag,),
            )

    def _canvas_handle(self, view: InteractiveImageCanvas, x: float, y: float, color: str, radius: int) -> None:
        tag = view.overlay_tag
        view.canvas.create_oval(
            x - radius - 2,
            y - radius - 2,
            x + radius + 2,
            y + radius + 2,
            fill="#ffffff",
            outline="",
            tags=(tag,),
        )
        view.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="#102033",
            width=1,
            tags=(tag,),
        )

    def _drag_key(self) -> str | None:
        if self.drag_target is None:
            return None
        if self.drag_target[0] == "point":
            return self.drag_target[1]
        return f"{self.drag_target[1]}_{self.drag_target[2]}"

    def _hit_test(self, x: float, y: float) -> tuple[str, ...] | None:
        target = np.array([x, y], dtype=np.float32)
        radius = 20.0 / max(self.input_view.zoom, 1e-6)
        hits: list[tuple[float, int, tuple[str, ...]]] = []
        priority = 0
        for line_name, endpoints in self.lines.items():
            for endpoint, point in endpoints.items():
                distance = float(np.linalg.norm(point - target))
                if distance <= radius:
                    hits.append((distance, -priority, ("line", line_name, endpoint)))
                priority += 1
        for name, point in self.points.items():
            distance = float(np.linalg.norm(point - target))
            if distance <= radius:
                hits.append((distance, -priority, ("point", name)))
            priority += 1
        if not hits:
            return None
        return min(hits, key=lambda item: (round(item[0], 4), item[1]))[2]

    def _snapshot(self) -> dict[str, Any]:
        return {
            "points": clone_points(self.points),
            "lines": clone_lines(self.lines),
            "edited_keys": set(self.edited_keys),
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.points = clone_points(snapshot["points"])
        self.lines = clone_lines(snapshot["lines"])
        self.edited_keys = set(snapshot["edited_keys"])

    def _snapshot_coordinate(self, snapshot: dict[str, Any], key: str) -> np.ndarray:
        if key in snapshot["points"]:
            return snapshot["points"][key]
        match = re.fullmatch(r"(upper_line|lower_line)_(p1|p2)", key)
        if match:
            return snapshot["lines"][match.group(1)][match.group(2)]
        raise KeyError(key)

    def _on_canvas_press(self, x: float, y: float) -> None:
        if self.busy:
            return
        if (
            self._input_scope() == "bilateral"
            and self.raw_image is not None
            and not self.roi_confirmed
            and self.analysis is None
        ):
            width = self.raw_image.shape[1]
            split_x = width * float(self.roi_split_var.get()) / 100.0
            self.screen_side_var.set(SCREEN_SIDE_LABELS["left" if x < split_x else "right"])
            self._invalidate_roi("画像上で対象脚を選択しました。ROIを確認して解析してください。")
            return
        if self.analysis is None:
            return
        self.drag_target = self._hit_test(x, y)
        self.drag_snapshot = self._snapshot() if self.drag_target is not None else None
        self.drag_changed = False
        self.input_view.redraw_overlay()

    def _move_drag_target(self, x: float, y: float) -> None:
        if self.drag_target is None:
            return
        point = np.array([x, y], dtype=np.float32)
        if self.drag_target[0] == "point":
            self.points[self.drag_target[1]] = point
        else:
            self.lines[self.drag_target[1]][self.drag_target[2]] = point

    def _on_canvas_drag(self, x: float, y: float) -> None:
        if self.drag_target is None:
            return
        self._move_drag_target(x, y)
        self.drag_changed = True
        self.input_view.redraw_overlay()

    def _on_canvas_release(self, x: float, y: float) -> None:
        if self.drag_target is None:
            return
        self._move_drag_target(x, y)
        key = self._drag_key()
        if key is not None and self.drag_snapshot is not None:
            original = self._snapshot_coordinate(self.drag_snapshot, key)
            final = self._prediction_coordinate(key)
            self.drag_changed = final is not None and float(np.linalg.norm(final - original)) > 0.1
        if self.drag_changed and self.drag_snapshot is not None:
            self.history.append(self.drag_snapshot)
            if key:
                self.edited_keys.add(key)
        self.drag_target = None
        self.drag_snapshot = None
        self.input_view.redraw_overlay()
        self._refresh_coordinate_table()
        if self.drag_changed:
            self._recalculate_after_edit()
        self.drag_changed = False
        self._refresh_action_states()

    def _recalculate_after_edit(self) -> None:
        if self.analysis is None or self.raw_image is None or self.raw_path is None:
            return
        try:
            side = self._effective_side()
            self.measurement = measurement_from_coordinates(
                self.raw_image,
                self.raw_path,
                side,
                self.points,
                self.lines,
                render_component_images=False,
            )
            side_source = self.side_source_hint if self.side_source_hint != "unknown" else self.analysis.side_source
            self.analysis = replace(
                self.analysis,
                side=side,
                side_source=side_source,
                measurement=self.measurement,
            )
        except Exception as exc:
            self._clear_measurement_display(f"修正後のランドマーク配置では角度を計算できません\n{exc}")
            self.result_state_var.set("手動修正・計算エラー")
            self.warning_title_var.set("修正後の計算に失敗しました")
            self._set_warning_text([str(exc), "今回の修正を元に戻すか、AI推定位置に戻してください。"])
            self._set_warning_banner("⚠ 修正後の計算に失敗しました", "warning", True)
            self.notebook.tab(self.quality_tab, text="計算エラー・AIモデル")
            self._show_quality_tab()
            self.status_var.set(f"修正後の計算に失敗しました：{exc}")
            return
        self._display_measurement(reset_view=False)
        if self.edited_keys:
            self.result_state_var.set("手動修正あり・要確認")
            self.status_var.set("手動修正を適用し、すべての角度を再計算しました。")
        else:
            self.result_state_var.set("AI推定結果・要確認")
            self.status_var.set("AI推定位置に戻し、すべての角度を再計算しました。")
        self._update_quality_display()

    def undo_edit(self) -> None:
        if self.busy or not self.history or not self._model_selection_is_active():
            return
        self._restore_snapshot(self.history.pop())
        self.input_view.redraw_overlay()
        self._refresh_coordinate_table()
        self._recalculate_after_edit()
        self._refresh_action_states()

    def reset_to_prediction(self) -> None:
        if self.busy or self.analysis is None or not self._model_selection_is_active():
            return
        self.history.append(self._snapshot())
        self.points = clone_points(self.analysis.prediction.points)
        self.lines = clone_lines(self.analysis.prediction.lines)
        self.edited_keys.clear()
        self.input_view.redraw_overlay()
        self._refresh_coordinate_table()
        self._recalculate_after_edit()
        self._refresh_action_states()

    def _refresh_coordinate_table(self) -> None:
        self.coordinate_tree.delete(*self.coordinate_tree.get_children())
        if self.analysis is None:
            return
        scores = self.analysis.prediction.peak_scores
        for key, label in TABLE_ROWS:
            point = self._prediction_coordinate(key)
            if point is None:
                continue
            source = "手動" if key in self.edited_keys else "AI"
            score = scores.get(key)
            self.coordinate_tree.insert(
                "",
                "end",
                iid=key,
                values=(label, f"{float(point[0]):.1f}", f"{float(point[1]):.1f}", f"{score:.3f}" if score is not None else "—", source),
            )

    def _update_quality_display(self, select_details: bool = False) -> None:
        if self.analysis is None:
            return
        self._update_angle_alerts()
        case_warnings = combine_warnings(
            coordinate_geometry_warnings(self.points, self.lines, self.analysis.raw_image.shape),
            measurement_warnings(self.measurement) if self.measurement is not None else (),
            self._roi_coordinate_warnings(),
        )
        warnings = list(combine_warnings(self.analysis.model_warnings, case_warnings))
        if self.edited_keys:
            warnings.insert(
                0,
                f"{len(self.edited_keys)}個のランドマークを手動修正済みです。"
                "AIスコアは修正前の推定位置に対する値です。",
            )
        if self.preference_warning:
            warnings.insert(0, self.preference_warning)
        if warnings:
            self.warning_title_var.set(f"確認事項：{len(warnings)}件")
            self._set_warning_text(warnings)
            self._set_warning_banner(
                f"⚠ 確認事項：{len(warnings)}件 — 必ず詳細を確認してください",
                level="warning",
                detail_available=True,
            )
            self.notebook.tab(self.quality_tab, text=f"確認事項（{len(warnings)}）・AIモデル")
            if select_details and case_warnings:
                self._show_quality_tab()
        else:
            self.warning_title_var.set("自動チェックで異常は検出されませんでした")
            self._set_warning_text(
                ["警告がない場合も、すべてのランドマークと角度を医師が確認してください。"
                 "AIスコアは臨床的な確信度ではありません。"]
            )
            self._set_warning_banner(
                "自動チェック：警告0件 — すべてのランドマークと角度を医師が確認してください",
                level="ok",
                detail_available=False,
            )
            self.notebook.tab(self.quality_tab, text="確認事項（0）・AIモデル")
        self._update_model_details()

    def _set_warning_banner(
        self,
        text: str,
        level: str = "neutral",
        detail_available: bool = False,
    ) -> None:
        palette = {
            "neutral": ("#f1f5f9", COLORS["border"], COLORS["muted"]),
            "ok": ("#ecfdf5", "#86efac", COLORS["success"]),
            "warning": ("#fff7ed", "#f59e0b", "#9a3412"),
        }
        background, border, foreground = palette[level]
        self.warning_banner_var.set(text)
        self.warning_banner.configure(bg=background, highlightbackground=border)
        self.warning_banner_label.configure(bg=background, fg=foreground)
        self.warning_detail_button.configure(state="normal" if detail_available else "disabled")

    def _set_warning_text(self, messages: list[str]) -> None:
        self.warning_text.configure(state="normal")
        self.warning_text.delete("1.0", tk.END)
        self.warning_text.insert("1.0", "\n".join(f"• {message}" for message in messages))
        self.warning_text.configure(state="disabled")

    def export_result(self) -> None:
        if (
            self.analysis is None
            or self.measurement is None
            or self.busy
            or not self._model_selection_is_active()
            or not self._analysis_matches_current_input()
            or not self._coordinates_within_confirmed_roi()
        ):
            return
        initial_dir = self.raw_path.parent if self.raw_path is not None else Path.home()
        directory = filedialog.askdirectory(title="結果の保存先を選択", initialdir=str(initial_dir))
        if not directory:
            return
        output_dir = Path(directory)
        stem = measurement_export_stem(
            self.analysis.raw_path.stem,
            input_scope=self.analysis.input_scope,
            side=self.analysis.side,
        )
        json_path = output_dir / f"{stem}_measurement.json"
        overlay_path = output_dir / f"{stem}_measurement.png"
        if (json_path.exists() or overlay_path.exists()) and not messagebox.askyesno(
            "既存の結果を上書き",
            f"次のファイルは既に存在します。上書きしますか？\n\n{json_path.name}\n{overlay_path.name}",
        ):
            return
        try:
            payload = export_record(
                self.analysis,
                self.points,
                self.lines,
                self.measurement,
                APP_VERSION,
                manually_modified=bool(self.edited_keys),
                edited_keys=self.edited_keys,
                app_release_channel=APP_RELEASE_CHANNEL,
            )
            write_result_bundle(json_path, payload, overlay_path, self.measurement["combined_image"])
        except Exception as exc:
            messagebox.showerror("書き出しに失敗しました", str(exc))
            return
        self.status_var.set(f"{json_path.name} と {overlay_path.name} を書き出しました。")
        messagebox.showinfo("書き出し完了", f"保存しました：\n\n{json_path}\n{overlay_path}")

    def close(self) -> None:
        self.closing = True
        self.task_id += 1
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knee X-ray automated measurement desktop application.")
    parser.add_argument("--config", type=Path, default=None, help="Optional app/model configuration JSON.")
    parser.add_argument("--image", type=Path, default=None, help="Optional image to open on launch.")
    parser.add_argument("--side", choices=("L", "R", "Auto"), default="Auto")
    parser.add_argument("--validate-model", action="store_true", help="Validate the configured model and exit.")
    parser.add_argument(
        "--validate-models",
        action="store_true",
        help="Validate every configured built-in model and exit.",
    )
    parser.add_argument(
        "--expected-model-version",
        action="append",
        default=[],
        type=model_version_expectation,
        metavar="KEY=VERSION",
        help="Require an exact checkpoint version during --validate-models (repeat for all models).",
    )
    parser.add_argument(
        "--model-mode",
        choices=("auto", "bone", "tka", "mixed"),
        default="auto",
        help="Model route used by the headless smoke test.",
    )
    parser.add_argument(
        "--smoke-test-image",
        type=Path,
        default=None,
        help="Run a headless CPU image-to-export smoke test and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_model_version and not args.validate_models:
        raise SystemExit("--expected-model-version requires --validate-models")
    if args.validate_model or args.validate_models or args.smoke_test_image is not None:
        try:
            config = load_app_config(args.config)
            services: dict[str, KneeAnalysisService] = {}
            infos: dict[str, Any] = {}

            def load_model(model_key: str) -> tuple[KneeAnalysisService, Any]:
                if model_key in services:
                    return services[model_key], infos[model_key]
                spec = config.models[model_key]
                adapter = create_model_adapter(spec)
                info = adapter.load()
                if info.device != "cpu":
                    raise RuntimeError(
                        f"Release validation requires CPU execution; model {model_key} selected {info.device}."
                    )
                service = KneeAnalysisService(
                    adapter,
                    low_peak_threshold=float(spec.options.get("low_peak_threshold", 0.35)),
                    render_component_images=False,
                )
                services[model_key] = service
                infos[model_key] = info
                return service, info

            if args.validate_models:
                for key in config.models:
                    load_model(key)
                validate_expected_model_versions(
                    config,
                    infos,
                    args.expected_model_version,
                )
            elif args.validate_model:
                load_model(config.default_model_key)

            if args.smoke_test_image is not None:
                selection = resolve_model_selection(
                    args.model_mode,
                    config.models,
                    args.smoke_test_image,
                    fallback_model_key=config.auto_fallback_model_key,
                )
                service, info = load_model(selection.model_key)
                requested_side = normalize_measurement_side(args.side)
                analysis = replace(
                    service.analyze_path(args.smoke_test_image, requested_side=requested_side),
                    model_selection=selection,
                )
                if len(analysis.prediction.points) != 8:
                    raise RuntimeError("Smoke test did not produce eight landmarks.")
                line_endpoint_count = sum(len(endpoints) for endpoints in analysis.prediction.lines.values())
                if line_endpoint_count != 4:
                    raise RuntimeError("Smoke test did not produce four joint-line endpoints.")
                with tempfile.TemporaryDirectory(prefix="knee-xray-smoke-") as directory:
                    output_dir = Path(directory)
                    json_path = output_dir / "smoke_measurement.json"
                    overlay_path = output_dir / "smoke_measurement.png"
                    payload = export_record(
                        analysis,
                        analysis.prediction.points,
                        analysis.prediction.lines,
                        analysis.measurement,
                        APP_VERSION,
                        manually_modified=False,
                        app_release_channel=APP_RELEASE_CHANNEL,
                    )
                    write_result_bundle(
                        json_path,
                        payload,
                        overlay_path,
                        analysis.measurement["combined_image"],
                    )
                    summary = {
                        "status": "ok",
                        "device": info.device,
                        "requested_model_mode": selection.requested_mode,
                        "resolved_model_key": selection.model_key,
                        "model_selection_source": selection.source,
                        "model_sha256": info.checkpoint_sha256,
                        "side": analysis.side,
                        "point_count": len(analysis.prediction.points),
                        "line_endpoint_count": line_endpoint_count,
                        "angles_deg": payload["angles_deg"],
                        "warning_count": len(payload["analysis"]["warnings"]),
                        "json_bytes": json_path.stat().st_size,
                        "overlay_bytes": overlay_path.stat().st_size,
                    }
                print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
                return
        except Exception as exc:
            escaped_error = str(exc).encode("unicode_escape", errors="backslashreplace").decode("ascii")
            print(f"Model validation failed: {escaped_error}")
            raise SystemExit(2) from exc
        if args.validate_models:
            validation_payload: dict[str, Any] = {
                "status": "ok",
                "models": {key: asdict(info) for key, info in infos.items()},
            }
        else:
            validation_payload = asdict(infos[config.default_model_key])
        print(json.dumps(validation_payload, ensure_ascii=True, indent=2, sort_keys=True))
        return
    root = tk.Tk()
    app = KneeMeasurementApp(root, config_path=args.config)
    app.side_var.set(args.side if args.side in {"L", "R"} else "自動判定")
    if args.image is not None:
        root.after(120, lambda: app.open_path(args.image, requested_side=args.side))
    root.mainloop()


if __name__ == "__main__":
    main()

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
    resolve_side,
    save_model_preference,
    write_result_bundle,
)
from measure_angles import (
    ANNOTATION_LINE_NAMES,
    ANNOTATION_POINT_NAMES,
    infer_knee_side_from_sources,
    normalize_measurement_side,
)


APP_VERSION = "0.2.4"
APP_TITLE = "下肢全長X線 自動計測"

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


class KneeMeasurementApp:
    def __init__(self, root: tk.Tk, config_path: Path | None = None) -> None:
        self.root = root
        self.root.title(f"{APP_TITLE} · v{APP_VERSION}")
        self.root.geometry("1500x930")
        self.root.minsize(1080, 720)
        self.root.configure(bg=COLORS["page"])

        self.config: AppConfig | None = None
        self.default_model_spec: ModelSpec | None = None
        self.model_spec: ModelSpec | None = None
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
        self.history: list[dict[str, Any]] = []
        self.drag_target: tuple[str, ...] | None = None
        self.drag_snapshot: dict[str, Any] | None = None
        self.drag_changed = False

        self.task_events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.task_id = 0
        self.busy = False
        self.closing = False

        self.side_var = tk.StringVar(value="自動判定")
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
            self.model_spec, self.preference_warning = load_model_preference(self.default_model_spec)
        except Exception as exc:
            self._set_model_unavailable(str(exc))
        else:
            self.root.after(80, lambda: self._start_model_load(self.model_spec, startup=True))

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
            text="片側下肢全長X線・ランドマーク自動推定／角度計測",
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
            text="研究用・要医師確認",
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
        toolbar.grid_columnconfigure(6, weight=1)

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

        self.model_button = ttk.Menubutton(toolbar, text="AIモデル…")
        self.model_menu = tk.Menu(self.model_button, tearoff=False)
        self.model_menu.add_command(label="AIモデルファイルを選択…", command=self.browse_weight)
        self.model_menu.add_command(label="標準モデルに戻す", command=self.restore_builtin_model)
        self.model_button.configure(menu=self.model_menu)
        self.model_button.grid(row=0, column=3, rowspan=2, padx=(12, 4), pady=10)
        self.undo_button = ttk.Button(toolbar, text="修正を元に戻す", command=self.undo_edit, state="disabled")
        self.undo_button.grid(row=0, column=4, rowspan=2, padx=4, pady=10)
        self.reset_button = ttk.Button(toolbar, text="AI推定位置に戻す", command=self.reset_to_prediction, state="disabled")
        self.reset_button.grid(row=0, column=5, rowspan=2, padx=4, pady=10)

        path_box = tk.Frame(toolbar, bg=COLORS["card"])
        path_box.grid(row=0, column=6, rowspan=2, sticky="ew", padx=(14, 10), pady=8)
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
            text="JPG / PNG / BMP / TIFF（DICOM・両側未対応）",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            anchor="w",
            font=("TkDefaultFont", 9),
        ).pack(fill="x", pady=(3, 0))

        self.export_button = ttk.Button(toolbar, text="結果を書き出す…", command=self.export_result, state="disabled")
        self.export_button.grid(row=0, column=7, rowspan=2, padx=(4, 12), pady=10)

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
            empty_text="片側下肢全長X線画像を開いてください\nAIが8個の解剖学的ランドマークと2本の関節線を推定します",
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
        model_menu.add_command(label="標準モデルに戻す", command=self.restore_builtin_model)
        menubar.add_cascade(label="AIモデル", menu=model_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="このアプリについて", command=self._show_about)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        self.root.configure(menu=menubar)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "このアプリについて",
            f"{APP_TITLE}  v{APP_VERSION}\n\n"
            "研究用ソフトウェアです。すべてのランドマークと角度を医師が確認してください。\n"
            "GPUは不要です。CPUで動作し、画像や結果を外部へ送信しません。",
        )

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
        else:
            self.progress.stop()
            self.open_button.configure(state="normal")
            self.model_button.configure(state="normal")
            self.side_combo.configure(state="readonly")
            self.rerun_button.configure(state="normal" if self.raw_path and self.service else "disabled")
        self._refresh_action_states()

    def _refresh_action_states(self) -> None:
        has_result = self.analysis is not None and self.measurement is not None and not self.busy
        self.export_button.configure(state="normal" if has_result else "disabled")
        self.reset_button.configure(state="normal" if self.analysis is not None and not self.busy else "disabled")
        self.undo_button.configure(state="normal" if self.history and not self.busy else "disabled")

    def _set_model_unavailable(self, error: str) -> None:
        self.model_badge_var.set("モデル：読み込み不可")
        self.model_detail_var.set(error)
        self.warning_title_var.set("モデル設定エラー")
        self._set_warning_text([error])
        self._set_warning_banner("⚠ AIモデル設定エラー — 詳細を確認してください", "warning", True)
        self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
        self.status_var.set(f"モデルを使用できません：{error}")

    def _start_model_load(
        self,
        spec: ModelSpec,
        startup: bool = False,
        preference_action: str | None = None,
    ) -> None:
        self.task_id += 1
        task_id = self.task_id
        self._set_busy(True, "AIモデルを検証して読み込んでいます…")
        self.model_badge_var.set("モデル：読み込み中…")

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
                }
                self.task_events.put({"id": task_id, "kind": "model", "ok": True, "value": payload})
            except Exception as exc:
                self.task_events.put(
                    {"id": task_id, "kind": "model", "ok": False, "error": str(exc), "startup": startup}
                )

        threading.Thread(target=worker, name="knee-model-loader", daemon=True).start()

    def _start_inference(self) -> None:
        if self.raw_path is None or self.service is None:
            return
        try:
            side = self._effective_side()
        except SideRequiredError:
            self._clear_measurement_display("LまたはRを選択してください。以前の角度は消去されました。")
            self.result_state_var.set("左右の選択待ち")
            self.status_var.set("ファイル名から左右を判定できません。上部でLまたはRを選択してください。")
            self.warning_title_var.set("左右を選択してください")
            self._set_warning_text(["mLDFA、MPTA、JLCA、HKAの解剖学的方向には明確なL/R情報が必要です。"])
            self._set_warning_banner("⚠ 左右を選択してください", "warning", True)
            self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
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
        self._clear_analysis(keep_raw=True)
        self._set_busy(True, f"{raw_path.name}を解析しています（{side}側）…")
        self.result_state_var.set("AI解析中")

        def worker() -> None:
            try:
                value = service.analyze_path(raw_path, requested_side=explicit_side)
                self.task_events.put({"id": task_id, "kind": "inference", "ok": True, "value": value})
            except Exception as exc:
                self.task_events.put({"id": task_id, "kind": "inference", "ok": False, "error": str(exc)})

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
            if (
                self.service is None
                and event.get("startup")
                and self.default_model_spec is not None
                and self.model_spec is not None
                and self.model_spec.checkpoint != self.default_model_spec.checkpoint
            ):
                self.preference_warning = (
                    "前回選択した外部AIモデルを読み込めなかったため、標準モデルに戻しました："
                    f"{error}"
                )
                try:
                    clear_model_preference()
                except Exception as exc:
                    self.preference_warning += f"（設定の消去にも失敗しました：{exc}）"
                self.model_spec = self.default_model_spec
                self.status_var.set(self.preference_warning)
                self._start_model_load(self.default_model_spec, startup=False)
                return
            if self.service is None:
                self._set_model_unavailable(error)
            else:
                self._update_model_badge()
                self.status_var.set("新しいAIモデルを読み込めませんでした。現在のモデルを継続して使用します。")
                messagebox.showerror(
                    "AIモデルの読み込みに失敗しました",
                    f"新しいAIモデルは適用されていません。現在のモデルは引き続き使用できます。\n\n{error}",
                )
            self._set_busy(False)
            return

        payload = event["value"]
        spec = payload["spec"]
        self.service = payload["service"]
        self.model_spec = spec
        # A successful model switch invalidates every result produced by the
        # previous service.  Clear it before attempting the rerun so that an
        # unresolved side cannot leave old coordinates under the new badge.
        if self.raw_path is not None:
            self._clear_analysis(keep_raw=True)
        preference_action = payload.get("preference_action")
        try:
            if preference_action == "save":
                save_model_preference(
                    spec,
                    expected_sha256=payload["info"].checkpoint_sha256,
                )
                self.preference_warning = None
            elif preference_action == "clear":
                clear_model_preference()
                self.preference_warning = None
        except Exception as exc:
            self.preference_warning = f"AIモデルは読み込まれましたが、選択内容を保存できません：{exc}"
        self._update_model_badge()
        self._update_model_details()
        ready_status = "AIモデルの準備が完了しました。片側下肢全長X線画像を開いてください。"
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
        if self.raw_path is not None:
            self._start_inference()

    def _finish_inference_event(self, event: dict[str, Any]) -> None:
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
        self.status_var.set(
            f"解析完了：{analysis.side}側・{analysis.total_elapsed_ms:.0f} ms。左側画像のマーカーをドラッグして修正できます。"
        )
        self._refresh_action_states()

    def _update_model_badge(self) -> None:
        if self.service is None:
            self.model_badge_var.set("モデル：未読み込み")
            return
        info = self.service.adapter.info
        version_label = info.version if len(info.version) <= 24 else f"{info.version[:21]}…"
        self.model_badge_var.set(f"モデル準備完了・{version_label}・{info.device.upper()}")

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
        self.model_detail_var.set(
            f"{info.display_name}・{info.version}・入力 {info.input_width}×{info.input_height}\n"
            f"実行環境：{info.device.upper()}・モデルID：{info.short_hash}\n"
            f"対象：{info.cohort}・メタデータ：{metadata_label}{metric_text}"
        )

    def open_image(self) -> None:
        if self.busy:
            return
        filename = filedialog.askopenfilename(
            title="片側下肢全長X線画像を選択",
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
        self._clear_analysis(keep_raw=False)
        self.path_var.set(path.name)
        self.raw_path = path
        self.path_var.set(path.name)
        explicit = normalize_measurement_side(requested_side)
        inferred = infer_knee_side_from_sources(path)
        self.side_var.set(explicit or inferred or "自動判定")
        self.side_source_hint = "explicit" if explicit else ("filename" if inferred else "unknown")
        if explicit is None and inferred is None:
            self.result_state_var.set("左右の選択待ち")
            self.status_var.set("画像を選択しました。ファイル名から左右を判定できないため、LまたはRを選択してください。")
            self.warning_title_var.set("左右を選択してください")
            self._set_warning_text(["左右はAIモデルの出力ではありません。元の検査情報に基づいてLまたはRを選択してください。"])
            self._set_warning_banner("⚠ 左右を選択してください", "warning", True)
            self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
            return
        if self.service is None:
            self.status_var.set("画像を選択しました。AIモデルの準備完了後に自動で解析します。")
            return
        self._start_inference()

    def rerun_inference(self) -> None:
        if not self.busy and self.raw_path is not None and self.service is not None:
            self._start_inference()

    def browse_weight(self) -> None:
        if self.busy or self.model_spec is None:
            return
        filename = filedialog.askopenfilename(
            title="互換性のあるAIモデルファイルを選択",
            filetypes=[("PyTorchモデルファイル", "*.pt *.pth"), ("すべてのファイル", "*.*")],
        )
        if not filename:
            return
        path = Path(filename).expanduser().resolve()
        spec = model_spec_with_checkpoint(self.model_spec, path)
        spec = replace(
            spec,
            display_name=f"外部モデル：{path.stem}",
            version="auto",
            cohort="対象データ未指定（外部モデル）",
            options={**spec.options, "allow_legacy_checkpoint": False},
        )
        self._start_model_load(spec, startup=False, preference_action="save")

    def restore_builtin_model(self) -> None:
        if self.busy or self.default_model_spec is None:
            return
        self._start_model_load(self.default_model_spec, startup=False, preference_action="clear")

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
        try:
            side = self._effective_side()
        except SideRequiredError:
            self._clear_measurement_display("LまたはRを選択してください。以前の角度は消去されました。")
            self.result_state_var.set("左右の選択待ち")
            self.status_var.set("LまたはRを選択してください。")
            self.warning_title_var.set("左右を選択してください")
            self._set_warning_text(["左右を判定できないため、書き出しを無効にして以前の角度を消去しました。"])
            self._set_warning_banner("⚠ 左右を選択してください", "warning", True)
            self.notebook.tab(self.quality_tab, text="確認事項（1）・AIモデル")
            return
        side_source = "manual_override" if explicit_side else "filename"
        self.side_source_hint = side_source
        if self.analysis is None or not self.points:
            if self.service is not None:
                self._start_inference()
            return
        try:
            self.measurement = measurement_from_coordinates(
                self.raw_image,
                self.raw_path,
                side,
                self.points,
                self.lines,
                render_component_images=False,
            )
            self.analysis = replace(
                self.analysis,
                side=side,
                side_source=side_source,
                measurement=self.measurement,
            )
        except Exception as exc:
            self._clear_measurement_display(f"左右変更後に角度を計算できません：{exc}")
            return
        self._display_measurement(reset_view=False)
        self._update_quality_display()
        self.result_state_var.set("左右変更済み・要確認")
        self.status_var.set(f"{side}側として角度を再計算しました。ランドマーク座標は変更していません。")

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
            self.side_var.set("自動判定")
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

    def _draw_prediction_overlay(self, view: InteractiveImageCanvas) -> None:
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
        if self.analysis is None or self.busy:
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
        if self.busy or not self.history:
            return
        self._restore_snapshot(self.history.pop())
        self.input_view.redraw_overlay()
        self._refresh_coordinate_table()
        self._recalculate_after_edit()
        self._refresh_action_states()

    def reset_to_prediction(self) -> None:
        if self.busy or self.analysis is None:
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
        if self.analysis is None or self.measurement is None or self.busy:
            return
        initial_dir = self.raw_path.parent if self.raw_path is not None else Path.home()
        directory = filedialog.askdirectory(title="結果の保存先を選択", initialdir=str(initial_dir))
        if not directory:
            return
        output_dir = Path(directory)
        stem = safe_export_stem(self.analysis.raw_path.stem)
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
        "--smoke-test-image",
        type=Path,
        default=None,
        help="Run a headless CPU image-to-export smoke test and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_model or args.smoke_test_image is not None:
        try:
            config = load_app_config(args.config)
            adapter = create_model_adapter(config.model)
            info = adapter.load()
            if args.smoke_test_image is not None:
                if info.device != "cpu":
                    raise RuntimeError(f"配布用スモークテストはCPU実行が必要ですが、{info.device}が選択されました。")
                threshold = float(config.model.options.get("low_peak_threshold", 0.35))
                service = KneeAnalysisService(
                    adapter,
                    low_peak_threshold=threshold,
                    render_component_images=False,
                )
                requested_side = normalize_measurement_side(args.side)
                analysis = service.analyze_path(args.smoke_test_image, requested_side=requested_side)
                if len(analysis.prediction.points) != 8:
                    raise RuntimeError("スモークテストで8個のランドマークを取得できませんでした。")
                line_endpoint_count = sum(len(endpoints) for endpoints in analysis.prediction.lines.values())
                if line_endpoint_count != 4:
                    raise RuntimeError("スモークテストで2本の関節線を取得できませんでした。")
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
                        "model_sha256": info.checkpoint_sha256,
                        "side": analysis.side,
                        "point_count": len(analysis.prediction.points),
                        "line_endpoint_count": line_endpoint_count,
                        "angles_deg": payload["angles_deg"],
                        "warning_count": len(payload["analysis"]["warnings"]),
                        "json_bytes": json_path.stat().st_size,
                        "overlay_bytes": overlay_path.stat().st_size,
                    }
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return
        except Exception as exc:
            print(f"モデル検証に失敗しました：{exc}")
            raise SystemExit(2) from exc
        print(json.dumps(asdict(info), ensure_ascii=False, indent=2, sort_keys=True))
        return
    root = tk.Tk()
    app = KneeMeasurementApp(root, config_path=args.config)
    app.side_var.set(args.side if args.side in {"L", "R"} else "自動判定")
    if args.image is not None:
        root.after(120, lambda: app.open_path(args.image, requested_side=args.side))
    root.mainloop()


if __name__ == "__main__":
    main()

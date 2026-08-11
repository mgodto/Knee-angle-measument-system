#!/usr/bin/env python3

"""Model-adapter and analysis service for the knee measurement application."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from knee_xray.core.measure_angles import (
    ANNOTATION_LINE_NAMES,
    ANNOTATION_POINT_NAMES,
    RENDER_STYLE_CLINICAL,
    infer_knee_side_from_sources,
    measure_from_named_points,
    normalize_measurement_side,
    read_color,
)


APP_CONFIG_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE_ID = "small_heatmap_v1"
PREPROCESSING_ID = "grayscale_resize_percentile_1_99_v1"
DEFAULT_CONFIG_FILENAME = "knee_measurement_app.json"
BUILTIN_MODEL_KEYS = ("bone", "tka", "mixed")
EXPECTED_KEYPOINT_NAMES = (
    *ANNOTATION_POINT_NAMES,
    "upper_line_p1",
    "upper_line_p2",
    "lower_line_p1",
    "lower_line_p2",
)
LINE_ENDPOINT_NAMES = {
    "upper_line_p1": ("upper_line", "p1"),
    "upper_line_p2": ("upper_line", "p2"),
    "lower_line_p1": ("lower_line", "p1"),
    "lower_line_p2": ("lower_line", "p2"),
}
COORDINATE_DISPLAY_NAMES = {
    "hip": "点1・股関節中心",
    "upper_left": "点2・大腿骨関節線点A",
    "upper_center": "点3・大腿骨関節線中央点",
    "upper_right": "点4・大腿骨関節線点B",
    "lower_left": "点5・脛骨関節線点A",
    "lower_center": "点6・脛骨関節線中央点",
    "lower_right": "点7・脛骨関節線点B",
    "ankle": "点8・足関節中心",
    "upper_line_p1": "大腿骨関節線端点1",
    "upper_line_p2": "大腿骨関節線端点2",
    "lower_line_p1": "脛骨関節線端点1",
    "lower_line_p2": "脛骨関節線端点2",
}
LINE_DISPLAY_NAMES = {
    "upper_line": "大腿骨関節線",
    "lower_line": "脛骨関節線",
}


def coordinate_display_name(name: str) -> str:
    return COORDINATE_DISPLAY_NAMES.get(name, name)


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint cannot be safely activated."""


class InferenceError(RuntimeError):
    """Raised when a loaded adapter cannot produce a valid prediction."""


class SideRequiredError(ValueError):
    """Raised when anatomical laterality cannot be resolved."""


@dataclass(frozen=True)
class ModelSpec:
    adapter: str
    checkpoint: Path
    display_name: str = "膝関節ランドマーク推定モデル"
    version: str = "unversioned"
    cohort: str = "片側下肢"
    device: str = "cpu"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    model: ModelSpec
    source_path: Path
    models: dict[str, ModelSpec] = field(default_factory=dict)
    default_model_key: str = "mixed"
    auto_fallback_model_key: str = "mixed"


@dataclass(frozen=True)
class ModelSelection:
    requested_mode: str
    model_key: str
    source: str


@dataclass(frozen=True)
class ModelInfo:
    adapter: str
    display_name: str
    version: str
    cohort: str
    checkpoint_path: str
    checkpoint_sha256: str
    device: str
    input_width: int
    input_height: int
    stride: int
    epoch: int | None
    val_metrics: dict[str, float]
    checkpoint_schema_version: int
    architecture_id: str
    preprocessing_id: str
    metadata_source: str
    training_manifest_sha256: str | None
    decoder_id: str = "hard_argmax_v1"

    @property
    def short_hash(self) -> str:
        return self.checkpoint_sha256[:12]


@dataclass(frozen=True)
class LandmarkPrediction:
    schema_version: int
    points: dict[str, np.ndarray]
    lines: dict[str, dict[str, np.ndarray]]
    peak_scores: dict[str, float]
    model_info: ModelInfo
    elapsed_ms: float

    def named_points_payload(self) -> dict[str, dict[str, float]]:
        return {
            name: {"x": float(self.points[name][0]), "y": float(self.points[name][1])}
            for name in ANNOTATION_POINT_NAMES
        }

    def named_lines_payload(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            line_name: {
                endpoint: {"x": float(endpoints[endpoint][0]), "y": float(endpoints[endpoint][1])}
                for endpoint in ("p1", "p2")
            }
            for line_name, endpoints in self.lines.items()
        }


@dataclass(frozen=True)
class AnalysisResult:
    raw_path: Path
    raw_image: np.ndarray
    side: str
    prediction: LandmarkPrediction
    measurement: dict[str, Any]
    model_warnings: tuple[str, ...]
    warnings: tuple[str, ...]
    source_sha256: str
    total_elapsed_ms: float
    side_source: str
    model_selection: ModelSelection | None = None
    input_scope: str = "single-leg raster X-ray"
    inference_roi: tuple[int, int, int, int] | None = None
    roi_selection_method: str | None = None
    roi_confirmed: bool = False


@runtime_checkable
class CoordinateModelAdapter(Protocol):
    @property
    def info(self) -> ModelInfo:
        ...

    def load(self) -> ModelInfo:
        ...

    def predict(self, image_bgr: np.ndarray) -> LandmarkPrediction:
        ...


def bundled_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    override = os.environ.get("KNEE_XRAY_APP_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    root = bundled_root()
    if not getattr(sys, "frozen", False):
        organized_path = root / "config" / DEFAULT_CONFIG_FILENAME
        if organized_path.is_file():
            return organized_path
    return root / DEFAULT_CONFIG_FILENAME


def _resolve_config_resource(value: str | Path, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


def _parse_model_spec(model_payload: dict[str, Any], config_dir: Path) -> ModelSpec:
    if not model_payload.get("adapter") or not model_payload.get("checkpoint"):
        raise ModelLoadError("モデル設定には 'adapter' と 'checkpoint' の両方が必要です。")

    options = dict(model_payload.get("options") or {})
    try:
        low_peak_threshold = float(options.get("low_peak_threshold", 0.35))
    except (TypeError, ValueError) as exc:
        raise ModelLoadError("low_peak_threshold は0～1の有限数で指定してください。") from exc
    if not math.isfinite(low_peak_threshold) or not 0.0 <= low_peak_threshold <= 1.0:
        raise ModelLoadError("low_peak_threshold は0～1の有限数で指定してください。")
    options["low_peak_threshold"] = low_peak_threshold

    return ModelSpec(
        adapter=str(model_payload["adapter"]),
        checkpoint=_resolve_config_resource(model_payload["checkpoint"], config_dir),
        display_name=str(model_payload.get("display_name", "膝関節ランドマーク推定モデル")),
        version=str(model_payload.get("version", "unversioned")),
        cohort=str(model_payload.get("cohort", "片側下肢")),
        device=str(model_payload.get("device", "cpu")),
        options=options,
    )


def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path or default_config_path()).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ModelLoadError(f"アプリ設定ファイルを読み込めません：{config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"アプリ設定ファイルのJSON形式が正しくありません：{config_path}") from exc

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != APP_CONFIG_SCHEMA_VERSION:
        raise ModelLoadError(
            f"未対応のアプリ設定スキーマです（{schema_version}）。必要なバージョン：{APP_CONFIG_SCHEMA_VERSION}。"
        )
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ModelLoadError("アプリ設定には 'model' オブジェクトが必要です。")
    resource_root = config_path.parent
    source_config_path = bundled_root() / "config" / DEFAULT_CONFIG_FILENAME
    if not getattr(sys, "frozen", False) and config_path == source_config_path:
        resource_root = bundled_root()
    legacy_model = _parse_model_spec(model_payload, resource_root)
    models_payload = payload.get("models")
    if models_payload is None:
        models = {"mixed": legacy_model}
    else:
        if not isinstance(models_payload, dict) or not models_payload:
            raise ModelLoadError("アプリ設定の 'models' には1件以上のモデル設定が必要です。")
        models: dict[str, ModelSpec] = {}
        for raw_key, raw_spec in models_payload.items():
            key = str(raw_key).strip().lower()
            if not key or not key.replace("_", "").isalnum():
                raise ModelLoadError(f"モデルキーの形式が正しくありません：{raw_key}")
            if key in models:
                raise ModelLoadError(f"モデルキーが重複しています：{key}")
            if not isinstance(raw_spec, dict):
                raise ModelLoadError(f"モデル '{key}' の設定はオブジェクトで指定してください。")
            models[key] = _parse_model_spec(raw_spec, resource_root)

    default_model_key = str(payload.get("default_model_key", "mixed")).strip().lower()
    auto_fallback_model_key = str(payload.get("auto_fallback_model_key", default_model_key)).strip().lower()
    if models_payload is not None:
        missing_builtin = sorted(set(BUILTIN_MODEL_KEYS).difference(models))
        if missing_builtin:
            raise ModelLoadError(f"内蔵モデル設定が不足しています：{', '.join(missing_builtin)}")
        unexpected_models = sorted(set(models).difference(BUILTIN_MODEL_KEYS))
        if unexpected_models:
            raise ModelLoadError(f"未対応の内蔵モデル設定です：{', '.join(unexpected_models)}")
    if default_model_key not in models:
        raise ModelLoadError(f"既定モデル '{default_model_key}' が models にありません。")
    if auto_fallback_model_key not in models:
        raise ModelLoadError(f"自動判定のfallbackモデル '{auto_fallback_model_key}' が models にありません。")
    if models_payload is not None and legacy_model != models[default_model_key]:
        raise ModelLoadError("従来形式の model 設定は default_model_key の内蔵モデルと一致する必要があります。")
    return AppConfig(
        schema_version=schema_version,
        model=models[default_model_key],
        source_path=config_path,
        models=models,
        default_model_key=default_model_key,
        auto_fallback_model_key=auto_fallback_model_key,
    )


def _model_key_candidates(*sources: object) -> set[str]:
    candidates: set[str] = set()
    bone_phrases = (
        "未加入人工關節",
        "未加入人工関節",
        "人工關節なし",
        "人工関節なし",
        "非人工關節",
        "非人工関節",
        "without implant",
        "non-tka",
        "non_tka",
    )
    tka_phrases = (
        "加入人工關節",
        "加入人工関節",
        "人工關節あり",
        "人工関節あり",
        "人工膝關節",
        "人工膝関節",
        "膝關節置換",
        "膝関節置換",
        "total knee arthroplasty",
    )
    for source in sources:
        if source is None:
            continue
        text = unicodedata.normalize("NFKC", str(source)).casefold()
        bone_match = any(phrase in text for phrase in bone_phrases)
        for phrase in bone_phrases:
            text = text.replace(phrase, " ")
        tka_match = any(phrase in text for phrase in tka_phrases)
        for phrase in tka_phrases:
            text = text.replace(phrase, " ")
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", text)))
        bone_match = bone_match or "bone" in tokens
        tka_match = tka_match or "tka" in tokens
        mixed_match = "mixed" in tokens
        if bone_match:
            candidates.add("bone")
        if tka_match:
            candidates.add("tka")
        if mixed_match:
            candidates.add("mixed")
    return candidates


def resolve_model_selection(
    requested_mode: str,
    available_model_keys: Iterable[str],
    *sources: object,
    fallback_model_key: str = "mixed",
) -> ModelSelection:
    available = {str(key).strip().lower() for key in available_model_keys}
    mode = str(requested_mode).strip().lower()
    if mode != "auto":
        if mode not in available:
            raise ModelLoadError(f"選択したモデル '{mode}' は利用できません。")
        return ModelSelection(requested_mode=mode, model_key=mode, source="manual_override")

    candidates = _model_key_candidates(*sources)
    inferred = next(iter(candidates)) if len(candidates) == 1 else None
    if inferred in available:
        return ModelSelection(requested_mode="auto", model_key=inferred, source="filename")
    fallback = str(fallback_model_key).strip().lower()
    if fallback not in available:
        raise ModelLoadError(f"自動判定のfallbackモデル '{fallback}' は利用できません。")
    source = "auto_fallback_conflict" if len(candidates) > 1 else "auto_fallback_unknown"
    return ModelSelection(requested_mode="auto", model_key=fallback, source=source)


def model_spec_with_checkpoint(spec: ModelSpec, checkpoint: Path) -> ModelSpec:
    return replace(spec, checkpoint=Path(checkpoint).expanduser().resolve())


def user_preferences_path() -> Path:
    override = os.environ.get("KNEE_XRAY_APP_DATA")
    if override:
        base = Path(override).expanduser()
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "KneeXrayMeasurement"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "KneeXrayMeasurement"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "KneeXrayMeasurement"
    return base / "model.json"


def load_model_preference(base_spec: ModelSpec) -> tuple[ModelSpec, str | None]:
    environment_checkpoint = os.environ.get("KNEE_XRAY_MODEL_CHECKPOINT")
    if environment_checkpoint:
        checkpoint = Path(environment_checkpoint).expanduser().resolve()
        return model_spec_with_checkpoint(base_spec, checkpoint), None

    path = user_preferences_path()
    if not path.is_file():
        return base_spec, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("未対応の設定形式です")
        checkpoint = Path(payload["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            return base_spec, f"前回選択した外部AIモデルが見つからないため、標準モデルに戻しました：{checkpoint}"
        expected_sha256 = str(payload.get("checkpoint_sha256", "")).strip().lower()
        if expected_sha256 and sha256_file(checkpoint) != expected_sha256:
            return base_spec, "保存済みの外部AIモデルが変更されているため、標準モデルに戻しました。"
        spec = replace(
            base_spec,
            adapter=str(payload.get("adapter", base_spec.adapter)),
            checkpoint=checkpoint,
            display_name=str(payload.get("display_name", base_spec.display_name)),
            version=str(payload.get("version", "auto")),
            cohort=str(payload.get("cohort", "対象データ未指定（外部モデル）")),
            # Release runtime is CPU-only. Ignore device values left by older
            # preferences so an upgrade cannot re-enable CUDA/MPS implicitly.
            device=base_spec.device,
            options=dict(payload.get("options") or base_spec.options),
        )
        return spec, None
    except Exception as exc:
        return base_spec, f"外部AIモデル設定を読み込めないため、標準モデルに戻しました：{exc}"


def save_model_preference(spec: ModelSpec, expected_sha256: str | None = None) -> Path:
    """Persist a validated model as a content-addressed managed copy.

    ``expected_sha256`` should be the digest reported by the adapter that
    successfully loaded ``spec.checkpoint``.  Keeping it optional preserves
    compatibility with existing administrative callers, while the GUI can use
    it to reject a source file that changed after validation.
    """

    path = user_preferences_path()
    source = Path(spec.checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"AIモデルファイルが見つかりません：{source}")
    expected_digest = str(expected_sha256 or "").strip().lower()
    if expected_digest and (
        len(expected_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError("expected_sha256 は64文字のSHA256値で指定してください。")
    digest = sha256_file(source)
    if expected_digest and digest != expected_digest:
        raise ModelLoadError(
            "選択したAIモデルファイルは読み込み後に変更されたため、設定を保存できません。"
            "ファイルの更新が完了してから、もう一度選択してください。"
        )
    managed_dir = path.parent / "models"
    managed_dir.mkdir(parents=True, exist_ok=True)
    managed_checkpoint = managed_dir / f"{digest}{source.suffix.lower() or '.pt'}"
    if source != managed_checkpoint:
        temporary = managed_dir / f".{managed_checkpoint.name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != digest:
                raise ModelLoadError(
                    "AIモデルファイルが保存中に変更されたため、設定を保存できません。"
                )
            temporary.replace(managed_checkpoint)
        finally:
            if temporary.exists():
                temporary.unlink()
    write_json(
        path,
        {
            "schema_version": 1,
            "adapter": spec.adapter,
            "checkpoint": str(managed_checkpoint),
            "checkpoint_sha256": digest,
            "display_name": spec.display_name,
            "version": spec.version,
            "cohort": spec.cohort,
            "device": spec.device,
            "options": spec.options,
        },
    )
    return path


def clear_model_preference() -> None:
    path = user_preferences_path()
    managed_checkpoint: Path | None = None
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidate = Path(payload.get("checkpoint", "")).expanduser().resolve()
            managed_dir = (path.parent / "models").resolve()
            if candidate.is_relative_to(managed_dir):
                managed_checkpoint = candidate
        except Exception:
            managed_checkpoint = None
    if path.exists():
        path.unlink()
    if managed_checkpoint is not None and managed_checkpoint.is_file():
        managed_checkpoint.unlink()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class SmallHeatmapV1Adapter:
    """Safe runtime wrapper for the current SmallHeatmapNet state-dict format."""

    ADAPTER_ID = "small_heatmap_v1"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._info: ModelInfo | None = None
        self._image_width = 0
        self._image_height = 0
        self._stride = 0
        self._decoder_id = "hard_argmax_v1"

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            raise ModelLoadError("AIモデルが読み込まれていません。")
        return self._info

    def load(self) -> ModelInfo:
        checkpoint_path = self.spec.checkpoint
        if not checkpoint_path.is_file():
            raise ModelLoadError(f"AIモデルファイルが見つかりません：{checkpoint_path}")

        try:
            import torch

            from knee_xray.ml.knee_keypoint_model import (
                HARD_ARGMAX_DECODER_ID,
                KEYPOINT_NAMES,
                SUPPORTED_DECODER_IDS,
                SmallHeatmapNet,
                select_device,
            )
        except ImportError as exc:
            raise ModelLoadError(
                "アプリに必要な推論ライブラリが含まれていません。管理者にお問い合わせください。"
            ) from exc

        try:
            device = select_device(self.spec.device)
            # Load and hash one immutable byte snapshot.  Hashing the path after
            # torch.load would let an in-place file replacement make ModelInfo
            # describe bytes different from the model that was activated.
            checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
            checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
        except TypeError as exc:
            raise ModelLoadError("AIモデルを安全に読み込むための推論ライブラリが古すぎます。") from exc
        except Exception as exc:
            raise ModelLoadError(f"AIモデルファイルを読み込めません：{checkpoint_path}\n{exc}") from exc

        if not isinstance(checkpoint, dict):
            raise ModelLoadError("AIモデルファイルの形式が正しくありません。model_state とメタデータが必要です。")
        required = {"model_state", "keypoint_names", "image_width", "image_height", "stride"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ModelLoadError(f"AIモデルファイルに必要な項目がありません：{', '.join(missing)}")

        keypoint_names = tuple(checkpoint["keypoint_names"])
        if keypoint_names != EXPECTED_KEYPOINT_NAMES or keypoint_names != tuple(KEYPOINT_NAMES):
            raise ModelLoadError(
                "ランドマーク定義に互換性がありません。"
                f"必要な定義：{EXPECTED_KEYPOINT_NAMES}、ファイル内：{keypoint_names}。"
            )
        image_width = int(checkpoint["image_width"])
        image_height = int(checkpoint["image_height"])
        stride = int(checkpoint["stride"])
        if image_width <= 0 or image_height <= 0 or stride <= 0:
            raise ModelLoadError("AIモデルの入力サイズとstrideは正の値である必要があります。")
        if image_width % stride or image_height % stride:
            raise ModelLoadError("AIモデルの入力サイズはstrideで割り切れる必要があります。")

        checkpoint_schema = int(checkpoint.get("checkpoint_schema_version", 0))
        architecture_id = str(checkpoint.get("architecture_id", self.ADAPTER_ID))
        adapter_id = str(checkpoint.get("adapter_id", self.ADAPTER_ID))
        preprocessing_id = str(checkpoint.get("preprocessing_id", PREPROCESSING_ID))
        decoder_id = str(checkpoint.get("decoder_id", HARD_ARGMAX_DECODER_ID))
        if checkpoint_schema not in {0, CHECKPOINT_SCHEMA_VERSION}:
            raise ModelLoadError(
                f"未対応のAIモデル形式です（{checkpoint_schema}）。必要なバージョン：{CHECKPOINT_SCHEMA_VERSION}。"
            )
        if checkpoint_schema == 0 and not bool(self.spec.options.get("allow_legacy_checkpoint", False)):
            raise ModelLoadError(
                "外部AIモデルには形式バージョン、前処理、モデル構造のメタデータが必要です。"
                "現在の学習コードで書き出したスキーマバージョン1のモデルを選択してください。"
            )
        if architecture_id != ARCHITECTURE_ID or adapter_id != self.ADAPTER_ID:
            raise ModelLoadError(
                f"AIモデルの構造またはアダプターに互換性がありません：{architecture_id}/{adapter_id}。"
            )
        if preprocessing_id != PREPROCESSING_ID:
            raise ModelLoadError(
                f"前処理 '{preprocessing_id}' は必要な形式 '{PREPROCESSING_ID}' と一致しません。"
            )
        if decoder_id not in SUPPORTED_DECODER_IDS:
            raise ModelLoadError(f"座標デコーダー '{decoder_id}' には対応していません。")
        metadata_source = "checkpoint_manifest" if checkpoint_schema else "legacy_assumed_contract"

        model = SmallHeatmapNet(out_channels=len(EXPECTED_KEYPOINT_NAMES)).to(device)
        try:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            with torch.inference_mode():
                output = model(torch.zeros((1, 1, image_height, image_width), dtype=torch.float32, device=device))
        except Exception as exc:
            raise ModelLoadError(f"AIモデルのパラメータに互換性がありません（{self.ADAPTER_ID}）：{exc}") from exc

        expected_shape = (1, len(EXPECTED_KEYPOINT_NAMES), image_height // stride, image_width // stride)
        if tuple(output.shape) != expected_shape:
            raise ModelLoadError(f"モデル自己テストの出力形状が不正です：{tuple(output.shape)}（必要：{expected_shape}）。")
        if not bool(torch.isfinite(output).all().item()):
            raise ModelLoadError("モデル自己テストでNaNまたは無限大のヒートマップ値が検出されました。")

        val_metrics = {
            str(key): float(value)
            for key, value in dict(checkpoint.get("val_metrics") or {}).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        epoch = int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None
        configured_version = self.spec.version.strip()
        if not configured_version or configured_version.lower() in {"auto", "unversioned"}:
            checkpoint_version = checkpoint.get("model_version")
            version = str(checkpoint_version) if checkpoint_version else checkpoint_path.stem
            if epoch is not None and checkpoint_version is None:
                version = f"{version}-epoch{epoch}"
        else:
            version = configured_version

        if checkpoint_schema:
            display_name = str(checkpoint.get("model_name", self.spec.display_name))
            cohort = str(checkpoint.get("model_scope", checkpoint.get("cohort", self.spec.cohort)))
        else:
            display_name = self.spec.display_name
            cohort = self.spec.cohort
        info = ModelInfo(
            adapter=self.ADAPTER_ID,
            display_name=display_name,
            version=version,
            cohort=cohort,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            device=str(device),
            input_width=image_width,
            input_height=image_height,
            stride=stride,
            epoch=epoch,
            val_metrics=val_metrics,
            checkpoint_schema_version=checkpoint_schema,
            architecture_id=architecture_id,
            preprocessing_id=preprocessing_id,
            metadata_source=metadata_source,
            training_manifest_sha256=(
                str(checkpoint["training_manifest_sha256"])
                if checkpoint.get("training_manifest_sha256")
                else None
            ),
            decoder_id=decoder_id,
        )

        # Atomic activation: keep the previous model untouched until all checks pass.
        self._torch = torch
        self._device = device
        self._model = model
        self._image_width = image_width
        self._image_height = image_height
        self._stride = stride
        self._decoder_id = decoder_id
        self._info = info
        return info

    def predict(self, image_bgr: np.ndarray) -> LandmarkPrediction:
        if self._model is None or self._torch is None or self._device is None:
            raise InferenceError("解析前にAIモデルを読み込んでください。")

        from knee_xray.ml.knee_keypoint_model import (
            KEYPOINT_NAMES,
            decode_heatmaps_for_shape,
            preprocess_xray_array,
        )

        original_height, original_width = image_bgr.shape[:2]
        started = time.perf_counter()
        try:
            image = preprocess_xray_array(image_bgr, self._image_width, self._image_height)
            tensor = self._torch.from_numpy(image[None, None, ...]).to(self._device)
            with self._torch.inference_mode():
                logits = self._model(tensor)[0]
            coords, scores = decode_heatmaps_for_shape(
                logits,
                original_width=original_width,
                original_height=original_height,
                image_width=self._image_width,
                image_height=self._image_height,
                stride=self._stride,
                decoder_id=self._decoder_id,
            )
        except Exception as exc:
            raise InferenceError(f"AI解析に失敗しました：{exc}") from exc

        if (
            coords.shape != (len(EXPECTED_KEYPOINT_NAMES), 2)
            or scores.shape != (len(EXPECTED_KEYPOINT_NAMES),)
            or not np.all(np.isfinite(coords))
            or not np.all(np.isfinite(scores))
        ):
            raise InferenceError("AIモデルがNaN、無限大、または不正なランドマーク推定結果を返しました。")

        points: dict[str, np.ndarray] = {}
        lines: dict[str, dict[str, np.ndarray]] = {name: {} for name in ANNOTATION_LINE_NAMES}
        peak_scores: dict[str, float] = {}
        for name, coordinate, score in zip(KEYPOINT_NAMES, coords, scores):
            coordinate = np.asarray(coordinate, dtype=np.float32)
            peak_scores[name] = float(score)
            if name in ANNOTATION_POINT_NAMES:
                points[name] = coordinate
            else:
                line_name, endpoint = LINE_ENDPOINT_NAMES[name]
                lines[line_name][endpoint] = coordinate

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LandmarkPrediction(
            schema_version=PREDICTION_SCHEMA_VERSION,
            points=points,
            lines=lines,
            peak_scores=peak_scores,
            model_info=self.info,
            elapsed_ms=elapsed_ms,
        )


def create_model_adapter(spec: ModelSpec) -> CoordinateModelAdapter:
    if spec.adapter in {SmallHeatmapV1Adapter.ADAPTER_ID, "torch_state_dict_heatmap_v1"}:
        return SmallHeatmapV1Adapter(spec)

    if ":" not in spec.adapter:
        raise ModelLoadError(
            f"不明なモデルアダプターです：'{spec.adapter}'。"
        )
    module_name, class_name = spec.adapter.split(":", 1)
    try:
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        adapter = adapter_class(spec)
    except Exception as exc:
        raise ModelLoadError(f"カスタムモデルアダプター '{spec.adapter}' を作成できません：{exc}") from exc
    if not isinstance(adapter, CoordinateModelAdapter):
        raise ModelLoadError(f"カスタムモデルアダプター '{spec.adapter}' は必要な実行インターフェースを実装していません。")
    return adapter


def resolve_side(raw_path: Path, requested_side: str | None) -> str:
    explicit = normalize_measurement_side(requested_side)
    side = explicit or infer_knee_side_from_sources(raw_path)
    if side is None:
        raise SideRequiredError("ファイル名から左右を判定できません。解析前にLまたはRを選択してください。")
    return side


def _all_coordinates(
    points: dict[str, np.ndarray],
    lines: dict[str, dict[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    values = list(points.items())
    for line_name, endpoints in lines.items():
        for endpoint, point in endpoints.items():
            values.append((f"{line_name}_{endpoint}", point))
    return values


def coordinate_geometry_warnings(
    points: dict[str, np.ndarray],
    lines: dict[str, dict[str, np.ndarray]],
    image_shape: tuple[int, ...],
) -> tuple[str, ...]:
    image_height, image_width = image_shape[:2]
    warnings: list[str] = []
    out_of_bounds = [
        name
        for name, point in _all_coordinates(points, lines)
        if point[0] < 0 or point[0] >= image_width or point[1] < 0 or point[1] >= image_height
    ]
    if out_of_bounds:
        labels = ", ".join(coordinate_display_name(name) for name in out_of_bounds)
        warnings.append(f"画像範囲外のランドマークがあります：{labels}")

    center_checks = (
        (
            "upper_center",
            "upper_left",
            "upper_right",
            "点3（大腿骨側中央点）が点2と点4の水平方向の間にありません。mLDFAとHKAを計算する前に位置を確認してください。",
        ),
        (
            "lower_center",
            "lower_left",
            "lower_right",
            "点6（脛骨側中央点）が点5と点7の水平方向の間にありません。MPTAとHKAを計算する前に位置を確認してください。",
        ),
    )
    for center_name, outer_a_name, outer_b_name, message in center_checks:
        center_x = float(points[center_name][0])
        outer_a_x = float(points[outer_a_name][0])
        outer_b_x = float(points[outer_b_name][0])
        if not min(outer_a_x, outer_b_x) < center_x < max(outer_a_x, outer_b_x):
            warnings.append(message)

    hip_y = float(points["hip"][1])
    upper_y = float(points["upper_center"][1])
    lower_y = float(points["lower_center"][1])
    ankle_y = float(points["ankle"][1])
    if not (hip_y < upper_y < lower_y < ankle_y):
        warnings.append("点1・点3・点6・点8の上下方向の解剖学的順序が不自然です。位置を確認・修正してください。")

    if float(np.linalg.norm(points["hip"] - points["upper_center"])) < 8.0:
        warnings.append("大腿骨の機械軸を定義する2点が近すぎるため、角度を正しく計算できません。")
    if float(np.linalg.norm(points["lower_center"] - points["ankle"])) < 8.0:
        warnings.append("脛骨の機械軸を定義する2点が近すぎるため、角度を正しく計算できません。")
    for line_name, endpoints in lines.items():
        if float(np.linalg.norm(endpoints["p2"] - endpoints["p1"])) < 8.0:
            warnings.append(f"{LINE_DISPLAY_NAMES.get(line_name, line_name)}の2端点が近すぎるため、角度が不正確な可能性があります。")
    return tuple(warnings)


def ensure_valid_coordinate_geometry(
    points: dict[str, np.ndarray],
    lines: dict[str, dict[str, np.ndarray]],
) -> None:
    for name in ANNOTATION_POINT_NAMES:
        if name not in points or not np.all(np.isfinite(points[name])):
            raise InferenceError(f"ランドマーク座標がない、または無効です：{coordinate_display_name(name)}。")
    for line_name in ANNOTATION_LINE_NAMES:
        endpoints = lines.get(line_name, {})
        for endpoint in ("p1", "p2"):
            if endpoint not in endpoints or not np.all(np.isfinite(endpoints[endpoint])):
                key = f"{line_name}_{endpoint}"
                raise InferenceError(f"関節線端点の座標がない、または無効です：{coordinate_display_name(key)}。")
    if float(np.linalg.norm(points["hip"] - points["upper_center"])) < 8.0:
        raise InferenceError("大腿骨の機械軸を定義する2点が近すぎます。")
    if float(np.linalg.norm(points["lower_center"] - points["ankle"])) < 8.0:
        raise InferenceError("脛骨の機械軸を定義する2点が近すぎます。")
    for line_name, endpoints in lines.items():
        if float(np.linalg.norm(endpoints["p2"] - endpoints["p1"])) < 8.0:
            raise InferenceError(f"{LINE_DISPLAY_NAMES.get(line_name, line_name)}の2端点が近すぎます。")


def model_quality_warnings(
    prediction: LandmarkPrediction,
    low_peak_threshold: float,
) -> tuple[str, ...]:
    warnings: list[str] = []
    low_scores = [
        name
        for name, score in prediction.peak_scores.items()
        if not math.isfinite(score) or score < low_peak_threshold
    ]
    if low_scores:
        labels = ", ".join(coordinate_display_name(name) for name in low_scores)
        warnings.append(f"AIスコアが低い、または無効です。次のランドマークを優先して確認してください：{labels}")

    val_metrics = prediction.model_info.val_metrics
    if not val_metrics:
        warnings.append("このAIモデルには検証指標が含まれていないため、検証時の誤差を表示できません。")
    else:
        required_angle_metrics = {"mldfa_mae_deg", "mpta_mae_deg"}
        missing_metrics = sorted(required_angle_metrics.difference(val_metrics))
        if missing_metrics:
            metric_labels = {
                "mldfa_mae_deg": "mLDFAの平均絶対誤差",
                "mpta_mae_deg": "MPTAの平均絶対誤差",
            }
            warnings.append(
                "このAIモデルには必要な角度検証指標がありません："
                f"{', '.join(metric_labels.get(name, name) for name in missing_metrics)}。"
                "検証時の誤差を完全には表示できません。"
            )
        elif max(val_metrics["mldfa_mae_deg"], val_metrics["mpta_mae_deg"]) > 5.0:
            warnings.append("このAIモデルは検証時の角度誤差が大きいため、すべての結果を必ず医師が確認してください。")
    if prediction.model_info.metadata_source == "legacy_assumed_contract":
        warnings.append("このモデルファイルは旧形式で、前処理およびモデル構造の情報が明示されていません。互換設定で読み込みました。")
    return tuple(warnings)


def ensure_finite_measurement(measurement: dict[str, Any]) -> None:
    for key in ("mldfa_angle", "mpta_angle", "jlca_angle", "hka_angle"):
        if not math.isfinite(float(measurement[key])):
            raise InferenceError(f"計測値 {key} がNaNまたは無限大になりました。")


def measurement_out_of_range_angles(measurement: dict[str, Any]) -> tuple[str, ...]:
    """Return angle labels that fail the app's broad technical range checks."""

    ensure_finite_measurement(measurement)
    out_of_range: list[str] = []
    if not 45.0 <= float(measurement["mldfa_angle"]) <= 135.0:
        out_of_range.append("mLDFA")
    if not 45.0 <= float(measurement["mpta_angle"]) <= 135.0:
        out_of_range.append("MPTA")
    if abs(float(measurement["jlca_angle"])) > 30.0:
        out_of_range.append("JLCA")
    if abs(float(measurement["hka_angle"])) > 45.0:
        out_of_range.append("HKA")
    return tuple(out_of_range)


def measurement_warnings(measurement: dict[str, Any]) -> tuple[str, ...]:
    out_of_range = set(measurement_out_of_range_angles(measurement))
    warnings: list[str] = []
    mldfa = float(measurement["mldfa_angle"])
    mpta = float(measurement["mpta_angle"])
    jlca = float(measurement["jlca_angle"])
    hka = float(measurement["hka_angle"])
    if "mLDFA" in out_of_range:
        warnings.append(f"mLDFA={mldfa:.1f}° は、広めに設定した技術的チェック範囲（45～135°）を外れています。")
    if "MPTA" in out_of_range:
        warnings.append(f"MPTA={mpta:.1f}° は、広めに設定した技術的チェック範囲（45～135°）を外れています。")
    if "JLCA" in out_of_range:
        warnings.append(f"|JLCA|={abs(jlca):.1f}° は、広めに設定した技術的チェック範囲（30°以内）を外れています。")
    if "HKA" in out_of_range:
        warnings.append(f"|HKA|={abs(hka):.1f}° は、広めに設定した技術的チェック範囲（45°以内）を外れています。")
    return tuple(warnings)


def combine_warnings(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    combined: list[str] = []
    for group in groups:
        for warning in group:
            if warning not in combined:
                combined.append(warning)
    return tuple(combined)


def _normalize_crop_box(
    crop_box: tuple[int, int, int, int] | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    if crop_box is None:
        return None
    if len(crop_box) != 4:
        raise ValueError("ROIは (x0, y0, x1, y1) の4つの座標で指定してください。")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in crop_box):
        raise ValueError("ROIの座標は整数で指定してください。")

    x0, y0, x1, y1 = (int(value) for value in crop_box)
    image_height, image_width = image_shape[:2]
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        raise ValueError(
            f"ROI ({x0}, {y0}, {x1}, {y1}) が元画像の範囲 "
            f"(0, 0, {image_width}, {image_height}) に収まっていません。"
        )
    return x0, y0, x1, y1


def _offset_prediction(
    prediction: LandmarkPrediction,
    offset_x: int,
    offset_y: int,
) -> LandmarkPrediction:
    offset = np.asarray([offset_x, offset_y], dtype=np.float32)
    points = {
        name: np.asarray(point, dtype=np.float32) + offset
        for name, point in prediction.points.items()
    }
    lines = {
        line_name: {
            endpoint: np.asarray(point, dtype=np.float32) + offset
            for endpoint, point in endpoints.items()
        }
        for line_name, endpoints in prediction.lines.items()
    }
    return replace(prediction, points=points, lines=lines)


class KneeAnalysisService:
    def __init__(
        self,
        adapter: CoordinateModelAdapter,
        low_peak_threshold: float = 0.35,
        render_component_images: bool = True,
    ) -> None:
        self.adapter = adapter
        self.low_peak_threshold = float(low_peak_threshold)
        self.render_component_images = bool(render_component_images)
        if not math.isfinite(self.low_peak_threshold) or not 0.0 <= self.low_peak_threshold <= 1.0:
            raise ModelLoadError("low_peak_threshold は0～1の有限数で指定してください。")

    def analyze_path(
        self,
        raw_path: Path,
        requested_side: str | None = None,
        crop_box: tuple[int, int, int, int] | None = None,
        roi_selection_method: str = "explicit",
        roi_confirmed: bool = True,
    ) -> AnalysisResult:
        started = time.perf_counter()
        raw_path = Path(raw_path).expanduser().resolve()
        explicit_side = normalize_measurement_side(requested_side)
        side = resolve_side(raw_path, explicit_side)
        side_source = "explicit" if explicit_side is not None else "filename"
        source_sha256 = sha256_file(raw_path)
        raw_image = read_color(raw_path)
        if sha256_file(raw_path) != source_sha256:
            raise InferenceError("解析中に元画像が変更されました。画像を開き直してください。")
        inference_roi = _normalize_crop_box(crop_box, raw_image.shape)
        inference_image = raw_image
        selection_method: str | None = None
        if inference_roi is not None:
            selection_method = str(roi_selection_method).strip()
            if not selection_method:
                raise ValueError("ROIの選択方法は空文字にできません。")
            x0, y0, x1, y1 = inference_roi
            inference_image = np.ascontiguousarray(raw_image[y0:y1, x0:x1])

        prediction = self.adapter.predict(inference_image)
        if inference_roi is not None:
            prediction = _offset_prediction(prediction, inference_roi[0], inference_roi[1])
        ensure_valid_coordinate_geometry(prediction.points, prediction.lines)
        measurement, _debug = measure_from_named_points(
            raw_image,
            prediction.named_points_payload(),
            raw_path=raw_path,
            named_lines=prediction.named_lines_payload(),
            side=side,
            render_style=RENDER_STYLE_CLINICAL,
            render_component_images=self.render_component_images,
        )
        ensure_finite_measurement(measurement)
        model_warnings = model_quality_warnings(prediction, self.low_peak_threshold)
        warnings = combine_warnings(
            coordinate_geometry_warnings(prediction.points, prediction.lines, raw_image.shape),
            model_warnings,
            measurement_warnings(measurement),
        )
        return AnalysisResult(
            raw_path=raw_path,
            raw_image=raw_image,
            side=side,
            prediction=prediction,
            measurement=measurement,
            model_warnings=model_warnings,
            warnings=warnings,
            source_sha256=source_sha256,
            total_elapsed_ms=(time.perf_counter() - started) * 1000.0,
            side_source=side_source,
            input_scope=(
                "bilateral raster X-ray"
                if inference_roi is not None
                else "single-leg raster X-ray"
            ),
            inference_roi=inference_roi,
            roi_selection_method=selection_method,
            roi_confirmed=bool(roi_confirmed) if inference_roi is not None else False,
        )


def measurement_from_coordinates(
    raw_image: np.ndarray,
    raw_path: Path,
    side: str,
    points: dict[str, np.ndarray],
    lines: dict[str, dict[str, np.ndarray]],
    render_component_images: bool = True,
) -> dict[str, Any]:
    ensure_valid_coordinate_geometry(points, lines)
    named_points = {
        name: {"x": float(points[name][0]), "y": float(points[name][1])}
        for name in ANNOTATION_POINT_NAMES
    }
    named_lines = {
        line_name: {
            endpoint: {"x": float(endpoints[endpoint][0]), "y": float(endpoints[endpoint][1])}
            for endpoint in ("p1", "p2")
        }
        for line_name, endpoints in lines.items()
    }
    measurement, _debug = measure_from_named_points(
        raw_image,
        named_points,
        raw_path=raw_path,
        named_lines=named_lines,
        side=side,
        render_style=RENDER_STYLE_CLINICAL,
        render_component_images=render_component_images,
    )
    ensure_finite_measurement(measurement)
    return measurement


def export_record(
    analysis: AnalysisResult,
    points: dict[str, np.ndarray],
    lines: dict[str, dict[str, np.ndarray]],
    measurement: dict[str, Any],
    app_version: str,
    manually_modified: bool,
    edited_keys: set[str] | None = None,
    app_release_channel: str | None = None,
) -> dict[str, Any]:
    ensure_finite_measurement(measurement)
    recomputed = measurement_from_coordinates(
        analysis.raw_image,
        analysis.raw_path,
        analysis.side,
        points,
        lines,
        render_component_images=False,
    )
    for key in ("mldfa_angle", "mpta_angle", "jlca_angle", "hka_angle"):
        if abs(float(recomputed[key]) - float(measurement[key])) > 1e-5:
            raise InferenceError("ランドマーク座標と表示中の角度が一致しないため、書き出しを中止しました。")
    edited = set(edited_keys or ())
    current_warnings = combine_warnings(
        analysis.model_warnings,
        coordinate_geometry_warnings(points, lines, analysis.raw_image.shape),
        measurement_warnings(measurement),
    )

    def point_record(name: str, point: np.ndarray, original: np.ndarray) -> dict[str, Any]:
        return {
            "x": float(point[0]),
            "y": float(point[1]),
            "source": "manual" if name in edited else "model",
            "model_prediction": {"x": float(original[0]), "y": float(original[1])},
            "model_peak_score": analysis.prediction.peak_scores.get(name),
        }

    analysis_payload: dict[str, Any] = {
        "side": analysis.side,
        "side_source": analysis.side_source,
        "manually_modified": bool(manually_modified),
        "edited_keys": sorted(edited),
        "prediction_schema_version": analysis.prediction.schema_version,
        "coordinate_space": {
            "unit": "pixel",
            "origin": "top-left",
            "x_direction": "right",
            "y_direction": "down",
        },
        "inference_elapsed_ms": round(analysis.prediction.elapsed_ms, 3),
        "total_elapsed_ms": round(analysis.total_elapsed_ms, 3),
        "warnings": list(current_warnings),
    }
    if analysis.model_selection is not None:
        analysis_payload["model_selection"] = asdict(analysis.model_selection)
    if analysis.inference_roi is not None:
        x0, y0, x1, y1 = analysis.inference_roi
        analysis_payload["inference_roi"] = {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "width": x1 - x0,
            "height": y1 - y0,
            "coordinate_space": "source_image_pixels",
            "selection_method": analysis.roi_selection_method,
            "confirmed": analysis.roi_confirmed,
        }

    app_payload = {"name": "Knee X-ray Auto Measurement", "version": app_version}
    if app_release_channel:
        app_payload["release_channel"] = app_release_channel

    return {
        "schema_version": 1,
        "app": app_payload,
        "source": {
            "filename": analysis.raw_path.name,
            "sha256": analysis.source_sha256,
            "image_width": int(analysis.raw_image.shape[1]),
            "image_height": int(analysis.raw_image.shape[0]),
            "input_scope": analysis.input_scope,
        },
        "model": asdict(analysis.prediction.model_info),
        "analysis": analysis_payload,
        "points": {
            name: point_record(name, points[name], analysis.prediction.points[name])
            for name in ANNOTATION_POINT_NAMES
        },
        "lines": {
            line_name: {
                endpoint: point_record(
                    f"{line_name}_{endpoint}",
                    endpoints[endpoint],
                    analysis.prediction.lines[line_name][endpoint],
                )
                for endpoint in ("p1", "p2")
            }
            for line_name, endpoints in lines.items()
        },
        "angle_convention": "HKA and JLCA: positive=varus, negative=valgus",
        "angles_deg": {
            "mLDFA": float(measurement["mldfa_angle"]),
            "MPTA": float(measurement["mpta_angle"]),
            "JLCA": float(measurement["jlca_angle"]),
            "HKA": float(measurement["hka_angle"]),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_overlay(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    extension = path.suffix.lower() or ".png"
    if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise ValueError(f"未対応の出力画像形式です：{extension}")
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"計測結果画像を {extension} 形式で保存できません。")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_result_bundle(
    json_path: Path,
    payload: dict[str, Any],
    overlay_path: Path,
    image: np.ndarray,
) -> None:
    """Commit JSON and overlay together, rolling both back if either replacement fails."""

    json_path = Path(json_path)
    overlay_path = Path(overlay_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    json_temp = json_path.with_name(f".{json_path.name}.{token}.tmp")
    overlay_temp = overlay_path.with_name(f".{overlay_path.stem}.{token}.tmp{overlay_path.suffix}")
    json_backup = json_path.with_name(f".{json_path.name}.{token}.bak")
    overlay_backup = overlay_path.with_name(f".{overlay_path.name}.{token}.bak")
    committed: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        json_temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        extension = overlay_path.suffix.lower() or ".png"
        ok, encoded = cv2.imencode(extension, image)
        if not ok:
            raise ValueError(f"計測結果画像を {extension} 形式で保存できません。")
        overlay_temp.write_bytes(encoded.tobytes())

        for target, backup in ((json_path, json_backup), (overlay_path, overlay_backup)):
            if target.exists():
                target.replace(backup)
                backed_up.append((target, backup))
        json_temp.replace(json_path)
        committed.append(json_path)
        overlay_temp.replace(overlay_path)
        committed.append(overlay_path)
    except Exception:
        for target in committed:
            if target.exists():
                target.unlink()
        for target, backup in backed_up:
            if backup.exists():
                backup.replace(target)
        raise
    finally:
        for path in (json_temp, overlay_temp, json_backup, overlay_backup):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

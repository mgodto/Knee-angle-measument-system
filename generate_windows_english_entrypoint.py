#!/usr/bin/env python3

"""Generate the Windows English GUI entrypoint from the canonical Japanese GUI.

The generated module is deliberately not maintained by hand.  This keeps the
model-selection and measurement behavior in ``knee_measurement_app.py`` as the
single source of truth while allowing the Windows package to avoid locale and
code-page problems in doctor-facing text.
"""

from __future__ import annotations

import argparse
import py_compile
import re
from pathlib import Path


DEFAULT_SOURCE = Path("knee_measurement_app.py")
DEFAULT_OUTPUT = Path("knee_measurement_app_windows.py")
CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


# Longest matches are applied first, so full sentences take precedence over
# shared UI fragments.  Every doctor-visible CJK string in the canonical app
# must be represented here; generation fails closed if a new one is added.
TRANSLATIONS = {
    "AIモデルの準備が完了しました。片側下肢全長X線画像を開いてください。": "AI model ready. Open a single-leg full-length X-ray image.",
    "片側下肢全長X線画像を開いてください\nAIが8個の解剖学的ランドマークと2本の関節線を推定します": "Open a single-leg full-length X-ray image.\nAI will estimate 8 anatomical landmarks and 2 joint lines.",
    "\n\n研究用ソフトウェアです。すべてのランドマークと角度を医師が確認してください。\nGPUは不要です。CPUで動作し、画像や結果を外部へ送信しません。": "\n\nResearch software. A doctor must review every landmark and angle.\nNo GPU is required. Processing runs on the CPU and does not upload images or results.",
    "GPUは不要です。CPUで動作し、画像や結果を外部へ送信しません。": "No GPU is required. Processing runs on the CPU and does not upload images or results.",
    "研究用ソフトウェアです。": "Research software. ",
    "警告がない場合も、すべてのランドマークと角度を医師が確認してください。AIスコアは臨床的な確信度ではありません。": "Even when no warning is shown, a doctor must review every landmark and angle. The AI score is not clinical confidence.",
    "警告がない場合も、": "Even when no warning is shown, ",
    "新しいAIモデルは適用されていません。以前の結果は消去しました。使用するモデルを再度選択してください。\n\n": "The new AI model was not applied. The previous result was cleared. Select a model again.\n\n",
    "左右はAIモデルの出力ではありません。元の検査情報に基づいてLまたはRを選択してください。": "Laterality is not predicted by the AI model. Select L or R from the original examination information.",
    "画像を選択しました。ファイル名から左右を判定できないため、LまたはRを選択してください。": "Image selected. Laterality could not be inferred from the filename; select L or R.",
    "前回の結果を消去しました。画像、左右、AIモデルを確認して再実行してください。": "The previous result was cleared. Check the image, laterality, and AI model, then run again.",
    "前回選択した外部AIモデルを読み込めなかったため、標準モデルに戻しました：": "The previously selected external AI model could not be loaded; restored a built-in model: ",
    "自動チェック：警告0件 — すべてのランドマークと角度を医師が確認してください": "Automated checks: 0 warnings - a doctor must review every landmark and angle",
    "左右を判定できないため、書き出しを無効にして以前の角度を消去しました。": "Laterality is unresolved, so export was disabled and previous angles were cleared.",
    "mLDFA、MPTA、JLCA、HKAの解剖学的方向には明確なL/R情報が必要です。": "The anatomical directions for mLDFA, MPTA, JLCA, and HKA require explicit L/R information.",
    "個のランドマークを手動修正済みです。AIスコアは修正前の推定位置に対する値です。": " landmark(s) were manually edited. AI scores refer to the original predictions.",
    "個のランドマークを手動修正済みです。": " landmark(s) were manually edited. ",
    "AIスコアは修正前の推定位置に対する値です。": "AI scores refer to the original predictions.",
    "ファイル名から左右を判定できません。上部でLまたはRを選択してください。": "Laterality could not be inferred from the filename. Select L or R above.",
    "手動修正を適用し、すべての角度を再計算しました。": "Manual edits applied and all angles recalculated.",
    "AI推定位置に戻し、すべての角度を再計算しました。": "Restored AI predictions and recalculated all angles.",
    "修正後のランドマーク配置では角度を計算できません\n": "Angles could not be calculated from the edited landmark positions.\n",
    "修正後のランドマーク配置では角度を計算できません": "Angles could not be calculated from the edited landmark positions.",
    "AIモデルは読み込まれましたが、選択内容を保存できません：": "The AI model was loaded, but the selection could not be saved: ",
    "新しいAIモデルを読み込めませんでした。以前の解析結果は消去しました。": "The new AI model could not be loaded. The previous analysis was cleared.",
    "マーカーをドラッグして修正できます。離すと角度を再計算します。": "Drag markers to edit them. Angles are recalculated on release.",
    "解析後、AIモデルと計測結果の確認事項を表示します。": "AI model and measurement review items will appear after analysis.",
    "解析後、計測結果を重ねた画像がここに表示されます": "The measurement overlay will appear here after analysis.",
    "AIモデルの準備が完了しました。LまたはRを選択してください。": "AI model ready. Select L or R.",
    "画像を選択しました。AIモデルの準備完了後に自動で解析します。": "Image selected. Analysis will start when the AI model is ready.",
    "次のファイルは既に存在します。上書きしますか？\n\n": "The following files already exist. Overwrite them?\n\n",
    "側として角度を再計算しました。ランドマーク座標は変更していません。": " side. Landmark coordinates were unchanged.",
    "今回の修正を元に戻すか、AI推定位置に戻してください。": "Undo this edit or restore the AI predictions.",
    "LまたはRを選択してください。以前の角度は消去されました。": "Select L or R. Previous angles were cleared.",
    "自動チェックで異常は検出されませんでした": "Automated checks found no abnormal condition",
    "計測結果画像・各角度・全ランドマーク座標": "Measurement overlay, angles, and all landmark coordinates",
    "片側下肢全長X線・ランドマーク自動推定／角度計測": "Single-leg full-length X-ray - automated landmarks and angle measurement",
    "JPG / PNG / BMP / TIFF（DICOM・両側未対応）": "JPG / PNG / BMP / TIFF (DICOM and bilateral images are not supported)",
    "ホイール：拡縮／右ドラッグ：移動／ダブルクリック：全体": "Wheel: zoom / right-drag: pan / double-click: fit",
    "AIモデル設定エラー — 詳細を確認してください": "AI model configuration error - review details",
    "解析に失敗しました — 詳細を確認してください": "Analysis failed - review details",
    "新しいAIモデルは適用されていません。": "The new AI model was not applied.",
    "以前の結果は消去しました。使用するモデルを再度選択してください。": "The previous result was cleared. Select a model again.",
    "左右変更後に角度を計算できません：": "Angles could not be calculated after changing laterality: ",
    "修正後の計算に失敗しました：": "Calculation after editing failed: ",
    "修正後の計算に失敗しました": "Calculation after editing failed",
    "前回選択した外部AIモデル": "Previously selected external AI model",
    "標準モデルに戻しました": "restored a built-in model",
    "AIモデルを検証して読み込んでいます": "Validating and loading the AI model",
    "検証済みのAIモデルを切り替えました。": "Switched to the validated AI model.",
    "片側下肢全長X線画像を選択": "Select a single-leg full-length X-ray image",
    "互換性のあるAIモデルファイルを選択": "Select a compatible AI model file",
    "対象データ未指定（外部モデル）": "Target cohort not specified (external model)",
    "ファイル名／フォルダから自動判定": "Automatic selection from file/folder name",
    "種類の候補が競合したため Mixed を使用": "Conflicting type indicators; using Mixed",
    "種類不明のため Mixed を使用": "Type unknown; using Mixed",
    "外部モデルを手動指定": "External model selected manually",
    "モデル情報は読み込み後に表示されます。": "Model information appears after loading.",
    "モデル選択：自動判定（画像未選択）": "Model selection: Auto (no image selected)",
    "AIモデル設定の確認事項：1件": "AI model configuration: 1 review item",
    "AIモデル設定の警告": "AI model configuration warning",
    "今回の解析に失敗しました": "This analysis failed",
    "解析完了後に結果を表示します": "Results appear after analysis",
    "モデルを使用できません：": "The model cannot be used: ",
    "使用モデル：": "Active model: ",
    "モデル：読み込み不可": "Model: unavailable",
    "モデル選択：読み込み不可": "Model selection: unavailable",
    "モデル：未読み込み": "Model: not loaded",
    "モデル設定エラー": "Model configuration error",
    "AIモデルの読み込みに失敗しました": "Failed to load the AI model",
    "自動計測に失敗しました": "Automated measurement failed",
    "解析に失敗しました：": "Analysis failed: ",
    "修正を元に戻す": "Undo edit",
    "AI推定位置に戻す": "Restore AI predictions",
    "左右変更済み・要確認": "Laterality changed - review required",
    "手動修正あり・要確認": "Manual edits - review required",
    "手動修正・計算エラー": "Manual edit - calculation error",
    "AI推定結果・要確認": "AI result - review required",
    "解析エラー・AIモデル": "Analysis error / AI model",
    "解析エラー": "Analysis error",
    "計算エラー・AIモデル": "Calculation error / AI model",
    "確認事項（0）・AIモデル": "Review items (0) / AI model",
    "確認事項（1）・AIモデル": "Review items (1) / AI model",
    "自動判定に戻す": "Return to Auto",
    "AIモデルファイルを選択": "Select AI model file",
    "結果を書き出す": "Export results",
    "X線画像を開く": "Open X-ray image",
    "X線画像": "X-ray image",
    "すべてのファイル": "All files",
    "PyTorchモデルファイル": "PyTorch model files",
    "画像が選択されていません": "No image selected",
    "AIモデルを準備しています": "Preparing AI model",
    "画像が読み込まれていません。": "No image is loaded.",
    "LまたはRを選択してください。": "Select L or R.",
    "左右を選択してください": "Select laterality",
    "左右の選択待ち": "Waiting for laterality",
    "解析結果はありません": "No analysis result",
    "確認事項：解析後に表示します": "Review items will appear after analysis",
    "結果の保存先を選択": "Select output folder",
    "既存の結果を上書き": "Overwrite existing results",
    "書き出しに失敗しました": "Export failed",
    "書き出し完了": "Export complete",
    "保存しました：\n\n": "Saved:\n\n",
    "研究用・要医師確認": "Research use - doctor review required",
    "下肢全長X線 自動計測": "Full-Length Leg X-ray Automated Measurement",
    "2  自動計測結果": "2  Automated measurement results",
    "1  元画像とAI推定点": "1  Source image and AI predictions",
    "このアプリについて": "About this application",
    "大腿骨関節線端点 1": "Femoral joint-line endpoint 1",
    "大腿骨関節線端点 2": "Femoral joint-line endpoint 2",
    "脛骨関節線端点 1": "Tibial joint-line endpoint 1",
    "脛骨関節線端点 2": "Tibial joint-line endpoint 2",
    "機械的外側遠位大腿骨角": "Mechanical lateral distal femoral angle",
    "内側近位脛骨角": "Medial proximal tibial angle",
    "関節裂隙収束角": "Joint line convergence angle",
    "股関節–膝関節–足関節角": "Hip-knee-ankle angle",
    "ファイル名から左右を判定できません": "Laterality could not be inferred from the filename",
    "左側画像のマーカーをドラッグして修正できます。": "Drag markers on the left image to edit them.",
    "ランドマーク座標は変更していません。": "Landmark coordinates were unchanged.",
    "すべてのランドマークと角度を医師が確認してください": "A doctor must review every landmark and angle",
    "AIスコアは臨床的な確信度ではありません。": "The AI score is not clinical confidence.",
    "モデル選択：": "Model selection: ",
    "使用中 ": "active ",
    "読み込み中": "loading",
    "読み込み中：": "Loading: ",
    "を解析しています": "Analyzing ",
    "解析完了：": "Analysis complete: ",
    "入力待ち": "Waiting for input",
    "入力 ": "input ",
    "AI解析中": "AI analysis in progress",
    "詳細を確認": "Review details",
    "ランドマーク": "Landmark",
    "修正区分": "Edit source",
    "座標一覧（12）": "Coordinates (12)",
    "AIランドマーク": "AI landmarks",
    "AI関節線": "AI joint lines",
    "手動修正": "Manual edits",
    "手動指定": "Manual selection",
    "手動": "Manual",
    "自動判定": "Auto",
    "外部モデル": "External model",
    "外部": "External",
    "画像種類 / AIモデル": "Image type / AI model",
    "左右": "Laterality",
    "再解析": "Run again",
    "終了": "Exit",
    "ファイル": "File",
    "編集": "Edit",
    "ヘルプ": "Help",
    "AIモデル": "AI model",
    "AIモデルを選択できません": "Cannot select AI model",
    "AIスコア": "AI score",
    "モデル：": "Model: ",
    "読込中：": "Loading: ",
    "検証時ランドマークMAE ": "Validation landmark MAE ",
    "旧形式（互換読み込み）": "Legacy format (compatibility mode)",
    "埋め込みマニフェスト": "Embedded manifest",
    "対象：": "Target: ",
    "メタデータ：": "Metadata: ",
    "実行環境：": "Runtime: ",
    "選択：": "Selection: ",
    "判定不明": "unknown type",
    "人工関節なし": "no knee implant",
    "人工関節あり": "knee implant",
    "側として角度を再計算しました。": " side selected; angles recalculated.",
    "件 — 必ず詳細を確認してください": " item(s) - review details",
    "確認事項：": "Review items: ",
    "確認事項（": "Review items (",
    "）・AIモデル": ") / AI model",
    "件": " item(s)",
    " と ": " and ",
    " を書き出しました。": " were exported.",
    "（設定の消去にも失敗しました：": " (also failed to clear the setting: ",
    "を読み込み中": " loading",
    "現在 ": "currently ",
    "側": " side",
}


# Messages supplied by the shared runtime are translated only at presentation
# boundaries.  The runtime itself remains untouched, preserving Japanese type
# tokens used by automatic Bone/TKA routing.
RUNTIME_TEXT_REPLACEMENTS = {
    "ファイル名から左右を判定できません。解析前にLまたはRを選択してください。": "Laterality could not be inferred from the filename. Select L or R before analysis.",
    "点3（大腿骨側中央点）が点2と点4の水平方向の間にありません。mLDFAとHKAを計算する前に位置を確認してください。": "Point 3 is not horizontally between points 2 and 4. Review it before using mLDFA or HKA.",
    "点6（脛骨側中央点）が点5と点7の水平方向の間にありません。MPTAとHKAを計算する前に位置を確認してください。": "Point 6 is not horizontally between points 5 and 7. Review it before using MPTA or HKA.",
    "点1・点3・点6・点8の上下方向の解剖学的順序が不自然です。位置を確認・修正してください。": "The vertical anatomical order of points 1, 3, 6, and 8 is unusual. Review and correct their positions.",
    "大腿骨の機械軸を定義する2点が近すぎるため、角度を正しく計算できません。": "The two femoral mechanical-axis points are too close for reliable angle calculation.",
    "脛骨の機械軸を定義する2点が近すぎるため、角度を正しく計算できません。": "The two tibial mechanical-axis points are too close for reliable angle calculation.",
    "このAIモデルには検証指標が含まれていないため、検証時の誤差を表示できません。": "This AI model has no validation metrics, so validation error cannot be displayed.",
    "このAIモデルは検証時の角度誤差が大きいため、すべての結果を必ず医師が確認してください。": "This AI model had high validation angle error. A doctor must review every result.",
    "このモデルファイルは旧形式で、前処理およびモデル構造の情報が明示されていません。互換設定で読み込みました。": "This is a legacy model without explicit preprocessing and architecture metadata. It was loaded in compatibility mode.",
    "AIスコアが低い、または無効です。次のランドマークを優先して確認してください：": "AI scores are low or invalid. Review these landmarks first: ",
    "mLDFAの平均絶対誤差": "mLDFA mean absolute error",
    "MPTAの平均絶対誤差": "MPTA mean absolute error",
    "大腿骨関節線端点1": "Femoral joint-line endpoint 1",
    "大腿骨関節線端点2": "Femoral joint-line endpoint 2",
    "脛骨関節線端点1": "Tibial joint-line endpoint 1",
    "脛骨関節線端点2": "Tibial joint-line endpoint 2",
    "大腿骨関節線": "Femoral joint line",
    "脛骨関節線": "Tibial joint line",
    "点1・股関節中心": "Point 1 - hip center",
    "点2・大腿骨関節線点A": "Point 2 - femoral joint-line point A",
    "点3・大腿骨関節線中央点": "Point 3 - femoral joint-line center",
    "点4・大腿骨関節線点B": "Point 4 - femoral joint-line point B",
    "点5・脛骨関節線点A": "Point 5 - tibial joint-line point A",
    "点6・脛骨関節線中央点": "Point 6 - tibial joint-line center",
    "点7・脛骨関節線点B": "Point 7 - tibial joint-line point B",
    "点8・足関節中心": "Point 8 - ankle center",
    "ランドマーク座標と表示中の角度が一致しないため、書き出しを中止しました。": "Export stopped because landmark coordinates do not match the displayed angles.",
    "解析中に元画像が変更されました。画像を開き直してください。": "The source image changed during analysis. Reopen the image.",
    "画像範囲外のランドマークがあります：": "Landmarks outside the image: ",
    "の2端点が近すぎるため、角度が不正確な可能性があります。": " endpoints are too close; angles may be inaccurate.",
    "広めに設定した技術的チェック範囲": "the broad technical check range",
    "° は、広めに設定した技術的チェック範囲（45～135°）を外れています。": " deg is outside the broad technical check range (45-135 deg).",
    "° は、広めに設定した技術的チェック範囲（30°以内）を外れています。": " deg is outside the broad technical check range (within 30 deg).",
    "° は、広めに設定した技術的チェック範囲（45°以内）を外れています。": " deg is outside the broad technical check range (within 45 deg).",
    "を外れています。": "is outside.",
    "AIモデルが読み込まれていません。": "No AI model is loaded.",
    "解析前にAIモデルを読み込んでください。": "Load an AI model before analysis.",
    "AI解析に失敗しました：": "AI analysis failed: ",
    "画像を読み込めません：": "Could not read image: ",
    "画像が見つかりません：": "Image not found: ",
    "AIモデルファイルが見つかりません：": "AI model file not found: ",
    "AIモデルファイルを読み込めません：": "Could not read AI model file: ",
    "アプリ設定ファイルを読み込めません：": "Could not read application configuration: ",
    "アプリ設定ファイルのJSON形式が正しくありません：": "Application configuration is not valid JSON: ",
    "アプリに必要な推論ライブラリが含まれていません。管理者にお問い合わせください。": "The application is missing a required inference library. Contact the administrator.",
    "AIモデルのパラメータに互換性がありません": "AI model parameters are incompatible",
    "AIモデルファイルの形式が正しくありません。": "The AI model file format is invalid.",
    "ランドマーク座標がない、または無効です：": "Landmark coordinates are missing or invalid: ",
    "関節線端点の座標がない、または無効です：": "Joint-line endpoint coordinates are missing or invalid: ",
    "ランドマーク定義に互換性がありません。": "Landmark definitions are incompatible.",
    "モデル自己テスト": "Model self-test",
    "膝関節ランドマーク推定モデル": "Knee landmark model",
    "対象データ未指定（外部モデル）": "Target cohort not specified (external model)",
    "人工関節なし": "no knee implant",
    "人工関節あり": "knee implant",
    "片側下肢": "single-leg full-length",
    "未対応": "Unsupported",
    "正しくありません": "is invalid",
    "必要です": "is required",
}


POINT_LABEL_REPLACEMENT = """POINT_LABELS = {
    \"hip\": \"Point 1 - Hip center\",
    \"upper_left\": \"Point 2 - Femoral joint-line point A\",
    \"upper_center\": \"Point 3 - Femoral joint-line center\",
    \"upper_right\": \"Point 4 - Femoral joint-line point B\",
    \"lower_left\": \"Point 5 - Tibial joint-line point A\",
    \"lower_center\": \"Point 6 - Tibial joint-line center\",
    \"lower_right\": \"Point 7 - Tibial joint-line point B\",
    \"ankle\": \"Point 8 - Ankle center\",
}"""


def _runtime_translation_block() -> str:
    pairs = ",\n".join(
        f"    ({source!a}, {target!a})"
        for source, target in sorted(
            RUNTIME_TEXT_REPLACEMENTS.items(), key=lambda item: (-len(item[0]), item[0])
        )
    )
    return f'''\n\nWINDOWS_ENGLISH_BUILD = True
_WINDOWS_RUNTIME_TEXT_REPLACEMENTS = (\n{pairs}\n)


def windows_english_text(value: object) -> str:
    text = str(value)
    for source, target in _WINDOWS_RUNTIME_TEXT_REPLACEMENTS:
        text = text.replace(source, target)
    for source, target in (("\\uFF08", " ("), ("\\uFF09", ")"), ("\\u30FB", " / "), ("\\uFF1A", ": "), ("\\u3001", ", "), ("\\u3002", "."), ("\\uFF5E", "-"), ("\\u00B0", " deg")):
        text = text.replace(source, target)
    if re.search(r"[\\u3040-\\u30ff\\u3400-\\u9fff]", text):
        detail = text.encode("unicode_escape", errors="backslashreplace").decode("ascii")
        return "A runtime message could not be fully localized. Review the image, landmarks, laterality, and model. Details: " + detail
    return text
'''


def generate(source_path: Path, output_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    if "WINDOWS_ENGLISH_BUILD = True" in source:
        raise RuntimeError("source already appears to be a generated Windows entrypoint")

    translated = source
    for original, replacement in sorted(
        TRANSLATIONS.items(), key=lambda item: (-len(item[0]), item[0])
    ):
        translated = translated.replace(original, replacement)
        if "\n" in original:
            translated = translated.replace(
                original.replace("\n", "\\n"),
                replacement.replace("\n", "\\n"),
            )

    point_label_source = (
        "POINT_LABELS = {name: coordinate_display_name(name) "
        "for name in ANNOTATION_POINT_NAMES}"
    )
    if translated.count(point_label_source) != 1:
        raise RuntimeError("canonical POINT_LABELS expression changed; update the generator")
    translated = translated.replace(point_label_source, POINT_LABEL_REPLACEMENT)

    marker = '\n\nAPP_VERSION = '
    if translated.count(marker) != 1:
        raise RuntimeError("canonical APP_VERSION marker changed; update the generator")
    translated = translated.replace(marker, _runtime_translation_block() + marker, 1)

    # Translate shared-runtime exceptions and warnings at UI boundaries without
    # changing the shared inference or measurement modules.
    translated = translated.replace("str(exc)", "windows_english_text(exc)")
    translated = translated.replace("{exc}", "{windows_english_text(exc)}")
    warning_line = '"\\n".join(f"• {message}" for message in messages)'
    translated_warning_line = (
        '"\\n".join(f"- {windows_english_text(message)}" for message in messages)'
    )
    if translated.count(warning_line) != 1:
        raise RuntimeError("canonical warning rendering changed; update the generator")
    translated = translated.replace(warning_line, translated_warning_line)
    translated = translated.replace("{info.display_name}", "{windows_english_text(info.display_name)}")
    translated = translated.replace("{info.cohort}", "{windows_english_text(info.cohort)}")

    # Normalize remaining Japanese punctuation so the generated source and
    # Windows UI are code-page friendly.  Degree symbols are converted too.
    punctuation = {
        "（": " (",
        "）": ")",
        "・": " / ",
        "／": " / ",
        "：": ": ",
        "、": ", ",
        "。": ".",
        "…": "...",
        "—": "-",
        "–": "-",
        "×": "x",
        "°": " deg",
        "⚠": "WARNING:",
        "•": "-",
        "·": "|",
        "～": "-",
        "→": "->",
        "●": "*",
    }
    for original, replacement in punctuation.items():
        translated = translated.replace(original, replacement)

    remaining = sorted(set(CJK_PATTERN.findall(translated)))
    if remaining:
        lines = [
            str(index)
            for index, line in enumerate(translated.splitlines(), start=1)
            if CJK_PATTERN.search(line)
        ]
        raise RuntimeError(
            "untranslated CJK text remains in generated entrypoint on line(s): "
            + ", ".join(lines[:20])
        )

    header = (
        "# GENERATED FILE - DO NOT EDIT.\n"
        "# Source: knee_measurement_app.py\n"
        "# Generator: generate_windows_english_entrypoint.py\n"
    )
    if translated.startswith("#!/usr/bin/env python3\n"):
        translated = "#!/usr/bin/env python3\n" + header + translated.split("\n", 1)[1]
    else:
        translated = header + translated
    if not translated.endswith("\n"):
        translated += "\n"

    output_path.write_text(translated, encoding="utf-8", newline="\n")
    py_compile.compile(str(output_path), doraise=True)
    return translated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Windows English GUI entrypoint.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.source.resolve(), args.output.resolve())
    print(f"Generated and compiled Windows English entrypoint: {args.output}")


if __name__ == "__main__":
    main()

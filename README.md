# Knee-angle-measument-system

## Final Auto-Measurement App

`knee_measurement_app.py` is the doctor-facing raw-image workflow:

1. Open one **single-leg, full-length** X-ray (`JPG`, `PNG`, `BMP`, or `TIFF`).
2. Confirm `L` or `R` when laterality cannot be inferred from the filename.
3. The configured model predicts 8 anatomical points and 2 joint-line endpoint pairs.
4. The right panel immediately shows the overlay, all 12 coordinates, and mLDFA, MPTA, JLCA, and HKA.
5. A doctor can drag any predicted handle; all measurements are recalculated after release.
6. Export writes a traceable `*_measurement.json` and `*_measurement.png` containing model hash/version and whether the coordinates were manually edited.

Run from source:

```bash
python -m pip install -r requirements-app.txt
python knee_measurement_app.py
```

Open a known non-PHI test image directly:

```bash
python knee_measurement_app.py \
  --image images/annotation_processed_combined/001L_raw.jpg \
  --side L
```

The Japanese doctor-facing quick guide is in `DOCTOR_GUIDE_JA.md`.

The v0.3.0 release bundles the three reviewed weights listed in
`models/README.md`: Bone for non-arthroplasty images, TKA for arthroplasty
images, and Mixed as the fallback. **自動判定** uses explicit tokens in the
image path/name and falls back to Mixed when the type is unknown or conflicting;
it does not diagnose an implant from image pixels. The user can always select
Bone, TKA, or Mixed manually, and the manual choice takes priority.

A compatible external weight can still be selected from **AIモデル… →
AIモデルファイルを選択…**. That selection is validated, copied to the user's
application-data folder, and persisted outside the signed app bundle;
**標準モデルに戻す** clears the external override. With `version: "auto"`,
the displayed version is taken from the checkpoint manifest.

The current `small_heatmap_v1` adapter accepts weights only when the model
architecture, preprocessing, and 12-keypoint order are compatible. A future
architecture can be added behind another adapter (or delivered as ONNX / TorchScript)
without changing the GUI contract. A new Python adapter/runtime dependency still
requires updating the PyInstaller build; only compatible `small_heatmap_v1`
weights are file-only replacements in the current package.

Important current scope limitations:

- DICOM and un-cropped bilateral X-rays are not supported yet.
- The bundled Bone, TKA, and Mixed checkpoints are research models with internal
  case-level 5-fold cross-validation only. Every landmark and angle must still
  be reviewed by a doctor, especially on bilateral or contralateral anatomy.
- This software is not a cleared medical device and must not be used as the sole
  basis for diagnosis or treatment.

Run automated checks:

```bash
python -m unittest discover -s tests -v
python knee_measurement_app.py --validate-models
```

Build the standalone inference app on the target operating system:

```bash
./build_measurement_mac.sh
```

or on Windows:

```bat
build_measurement_windows.bat
```

The canonical GUI source remains the Japanese macOS interface. The Windows
build deterministically generates a temporary English entrypoint from that same
source with `generate_windows_english_entrypoint.py`; the generated file is
compiled and checked for untranslated CJK text before PyInstaller runs. It is
ignored by Git and must not be edited or committed. This keeps model routing,
measurement, and export behavior in one implementation while avoiding Japanese
UI/code-page problems on Windows. The Windows ZIP contains
`README_DOCTOR_EN.txt`; the macOS ZIP keeps `README_DOCTOR_JA.txt`.

Builds require Python 3.10–3.12. The current release target is Apple Silicon
`arm64` on macOS 12.1 or newer; it is not a universal binary. The release config
forces CPU inference. The doctor does not need Python, a GPU, package installs,
or network access. Each build clears and sanitizes its virtual environment,
runs unit tests, validates all three configured models, bundles
Python/Tk/Torch/OpenCV, then runs real image-to-JSON/PNG CPU smoke tests for all
three models from the frozen executable in a stripped environment. The release
archive audit permits only the application, doctor README, and the exact three
approved model hashes; source data, caches, build trees, or an old `current.pt`
cause the build to fail.

The generated macOS app is ad-hoc signed for local testing. A release sent to
doctors still needs an organization Developer ID signature and notarization;
the Windows build likewise needs production code signing.

## Annotation Tool

Use the raw-image annotation tool to create stable training labels directly on the original X-ray:

```bash
python annotate_gui.py
```

The tool stores:

- `*_annotation.json`: source-of-truth labels in raw-image coordinates
- `*_point.jpg`: same-size preview with the 8 points
- `*_line.jpg`: same-size preview with the 2 joint lines
- `*_combined.jpg`: same-size measurement preview

The measurement side matters because `mLDFA` uses the lateral distal femur angle and `MPTA` uses the medial proximal tibia angle. If the filename cannot identify a single side, choose `L` or `R` in the annotation tool before previewing or saving. The JSON stores this as `side`.

## Mac Packaging

The `KneeAnnotationTool.app` at the project root is only a development launcher. It expects `annotate_gui.py`, `measure_angles.py`, Python, and Python packages to exist next to it on the same machine.

To create a portable macOS app for another user, build the PyInstaller app on macOS:

```bash
./build_mac.sh
```

The portable output will be:

```text
KneeAnnotationTool-macOS.zip
```

Send `KneeAnnotationTool-macOS.zip`, not the development launcher at the project root. This zip intentionally contains the same `dist/` folder layout as the previous working package. The standalone app writes exported annotations to:

```text
~/Documents/Knee_Xray_annotations
```

The build script removes `.DS_Store`, AppleDouble `._*` files, and `__MACOSX`
folders before zipping, then checks the zip contents. If AppleDouble metadata is
still present, the build fails so the package is not sent with files that can
break macOS signature verification after extraction.

If macOS blocks the app because it was downloaded from the internet, right-click the app and choose `Open`.

## Windows Packaging

If you want to send the tool to a doctor as a standalone Windows app, build it on a Windows machine:

```bat
build_windows.bat
```

The output will be:

```text
dist\KneeAnnotationTool\
```

Send the whole `dist\KneeAnnotationTool` folder. The doctor should launch:

```text
KneeAnnotationTool.exe
```

This is the practical rule:

- macOS `.app` should be built/tested on macOS
- Windows `.exe` should be built/tested on Windows

Do not try to build the final Windows release on macOS and assume it will be reliable.

## Packaging Dependencies

Packaging uses:

- `numpy`
- `opencv-python`
- `Pillow`
- `pyinstaller`

Install them with:

```bash
pip install -r requirements-packaging.txt
```

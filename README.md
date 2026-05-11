# Knee-angle-measument-system

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
dist/KneeAnnotationTool.app
```

Send `dist/KneeAnnotationTool.app`, not the development launcher at the project root. The standalone app writes exported annotations to:

```text
~/Documents/Knee_Xray_annotations
```

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

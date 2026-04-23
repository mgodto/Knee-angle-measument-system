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

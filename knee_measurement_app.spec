# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


project_root = Path(SPECPATH).resolve()
target_arch = os.environ.get("KNEE_TARGET_ARCH") or None
entry_script_name = (
    "knee_measurement_app_windows.py"
    if sys.platform == "win32"
    else "knee_measurement_app.py"
)
entry_script = project_root / entry_script_name
config_path = project_root / "knee_measurement_app.json"
checkpoint_paths = tuple(
    project_root / "models" / filename
    for filename in ("bone.pt", "tka.pt", "mixed.pt")
)

if not entry_script.is_file():
    raise FileNotFoundError(
        f"Missing build entrypoint: {entry_script}. "
        "On Windows, run generate_windows_english_entrypoint.py first."
    )
if not config_path.is_file():
    raise FileNotFoundError(f"Missing app configuration: {config_path}")
missing_checkpoints = [str(path) for path in checkpoint_paths if not path.is_file()]
if missing_checkpoints:
    raise FileNotFoundError(
        "Missing required bundled checkpoint(s): " + ", ".join(missing_checkpoints)
    )

hiddenimports = ["knee_keypoint_model"]
datas = [
    (str(config_path), "."),
    *((str(checkpoint_path), "models") for checkpoint_path in checkpoint_paths),
] + copy_metadata("torch")

a = Analysis(
    [str(entry_script)],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "matplotlib",
        "pandas",
        "scipy",
        "torchvision",
        "torchaudio",
        "jax",
        "jaxlib",
        "tensorflow",
        "sentencepiece",
        "transformers",
        "lxml",
        "cryptography",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KneeXrayMeasurement",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="KneeXrayMeasurement",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="KneeXrayMeasurement.app",
        icon=None,
        bundle_identifier="com.kneexray.measurement",
        info_plist={
            "CFBundleName": "下肢全長X線自動計測",
            "CFBundleDisplayName": "下肢全長X線 自動計測",
            "CFBundleShortVersionString": "0.6.0",
            "CFBundleVersion": "9",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.1",
        },
    )

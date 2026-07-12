# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


project_root = Path(SPECPATH).resolve()
target_arch = os.environ.get("KNEE_TARGET_ARCH") or None
config_path = project_root / "knee_measurement_app.json"
checkpoint_path = project_root / "models" / "current.pt"

if not config_path.is_file():
    raise FileNotFoundError(f"Missing app configuration: {config_path}")
if not checkpoint_path.is_file():
    raise FileNotFoundError(
        "Missing default checkpoint. Put the production-compatible weight at "
        f"{checkpoint_path} before building."
    )

hiddenimports = ["knee_keypoint_model"]
datas = [
    (str(config_path), "."),
    (str(checkpoint_path), "models"),
] + copy_metadata("torch")

a = Analysis(
    ["knee_measurement_app.py"],
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
            "CFBundleShortVersionString": "0.2.3",
            "CFBundleVersion": "5",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.1",
        },
    )

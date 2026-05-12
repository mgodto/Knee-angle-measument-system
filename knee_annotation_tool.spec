# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = Path(SPECPATH).resolve()
hiddenimports = collect_submodules("PIL") + collect_submodules("cv2")

a = Analysis(
    ["annotate_gui.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KneeAnnotationTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="KneeAnnotationTool",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="KneeAnnotationTool.app",
        icon=None,
        bundle_identifier="com.kneexray.annotationtool",
        info_plist={
            "CFBundleName": "KneeAnnotationTool",
            "CFBundleDisplayName": "KneeAnnotationTool",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )

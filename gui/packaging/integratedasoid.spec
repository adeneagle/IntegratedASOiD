# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the IntegratedASOiD desktop GUI (gui/app.py).

Build from the repo root, in the GUI venv with PyInstaller installed:

    .venv-gui/Scripts/pip install pyinstaller          # Windows
    .venv-gui/Scripts/pyinstaller gui/packaging/integratedasoid.spec

Output: dist/IntegratedASOiD/ (onedir). Run PyInstaller on each target OS
separately — a build is not cross-compiling.

Notes:
  * The backend (FastAPI + scikit-learn) is NOT bundled. The app spawns a
    local backend from the project's pixi env, or connects to a remote one;
    either way sklearn/numba are excluded here to keep the GUI bundle small.
  * PyAV needs ffmpeg shared libs -> collect_dynamic_libs("av").
"""

import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

GUI_DIR = os.path.dirname(SPECPATH)          # .../gui
ENTRY = os.path.join(GUI_DIR, "app.py")

hiddenimports = collect_submodules("labeling_shell") + collect_submodules("plugins")
binaries = collect_dynamic_libs("av")        # ffmpeg libs PyAV loads at runtime

a = Analysis(
    [ENTRY],
    pathex=[GUI_DIR],                        # so labeling_shell/plugins resolve
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["sklearn", "numba", "torch", "torchvision", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IntegratedASOiD",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="IntegratedASOiD")

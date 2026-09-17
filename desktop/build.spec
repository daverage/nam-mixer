# PyInstaller spec for the NAM Mixer standalone desktop app.
#
# Produces a --onedir bundle (not --onefile): the nam_render native binary
# and assets/templates/static need to sit as plain files next to the
# executable, which --onedir gives for free and --onefile would otherwise
# force an unpack-to-tempdir step for on every launch.
#
# Usage (run from repo root, after building/downloading nam_render -- see
# docs/standalone_packaging.md):
#   pyinstaller desktop/build.spec --noconfirm
#
# Output: dist/NAMMixer/ (Windows/Linux) or dist/NAMMixer.app (macOS).

import sys
from pathlib import Path

block_cipher = None
REPO_ROOT = Path(SPECPATH).resolve().parent

nam_render_exe = "nam_render.exe" if sys.platform.startswith("win") else "nam_render"
nam_render_src = REPO_ROOT / "native" / "nam_render" / "build" / nam_render_exe
if not nam_render_src.is_file():
    # Also check the MSVC Release subfolder used on Windows.
    alt = REPO_ROOT / "native" / "nam_render" / "build" / "Release" / nam_render_exe
    if alt.is_file():
        nam_render_src = alt

if not nam_render_src.is_file():
    raise SystemExit(
        f"nam_render binary not found at {nam_render_src}. "
        "Build it (native/nam_render/README.md) or download it "
        "(scripts/download_nam_render.sh / .ps1) before packaging."
    )

datas = [
    (str(REPO_ROOT / "templates"), "templates"),
    (str(REPO_ROOT / "static"), "static"),
    (str(REPO_ROOT / "assets" / "di"), "assets/di"),
    (str(REPO_ROOT / "assets" / "training"), "assets/training"),
    # These are source resources for the optional external-Python training
    # process, not imports for the render-only frozen interpreter.
    (str(REPO_ROOT / "hybrid"), "training_runtime/hybrid"),
    (str(REPO_ROOT / "scripts" / "train_a2.py"), "training_runtime/scripts"),
    (str(REPO_ROOT / "requirements-training.txt"), "training_runtime"),
    (str(REPO_ROOT / "cloud" / "kaggle" / "train_a2_cloud.py"), "training_runtime/cloud/kaggle"),
]
binaries = [
    (str(nam_render_src), "nam_render"),
]

# Icons rendered from static/nam-mixer-logo.png -- see desktop/icons/README.md
# for how to regenerate them if that source logo ever changes.
if sys.platform == "darwin":
    icon_path = REPO_ROOT / "desktop" / "icons" / "icon.icns"
elif sys.platform.startswith("win"):
    icon_path = REPO_ROOT / "desktop" / "icons" / "icon.ico"
else:
    icon_path = REPO_ROOT / "desktop" / "icons" / "icon.png"
icon_arg = str(icon_path) if icon_path.is_file() else None

a = Analysis(
    [str(REPO_ROOT / "desktop" / "main.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["hybrid", "app"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchaudio"],  # never bundle the training-only env, see CLAUDE.md
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NAMMixer",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon_arg,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="NAMMixer",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="NAMMixer.app",
        icon=icon_arg,
        bundle_identifier="com.nammixer.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )

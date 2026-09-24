# PyInstaller spec for the desktop shell's bundled backend -- this is what
# removes the one real blocker from the desktop readiness review: the
# packaged .app previously shelled out to a bare `python3` on PATH, which a
# real end user does not have set up with flask/numpy/scipy/soundfile.
#
# This backend's own RUNTIME (the frozen exe/pyz) mirrors requirements.txt
# ONLY -- never requirements-training.txt; Torch/neural-amp-modeler are never
# installed into it. Local A2 training creates its OWN separate, dedicated
# venv (hybrid/training/local_training.py's LocalTrainingManager) that this frozen
# process merely launches as a subprocess -- see CLAUDE.md's "Kaggle GPU
# training backend" note and the desktop-architecture note for why a frozen
# build must never try to relaunch itself as a generic interpreter. That
# subprocess still needs real loose .py source + requirements-training.txt
# on disk to run against, though -- see the "training_support" datas below.
#
# Build (from repo root):
#   .venv/bin/pyinstaller packaging/backend/nam_mixer_backend.spec --distpath packaging/backend/dist --workpath packaging/backend/build --noconfirm
#
# Output: packaging/backend/dist/nam-mixer-backend/ (onedir -- NOT onefile;
# onefile re-extracts to a fresh temp dir on every launch, which would
# break the DI_DIR/nam_render relative-path lookups bundled below as datas/
# binaries. WORK_DIR itself is redirected to a real writable, persistent
# location via NAM_MIXER_DATA_DIR -- set by the Tauri shell, never left at
# this read-only bundle's own default).
from pathlib import Path

REPO_ROOT = Path.cwd()

datas = [
    (str(REPO_ROOT / "templates"), "templates"),
    (str(REPO_ROOT / "static"), "static"),
    (str(REPO_ROOT / "assets" / "di"), "assets/di"),
    # assets/nam_models is deliberately NOT bundled: it holds the developer's
    # own (gitignored) personal captures, and nothing reads it at runtime.
    # The official NAM v3.0.0 training input (assets/training/README.md) --
    # app.py auto-seeds work/training_input/input.wav from this on first
    # run so training works without a manual upload. Omitting it here
    # doesn't break anything visibly (the app just falls back to asking
    # for a manual upload) but silently loses that zero-setup behavior --
    # exactly what happened here before this fix.
    (str(REPO_ROOT / "assets" / "training"), "assets/training"),
    # Local A2 training runs scripts/train_a2.py in its OWN separate,
    # dedicated venv (.venv-a2) -- a real, unfrozen Python environment, not
    # this app's own packed/compiled modules. That subprocess needs actual
    # loose .py source files and requirements-training.txt to exist
    # somewhere on disk to import/install from; nothing did that for the
    # desktop build before this, so "Set up local training" always failed
    # with "No such file or directory: requirements-training.txt" once it
    # got past finding a real Python. main.rs points
    # NAM_MIXER_TRAINING_ROOT at this exact folder.
    (str(REPO_ROOT / "hybrid"), "training_support/hybrid"),
    (str(REPO_ROOT / "scripts" / "train_a2.py"), "training_support/scripts"),
    (str(REPO_ROOT / "requirements-training.txt"), "training_support"),
    # Kaggle GPU training stages this worker script into the kernel; app.py
    # looks for it at TRAINING_ROOT/cloud/kaggle/train_a2_cloud.py.
    (str(REPO_ROOT / "cloud" / "kaggle" / "train_a2_cloud.py"), "training_support/cloud/kaggle"),
]

binaries = []
nam_render_exe = REPO_ROOT / "native" / "nam_render" / "build" / "nam_render"
if nam_render_exe.is_file():
    binaries.append((str(nam_render_exe), "native/nam_render/build"))
    # hybrid.core.render.find_nam_render_exe() looks for this relative to
    # wherever ITS OWN hybrid/ copy sits -- training_support/hybrid/core/render.py
    # resolves that to training_support/native/nam_render/build/, a
    # separate lookup from the main app's copy above.
    binaries.append((str(nam_render_exe), "training_support/native/nam_render/build"))
nam_render_exe_win = REPO_ROOT / "native" / "nam_render" / "build" / "Release" / "nam_render.exe"
if nam_render_exe_win.is_file():
    binaries.append((str(nam_render_exe_win), "native/nam_render/build/Release"))
    binaries.append((str(nam_render_exe_win), "training_support/native/nam_render/build/Release"))
if not binaries:
    # Without the renderer every render/preview/generate in the shipped app
    # fails; never produce a "successful" bundle that can't render.
    raise SystemExit("nam_render was not built (native/nam_render/build/[Release/]nam_render[.exe]) -- build it first")

a = Analysis(
    [str(REPO_ROOT / "app.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["soundfile", "scipy.signal", "scipy.ndimage", "scipy.interpolate"],
    hookspath=[],
    excludes=["torch", "pytorch_lightning", "nam", "matplotlib", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="nam-mixer-backend",
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="nam-mixer-backend",
)

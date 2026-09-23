# NAM Mixer desktop shell

A thin Tauri wrapper around the existing Flask app (`app.py`), not a
reimplementation -- see `src-tauri/src/main.rs`'s module docstring. The web
UI it shows is byte-for-byte the same as `python app.py` in a browser.

## Build order matters

The backend must be built **before** the Rust side, every time:

```bash
# 1. Build the bundled backend (PyInstaller, from repo root):
.venv/bin/pyinstaller packaging/backend/nam_mixer_backend.spec \
  --distpath packaging/backend/dist --workpath packaging/backend/build --noconfirm

# 2. Then build/run the desktop shell (from desktop/src-tauri):
cargo build            # dev binary -- also uses the bundled backend if step 1 ran
cargo tauri build      # real .app/.dmg/.exe/.AppImage
```

`tauri.conf.json`'s `bundle.resources` points at
`packaging/backend/dist/nam-mixer-backend` unconditionally -- **even a plain
`cargo build` fails outright** if that directory doesn't exist yet, because
Tauri's build script validates every declared resource path up front. Run
step 1 first on a fresh clone, or after any `app.py`/`hybrid/` change you
want the desktop build to pick up (the PyInstaller build is not
automatically re-run by `cargo build`).

If you skip step 1 entirely (delete `packaging/backend/dist/`), you'll need
to remove the `resources` entry from `tauri.conf.json` too, or `cargo
build` won't run at all.

## Why bundle instead of a Tauri sidecar

Tauri's sidecar mechanism expects one standalone executable file.
PyInstaller's `--onedir` output (used here on purpose -- see the `.spec`
file's docstring) is an executable plus a required sibling `_internal/`
folder, which doesn't fit that model. Bundling the whole folder as a Tauri
**resource** and spawning the executable inside it directly (see
`bundled_backend_exe()` in `main.rs`) sidesteps that mismatch.

## What's NOT bundled

Local A2 training (`hybrid/training/local_training.py`) creates its own separate,
dedicated Python venv (`.venv-a2`, with Torch) completely independently of
whatever runs this Flask process -- see `CLAUDE.md`. Bundling this desktop
backend never touches that; a user who wants local training still needs to
click "Set up local training" once, same as on the web version.

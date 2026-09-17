# Standalone desktop packaging

Hybrid NAM Builder normally runs as `python app.py` + a browser tab. The
`desktop/` directory adds an alternative: a double-clickable native app for
Windows, macOS, and Linux, with no Python install, `pip install`, or
manually-built `nam_render` step required by the end user.

## How it works

- **`desktop/main.py`** is the entry point PyInstaller freezes. It starts
  the existing Flask app (`app.py`, unchanged) on a local loopback port in a
  background thread, then opens a native OS window over it via
  [`pywebview`](https://pywebview.flet.dev/) (the OS's own webview --
  WebView2 on Windows, WKWebView on macOS, WebKitGTK on Linux -- not a
  bundled Chromium, which keeps the app small).
- **`desktop/build.spec`** is the PyInstaller spec: `--onedir` (not
  `--onefile`), because the bundled `nam_render` binary and the
  templates/static/assets need to sit as plain files next to the executable
  rather than unpacked to a temp dir on every launch. It explicitly excludes
  `torch`/`torchaudio` -- this app is render/design/preview only; training a
  bundle into a `.nam` still requires the separate environment described in
  the top-level CLAUDE.md and `requirements-training.txt`.
- **`hybrid/render.py`'s existing `NAM_RENDER_EXE` env var** is how the
  desktop launcher points the app at its bundled `nam_render` binary --
  no changes to `render.py` itself were needed.
- **`.github/workflows/build-desktop.yml`** builds `nam_render` from source
  and packages the app on all three OSes on every relevant push/PR, and
  attaches release assets on a `desktop-v*` tag (mirrors the existing
  `build-nam-render.yml` pattern for the CLI-only binary).

## Building locally

```bash
pip install -r requirements-desktop.txt

# Build (or download -- see native/nam_render/README.md /
# scripts/download_nam_render.sh) nam_render first:
cmake -B native/nam_render/build -S native/nam_render -DCMAKE_BUILD_TYPE=Release
cmake --build native/nam_render/build --config Release --target nam_render

pyinstaller desktop/build.spec --noconfirm
```

Output: `dist/HybridNAMBuilder/` (Windows/Linux) or
`dist/HybridNAMBuilder.app` (macOS). PyInstaller does not cross-compile --
build on each target OS, or rely on the CI matrix.

Running unfrozen for development (no packaging needed):

```bash
python desktop/main.py
```

## Settings page

A packaged app has no shell to `export` an environment variable into, but
several genuinely user-facing knobs are only ever read from the environment
today (`NAM_RENDER_EXE`, the optional local-AI assistant's endpoint/model,
etc.). The **Settings** tab in the UI (`/api/settings` GET/POST,
`hybrid/settings.py`) exposes exactly the same env vars the app has always
read -- it's not a new configuration surface, just a UI for the existing
one.

Values are persisted to a `.env` file via `hybrid/env_file.py` (already used
by the local-AI assistant settings) and applied to the running process's
`os.environ` immediately, so most changes take effect without a restart --
the registry in `hybrid/settings.py` flags the exceptions
(`restart_required=True`, e.g. the server `PORT`, which is only read once at
`app.run()`).

For the packaged desktop app specifically, `desktop/main.py` points
`NAM_MIXER_ENV_FILE` at a per-user config directory instead of the repo
root (which is commonly read-only once installed, e.g. under
`/Applications` or `Program Files`):

| OS      | Settings file location                                      |
|---------|--------------------------------------------------------------|
| Windows | `%APPDATA%\HybridNAMBuilder\.env`                             |
| macOS   | `~/Library/Application Support/HybridNAMBuilder/.env`         |
| Linux   | `$XDG_CONFIG_HOME/hybrid-nam-builder/.env` (default `~/.config/...`) |

Running unfrozen (`python app.py` or `python desktop/main.py` from a git
checkout) keeps using the repo-root `.env`, unchanged from before.

## What's intentionally out of scope

- **Training** (`scripts/train_a2.py`, the Kaggle GPU backend) is not
  bundled and is not expected to run from the packaged app -- it needs a
  separate torch environment per the existing project design (see
  CLAUDE.md). The packaged app covers render/preview/design/generate
  (everything Flask already serves), not local GPU training.
- **Code signing/notarization** is not set up. Unsigned builds will trigger
  Gatekeeper/SmartScreen warnings on macOS/Windows; users can still run them
  (right-click > Open on macOS, "More info > Run anyway" on Windows), but a
  production release aimed at less technical users should add signing
  before distributing built artifacts widely.

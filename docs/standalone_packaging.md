# Standalone desktop packaging

NAM Mixer normally runs as `python app.py` + a browser tab. The
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
  `torch`/`torchaudio`, but bundles the small training scripts and
  `requirements-training.txt` needed to create a separate local A2
  environment. The player still needs an installed Python 3 interpreter for
  that optional step; heavyweight training dependencies are never installed
  into the app bundle.
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

Output: `dist/NAMMixer/` (Windows/Linux) or
`dist/NAMMixer.app` (macOS). PyInstaller does not cross-compile --
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
`app.run()`). On a later packaged-app launch, values stored in this settings
file take precedence over matching values inherited from the launcher. This
avoids a stale terminal/IDE variable (especially `TONE3000_API_KEY`) masking
the value saved in the Settings page.

For the packaged desktop app specifically, `desktop/main.py` points
`NAM_MIXER_ENV_FILE` at a per-user config directory instead of the repo
root (which is commonly read-only once installed, e.g. under
`/Applications` or `Program Files`):

| OS      | Settings file location                                      |
|---------|--------------------------------------------------------------|
| Windows | `%APPDATA%\NAMMixer\.env`                                     |
| macOS   | `~/Library/Application Support/NAMMixer/.env`                 |
| Linux   | `$XDG_CONFIG_HOME/nam-mixer/.env` (default `~/.config/...`)   |

Running unfrozen (`python app.py` or `python desktop/main.py` from a git
checkout) keeps using the repo-root `.env`, unchanged from before.

The packaged app also stores uploads, sessions, generated bundles, local A2
environments, and downloaded/generated outputs below that same per-user
directory's `data/` folder. Nothing mutable is written into the installed app
bundle. For optional local A2 training, it finds `python3`/`python` on the
system (or honours an explicit `NAM_MIXER_TRAINING_PYTHON` executable path),
then creates the dedicated environment inside that data folder.

## What's intentionally out of scope

- **A2 dependencies** (`torch`, `neural-amp-modeler`) are not bundled. Local
  training remains opt-in and installs them into a separate per-user virtual
  environment after the player requests setup; Kaggle training uses the
  player's own Kaggle CLI/account.
- **Code signing/notarization** is not set up. Unsigned builds will trigger
  Gatekeeper/SmartScreen warnings on macOS/Windows; users can still run them
  (right-click > Open on macOS, "More info > Run anyway" on Windows), but a
  production release aimed at less technical users should add signing
  before distributing built artifacts widely.

"""Standalone desktop entry point for NAM Mixer.

Wraps the existing Flask app (app.py) in a native OS window via `pywebview`,
so the whole tool runs as a single double-clickable app on Windows/macOS/
Linux with no separate browser tab, `pip install`, or manually-built
`nam_render` step required by the end user.

This module is the thing PyInstaller freezes (see desktop/build.spec) -- it
never changes hybrid/ or app.py's behavior when run the normal way
(`python app.py`); it only adds a launcher on top.

Responsibilities:
  1. Locate the bundled `nam_render` binary next to this executable (when
     frozen) and point `hybrid.render.find_nam_render_exe()` at it via the
     `NAM_RENDER_EXE` env var it already supports -- see hybrid/render.py.
  2. Run the Flask app on a local loopback port in a background thread.
  3. Open a native webview window pointed at that port. Closing the window
     shuts the whole process down (no orphaned server process).
"""
from __future__ import annotations

import os
import socket
import sys
import threading
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _run_kaggle_cli_and_exit(args: list[str]) -> "int":
    """Re-exec target for hybrid.kaggle_training.KaggleCli's frozen-build
    fallback: `[sys.executable, "--run-kaggle-cli", *args]` instead of the
    normal-interpreter `[sys.executable, "-m", "kaggle", *args]`, since a
    frozen app's own executable has no `-m` support. Dispatched here,
    before any Flask/webview import, so this process becomes a plain
    one-shot `kaggle` CLI call and exits -- never opens a second app window.
    desktop/build.spec bundles the `kaggle` package via collect_all("kaggle")
    specifically so this import succeeds in the frozen build.
    """
    from kaggle.cli import main as kaggle_main

    sys.argv = ["kaggle", *args]
    try:
        kaggle_main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _bundle_dir() -> Path:
    """Directory containing bundled resources (nam_render binary, etc.).

    PyInstaller (--onedir) exposes this as the folder containing the
    executable; sys._MEIPASS is only for the temp extraction dir used by
    --onefile mode, which is not what desktop/build.spec targets, but is
    supported here too for completeness.
    """
    if _is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        executable_dir = Path(sys.executable).resolve().parent
        # A macOS .app stores PyInstaller data resources in
        # Contents/Resources, rather than alongside the executable in
        # Contents/MacOS. Windows/Linux --onedir builds keep the latter.
        mac_resources = executable_dir.parent / "Resources"
        candidates = ([Path(meipass)] if meipass else []) + [executable_dir, mac_resources]
        # `_MEIPASS` is not consistently the data-resource directory across
        # PyInstaller's onedir/onefile/macOS bundle layouts. Prefer the first
        # candidate that actually contains one of our bundled resources.
        for candidate in candidates:
            if (candidate / "nam_render").is_dir() or (candidate / "training_runtime").is_dir():
                return candidate
        return candidates[0]
    return Path(__file__).resolve().parent.parent


def _nam_render_binary_name() -> str:
    if sys.platform.startswith("win"):
        return "nam_render.exe"
    return "nam_render"


def _configure_nam_render_exe() -> None:
    """Point NAM_RENDER_EXE at the bundled binary, if present and not already set."""
    if os.environ.get("NAM_RENDER_EXE"):
        return
    candidate = _bundle_dir() / "nam_render" / _nam_render_binary_name()
    if candidate.is_file():
        os.environ["NAM_RENDER_EXE"] = str(candidate)


def _user_config_dir() -> Path:
    """Per-user, writable config directory for the Settings page's .env file.

    A frozen app's own install directory (e.g. Program Files, /Applications)
    is commonly read-only, unlike the git checkout `python app.py` runs
    from -- see hybrid/env_file.py's NAM_MIXER_ENV_FILE override.
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "NAMMixer"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        return base / "NAMMixer"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
        return base / "nam-mixer"


def _configure_env_file() -> None:
    if os.environ.get("NAM_MIXER_ENV_FILE"):
        return
    if not _is_frozen():
        return  # dev checkout: keep using the repo-root .env as before
    config_dir = _user_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NAM_MIXER_ENV_FILE"] = str(config_dir / ".env")


def _configure_runtime_paths() -> None:
    """Keep mutable desktop data outside the read-only application bundle."""
    if not _is_frozen():
        return
    config_dir = _user_config_dir()
    data_dir = config_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NAM_MIXER_DATA_DIR", str(data_dir))

    # The dedicated training sources are bundled as plain files because the
    # external Python used for A2 training cannot import PyInstaller's PYZ.
    training_root = _bundle_dir() / "training_runtime"
    if training_root.is_dir():
        os.environ.setdefault("NAM_MIXER_TRAINING_ROOT", str(training_root))


def _load_saved_desktop_settings() -> None:
    """Make saved frozen-app settings win over inherited launch variables.

    A desktop app may be started from a terminal, Finder, an IDE, or a
    launcher.  Its parent environment is therefore not a reliable source of
    user preferences; in particular a stale TONE3000 key can otherwise mask
    the key saved in the Settings page.  Only values actually present in the
    per-user settings file replace inherited values.  Unconfigured settings
    remain available as normal environment-variable overrides.
    """
    from hybrid.env_file import read_saved_env_values
    from hybrid.settings import SETTINGS

    saved = read_saved_env_values({field.name for field in SETTINGS})
    os.environ.update(saved)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_flask_app(flask_app, port: int) -> None:
    """Run `flask_app` on 127.0.0.1:`port` -- pulled out of main() so this
    exact call (and its load_dotenv=False) is independently testable; see
    tests/test_desktop_main.py's regression test for why that flag matters.
    """
    # load_dotenv=False: Flask's app.run() defaults to auto-loading a .env
    # from the current working directory (see flask.cli.load_dotenv), which
    # would silently bypass the NAM_MIXER_ENV_FILE isolation
    # _configure_env_file() just set up -- e.g. a real secret from a
    # developer's repo-root .env leaking into the packaged app if it's ever
    # launched with that directory as its CWD. This app's own env
    # resolution (hybrid/env_file.py) is deliberate and sufficient on its
    # own; Flask's independent auto-load must stay off here.
    flask_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, load_dotenv=False)


class DesktopApi:
    """Exposed to the page as `window.pywebview.api` (via `js_api=` below).

    pywebview's native OS webview does not honor `<a download>` the way a
    real browser does -- clicking one just navigates/opens the content
    inline instead of prompting to save. static/app.js detects
    `window.pywebview` and routes every download (NAM Tools exports,
    session files, TONE3000 downloads, the AI Assistant's Markdown export)
    through `save_file` instead, so "download" in the desktop app always
    means an actual native Save dialog, matching the plain-browser behavior.
    """

    def save_file(self, filename: str, data_base64: str) -> "str | None":
        import base64

        import webview

        window = webview.windows[0]
        result = window.create_file_dialog(webview.SAVE_DIALOG, save_filename=filename)
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        with open(path, "wb") as f:
            f.write(base64.b64decode(data_base64))
        return str(path)


def main() -> int:
    _configure_env_file()
    _configure_runtime_paths()

    # Import app.py lazily, after sys.path includes the repo root when
    # running unfrozen from desktop/, and before configuring NAM_RENDER_EXE
    # so hybrid.settings' saved values (from a previous Settings-page save,
    # persisted to the .env file _configure_env_file just pointed at) load
    # into os.environ before the bundled-binary fallback below runs.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    _load_saved_desktop_settings()

    _configure_nam_render_exe()

    import app as flask_app_module  # noqa: E402

    flask_app = flask_app_module.app
    port = _free_port()

    server_thread = threading.Thread(target=lambda: run_flask_app(flask_app, port), daemon=True)
    server_thread.start()

    import webview  # deferred: only needed for the desktop launcher, not the plain Flask app

    window = webview.create_window(
        "NAM Mixer",
        f"http://127.0.0.1:{port}/",
        width=1400,
        height=900,
        min_size=(900, 600),
        js_api=DesktopApi(),
    )
    webview.start()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-kaggle-cli":
        raise SystemExit(_run_kaggle_cli_and_exit(sys.argv[2:]))
    raise SystemExit(main())

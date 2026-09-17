"""Standalone desktop entry point for Hybrid NAM Builder.

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


def _bundle_dir() -> Path:
    """Directory containing bundled resources (nam_render binary, etc.).

    PyInstaller (--onedir) exposes this as the folder containing the
    executable; sys._MEIPASS is only for the temp extraction dir used by
    --onefile mode, which is not what desktop/build.spec targets, but is
    supported here too for completeness.
    """
    if _is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
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
        return Path(base) / "HybridNAMBuilder"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "HybridNAMBuilder"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "hybrid-nam-builder"


def _configure_env_file() -> None:
    if os.environ.get("NAM_MIXER_ENV_FILE"):
        return
    if not _is_frozen():
        return  # dev checkout: keep using the repo-root .env as before
    config_dir = _user_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NAM_MIXER_ENV_FILE"] = str(config_dir / ".env")


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

    # Import app.py lazily, after sys.path includes the repo root when
    # running unfrozen from desktop/, and before configuring NAM_RENDER_EXE
    # so hybrid.settings' saved values (from a previous Settings-page save,
    # persisted to the .env file _configure_env_file just pointed at) load
    # into os.environ before the bundled-binary fallback below runs.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hybrid.settings import get_settings as _get_settings
    for field in _get_settings():
        if field["value"]:
            os.environ.setdefault(field["name"], field["value"])

    _configure_nam_render_exe()

    import app as flask_app_module  # noqa: E402

    flask_app = flask_app_module.app
    port = _free_port()

    server_thread = threading.Thread(target=lambda: run_flask_app(flask_app, port), daemon=True)
    server_thread.start()

    import webview  # deferred: only needed for the desktop launcher, not the plain Flask app

    window = webview.create_window(
        "Hybrid NAM Builder",
        f"http://127.0.0.1:{port}/",
        width=1400,
        height=900,
        min_size=(900, 600),
        js_api=DesktopApi(),
    )
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

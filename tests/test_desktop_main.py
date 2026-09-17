"""Tests for desktop/main.py -- the standalone desktop app's launcher.

desktop/ isn't a package (no __init__.py, imported by absolute path), so
these load it via importlib, the same way tests/test_a2_training_settings.py
loads the self-contained cloud worker module.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_desktop_main():
    path = REPO_ROOT / "desktop" / "main.py"
    spec = importlib.util.spec_from_file_location("desktop_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_flask_app_disables_flask_auto_dotenv_loading():
    """Regression test for a real leak: Flask's app.run() defaults to
    load_dotenv=True, which auto-loads a .env from the CURRENT WORKING
    DIRECTORY (flask.cli.load_dotenv) independently of and prior to this
    app's own NAM_MIXER_ENV_FILE-aware resolution (hybrid/env_file.py) --
    confirmed by hand: running the packaged desktop app from a checkout
    containing a real .env leaked its secrets (e.g. TONE3000_API_KEY) into
    the running process even though _configure_env_file() had correctly
    pointed NAM_MIXER_ENV_FILE at an isolated, nonexistent per-user file.
    load_dotenv=False on this call is what actually prevents that.
    """
    desktop_main = _load_desktop_main()
    fake_app = MagicMock()
    desktop_main.run_flask_app(fake_app, 5001)
    fake_app.run.assert_called_once_with(
        host="127.0.0.1", port=5001, debug=False, use_reloader=False, load_dotenv=False,
    )


def test_configure_env_file_uses_isolated_path_when_frozen(monkeypatch, tmp_path):
    desktop_main = _load_desktop_main()
    monkeypatch.delenv("NAM_MIXER_ENV_FILE", raising=False)
    monkeypatch.setattr(desktop_main.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop_main, "_user_config_dir", lambda: tmp_path / "config")

    desktop_main._configure_env_file()

    import os
    assert os.environ["NAM_MIXER_ENV_FILE"] == str(tmp_path / "config" / ".env")
    monkeypatch.delenv("NAM_MIXER_ENV_FILE", raising=False)


def test_configure_env_file_is_noop_when_not_frozen(monkeypatch):
    desktop_main = _load_desktop_main()
    monkeypatch.delenv("NAM_MIXER_ENV_FILE", raising=False)
    monkeypatch.setattr(desktop_main.sys, "frozen", False, raising=False)

    desktop_main._configure_env_file()

    import os
    assert "NAM_MIXER_ENV_FILE" not in os.environ

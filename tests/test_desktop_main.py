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


def test_configure_runtime_paths_uses_user_data_and_bundled_training_sources(monkeypatch, tmp_path):
    desktop_main = _load_desktop_main()
    monkeypatch.setattr(desktop_main.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop_main, "_user_config_dir", lambda: tmp_path / "config")
    training_root = tmp_path / "bundle" / "training_runtime"
    training_root.mkdir(parents=True)
    monkeypatch.setattr(desktop_main, "_bundle_dir", lambda: training_root.parent)
    monkeypatch.delenv("NAM_MIXER_DATA_DIR", raising=False)
    monkeypatch.delenv("NAM_MIXER_TRAINING_ROOT", raising=False)

    desktop_main._configure_runtime_paths()

    import os
    assert os.environ["NAM_MIXER_DATA_DIR"] == str(tmp_path / "config" / "data")
    assert os.environ["NAM_MIXER_TRAINING_ROOT"] == str(training_root)
    assert (tmp_path / "config" / "data").is_dir()


def test_bundle_dir_uses_resources_folder_in_a_macos_app(monkeypatch, tmp_path):
    desktop_main = _load_desktop_main()
    executable = tmp_path / "NAMMixer.app" / "Contents" / "MacOS" / "NAMMixer"
    resources = executable.parent.parent / "Resources"
    (resources / "training_runtime").mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(desktop_main.sys, "frozen", True, raising=False)
    monkeypatch.delattr(desktop_main.sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(desktop_main.sys, "executable", str(executable))

    assert desktop_main._bundle_dir() == resources


def test_bundle_dir_does_not_let_non_resource_meipass_hide_macos_resources(monkeypatch, tmp_path):
    desktop_main = _load_desktop_main()
    executable = tmp_path / "NAMMixer.app" / "Contents" / "MacOS" / "NAMMixer"
    resources = executable.parent.parent / "Resources"
    (resources / "training_runtime").mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    executable.touch()
    meipass = tmp_path / "NAMMixer.app" / "Contents" / "Frameworks"
    meipass.mkdir()
    monkeypatch.setattr(desktop_main.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop_main.sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(desktop_main.sys, "executable", str(executable))

    assert desktop_main._bundle_dir() == resources


def test_saved_desktop_settings_override_stale_inherited_values(monkeypatch, tmp_path):
    desktop_main = _load_desktop_main()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "NAM_MIXER_LOCAL_LLM_MODEL=gemma4:e4b\n"
        "TONE3000_API_KEY=t3k_cs_saved\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(env_path))
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "stale-model")
    monkeypatch.setenv("TONE3000_API_KEY", "stale_shell_value")

    desktop_main._load_saved_desktop_settings()

    import os
    assert os.environ["NAM_MIXER_LOCAL_LLM_MODEL"] == "gemma4:e4b"
    assert os.environ["TONE3000_API_KEY"] == "t3k_cs_saved"


def test_desktop_api_save_file_writes_chosen_path(tmp_path, monkeypatch):
    """Regression test: pywebview's native webview ignores <a download> --
    static/app.js routes desktop downloads through this instead, so a
    native Save dialog (not an inline open) is what actually happens."""
    import base64
    import sys
    import types

    desktop_main = _load_desktop_main()
    dest = tmp_path / "chosen.nam"

    class FakeWindow:
        def create_file_dialog(self, dialog_type, save_filename=None):
            assert save_filename == "model.nam"
            return [str(dest)]

    fake_webview = types.SimpleNamespace(windows=[FakeWindow()], SAVE_DIALOG="save")
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    api = desktop_main.DesktopApi()
    payload = base64.b64encode(b"hello nam").decode()
    result = api.save_file("model.nam", payload)

    assert result == str(dest)
    assert dest.read_bytes() == b"hello nam"


def test_desktop_api_save_file_returns_none_when_dialog_cancelled(monkeypatch):
    import sys
    import types

    desktop_main = _load_desktop_main()

    class FakeWindow:
        def create_file_dialog(self, dialog_type, save_filename=None):
            return None

    fake_webview = types.SimpleNamespace(windows=[FakeWindow()], SAVE_DIALOG="save")
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    api = desktop_main.DesktopApi()
    assert api.save_file("model.nam", "aGVsbG8=") is None


def test_run_kaggle_cli_dispatches_to_kaggle_module_main_and_returns_exit_code(monkeypatch):
    """Regression test for KaggleCli's frozen-build fallback
    ([sys.executable, "--run-kaggle-cli", *args]): this is the receiving end
    -- it must call the real kaggle.cli.main() (bundled via
    desktop/build.spec's collect_all("kaggle")), not relaunch the GUI app."""
    import sys
    import types

    desktop_main = _load_desktop_main()

    captured = {}

    def fake_kaggle_main():
        captured["argv"] = list(sys.argv)
        raise SystemExit(0)

    fake_cli_module = types.SimpleNamespace(main=fake_kaggle_main)
    monkeypatch.setitem(sys.modules, "kaggle", types.SimpleNamespace(cli=fake_cli_module))
    monkeypatch.setitem(sys.modules, "kaggle.cli", fake_cli_module)

    result = desktop_main._run_kaggle_cli_and_exit(["auth", "login"])

    assert result == 0
    assert captured["argv"] == ["kaggle", "auth", "login"]


def test_run_kaggle_cli_returns_nonzero_exit_code(monkeypatch):
    import sys
    import types

    desktop_main = _load_desktop_main()

    def fake_kaggle_main():
        raise SystemExit(2)

    fake_cli_module = types.SimpleNamespace(main=fake_kaggle_main)
    monkeypatch.setitem(sys.modules, "kaggle", types.SimpleNamespace(cli=fake_cli_module))
    monkeypatch.setitem(sys.modules, "kaggle.cli", fake_cli_module)

    assert desktop_main._run_kaggle_cli_and_exit(["invalid"]) == 2

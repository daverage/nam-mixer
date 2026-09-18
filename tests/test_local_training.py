from __future__ import annotations

import sys
import time
import pytest

from hybrid.local_training import LocalTrainingManager


def test_cancel_stops_an_app_owned_process_and_reports_cancelled(tmp_path):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    manager._start([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"], "training")

    manager.cancel()
    deadline = time.monotonic() + 2
    while manager.status()["state"] == "cancelling" and time.monotonic() < deadline:
        time.sleep(0.01)

    status = manager.status()
    assert status["state"] == "cancelled"
    assert status["exit_code"] is not None
    assert "Cancellation requested" in status["log_tail"]


def test_setup_cannot_expose_previous_training_validation_report(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    old_manifest = tmp_path / "old-manifest.json"
    old_manifest.write_text('{"training":{"validation_report":{"state":"passed"}}}')
    manager.manifest_path = old_manifest
    monkeypatch.setattr(manager, "_start", lambda command, state: setattr(manager, "state", state))

    manager.setup()

    assert manager.manifest_path is None
    assert manager.status()["validation_report"] is None
    assert manager.design_id is None


def test_setup_uses_own_interpreter_when_it_already_meets_the_version_floor(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.setattr("hybrid.local_training._training_python_version", lambda argv: (3, 11))
    started = {}
    monkeypatch.setattr(manager, "_start", lambda command, state: started.update(command=command, state=state))

    manager.setup()

    assert started["command"][:2] == [sys.executable, "-c"]
    assert started["state"] == "setting_up"


def test_setup_falls_back_to_system_python_when_own_interpreter_is_too_old(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)
    monkeypatch.setattr("hybrid.local_training.shutil.which", lambda name: "/usr/local/bin/python3" if name == "python3" else None)

    def fake_version(argv):
        return (3, 9) if argv == [sys.executable] else (3, 11)
    monkeypatch.setattr("hybrid.local_training._training_python_version", fake_version)
    started = {}
    monkeypatch.setattr(manager, "_start", lambda command, state: started.update(command=command, state=state))

    manager.setup()

    assert started["command"][:2] == ["/usr/local/bin/python3", "-c"]
    assert started["state"] == "setting_up"


def test_setup_explains_when_no_system_python_is_available(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)
    monkeypatch.setattr("hybrid.local_training.shutil.which", lambda _name: None)

    def fake_version(argv):
        return (3, 9) if argv == [sys.executable] else None
    monkeypatch.setattr("hybrid.local_training._training_python_version", fake_version)

    with pytest.raises(RuntimeError, match=r"Python 3\.10\+ installed"):
        manager.setup()


def test_setup_skips_a_too_old_python_and_uses_a_valid_one(tmp_path, monkeypatch):
    """Regression test for a real failure: shutil.which("python3") picked up
    macOS's ancient Xcode-bundled Python 3.9 (earlier on PATH than any real
    installed Python), and pip then failed opaquely on
    neural-amp-modeler==0.13.0 (which needs Python 3.10+) instead of a clear
    "wrong Python version" message. The bootstrap must now actually check
    each candidate's reported version, not just that it exists."""
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)

    def fake_which(name):
        return {"python3.12": None, "python3.11": None, "python3.10": None,
                "python3.13": None, "python3.14": None,
                "python3": "/usr/bin/python3", "python": None}.get(name)
    monkeypatch.setattr("hybrid.local_training.shutil.which", fake_which)

    def fake_version(argv):
        # Simulate the exact bug: the only thing on PATH is the ancient
        # Xcode-bundled Python 3.9. sys.executable is treated as equally too
        # old here, forcing the candidate search.
        return (3, 9) if argv in ([sys.executable], ["/usr/bin/python3"]) else None
    monkeypatch.setattr("hybrid.local_training._training_python_version", fake_version)

    with pytest.raises(RuntimeError, match=r"only found: 3\.9 \(/usr/bin/python3\)"):
        manager.setup()


def test_setup_prefers_a_versioned_python_over_the_bare_name(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)

    def fake_which(name):
        return {"python3.12": "/opt/homebrew/bin/python3.12", "python3": "/usr/bin/python3"}.get(name)
    monkeypatch.setattr("hybrid.local_training.shutil.which", fake_which)

    def fake_version(argv):
        return {tuple([sys.executable]): (3, 9),
                ("/opt/homebrew/bin/python3.12",): (3, 12),
                ("/usr/bin/python3",): (3, 9)}.get(tuple(argv))
    monkeypatch.setattr("hybrid.local_training._training_python_version", fake_version)

    started = {}
    monkeypatch.setattr(manager, "_start", lambda command, state: started.update(command=command, state=state))
    manager.setup()

    assert started["command"][0] == "/opt/homebrew/bin/python3.12"


def test_configured_training_python_below_minimum_version_is_rejected(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    old_python = tmp_path / "old_python"
    old_python.touch()
    monkeypatch.setenv("NAM_MIXER_TRAINING_PYTHON", str(old_python))

    def fake_version(argv):
        return (3, 9)
    monkeypatch.setattr("hybrid.local_training._training_python_version", fake_version)

    with pytest.raises(RuntimeError, match=r"reports Python 3\.9.*needs Python 3\.10\+"):
        manager.setup()


def test_custom_venv_directory_is_separate_from_training_sources(tmp_path):
    source_root = tmp_path / "bundled-training-sources"
    data_venv = tmp_path / "user-data" / ".venv-a2"
    manager = LocalTrainingManager(source_root, tmp_path / "user-data" / "a2", venv_dir=data_venv)

    assert manager.venv_dir == data_venv
    assert manager.python == data_venv / "bin" / "python"


def test_training_ownership_follows_accepted_manifest_and_survives_launch_failure(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "a2")
    manager.python.parent.mkdir(parents=True)
    manager.python.touch()
    manifests = []
    for name in ("first", "second"):
        path = manager.output_root / name / "training_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        manifests.append(path)
    monkeypatch.setattr(manager, "_start", lambda *args: None)
    manager.train(manifests[0], "standard")
    assert manager.design_id == "first"

    def fail_start(*args):
        raise OSError("cannot launch")
    monkeypatch.setattr(manager, "_start", fail_start)
    with pytest.raises(OSError, match="cannot launch"):
        manager.train(manifests[1], "standard")
    assert manager.design_id == "first"
    assert manager.manifest_path == manifests[0]

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


def test_frozen_setup_uses_system_python_not_the_app_executable(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr("hybrid.local_training.shutil.which", lambda name: "/usr/local/bin/python3" if name == "python3" else None)
    started = {}
    monkeypatch.setattr(manager, "_start", lambda command, state: started.update(command=command, state=state))

    manager.setup()

    assert started["command"][:2] == ["/usr/local/bin/python3", "-c"]
    assert started["state"] == "setting_up"


def test_frozen_setup_explains_when_no_system_python_is_available(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)
    monkeypatch.setattr("hybrid.local_training.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Python 3 installed outside the packaged app"):
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

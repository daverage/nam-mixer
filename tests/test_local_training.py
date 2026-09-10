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

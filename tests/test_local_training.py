from __future__ import annotations

import sys
import time
import pytest

from hybrid.training.local_training import LocalTrainingManager


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


def test_carriage_return_progress_bar_is_visible_before_the_epoch_finishes(tmp_path):
    """PyTorch Lightning's own progress bar updates via '\\r' on a single
    line and never emits '\\n' until an epoch actually completes -- a
    naive `for line in process.stdout` misses it entirely until then,
    which is exactly what "we never see anything during the epoch phase"
    was. Simulates that shape directly: a long-running '\\r' update with
    no trailing newline while the process is still alive."""
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    manager._start(
        [sys.executable, "-c",
         "import sys, time; sys.stdout.write('Epoch 2/59: 40%\\r'); sys.stdout.flush(); time.sleep(30)"],
        "training",
    )
    deadline = time.monotonic() + 2
    status = manager.status()
    while "Epoch 2/59" not in status["log_tail"] and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.status()
    assert "Epoch 2/59" in status["log_tail"]
    assert status["progress"] == {"epoch": 3, "total_epochs": 60}
    manager.cancel()


def test_lightning_epoch_colon_progress_is_parsed_with_preset_total(tmp_path):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    manager._start(
        [sys.executable, "-c",
         "import sys,time; print('--epoch-preset=standard: 60 epochs.', flush=True); sys.stdout.write('Epoch 2: 40%|####\\r'); sys.stdout.flush(); time.sleep(30)"],
        "training",
    )
    deadline = time.monotonic() + 2
    status = manager.status()
    while status["progress"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.status()
    assert status["progress"] == {"epoch": 3, "total_epochs": 60}
    assert status["latest_line"].startswith("Epoch 2: 40%")
    manager.cancel()


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


def test_status_surfaces_embedded_artifact_so_the_ui_can_offer_it(tmp_path, monkeypatch):
    """Sequential Embedded's real deliverable is the packaged Sequential
    (head + cab) file recorded at manifest["training"]["embedded_artifact"],
    never the bare SlimmableContainer head that `status()` already exposed
    via `output_nam_path`. Before this fix, `status()` dropped
    `embedded_artifact` entirely, so the UI had no way to know a validated
    embedded package existed and could only ever link to the head -- the
    file a user would (wrongly) call "Sequential Embedded.nam" and observe
    as amp-only/gear_type amp, even though the amp+cab package genuinely
    reports gear_type amp_cab (see tests/test_cab_export_modes.py)."""
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"training": {"validation_report": {"state": "passed"}, '
        '"embedded_artifact": {"state": "validated", '
        '"artifacts": {"sequential_nam_path": "/tmp/x-embedded-experimental-full.nam"}}}}'
    )
    manager.manifest_path = manifest
    monkeypatch.setattr(manager, "state", "complete")

    status = manager.status()

    assert status["embedded_artifact"]["state"] == "validated"
    assert status["embedded_artifact"]["artifacts"]["sequential_nam_path"].endswith("-embedded-experimental-full.nam")


def test_setup_uses_own_interpreter_when_it_already_meets_the_version_floor(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.setattr("hybrid.training.local_training._training_python_version", lambda argv: (3, 11))
    started = {}
    monkeypatch.setattr(manager, "_start", lambda command, state: started.update(command=command, state=state))

    manager.setup()

    assert started["command"][:2] == [sys.executable, "-c"]
    assert started["state"] == "setting_up"


def test_setup_falls_back_to_system_python_when_own_interpreter_is_too_old(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)
    monkeypatch.setattr("hybrid.training.local_training.shutil.which", lambda name: "/usr/local/bin/python3" if name == "python3" else None)

    def fake_version(argv):
        return (3, 9) if argv == [sys.executable] else (3, 11)
    monkeypatch.setattr("hybrid.training.local_training._training_python_version", fake_version)
    started = {}
    monkeypatch.setattr(manager, "_start", lambda command, state: started.update(command=command, state=state))

    manager.setup()

    assert started["command"][:2] == ["/usr/local/bin/python3", "-c"]
    assert started["state"] == "setting_up"


def test_setup_explains_when_no_system_python_is_available(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)
    monkeypatch.setattr("hybrid.training.local_training.shutil.which", lambda _name: None)

    def fake_version(argv):
        return (3, 9) if argv == [sys.executable] else None
    monkeypatch.setattr("hybrid.training.local_training._training_python_version", fake_version)

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
    monkeypatch.setattr("hybrid.training.local_training.shutil.which", fake_which)

    def fake_version(argv):
        # Simulate the exact bug: the only thing on PATH is the ancient
        # Xcode-bundled Python 3.9. sys.executable is treated as equally too
        # old here, forcing the candidate search.
        return (3, 9) if argv in ([sys.executable], ["/usr/bin/python3"]) else None
    monkeypatch.setattr("hybrid.training.local_training._training_python_version", fake_version)

    with pytest.raises(RuntimeError, match=r"only found: 3\.9 \(/usr/bin/python3\)"):
        manager.setup()


def test_setup_prefers_a_versioned_python_over_the_bare_name(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    monkeypatch.delenv("NAM_MIXER_TRAINING_PYTHON", raising=False)

    def fake_which(name):
        return {"python3.12": "/opt/homebrew/bin/python3.12", "python3": "/usr/bin/python3"}.get(name)
    monkeypatch.setattr("hybrid.training.local_training.shutil.which", fake_which)

    def fake_version(argv):
        return {tuple([sys.executable]): (3, 9),
                ("/opt/homebrew/bin/python3.12",): (3, 12),
                ("/usr/bin/python3",): (3, 9)}.get(tuple(argv))
    monkeypatch.setattr("hybrid.training.local_training._training_python_version", fake_version)

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
    monkeypatch.setattr("hybrid.training.local_training._training_python_version", fake_version)

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
    manager._setup_complete_marker.touch()
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


def test_collect_keeps_crlf_lines_and_multibyte_characters_split_across_reads(tmp_path):
    """Windows ends every line with '\\r\\n', which must not erase the line; a
    bare '\\r' is still a progress-bar overwrite; and a UTF-8 character split
    across two os.read chunks must decode intact."""
    import os
    from types import SimpleNamespace

    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    read_fd, write_fd = os.pipe()
    bar = "\u2588".encode("utf-8")  # a tqdm block glyph, 3 bytes
    os.write(write_fd, b"Epoch 1/60 done\r\nprogress 10%\rprogress 90%\r\nbar " + bar[:1])
    os.write(write_fd, bar[1:] + b" end\r\n")
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stdout:
        manager.process = SimpleNamespace(stdout=stdout, wait=lambda: 0)
        manager._collect()
    assert list(manager.log)[-3:] == ["Epoch 1/60 done", "progress 90%", "bar \u2588 end"]



def _fake_venv_python(manager):
    manager.python.parent.mkdir(parents=True, exist_ok=True)
    manager.python.write_text("")
    return manager.python


def test_adoption_check_imports_neural_amp_modeler_and_caches_a_failure(tmp_path, monkeypatch):
    import os
    import subprocess

    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2", venv_dir=tmp_path / "venv")
    python = _fake_venv_python(manager)
    calls = []

    def failing_run(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr("hybrid.training.local_training.subprocess.run", failing_run)
    assert manager.is_ready is False and manager.is_ready is False
    assert len(calls) == 1                                  # the failed check is not re-run on every poll
    assert "nam" in calls[0][-1].replace(" ", "").split("import")[-1].split(",")
    os.utime(python, (python.stat().st_atime, python.stat().st_mtime + 10))   # the venv changed
    assert manager.is_ready is False and len(calls) == 2


def test_a_failed_adoption_check_is_retried_after_packages_change_or_a_minute(tmp_path, monkeypatch):
    import os
    import subprocess

    import hybrid.training.local_training as lt

    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2", venv_dir=tmp_path / "venv")
    _fake_venv_python(manager)
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"; site.mkdir(parents=True)
    outcome = {"ok": False, "calls": 0}

    def run(argv, **kwargs):
        outcome["calls"] += 1
        if not outcome["ok"]:
            raise subprocess.TimeoutExpired(argv, 15)             # e.g. a slow first torch import
    monkeypatch.setattr(lt.subprocess, "run", run)
    clock = [1000.0]
    monkeypatch.setattr(lt.time, "monotonic", lambda: clock[0])

    assert manager.is_ready is False and manager.is_ready is False and outcome["calls"] == 1
    # pip finishes installing into the existing venv: the interpreter is untouched, site-packages is not
    outcome["ok"] = True
    os.utime(site, (site.stat().st_atime, site.stat().st_mtime + 10))
    assert manager.is_ready is True and outcome["calls"] == 2

    # a transient failure alone is retried once the minute is up
    manager2 = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2", venv_dir=tmp_path / "venv2")
    _fake_venv_python(manager2)
    outcome.update(ok=False, calls=0)
    assert manager2.is_ready is False and manager2.is_ready is False and outcome["calls"] == 1
    outcome["ok"] = True
    clock[0] += lt.FAILED_IMPORT_CHECK_RETRY_S + 1
    assert manager2.is_ready is True and outcome["calls"] == 2


def test_packaged_app_never_probes_its_own_executable_as_python(tmp_path, monkeypatch):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    configured = tmp_path / "python3.11"
    configured.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("NAM_MIXER_TRAINING_PYTHON", str(configured))
    probed = []
    monkeypatch.setattr("hybrid.training.local_training._training_python_version",
                        lambda argv: probed.append(argv) or (3, 11))
    command = manager._bootstrap_python()
    assert [sys.executable] not in probed and command[0] != sys.executable

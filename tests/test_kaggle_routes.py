"""Flask route tests for /api/kaggle/* -- see hybrid/kaggle_training.py.
KaggleJobManager is monkeypatched at the app-module boundary, so these never
touch a real Kaggle CLI/network/credentials.
"""
from __future__ import annotations

import json as jsonlib

import pytest

import app as app_module
from hybrid.kaggle_training import KaggleJob, KaggleTrainingError


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_status_reports_cli_not_installed(client, monkeypatch):
    monkeypatch.setattr(app_module._kaggle_manager, "status", lambda: {
        "cli_installed": False, "cli_version": None, "authenticated": False,
        "accelerator": "NvidiaTeslaT4", "quota_available": False, "quota_error": None,
    })
    resp = client.get("/api/kaggle/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["cli_installed"] is False


def test_auth_start_without_cli_returns_actionable_error(client, monkeypatch):
    monkeypatch.setattr(app_module._kaggle_manager.cli, "is_installed", lambda: False)
    resp = client.post("/api/kaggle/auth/start")
    assert resp.status_code == 400
    data = resp.get_json()
    assert "pip install kaggle" in data["command"]


def test_auth_start_falls_back_to_python_module_when_console_script_missing(client, monkeypatch):
    """Regression: a `kaggle` package importable but with no console script
    on PATH (e.g. a GUI app's PATH not including the user-site bin dir on
    macOS) reports is_installed()=True with cli.executable=None. Popen-ing
    `[None, "auth", "login"]` directly raised an uncaught TypeError (not
    OSError) and surfaced as a 500 -- launch_auth_login() must fall back to
    `sys.executable -m kaggle` instead."""
    monkeypatch.setattr(app_module._kaggle_manager.cli, "is_installed", lambda: True)
    monkeypatch.setattr(app_module._kaggle_manager.cli, "executable", None)
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        class _P:
            pass
        return _P()
    monkeypatch.setattr("hybrid.kaggle_training.subprocess.Popen", fake_popen)

    resp = client.post("/api/kaggle/auth/start")
    assert resp.status_code == 200
    assert resp.get_json()["started"] is True
    assert captured["argv"][1:] == ["-m", "kaggle", "auth", "login"]
    assert None not in captured["argv"]


def test_train_requires_design_id(client):
    resp = client.post("/api/kaggle/train", json={})
    assert resp.status_code == 400
    assert "design_id" in resp.get_json()["error"]


def test_train_requires_existing_bundle(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    resp = client.post("/api/kaggle/train", json={"design_id": "nonexistent"})
    assert resp.status_code == 400
    assert "generate" in resp.get_json()["error"].lower()


def test_train_surfaces_manager_error_distinctly(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    bundle_dir = tmp_path / "mydesign"
    bundle_dir.mkdir()
    (bundle_dir / "training_manifest.json").write_text("{}")

    def fake_submit(design_id, bundle_dir_arg, epoch_preset=None):
        raise KaggleTrainingError("Kaggle CLI is not authenticated. Run: kaggle auth login")
    monkeypatch.setattr(app_module._kaggle_manager, "submit_async", fake_submit)

    resp = client.post("/api/kaggle/train", json={"design_id": "mydesign"})
    assert resp.status_code == 400
    assert "kaggle auth login" in resp.get_json()["error"]


def test_train_success_returns_job_id(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    bundle_dir = tmp_path / "mydesign"
    bundle_dir.mkdir()
    (bundle_dir / "training_manifest.json").write_text("{}")

    captured = {}

    def fake_submit(design_id, bundle_dir_arg, epoch_preset=None):
        captured["epoch_preset"] = epoch_preset
        return KaggleJob(job_id="abc123", design_id=design_id, state="submitted")

    monkeypatch.setattr(app_module._kaggle_manager, "submit_async", fake_submit)
    resp = client.post("/api/kaggle/train", json={"design_id": "mydesign"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["job_id"] == "abc123"
    assert data["state"] == "submitted"
    assert captured["epoch_preset"] == "standard"  # default when unspecified


def test_train_passes_through_requested_epoch_preset(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    bundle_dir = tmp_path / "mydesign"
    bundle_dir.mkdir()
    (bundle_dir / "training_manifest.json").write_text("{}")

    captured = {}

    def fake_submit(design_id, bundle_dir_arg, epoch_preset=None):
        captured["epoch_preset"] = epoch_preset
        return KaggleJob(job_id="abc123", design_id=design_id, state="submitted", epoch_preset=epoch_preset)

    monkeypatch.setattr(app_module._kaggle_manager, "submit_async", fake_submit)
    resp = client.post("/api/kaggle/train", json={"design_id": "mydesign", "epoch_preset": "high_def"})
    assert resp.status_code == 200
    assert captured["epoch_preset"] == "high_def"


def test_train_rejects_unknown_epoch_preset(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    bundle_dir = tmp_path / "mydesign"
    bundle_dir.mkdir()
    (bundle_dir / "training_manifest.json").write_text("{}")

    resp = client.post("/api/kaggle/train", json={"design_id": "mydesign", "epoch_preset": "ultra"})
    assert resp.status_code == 400
    assert "epoch_preset" in resp.get_json()["error"]


def test_job_status_requires_design_id(client):
    resp = client.get("/api/kaggle/jobs/abc123")
    assert resp.status_code == 400


def test_job_status_unknown_job_returns_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    resp = client.get("/api/kaggle/jobs/doesnotexist?design_id=mydesign")
    assert resp.status_code == 404


def test_job_status_refreshes_and_returns_state(client, tmp_path, monkeypatch):
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="running")
    save_job(tmp_path, job)

    monkeypatch.setattr(app_module._kaggle_manager, "refresh", lambda j: j)
    # Status for a non-terminal job pulls fresh logs (fetch_logs), not just
    # the local file (read_log_tail) -- mock both explicitly so this test
    # doesn't rely on the incidental kernel_ref=None short-circuit inside
    # fetch_logs to avoid a real Kaggle CLI call.
    monkeypatch.setattr(app_module._kaggle_manager, "fetch_logs", lambda j: "Epoch 3/100\n")
    monkeypatch.setattr(app_module._kaggle_manager, "read_log_tail", lambda j, n=200: "")

    resp = client.get("/api/kaggle/jobs/j1?design_id=mydesign")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["state"] == "running"
    assert data["progress"] == {"epoch": 3, "total_epochs": 100}
    assert data["log_tail"] == "Epoch 3/100"


def test_job_logs_bounded(client, tmp_path, monkeypatch):
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="running")
    save_job(tmp_path, job)

    monkeypatch.setattr(app_module._kaggle_manager, "fetch_logs", lambda j: "Epoch 5/100\n")
    resp = client.get("/api/kaggle/jobs/j1/logs?design_id=mydesign")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["progress"] == {"epoch": 5, "total_epochs": 100}


def test_cleanup_requires_design_id(client):
    resp = client.post("/api/kaggle/jobs/j1/cleanup", json={})
    assert resp.status_code == 400


def test_cleanup_unknown_job_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    resp = client.post("/api/kaggle/jobs/doesnotexist/cleanup", json={"design_id": "mydesign"})
    assert resp.status_code == 404


def test_cleanup_surfaces_error_distinctly(client, tmp_path, monkeypatch):
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="running")
    save_job(tmp_path, job)

    def fake_cleanup(j):
        raise KaggleTrainingError("cannot clean up a job that is not finished (state=running)")
    monkeypatch.setattr(app_module._kaggle_manager, "cleanup", fake_cleanup)

    resp = client.post("/api/kaggle/jobs/j1/cleanup", json={"design_id": "mydesign"})
    assert resp.status_code == 400
    assert "not finished" in resp.get_json()["error"]


def test_job_status_download_filename_matches_what_download_endpoint_serves(client, tmp_path, monkeypatch):
    """Regression: the job dict returned to the UI must carry the EXACT
    filename GET /api/kaggle/jobs/<id>/download will save the file as, not
    the unrelated internal `hybrid_a2.nam` export basename baked inside the
    Kaggle kernel -- the UI's download button used to display that internal
    name while the browser actually saved a differently-named file."""
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    nam_path = tmp_path / "hybrid_a2.nam"  # internal Kaggle kernel export basename
    nam_path.write_bytes(b"fake nam contents")
    job = KaggleJob(job_id="j1", design_id="mydesign", state="complete", output_nam_path=str(nam_path))
    save_job(tmp_path, job)

    monkeypatch.setattr(app_module._kaggle_manager, "refresh", lambda j: j)
    monkeypatch.setattr(app_module._kaggle_manager, "read_log_tail", lambda j, n=200: "")

    status_resp = client.get("/api/kaggle/jobs/j1?design_id=mydesign")
    assert status_resp.status_code == 200
    download_filename = status_resp.get_json()["download_filename"]
    assert download_filename != "hybrid_a2.nam"  # must NOT be the internal export basename

    download_resp = client.get("/api/kaggle/jobs/j1/download?design_id=mydesign")
    assert download_resp.status_code == 200
    content_disposition = download_resp.headers.get("Content-Disposition", "")
    assert download_filename in content_disposition


def test_kaggle_status_job_dict_also_carries_download_filename(client, tmp_path, monkeypatch):
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="complete", output_nam_path=str(tmp_path / "hybrid_a2.nam"))
    save_job(tmp_path, job)
    monkeypatch.setattr(app_module._kaggle_manager, "status", lambda: {
        "cli_installed": True, "cli_version": "1.0", "authenticated": True,
        "accelerator": "NvidiaTeslaT4", "quota_available": True, "quota_error": None,
    })

    resp = client.get("/api/kaggle/status?design_id=mydesign")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["job"]["download_filename"] == "mydesign.nam"


# --- GET /api/kaggle/jobs/<job_id>/download -------------------------------

def test_download_requires_design_id(client):
    resp = client.get("/api/kaggle/jobs/j1/download")
    assert resp.status_code == 400
    assert "design_id" in resp.get_json()["error"]


def test_download_unknown_job_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    resp = client.get("/api/kaggle/jobs/doesnotexist/download?design_id=mydesign")
    assert resp.status_code == 404


def test_download_rejects_incomplete_job(client, tmp_path, monkeypatch):
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="running")
    save_job(tmp_path, job)

    resp = client.get("/api/kaggle/jobs/j1/download?design_id=mydesign")
    assert resp.status_code == 400
    assert "no downloadable model" in resp.get_json()["error"]


def test_download_rejects_complete_job_with_missing_recorded_path(client, tmp_path, monkeypatch):
    """Defensive: even a 'complete' job must not crash if its recorded .nam
    somehow no longer exists on disk (moved/deleted out of band)."""
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="complete",
                     output_nam_path=str(tmp_path / "gone.nam"))
    save_job(tmp_path, job)

    resp = client.get("/api/kaggle/jobs/j1/download?design_id=mydesign")
    assert resp.status_code == 404
    assert "no longer exists" in resp.get_json()["error"]


def test_download_serves_the_recorded_nam_file(client, tmp_path, monkeypatch):
    from hybrid.kaggle_training import save_job

    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path)
    nam_path = tmp_path / "output" / "hybrid_a2.nam"
    nam_path.parent.mkdir(parents=True, exist_ok=True)
    nam_content = b'{"architecture": "WaveNet"}'
    nam_path.write_bytes(nam_content)
    job = KaggleJob(job_id="j1", design_id="mydesign", state="complete", output_nam_path=str(nam_path))
    save_job(tmp_path, job)

    resp = client.get("/api/kaggle/jobs/j1/download?design_id=mydesign")
    assert resp.status_code == 200
    assert resp.data == nam_content
    assert "attachment" in resp.headers.get("Content-Disposition", "")
    assert "mydesign.nam" in resp.headers.get("Content-Disposition", "")

"""Unit tests for hybrid/kaggle_training.py -- all Kaggle CLI interaction is
mocked at the subprocess boundary. No real network/CLI call, no credentials,
no Kaggle quota consumed -- see docs/kaggle_training.md.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hybrid.kaggle_training import (
    ACCELERATOR,
    FORBIDDEN_ACCELERATORS,
    STAGED_BUNDLE_FILES,
    CliResult,
    KaggleCli,
    KaggleJob,
    KaggleJobManager,
    KaggleTrainingError,
    _redact,
    _safe_slug,
    find_active_job,
    load_job,
    save_job,
)


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


DEFAULT_CONFIG_VIEW = FakeCompleted(0, "Configuration values from C:\\Users\\x\\.kaggle\n- username: testuser\n- auth_method: OAUTH\n", "")


def make_cli(monkeypatch, executable="/usr/bin/kaggle", responses=None):
    """responses: dict mapping a tuple of argv (after the executable) to a
    FakeCompleted, or a callable(argv) -> FakeCompleted. `config view`
    defaults to reporting username "testuser" (needed by create_dataset's
    KaggleCli.username() call) unless a dict `responses` overrides it."""
    responses = responses or {}
    calls = []

    def fake_run(argv, shell, capture_output, text, timeout):
        assert shell is False, "must never invoke the Kaggle CLI via a shell"
        assert isinstance(argv, list)
        calls.append(argv)
        key = tuple(argv[1:])
        if callable(responses):
            return responses(argv)
        if key in responses:
            return responses[key]
        if key == ("config", "view"):
            return DEFAULT_CONFIG_VIEW
        return FakeCompleted(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cli = KaggleCli(executable=executable)
    return cli, calls


# --- KaggleCli ---------------------------------------------------------

def test_not_installed_reports_cleanly(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    cli = KaggleCli()
    assert cli.is_installed() is False
    assert cli.version() is None
    assert cli.is_authenticated() is False


def test_version_parsing(monkeypatch):
    cli, calls = make_cli(monkeypatch, responses={("--version",): FakeCompleted(0, "Kaggle API 2.2.4\n", "")})
    assert cli.version() == "2.2.4"
    assert calls[0] == ["/usr/bin/kaggle", "--version"]


def test_is_authenticated_true_and_false(monkeypatch):
    cli, _ = make_cli(monkeypatch, responses={
        ("kernels", "list", "-m", "--page-size", "1"): FakeCompleted(0, "", ""),
    })
    assert cli.is_authenticated() is True

    def unauth(argv):
        return FakeCompleted(1, "", "401 Unauthorized")
    cli2, _ = make_cli(monkeypatch, responses=unauth)
    assert cli2.is_authenticated() is False


def test_quota_unavailable_is_not_fatal(monkeypatch):
    def responses(argv):
        if argv[1:] == ["quota", "--help"]:
            return FakeCompleted(1, "", "unknown command")
        return FakeCompleted(0, "", "")
    cli, _ = make_cli(monkeypatch, responses=responses)
    result = cli.quota()
    assert result["quota_available"] is False
    assert "quota_error" in result


def test_quota_success(monkeypatch):
    def responses(argv):
        if argv[1:] == ["quota", "--help"]:
            return FakeCompleted(0, "", "")
        if argv[1:] == ["quota"]:
            return FakeCompleted(0, "GPU: 30h remaining", "")
        return FakeCompleted(0, "", "")
    cli, _ = make_cli(monkeypatch, responses=responses)
    result = cli.quota()
    assert result["quota_available"] is True
    assert "30h" in result["quota_raw"]


def test_p100_is_never_selectable(monkeypatch, tmp_path):
    cli, calls = make_cli(monkeypatch)
    with pytest.raises(ValueError):
        cli.kernels_push(tmp_path, accelerator="NvidiaTeslaP100")
    assert not calls  # never even attempted a subprocess call
    assert "NvidiaTeslaP100" in FORBIDDEN_ACCELERATORS


def test_kernels_push_uses_t4_by_default(monkeypatch, tmp_path):
    cli, calls = make_cli(monkeypatch)
    cli.kernels_push(tmp_path)
    assert calls[0] == ["/usr/bin/kaggle", "kernels", "push", "-p", str(tmp_path), "--accelerator", ACCELERATOR]
    assert ACCELERATOR == "NvidiaTeslaT4"


def test_redact_scrubs_secret_shaped_text():
    text = "auth token=abc123XYZ done"
    redacted = _redact(text)
    assert "abc123XYZ" not in redacted


def test_safe_slug_sanitisation():
    assert _safe_slug("../../etc/passwd") == "etc-passwd"
    assert _safe_slug("My Design #1!!") == "my-design-1"
    assert _safe_slug("") == "job"


# --- staging / manager --------------------------------------------------

@pytest.fixture
def bundle_dir(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    sf.write(d / "input.wav", np.zeros(100, dtype=np.float32), 48000, subtype="FLOAT")
    sf.write(d / "hybrid_target.wav", np.zeros(100, dtype=np.float32), 48000, subtype="FLOAT")
    (d / "training_manifest.json").write_text(json.dumps({"amp_a": {"filename": "A.nam"}, "amp_b": {"filename": "B.nam"}}))
    # Extra files that must NEVER be staged.
    (d / "amp_a_source.nam").write_text("not real, must not be uploaded")
    (d / "some_di.wav").write_text("not real, must not be uploaded")
    return d


@pytest.fixture
def cloud_script(tmp_path, monkeypatch):
    """hybrid/kaggle_training.py locates cloud/kaggle/train_a2_cloud.py
    relative to the real repo -- that file exists for real, so no fixture
    needed; this fixture is a no-op placeholder kept for clarity."""
    return None


def test_staging_allow_list(tmp_path, bundle_dir):
    manager = KaggleJobManager(tmp_path)
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)

    staged_names = {p.name for p in staging.iterdir()}
    for name in STAGED_BUNDLE_FILES:
        assert name in staged_names
    assert "amp_a_source.nam" not in staged_names
    assert "some_di.wav" not in staged_names
    assert "cloud_job.json" in staged_names
    assert "train_a2_cloud.py" in staged_names
    assert job.state == "uploading"


def test_staging_missing_file_raises(tmp_path, bundle_dir):
    (bundle_dir / "input.wav").unlink()
    manager = KaggleJobManager(tmp_path)
    job = KaggleJob(job_id="abc", design_id="d")
    with pytest.raises(KaggleTrainingError):
        manager.stage(job, bundle_dir)


def test_create_dataset_is_always_private(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path)
    cli, calls = make_cli(monkeypatch)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, staging)

    metadata = json.loads((staging / "dataset-metadata.json").read_text())
    assert "public" not in json.dumps(metadata).lower() or metadata.get("public") is not True
    push_call = [c for c in calls if c[1:3] == ["datasets", "create"]]
    assert push_call
    assert "--public" not in push_call[0]
    assert job.dataset_ref == metadata["id"]
    # Kaggle's dataset_create_new does ref.split("/")[1] unconditionally --
    # a bare slug crashes with IndexError, so id must be "username/slug".
    assert metadata["id"] == f"testuser/{metadata['title']}"
    assert 6 <= len(metadata["title"]) <= 50
    assert job.state == "waiting_for_dataset"


def test_create_dataset_fails_cleanly_without_username(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path)
    cli, _ = make_cli(monkeypatch, responses={("config", "view"): FakeCompleted(0, "no username line here", "")})
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)
    with pytest.raises(KaggleTrainingError, match="username"):
        manager.create_dataset(job, staging)
    assert job.state == "failed"


def test_create_dataset_slug_bounded_for_long_ids(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path)
    cli, _ = make_cli(monkeypatch)
    manager.cli = cli
    job = KaggleJob(job_id="a" * 40, design_id="a-very-long-design-name-that-goes-on-and-on-and-on")
    staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, staging)
    metadata = json.loads((staging / "dataset-metadata.json").read_text())
    assert 6 <= len(metadata["title"]) <= 50
    assert metadata["id"].startswith("testuser/")


def test_create_kernel_is_always_private_and_t4(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0)
    cli, calls = make_cli(monkeypatch)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, staging)
    manager.create_kernel(job, staging)

    metadata = json.loads((staging / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    push_call = [c for c in calls if c[1:3] == ["kernels", "push"]]
    assert push_call
    assert "--accelerator" in push_call[0]
    assert push_call[0][push_call[0].index("--accelerator") + 1] == ACCELERATOR
    assert job.state == "submitted"


def test_create_kernel_id_and_ref_are_username_qualified(tmp_path, bundle_dir, monkeypatch):
    """Production bug: create_kernel() persisted a BARE kernel slug (no
    "username/" prefix) as both kernel-metadata.json's "id" and
    job.kernel_ref, so `kaggle kernels status <bare-slug>` (and every other
    command built from job.kernel_ref) resolved to nothing on the real
    Kaggle API even though `kernels push` exited 0."""
    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0)
    cli, calls = make_cli(monkeypatch)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, staging)
    manager.create_kernel(job, staging)

    metadata = json.loads((staging / "kernel-metadata.json").read_text())
    assert metadata["id"] == f"testuser/{metadata['title']}"
    assert job.kernel_ref == metadata["id"]
    assert job.kernel_ref.startswith("testuser/")

    # Every subsequent command built from job.kernel_ref must carry the full
    # owner/slug ref, never the bare slug.
    status_calls = [c for c in calls if c[1:3] == ["kernels", "status"]]
    assert status_calls
    assert all(c[3] == job.kernel_ref for c in status_calls)


def test_create_kernel_fails_when_push_succeeds_but_status_never_resolves(tmp_path, bundle_dir, monkeypatch):
    """A successful `kernels push` (exit 0) is NOT sufficient evidence the
    kernel exists -- this is exactly the production bug's failure mode:
    Kaggle silently never created/resolved the kernel. The job must be
    marked failed, never left "submitted"."""
    def responses(argv):
        if argv[1:3] == ["kernels", "status"]:
            return FakeCompleted(1, "", "Permission 'kernels.get' was denied")
        return FakeCompleted(0, "" if argv[1] != "config" else DEFAULT_CONFIG_VIEW.stdout, "")

    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0, kernel_verify_attempts=2)
    cli, calls = make_cli(monkeypatch, responses=responses)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, staging)

    with pytest.raises(KaggleTrainingError, match="did not create/resolve the kernel"):
        manager.create_kernel(job, staging)

    assert job.state == "failed"
    assert job.kernel_ref is None  # never persisted -- it was never real
    status_calls = [c for c in calls if c[1:3] == ["kernels", "status"]]
    assert len(status_calls) == 2  # bounded retry, not infinite


def test_create_kernel_succeeds_when_status_resolves_on_second_attempt(tmp_path, bundle_dir, monkeypatch):
    """Eventual consistency: the first status check can fail immediately
    after push without that being a real failure, as long as it resolves
    within the bounded retry window."""
    attempt_count = {"n": 0}

    def responses(argv):
        if argv[1:3] == ["kernels", "status"]:
            attempt_count["n"] += 1
            if attempt_count["n"] == 1:
                return FakeCompleted(1, "", "Not found")
            return FakeCompleted(0, "queued", "")
        if argv[1:3] == ["config", "view"]:
            return DEFAULT_CONFIG_VIEW
        return FakeCompleted(0, "", "")

    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0, kernel_verify_attempts=3)
    cli, _ = make_cli(monkeypatch, responses=responses)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, staging)
    manager.create_kernel(job, staging)

    assert job.state == "submitted"
    assert job.kernel_ref == f"testuser/hybrid-a2-train-mydesign-{job.job_id[:12]}"


def test_submit_requires_cli_installed(tmp_path, bundle_dir, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    manager = KaggleJobManager(tmp_path)
    with pytest.raises(KaggleTrainingError, match="not installed"):
        manager.submit("mydesign", bundle_dir)


def test_submit_requires_authentication(tmp_path, bundle_dir, monkeypatch):
    cli, _ = make_cli(monkeypatch, responses=lambda argv: FakeCompleted(1, "", "unauthorized"))
    manager = KaggleJobManager(tmp_path, cli=cli)
    with pytest.raises(KaggleTrainingError, match="not authenticated"):
        manager.submit("mydesign", bundle_dir)


def test_submit_rejects_second_concurrent_job(tmp_path, bundle_dir, monkeypatch):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    job1 = manager.submit("mydesign", bundle_dir)
    assert job1.state == "submitted"
    with pytest.raises(KaggleTrainingError, match="already in progress"):
        manager.submit("mydesign", bundle_dir)


# --- refresh() migration path for pre-fix bare kernel_ref jobs -----------

def test_refresh_migrates_bare_kernel_ref_when_it_actually_resolves(tmp_path, monkeypatch):
    """A job persisted by the buggy pre-fix version has a bare kernel_ref
    (no "username/" prefix). If it happens to still resolve once qualified
    with the authenticated username, migrate it in place rather than
    failing a job that could actually proceed."""
    def responses(argv):
        if argv[1:3] == ["kernels", "status"] and argv[3] == "testuser/bare-kernel-slug":
            return FakeCompleted(0, "running", "")
        if argv[1:3] == ["config", "view"]:
            return DEFAULT_CONFIG_VIEW
        return FakeCompleted(1, "", "Not found")  # bare ref alone never resolves
    cli, calls = make_cli(monkeypatch, responses=responses)
    manager = KaggleJobManager(tmp_path, cli=cli)

    job = KaggleJob(job_id="j1", design_id="mydesign", state="submitted",
                     dataset_ref="testuser/ds1", kernel_ref="bare-kernel-slug")
    save_job(tmp_path, job)

    refreshed = manager.refresh(job)
    assert refreshed.kernel_ref == "testuser/bare-kernel-slug"
    assert refreshed.state != "failed"


def test_refresh_fails_stuck_job_with_unresolvable_bare_kernel_ref(tmp_path, monkeypatch):
    """Exactly the production incident: a bare kernel_ref that was never a
    real Kaggle kernel. refresh() must fail the job (never leave it
    "submitted" forever) rather than blocking find_active_job permanently."""
    cli, _ = make_cli(monkeypatch, responses=lambda argv: FakeCompleted(1, "", "Not found"))
    manager = KaggleJobManager(tmp_path, cli=cli)

    job = KaggleJob(job_id="j1", design_id="mydesign", state="submitted",
                     dataset_ref="testuser/ds1", kernel_ref="bare-kernel-slug")
    save_job(tmp_path, job)

    refreshed = manager.refresh(job)
    assert refreshed.state == "failed"
    assert "stuck job" in refreshed.error
    assert refreshed.kernel_ref == "bare-kernel-slug"  # left as-is, not silently rewritten to a lie


def test_submit_unblocks_when_existing_job_is_a_stuck_bare_ref(tmp_path, bundle_dir, monkeypatch):
    cli, _ = make_cli(monkeypatch, responses=lambda argv: (
        FakeCompleted(1, "", "Not found") if argv[1:3] == ["kernels", "status"] else FakeCompleted(0, "", "")
    ))
    manager = KaggleJobManager(tmp_path, cli=cli, kernel_verify_delay_s=0)

    stuck = KaggleJob(job_id="stuck1", design_id="mydesign", state="submitted",
                       dataset_ref="testuser/ds1", kernel_ref="bare-kernel-slug")
    save_job(tmp_path, stuck)

    # A fresh submit should refresh (and fail) the stuck job rather than
    # being permanently blocked by it -- but the new submission itself also
    # fails here (kernels_status always denies), which is fine: the point is
    # it's not rejected with "already in progress".
    with pytest.raises(KaggleTrainingError) as excinfo:
        manager.submit("mydesign", bundle_dir)
    assert "already in progress" not in str(excinfo.value)


# --- job persistence ----------------------------------------------------

def test_job_save_and_load_roundtrip(tmp_path):
    job = KaggleJob(job_id="j1", design_id="d1", state="running", dataset_ref="ds1", kernel_ref="k1")
    save_job(tmp_path, job)
    loaded = load_job(tmp_path, "d1", "j1")
    assert loaded is not None
    assert loaded.state == "running"
    assert loaded.dataset_ref == "ds1"


def test_job_json_never_contains_secret_shaped_strings(tmp_path):
    job = KaggleJob(job_id="j1", design_id="d1", error="dataset upload failed: token=SHOULDNOTLEAK")
    save_job(tmp_path, job)
    raw = (tmp_path / "d1" / "kaggle" / "j1" / "job.json").read_text()
    assert "SHOULDNOTLEAK" not in raw


def test_find_active_job_prefers_non_terminal(tmp_path):
    job_done = KaggleJob(job_id="a", design_id="d", state="complete", updated_at=1.0)
    job_running = KaggleJob(job_id="b", design_id="d", state="running", updated_at=2.0)
    save_job(tmp_path, job_done)
    save_job(tmp_path, job_running)
    active = find_active_job(tmp_path, "d")
    assert active.job_id == "b"


def test_find_active_job_no_jobs(tmp_path):
    assert find_active_job(tmp_path, "nonexistent") is None


# --- cleanup --------------------------------------------------------------

def test_cleanup_refuses_non_terminal_job(tmp_path, monkeypatch):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d", state="running")
    with pytest.raises(KaggleTrainingError):
        manager.cleanup(job)


def test_cleanup_refuses_complete_without_local_validation(tmp_path, monkeypatch):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d", state="complete", local_validation=None)
    with pytest.raises(KaggleTrainingError):
        manager.cleanup(job)


def test_cleanup_after_valid_completion_deletes_remote_only(tmp_path, monkeypatch, bundle_dir):
    cli, calls = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    nam_path = bundle_dir / "output.nam"
    nam_path.write_text("fake nam contents")
    job = KaggleJob(
        job_id="j", design_id="d", state="complete",
        dataset_ref="ds1", kernel_ref="k1",
        local_validation={"full": {"rendered_ok": True}},
        output_nam_path=str(nam_path),
    )
    result = manager.cleanup(job)
    assert result.cleanup_state == "cleaned"
    delete_calls = [c for c in calls if "delete" in c]
    assert any(c[1] == "datasets" for c in delete_calls)
    assert any(c[1] == "kernels" for c in delete_calls)
    assert nam_path.is_file(), "cleanup must never delete the downloaded local .nam"


def test_cleanup_failure_does_not_delete_local_nam(tmp_path, monkeypatch, bundle_dir):
    cli, _ = make_cli(monkeypatch, responses=lambda argv: FakeCompleted(1, "", "delete failed"))
    manager = KaggleJobManager(tmp_path, cli=cli)
    nam_path = bundle_dir / "output.nam"
    nam_path.write_text("fake nam contents")
    job = KaggleJob(
        job_id="j", design_id="d", state="complete",
        dataset_ref="ds1", kernel_ref="k1",
        local_validation={"full": {"rendered_ok": True}},
        output_nam_path=str(nam_path),
    )
    result = manager.cleanup(job)
    assert result.cleanup_state == "cleanup_pending"
    assert result.cleanup_error
    assert nam_path.is_file()


# --- progress parsing (UX-only, never fatal) -----------------------------

def test_parse_progress_extracts_latest_epoch():
    log = "starting...\nEpoch 3/100 loss=1.2\nEpoch 17/100 loss=0.4\n"
    progress = KaggleJobManager.parse_progress(log)
    assert progress == {"epoch": 17, "total_epochs": 100}


def test_parse_progress_returns_none_when_unparseable():
    assert KaggleJobManager.parse_progress("no useful lines here") is None
    assert KaggleJobManager.parse_progress("") is None
    assert KaggleJobManager.parse_progress(None) is None

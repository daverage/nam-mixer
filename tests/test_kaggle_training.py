"""Unit tests for hybrid/kaggle_training.py -- all Kaggle CLI interaction is
mocked at the subprocess boundary. No real network/CLI call, no credentials,
no Kaggle quota consumed -- see docs/kaggle_training.md.
"""
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import soundfile as sf

import hybrid.kaggle_training as kaggle_training
from hybrid.kaggle_training import (
    A2_EPOCH_PRESETS,
    ACCELERATOR,
    DEFAULT_EPOCH_PRESET,
    FORBIDDEN_ACCELERATORS,
    REQUIRED_DATASET_FILES,
    STAGED_BUNDLE_FILES,
    CliResult,
    KaggleCli,
    KaggleJob,
    KaggleJobManager,
    KaggleTrainingError,
    _job_dir,
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


class FakePopenResult:
    """Fakes just enough of subprocess.Popen for KaggleCli.run_streaming:
    .stdout.readline() yields the response's combined output line-by-line,
    then "" (EOF); .wait()/.kill() report the response's returncode."""

    def __init__(self, response):
        combined = (response.stdout or "") + (response.stderr or "")
        self._lines = combined.splitlines(keepends=True)
        self._idx = 0
        self._returncode = response.returncode
        self.stdout = self

    def readline(self):
        if self._idx >= len(self._lines):
            return ""
        line = self._lines[self._idx]
        self._idx += 1
        return line

    def wait(self, timeout=None):
        return self._returncode

    def kill(self):
        pass


def _auto_datasets_files_response(directory: Path) -> FakeCompleted:
    """Auto-generates a `datasets files -v`-shaped CSV response reflecting
    whatever files actually exist in `directory` -- matches real Kaggle
    behavior for a genuinely complete upload, so tests that don't care about
    verification specifically don't need to hand-construct a matching CSV."""
    rows = ["name,size,creationDate"]
    if directory.is_dir():
        for p in sorted(directory.iterdir()):
            if p.is_file():
                rows.append(f"{p.name},{p.stat().st_size},2026-01-01 00:00:00.000000")
    return FakeCompleted(0, "\n".join(rows), "")


def make_cli(monkeypatch, executable="/usr/bin/kaggle", responses=None):
    """responses: dict mapping a tuple of argv (after the executable) to a
    FakeCompleted, or a callable(argv) -> FakeCompleted. `config view`
    defaults to reporting username "testuser" (needed by create_dataset's
    KaggleCli.username() call); `datasets files` defaults to reflecting
    whatever the most recent `datasets create -p <dir>` call actually staged
    (see _auto_datasets_files_response) -- both unless a dict `responses`
    overrides them. Mocks both subprocess.run (short calls) and
    subprocess.Popen (KaggleCli.run_streaming, used for dataset upload)."""
    responses = responses or {}
    calls = []
    state = {"last_dataset_dir": None}

    def _lookup(argv):
        key = tuple(argv[1:])
        if key[:2] == ("datasets", "create") and "-p" in argv:
            state["last_dataset_dir"] = Path(argv[argv.index("-p") + 1])
        if callable(responses):
            return responses(argv)
        if key in responses:
            return responses[key]
        if key == ("config", "view"):
            return DEFAULT_CONFIG_VIEW
        if key[:2] == ("datasets", "status"):
            # Default: dataset is immediately ready -- tests that care about
            # settling/eventual-consistency override this via `responses`.
            return FakeCompleted(0, "ready", "")
        if key[:2] == ("datasets", "files") and state["last_dataset_dir"] is not None:
            return _auto_datasets_files_response(state["last_dataset_dir"])
        return FakeCompleted(returncode=0, stdout="", stderr="")

    def fake_run(argv, shell, capture_output, text, timeout, encoding=None, errors=None):
        assert shell is False, "must never invoke the Kaggle CLI via a shell"
        assert isinstance(argv, list)
        calls.append(argv)
        return _lookup(argv)

    def fake_popen(argv, shell, stdout, stderr, text, bufsize, encoding=None, errors=None):
        assert shell is False, "must never invoke the Kaggle CLI via a shell"
        assert isinstance(argv, list)
        calls.append(argv)
        return FakePopenResult(_lookup(argv))

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    cli = KaggleCli(executable=executable)
    return cli, calls


# --- KaggleCli ---------------------------------------------------------

def test_not_installed_reports_cleanly(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(KaggleCli, "_module_available", staticmethod(lambda: False))
    cli = KaggleCli()
    assert cli.is_installed() is False
    assert cli.version() is None
    assert cli.is_authenticated() is False


def test_launch_auth_login_falls_back_to_python_module_when_no_console_script(monkeypatch):
    """Regression: importlib reports the `kaggle` package available (so
    is_installed() is True) but shutil.which("kaggle") finds nothing --
    e.g. a GUI app's PATH not including the user-site console-script dir on
    macOS. `executable` is then None; launching auth login must go through
    `_build_argv` (falling back to `sys.executable -m kaggle`) rather than
    Popen-ing `[None, "auth", "login"]`, which raised an uncaught TypeError."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    cli = KaggleCli()
    assert cli.executable is None

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        class _P:
            pass
        return _P()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert cli.launch_auth_login() is True
    assert None not in captured["argv"]
    assert captured["argv"][-2:] == ["auth", "login"]


def test_launch_auth_login_returns_false_on_oserror(monkeypatch):
    cli = KaggleCli(executable="/usr/bin/kaggle")

    def raising_popen(argv, **kwargs):
        raise OSError("no such file")
    monkeypatch.setattr(subprocess, "Popen", raising_popen)

    assert cli.launch_auth_login() is False


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
    dataset_staging, kernel_staging = manager.stage(job, bundle_dir)

    dataset_names = {p.name for p in dataset_staging.iterdir()}
    kernel_names = {p.name for p in kernel_staging.iterdir()}
    for name in STAGED_BUNDLE_FILES:
        assert name in dataset_names
    assert "amp_a_source.nam" not in dataset_names
    assert "some_di.wav" not in dataset_names
    assert "cloud_job.json" in dataset_names
    # The cloud worker script belongs ONLY in kernel staging -- never
    # re-uploaded as part of the dataset payload.
    assert "train_a2_cloud.py" not in dataset_names
    assert "train_a2_cloud.py" in kernel_names
    assert job.state == "uploading"


def test_staging_writes_job_epoch_preset_into_cloud_job_json(tmp_path, bundle_dir):
    manager = KaggleJobManager(tmp_path)
    job = KaggleJob(job_id="abc123", design_id="mydesign", epoch_preset="high_def")
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)

    cloud_job = json.loads((dataset_staging / "cloud_job.json").read_text())
    assert cloud_job["epoch_preset"] == "high_def"


def test_kaggle_job_defaults_to_standard_epoch_preset():
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    assert job.epoch_preset == DEFAULT_EPOCH_PRESET == "standard"


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
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, dataset_staging)

    metadata = json.loads((dataset_staging / "dataset-metadata.json").read_text())
    assert "public" not in json.dumps(metadata).lower() or metadata.get("public") is not True
    push_call = [c for c in calls if c[1:3] == ["datasets", "create"]]
    assert push_call
    assert "--public" not in push_call[0]
    assert job.dataset_ref == metadata["id"]
    # Kaggle's dataset_create_new does ref.split("/")[1] unconditionally --
    # a bare slug crashes with IndexError, so id must be "username/slug".
    assert metadata["id"] == f"testuser/{metadata['title']}"
    assert 6 <= len(metadata["title"]) <= 50
    assert job.state == "verifying_dataset"


def test_create_dataset_fails_cleanly_without_username(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path)
    cli, _ = make_cli(monkeypatch, responses={("config", "view"): FakeCompleted(0, "no username line here", "")})
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)
    with pytest.raises(KaggleTrainingError, match="username"):
        manager.create_dataset(job, dataset_staging)
    assert job.state == "failed"


def test_create_dataset_slug_bounded_for_long_ids(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path)
    cli, _ = make_cli(monkeypatch)
    manager.cli = cli
    job = KaggleJob(job_id="a" * 40, design_id="a-very-long-design-name-that-goes-on-and-on-and-on")
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, dataset_staging)
    metadata = json.loads((dataset_staging / "dataset-metadata.json").read_text())
    assert 6 <= len(metadata["title"]) <= 50
    assert metadata["id"].startswith("testuser/")


def test_create_kernel_is_always_private_and_t4(tmp_path, bundle_dir, monkeypatch):
    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0)
    cli, calls = make_cli(monkeypatch)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, kernel_staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, dataset_staging)
    manager.create_kernel(job, kernel_staging)

    metadata = json.loads((kernel_staging / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    push_call = [c for c in calls if c[1:3] == ["kernels", "push"]]
    assert push_call
    assert "--accelerator" in push_call[0]
    assert push_call[0][push_call[0].index("--accelerator") + 1] == ACCELERATOR
    assert job.state == "queued"


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
    dataset_staging, kernel_staging = manager.stage(job, bundle_dir)
    manager.create_dataset(job, dataset_staging)
    manager.create_kernel(job, kernel_staging)

    metadata = json.loads((kernel_staging / "kernel-metadata.json").read_text())
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
    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0, kernel_verify_attempts=2)
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, kernel_staging = manager.stage(job, bundle_dir)

    def responses(argv):
        if argv[1:3] == ["kernels", "status"]:
            return FakeCompleted(1, "", "Permission 'kernels.get' was denied")
        if argv[1:3] == ["datasets", "status"]:
            return FakeCompleted(0, "ready", "")
        if argv[1:3] == ["datasets", "files"]:
            return _auto_datasets_files_response(dataset_staging)
        return FakeCompleted(0, "" if argv[1] != "config" else DEFAULT_CONFIG_VIEW.stdout, "")

    cli, calls = make_cli(monkeypatch, responses=responses)
    manager.cli = cli
    manager.create_dataset(job, dataset_staging)

    with pytest.raises(KaggleTrainingError, match="did not create/resolve the kernel"):
        manager.create_kernel(job, kernel_staging)

    assert job.state == "failed"
    assert job.kernel_ref is None  # never persisted -- it was never real
    status_calls = [c for c in calls if c[1:3] == ["kernels", "status"]]
    assert len(status_calls) == 2  # bounded retry, not infinite


def test_create_kernel_succeeds_when_status_resolves_on_second_attempt(tmp_path, bundle_dir, monkeypatch):
    """Eventual consistency: the first status check can fail immediately
    after push without that being a real failure, as long as it resolves
    within the bounded retry window."""
    attempt_count = {"n": 0}
    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0, kernel_verify_attempts=3)
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, kernel_staging = manager.stage(job, bundle_dir)

    def responses(argv):
        if argv[1:3] == ["kernels", "status"]:
            attempt_count["n"] += 1
            if attempt_count["n"] == 1:
                return FakeCompleted(1, "", "Not found")
            return FakeCompleted(0, "queued", "")
        if argv[1:3] == ["config", "view"]:
            return DEFAULT_CONFIG_VIEW
        if argv[1:3] == ["datasets", "status"]:
            return FakeCompleted(0, "ready", "")
        if argv[1:3] == ["datasets", "files"]:
            return _auto_datasets_files_response(dataset_staging)
        return FakeCompleted(0, "", "")

    cli, _ = make_cli(monkeypatch, responses=responses)
    manager.cli = cli
    manager.create_dataset(job, dataset_staging)
    manager.create_kernel(job, kernel_staging)

    assert job.state == "queued"
    assert job.kernel_ref == f"testuser/hybrid-a2-train-mydesign-{job.job_id[:12]}"


def test_submit_requires_cli_installed(tmp_path, bundle_dir, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(KaggleCli, "_module_available", staticmethod(lambda: False))
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
    assert job1.state == "queued"
    with pytest.raises(KaggleTrainingError, match="already in progress"):
        manager.submit("mydesign", bundle_dir)


# --- epoch preset selection (draft=20 / standard=60 / high_def=120) ------

def test_submit_rejects_unknown_epoch_preset(tmp_path, bundle_dir, monkeypatch):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    with pytest.raises(KaggleTrainingError, match="unknown epoch_preset"):
        manager.submit("mydesign", bundle_dir, epoch_preset="ultra")


@pytest.mark.parametrize("preset", sorted(A2_EPOCH_PRESETS))
def test_submit_threads_epoch_preset_through_to_staged_cloud_job(tmp_path, bundle_dir, monkeypatch, preset):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = manager.submit("mydesign", bundle_dir, epoch_preset=preset)

    assert job.epoch_preset == preset
    cloud_job = json.loads(
        (_job_dir(tmp_path, "mydesign", job.job_id) / "dataset_staging" / "cloud_job.json").read_text()
    )
    assert cloud_job["epoch_preset"] == preset


def test_submit_defaults_to_standard_epoch_preset_when_unspecified(tmp_path, bundle_dir, monkeypatch):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = manager.submit("mydesign", bundle_dir)
    assert job.epoch_preset == "standard"


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


# --- streaming subprocess (run_streaming) --------------------------------

def test_run_streaming_writes_lines_incrementally_to_log(tmp_path, monkeypatch):
    response = FakeCompleted(0, "line one\nline two\nline three\n", "")
    cli, calls = make_cli(monkeypatch, responses=lambda argv: response)
    log_path = tmp_path / "kaggle.log"
    result = cli.run_streaming(["datasets", "create", "-p", str(tmp_path)], log_path, timeout=30)
    assert result.ok
    assert result.returncode == 0
    logged = log_path.read_text()
    assert "line one" in logged and "line two" in logged and "line three" in logged
    assert calls  # went through Popen, not subprocess.run


def test_run_streaming_redacts_secrets_in_log(tmp_path, monkeypatch):
    response = FakeCompleted(0, "auth token=SHOULDNOTLEAK\n", "")
    cli, _ = make_cli(monkeypatch, responses=lambda argv: response)
    log_path = tmp_path / "kaggle.log"
    cli.run_streaming(["datasets", "create", "-p", str(tmp_path)], log_path, timeout=30)
    assert "SHOULDNOTLEAK" not in log_path.read_text()


def test_run_streaming_times_out_without_hanging(tmp_path, monkeypatch):
    """Simulates a child that produces output forever without ever exiting
    (the real-world equivalent of Kaggle's bounded-but-very-long resumable-
    upload retry loop under persistent network trouble) -- run_streaming
    must still return within a bounded time, never hang indefinitely."""
    class NeverEndingPopen:
        def __init__(self, argv, shell, stdout, stderr, text, bufsize, encoding=None, errors=None):
            self.stdout = self
            self._killed = False

        def readline(self):
            if self._killed:
                return ""
            time.sleep(0.05)
            return "still going...\n"

        def wait(self, timeout=None):
            return -9 if self._killed else 0

        def kill(self):
            self._killed = True

    monkeypatch.setattr(subprocess, "Popen", NeverEndingPopen)
    cli = KaggleCli(executable="/usr/bin/kaggle")
    log_path = tmp_path / "kaggle.log"

    start = time.time()
    result = cli.run_streaming(["datasets", "create", "-p", str(tmp_path)], log_path, timeout=0.3)
    elapsed = time.time() - start

    assert not result.ok
    assert "timed out" in result.stderr
    assert elapsed < 5.0  # bounded, not hung -- generous margin for CI jitter
    assert "still going" in log_path.read_text()  # partial output was still captured


# --- dataset payload verification (mandatory before kernel creation) ----

class _StubCli:
    """Minimal stand-in for KaggleCli when a test only needs to control
    datasets_files()'s (and optionally datasets_status()'s) return value,
    without going through the full subprocess-mocking machinery. Status
    defaults to "ready" so tests that only care about files() behavior don't
    need to know about the settling loop's status-polling step."""
    def __init__(self, files_result: FakeCompleted, status_result: Optional[FakeCompleted] = None):
        self._files_result = CliResult(
            ok=(files_result.returncode == 0), returncode=files_result.returncode,
            stdout=files_result.stdout, stderr=files_result.stderr,
        )
        status_result = status_result or FakeCompleted(0, "ready", "")
        self._status_result = CliResult(
            ok=(status_result.returncode == 0), returncode=status_result.returncode,
            stdout=status_result.stdout, stderr=status_result.stderr,
        )

    def datasets_files(self, dataset_ref):
        return self._files_result

    def datasets_status(self, dataset_ref, json_format=False):
        return self._status_result


def test_verify_dataset_payload_accepts_matching_complete_payload(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    for name, content in (("input.wav", b"a" * 100), ("hybrid_target.wav", b"b" * 200),
                          ("training_manifest.json", b"{}"), ("cloud_job.json", b"{}")):
        (staging / name).write_bytes(content)

    csv_text = "name,size,creationDate\n" + "\n".join(
        f"{name},{ (staging / name).stat().st_size },2026-01-01 00:00:00"
        for name in ("input.wav", "hybrid_target.wav", "training_manifest.json", "cloud_job.json")
    )
    manager = KaggleJobManager(tmp_path, cli=_StubCli(FakeCompleted(0, csv_text, "")))
    ok, error = manager.verify_dataset_payload("owner/slug", staging)
    assert ok is True
    assert error == ""


def test_verify_dataset_payload_rejects_missing_file():
    csv_text = "name,size,creationDate\ncloud_job.json,103,2026-01-01 00:00:00"
    manager = KaggleJobManager(Path("."), cli=_StubCli(FakeCompleted(0, csv_text, "")))
    ok, error = manager.verify_dataset_payload("owner/slug", Path("/nonexistent"))
    assert ok is False
    assert "input.wav" in error
    assert "hybrid_target.wav" in error
    assert "training_manifest.json" in error


def test_verify_dataset_payload_rejects_wrong_remote_size(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "input.wav").write_bytes(b"a" * 1000)
    (staging / "hybrid_target.wav").write_bytes(b"b" * 2000)
    (staging / "training_manifest.json").write_bytes(b"{}")
    (staging / "cloud_job.json").write_bytes(b"{}")

    # input.wav reported with a truncated remote size -- a partial upload
    # that still produced a listing entry, exactly the incident this check
    # exists for.
    csv_text = (
        "name,size,creationDate\n"
        "input.wav,16384,2026-01-01 00:00:00\n"
        "hybrid_target.wav,2000,2026-01-01 00:00:00\n"
        "training_manifest.json,2,2026-01-01 00:00:00\n"
        "cloud_job.json,2,2026-01-01 00:00:00\n"
    )
    manager = KaggleJobManager(tmp_path, cli=_StubCli(FakeCompleted(0, csv_text, "")))
    ok, error = manager.verify_dataset_payload("owner/slug", staging)
    assert ok is False
    assert "input.wav" in error
    assert "1000" in error and "16384" in error


def test_verify_dataset_payload_rejects_when_ready_but_partial():
    """The exact real-world incident: `datasets status` reported "ready" for
    a dataset containing only cloud_job.json -- 5 of 6 files never arrived.
    `datasets files -v`-based verification must still reject this."""
    csv_text = "name,size,creationDate\ncloud_job.json,103,2026-09-05 19:46:45.585000"
    manager = KaggleJobManager(Path("."), cli=_StubCli(FakeCompleted(0, csv_text, "")))
    ok, error = manager.verify_dataset_payload(
        "andrzejmarczewski/hybrid-a2-20260905t192736z-87206825efe1", Path("/nonexistent"),
    )
    assert ok is False
    assert "missing" in error.lower()


def test_verify_dataset_payload_surfaces_files_command_failure():
    """A files() call that NEVER succeeds within the settling window (not
    just transiently) must still fail, with a clear timeout message -- the
    retry window is bounded, not infinite."""
    manager = KaggleJobManager(Path("."), cli=_StubCli(FakeCompleted(1, "", "403 Forbidden")))
    ok, error = manager.verify_dataset_payload("owner/slug", Path("/nonexistent"), timeout=0.05, interval=0.01)
    assert ok is False
    assert "could not be read back for verification" in error.lower()


def test_create_dataset_never_creates_kernel_when_verification_fails(tmp_path, bundle_dir, monkeypatch):
    """Kernel creation must never even be attempted when the dataset payload
    fails verification -- checked here by asserting no "kernels" subcommand
    ever appears in the recorded calls when driving the full pipeline."""
    def responses(argv):
        if argv[1:3] == ["config", "view"]:
            return DEFAULT_CONFIG_VIEW
        if argv[1:3] == ["datasets", "status"]:
            return FakeCompleted(0, "ready", "")
        if argv[1:3] == ["datasets", "files"]:
            return FakeCompleted(0, "name,size,creationDate\ncloud_job.json,103,2026-01-01", "")
        return FakeCompleted(0, "", "")

    manager = KaggleJobManager(tmp_path)
    cli, calls = make_cli(monkeypatch, responses=responses)
    manager.cli = cli
    job = KaggleJob(job_id="abc123", design_id="mydesign")

    with pytest.raises(KaggleTrainingError, match="training files are incomplete"):
        manager._run_pipeline(job, bundle_dir)

    assert job.state == "failed"
    assert not any(c[1] == "kernels" for c in calls)


def test_dataset_upload_is_never_automatically_retried(tmp_path, bundle_dir, monkeypatch):
    """A deliberate design decision (see docs/kaggle_training.md): on upload
    failure we fail clearly and preserve diagnostics rather than blindly
    retrying and risking multiple orphaned partial datasets. Locks that in."""
    def responses(argv):
        if argv[1:3] == ["config", "view"]:
            return DEFAULT_CONFIG_VIEW
        if argv[1:3] == ["datasets", "create"]:
            return FakeCompleted(1, "", "network error")
        return FakeCompleted(0, "", "")

    cli, calls = make_cli(monkeypatch, responses=responses)
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)

    with pytest.raises(KaggleTrainingError):
        manager.create_dataset(job, dataset_staging)

    create_calls = [c for c in calls if c[1:3] == ["datasets", "create"]]
    assert len(create_calls) == 1
    assert job.state == "failed"
    assert job.dataset_ref is not None  # the intended ref stays on record for diagnosis


# --- background worker (submit_async) ------------------------------------

def test_submit_async_returns_immediately_without_waiting_for_pipeline(tmp_path, bundle_dir, monkeypatch):
    cli, _ = make_cli(monkeypatch)
    manager = KaggleJobManager(tmp_path, cli=cli)

    pipeline_started = threading.Event()
    pipeline_may_finish = threading.Event()

    def slow_pipeline(job, bundle_dir_arg):
        pipeline_started.set()
        pipeline_may_finish.wait(timeout=5)
        job.state = "queued"
        save_job(tmp_path, job)

    monkeypatch.setattr(manager, "_run_pipeline", slow_pipeline)

    start = time.time()
    job = manager.submit_async("mydesign", bundle_dir)
    elapsed = time.time() - start

    assert elapsed < 1.0, "submit_async must return before the pipeline finishes"
    assert pipeline_started.wait(timeout=2), "background thread never started the pipeline"
    assert job.state == "preparing"  # pre-checks only; pipeline hasn't run yet

    pipeline_may_finish.set()
    for _ in range(50):
        reloaded = load_job(tmp_path, "mydesign", job.job_id)
        if reloaded.state == "queued":
            break
        time.sleep(0.05)
    assert reloaded.state == "queued"


def test_submit_async_persists_dataset_ref_before_long_upload_completes(tmp_path, bundle_dir, monkeypatch):
    """dataset_ref must be readable via job.json WHILE the upload is still
    in flight, not only after it finishes -- proves the "persist before
    upload" ordering end-to-end through the real create_dataset() path."""
    upload_started = threading.Event()
    upload_may_finish = threading.Event()

    class SlowPopen:
        def __init__(self, argv, shell, stdout, stderr, text, bufsize, encoding=None, errors=None):
            self.stdout = self
            upload_started.set()

        def readline(self):
            upload_may_finish.wait(timeout=5)
            return ""

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", SlowPopen)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: FakeCompleted(
        0, DEFAULT_CONFIG_VIEW.stdout if argv[1:3] == ["config", "view"] else "", "",
    ))

    # Short verify settling window: this test's fake CLI answers "datasets
    # status"/"datasets files" with empty/unrecognized responses (it only
    # cares about the upload phase), which would otherwise make the new
    # settling loop retry for its full default timeout on a background
    # thread that outlives this test's monkeypatch teardown -- letting a
    # stray subprocess.run call reach the REAL kaggle CLI once unmocked.
    manager = KaggleJobManager(tmp_path, dataset_verify_timeout_s=0.2, dataset_verify_interval_s=0.05)
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)

    def run_create_dataset():
        try:
            manager.create_dataset(job, dataset_staging)
        except KaggleTrainingError:
            pass

    t = threading.Thread(target=run_create_dataset, daemon=True)
    t.start()
    assert upload_started.wait(timeout=2)

    reloaded = load_job(tmp_path, "mydesign", job.job_id)
    assert reloaded.dataset_ref is not None
    assert reloaded.dataset_ref.startswith("testuser/")
    assert reloaded.state == "uploading_dataset"

    upload_may_finish.set()
    t.join(timeout=5)


# --- dataset verification settling/retry loop (Kaggle eventual consistency) --
#
# A real production dataset (andrzejmarczewski/hybrid-a2-20260905t200558z-
# ab2994879d66) proved `datasets files` can transiently 403 on the very first
# call right after a genuinely successful `datasets create`, and that the
# SAME dataset was fully readable and correct moments later. These tests
# drive verify_dataset_payload() directly with a scripted fake CLI (no real
# subprocess, no real waiting -- `interval` is always ~0) to prove the
# settling loop tolerates that without weakening the exact-size check for a
# genuinely incomplete/wrong payload.

class _SequenceCli:
    """A fake CLI whose datasets_status()/datasets_files() responses follow a
    scripted sequence (the last entry repeats once exhausted), so a test can
    say precisely "fail N times, then succeed" without any real subprocess or
    real waiting. Call counts are recorded for bounded-retry assertions."""

    def __init__(self, status_sequence=None, files_sequence=None):
        self._status_seq = list(status_sequence or [FakeCompleted(0, "ready", "")])
        self._files_seq = list(files_sequence or [FakeCompleted(0, "name,size,creationDate", "")])
        self.status_calls = 0
        self.files_calls = 0

    @staticmethod
    def _at(seq, count):
        return seq[min(count, len(seq) - 1)]

    @staticmethod
    def _to_result(fake: FakeCompleted) -> CliResult:
        return CliResult(ok=(fake.returncode == 0), returncode=fake.returncode, stdout=fake.stdout, stderr=fake.stderr)

    def datasets_status(self, dataset_ref, json_format=False):
        result = self._to_result(self._at(self._status_seq, self.status_calls))
        self.status_calls += 1
        return result

    def datasets_files(self, dataset_ref):
        result = self._to_result(self._at(self._files_seq, self.files_calls))
        self.files_calls += 1
        return result


def _make_complete_staging(tmp_path: Path) -> Path:
    staging = tmp_path / "verify_staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "input.wav").write_bytes(b"i" * 1000)
    (staging / "hybrid_target.wav").write_bytes(b"t" * 2000)
    (staging / "training_manifest.json").write_bytes(b"{}")
    (staging / "cloud_job.json").write_bytes(b"{}")
    return staging


def _complete_files_csv(staging_dir: Path) -> str:
    rows = ["name,size,creationDate"]
    for name in REQUIRED_DATASET_FILES:
        rows.append(f"{name},{(staging_dir / name).stat().st_size},2026-01-01 00:00:00")
    return "\n".join(rows)


def test_verify_retries_transient_403_then_succeeds(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = _complete_files_csv(staging)
    cli = _SequenceCli(files_sequence=[FakeCompleted(1, "", "403 Client Error: Forbidden"), FakeCompleted(0, csv, "")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is True
    assert error == ""
    assert cli.files_calls == 2
    assert job.dataset_verified_at is not None
    log = manager.read_log_tail(job)
    assert "403" in log
    assert "verified" in log.lower()


def test_verify_retries_transient_404_then_succeeds(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = _complete_files_csv(staging)
    cli = _SequenceCli(files_sequence=[FakeCompleted(1, "", "404 Not Found"), FakeCompleted(0, csv, "")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is True
    assert cli.files_calls == 2


def test_verify_status_processing_then_ready_then_files_success(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = _complete_files_csv(staging)
    cli = _SequenceCli(
        status_sequence=[FakeCompleted(0, "processing", ""), FakeCompleted(0, "ready", "")],
        files_sequence=[FakeCompleted(0, csv, "")],
    )
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is True
    assert cli.status_calls == 2
    assert cli.files_calls == 1  # files is only attempted once status says ready
    assert job.dataset_ready_at is not None


def test_verify_status_ready_but_files_still_403_then_retries(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = _complete_files_csv(staging)
    cli = _SequenceCli(
        status_sequence=[FakeCompleted(0, "ready", "")],
        files_sequence=[FakeCompleted(1, "", "403 Forbidden"), FakeCompleted(1, "", "403 Forbidden"), FakeCompleted(0, csv, "")],
    )
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is True
    assert cli.files_calls == 3
    # status is only polled again while not yet "ready" -- once ready, only files is retried.
    assert cli.status_calls == 1


def test_verify_repeated_403_until_timeout_fails(tmp_path):
    staging = _make_complete_staging(tmp_path)
    cli = _SequenceCli(files_sequence=[FakeCompleted(1, "", "403 Forbidden")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=0.05, interval=0.01)
    assert ok is False
    assert "could not be read back for verification" in error.lower()
    assert cli.files_calls >= 2  # actually retried, not just one attempt


def test_verify_repeated_404_until_timeout_fails(tmp_path):
    staging = _make_complete_staging(tmp_path)
    cli = _SequenceCli(files_sequence=[FakeCompleted(1, "", "404 Not Found")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=0.05, interval=0.01)
    assert ok is False
    assert cli.files_calls >= 2


def test_verify_readable_listing_missing_input_wav_fails_immediately(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = "name,size,creationDate\n" + "\n".join(
        f"{name},{(staging / name).stat().st_size},2026-01-01"
        for name in ("hybrid_target.wav", "training_manifest.json", "cloud_job.json")
    )
    cli = _SequenceCli(files_sequence=[FakeCompleted(0, csv, "")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is False
    assert "input.wav" in error
    assert cli.files_calls == 1  # readable-but-wrong fails immediately, never retried


def test_verify_readable_listing_missing_hybrid_target_fails_immediately(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = "name,size,creationDate\n" + "\n".join(
        f"{name},{(staging / name).stat().st_size},2026-01-01"
        for name in ("input.wav", "training_manifest.json", "cloud_job.json")
    )
    cli = _SequenceCli(files_sequence=[FakeCompleted(0, csv, "")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is False
    assert "hybrid_target.wav" in error
    assert cli.files_calls == 1


def test_verify_readable_listing_wrong_byte_size_fails_immediately(tmp_path):
    staging = _make_complete_staging(tmp_path)
    csv = "name,size,creationDate\n" + "\n".join(
        f"{name},{999999 if name == 'input.wav' else (staging / name).stat().st_size},2026-01-01"
        for name in REQUIRED_DATASET_FILES
    )
    cli = _SequenceCli(files_sequence=[FakeCompleted(0, csv, "")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d")

    ok, error = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is False
    assert "size mismatch" in error
    assert cli.files_calls == 1  # readable-but-wrong fails immediately, never retried


def test_verify_settling_never_starts_kernel_creation(tmp_path, bundle_dir, monkeypatch):
    """End-to-end through _run_pipeline: a transient 403 that later resolves
    must let the pipeline proceed to a real kernel push -- but only after
    verification actually passes, never before."""
    manager = KaggleJobManager(tmp_path, kernel_verify_delay_s=0)
    job = KaggleJob(job_id="abc123", design_id="mydesign")
    dataset_staging, _kernel_staging = manager.stage(job, bundle_dir)
    csv_holder = {"csv": None}

    def responses(argv):
        if argv[1:3] == ["config", "view"]:
            return DEFAULT_CONFIG_VIEW
        if argv[1:3] == ["datasets", "status"]:
            return FakeCompleted(0, "ready", "")
        if argv[1:3] == ["datasets", "files"]:
            if csv_holder["csv"] is None:
                csv_holder["csv"] = "seen"
                return FakeCompleted(1, "", "403 Forbidden")
            return _auto_datasets_files_response(dataset_staging)
        return FakeCompleted(0, "", "")

    cli, calls = make_cli(monkeypatch, responses=responses)
    manager.cli = cli

    kernel_calls_before_verified = []

    real_create_kernel = manager.create_kernel

    def wrapped_create_kernel(job_arg, staging_arg):
        assert job_arg.dataset_verified_at is not None, "kernel creation started before dataset verification completed"
        return real_create_kernel(job_arg, staging_arg)

    manager.create_kernel = wrapped_create_kernel
    manager._run_pipeline(job, bundle_dir)

    assert job.state == "queued"
    assert job.dataset_verified_at is not None
    files_calls = [c for c in calls if c[1:3] == ["datasets", "files"]]
    assert len(files_calls) == 2  # bounded: one transient 403, one success -- not unbounded


def test_verify_dataset_payload_job_state_stays_verifying_during_retries(tmp_path):
    """job.state is set to "verifying_dataset" by create_dataset() before the
    settling loop starts and stays there across retries -- confirmed here by
    checking it's never mutated by verify_dataset_payload itself (only the
    caller, create_dataset, transitions state), so UI polling mid-retry keeps
    reporting the correct state."""
    staging = _make_complete_staging(tmp_path)
    csv = _complete_files_csv(staging)
    cli = _SequenceCli(files_sequence=[FakeCompleted(1, "", "403 Forbidden"), FakeCompleted(0, csv, "")])
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j", design_id="d", state="verifying_dataset")
    save_job(tmp_path, job)

    ok, _ = manager.verify_dataset_payload("owner/ds", staging, job=job, timeout=5, interval=0.001)
    assert ok is True
    assert job.state == "verifying_dataset"  # unchanged by verify_dataset_payload itself

    reloaded = load_job(tmp_path, "d", "j")
    assert reloaded.state == "verifying_dataset"


# --- kernel output download (the `--file-pattern` regex incident) -------
#
# A real production job (andrzejmarczewski/hybrid-a2-train-20260905t202635z-
# e919ae705ff4) completed training on Kaggle (KernelWorkerStatus.COMPLETE)
# but the app itself failed with "Invalid regex pattern '*.nam|*.json':
# nothing to repeat at position 0" -- Kaggle's `--file-pattern` is a REGEX,
# not a shell glob, and that string was never valid regex. Production now
# downloads the whole (small) kernel output directory and filters locally.

def _write_nam(path: Path) -> Path:
    path.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000.0}), encoding="utf-8")
    return path


def _fake_render(model, audio, sr, **kwargs):
    return np.asarray(audio, dtype=np.float32).copy()


class _DownloadStubCli:
    """Stand-in for KaggleCli implementing only what
    _download_and_validate()/retry_download() actually call. Records every
    kernels_output() call (in particular: whether a file_pattern was ever
    passed) without touching the filesystem -- tests populate the output
    directory directly to simulate what a real `kaggle kernels output -p
    <dir>` download would have written. datasets_create/kernels_push raise
    if ever called, since recovery must never re-create either."""

    def __init__(self, kernel_status_text="andrzejmarczewski/foo has status \"KernelWorkerStatus.COMPLETE\"",
                 kernel_status_ok=True, output_ok=True, output_error="",
                 nam_name="hybrid_a2.nam", training_result=None, extra_files=None,
                 write_despite_failure=False):
        self.kernel_status_text = kernel_status_text
        self.kernel_status_ok = kernel_status_ok
        self.output_ok = output_ok
        self.output_error = output_error
        self.nam_name = nam_name
        self.training_result = training_result if training_result is not None else {"success": True}
        self.extra_files = extra_files
        self.write_despite_failure = write_despite_failure
        self.kernels_output_calls: list[dict] = []

    def kernels_status(self, kernel_ref):
        return CliResult(ok=self.kernel_status_ok, returncode=0 if self.kernel_status_ok else 1,
                          stdout=self.kernel_status_text, stderr="")

    def kernels_output(self, kernel_ref, out_dir, file_pattern=None):
        self.kernels_output_calls.append({"kernel_ref": kernel_ref, "out_dir": Path(out_dir), "file_pattern": file_pattern})
        if not self.output_ok:
            if self.write_despite_failure:
                # Reproduces the real Kaggle-CLI-on-Windows incident: every
                # file is written successfully, but the CLI process still
                # exits non-zero due to its own internal unicode-printing
                # crash after the download finished.
                _populate_output(Path(out_dir), nam_name=self.nam_name, training_result=self.training_result,
                                  extra_files=self.extra_files)
            return CliResult(ok=False, returncode=1, stdout="", stderr=self.output_error)
        # Simulates what a real `kaggle kernels output -p <dir>` download
        # would have written -- called AFTER _download_and_validate's own
        # pre-clear of `out_dir`, exactly like the real CLI would populate
        # an empty directory.
        _populate_output(Path(out_dir), nam_name=self.nam_name, training_result=self.training_result,
                          extra_files=self.extra_files)
        return CliResult(ok=True, returncode=0, stdout="", stderr="")

    def datasets_create(self, *a, **k):
        raise AssertionError("recovery must never create a new dataset")

    def datasets_create_streaming(self, *a, **k):
        raise AssertionError("recovery must never create a new dataset")

    def kernels_push(self, *a, **k):
        raise AssertionError("recovery must never push a new kernel")


def _populate_output(output_dir: Path, nam_name="hybrid_a2.nam", training_result=None, extra_files=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_nam(output_dir / nam_name)
    if training_result is not None:
        (output_dir / "training_result.json").write_text(json.dumps(training_result), encoding="utf-8")
    for name, content in (extra_files or {}).items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


def _output_dir_for(tmp_path: Path, design_id: str, job_id: str) -> Path:
    return _job_dir(tmp_path, design_id, job_id) / "output"


def _write_bundle_wavs(a2_output_dir: Path, design_id: str, n: int = 1000, sr: int = 48000) -> None:
    """validate_downloaded_model() reads the design's own input.wav/
    hybrid_target.wav (as bundle_dir / "input.wav" next to the job's kaggle/
    subdirectory) to compare against -- these must exist for
    _download_and_validate()/retry_download() to reach local validation."""
    bundle_dir = a2_output_dir / design_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(n, dtype=np.float32)
    sf.write(bundle_dir / "input.wav", audio, sr, subtype="FLOAT")
    sf.write(bundle_dir / "hybrid_target.wav", audio, sr, subtype="FLOAT")


def test_validate_downloaded_model_runs_low_level_response_check_when_manifest_is_character_mode(monkeypatch, tmp_path):
    """docs/blend-mode-fixes.md Phase 10/11: a Kaggle-trained Character
    Blend A2 gets the same low-level-response bar as a locally-trained one,
    via the SAME hybrid.character_training_target.check_full_low_level_
    response function scripts/train_a2.py uses."""
    import hybrid.character_training_target as character_training_target

    nam_path = _write_nam(tmp_path / "model.nam")
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    audio = np.zeros(4800, dtype=np.float32)
    sf.write(input_path, audio, 48000, subtype="FLOAT")
    sf.write(target_path, audio, 48000, subtype="FLOAT")

    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    monkeypatch.setattr(character_training_target, "render", _fake_render)

    manifest = {"mode": "character", "low_level_response": {"levels_db": [0.0], "output_rms_dbfs": [-120.0], "dead_zone_detected": False}}
    report = kaggle_training.validate_downloaded_model(nam_path, input_path, target_path, manifest=manifest)
    assert "low_level_response_check" in report

    report_without_manifest = kaggle_training.validate_downloaded_model(nam_path, input_path, target_path)
    assert "low_level_response_check" not in report_without_manifest


def test_kernels_output_never_passes_the_invalid_glob_pattern(monkeypatch, tmp_path):
    """Reproduces the real incident: `_download_and_validate` must call
    kernels_output() with no file_pattern at all -- never the glob-shaped
    string that broke production."""
    cli = _DownloadStubCli(training_result={"success": True})
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="e919ae705ff4", design_id="mydesign", state="downloading",
                     kernel_ref="testuser/some-kernel")
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, "mydesign")

    manager._download_and_validate(job)

    assert len(cli.kernels_output_calls) == 1
    assert cli.kernels_output_calls[0]["file_pattern"] is None


def test_kernels_output_rejects_glob_shaped_pattern_as_invalid_regex(tmp_path):
    """If file_pattern support is ever used again, it must be validated as a
    REAL regex up front -- `*.nam|*.json` (a glob, not a regex) must never
    reach the Kaggle CLI, since that's exactly what broke production."""
    cli = KaggleCli(executable="/usr/bin/kaggle")
    with pytest.raises(ValueError, match="not a valid regex"):
        cli.kernels_output("owner/kernel", tmp_path, file_pattern="*.nam|*.json")


def test_kernels_output_accepts_a_real_regex_pattern(monkeypatch, tmp_path):
    cli, calls = make_cli(monkeypatch)
    cli.kernels_output("owner/kernel", tmp_path, file_pattern=r".*\.nam$")
    assert calls[0] == ["/usr/bin/kaggle", "kernels", "output", "owner/kernel", "-p", str(tmp_path),
                         "--file-pattern", r".*\.nam$"]


def test_download_and_validate_discovers_nam_and_training_result(monkeypatch, tmp_path):
    cli = _DownloadStubCli(nam_name="hybrid_a2.nam", training_result={"success": True, "epochs": 100})
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, "d1")

    manager._download_and_validate(job)

    assert job.state == "complete"
    assert job.output_nam_path is not None
    assert Path(job.output_nam_path).name == "hybrid_a2.nam"
    assert job.training_result == {"success": True, "epochs": 100}
    assert job.local_validation is not None
    assert job.local_validation["full"]["rendered_ok"] is True
    assert job.local_validation["lite"]["rendered_ok"] is True


def test_download_and_validate_unrelated_files_do_not_interfere(monkeypatch, tmp_path):
    """A real kernel output directory also contains lightning_logs/,
    checkpoint files, and a training log -- none of that should confuse
    discovery of the one .nam and one training_result.json that matter."""
    cli = _DownloadStubCli(
        nam_name="hybrid_a2.nam", training_result={"success": True},
        extra_files={
            "hybrid-a2-train-job.log": "",
            "a2_output/packed_best.json": "{}",
            "a2_output/packed_best_submodel_0.ckpt": b"\x00" * 16,
            "a2_output/lightning_logs/version_0/events.out": b"\x00",
        },
    )
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, "d1")

    manager._download_and_validate(job)

    assert job.state == "complete"
    assert Path(job.output_nam_path).name == "hybrid_a2.nam"


def test_download_and_validate_missing_nam_fails_clearly(tmp_path):
    def _write_result_only(kernel_ref, out_dir, file_pattern=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "training_result.json").write_text(json.dumps({"success": True}), encoding="utf-8")
        return CliResult(ok=True, returncode=0, stdout="", stderr="")

    cli = type("Cli", (), {"kernels_output": staticmethod(_write_result_only)})()
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")

    manager._download_and_validate(job)

    assert job.state == "failed"
    assert "no .nam file" in job.error


def test_download_and_validate_download_failure_still_fails_clearly(tmp_path):
    """The exact real incident, reproduced directly: kernels_output()
    failing with the CLI's own regex-rejection error must still surface as a
    clear job failure -- proves the failure path itself (pre-fix behavior)
    without needing the real Kaggle CLI."""
    cli = _DownloadStubCli(output_ok=False, output_error="Invalid regex pattern '*.nam|*.json': nothing to repeat at position 0")
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")

    manager._download_and_validate(job)

    assert job.state == "failed"
    assert "output download failed" in job.error


def test_download_and_validate_tolerates_nonzero_exit_when_files_were_written(monkeypatch, tmp_path):
    """A real recovery of a genuinely completed job hit exactly this: the
    installed Kaggle CLI wrote every output file successfully (byte-for-byte
    identical to a from-scratch download, confirmed via SHA256) but still
    exited non-zero, crashing internally with `'charmap' codec can't encode
    characters...` while printing its own progress on a Windows console
    whose codepage can't represent every character it prints. The exit code
    is not trustworthy here; what actually landed on disk is."""
    cli = _DownloadStubCli(
        output_ok=False, write_despite_failure=True,
        output_error="'charmap' codec can't encode characters in position 519-558: character maps to <undefined>",
        training_result={"success": True},
    )
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, "d1")

    manager._download_and_validate(job)

    assert job.state == "complete"
    assert job.error is None
    assert Path(job.output_nam_path).name == "hybrid_a2.nam"
    log = manager.read_log_tail(job)
    assert "non-fatal" in log.lower()


def test_download_and_validate_nonzero_exit_with_nothing_written_still_fails(tmp_path):
    """The tolerance above must not become a blanket "ignore all CLI
    failures" -- a download that reports failure AND wrote nothing at all is
    still a real, immediate failure."""
    cli = _DownloadStubCli(output_ok=False, write_despite_failure=False, output_error="403 Forbidden")
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")

    manager._download_and_validate(job)

    assert job.state == "failed"
    assert "output download failed" in job.error


def test_download_and_validate_clears_stale_partial_output_first(monkeypatch, tmp_path):
    """A previous failed/interrupted download attempt could leave a stale
    .nam or json behind on disk -- _download_and_validate must clear the
    output directory BEFORE invoking kernels_output, so a stale file can
    never be mistaken for the current attempt's result even if kernels_
    output itself doesn't happen to overwrite/remove it."""
    manager = KaggleJobManager(tmp_path)
    job = KaggleJob(job_id="j1", design_id="d1", state="downloading", kernel_ref="testuser/k1")
    output_dir = _output_dir_for(tmp_path, "d1", "j1")
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, "d1")

    # Stale leftover from a previous attempt, sitting in the output dir
    # BEFORE this download attempt starts.
    _populate_output(output_dir, nam_name="stale_old_model.nam", training_result={"success": True})
    assert (output_dir / "stale_old_model.nam").is_file()

    # This attempt's kernels_output writes only the fresh, correct file --
    # it does NOT know about (and would not remove) the stale one itself;
    # only _download_and_validate's own pre-clear can be responsible for it
    # being gone afterward.
    def _write_fresh_output(kernel_ref, out_dir, file_pattern=None):
        _populate_output(Path(out_dir), nam_name="hybrid_a2.nam", training_result={"success": True})
        return CliResult(ok=True, returncode=0, stdout="", stderr="")

    manager.cli = type("Cli", (), {"kernels_output": staticmethod(_write_fresh_output)})()
    manager._download_and_validate(job)

    assert job.state == "complete"
    assert not (output_dir / "stale_old_model.nam").is_file()
    assert (output_dir / "hybrid_a2.nam").is_file()


# --- recovery: completed kernel, failed local download -------------------

def _failed_download_job(design_id="mydesign", job_id="j1") -> KaggleJob:
    return KaggleJob(
        job_id=job_id, design_id=design_id, state="failed",
        dataset_ref=f"testuser/hybrid-a2-{design_id}-{job_id}",
        kernel_ref=f"testuser/hybrid-a2-train-{design_id}-{job_id}",
        error="output download failed: Invalid regex pattern '*.nam|*.json': nothing to repeat at position 0",
    )


def test_retry_download_recovers_completed_job_without_new_kernel_or_dataset(monkeypatch, tmp_path):
    cli = _DownloadStubCli(training_result={"success": True})
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = _failed_download_job()
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, job.design_id)

    recovered = manager.retry_download(job)

    assert recovered.state == "complete"
    assert recovered.error is None
    assert recovered.local_validation is not None
    assert recovered.local_validation["full"]["rendered_ok"] is True
    assert recovered.local_validation["lite"]["rendered_ok"] is True
    # datasets_create/kernels_push raise on _DownloadStubCli if ever called --
    # reaching `complete` here already proves neither was invoked.


def test_retry_download_transitions_failed_downloading_validating_complete(monkeypatch, tmp_path):
    seen_states: list[str] = []
    cli = _DownloadStubCli(training_result={"success": True})
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = _failed_download_job()
    monkeypatch.setattr(kaggle_training, "load_nam", lambda path: object())
    monkeypatch.setattr(kaggle_training, "render", _fake_render)
    _write_bundle_wavs(tmp_path, job.design_id)

    real_save_job = kaggle_training.save_job
    def recording_save_job(a2_output_dir, j):
        seen_states.append(j.state)
        return real_save_job(a2_output_dir, j)
    monkeypatch.setattr(kaggle_training, "save_job", recording_save_job)

    manager.retry_download(job)

    assert seen_states[0] == "downloading"
    assert "validating" in seen_states
    assert seen_states[-1] == "complete"


def test_retry_download_refuses_when_remote_kernel_not_complete(tmp_path):
    cli = _DownloadStubCli(kernel_status_text="andrzejmarczewski/foo has status \"KernelWorkerStatus.RUNNING\"")
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = _failed_download_job()

    with pytest.raises(KaggleTrainingError, match="not reporting a completed status"):
        manager.retry_download(job)
    assert job.state == "failed"  # never moved to downloading
    assert not cli.kernels_output_calls


def test_retry_download_refuses_non_failed_job(tmp_path):
    cli = _DownloadStubCli()
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = _failed_download_job()
    job.state = "running"

    with pytest.raises(KaggleTrainingError, match="can only recover"):
        manager.retry_download(job)


def test_retry_download_refuses_job_with_no_kernel_ref(tmp_path):
    cli = _DownloadStubCli()
    manager = KaggleJobManager(tmp_path, cli=cli)
    job = _failed_download_job()
    job.kernel_ref = None

    with pytest.raises(KaggleTrainingError, match="no kernel_ref"):
        manager.retry_download(job)

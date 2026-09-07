"""Kaggle GPU training backend -- see docs/kaggle_training.md.

Service layer between Flask (`app.py`'s `/api/kaggle/*` routes) and the
Kaggle CLI. Deliberately torch-free and import-safe in the normal Flask
runtime environment: this module only ever shells out to the external
`kaggle` executable, never imports `kaggle` (the Python package) or any
training dependency itself.

Two costs, mirroring hybrid/pipeline.py's render_pair()/build_hybrid() split:

- `KaggleCli` is a thin, fully-mocked-in-tests wrapper around individual
  `kaggle ...` subprocess invocations. It never raises for an expected
  failure (missing CLI, not authenticated, quota unavailable) -- callers
  inspect `.ok`/`.stdout`/`.stderr` and decide what that means.
- `KaggleJobManager` orchestrates a `KaggleJob`'s state machine: stage ->
  create private dataset -> create private kernel -> poll -> download ->
  locally validate -> complete (or fail, with Kaggle artefacts preserved).

Security invariants enforced throughout:
  - every subprocess call uses an argv list, never shell=True
  - Kaggle credentials are never read, stored, or logged by this module --
    authentication is entirely the installed `kaggle` CLI's own business
  - datasets/kernels created here are always private
  - only an explicit allow-list of files is ever staged/uploaded
  - the accelerator is always NvidiaTeslaT4 -- P100 is never requested
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .a2_training_settings import A2_EPOCH_PRESETS, DEFAULT_EPOCH_PRESET
from .nam_loader import load_nam
from .render import NamRenderError, render
from .validation import compute_esr_metrics

ACCELERATOR = "NvidiaTeslaT4"
FORBIDDEN_ACCELERATORS = {"NvidiaTeslaP100", "TPU"}

# `kaggle datasets create` for the real ~64MB training payload was observed
# to hang for 10+ minutes in production with no error surfaced -- root-cause
# investigation (direct CLI reproduction, inspecting the installed kaggle
# package's ResumableUploadContext) found the upload mechanism itself works
# correctly on this machine/network (repeatable ~22s uploads of the full
# payload), but Kaggle's own client-side retry loop is bounded yet can still
# run very long under real transient network conditions (up to
# MAX_UPLOAD_RESUME_ATTEMPTS=10 resumable-upload attempts, each with its own
# HTTP-level Retry(total=10, backoff_factor=0.5)) -- and our old fully
# synchronous, un-streamed subprocess.run(capture_output=True) gave zero
# visibility while that ran and left the job frozen with no error if the
# Flask process was interrupted mid-upload. This timeout is generous
# specifically because that retry behavior is legitimate, not a hang to
# short-circuit aggressively.
DATASET_UPLOAD_TIMEOUT_S = 1800

# A real production dataset (andrzejmarczewski/hybrid-a2-20260905t200558z-
# ab2994879d66) proved that `datasets create` exiting 0 can be immediately
# followed by a transient 403 from `datasets files`/`datasets status` --
# manually re-checking the SAME dataset moments later showed status "ready"
# and a complete, correctly-sized file listing. That is Kaggle-side eventual
# consistency after dataset creation, not a real permission or content
# problem, so verification must tolerate a bounded settling window rather
# than failing on the first transient response. This window is much shorter
# than DATASET_UPLOAD_TIMEOUT_S above -- it's for post-create propagation,
# not for the upload itself.
DATASET_VERIFY_TIMEOUT_S = 120
DATASET_VERIFY_INTERVAL_S = 3

# `kernels push` returning exit code 0 is NOT sufficient evidence the kernel
# actually exists on Kaggle's side -- a real production run observed a
# "submitted" job whose kernel could never be resolved (wrong/bare slug).
# Bound the post-push existence check so a Kaggle eventual-consistency delay
# doesn't fail us immediately, without blocking forever either.
KERNEL_VERIFY_ATTEMPTS = 3
KERNEL_VERIFY_DELAY_S = 5

# Allow-list of files staged into the Kaggle dataset for a job -- nothing
# else is ever copied out of a design's bundle dir, in particular never
# assets/nam_models/*.nam or any preview DI (see docs/kaggle_training.md).
STAGED_BUNDLE_FILES = ("input.wav", "hybrid_target.wav", "training_manifest.json")

# The files that MUST actually exist in the remote dataset, with sizes
# matching the local staged copies, before a kernel is ever created against
# it. `datasets status == ready` is NOT sufficient evidence of this -- see
# KaggleCli.datasets_files's docstring for the real incident that proved it.
REQUIRED_DATASET_FILES = (*STAGED_BUNDLE_FILES, "cloud_job.json")

# Bounded so a Flask route never returns an unbounded log file.
LOG_TAIL_LINES = 200

_SECRET_PATTERN = re.compile(r"(?i)(key|token|secret|password)\s*[:=]\s*\S+")


def _redact(text: str) -> str:
    if not text:
        return text
    return _SECRET_PATTERN.sub(lambda m: m.group(0).split(next(c for c in ":=" if c in m.group(0)))[0] + "=<redacted>", text)


def _safe_slug(text: str, max_len: int = 40) -> str:
    """Kaggle dataset/kernel slugs: lowercase alnum + dashes only."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text)).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug or "job")[:max_len].strip("-") or "job"


def _parse_datasets_files_csv(text: str) -> dict[str, int]:
    """Parses `kaggle datasets files <ref> -v`'s CSV output
    (`name,size,creationDate` header + one row per file) into
    {filename: size_in_bytes}. Never raises on a malformed/empty row --
    verification callers treat a name that fails to parse as absent."""
    sizes: dict[str, int] = {}
    reader = csv.DictReader(io.StringIO(text.strip()))
    for row in reader:
        name = (row.get("name") or "").strip()
        size_str = (row.get("size") or "").strip()
        if not name:
            continue
        try:
            sizes[name] = int(size_str)
        except ValueError:
            continue
    return sizes


def _parse_dataset_status(result: "CliResult") -> Optional[str]:
    """Extracts a normalized status string (e.g. "ready") from a
    `datasets_status` CliResult, whether it came back as `--format json`
    (`{"status": "ready", ...}`) or plain text. Returns None only when the
    call itself failed -- callers must not confuse "failed to check" with
    "checked and not ready"."""
    if not result.ok:
        return None
    text = result.combined.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "status" in data:
            return str(data["status"]).strip().lower()
    except (json.JSONDecodeError, TypeError):
        pass
    lower = text.lower()
    if "ready" in lower:
        return "ready"
    return lower or None


@dataclass
class CliResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


class KaggleCliUnavailable(RuntimeError):
    """Raised only for a truly unrecoverable call -- most callers should
    prefer inspecting a CliResult instead of catching this."""


class KaggleCli:
    """Thin wrapper around the `kaggle` executable. Every method returns a
    CliResult rather than raising, except `_run` itself raising
    KaggleCliUnavailable when the executable cannot be found/started at all
    -- that is the one case every caller genuinely cannot proceed past.
    """

    def __init__(self, executable: Optional[str] = None, timeout: int = 120):
        self.executable = executable or self._locate()
        self.timeout = timeout

    @staticmethod
    def _locate() -> Optional[str]:
        found = shutil.which("kaggle")
        if found:
            return found
        return None

    @staticmethod
    def _module_available() -> bool:
        """The macOS user-site console-script directory is commonly absent
        from GUI-app PATHs.  The package is still a valid CLI via
        ``sys.executable -m kaggle``; detect it without importing Kaggle or
        touching its credential directory."""
        return importlib.util.find_spec("kaggle") is not None

    def is_installed(self) -> bool:
        return self.executable is not None or self._module_available()

    def _build_argv(self, args: list[str]) -> list[str]:
        if self.executable is None:
            # Fall back to `python -m kaggle` in case the console script
            # isn't on PATH but the package is importable in this interpreter.
            return [sys.executable, "-m", "kaggle", *args]
        return [self.executable, *args]

    def _run(self, args: list[str], timeout: Optional[int] = None) -> CliResult:
        argv = self._build_argv(args)
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                # A real `kernels output` download failed with
                # "'charmap' codec can't encode/decode..." -- `text=True`
                # without an explicit encoding decodes with the OS's default
                # locale encoding, which on Windows is a legacy codepage
                # (e.g. cp1252) that cannot represent every character
                # Kaggle's CLI prints (checkmarks, etc). UTF-8 with
                # replacement never raises on unexpected bytes.
                encoding="utf-8",
                errors="replace",
                timeout=timeout or self.timeout,
            )
        except FileNotFoundError as exc:
            raise KaggleCliUnavailable(f"kaggle CLI not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            return CliResult(ok=False, returncode=-1, stdout=_redact(exc.stdout or ""), stderr=f"timed out after {timeout or self.timeout}s")
        return CliResult(
            ok=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=_redact(proc.stdout or ""),
            stderr=_redact(proc.stderr or ""),
        )

    def run_streaming(self, args: list[str], log_path: Path, timeout: int) -> CliResult:
        """For long operations (dataset upload) -- streams combined
        stdout+stderr line-by-line into `log_path` as they arrive, instead of
        `_run`'s capture_output=True (which only returns output after the
        whole process exits, giving zero visibility during a multi-minute
        upload and no way to show elapsed progress).

        stderr is merged into stdout (never a second pipe) specifically to
        avoid the classic two-pipe deadlock; a background reader thread
        drains the single pipe into a queue so the timeout can be checked
        even during a period of total silence from the child (readline()
        alone would block past the timeout if the child produced no output
        at all -- Windows pipes have no select()-based readline-with-timeout).
        """
        argv = self._build_argv(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.Popen(
                argv, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError as exc:
            raise KaggleCliUnavailable(f"kaggle CLI not found: {exc}") from exc

        line_queue: "queue.Queue[Optional[str]]" = queue.Queue()

        def _reader() -> None:
            try:
                for line in iter(proc.stdout.readline, ""):
                    line_queue.put(line)
            finally:
                line_queue.put(None)  # sentinel: stream closed

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        start = time.time()
        lines: list[str] = []
        timed_out = False
        with open(log_path, "a", encoding="utf-8") as logf:
            while True:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = line_queue.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
                if line is None:
                    break
                redacted = _redact(line)
                logf.write(redacted)
                logf.flush()
                lines.append(redacted)

        if timed_out:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return CliResult(ok=False, returncode=-1, stdout="".join(lines), stderr=f"timed out after {timeout}s")

        try:
            returncode = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = -1
        return CliResult(ok=(returncode == 0), returncode=returncode, stdout="".join(lines), stderr="")

    def version(self) -> Optional[str]:
        if not self.is_installed():
            return None
        result = self._run(["--version"], timeout=15)
        if not result.ok:
            return None
        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", result.combined)
        return match.group(1) if match else result.combined.strip() or None

    def is_authenticated(self) -> bool:
        """A harmless authenticated call whose exit status tells us whether
        credentials are valid -- deliberately NOT `auth print-access-token`
        (we never need or want to see the secret value)."""
        if not self.is_installed():
            return False
        result = self._run(["kernels", "list", "-m", "--page-size", "1"], timeout=30)
        return result.ok

    def supports(self, *subcommand: str) -> bool:
        """Feature-detect a subcommand via its own --help, so we never call
        a flag/command an older installed CLI doesn't have."""
        if not self.is_installed():
            return False
        result = self._run([*subcommand, "--help"], timeout=15)
        return result.ok

    def quota(self) -> dict:
        if not self.supports("quota"):
            return {"quota_available": False, "quota_error": "installed CLI does not support `kaggle quota`"}
        result = self._run(["quota"], timeout=30)
        if not result.ok:
            return {"quota_available": False, "quota_error": result.stderr.strip() or "quota command failed"}
        return {"quota_available": True, "quota_raw": result.stdout.strip()}

    def username(self) -> Optional[str]:
        """The authenticated Kaggle username -- NOT a credential (it's the
        public handle, not a secret), needed because a Kaggle dataset's
        `id` must be `"<username>/<slug>"`; the installed CLI's own
        `dataset_create_new` crashes with an IndexError on a bare slug (it
        does `ref.split("/")[1]` unconditionally). `kaggle config view` is
        the CLI's own documented way to read it back -- never
        `auth print-access-token`."""
        if not self.is_installed():
            return None
        result = self._run(["config", "view"], timeout=15)
        if not result.ok:
            return None
        match = re.search(r"username:\s*(\S+)", result.combined)
        return match.group(1) if match else None

    def datasets_create(self, dataset_dir: Path) -> CliResult:
        return self._run(["datasets", "create", "-p", str(dataset_dir)], timeout=600)

    def datasets_create_streaming(self, dataset_dir: Path, log_path: Path, timeout: int = DATASET_UPLOAD_TIMEOUT_S) -> CliResult:
        """Like `datasets_create`, but streams progress into `log_path` as it
        happens instead of returning only once the whole upload finishes --
        see `run_streaming`'s docstring for why this matters for a
        potentially multi-minute upload."""
        return self.run_streaming(["datasets", "create", "-p", str(dataset_dir)], log_path, timeout=timeout)

    def datasets_status(self, dataset_ref: str, json_format: bool = False) -> CliResult:
        args = ["datasets", "status", dataset_ref]
        if json_format:
            args += ["--format", "json"]
        return self._run(args, timeout=30)

    def supports_datasets_status_json(self) -> bool:
        """Feature-detects `--format json` on `datasets status` via its own
        --help text, so we never pass a flag an older installed CLI doesn't
        recognize. Best-effort: any failure just means we fall back to
        parsing plain-text status output."""
        if not self.is_installed():
            return False
        result = self._run(["datasets", "status", "--help"], timeout=15)
        return result.ok and "--format" in result.combined

    def datasets_files(self, dataset_ref: str) -> CliResult:
        """`kaggle datasets files <ref> -v` -- CSV listing of what actually
        landed remotely. Never trust `datasets_status`'s "ready" alone: a
        real production incident showed Kaggle reporting "ready" for a
        dataset that contained only 1 of 6 intended files (the upload never
        reached the point of comparing against the client's actual intent),
        so this is the mandatory follow-up check."""
        return self._run(["datasets", "files", dataset_ref, "-v"], timeout=30)

    def datasets_delete(self, dataset_ref: str) -> CliResult:
        return self._run(["datasets", "delete", dataset_ref, "--yes"], timeout=60)

    def kernels_push(self, kernel_dir: Path, accelerator: str = ACCELERATOR) -> CliResult:
        if accelerator in FORBIDDEN_ACCELERATORS:
            raise ValueError(f"refusing to push with forbidden accelerator {accelerator!r}")
        return self._run(["kernels", "push", "-p", str(kernel_dir), "--accelerator", accelerator], timeout=300)

    def kernels_status(self, kernel_ref: str) -> CliResult:
        return self._run(["kernels", "status", kernel_ref], timeout=30)

    def kernels_logs(self, kernel_ref: str) -> CliResult:
        if not self.supports("kernels", "logs"):
            return CliResult(ok=False, returncode=-1, stdout="", stderr="installed CLI does not support `kaggle kernels logs`")
        return self._run(["kernels", "logs", kernel_ref], timeout=60)

    def kernels_output(self, kernel_ref: str, out_dir: Path, file_pattern: Optional[str] = None) -> CliResult:
        """Downloads a kernel's output files. `file_pattern` is passed
        straight to Kaggle's own `--file-pattern`, which Kaggle documents and
        implements as a REGULAR EXPRESSION, not a shell glob -- a real
        production job failed with `Invalid regex pattern '*.nam|*.json':
        nothing to repeat at position 0` because that string is a glob, not
        valid regex (a bare leading `*` has nothing to repeat). Validating
        it here catches that class of mistake immediately instead of only at
        the Kaggle API boundary. Production code should prefer omitting
        `file_pattern` entirely (see `_download_and_validate`) and filtering
        the small, fully-downloaded output locally instead."""
        args = ["kernels", "output", kernel_ref, "-p", str(out_dir)]
        if file_pattern:
            try:
                re.compile(file_pattern)
            except re.error as exc:
                raise ValueError(
                    f"file_pattern {file_pattern!r} is not a valid regex -- Kaggle's "
                    f"--file-pattern is a REGEX, not a shell glob ({exc})"
                ) from exc
            args += ["--file-pattern", file_pattern]
        return self._run(args, timeout=600)

    def kernels_delete(self, kernel_ref: str) -> CliResult:
        return self._run(["kernels", "delete", kernel_ref, "--yes"], timeout=60)


# --- Job state machine -----------------------------------------------------

JOB_STATES = (
    "preparing", "uploading_dataset", "verifying_dataset", "creating_kernel",
    "verifying_kernel", "queued", "running", "downloading", "validating",
    "complete", "failed", "cleanup_pending", "cleaned",
)
TERMINAL_STATES = ("complete", "failed")
# Note: a job.json written by a pre-rewrite version of this module may still
# have state "uploading"/"waiting_for_dataset"/"submitted" -- every check
# below tests `state not in TERMINAL_STATES` (never an exact new-state
# match), so those old values keep behaving as non-terminal without needing
# a migration table.

# Kaggle's own kernel statuses, mapped defensively -- anything unrecognized
# falls back to "running" rather than raising, per docs/kaggle_training.md
# ("tolerant of unexpected strings").
_KERNEL_STATUS_MAP = {
    "queued": "queued",
    "running": "running",
    "complete": "downloading",
    "error": "failed",
    "cancelAcknowledged": "failed",
    "cancelRequested": "failed",
}


@dataclass
class KaggleJob:
    job_id: str
    design_id: str
    state: str = "preparing"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    dataset_ref: Optional[str] = None
    kernel_ref: Optional[str] = None
    accelerator: str = ACCELERATOR
    epoch_preset: str = DEFAULT_EPOCH_PRESET
    upload_completed_at: Optional[float] = None
    dataset_ready_at: Optional[float] = None
    dataset_verified_at: Optional[float] = None
    raw_kernel_status: Optional[str] = None
    output_nam_path: Optional[str] = None
    output_nam_sha256: Optional[str] = None
    training_result: Optional[dict] = None
    local_validation: Optional[dict] = None
    error: Optional[str] = None
    cleanup_state: Optional[str] = None
    cleanup_error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "KaggleJob":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _job_dir(a2_output_dir: Path, design_id: str, job_id: str) -> Path:
    return a2_output_dir / design_id / "kaggle" / job_id


def _atomic_write_json(path: Path, data: dict) -> None:
    """job.json is now genuinely written from a background worker thread
    while Flask GET routes read it concurrently (see submit_async) --
    os.replace can transiently fail on Windows with PermissionError if
    another handle has the destination open at that exact instant. Retry a
    few times with a tiny backoff rather than letting a real state update
    get lost to a race that resolves itself within milliseconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    last_exc: Optional[OSError] = None
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.02 * (attempt + 1))
    raise last_exc


def save_job(a2_output_dir: Path, job: KaggleJob) -> None:
    """Persists the job atomically. Defense-in-depth redaction on top of
    KaggleCli._run's own stdout/stderr scrubbing -- `error`/`cleanup_error`
    are free-text fields that could in principle carry secret-shaped text
    from any source, and job.json must never contain one (docs/kaggle_training.md)."""
    job.updated_at = time.time()
    data = job.to_dict()
    if data.get("error"):
        data["error"] = _redact(data["error"])
    if data.get("cleanup_error"):
        data["cleanup_error"] = _redact(data["cleanup_error"])
    _atomic_write_json(_job_dir(a2_output_dir, job.design_id, job.job_id) / "job.json", data)


def load_job(a2_output_dir: Path, design_id: str, job_id: str) -> Optional[KaggleJob]:
    path = _job_dir(a2_output_dir, design_id, job_id) / "job.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return KaggleJob.from_dict(json.load(f))


def find_active_job(a2_output_dir: Path, design_id: str) -> Optional[KaggleJob]:
    """The brief permits one active job per design -- used to reject a
    second concurrent `train` request rather than pretending to support
    concurrency we haven't actually isolated."""
    kaggle_dir = a2_output_dir / design_id / "kaggle"
    if not kaggle_dir.is_dir():
        return None
    candidates = []
    for job_json in kaggle_dir.glob("*/job.json"):
        with open(job_json, "r", encoding="utf-8") as f:
            job = KaggleJob.from_dict(json.load(f))
        candidates.append(job)
    active = [j for j in candidates if j.state not in TERMINAL_STATES]
    if active:
        return max(active, key=lambda j: j.updated_at)
    return max(candidates, key=lambda j: j.updated_at) if candidates else None


class KaggleTrainingError(RuntimeError):
    """Raised for a distinguishable, user-facing failure category -- callers
    (Flask routes) report `str(exc)` rather than collapsing everything into
    a generic 'Training failed'."""


class KaggleJobManager:
    def __init__(
        self,
        a2_output_dir: Path,
        cli: Optional[KaggleCli] = None,
        kernel_verify_attempts: int = KERNEL_VERIFY_ATTEMPTS,
        kernel_verify_delay_s: float = KERNEL_VERIFY_DELAY_S,
        dataset_upload_timeout_s: int = DATASET_UPLOAD_TIMEOUT_S,
        dataset_verify_timeout_s: float = DATASET_VERIFY_TIMEOUT_S,
        dataset_verify_interval_s: float = DATASET_VERIFY_INTERVAL_S,
    ):
        self.a2_output_dir = Path(a2_output_dir)
        self.cli = cli or KaggleCli()
        self.kernel_verify_attempts = kernel_verify_attempts
        self.kernel_verify_delay_s = kernel_verify_delay_s
        self.dataset_upload_timeout_s = dataset_upload_timeout_s
        self.dataset_verify_timeout_s = dataset_verify_timeout_s
        self.dataset_verify_interval_s = dataset_verify_interval_s
        self._status_json_supported: Optional[bool] = None

    # -- status -------------------------------------------------------

    def status(self) -> dict:
        installed = self.cli.is_installed()
        info: dict = {
            "cli_installed": installed,
            "cli_version": self.cli.version() if installed else None,
            "authenticated": self.cli.is_authenticated() if installed else False,
            "accelerator": ACCELERATOR,
        }
        if info["authenticated"]:
            info.update(self.cli.quota())
        else:
            info["quota_available"] = False
            info["quota_error"] = None if not installed else "not authenticated"
        return info

    # -- staging --------------------------------------------------------

    def stage(self, job: KaggleJob, bundle_dir: Path) -> tuple[Path, Path]:
        """Returns (dataset_staging, kernel_staging) -- split so the private
        DATASET (the actual training data: input.wav/hybrid_target.wav/
        training_manifest.json/cloud_job.json) never bundles in the cloud
        worker script, and the private KERNEL push never re-uploads the
        training data it already gets via `dataset_sources` -- see
        docs/kaggle_training.md."""
        job_dir = _job_dir(self.a2_output_dir, job.design_id, job.job_id)
        dataset_staging = job_dir / "dataset_staging"
        kernel_staging = job_dir / "kernel_staging"
        dataset_staging.mkdir(parents=True, exist_ok=True)
        kernel_staging.mkdir(parents=True, exist_ok=True)

        bundle_dir = Path(bundle_dir)
        for name in STAGED_BUNDLE_FILES:
            src = bundle_dir / name
            if not src.is_file():
                raise KaggleTrainingError(f"training bundle is missing required file: {src}")
            shutil.copyfile(src, dataset_staging / name)

        cloud_job = {
            "job_id": job.job_id,
            "design_id": job.design_id,
            "accelerator": job.accelerator,
            "epoch_preset": job.epoch_preset,
        }
        _atomic_write_json(dataset_staging / "cloud_job.json", cloud_job)

        cloud_script = Path(__file__).resolve().parent.parent / "cloud" / "kaggle" / "train_a2_cloud.py"
        if not cloud_script.is_file():
            raise KaggleTrainingError(f"cloud worker script missing: {cloud_script}")
        shutil.copyfile(cloud_script, kernel_staging / cloud_script.name)

        job.state = "uploading"
        save_job(self.a2_output_dir, job)
        return dataset_staging, kernel_staging

    def create_dataset(self, job: KaggleJob, staging_dir: Path) -> None:
        # Kaggle requires dataset-metadata.json's "id" to be
        # "<username>/<slug>" -- the installed CLI's own dataset_create_new
        # does `ref.split("/")[1]` unconditionally and raises an unhandled
        # IndexError on a bare slug. Also keep the bare slug within Kaggle's
        # 6-50 character title/slug bound regardless of how long design_id/
        # job_id happen to be.
        slug = f"hybrid-a2-{_safe_slug(job.design_id, 20)}-{_safe_slug(job.job_id, 12)}"
        username = self.cli.username()
        if not username:
            job.state = "failed"
            job.error = "could not determine the authenticated Kaggle username (required to create a private dataset)"
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)
        dataset_ref = f"{username}/{slug}"

        # Persist the INTENDED dataset reference before the actual upload
        # even starts -- this is a deterministic reference, not proof the
        # dataset exists yet, but it means a Flask restart or crash mid-
        # upload doesn't lose track of what to look for/clean up. Root-cause
        # investigation of a real production incident found this upload can
        # legitimately take several minutes under real network conditions
        # (Kaggle's own resumable-upload retry logic), so `state` reflects
        # that this may run for a while, not that anything has gone wrong.
        job.dataset_ref = dataset_ref
        job.state = "uploading_dataset"
        save_job(self.a2_output_dir, job)

        dataset_metadata = {
            "title": slug,
            "id": dataset_ref,
            "licenses": [{"name": "CC0-1.0"}],
        }
        _atomic_write_json(staging_dir / "dataset-metadata.json", dataset_metadata)

        log_path = self._tail_log_path(job)
        # Streamed (not capture_output=True) so the job log shows real
        # progress during a long upload instead of nothing until it exits --
        # see KaggleCli.run_streaming's docstring.
        result = self.cli.datasets_create_streaming(staging_dir, log_path, timeout=self.dataset_upload_timeout_s)
        if not result.ok:
            job.state = "failed"
            timed_out = "timed out" in (result.stderr or "")
            job.error = (
                f"dataset upload {'timed out' if timed_out else 'failed'}: "
                f"{result.stderr.strip() or result.stdout.strip()[-500:]}"
            )
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)

        job.upload_completed_at = time.time()
        # `datasets create` exiting 0 (and even `datasets status == ready`)
        # is NOT sufficient evidence every intended file actually arrived --
        # a real production dataset reported "ready" with only 1 of 6 files
        # present, and a separate real dataset (andrzejmarczewski/hybrid-a2-
        # 20260905t200558z-ab2994879d66) proved the FIRST post-create
        # `datasets files` call can transiently 403 even though the upload
        # genuinely succeeded. Verify the exact remote payload -- tolerating
        # a bounded settling window for that eventual consistency -- before
        # ever creating a kernel against it.
        job.state = "verifying_dataset"
        save_job(self.a2_output_dir, job)
        ok, error = self.verify_dataset_payload(dataset_ref, staging_dir, job=job)
        if not ok:
            job.state = "failed"
            job.error = error
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)

    def verify_dataset_payload(
        self,
        dataset_ref: str,
        staging_dir: Path,
        job: Optional[KaggleJob] = None,
        timeout: Optional[float] = None,
        interval: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Confirms every file in REQUIRED_DATASET_FILES is actually present
        remotely with a size matching the local staged copy, tolerating
        Kaggle's post-create eventual consistency (see DATASET_VERIFY_*
        constants and the create_dataset() comment above).

        Two very different failure classes, handled differently:
          - CANNOT INSPECT YET (403/404/not-found/still-processing/empty
            listing): retried within the bounded settling window.
          - DATASET IS READABLE BUT WRONG (missing file, wrong size): failed
            immediately, never retried -- a genuinely incomplete upload does
            not become complete by waiting.

        `job` is optional (existing direct-call tests omit it) -- when given,
        attempts are appended to its Kaggle log and dataset_ready_at/
        dataset_verified_at are stamped as they happen.
        """
        timeout = self.dataset_verify_timeout_s if timeout is None else timeout
        interval = self.dataset_verify_interval_s if interval is None else interval
        deadline = time.time() + timeout
        attempt = 0
        dataset_ready = False

        while True:
            attempt += 1

            if not dataset_ready:
                status_result = self.cli.datasets_status(dataset_ref, json_format=self._datasets_status_json_supported())
                status_value = _parse_dataset_status(status_result)
                if status_value == "ready":
                    dataset_ready = True
                    if job is not None and job.dataset_ready_at is None:
                        job.dataset_ready_at = time.time()
                        save_job(self.a2_output_dir, job)
                    self._log_verify_attempt(job, attempt, "dataset status ready")
                else:
                    reason = (
                        f"dataset still processing (status={status_value or 'unknown'})"
                        if status_result.ok
                        else f"dataset status check failed ({(status_result.stderr or status_result.stdout).strip() or 'no response'})"
                    )
                    if time.time() + interval > deadline:
                        return False, self._verify_timeout_message(dataset_ref, timeout)
                    self._log_verify_attempt(job, attempt, f"{reason}; retrying in {interval}s")
                    time.sleep(interval)
                    continue

            files_result = self.cli.datasets_files(dataset_ref)
            if files_result.ok:
                remote_sizes = _parse_datasets_files_csv(files_result.stdout)
                if remote_sizes:
                    self._log_verify_attempt(job, attempt, "remote file listing available")
                    ok, error = self._compare_required_files(dataset_ref, remote_sizes, staging_dir)
                    if ok:
                        self._log_verify_attempt(job, attempt, "verified " + ", ".join(
                            f"{name} {remote_sizes.get(name)}" for name in REQUIRED_DATASET_FILES
                        ))
                        if job is not None:
                            job.dataset_verified_at = time.time()
                            save_job(self.a2_output_dir, job)
                    return ok, error
                reason = "remote file listing not available yet (empty)"
            else:
                reason = f"file listing not available yet ({(files_result.stderr or files_result.stdout).strip() or 'unknown error'})"

            if time.time() + interval > deadline:
                return False, self._verify_timeout_message(dataset_ref, timeout)
            self._log_verify_attempt(job, attempt, f"{reason}; retrying in {interval}s")
            time.sleep(interval)

    @staticmethod
    def _compare_required_files(dataset_ref: str, remote_sizes: dict[str, int], staging_dir: Path) -> tuple[bool, str]:
        missing: list[str] = []
        mismatched: list[str] = []
        for name in REQUIRED_DATASET_FILES:
            local_path = staging_dir / name
            local_size = local_path.stat().st_size if local_path.is_file() else None
            remote_size = remote_sizes.get(name)
            if remote_size is None:
                missing.append(name)
            elif local_size is not None and remote_size != local_size:
                mismatched.append(f"{name} (local {local_size}B, remote {remote_size}B)")

        if not missing and not mismatched:
            return True, ""

        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if mismatched:
            parts.append("size mismatch: " + ", ".join(mismatched))
        return False, (
            "Kaggle dataset creation returned ready but training files are incomplete: "
            + "; ".join(parts) + f" (dataset={dataset_ref})"
        )

    @staticmethod
    def _verify_timeout_message(dataset_ref: str, timeout: float) -> str:
        return (
            f"Kaggle accepted the upload, but the private dataset ({dataset_ref}) could not be "
            f"read back for verification after {timeout:.0f}s. No GPU kernel was started."
        )

    def _datasets_status_json_supported(self) -> bool:
        if self._status_json_supported is None:
            self._status_json_supported = bool(
                hasattr(self.cli, "supports_datasets_status_json") and self.cli.supports_datasets_status_json()
            )
        return self._status_json_supported

    def _log_verify_attempt(self, job: Optional[KaggleJob], attempt: int, message: str) -> None:
        if job is None:
            return
        self._append_log(job, f"Dataset verification attempt {attempt}: {message}")

    def _append_log(self, job: KaggleJob, text: str) -> None:
        log_path = self._tail_log_path(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else text + "\n")

    def create_kernel(self, job: KaggleJob, staging_dir: Path) -> None:
        if job.dataset_ref is None:
            raise KaggleTrainingError("cannot create kernel before a dataset exists for this job")

        job.state = "creating_kernel"
        save_job(self.a2_output_dir, job)

        # Kaggle requires kernel-metadata.json's "id" to be
        # "<username>/<slug>" too -- a bare slug silently resolves to
        # something that never matches the kernel Kaggle actually creates
        # (observed in production: kernels.get denied / "Not found" against
        # the bare-slug ref this app had persisted). Exactly mirrors
        # create_dataset()'s fix above.
        kernel_slug = f"hybrid-a2-train-{_safe_slug(job.design_id, 20)}-{_safe_slug(job.job_id, 12)}"
        username = self.cli.username()
        if not username:
            job.state = "failed"
            job.error = "could not determine the authenticated Kaggle username (required to create a private kernel)"
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)
        kernel_ref = f"{username}/{kernel_slug}"

        kernel_metadata = {
            "id": kernel_ref,
            "title": kernel_slug,
            "code_file": "train_a2_cloud.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [job.dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        _atomic_write_json(staging_dir / "kernel-metadata.json", kernel_metadata)

        result = self.cli.kernels_push(staging_dir, accelerator=job.accelerator)
        if not result.ok:
            job.state = "failed"
            job.error = f"kernel push failed: {result.stderr.strip() or result.stdout.strip()}"
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)

        # `kernels push` exiting 0 is NOT sufficient evidence the kernel
        # actually exists -- verify it resolves (with bounded retry for
        # Kaggle-side eventual consistency) before ever calling this job
        # "queued". Never leave it looking submitted for an unresolved
        # kernel -- the dataset/staging are preserved either way for
        # diagnosis (we never delete them here).
        job.state = "verifying_kernel"
        save_job(self.a2_output_dir, job)
        if not self._verify_kernel_exists(kernel_ref):
            job.state = "failed"
            job.error = (
                f"kernel push reported success but Kaggle did not create/resolve the kernel "
                f"({kernel_ref}) after {self.kernel_verify_attempts} attempts -- the dataset "
                f"({job.dataset_ref}) and staging directory are preserved for diagnosis"
            )
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)

        job.kernel_ref = kernel_ref
        job.state = "queued"
        save_job(self.a2_output_dir, job)

    def _verify_kernel_exists(self, kernel_ref: str) -> bool:
        for attempt in range(self.kernel_verify_attempts):
            if self.cli.kernels_status(kernel_ref).ok:
                return True
            if attempt < self.kernel_verify_attempts - 1:
                time.sleep(self.kernel_verify_delay_s)
        return False

    def _precheck_and_reserve_job(self, design_id: str, epoch_preset: str = DEFAULT_EPOCH_PRESET) -> KaggleJob:
        """Shared by `submit`/`submit_async`: CLI/auth checks, resolving a
        stuck existing job, and reserving a fresh job_id -- all fast/cheap,
        safe to run synchronously inside the Flask request."""
        if epoch_preset not in A2_EPOCH_PRESETS:
            raise KaggleTrainingError(
                f"unknown epoch_preset {epoch_preset!r} -- choose one of {sorted(A2_EPOCH_PRESETS)}"
            )
        if not self.cli.is_installed():
            raise KaggleTrainingError("Kaggle CLI is not installed. Run: pip install kaggle")
        if not self.cli.is_authenticated():
            raise KaggleTrainingError("Kaggle CLI is not authenticated. Run: kaggle auth login")

        existing = find_active_job(self.a2_output_dir, design_id)
        if existing is not None and existing.state not in TERMINAL_STATES:
            # Give a stuck job (e.g. one left mid-upload by an interrupted
            # previous attempt, or "submitted" by the pre-fix bare-kernel-
            # slug bug) a chance to resolve to failed via the same migration
            # path refresh() uses, rather than letting a job that was never
            # real block this design forever.
            existing = self.refresh(existing)
            if existing.state not in TERMINAL_STATES:
                raise KaggleTrainingError(
                    f"a Kaggle job is already in progress for this design ({existing.job_id}, state={existing.state})"
                )

        job = KaggleJob(job_id=uuid.uuid4().hex[:12], design_id=design_id, epoch_preset=epoch_preset)
        save_job(self.a2_output_dir, job)
        return job

    def _run_pipeline(self, job: KaggleJob, bundle_dir: Path) -> None:
        """stage -> upload dataset -> verify -> create kernel -> verify.
        The actually-long-running part of submission; called synchronously
        by `submit()` (tests, and callers that genuinely want to block) or
        from a background thread by `submit_async()`."""
        dataset_staging, kernel_staging = self.stage(job, bundle_dir)
        self.create_dataset(job, dataset_staging)
        self.create_kernel(job, kernel_staging)

    def submit(self, design_id: str, bundle_dir: Path, epoch_preset: str = DEFAULT_EPOCH_PRESET) -> KaggleJob:
        """Synchronous end-to-end submission -- blocks for the entire
        upload. Kept for tests and any caller that genuinely wants to wait;
        the Flask route uses `submit_async` instead so a slow/flaky Kaggle
        upload (see docs/kaggle_training.md -- a real production run took
        several minutes under real network conditions) never blocks the
        request thread."""
        job = self._precheck_and_reserve_job(design_id, epoch_preset)
        self._run_pipeline(job, bundle_dir)
        return job

    def submit_async(self, design_id: str, bundle_dir: Path, epoch_preset: str = DEFAULT_EPOCH_PRESET) -> KaggleJob:
        """Returns immediately (job in state "preparing"/"uploading_dataset")
        once CLI/auth pre-checks pass; the actual stage/upload/verify/kernel
        pipeline runs on a background thread. A plain daemon thread is
        acceptable here -- this is a single-user local app (see
        docs/kaggle_training.md), not a multi-tenant server -- but every
        step still persists job state to disk immediately, so a Flask
        restart mid-upload loses only the ability to keep watching that one
        upload live, never the record of what happened."""
        job = self._precheck_and_reserve_job(design_id, epoch_preset)

        def _worker() -> None:
            try:
                self._run_pipeline(job, bundle_dir)
            except KaggleTrainingError:
                pass  # already persisted (state=="failed" + job.error) by the failing step
            except Exception as exc:  # noqa: BLE001 -- a background thread's exception has nowhere else to go
                job.state = "failed"
                job.error = f"unexpected error during Kaggle submission: {exc}"
                save_job(self.a2_output_dir, job)

        threading.Thread(target=_worker, daemon=True, name=f"kaggle-submit-{job.job_id}").start()
        return job

    # -- polling ----------------------------------------------------------

    def refresh(self, job: KaggleJob) -> KaggleJob:
        if job.state in TERMINAL_STATES or job.kernel_ref is None:
            return job

        if "/" not in job.kernel_ref:
            # Migration path for a job stuck by the pre-fix version, which
            # persisted a bare kernel slug that never actually resolved on
            # Kaggle. Try to safely qualify it with the authenticated
            # username and confirm it actually resolves; if it doesn't,
            # this job can never proceed (the kernel it thinks it has was
            # never real) -- fail it rather than blocking find_active_job
            # forever.
            username = self.cli.username()
            qualified = f"{username}/{job.kernel_ref}" if username else None
            if qualified and self.cli.kernels_status(qualified).ok:
                job.kernel_ref = qualified
            else:
                job.state = "failed"
                job.error = (
                    f"stuck job from a previous buggy version: kernel_ref {job.kernel_ref!r} "
                    "was never a valid Kaggle kernel reference and could not be resolved -- "
                    "submit a new training job for this design"
                )
                save_job(self.a2_output_dir, job)
                return job

        result = self.cli.kernels_status(job.kernel_ref)
        if not result.ok:
            # A transient status-check failure is not itself a training
            # failure -- keep the job in its current non-terminal state.
            return job

        raw = result.combined.strip()
        job.raw_kernel_status = raw
        mapped = "running"
        for key, value in _KERNEL_STATUS_MAP.items():
            if key.lower() in raw.lower():
                mapped = value
                break
        job.state = mapped
        save_job(self.a2_output_dir, job)

        if job.state == "downloading":
            self._download_and_validate(job)
        return job

    def _tail_log_path(self, job: KaggleJob) -> Path:
        return _job_dir(self.a2_output_dir, job.design_id, job.job_id) / "logs" / "kaggle.log"

    def fetch_logs(self, job: KaggleJob) -> str:
        if job.kernel_ref is None:
            return ""
        result = self.cli.kernels_logs(job.kernel_ref)
        text = result.combined
        log_path = self._tail_log_path(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text)
            if text and not text.endswith("\n"):
                f.write("\n")
        return self._read_log_tail(job)

    def read_log_tail(self, job: KaggleJob, n: int = LOG_TAIL_LINES) -> str:
        return self._read_log_tail(job, n)

    def _read_log_tail(self, job: KaggleJob, n: int = LOG_TAIL_LINES) -> str:
        log_path = self._tail_log_path(job)
        if not log_path.is_file():
            return ""
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n:])

    @staticmethod
    def parse_progress(log_text: str) -> Optional[dict]:
        """Best-effort epoch-progress extraction -- UX only, never allowed to
        raise or affect job correctness (docs/kaggle_training.md)."""
        try:
            matches = re.findall(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)", log_text or "")
            if not matches:
                return None
            current, total = matches[-1]
            return {"epoch": int(current), "total_epochs": int(total)}
        except (ValueError, TypeError):
            return None

    # -- download + local validation --------------------------------------

    def _download_and_validate(self, job: KaggleJob) -> None:
        job_dir = _job_dir(self.a2_output_dir, job.design_id, job.job_id)
        output_dir = job_dir / "output"
        # A real production job failed here with `Invalid regex pattern
        # '*.nam|*.json': nothing to repeat at position 0` -- Kaggle's
        # `--file-pattern` is a REGEX, not a shell glob, and that string was
        # never valid regex. The kernel output for this app is always tiny
        # compared to the training dataset, so there is no real cost to
        # downloading it in full and filtering locally afterwards -- that
        # removes the CLI-regex compatibility/failure point entirely rather
        # than just fixing the one pattern. Clear any partial output from a
        # previous failed/interrupted download attempt first so stale files
        # can never be mistaken for this attempt's result.
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.cli.kernels_output(job.kernel_ref, output_dir)
        downloaded_something = output_dir.is_dir() and any(output_dir.rglob("*"))
        if not result.ok:
            if downloaded_something:
                # A real recovery of a genuinely completed job hit exactly
                # this: `kernels output` wrote every file successfully (byte-
                # for-byte identical to a from-scratch download, confirmed
                # via SHA256) but still exited non-zero, because the
                # installed Kaggle CLI crashes internally with `'charmap'
                # codec can't encode characters...` while printing its own
                # progress output on a Windows console whose codepage can't
                # represent every character it prints -- a bug in the CLI's
                # own output handling, unrelated to whether the download
                # itself succeeded. The exit code is therefore not a
                # trustworthy signal here; what's actually on disk is. A
                # download that reports failure with literally nothing
                # written is still a real, immediate failure below.
                self._append_log(
                    job,
                    "kernels output exited non-zero but files were written to disk -- "
                    f"treating as a non-fatal CLI-side error and continuing: "
                    f"{(result.stderr.strip() or result.stdout.strip())[:300]}",
                )
            else:
                job.state = "failed"
                job.error = f"output download failed: {result.stderr.strip() or result.stdout.strip()}"
                save_job(self.a2_output_dir, job)
                return

        job.state = "validating"
        save_job(self.a2_output_dir, job)

        nam_candidates = sorted(output_dir.rglob("*.nam"))
        if not nam_candidates:
            job.state = "failed"
            job.error = "Kaggle kernel finished but produced no .nam file"
            save_job(self.a2_output_dir, job)
            return
        nam_path = nam_candidates[0]

        result_json_candidates = sorted(output_dir.rglob("training_result.json"))
        training_result = None
        if result_json_candidates:
            try:
                with open(result_json_candidates[0], "r", encoding="utf-8") as f:
                    training_result = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                job.error = f"malformed training_result.json: {exc}"
        job.training_result = training_result

        if training_result is not None and training_result.get("success") is False:
            job.state = "failed"
            job.error = training_result.get("error", "cloud training reported failure")
            save_job(self.a2_output_dir, job)
            return

        try:
            bundle_dir = self.a2_output_dir / job.design_id
            validation = validate_downloaded_model(nam_path, bundle_dir / "input.wav", bundle_dir / "hybrid_target.wav")
        except (NamRenderError, OSError, ValueError) as exc:
            job.state = "failed"
            job.error = f"local validation of downloaded model failed: {exc}"
            save_job(self.a2_output_dir, job)
            return

        job.local_validation = validation
        job.output_nam_path = str(nam_path)
        job.output_nam_sha256 = validation["sha256"]
        job.state = "complete"
        save_job(self.a2_output_dir, job)

    # -- recovery -----------------------------------------------------

    def retry_download(self, job: KaggleJob) -> KaggleJob:
        """Recovers a job whose Kaggle training genuinely COMPLETED but whose
        LOCAL output download/validation failed for a reason that has
        nothing to do with the training run itself (the `--file-pattern`
        regex bug being the real incident this exists for). Never
        re-uploads the dataset, never re-pushes the kernel, never re-runs
        training -- a completed remote kernel's output is downloaded and
        re-validated exactly as `refresh()` would have done the first time.

        Only usable on a job that is 'failed' and still has a kernel_ref;
        refuses if the remote kernel doesn't actually report a completed
        status, since retrying a download for a kernel that never finished
        (or errored) would just reproduce a different failure with a
        misleading "recovery" label."""
        if job.kernel_ref is None:
            raise KaggleTrainingError(
                "cannot recover this job -- it has no kernel_ref, so no training run to recover output from"
            )
        if job.state != "failed":
            raise KaggleTrainingError(f"can only recover a job in state 'failed' (job is {job.state!r})")

        status_result = self.cli.kernels_status(job.kernel_ref)
        raw = status_result.combined.strip()
        if not status_result.ok or "complete" not in raw.lower():
            raise KaggleTrainingError(
                f"cannot recover: Kaggle kernel {job.kernel_ref} is not reporting a completed status "
                f"(status: {raw or 'no response'}) -- this only recovers a job whose training genuinely finished"
            )

        job.raw_kernel_status = raw
        job.error = None
        job.state = "downloading"
        save_job(self.a2_output_dir, job)
        self._download_and_validate(job)
        return job

    # -- cleanup ------------------------------------------------------

    def cleanup(self, job: KaggleJob) -> KaggleJob:
        if job.state not in TERMINAL_STATES:
            raise KaggleTrainingError(f"cannot clean up a job that is not finished (state={job.state})")
        if job.state == "complete" and job.local_validation is None:
            raise KaggleTrainingError("refusing to clean up: job is 'complete' but has no recorded local validation")

        errors = []
        if job.dataset_ref:
            result = self.cli.datasets_delete(job.dataset_ref)
            if not result.ok:
                errors.append(f"dataset delete failed: {result.stderr.strip() or result.stdout.strip()}")
        if job.kernel_ref:
            result = self.cli.kernels_delete(job.kernel_ref)
            if not result.ok:
                errors.append(f"kernel delete failed: {result.stderr.strip() or result.stdout.strip()}")

        if errors:
            job.cleanup_state = "cleanup_pending"
            job.cleanup_error = "; ".join(errors)
        else:
            job.cleanup_state = "cleaned"
            job.cleanup_error = None
        save_job(self.a2_output_dir, job)
        return job


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_downloaded_model(nam_path: Path, training_input_path: Path, target_path: Path) -> dict:
    """Reuses the exact same NAMCore verification `scripts/train_a2.py`
    performs for the local trainer's own export -- Full/Lite render, finite,
    correct length, plus ESR comparison against the target -- so a Kaggle
    "success" only counts once it passes the identical bar."""
    import soundfile as sf

    nam_path = Path(nam_path)
    model = load_nam(nam_path)
    input_audio, sr = sf.read(training_input_path, dtype="float32", always_2d=False)
    target_audio, _ = sf.read(target_path, dtype="float32", always_2d=False)

    report: dict = {"sha256": _sha256_file(nam_path), "sample_rate": sr}

    for label, slim in (("full", 0.0), ("lite", 1.0)):
        try:
            rendered = render(model, input_audio, sr, slim=slim)
        except NamRenderError as exc:
            report[label] = {"rendered_ok": False, "error": str(exc)}
            continue
        if rendered.ndim != 1 or len(rendered) != len(input_audio) or not np.all(np.isfinite(rendered)):
            report[label] = {"rendered_ok": False, "error": "invalid render shape/length/finiteness"}
            continue
        report[label] = {"rendered_ok": True, "metrics": compute_esr_metrics(rendered, target_audio)}

    if not report["full"].get("rendered_ok"):
        raise NamRenderError(f"Full submodel failed to render: {report['full'].get('error')}")

    return report

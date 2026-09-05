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

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .nam_loader import load_nam
from .render import NamRenderError, render
from .validation import compute_esr_metrics

ACCELERATOR = "NvidiaTeslaT4"
FORBIDDEN_ACCELERATORS = {"NvidiaTeslaP100", "TPU"}

# Allow-list of files staged into the Kaggle dataset for a job -- nothing
# else is ever copied out of a design's bundle dir, in particular never
# assets/nam_models/*.nam or any preview DI (see docs/kaggle_training.md).
STAGED_BUNDLE_FILES = ("input.wav", "hybrid_target.wav", "training_manifest.json")

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

    def is_installed(self) -> bool:
        return self.executable is not None

    def _run(self, args: list[str], timeout: Optional[int] = None) -> CliResult:
        if not self.is_installed():
            # Fall back to `python -m kaggle` in case the console script
            # isn't on PATH but the package is importable in this interpreter.
            argv = [sys.executable, "-m", "kaggle", *args]
        else:
            argv = [self.executable, *args]
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
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

    def datasets_create(self, dataset_dir: Path) -> CliResult:
        return self._run(["datasets", "create", "-p", str(dataset_dir)], timeout=600)

    def datasets_status(self, dataset_ref: str) -> CliResult:
        return self._run(["datasets", "status", dataset_ref], timeout=30)

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
        args = ["kernels", "output", kernel_ref, "-p", str(out_dir)]
        if file_pattern:
            args += ["--file-pattern", file_pattern]
        return self._run(args, timeout=600)

    def kernels_delete(self, kernel_ref: str) -> CliResult:
        return self._run(["kernels", "delete", kernel_ref, "--yes"], timeout=60)


# --- Job state machine -----------------------------------------------------

JOB_STATES = (
    "preparing", "uploading", "waiting_for_dataset", "submitted", "queued",
    "running", "downloading", "validating", "complete", "failed",
    "cleanup_pending", "cleaned",
)
TERMINAL_STATES = ("complete", "failed")

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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


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
    def __init__(self, a2_output_dir: Path, cli: Optional[KaggleCli] = None):
        self.a2_output_dir = Path(a2_output_dir)
        self.cli = cli or KaggleCli()

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

    def stage(self, job: KaggleJob, bundle_dir: Path) -> Path:
        job_dir = _job_dir(self.a2_output_dir, job.design_id, job.job_id)
        staging = job_dir / "staging"
        staging.mkdir(parents=True, exist_ok=True)

        bundle_dir = Path(bundle_dir)
        for name in STAGED_BUNDLE_FILES:
            src = bundle_dir / name
            if not src.is_file():
                raise KaggleTrainingError(f"training bundle is missing required file: {src}")
            shutil.copyfile(src, staging / name)

        cloud_job = {
            "job_id": job.job_id,
            "design_id": job.design_id,
            "accelerator": job.accelerator,
        }
        _atomic_write_json(staging / "cloud_job.json", cloud_job)

        cloud_script = Path(__file__).resolve().parent.parent / "cloud" / "kaggle" / "train_a2_cloud.py"
        if not cloud_script.is_file():
            raise KaggleTrainingError(f"cloud worker script missing: {cloud_script}")
        shutil.copyfile(cloud_script, staging / cloud_script.name)

        job.state = "uploading"
        save_job(self.a2_output_dir, job)
        return staging

    def create_dataset(self, job: KaggleJob, staging_dir: Path) -> None:
        slug = f"hybrid-a2-{_safe_slug(job.design_id)}-{_safe_slug(job.job_id, 12)}"
        dataset_metadata = {
            "title": slug,
            "id": slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        _atomic_write_json(staging_dir / "dataset-metadata.json", dataset_metadata)

        result = self.cli.datasets_create(staging_dir)
        if not result.ok:
            job.state = "failed"
            job.error = f"dataset upload failed: {result.stderr.strip() or result.stdout.strip()}"
            save_job(self.a2_output_dir, job)
            raise KaggleTrainingError(job.error)

        job.dataset_ref = slug
        job.state = "waiting_for_dataset"
        save_job(self.a2_output_dir, job)

    def create_kernel(self, job: KaggleJob, staging_dir: Path) -> None:
        if job.dataset_ref is None:
            raise KaggleTrainingError("cannot create kernel before a dataset exists for this job")

        kernel_slug = f"hybrid-a2-train-{_safe_slug(job.design_id)}-{_safe_slug(job.job_id, 12)}"
        kernel_metadata = {
            "id": kernel_slug,
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

        job.kernel_ref = kernel_slug
        job.state = "submitted"
        save_job(self.a2_output_dir, job)

    def submit(self, design_id: str, bundle_dir: Path) -> KaggleJob:
        if not self.cli.is_installed():
            raise KaggleTrainingError("Kaggle CLI is not installed. Run: pip install kaggle")
        if not self.cli.is_authenticated():
            raise KaggleTrainingError("Kaggle CLI is not authenticated. Run: kaggle auth login")

        existing = find_active_job(self.a2_output_dir, design_id)
        if existing is not None and existing.state not in TERMINAL_STATES:
            raise KaggleTrainingError(
                f"a Kaggle job is already in progress for this design ({existing.job_id}, state={existing.state})"
            )

        job = KaggleJob(job_id=uuid.uuid4().hex[:12], design_id=design_id)
        save_job(self.a2_output_dir, job)
        try:
            staging = self.stage(job, bundle_dir)
            self.create_dataset(job, staging)
            self.create_kernel(job, staging)
        except KaggleTrainingError:
            raise
        return job

    # -- polling ----------------------------------------------------------

    def refresh(self, job: KaggleJob) -> KaggleJob:
        if job.state in TERMINAL_STATES or job.kernel_ref is None:
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
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.cli.kernels_output(job.kernel_ref, output_dir, file_pattern="*.nam|*.json")
        if not result.ok:
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

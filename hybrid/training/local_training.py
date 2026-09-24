"""Cross-platform, local-only runner for the dedicated A2 environment."""
from __future__ import annotations

import codecs
import os
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path


# neural-amp-modeler==0.13.0 (requirements-training.txt) itself requires
# Python 3.10+ -- verified against a real failure: shutil.which("python3")
# picked up macOS's ancient Xcode-bundled Python 3.9 (earlier on this
# machine's PATH than any real installed Python), and pip then failed with
# "No matching distribution found for neural-amp-modeler==0.13.0" instead of
# a clear "wrong Python version" message.
MIN_TRAINING_PYTHON = (3, 10)
# Every package scripts/train_a2.py imports at training time; `nam` is
# neural-amp-modeler itself, the trainer.
TRAINING_IMPORT_CHECK = "import torch, soundfile, numpy, scipy, nam"


def _candidate_training_pythons() -> list[list[str]]:
    """Every plausible system Python, most-preferred first.

    Versioned names (python3.12, etc.) are checked before the bare `python3`/
    `python` a distro's `update-alternatives`-style symlink might point at
    literally anything -- preferring 3.12/3.11/3.10 specifically because
    Torch wheels lag behind the newest CPython release (see
    requirements-training.txt's own comment about avoiding a too-new
    interpreter), while still accepting a newer one if that's all that
    exists.
    """
    candidates = []
    for name in ("python3.12", "python3.11", "python3.10", "python3.13", "python3.14", "python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append([found])
    if os.name == "nt":
        py_launcher = shutil.which("py")
        if py_launcher:
            candidates.append([py_launcher, "-3"])
    return candidates


def _training_python_version(argv: list[str]) -> "tuple[int, int] | None":
    """The (major, minor) an interpreter actually reports, or None if it
    can't be run at all -- never guessed from a filename/version-manager
    symlink, which can point anywhere."""
    try:
        result = subprocess.run(
            [*argv, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor = result.stdout.strip().split(".")
        return (int(major), int(minor))
    except ValueError:
        return None


# A failed venv adoption check is retried after this long even if nothing changed.
FAILED_IMPORT_CHECK_RETRY_S = 60.0


class LocalTrainingManager:
    def __init__(self, repo_root: Path, output_root: Path, *, venv_dir: Path | None = None):
        self.repo_root, self.output_root = Path(repo_root), Path(output_root)
        self.venv_dir = Path(venv_dir) if venv_dir is not None else self.repo_root / ".venv-a2"
        self.process: subprocess.Popen | None = None
        self.state = "not_configured"
        self.log = deque(maxlen=300)
        # The most recent '\r'-updated, not-yet-newline-terminated line
        # (a live progress bar) -- see _collect()'s docstring. Folded into
        # status()'s log_tail/progress parsing, never into self.log itself.
        self._live_line = ""
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.cancel_requested = False
        self.manifest_path: Path | None = None
        self._lock = threading.Lock()
        # (python path, mtime) of a venv whose adoption import check already
        # failed, so status polls don't re-run a slow torch import each time.
        # (venv identity, monotonic time) of the last failed adoption check.
        self._failed_import_check: tuple[tuple, float] | None = None

    @property
    def design_id(self) -> str | None:
        """The accepted training manifest owns the job, never a pending request."""
        return self.manifest_path.parent.name if self.manifest_path else None

    @property
    def python(self) -> Path:
        return self.venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    @property
    def _setup_complete_marker(self) -> Path:
        return self.venv_dir / ".setup_complete"

    def _venv_identity(self) -> tuple:
        """What changes when this venv's interpreter or installed packages change."""
        site_packages = [*self.venv_dir.glob("lib/python*/site-packages"), self.venv_dir / "Lib" / "site-packages"]
        return (str(self.python), self.python.stat().st_mtime,
                tuple(sorted((str(p), p.stat().st_mtime) for p in site_packages if p.is_dir())))

    @property
    def is_ready(self) -> bool:
        """True only once a setup run has actually finished successfully.

        `self.python.is_file()` alone is NOT enough: a venv directory (and
        its `bin/python`) exists the moment `python -m venv` runs, long
        before pip has installed anything into it. An interrupted or failed
        setup (killed process, network error, disk full, a still-missing
        package like the `soundfile` import failing) leaves exactly that
        half-built venv behind. Gating readiness on a marker written only
        after every install step -- including a real import of every package
        `scripts/train_a2.py` needs -- succeeded means Train can never be
        enabled against a broken environment.

        A venv built by the OTHER supported path -- manually running
        scripts/setup_a2_env.sh/.ps1 per the README, into this same
        `venv_dir` -- never writes that marker either, since that script
        knows nothing about this app. Rather than declare it not-ready and
        push a user into re-running a multi-GB install that already
        succeeded, this does one lightweight adoption check: if the
        packages actually import, stamp the marker so every later call is
        the cheap file-existence check again.
        """
        if self._setup_complete_marker.is_file():
            return self.python.is_file()
        if not self.python.is_file():
            return False
        if self.process and self.process.poll() is None:
            # A setup/training run is currently writing into this venv --
            # never race an import check against it mid-install.
            return False
        try:
            identity = self._venv_identity()
        except OSError:
            return False
        # Don't re-run a failed 15 s check on every status poll -- but retry
        # once the packages change (pip adds/removes entries in site-packages,
        # which leaves the interpreter itself untouched) or after a minute, so
        # one slow first import can't mark the venv broken for the session.
        if self._failed_import_check is not None:
            failed_identity, failed_at = self._failed_import_check
            if failed_identity == identity and time.monotonic() - failed_at < FAILED_IMPORT_CHECK_RETRY_S:
                return False
        try:
            subprocess.run(
                [str(self.python), "-c", TRAINING_IMPORT_CHECK],
                capture_output=True, timeout=15, check=True,
            )
        except (subprocess.SubprocessError, OSError):
            self._failed_import_check = (identity, time.monotonic())
            return False
        self._failed_import_check = None
        self._setup_complete_marker.write_text("ok")
        return True

    def _bootstrap_python(self) -> list[str]:
        """Return a real CPython command for making the training venv.

        Prefers the interpreter running this app (``sys.executable``) when it
        already satisfies neural-amp-modeler's own Python floor, since that
        is the environment the user actually ran `python app.py` from.
        Otherwise falls back to `NAM_MIXER_TRAINING_PYTHON` (an explicit
        override) or a search across other installed interpreters -- the app
        environment itself deliberately stays torch-free (see CLAUDE.md), so
        it may not meet the training venv's own requirement.
        """
        # In the packaged (PyInstaller) app sys.executable is the app itself,
        # not a Python interpreter: running it with -c would start a second app.
        if not getattr(sys, "frozen", False):
            own_version = _training_python_version([sys.executable])
            if own_version is not None and own_version >= MIN_TRAINING_PYTHON:
                return [sys.executable]

        configured = os.environ.get("NAM_MIXER_TRAINING_PYTHON", "").strip()
        if configured:
            executable = Path(configured).expanduser()
            if not executable.is_file():
                raise RuntimeError("NAM_MIXER_TRAINING_PYTHON does not point to a Python executable.")
            version = _training_python_version([str(executable)])
            if version is None or version < MIN_TRAINING_PYTHON:
                found = ".".join(map(str, version)) if version else "an unrecognized version"
                raise RuntimeError(
                    f"NAM_MIXER_TRAINING_PYTHON ({executable}) reports Python {found}, but "
                    f"neural-amp-modeler needs Python {'.'.join(map(str, MIN_TRAINING_PYTHON))}+."
                )
            return [str(executable)]

        checked: list[tuple[tuple[int, int], list[str]]] = []
        for argv in _candidate_training_pythons():
            version = _training_python_version(argv)
            if version is None:
                continue
            if version >= MIN_TRAINING_PYTHON:
                return argv
            checked.append((version, argv))

        min_str = ".".join(map(str, MIN_TRAINING_PYTHON))
        if checked:
            found_desc = ", ".join(f"{'.'.join(map(str, v))} ({' '.join(a)})" for v, a in checked)
            raise RuntimeError(
                f"Local training needs Python {min_str}+ (neural-amp-modeler's own requirement), "
                f"but only found: {found_desc}. Install a newer Python 3, restart the app, then "
                "click Set up local training again."
            )
        raise RuntimeError(
            f"Local training needs Python {min_str}+ installed outside the packaged app. "
            "Install Python 3, restart the app, then click Set up local training again."
        )

    def _start(self, command: list[str], state: str) -> None:
        self.log.clear()
        self._live_line = ""
        self.state = state
        self.started_at, self.finished_at, self.exit_code = time.time(), None, None
        self.cancel_requested = False
        # Without this, an op unimplemented on MPS raises a hard error instead
        # of falling back to CPU for that op -- surfaces as training dying
        # almost immediately with ~0 CPU time consumed and no output.
        # Without MPLBACKEND=Agg, nam.train.core's plt.show() calls open a
        # blocking native window on macOS -- training then sits at ~0 CPU
        # until a human closes it, which looks identical to a genuine hang
        # in this headless/background subprocess.
        # The child's own stdout is block-buffered (~8KB) because it's a
        # pipe, not a tty, so without PYTHONUNBUFFERED=1 every print()
        # (including the "Epoch X/Y" progress line _collect()/status()
        # parse) sits in the child's buffer and never reaches log_tail/
        # progress until it fills or the process exits -- looks like the
        # epoch count is frozen.
        #
        # bufsize=0 (raw, unbuffered binary mode -- deliberately NOT
        # text=True) is just as deliberate: PyTorch Lightning's progress
        # bar (tqdm) updates via '\r' on one line and text-mode's
        # universal-newlines translation combined with `.read(1)`
        # produced spurious empty reads that looked exactly like EOF while
        # the process was still very much alive and sleeping -- verified
        # directly against os.read() on the same fd, which behaved
        # correctly. _collect() below does its own '\r'/'\n' splitting on
        # raw decoded bytes instead of trusting a text-mode wrapper.
        env = {
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
        }
        process_group_options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )
        self.process = subprocess.Popen(
            command, cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, env=env, **process_group_options,
        )
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self) -> None:
        """Reads the subprocess's combined stdout/stderr as raw bytes off
        the pipe's file descriptor, splitting on '\\r'/'\\n' ourselves,
        rather than `for line in self.process.stdout` (which only ever
        yields on '\\n') or a text-mode `.read(1)` (which, verified
        directly, produced spurious empty reads indistinguishable from
        EOF while the process was still alive, apparently an interaction
        between universal-newlines translation and single-character
        decoding -- os.read() on the same fd does not have this problem).

        PyTorch Lightning's own progress bar (tqdm) updates via '\\r' on a
        single line, exactly like a terminal overwriting itself, and never
        emits '\\n' until an epoch actually finishes -- so naive line-based
        iteration left `log_tail` looking frozen for an entire epoch's
        duration (this is what "we never see anything" during the epoch
        phase was: the bytes were arriving fine, PYTHONUNBUFFERED=1 above
        already saw to that, but a '\\r'-only update was never a complete
        "line" to yield). A '\\r' update now REPLACES `self._live_line` in
        place, exactly like a real terminal would show it, without
        spamming the capped `self.log` deque with one entry per
        progress-bar tick (which happens several times a second and would
        otherwise push real output out of the last-300-lines window
        within seconds)."""
        assert self.process and self.process.stdout
        fd = self.process.stdout.fileno()
        # Incremental: a multi-byte character (e.g. tqdm's block glyphs) can
        # straddle two 4096-byte reads.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""
        # A '\r' is only a progress-bar overwrite if the next character is not
        # '\n'; '\r\n' (every line on Windows) must still complete the line.
        pending_cr = False
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            for char in decoder.decode(chunk):
                if pending_cr:
                    pending_cr = False
                    if char != "\n":
                        buffer = ""  # bare '\r': the next text overwrites the line
                if char == "\n":
                    with self._lock:
                        self.log.append(buffer)
                        self._live_line = ""
                    buffer = ""
                elif char == "\r":
                    with self._lock:
                        self._live_line = buffer
                    pending_cr = True
                else:
                    buffer += char
        buffer += decoder.decode(b"", final=True)
        if buffer:
            with self._lock:
                self.log.append(buffer)
                self._live_line = ""
        code = self.process.wait()
        with self._lock:
            self.exit_code, self.finished_at = code, time.time()
            self.state = "cancelled" if self.cancel_requested else ("complete" if code == 0 else "failed")

    def cancel(self) -> None:
        """Stop the app-owned setup or training subprocess, escalating if needed."""
        with self._lock:
            process = self.process
            if process is None or process.poll() is not None:
                raise RuntimeError("No local setup or training process is running.")
            self.cancel_requested = True
            self.state = "cancelling"
            self.log.append("Cancellation requested — stopping local process…")
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def setup(self) -> None:
        if self.process and self.process.poll() is None:
            raise RuntimeError("Local setup or training is already running.")
        # Setup completion is not training completion; never expose a report
        # left over from the preceding training run.
        self.manifest_path = None
        # Never let a stale marker from a previous successful setup claim
        # readiness while this run is in flight or if it fails partway.
        self._setup_complete_marker.unlink(missing_ok=True)
        # A tiny Python bootstrap avoids shell quoting and works on Windows/macOS.
        # `env` must be self.venv_dir itself (not a path derived from cwd) --
        # that is the exact directory `self.python`/`ready` check afterwards.
        bootstrap = (
            "import subprocess,sys,pathlib; "
            f"root=pathlib.Path.cwd(); env=pathlib.Path({str(self.venv_dir)!r}); "
            "env.parent.mkdir(parents=True, exist_ok=True); "
            "subprocess.check_call([sys.executable,'-m','venv',str(env)]); "
            "py=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python'); "
            "subprocess.check_call([str(py),'-m','pip','install','--upgrade','pip']); "
            "subprocess.check_call([str(py),'-m','pip','install','-r',str(root/'requirements-training.txt')]); "
            # The standard macOS wheel includes MPS support.  On Windows we
            # intentionally do not install a generic wheel here: that could
            # replace a user's CUDA-specific Torch installation.
            "(subprocess.check_call([str(py),'-m','pip','install','--upgrade','torch','torchvision']) if sys.platform=='darwin' else None); "
            # Actually import every package scripts/train_a2.py needs at
            # training time (not just torch) -- soundfile in particular is
            # a hybrid/core/cab_ir.py dependency that pip can silently skip if an
            # earlier install step was interrupted, which used to leave
            # `ready` true and Train enabled against a broken environment.
            f"import_check={TRAINING_IMPORT_CHECK + '; '!r} + \"print('MPS available: ' + str(torch.backends.mps.is_available())); print('MPS built: ' + str(torch.backends.mps.is_built()))\"; "
            "subprocess.check_call([str(py),'-c',import_check]); "
            f"pathlib.Path({str(self._setup_complete_marker)!r}).write_text('ok')"
        )
        self._start([*self._bootstrap_python(), "-c", bootstrap], "setting_up")

    def train(self, manifest: Path, preset: str) -> None:
        manifest = Path(manifest).resolve()
        if not manifest.is_file() or self.output_root.resolve() not in manifest.parents:
            raise RuntimeError("Training manifest must be a generated bundle inside work/a2.")
        if self.process and self.process.poll() is None:
            raise RuntimeError("Local setup or training is already running.")
        if not self.is_ready:
            raise RuntimeError("Local A2 environment is not ready. Click Set up local training first.")
        previous_manifest = self.manifest_path
        self.manifest_path = manifest
        try:
            self._start([str(self.python), "scripts/train_a2.py", str(manifest), "--epoch-preset", preset, "--progress"], "training")
        except Exception:
            self.manifest_path = previous_manifest
            raise

    def status(self) -> dict:
        running = bool(self.process and self.process.poll() is None)
        state = self.state if running or self.state in ("complete", "failed", "cancelled") else ("ready" if self.is_ready else "not_configured")
        with self._lock:
            # The live progress-bar line is appended last, exactly as it
            # would appear on a real terminal -- both for display and so
            # the epoch/total_epochs regex below can actually see it
            # (Lightning's own progress bar text is what usually carries
            # "Epoch X/Y" in the first place).
            tail = "\n".join(self.log) + (f"\n{self._live_line}" if self._live_line else "")
        # Lightning's rich progress output is commonly either ``Epoch 3/60``
        # or ``Epoch 3: 100%|...``.  The latter has no total in the line, so
        # pair it with the explicit ``--epoch-preset=...: N epochs`` message.
        matches = re.findall(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)", tail)
        if matches:
            progress = {"epoch": int(matches[-1][0]), "total_epochs": int(matches[-1][1])}
        else:
            current = re.findall(r"[Ee]poch\s+(\d+)\s*[:|]", tail)
            totals = re.findall(r"(?:epoch(?:s)?|for)\D{0,20}(\d+)\s+epochs?", tail, flags=re.IGNORECASE)
            progress = ({"epoch": int(current[-1]), "total_epochs": int(totals[-1])} if current and totals else None)
        meaningful_lines = [line.strip() for line in tail.splitlines() if line.strip()]
        latest_line = meaningful_lines[-1] if meaningful_lines else ""
        now = time.time()
        elapsed_s = None if self.started_at is None else int((self.finished_at or now) - self.started_at)
        validation_report = None
        embedded_artifact = None
        if state == "complete" and self.manifest_path and self.manifest_path.is_file():
            try:
                training = json.loads(self.manifest_path.read_text(encoding="utf-8")).get("training", {})
                validation_report = training.get("validation_report")
                # Surfaced so the UI can offer the experimental Sequential
                # (embedded-cab) download alongside the head model instead
                # of only ever exposing the head -- see app.py's
                # api_local_training_download `artifact` query param, which
                # the frontend previously had no data to ever request.
                embedded_artifact = training.get("embedded_artifact")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return {
            "state": state, "ready": self.is_ready, "python": str(self.python),
            "log_tail": tail, "started_at": self.started_at, "finished_at": self.finished_at,
            "elapsed_s": elapsed_s, "exit_code": self.exit_code,
            "progress": progress,
            "latest_line": latest_line[-500:],
            "validation_report": validation_report,
            "embedded_artifact": embedded_artifact,
        }

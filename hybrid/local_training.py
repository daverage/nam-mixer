"""Cross-platform, local-only runner for the dedicated A2 environment."""
from __future__ import annotations

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


class LocalTrainingManager:
    def __init__(self, repo_root: Path, output_root: Path, *, venv_dir: Path | None = None):
        self.repo_root, self.output_root = Path(repo_root), Path(output_root)
        self.venv_dir = Path(venv_dir) if venv_dir is not None else self.repo_root / ".venv-a2"
        self.process: subprocess.Popen | None = None
        self.state = "not_configured"
        self.log = deque(maxlen=300)
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.cancel_requested = False
        self.manifest_path: Path | None = None
        self._lock = threading.Lock()

    @property
    def design_id(self) -> str | None:
        """The accepted training manifest owns the job, never a pending request."""
        return self.manifest_path.parent.name if self.manifest_path else None

    @property
    def python(self) -> Path:
        return self.venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

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
        # bufsize=1/text=True below only govern how WE read the pipe -- the
        # child's own stdout is block-buffered (~8KB) because it's a pipe,
        # not a tty, so without PYTHONUNBUFFERED=1 every print() (including
        # the "Epoch X/Y" progress line _collect()/status() parse) sits in
        # the child's buffer and never reaches log_tail/progress until it
        # fills or the process exits -- looks like the epoch count is frozen.
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
            text=True, bufsize=1, env=env, **process_group_options,
        )
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            with self._lock:
                self.log.append(line.rstrip())
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
        # A tiny Python bootstrap avoids shell quoting and works on Windows/macOS.
        bootstrap = (
            "import subprocess,sys,pathlib; "
            "root=pathlib.Path.cwd(); env=root/'.venv-a2'; "
            "subprocess.check_call([sys.executable,'-m','venv',str(env)]); "
            "py=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python'); "
            "subprocess.check_call([str(py),'-m','pip','install','--upgrade','pip']); "
            "subprocess.check_call([str(py),'-m','pip','install','-r',str(root/'requirements-training.txt')]); "
            # The standard macOS wheel includes MPS support.  On Windows we
            # intentionally do not install a generic wheel here: that could
            # replace a user's CUDA-specific Torch installation.
            "(subprocess.check_call([str(py),'-m','pip','install','--upgrade','torch','torchvision']) if sys.platform=='darwin' else None); "
            "import_torch=\"import torch; print('MPS available: ' + str(torch.backends.mps.is_available())); print('MPS built: ' + str(torch.backends.mps.is_built()))\"; "
            "subprocess.check_call([str(py),'-c',import_torch])"
        )
        self._start([*self._bootstrap_python(), "-c", bootstrap], "setting_up")

    def train(self, manifest: Path, preset: str) -> None:
        manifest = Path(manifest).resolve()
        if not manifest.is_file() or self.output_root.resolve() not in manifest.parents:
            raise RuntimeError("Training manifest must be a generated bundle inside work/a2.")
        if self.process and self.process.poll() is None:
            raise RuntimeError("Local setup or training is already running.")
        if not self.python.is_file():
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
        state = self.state if running or self.state in ("complete", "failed", "cancelled") else ("ready" if self.python.is_file() else "not_configured")
        with self._lock:
            tail = "\n".join(self.log)
        matches = re.findall(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)", tail)
        progress = None if not matches else {"epoch": int(matches[-1][0]), "total_epochs": int(matches[-1][1])}
        now = time.time()
        elapsed_s = None if self.started_at is None else int((self.finished_at or now) - self.started_at)
        validation_report = None
        if state == "complete" and self.manifest_path and self.manifest_path.is_file():
            try:
                validation_report = json.loads(self.manifest_path.read_text(encoding="utf-8")).get("training", {}).get("validation_report")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return {
            "state": state, "ready": self.python.is_file(), "python": str(self.python),
            "log_tail": tail, "started_at": self.started_at, "finished_at": self.finished_at,
            "elapsed_s": elapsed_s, "exit_code": self.exit_code,
            "progress": progress,
            "validation_report": validation_report,
        }

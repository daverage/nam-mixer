"""Cross-platform, local-only runner for the dedicated A2 environment."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path


class LocalTrainingManager:
    def __init__(self, repo_root: Path, output_root: Path):
        self.repo_root, self.output_root = Path(repo_root), Path(output_root)
        self.venv_dir = self.repo_root / ".venv-a2"
        self.process: subprocess.Popen | None = None
        self.state = "not_configured"
        self.log = deque(maxlen=300)
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self._lock = threading.Lock()

    @property
    def python(self) -> Path:
        return self.venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _start(self, command: list[str], state: str) -> None:
        self.log.clear()
        self.state = state
        self.started_at, self.finished_at, self.exit_code = time.time(), None, None
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
        self.process = subprocess.Popen(
            command, cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
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
            self.state = "complete" if code == 0 else "failed"

    def setup(self) -> None:
        if self.process and self.process.poll() is None:
            raise RuntimeError("Local setup or training is already running.")
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
        self._start([sys.executable, "-c", bootstrap], "setting_up")

    def train(self, manifest: Path, preset: str) -> None:
        manifest = Path(manifest).resolve()
        if not manifest.is_file() or self.output_root.resolve() not in manifest.parents:
            raise RuntimeError("Training manifest must be a generated bundle inside work/a2.")
        if self.process and self.process.poll() is None:
            raise RuntimeError("Local setup or training is already running.")
        if not self.python.is_file():
            raise RuntimeError("Local A2 environment is not ready. Click Set up local training first.")
        self._start([str(self.python), "scripts/train_a2.py", str(manifest), "--epoch-preset", preset, "--progress"], "training")

    def status(self) -> dict:
        running = bool(self.process and self.process.poll() is None)
        state = self.state if running or self.state in ("complete", "failed") else ("ready" if self.python.is_file() else "not_configured")
        with self._lock:
            tail = "\n".join(self.log)
        matches = re.findall(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)", tail)
        progress = None if not matches else {"epoch": int(matches[-1][0]), "total_epochs": int(matches[-1][1])}
        now = time.time()
        elapsed_s = None if self.started_at is None else int((self.finished_at or now) - self.started_at)
        return {
            "state": state, "ready": self.python.is_file(), "python": str(self.python),
            "log_tail": tail, "started_at": self.started_at, "finished_at": self.finished_at,
            "elapsed_s": elapsed_s, "exit_code": self.exit_code,
            "progress": progress,
        }

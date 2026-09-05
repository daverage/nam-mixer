"""Rendering audio through a loaded NAM model.

Implemented via the `nam_render` native tool built from sdatkinson's
NeuralAmpModelerCore (C++ inference library) rather than the Python
`neural-amp-modeler`/torch package: NAMCore is inference-only, has no torch
dependency, and its own `.nam` loading (`nam::get_dsp`) is the reference
implementation the model authors maintain, so there is no risk of guessing
wrong at the on-disk schema (see git history for why that was deliberately
not attempted directly in Python).

To build the native tool: see native/nam_render/README.md. In short --
CMake + FetchContent pulls NeuralAmpModelerCore v0.5.4 and its dependencies
(Eigen, AudioDSPTools) and compiles the `render` CLI it defines
(`native/nam_render/build/Release/nam_render.exe` on Windows).

This module shells out to that executable: writes `audio` to a temp WAV,
invokes `nam_render <model.nam> <in.wav> <out.wav>`, reads the result back.
It requires no torch or neural-amp-modeler installation.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .nam_loader import NamModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NAM_RENDER_EXE_CANDIDATES = (
    _REPO_ROOT / "native" / "nam_render" / "build" / "Release" / "nam_render.exe",
    _REPO_ROOT / "native" / "nam_render" / "build" / "nam_render.exe",
    _REPO_ROOT / "native" / "nam_render" / "build" / "Release" / "nam_render",
    _REPO_ROOT / "native" / "nam_render" / "build" / "nam_render",
)


class NamRenderError(RuntimeError):
    """Raised when the native nam_render tool can't be found or fails to run."""


class RenderNotImplementedError(NamRenderError):
    """Deprecated alias kept for backward compatibility with older callers.

    render() no longer raises this unconditionally now that NAM inference is
    wired in -- see the module docstring. Prefer catching NamRenderError.
    """


def find_nam_render_exe() -> Path:
    """Locate the built nam_render executable, or raise NamRenderError."""
    for candidate in _NAM_RENDER_EXE_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("nam_render")
    if found:
        return Path(found)
    searched = ", ".join(str(c) for c in _NAM_RENDER_EXE_CANDIDATES)
    raise NamRenderError(
        "nam_render executable not found. Build it first -- see "
        f"native/nam_render/README.md. Looked in: {searched}, and on PATH."
    )


def render(model: NamModel, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Render `audio` (mono float32, at `sample_rate`) through `model`.

    - Input and output are both mono float32 numpy arrays of the same length.
    - Resampling `audio` to `model.sample_rate` (if known and different) is
      the caller's responsibility, not this function's. If `sample_rate`
      doesn't match what the model expects, nam_render's own check will
      raise NamRenderError with the mismatch reported.
    - Raises NamRenderError if the native tool is missing or exits non-zero.
    """
    exe = find_nam_render_exe()
    audio = np.asarray(audio, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="hybrid_nam_render_") as tmp:
        tmp_dir = Path(tmp)
        in_path = tmp_dir / "input.wav"
        out_path = tmp_dir / "output.wav"
        sf.write(in_path, audio, sample_rate, subtype="FLOAT")

        result = subprocess.run(
            [str(exe), str(model.path), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not out_path.is_file():
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise NamRenderError(f"nam_render failed for {model.path}: {message}")

        rendered, _ = sf.read(out_path, dtype="float32")

    return rendered

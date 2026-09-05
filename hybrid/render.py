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


_SUBPROCESS_TIMEOUT_S = 120.0


def render(model: NamModel, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Render `audio` (mono float32, at `sample_rate`) through `model`.

    - Input and output are both mono float32 numpy arrays of the same length.
    - Resampling `audio` to `model.sample_rate` (if known and different) is
      the caller's responsibility, not this function's. If `sample_rate`
      doesn't match what the model expects, nam_render's own check will
      raise NamRenderError with the mismatch reported.
    - Raises NamRenderError if the native tool is missing, times out, exits
      non-zero, or its output doesn't match this function's contract (mono,
      same length, same sample rate, all-finite) -- this module is the
      boundary between native code and the rest of the app, so it verifies
      that contract rather than trusting the subprocess blindly.
    """
    exe = find_nam_render_exe()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise NamRenderError(f"NAM render currently requires mono audio, got shape {audio.shape}")

    with tempfile.TemporaryDirectory(prefix="hybrid_nam_render_") as tmp:
        tmp_dir = Path(tmp)
        in_path = tmp_dir / "input.wav"
        out_path = tmp_dir / "output.wav"
        sf.write(in_path, audio, sample_rate, subtype="FLOAT")

        try:
            result = subprocess.run(
                [str(exe), str(model.path), str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise NamRenderError(
                f"nam_render timed out after {_SUBPROCESS_TIMEOUT_S}s for {model.path}"
            ) from exc

        if result.returncode != 0 or not out_path.is_file():
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise NamRenderError(f"nam_render failed for {model.path}: {message}")

        rendered, out_sample_rate = sf.read(out_path, dtype="float32")

    if rendered.ndim != 1:
        raise NamRenderError(
            f"nam_render produced non-mono output for {model.path}: shape {rendered.shape}"
        )
    if len(rendered) != len(audio):
        raise NamRenderError(
            f"nam_render output length {len(rendered)} != input length {len(audio)} for {model.path}"
        )
    if out_sample_rate != sample_rate:
        raise NamRenderError(
            f"nam_render output sample rate {out_sample_rate} != input sample rate {sample_rate} for {model.path}"
        )
    if not np.all(np.isfinite(rendered)):
        raise NamRenderError(f"nam_render produced non-finite samples for {model.path}")

    return rendered

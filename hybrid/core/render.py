"""Rendering audio through a loaded NAM model.

Implemented via the `nam_render` native tool built from sdatkinson's
NeuralAmpModelerCore (C++ inference library) rather than the Python
`neural-amp-modeler`/torch package: NAMCore is inference-only, has no torch
dependency, and its own `.nam` loading (`nam::get_dsp`) is the reference
implementation the model authors maintain, so there is no risk of guessing
wrong at the on-disk schema (see git history for why that was deliberately
not attempted directly in Python).

To build the native tool: see native/nam_render/README.md. In short --
CMake + FetchContent pulls the pinned Sequential-capable NeuralAmpModelerCore
commit and its dependencies
(Eigen, AudioDSPTools) and compiles the `render` CLI it defines
(`native/nam_render/build/Release/nam_render.exe` on Windows).

This module shells out to that executable: writes `audio` to a temp WAV,
invokes `nam_render <model.nam> <in.wav> <out.wav>`, reads the result back.
It requires no torch or neural-amp-modeler installation.
"""
from __future__ import annotations

import shutil
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .nam_loader import NamModel
from ..paths import REPO_ROOT as _REPO_ROOT

_NAM_RENDER_EXE_CANDIDATES = (
    _REPO_ROOT / "native" / "nam_render" / "build" / "Release" / "nam_render.exe",
    _REPO_ROOT / "native" / "nam_render" / "build" / "nam_render.exe",
    _REPO_ROOT / "native" / "nam_render" / "build" / "Release" / "nam_render",
    _REPO_ROOT / "native" / "nam_render" / "build" / "nam_render",
)


# NAMCore slimmable-size values for a packed A2 (SlimmableContainer) export.
# ContainerModel::_get_index_for_slimmable_size picks the FIRST submodel whose
# max_value is greater than the requested size; A2 exports list the small
# (Lite, 3-channel, max_value 0.5) submodel first and the large (Full,
# 8-channel, max_value 1.0) one last. So 0.0 selects Lite and 1.0 falls
# through to Full -- the same model NAMCore uses when no size is given.
# Verified by rendering each extracted submodel (tests/test_render.py).
SLIM_FULL = 1.0
SLIM_LITE = 0.0


class NamRenderError(RuntimeError):
    """Raised when the native nam_render tool can't be found or fails to run."""


def find_nam_render_exe() -> Path:
    """Locate the built nam_render executable, or raise NamRenderError."""
    configured = os.environ.get("NAM_RENDER_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise NamRenderError(f"NAM_RENDER_EXE is set but is not an executable file: {candidate}")
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


def find_sequential_nam_render_exe() -> Path:
    """Return the renderer used for Sequential validation.

    The bundled renderer is pinned to the Sequential-capable NAMCore commit.
    ``NAM_RENDER_SEQUENTIAL_EXE`` remains an optional explicit override for
    testing another renderer.
    """
    configured = os.environ.get("NAM_RENDER_SEQUENTIAL_EXE")
    if not configured:
        return find_nam_render_exe()
    # resolve() makes relative configuration deterministic against the
    # application's current working directory and works for POSIX binaries as
    # well as a Windows .exe name; no suffix is assumed.
    candidate = Path(configured).expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise NamRenderError(f"NAM_RENDER_SEQUENTIAL_EXE is not an executable file: {candidate}")
    return candidate


def sequential_renderer_record() -> dict:
    """Auditable identity for embedded-validation metadata."""
    executable = find_sequential_nam_render_exe()
    return {"path": str(executable), "configured_path": os.environ.get("NAM_RENDER_SEQUENTIAL_EXE"),
            "required_namcore_commit": "2563c0fd4cb1f9ce457d89a761738ea15097e1f3"}


_SUBPROCESS_TIMEOUT_S = 120.0


def render(model: NamModel, audio: np.ndarray, sample_rate: int, slim: float | None = None,
           executable: Path | None = None) -> np.ndarray:
    """Render `audio` (mono float32, at `sample_rate`) through `model`.

    - Input and output are both mono float32 numpy arrays of the same length.
    - Resampling `audio` to `model.sample_rate` (if known and different) is
      the caller's responsibility, not this function's. If `sample_rate`
      doesn't match what the model expects, nam_render's own check will
      raise NamRenderError with the mismatch reported.
    - `slim`: optional NAMCore "slimmable size" in [0.0, 1.0] -- only
      meaningful for models built as a `SlimmableContainer`/slimmable WaveNet
      (e.g. an A2 packed model's Full/Lite submodels), forwarded to the
      native tool's `--slim` flag verbatim. Use SLIM_FULL (1.0) / SLIM_LITE
      (0.0) rather than literals -- see their definition for NAMCore's
      selection rule (native/nam_render/build/_deps/namcore-src/NAM/container.cpp).
      Left as `None` (the default, no flag passed) for ordinary
      non-slimmable models -- this is purely additive, existing callers are
      unaffected. docs/history/phase3.md section 20.
    - Raises NamRenderError if the native tool is missing, times out, exits
      non-zero, or its output doesn't match this function's contract (mono,
      same length, same sample rate, all-finite) -- this module is the
      boundary between native code and the rest of the app, so it verifies
      that contract rather than trusting the subprocess blindly.
    """
    exe = executable or find_nam_render_exe()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise NamRenderError(f"NAM render currently requires mono audio, got shape {audio.shape}")

    with tempfile.TemporaryDirectory(prefix="hybrid_nam_render_") as tmp:
        tmp_dir = Path(tmp)
        in_path = tmp_dir / "input.wav"
        out_path = tmp_dir / "output.wav"
        sf.write(in_path, audio, sample_rate, subtype="FLOAT")

        cmd = [str(exe)]
        if slim is not None:
            cmd += ["--slim", str(slim)]
        cmd += [str(model.path), str(in_path), str(out_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise NamRenderError(
                f"nam_render timed out after {_SUBPROCESS_TIMEOUT_S}s for {model.path}"
            ) from exc
        except OSError as exc:  # e.g. no exec bit, or a binary for another CPU
            raise NamRenderError(f"could not run nam_render at {exe}: {exc}") from exc

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

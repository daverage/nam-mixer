"""Shared cabinet-IR convolution stage, used identically for live preview and
for a baked A2 training target -- see docs/blend-mode.md "CAB IR PROCESSING".

The cabinet sits AFTER the amp combination (Hybrid crossfade or Fixed Blend
mix), never before it and never as two separate per-branch cabs -- see
`hybrid.fixed_blend`/`hybrid.training_target`/`hybrid.blend_training_target`
for where `apply_cab_ir` is actually called in each signal chain.

Design choices, all deliberate (see the doc for the "do not" list this
enforces):

- Ordinary causal FIR convolution only. No peak-alignment, no minimum-phase
  transform, no EQ/tone-matching, no cab gain control. If the IR were treated
  more cleverly than this, preview and baked-target processing could subtly
  diverge, and the whole point of this module is that they never do.
- The IR is downmixed to mono deterministically (plain channel average -- not
  "first channel", which would silently pick an arbitrary side of a stereo
  capture) and resampled to the driving audio's sample rate with
  `scipy.signal.resample_poly` (already a project dependency, see CLAUDE.md).
- A causal full convolution is computed and then the first N samples are kept
  (N = length of the source audio) -- NOT a circular convolution -- so the
  IR's own convolution tail is truncated for a training target exactly the
  way a real-time convolver would only ever see it, streaming-equivalent
  behavior per the doc.
- Leading near-silence in the IR (e.g. a captured impulse with some silence
  before the microphone actually picks up the speaker) is trimmed
  deterministically before convolution, using a fixed -40 dBFS-relative
  threshold (documented below) -- NOT peak alignment to some other point in
  the IR, which would already start reshaping what the cab does. The number
  of samples trimmed is recorded on `PreparedCabIr` for manifest provenance.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

# A sample below this level, relative to the IR's own peak, is considered
# "before the cab actually starts responding" and trimmed from the front
# before convolution -- consistent in spirit with the -50 dBFS ABSOLUTE
# active-signal convention used elsewhere (hybrid/coverage.py), but RELATIVE
# here because an IR's absolute level is arbitrary (unlike a calibrated DI).
LEADING_SILENCE_THRESHOLD_RELATIVE_DB = -40.0

_MAX_CACHE_ENTRIES = 8


class CabIrError(ValueError):
    """Raised when a cabinet IR file is missing, invalid, silent, or otherwise
    cannot be safely used -- generation/preview aborts rather than guessing."""


@dataclass(frozen=True)
class PreparedCabIr:
    """An IR that has been loaded, downmixed, trimmed, and resampled to a
    specific target sample rate -- ready to convolve via `apply_cab_ir`.
    Immutable and cacheable by (sha256, target_sample_rate)."""

    samples: np.ndarray  # mono float32, causal FIR taps
    sample_rate: int  # == target_sample_rate this was prepared for
    source_path: str
    sha256: str
    original_sample_rate: int
    original_channels: int
    original_frame_count: int
    prepared_frame_count: int
    leading_samples_trimmed: int


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _trim_leading_silence(ir: np.ndarray, threshold_relative_db: float) -> tuple[np.ndarray, int]:
    peak = float(np.max(np.abs(ir))) if len(ir) else 0.0
    if peak <= 0:
        return ir, 0
    threshold = peak * (10.0 ** (threshold_relative_db / 20.0))
    above = np.nonzero(np.abs(ir) >= threshold)[0]
    if len(above) == 0:
        return ir, 0
    first = int(above[0])
    return ir[first:], first


def load_and_prepare_cab_ir(
    path: str | Path,
    target_sample_rate: int,
    leading_silence_threshold_db: float = LEADING_SILENCE_THRESHOLD_RELATIVE_DB,
) -> PreparedCabIr:
    """Load a cabinet IR WAV and prepare it for convolution against audio at
    `target_sample_rate`. Raises `CabIrError` for anything that can't be
    safely used -- never silently coerces a bad file into "probably fine"."""
    path = Path(path)
    if not path.is_file():
        raise CabIrError(f"cabinet IR file not found: {path}")

    try:
        raw, original_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 -- any soundfile failure means "not a usable IR"
        raise CabIrError(f"could not read cabinet IR WAV {path}: {exc}") from exc

    original_frame_count, original_channels = raw.shape
    if original_frame_count == 0:
        raise CabIrError(f"cabinet IR file is empty: {path}")

    # Deterministic downmix: plain channel average, not "pick channel 0" --
    # see module docstring.
    mono = raw.mean(axis=1).astype(np.float32) if original_channels > 1 else raw[:, 0].astype(np.float32)

    if not np.all(np.isfinite(mono)):
        raise CabIrError(f"cabinet IR contains NaN/Inf samples: {path}")

    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak <= 1e-9:
        raise CabIrError(f"cabinet IR is silent/near-silent: {path}")

    trimmed, leading_trimmed = _trim_leading_silence(mono, leading_silence_threshold_db)

    if int(original_sample_rate) != int(target_sample_rate):
        # resample_poly needs an integer up/down ratio -- reduce via GCD so
        # long IRs don't get an absurdly large FFT internally.
        from math import gcd

        g = gcd(int(original_sample_rate), int(target_sample_rate))
        up = int(target_sample_rate) // g
        down = int(original_sample_rate) // g
        prepared = resample_poly(trimmed, up, down).astype(np.float32)
    else:
        prepared = trimmed.astype(np.float32)

    return PreparedCabIr(
        samples=prepared,
        sample_rate=int(target_sample_rate),
        source_path=str(path),
        sha256=_sha256_file(path),
        original_sample_rate=int(original_sample_rate),
        original_channels=int(original_channels),
        original_frame_count=int(original_frame_count),
        prepared_frame_count=int(len(prepared)),
        leading_samples_trimmed=int(leading_trimmed),
    )


# (sha256, target_sample_rate, leading_silence_threshold_db) -> PreparedCabIr.
# A tiny bounded LRU-ish cache -- interactive preview must not re-read/
# resample the IR file on every slider movement (see doc "CAB CACHE /
# PERFORMANCE"), but this is a single-process local tool (see CLAUDE.md), so
# a simple dict with an eviction cap is enough; no need for real LRU bookkeeping.
_prepared_cache: dict[tuple[str, int, float], PreparedCabIr] = {}


def get_prepared_cab_ir(
    path: str | Path,
    target_sample_rate: int,
    leading_silence_threshold_db: float = LEADING_SILENCE_THRESHOLD_RELATIVE_DB,
) -> PreparedCabIr:
    """Cached wrapper around `load_and_prepare_cab_ir`, keyed by the file's
    own content hash (not just its path) so a re-uploaded/replaced IR at the
    same path never serves a stale prepared IR."""
    path = Path(path)
    sha256 = _sha256_file(path)
    key = (sha256, int(target_sample_rate), float(leading_silence_threshold_db))
    cached = _prepared_cache.get(key)
    if cached is not None:
        return cached

    prepared = load_and_prepare_cab_ir(path, target_sample_rate, leading_silence_threshold_db)
    if len(_prepared_cache) >= _MAX_CACHE_ENTRIES:
        _prepared_cache.pop(next(iter(_prepared_cache)))
    _prepared_cache[key] = prepared
    return prepared


def apply_cab_ir(audio: np.ndarray, prepared: PreparedCabIr) -> np.ndarray:
    """Convolve `audio` (mono float) with `prepared`'s FIR taps, causal, and
    return exactly `len(audio)` samples (full convolution, tail truncated) --
    see module docstring. Does NOT apply any limiter/normalization; that
    remains the caller's job (`preview_safety_limiter` for preview,
    `apply_peak_ceiling` for a training target -- see hybrid/safety.py)."""
    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise CabIrError("cab input audio contains NaN/Inf samples")
    n = len(audio)
    full = fftconvolve(audio, prepared.samples, mode="full")
    return full[:n].astype(np.float32)


@dataclass(frozen=True)
class CabDesign:
    """Frozen provenance for the shared Cabinet IR stage -- attached to both
    `hybrid.design.HybridDesign` and `hybrid.fixed_blend.BlendDesign` (a cab
    is a shared, mode-independent post-amp stage, see docs/blend-mode.md).

    `ir_working_path` is the server-side working copy used to actually
    re-run the convolution at generation time -- functional, not sensitive
    (see hybrid.cab_ir module docstring / app.py's upload endpoint); the raw
    IR audio itself is never embedded in JSON provenance.
    """

    selected: bool = False
    ir_working_path: Optional[str] = None
    original_filename: Optional[str] = None
    sha256: Optional[str] = None
    preview_enabled: bool = False
    baked: bool = False
    original_sample_rate: Optional[int] = None
    prepared_sample_rate: Optional[int] = None
    original_channels: Optional[int] = None
    original_frame_count: Optional[int] = None
    prepared_frame_count: Optional[int] = None
    leading_samples_trimmed: Optional[int] = None
    fir_history_samples: Optional[int] = None  # prepared_frame_count - 1, the serial RF cost when baked

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def cab_design_from_prepared(
    prepared: PreparedCabIr,
    original_filename: str,
    preview_enabled: bool,
    baked: bool,
) -> CabDesign:
    fir_history_samples = max(0, prepared.prepared_frame_count - 1)
    return CabDesign(
        selected=True,
        ir_working_path=prepared.source_path,
        original_filename=original_filename,
        sha256=prepared.sha256,
        preview_enabled=preview_enabled,
        baked=baked,
        original_sample_rate=prepared.original_sample_rate,
        prepared_sample_rate=prepared.sample_rate,
        original_channels=prepared.original_channels,
        original_frame_count=prepared.original_frame_count,
        prepared_frame_count=prepared.prepared_frame_count,
        leading_samples_trimmed=prepared.leading_samples_trimmed,
        fir_history_samples=fir_history_samples,
    )

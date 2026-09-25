"""Sample-accurate alignment check/correction between two amp renders.

Different NAM models (different architectures, different training setups) can
introduce slightly different fixed latency. Blending two renders that are offset
by even a handful of samples smears transients and can sound like a phasey
crossfade rather than a clean amp swap, so we measure and correct this before
blend.py ever runs.

CAUTION: `estimate_offset`/`align_to_reference` cross-correlate Amp A's render
directly against Amp B's render. When A and B are tonally very different (e.g.
a clean amp vs. a heavily distorted one), differences in distortion, filtering,
compression and phase response can themselves reduce the correlation score,
which this algorithm cannot distinguish from genuine fixed latency -- it could
"correct" a real tonal difference as if it were a timing offset, and shifting
Amp B's samples that way risks its own causality problems downstream. Until
NAM inference is wired in and we've established what latency guarantees (if
any) the official inference API actually makes, pass `enabled=False` to
`align_to_reference` to skip correction (renders are still truncated/padded to
match length) rather than trusting this cross-correlation blind.

`hybrid.core.align_diagnostic.analyse_alignment` is the read-only check for
whether a stable fixed offset actually exists; see docs/alignment_diagnostic.md.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import correlate

logger = logging.getLogger(__name__)

_MAX_LAG_SAMPLES_DEFAULT = 2000  # ~45ms at 44.1kHz; generous for NAM model latency


def estimate_offset(reference: np.ndarray, other: np.ndarray, max_lag: int = _MAX_LAG_SAMPLES_DEFAULT) -> int:
    """Return the sample offset of `other` relative to `reference`.

    Positive return value means `other` lags `reference` (starts later) and
    should be shifted backward (trim `offset` samples from its start) to align.
    Uses normalized cross-correlation restricted to +/-max_lag for speed and to
    avoid spurious long-lag matches on repetitive material.
    """
    reference = np.asarray(reference, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    n = min(len(reference), len(other))
    if n == 0:
        return 0

    ref = reference[:n]
    oth = other[:n]
    ref = ref - ref.mean()
    oth = oth - oth.mean()

    max_lag = min(max_lag, n - 1)
    if max_lag < 0:
        return 0
    # Score every lag at once: one FFT cross-correlation for the dot products
    # and prefix sums of squares for the per-lag norms, instead of a Python
    # loop doing O(n) work per lag. For lag >= 0 this scores ref[:n-lag]
    # against oth[lag:]; for lag < 0, ref[-lag:] against oth[:n+lag].
    lags = np.arange(-max_lag, max_lag + 1)
    dots = correlate(oth, ref, mode="full", method="fft")[lags + n - 1]
    ref_sq = np.concatenate(([0.0], np.cumsum(ref * ref)))
    oth_sq = np.concatenate(([0.0], np.cumsum(oth * oth)))
    pos = np.maximum(lags, 0)   # samples dropped from the front of `oth`
    neg = np.maximum(-lags, 0)  # samples dropped from the front of `ref`
    ref_energy = ref_sq[n - pos] - ref_sq[neg]
    oth_energy = oth_sq[n - neg] - oth_sq[pos]
    denom = np.sqrt(np.clip(ref_energy, 0.0, None) * np.clip(oth_energy, 0.0, None))
    usable = (n - np.abs(lags) >= 2) & (denom >= 1e-12)
    if not usable.any():
        return 0
    scores = np.full(len(lags), -np.inf)
    scores[usable] = dots[usable] / denom[usable]
    # argmax takes the first maximum, i.e. the most negative lag on a tie --
    # the same tie-break as scanning lags upward with a strict ">".
    return int(lags[int(np.argmax(scores))])


# `alignment_method` recorded in a design whose Amp B timing correction is a
# single integer chosen at design time and applied verbatim everywhere after.
FIXED_FROZEN_OFFSET_METHOD = "fixed-frozen-offset"


class LegacyAlignmentDesignError(ValueError):
    """A design with alignment enabled but no frozen-offset method: its original
    semantics re-estimated the offset on each audio it was applied to."""


def apply_fixed_offset(audio: np.ndarray, offset_samples: int, output_length: int) -> np.ndarray:
    """Apply an already-chosen integer timing correction to Amp B. Pure and
    deterministic: nothing is measured here.

    Same sign convention as `estimate_offset`: a positive offset means Amp B
    lags Amp A, so the correction trims `offset_samples` from B's start; a
    negative offset pads B's start with zeros. The result is always exactly
    `output_length` samples (zero-padded or truncated at the end) and keeps
    `audio`'s dtype.
    """
    if isinstance(offset_samples, (bool, np.bool_)) or not isinstance(offset_samples, (int, np.integer)):
        raise TypeError(f"offset_samples must be an integer, got {offset_samples!r}")
    if output_length < 0:
        raise ValueError(f"output_length must be >= 0, got {output_length}")
    audio = np.asarray(audio)
    offset = int(offset_samples)
    if offset > 0:
        shifted = audio[offset:]
    elif offset < 0:
        shifted = np.concatenate([np.zeros(-offset, dtype=audio.dtype), audio])
    else:
        shifted = audio
    if len(shifted) < output_length:
        return np.pad(shifted, (0, output_length - len(shifted)))
    return shifted[:output_length]


def resolve_alignment_request(align_enabled: bool, alignment_offset_samples: int) -> int:
    """The offset to apply for an (enabled, offset) pair: 0 when disabled.
    A non-zero offset with alignment disabled is a caller bug, not a request."""
    if not align_enabled:
        if alignment_offset_samples:
            raise ValueError("alignment_offset_samples is set but alignment is disabled")
        return 0
    return int(alignment_offset_samples)


def frozen_alignment_offset(design) -> int:
    """The integer offset a frozen design says to apply to Amp B (0 when
    alignment is off). Target generation, teacher reconstruction and
    validation use this instead of measuring anything.

    A design with `alignment_enabled=True` but no `alignment_method` predates
    frozen offsets: its original meaning was "re-estimate on whatever audio
    this is applied to". That is refused rather than silently reinterpreted.
    """
    if not getattr(design, "alignment_enabled", False):
        return 0
    method = getattr(design, "alignment_method", None)
    if method != FIXED_FROZEN_OFFSET_METHOD:
        raise LegacyAlignmentDesignError(
            "This design has alignment enabled from before fixed timing offsets were frozen "
            f"(alignment_method={method!r}). Its original behaviour re-estimated the offset on "
            "each audio it was applied to, so it is not reinterpreted as a fixed offset. "
            "Recreate the design to choose a verified fixed timing offset correction.")
    return int(design.alignment_offset_samples)


def align_to_reference(
    reference: np.ndarray,
    other: np.ndarray,
    max_lag: int = _MAX_LAG_SAMPLES_DEFAULT,
    enabled: bool = False,
) -> tuple[np.ndarray, int]:
    """Shift `other` to align with `reference`; returns (aligned_other, offset_applied).

    MEASURES and applies in one step, so no production path uses it: preview,
    target generation and validation apply a frozen integer with
    `apply_fixed_offset` instead (see `frozen_alignment_offset`).

    Aligned output is truncated/zero-padded to the same length as `reference`
    so downstream blending never has to worry about length mismatches.

    Defaults to `enabled=False`: skips the cross-correlation entirely (offset
    forced to 0, `other` only length-matched to `reference`) -- see the module
    docstring for why this matters when `reference` and `other` are tonally
    dissimilar amp renders. Pass `enabled=True` to opt back into the
    cross-correlation-based correction.
    """
    n = len(reference)
    if not enabled:
        logger.debug("align: disabled, skipping cross-correlation")
        return apply_fixed_offset(other, 0, n), 0

    offset = estimate_offset(reference, other, max_lag=max_lag)
    if offset:
        logger.info("align: corrected offset of %d samples between renders", offset)
    return apply_fixed_offset(other, offset, n), offset

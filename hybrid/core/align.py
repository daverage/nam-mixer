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


def align_to_reference(
    reference: np.ndarray,
    other: np.ndarray,
    max_lag: int = _MAX_LAG_SAMPLES_DEFAULT,
    enabled: bool = False,
) -> tuple[np.ndarray, int]:
    """Shift `other` to align with `reference`; returns (aligned_other, offset_applied).

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
        aligned = other[:n]
        if len(aligned) < n:
            aligned = np.pad(aligned, (0, n - len(aligned)))
        logger.debug("align: disabled, skipping cross-correlation")
        return aligned, 0

    offset = estimate_offset(reference, other, max_lag=max_lag)

    if offset == 0:
        logger.debug("align: no correction needed (offset=0)")
        aligned = other[:n]
        if len(aligned) < n:
            aligned = np.pad(aligned, (0, n - len(aligned)))
        return aligned, 0

    if offset > 0:
        # other lags reference: drop `offset` samples from the front of other
        shifted = other[offset:]
    else:
        # other leads reference: pad the front of other
        shifted = np.concatenate([np.zeros(-offset, dtype=other.dtype), other])

    if len(shifted) < n:
        shifted = np.pad(shifted, (0, n - len(shifted)))
    else:
        shifted = shifted[:n]

    logger.info("align: corrected offset of %d samples between renders", offset)
    return shifted, offset

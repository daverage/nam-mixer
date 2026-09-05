"""Sample-accurate alignment check/correction between two amp renders.

Different NAM models (different architectures, different training setups) can
introduce slightly different fixed latency. Blending two renders that are offset
by even a handful of samples smears transients and can sound like a phasey
crossfade rather than a clean amp swap, so we measure and correct this before
blend.py ever runs.
"""
from __future__ import annotations

import logging

import numpy as np

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
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        # Testing ref[i] ~= oth[i + lag]: lag > 0 means oth's matching content
        # starts `lag` samples later than ref's (oth lags ref).
        if lag >= 0:
            a = ref[: n - lag]
            b = oth[lag:]
        else:
            a = ref[-lag:]
            b = oth[: n + lag]
        if len(a) < 2:
            continue
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            continue
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def align_to_reference(reference: np.ndarray, other: np.ndarray, max_lag: int = _MAX_LAG_SAMPLES_DEFAULT) -> tuple[np.ndarray, int]:
    """Shift `other` to align with `reference`; returns (aligned_other, offset_applied).

    Aligned output is truncated/zero-padded to the same length as `reference`
    so downstream blending never has to worry about length mismatches.
    """
    offset = estimate_offset(reference, other, max_lag=max_lag)
    n = len(reference)

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

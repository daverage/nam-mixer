"""Automatic level matching between Amp A and Amp B, focused on the crossover region.

Deliberately NOT a whole-file RMS normalization -- see the main README/brief for
why: what matters is how loud each amp sounds for the input material that
actually lives near the crossover point, not their overall average loudness
across the whole DI (which may spend most of its time far from the crossover).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-10


@dataclass
class LevelMatchResult:
    amp_a_region_dbfs: float
    amp_b_region_dbfs: float
    suggested_b_trim_db: float
    n_samples_in_region: int


def _region_mask(envelope_db: np.ndarray, crossover_dbfs: float, transition_width_db: float) -> np.ndarray:
    """Samples whose dry-input envelope falls within the crossover transition band."""
    lo = crossover_dbfs - transition_width_db / 2.0
    hi = crossover_dbfs + transition_width_db / 2.0
    return (envelope_db >= lo) & (envelope_db <= hi)


def _rms_dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return 20.0 * np.log10(max(rms, _EPS))


def compute_crossover_trim(
    dry_envelope_db: np.ndarray,
    amp_a_render: np.ndarray,
    amp_b_render: np.ndarray,
    crossover_dbfs: float,
    transition_width_db: float,
) -> LevelMatchResult:
    """Suggest a trim (dB) to apply to Amp B so it matches Amp A's loudness
    specifically over the input region near the crossover point.

    `dry_envelope_db` must be sample-aligned with amp_a_render/amp_b_render
    (same length, same hop -- i.e. the output of envelope.rms_envelope_db on the
    same dry input used to drive both amp renders).

    Returns amp_a_trim implicitly fixed at 0 dB (Amp A trim is the user-facing
    "0.0 dB" reference point per the UI spec); only Amp B's trim is calculated.
    If too few samples fall in the transition band (e.g. crossover point picked
    outside the DI's actual level range), the region is widened once to avoid
    returning a meaningless result from a handful of samples.
    """
    n = min(len(dry_envelope_db), len(amp_a_render), len(amp_b_render))
    envelope_db = dry_envelope_db[:n]
    a = amp_a_render[:n]
    b = amp_b_render[:n]

    mask = _region_mask(envelope_db, crossover_dbfs, transition_width_db)
    min_samples = max(1, n // 1000)
    if mask.sum() < min_samples:
        mask = _region_mask(envelope_db, crossover_dbfs, transition_width_db * 3.0)

    a_region_db = _rms_dbfs(a[mask])
    b_region_db = _rms_dbfs(b[mask])

    if np.isfinite(a_region_db) and np.isfinite(b_region_db):
        suggested_trim = a_region_db - b_region_db
    else:
        suggested_trim = 0.0

    return LevelMatchResult(
        amp_a_region_dbfs=a_region_db,
        amp_b_region_dbfs=b_region_db,
        suggested_b_trim_db=suggested_trim,
        n_samples_in_region=int(mask.sum()),
    )

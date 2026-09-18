"""Capture-derived response-coordinate analysis -- see
docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md.

`hybrid.continuous_gain` treats a capture's KNOB position (Gain 1..10,
Volume 1..10, etc.) as the interpolation coordinate directly. This module
investigates whether that is itself a source of error: a knob is not
guaranteed to be linear with electrical gain (control/pot-law nonlinearity),
and even a linear electrical gain can drive the amplifier through a
nonlinear clean -> breakup -> saturation response. This module builds an
alternative coordinate DERIVED FROM THE CAPTURES THEMSELVES (how much the
rendered output actually changed between neighbouring captures), and
compares interpolating in that space against the existing knob-linear
baseline.

Explicitly NOT claiming to recover the physical pot law or separate control-
law nonlinearity from amplifier-response nonlinearity -- see the report's
Part 5 for why the data available here cannot distinguish those two causes.
This module only asks: does the size of the OUTPUT change from one capture
to the next track knob spacing, and does modelling that improve anything.

Raw ESR is deliberately NOT used as a response-distance measure between
independently-rendered neighbouring captures -- see
docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md for why sample-domain comparison
of two different renders of heavily-driven material can be misleading
(clipping-instant decorrelation). `spectral_magnitude_correlation` is used
instead, which survives that failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from .audio_metrics import spectral_magnitude_correlation
from .continuous_gain import GainCapture, GainCaptureSet, GroundTruthHarnessResult


@dataclass
class AdjacentStepMetrics:
    """Measured differences between two ADJACENT training captures -- the
    raw material both the response axis (Part 2) and the capture-placement
    score (Part 4/6) are built from."""

    lower_label: str
    upper_label: str
    lower_position: float
    upper_position: float
    level_change_db: float      # |active RMS delta|
    peak_change_db: float       # |peak dBFS delta|
    spectral_distance: float    # 1 - spectral_magnitude_correlation (0 = identical, up to 2)


def compute_adjacent_metrics(harness: GroundTruthHarnessResult, capture_set: GainCaptureSet) -> list[AdjacentStepMetrics]:
    """Part 1: per-step measurements across a sweep's TRAINING captures,
    in knob order."""
    training = capture_set.training_captures()
    steps = []
    for i in range(len(training) - 1):
        lower = harness.by_label(training[i].label)
        upper = harness.by_label(training[i + 1].label)
        n = min(len(lower.output), len(upper.output))
        level_change = abs(upper.active_rms_dbfs - lower.active_rms_dbfs)
        peak_lower = 20.0 * np.log10(max(float(np.max(np.abs(lower.output))), 1e-10))
        peak_upper = 20.0 * np.log10(max(float(np.max(np.abs(upper.output))), 1e-10))
        peak_change = abs(peak_upper - peak_lower)
        correlation = spectral_magnitude_correlation(lower.output[:n], upper.output[:n])
        spectral_distance = (1.0 - correlation) if np.isfinite(correlation) else float("nan")
        steps.append(AdjacentStepMetrics(
            lower_label=training[i].label, upper_label=training[i + 1].label,
            lower_position=training[i].control_position, upper_position=training[i + 1].control_position,
            level_change_db=level_change, peak_change_db=peak_change, spectral_distance=spectral_distance,
        ))
    return steps


def _value_range(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (0.0, 0.0)
    return (float(np.min(finite)), float(np.max(finite)))


def _normalize_with_range(values: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """Min-max normalize against an EXTERNALLY supplied (lo, hi), so
    components measured in different units (dB vs a correlation-derived
    distance) don't dominate a combined score merely by having a larger
    numeric range. Deliberately takes the range as a parameter rather than
    computing it from `values` itself: min-max normalizing a SMALL sample
    (as few as 2 values, e.g. in `oracle_response_mismatch`) against its own
    range always collapses to exactly {0, 1} regardless of the true
    magnitude ratio, destroying the information being measured -- the range
    must come from a larger reference sample (the full sweep's steps) for
    the result to be meaningful. NaNs (e.g. a degenerate spectral
    correlation) are treated as 0 contribution rather than poisoning the
    whole normalization."""
    values = np.asarray(values, dtype=np.float64)
    lo, hi = value_range
    span = hi - lo
    if span <= 1e-12:
        return np.zeros(len(values))
    return np.nan_to_num((values - lo) / span, nan=0.0)


def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a series against ITS OWN range -- appropriate when
    `values` already represents the full reference sample (e.g. building a
    response axis from every step in a sweep). See `_normalize_with_range`
    for the case where a smaller sample must be normalized against a
    larger reference's range instead."""
    return _normalize_with_range(values, _value_range(values))


DEFAULT_RESPONSE_WEIGHTS = {"level": 1.0, "spectral": 1.0}
RMS_ONLY_WEIGHTS = {"level": 1.0, "spectral": 0.0}


def combined_step_scores(steps: list[AdjacentStepMetrics], weights: dict[str, float] = DEFAULT_RESPONSE_WEIGHTS) -> np.ndarray:
    """Part 2/6: one normalized, equally-weighted-by-default score per step,
    combining level and spectral distance. `RMS_ONLY_WEIGHTS` reproduces the
    level-only baseline used throughout the earlier reports. Weights are
    fixed/equal by design (doc: "Do not tune weights against the withheld
    test points until they produce a desired result"), not fit to any
    dataset here.
    """
    level = _normalize(np.array([s.level_change_db for s in steps]))
    spectral = _normalize(np.array([s.spectral_distance for s in steps]))
    total_weight = weights.get("level", 0.0) + weights.get("spectral", 0.0)
    if total_weight <= 0:
        raise ValueError("At least one of level/spectral weights must be positive.")
    return (weights.get("level", 0.0) * level + weights.get("spectral", 0.0) * spectral) / total_weight


@dataclass
class ResponseAxis:
    """A monotonic, capture-derived coordinate over a sweep's TRAINING
    positions: `coordinates[i]` is the cumulative normalized response
    distance travelled by `positions[i]`, rescaled to end at 1.0."""

    positions: list[float]
    coordinates: list[float]
    weights: dict[str, float]


def build_response_axis(steps: list[AdjacentStepMetrics], first_position: float, weights: dict[str, float] = DEFAULT_RESPONSE_WEIGHTS) -> ResponseAxis:
    """Part 2: r[0] = 0, r[n] = r[n-1] + score[n-1], then normalized to end at 1.0."""
    if not steps:
        raise ValueError("Need at least one adjacent step to build a response axis.")
    scores = combined_step_scores(steps, weights)
    cumulative = np.concatenate([[0.0], np.cumsum(scores)])
    total = cumulative[-1]
    normalized = cumulative / total if total > 1e-12 else cumulative
    positions = [first_position] + [s.upper_position for s in steps]
    return ResponseAxis(positions=positions, coordinates=[float(c) for c in normalized], weights=dict(weights))


def response_coordinate_for_position(axis: ResponseAxis, knob_position: float) -> float:
    """Non-oracle: monotone PCHIP interpolation/extrapolation of the
    TRAINING-only response axis, evaluated at an arbitrary (e.g. hidden)
    knob position. Only ever uses information available at inference time
    (the query's known knob position; never its own unmeasured output)."""
    if len(axis.positions) < 2:
        raise ValueError("Need at least 2 training points to build a response axis.")
    interpolator = PchipInterpolator(axis.positions, axis.coordinates, extrapolate=True)
    return float(np.clip(interpolator(knob_position), 0.0, 1.0))


def interpolate_output_response_coordinate(
    harness: GroundTruthHarnessResult,
    capture_set: GainCaptureSet,
    normalized_position: float,
    axis: ResponseAxis,
    *,
    neighbors: Optional[tuple[GainCapture, GainCapture]] = None,
) -> np.ndarray:
    """Part 3, Test B -- non-oracle response-coordinate interpolation: same
    bracket selection as `hybrid.continuous_gain.interpolate_output`
    (response coordinate is monotonic in knob position by construction, so
    the nearest-neighbour bracket is unchanged), but the blend WEIGHT within
    that bracket comes from the response axis instead of the raw knob
    fraction.
    """
    lower, upper = neighbors if neighbors is not None else capture_set.neighbors(normalized_position)
    training = capture_set.training_captures()
    lo_pos, hi_pos = training[0].control_position, training[-1].control_position
    knob_position = lo_pos + normalized_position * (hi_pos - lo_pos)

    r_query = response_coordinate_for_position(axis, knob_position)
    r_lower = axis.coordinates[axis.positions.index(lower.control_position)]
    r_upper = axis.coordinates[axis.positions.index(upper.control_position)]
    span = r_upper - r_lower
    local_blend = 0.0 if span <= 1e-12 else float(np.clip((r_query - r_lower) / span, 0.0, 1.0))

    lower_output = harness.by_label(lower.label).output
    upper_output = harness.by_label(upper.label).output
    n = min(len(lower_output), len(upper_output))
    return lower_output[:n] * (1.0 - local_blend) + upper_output[:n] * local_blend


@dataclass
class OracleMismatch:
    """Part 3, Test A -- DIAGNOSTIC ONLY, not a production method: how far a
    withheld capture's TRUE local response fraction (computed from its own
    real neighbouring distances, which a production system never has) sits
    from the knob-linear fraction a production system WOULD have used."""

    hidden_label: str
    knob_linear_fraction: float
    true_response_fraction: float
    mismatch: float


def oracle_response_mismatch(
    harness: GroundTruthHarnessResult,
    lower: GainCapture,
    hidden: GainCapture,
    upper: GainCapture,
    reference_steps: list[AdjacentStepMetrics],
    weights: dict[str, float] = DEFAULT_RESPONSE_WEIGHTS,
) -> OracleMismatch:
    """ORACLE/DIAGNOSTIC ONLY (see class docstring): uses the hidden
    capture's own real rendered output to measure where it actually sits,
    in response space, between its two real neighbours. Never use this to
    reconstruct the hidden capture itself -- that would be oracle leakage.

    `reference_steps` (the full sweep's OWN adjacent-step measurements, from
    `compute_adjacent_metrics`) supplies the normalization RANGE -- min-max
    normalizing just this function's two local distances against themselves
    would always collapse to exactly {0, 1} regardless of their true
    magnitude ratio (see `_normalize_with_range`'s docstring), which would
    make `true_response_fraction` meaningless.
    """
    lower_r = harness.by_label(lower.label)
    hidden_r = harness.by_label(hidden.label)
    upper_r = harness.by_label(upper.label)

    n1 = min(len(lower_r.output), len(hidden_r.output))
    n2 = min(len(hidden_r.output), len(upper_r.output))
    level1 = abs(hidden_r.active_rms_dbfs - lower_r.active_rms_dbfs)
    level2 = abs(upper_r.active_rms_dbfs - hidden_r.active_rms_dbfs)
    corr1 = spectral_magnitude_correlation(lower_r.output[:n1], hidden_r.output[:n1])
    corr2 = spectral_magnitude_correlation(hidden_r.output[:n2], upper_r.output[:n2])
    spectral1 = (1.0 - corr1) if np.isfinite(corr1) else 0.0
    spectral2 = (1.0 - corr2) if np.isfinite(corr2) else 0.0

    level_range = _value_range(np.array([s.level_change_db for s in reference_steps] + [level1, level2]))
    spectral_range = _value_range(np.array([s.spectral_distance for s in reference_steps] + [spectral1, spectral2]))
    level_norm = _normalize_with_range(np.array([level1, level2]), level_range)
    spectral_norm = _normalize_with_range(np.array([spectral1, spectral2]), spectral_range)
    total_weight = weights.get("level", 0.0) + weights.get("spectral", 0.0)
    d1 = (weights.get("level", 0.0) * level_norm[0] + weights.get("spectral", 0.0) * spectral_norm[0]) / total_weight
    d2 = (weights.get("level", 0.0) * level_norm[1] + weights.get("spectral", 0.0) * spectral_norm[1]) / total_weight

    true_fraction = 0.5 if (d1 + d2) <= 1e-12 else d1 / (d1 + d2)
    knob_fraction = (hidden.control_position - lower.control_position) / (upper.control_position - lower.control_position)
    return OracleMismatch(
        hidden_label=hidden.label, knob_linear_fraction=knob_fraction,
        true_response_fraction=true_fraction, mismatch=abs(true_fraction - knob_fraction),
    )

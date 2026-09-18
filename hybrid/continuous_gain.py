"""Continuous Gain Model research harness -- see docs/CONTINUOUS_GAIN.md.

This is the research branch's Phase 1 (Ground Truth Harness) and Phase 2
(baseline piecewise interpolation, "Approach A: Output Interpolation" in the
doc) ONLY. It intentionally does not implement Phase 3+ (improved
reconstruction methods, synthetic intermediate NAMs, a continuous teacher
with an explicit gain-position input, or student training) until Phase 2's
leave-one-out validation demonstrates the baseline is worth building on --
see the doc's "Stop Conditions"/"Development Order" sections.

Per the doc's three-concepts split, this module keeps distinct:

- `GainCapture.control_position` -- the physical amp-gain knob metadata
  (arbitrary units the user entered, e.g. "5" on a 1-10 knob).
- `normalized_position` -- that value rescaled to 0..1 across the TRAINING
  captures' range (hidden/withheld captures reuse the same scale so they can
  be compared at their true relative position, even outside 0..1).
- NAM player input level is NOT modelled here at all yet -- that is Phase 5
  ("Input-to-Gain Mapping" / "Compatibility Mapping"), which requires a
  student NAM to exist first.

Reuses existing primitives rather than inventing parallel ones (see the
doc's "Architecture Principle" and "Coding-Agent Instruction" sections):
`hybrid.calibration` for NAM input-level reconciliation, the active-signal
convention from `hybrid.coverage`/`hybrid.fixed_blend.compute_active_trim`
for output-level measurement, and `hybrid.validation.compute_esr_metrics`
for comparing a reconstructed capture against a withheld real one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio_metrics import band_energy_dbfs, rms_dbfs, spectral_magnitude_correlation
from .calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU, input_calibration_gain_db
from .coverage import ACTIVE_SIGNAL_THRESHOLD_DBFS, active_signal_mask
from .envelope import DEFAULT_BOUNDED_ENVELOPE_CONFIG, BoundedEnvelopeConfig, bounded_causal_envelope_db
from .nam_loader import NamModel
from .render import render
from .validation import compute_esr_metrics

# Mirrors hybrid.calibration._UNAVAILABLE_WARNING's reasoning, generalized
# from a pair to an arbitrary-size capture set: compensating some captures
# and not others would apply a real physical assumption asymmetrically
# across what is supposed to be a single controlled gain sweep.
_PARTIAL_CALIBRATION_WARNING = (
    "Input calibration unavailable for this capture set: not every capture "
    "reports input_level_dbu. Using raw digital level for all captures."
)


@dataclass
class GainCapture:
    """One source `.nam` capture plus its physical gain-control metadata.

    `control_position` is in whatever arbitrary units the user entered (a
    "1" to "10" knob, a 0.0-1.0 pot travel fraction, etc.) -- see the doc's
    "Minimum Capture Requirements" section for why irregular spacing must be
    supported rather than assuming evenly-spaced positions.

    `hidden=True` marks a withheld validation-only capture (doc: "Recommended
    Experimental Capture Set" / "Critical Validation Strategy") -- it is
    never used to build the interpolation, only to score it.
    """

    model: NamModel
    control_position: float
    label: str = ""
    hidden: bool = False

    def __post_init__(self):
        if not self.label:
            self.label = f"gain_{self.control_position:g}"


@dataclass
class GainCaptureSet:
    """An ordered collection of captures of the SAME amplifier configuration
    across its gain range -- the doc's "New Top-Level Model Type" data model,
    deliberately independent of the existing two-amp `amp_a`/`amp_b`
    abstractions (see "Internal Data Model" / "Architecture Principle")."""

    captures: list[GainCapture]

    def __post_init__(self):
        if len(self.training_captures()) < 2:
            raise ValueError(
                "A gain capture set needs at least 2 non-hidden (training) captures "
                "to interpolate between -- see docs/CONTINUOUS_GAIN.md 'Minimum Capture Requirements'."
            )
        positions = [c.control_position for c in self.captures]
        if len(set(positions)) != len(positions):
            raise ValueError("Gain capture control positions must be unique within a capture set.")

    def training_captures(self) -> list[GainCapture]:
        """Non-hidden captures, sorted by control position -- these define
        both the interpolation anchors and the normalization range."""
        return sorted((c for c in self.captures if not c.hidden), key=lambda c: c.control_position)

    def hidden_captures(self) -> list[GainCapture]:
        """Withheld validation-only captures, sorted by control position."""
        return sorted((c for c in self.captures if c.hidden), key=lambda c: c.control_position)

    def normalized_position(self, capture: GainCapture) -> float:
        """Rescale `capture.control_position` to the training captures' 0..1
        range (doc: "Internally, control positions should be normalised to
        0.0 -> 1.0"). A hidden capture between two training anchors lands
        inside 0..1; one outside the training range is extrapolated (still
        returned, since Phase 1 must be able to report that the requested
        position is unsupported rather than silently clamping it away)."""
        training = self.training_captures()
        lo = training[0].control_position
        hi = training[-1].control_position
        span = hi - lo
        if span <= 0:
            raise ValueError("Training captures must span a nonzero control-position range.")
        return (capture.control_position - lo) / span

    def neighbors(self, normalized_position: float) -> tuple[GainCapture, GainCapture]:
        """Return the two training captures bracketing `normalized_position`
        for piecewise interpolation (doc: "Piecewise Interpolation" -- do not
        interpolate the two extreme captures directly unless that IS the
        requested pair). Clamps to the outermost pair if out of range."""
        training = self.training_captures()
        positions = [self.normalized_position(c) for c in training]
        if normalized_position <= positions[0]:
            return training[0], training[1]
        if normalized_position >= positions[-1]:
            return training[-2], training[-1]
        for i in range(len(training) - 1):
            if positions[i] <= normalized_position <= positions[i + 1]:
                return training[i], training[i + 1]
        return training[-2], training[-1]  # unreachable, satisfies type checkers


def _capture_calibration_gains_db(
    captures: list[GainCapture],
    mode: str,
    reference_input_level_dbu: float,
) -> tuple[dict[str, float], Optional[str]]:
    """Per-capture NAM input-calibration gain (dB), reusing the official
    plugin formula from `hybrid.calibration.input_calibration_gain_db`.

    Generalizes `hybrid.calibration.resolve_calibration`'s two-model
    Auto/Raw rule (compensate only when BOTH models report calibration
    metadata) to an arbitrary-size capture set: "auto" compensates only when
    EVERY capture reports `input_level_dbu`; otherwise every capture falls
    back to raw (0 dB) rather than calibrating some captures and not others.
    """
    if mode == "raw":
        return {c.label: 0.0 for c in captures}, None
    levels = [c.model.input_level_dbu for c in captures]
    if any(level is None for level in levels):
        return {c.label: 0.0 for c in captures}, _PARTIAL_CALIBRATION_WARNING
    return {
        c.label: input_calibration_gain_db(reference_input_level_dbu, c.model.input_level_dbu)
        for c in captures
    }, None


@dataclass
class RenderedGainCapture:
    """Phase 1 per-capture record -- see the doc's Phase 1 requirement to
    record, for every source capture: control position, input calibration
    metadata, measured active RMS, output trim used, normalisation mode, and
    RF-related metadata (the last is 'not applicable' at Phase 1: no new
    temporal processing has been introduced yet -- see `rf_note`)."""

    capture: GainCapture
    normalized_position: float
    output: np.ndarray
    calibration_gain_db: float
    active_rms_dbfs: float
    n_active_samples: int
    rf_note: str = (
        "Phase 1 harness introduces no new temporal state beyond render()/"
        "bounded_causal_envelope_db (already covered by existing RF policy)."
    )


@dataclass
class GroundTruthHarnessResult:
    rendered: list[RenderedGainCapture]
    calibration_mode: str
    calibration_warning: Optional[str]
    output_normalization_amount: float

    def by_label(self, label: str) -> RenderedGainCapture:
        for r in self.rendered:
            if r.capture.label == label:
                return r
        raise KeyError(label)


def run_ground_truth_harness(
    capture_set: GainCaptureSet,
    dry: np.ndarray,
    sample_rate: int,
    *,
    calibration_mode: str = "auto",
    reference_input_level_dbu: float = DEFAULT_REFERENCE_INPUT_LEVEL_DBU,
    output_normalization_amount: float = 0.0,
    envelope_config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG,
) -> GroundTruthHarnessResult:
    """Phase 1: render identical `dry` material through EVERY capture
    (including hidden validation ones), and record the measurements Phase 2+
    needs -- see docs/CONTINUOUS_GAIN.md "Phase 1: Ground Truth Harness".

    `output_normalization_amount` implements the doc's "Candidate
    Partial-Normalisation Method": 0.0 preserves each capture's real output
    level, 1.0 RMS-matches every capture to the first (lowest-gain) training
    capture over the DI's active material, values between partially do both.
    Uses `hybrid.coverage.active_signal_mask` (the same active/silence
    convention `hybrid.fixed_blend.compute_active_trim` uses) so silence
    doesn't dominate the RMS measurement.

    No interpolation, no student training -- this only builds the ground
    truth dataset used by everything downstream.
    """
    if not (0.0 <= output_normalization_amount <= 1.0):
        raise ValueError("output_normalization_amount must be within 0.0..1.0")

    dry = np.asarray(dry, dtype=np.float32)
    envelope_db = bounded_causal_envelope_db(dry, sample_rate, envelope_config)
    mask = active_signal_mask(envelope_db, ACTIVE_SIGNAL_THRESHOLD_DBFS)

    all_captures = sorted(capture_set.captures, key=lambda c: c.control_position)
    calibration_gains_db, calibration_warning = _capture_calibration_gains_db(
        all_captures, calibration_mode, reference_input_level_dbu,
    )

    raw_outputs: dict[str, np.ndarray] = {}
    raw_active_dbfs: dict[str, float] = {}
    for capture in all_captures:
        gain_db = calibration_gains_db[capture.label]
        input_audio = dry * (10.0 ** (gain_db / 20.0)) if gain_db else dry
        output = render(capture.model, input_audio.astype(np.float32), sample_rate)
        raw_outputs[capture.label] = output
        n = min(len(output), len(mask))
        raw_active_dbfs[capture.label] = rms_dbfs(output[:n][mask[:n]])

    # Reference for partial normalisation: the lowest-gain TRAINING capture,
    # matching the doc's "choose one capture ... as the reference level".
    reference_label = capture_set.training_captures()[0].label
    reference_dbfs = raw_active_dbfs[reference_label]

    rendered = []
    for capture in all_captures:
        output = raw_outputs[capture.label]
        if output_normalization_amount > 0.0 and np.isfinite(raw_active_dbfs[capture.label]) and np.isfinite(reference_dbfs):
            full_trim_db = reference_dbfs - raw_active_dbfs[capture.label]
            applied_trim_db = full_trim_db * output_normalization_amount
            output = output * (10.0 ** (applied_trim_db / 20.0))
        n = min(len(output), len(mask))
        rendered.append(RenderedGainCapture(
            capture=capture,
            normalized_position=capture_set.normalized_position(capture),
            output=output,
            calibration_gain_db=calibration_gains_db[capture.label],
            active_rms_dbfs=rms_dbfs(output[:n][mask[:n]]),
            n_active_samples=int(mask[:n].sum()),
        ))

    return GroundTruthHarnessResult(
        rendered=sorted(rendered, key=lambda r: r.normalized_position),
        calibration_mode=calibration_mode,
        calibration_warning=calibration_warning,
        output_normalization_amount=output_normalization_amount,
    )


# Doc decision gate: "If errors mainly come from spectral differences above
# 3 kHz, a simpler frequency-aware correction may be more useful than the
# full Hybrid machinery" -- so the low/high split point is fixed at 3 kHz
# rather than left as an unlabelled magic number in every caller.
SPECTRAL_SPLIT_HZ = 3000.0


def _local_blend(capture_set: GainCaptureSet, lower: GainCapture, upper: GainCapture, normalized_position: float) -> float:
    lower_pos = capture_set.normalized_position(lower)
    upper_pos = capture_set.normalized_position(upper)
    span = upper_pos - lower_pos
    local_blend = 0.0 if span <= 0 else (normalized_position - lower_pos) / span
    return float(np.clip(local_blend, 0.0, 1.0))


@dataclass
class CaptureAnomaly:
    """One advisory finding from `detect_capture_anomalies` -- doc: 'Warnings
    should be advisory. Do not reject captures simply because an amplifier
    behaves unusually.'"""

    label: str
    kind: str  # "non_monotonic_level"
    detail: str


def detect_capture_anomalies(
    harness: GroundTruthHarnessResult,
    capture_set: GainCaptureSet,
    *,
    reversal_threshold_db: float = 0.5,
) -> list[CaptureAnomaly]:
    """Doc "Capture Validation": flag TRAINING captures whose active RMS goes
    the wrong way relative to their control-position neighbours (doc's
    "unusual level jumps" / "non-monotonic distortion changes") -- e.g. a
    Gain-4 capture measuring quieter than BOTH its Gain-3 and Gain-5
    neighbours, which never happens for a real, correctly-labelled gain
    sweep of an amp whose output level increases (even nonlinearly) with
    gain. Advisory only: this does not change interpolation behaviour, it
    only surfaces a finding a caller can choose to warn about or exclude
    the affected capture over.

    `reversal_threshold_db` filters out sub-threshold measurement noise --
    only a reversal at least this large (in dB) relative to BOTH neighbours
    is reported.
    """
    training = capture_set.training_captures()
    anomalies = []
    for i in range(1, len(training) - 1):
        prev_r = harness.by_label(training[i - 1].label).active_rms_dbfs
        curr_r = harness.by_label(training[i].label).active_rms_dbfs
        next_r = harness.by_label(training[i + 1].label).active_rms_dbfs
        if not all(np.isfinite(v) for v in (prev_r, curr_r, next_r)):
            continue
        drop_from_prev = prev_r - curr_r
        drop_from_next = next_r - curr_r
        if drop_from_prev >= reversal_threshold_db and drop_from_next >= reversal_threshold_db:
            anomalies.append(CaptureAnomaly(
                label=training[i].label,
                kind="non_monotonic_level",
                detail=(
                    f"{training[i].label} measures {curr_r:.2f} dBFS active RMS, "
                    f"{drop_from_prev:.2f} dB quieter than {training[i-1].label} ({prev_r:.2f} dBFS) "
                    f"and {drop_from_next:.2f} dB quieter than {training[i+1].label} ({next_r:.2f} dBFS) -- "
                    "both its control-position neighbours are louder, which should not happen for a "
                    "correctly labelled monotonic gain sweep."
                ),
            ))
    return anomalies


def interpolate_output(
    harness: GroundTruthHarnessResult,
    capture_set: GainCaptureSet,
    normalized_position: float,
    *,
    neighbors: Optional[tuple[GainCapture, GainCapture]] = None,
) -> np.ndarray:
    """Phase 2 baseline -- doc "Approach A: Output Interpolation": linear
    crossfade of the two bracketing TRAINING captures' rendered output.
    Deliberately the simplest possible reconstruction, to be used as the
    control comparison for any more elaborate method (Approach B/C in the
    doc) -- do not treat this as the final interpolation method.

    `neighbors`, if given, overrides `capture_set.neighbors()` -- used to
    test wider-than-nearest spacing (doc: "does capture spacing matter more
    than capture count?"), e.g. reconstructing Gain 6 from Gain 2/Gain 10
    instead of its immediate Gain 4/Gain 8 neighbours.
    """
    lower, upper = neighbors if neighbors is not None else capture_set.neighbors(normalized_position)
    local_blend = _local_blend(capture_set, lower, upper, normalized_position)

    lower_output = harness.by_label(lower.label).output
    upper_output = harness.by_label(upper.label).output
    n = min(len(lower_output), len(upper_output))
    return lower_output[:n] * (1.0 - local_blend) + upper_output[:n] * local_blend


def _split_bands(audio: np.ndarray, sample_rate: int, split_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Split `audio` into (below split_hz, at-or-above split_hz) bands via a
    hard FFT-domain mask -- the two halves sum back to the original signal
    exactly (up to float rounding), unlike `band_energy_dbfs` which only
    measures one band's level."""
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    low_mask = freqs < split_hz
    low_spectrum = np.zeros_like(spectrum)
    low_spectrum[low_mask] = spectrum[low_mask]
    low = np.fft.irfft(low_spectrum, n=len(audio))
    high = audio - low
    return low, high


def interpolate_output_hf_corrected(
    harness: GroundTruthHarnessResult,
    capture_set: GainCaptureSet,
    normalized_position: float,
    sample_rate: int,
    *,
    split_hz: float = SPECTRAL_SPLIT_HZ,
    neighbors: Optional[tuple[GainCapture, GainCapture]] = None,
) -> np.ndarray:
    """Phase 3 candidate -- doc "a simpler frequency-aware correction" for
    the case where the baseline's residual error concentrates above
    `split_hz` (see docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md). Still causal/
    production-plausible (unlike `interpolate_output_level_matched`): it
    predicts the missing capture's high-frequency band energy by linearly
    interpolating the TWO NEIGHBOURS' OWN measured high-band energy at the
    same knob-linear blend weight used for the waveform crossfade -- no
    access to the real hidden capture's level -- then rescales only the
    reconstruction's high band (>= split_hz) to match that prediction,
    leaving the low band untouched.

    This is a targeted correction, not a claim that linear HF-energy
    interpolation is the right model of a real amp's high-frequency gain
    response -- it exists to test whether ANY lightweight, local correction
    recovers more accuracy than plain `interpolate_output`, before reaching
    for full Hybrid/Character machinery.
    """
    lower, upper = neighbors if neighbors is not None else capture_set.neighbors(normalized_position)
    local_blend = _local_blend(capture_set, lower, upper, normalized_position)

    lower_output = harness.by_label(lower.label).output
    upper_output = harness.by_label(upper.label).output
    n = min(len(lower_output), len(upper_output))
    lower_output, upper_output = lower_output[:n], upper_output[:n]
    reconstructed = lower_output * (1.0 - local_blend) + upper_output * local_blend

    lower_hf_dbfs = band_energy_dbfs(lower_output, sample_rate, split_hz, None)
    upper_hf_dbfs = band_energy_dbfs(upper_output, sample_rate, split_hz, None)
    if not (np.isfinite(lower_hf_dbfs) and np.isfinite(upper_hf_dbfs)):
        return reconstructed
    predicted_hf_dbfs = lower_hf_dbfs * (1.0 - local_blend) + upper_hf_dbfs * local_blend

    low_band, high_band = _split_bands(reconstructed, sample_rate, split_hz)
    current_hf_dbfs = rms_dbfs(high_band)
    if not np.isfinite(current_hf_dbfs):
        return reconstructed
    trim_db = predicted_hf_dbfs - current_hf_dbfs
    return low_band + high_band * (10.0 ** (trim_db / 20.0))


def interpolate_output_level_matched(
    harness: GroundTruthHarnessResult,
    capture_set: GainCaptureSet,
    normalized_position: float,
    target_active_rms_dbfs: float,
    *,
    neighbors: Optional[tuple[GainCapture, GainCapture]] = None,
) -> np.ndarray:
    """Research ABLATION ONLY -- not a candidate production method.

    Doc: "The RMS progression is already nonlinear, so a 50/50 blend may not
    be the most accurate midpoint even when the knob position is halfway."
    This answers that question in isolation from everything else: it takes
    `interpolate_output`'s same waveform-shape crossfade, but rescales the
    RESULT to `target_active_rms_dbfs` -- the withheld capture's OWN real
    measured level, an oracle a production system does not have access to
    (the whole point of interpolation is not knowing the hidden capture's
    level in advance). Comparing this against plain `interpolate_output`
    isolates "does getting the waveform SHAPE right via linear-knob
    crossfade already capture most of the achievable accuracy" from "is
    knob-linear LEVEL blending the main source of error" -- the latter would
    point at a measured-level-based reconstruction method as more promising
    than knob-linear blending, without yet claiming the shape is also solved.
    """
    reconstructed = interpolate_output(harness, capture_set, normalized_position, neighbors=neighbors)
    current_dbfs = rms_dbfs(reconstructed)
    if not np.isfinite(current_dbfs) or not np.isfinite(target_active_rms_dbfs):
        return reconstructed
    trim_db = target_active_rms_dbfs - current_dbfs
    return reconstructed * (10.0 ** (trim_db / 20.0))


@dataclass
class LeaveOneOutResult:
    capture_label: str
    control_position: float
    normalized_position: float
    neighbor_labels: tuple[str, str]
    metrics: dict


def _spectral_band_metrics(reconstructed: np.ndarray, real: np.ndarray, sample_rate: int) -> dict:
    n = min(len(reconstructed), len(real))
    low_delta = abs(
        band_energy_dbfs(reconstructed[:n], sample_rate, 0.0, SPECTRAL_SPLIT_HZ)
        - band_energy_dbfs(real[:n], sample_rate, 0.0, SPECTRAL_SPLIT_HZ)
    )
    high_delta = abs(
        band_energy_dbfs(reconstructed[:n], sample_rate, SPECTRAL_SPLIT_HZ, None)
        - band_energy_dbfs(real[:n], sample_rate, SPECTRAL_SPLIT_HZ, None)
    )
    return {
        "low_freq_delta_db": low_delta,
        "high_freq_delta_db": high_delta,
        # See spectral_magnitude_correlation's docstring: raw ESR is a
        # sample-domain metric and can look catastrophic on heavily
        # saturated/high-gain material even when the reconstruction is
        # spectrally/tonally close -- this is the check that catches that.
        "spectral_correlation": spectral_magnitude_correlation(reconstructed[:n], real[:n]),
    }


def leave_one_out_validation(
    harness: GroundTruthHarnessResult,
    capture_set: GainCaptureSet,
    sample_rate: int,
    *,
    neighbors: Optional[tuple[GainCapture, GainCapture]] = None,
) -> list[LeaveOneOutResult]:
    """Doc "Critical Validation Strategy": for every HIDDEN capture, build the
    Phase 2 baseline reconstruction from its bracketing training captures and
    score it against the real rendered output for that same capture, using
    `hybrid.validation.compute_esr_metrics` (already the project's ESR/RMS/
    peak comparison, not a new metric implementation) plus a low/high
    spectral-band delta split at `SPECTRAL_SPLIT_HZ` (doc's Phase-3 decision
    gate on whether errors concentrate above 3 kHz).

    `neighbors` overrides the automatically-chosen bracketing pair for every
    hidden capture -- used to test wider-than-nearest spacing (only sensible
    with exactly one hidden capture per call in that case).

    Requires the hidden captures to already be inside the harness's
    positions and inside the training captures' normalized range (interior
    withheld points, per the doc's "Recommended Experimental Capture Set" --
    extrapolation beyond the training range is a different, not-yet-answered
    question and is rejected here rather than silently reported as if it
    were interpolation).
    """
    results = []
    for hidden in capture_set.hidden_captures():
        rendered_hidden = harness.by_label(hidden.label)
        pos = rendered_hidden.normalized_position
        training_positions = [capture_set.normalized_position(c) for c in capture_set.training_captures()]
        if not (min(training_positions) <= pos <= max(training_positions)):
            raise ValueError(
                f"Hidden capture {hidden.label!r} lies outside the training range -- "
                "leave-one-out validation only evaluates interpolation, not extrapolation."
            )
        pair = neighbors if neighbors is not None else capture_set.neighbors(pos)
        reconstructed = interpolate_output(harness, capture_set, pos, neighbors=pair)
        real = rendered_hidden.output
        n = min(len(reconstructed), len(real))
        metrics = compute_esr_metrics(reconstructed[:n], real[:n])
        metrics.update(_spectral_band_metrics(reconstructed, real, sample_rate))
        results.append(LeaveOneOutResult(
            capture_label=hidden.label,
            control_position=hidden.control_position,
            normalized_position=pos,
            neighbor_labels=(pair[0].label, pair[1].label),
            metrics=metrics,
        ))
    return results

"""Character Blend teacher construction: one donor path plus broad correction."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve, firwin2, minimum_phase

from .audio_metrics import rms_dbfs as _shared_rms_dbfs
from .cab_ir import CabDesign
from .calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU
from .character_analysis import AmpCharacterAnalysis, CharacterAnalysisConfig, analyse_rendered_audio
from .envelope import bounded_causal_envelope_db, bounded_envelope_max_history_ms

_EPS = 1e-10

# Version 1 used a centred donor crossfade, which made the output before a
# switch depend on future control samples.  Version 2 starts a bounded ramp at
# the switch instead.  Keep the old value readable so existing bundles are not
# silently reinterpreted when they are opened again.
CHARACTER_TEACHER_SEMANTICS_VERSION = 2
DONOR_TRANSITION_MS = 10.0


def _clamp(value: float) -> float: return max(0.0, min(1.0, float(value)))


def character_temporal_history_samples(sample_rate: int, smoothing_ms: float) -> dict[str, int]:
    """History introduced by the Character Blend teacher itself.

    The source NAM paths, envelope, and these paths run in parallel until the
    final output.  Within the drive-control path, however, envelope smoothing
    and donor transition state are serial dependencies.  The broad correction
    FIR is serial with each selected amp path.  Reporting the pieces avoids
    incorrectly treating all of them as one Dynamic-Hybrid envelope branch.
    """
    smoothing = max(0, int(sample_rate * max(0.0, smoothing_ms) / 1000.0) - 1)
    transition = max(0, int(round(sample_rate * DONOR_TRANSITION_MS / 1000.0)) - 1)
    # scipy.minimum_phase(..., half=True) yields ceil(129 / 2) = 65 taps.
    correction_fir = 64
    return {
        "drive_smoothing_serial_samples": smoothing,
        "donor_transition_serial_samples": transition,
        "correction_fir_serial_samples": correction_fir,
    }


@dataclass(frozen=True)
class CharacterBlendDesign:
    amp_a_path: str
    amp_b_path: str
    tone_mix_b: float = 0.5
    feel_mix_b: float = 0.5
    drive_mix_b: float = 0.5
    drive_low_mix_b: Optional[float] = None
    drive_mid_mix_b: Optional[float] = None
    drive_high_mix_b: Optional[float] = None
    analysis_a: Optional[dict] = None
    analysis_b: Optional[dict] = None
    analysis_config: dict = None
    eq_correction_limit_db: float = 4.0
    envelope_smoothing_ms: float = 40.0
    envelope_max_history_ms: float = bounded_envelope_max_history_ms()
    instrument_type: str = "guitar"
    design_reference_profile_id: str = "vintage_humbucker"
    design_reference_profile_gain_db: float = 0.0
    calibration_mode: str = "auto"
    reference_input_level_dbu: float = DEFAULT_REFERENCE_INPUT_LEVEL_DBU
    amp_a_input_level_dbu: Optional[float] = None
    amp_b_input_level_dbu: Optional[float] = None
    amp_a_calibration_gain_db: float = 0.0
    amp_b_calibration_gain_db: float = 0.0
    amp_a_input_gain_db: float = 0.0
    amp_b_input_gain_db: float = 0.0
    calibration_applied: bool = False
    calibration_effective_mode: str = "raw"
    calibration_warning: Optional[str] = None
    amp_a_sha256: str = ""
    amp_b_sha256: str = ""
    design_di_file: Optional[str] = None
    mode: str = "character"
    teacher_semantics_version: int = CHARACTER_TEACHER_SEMANTICS_VERSION
    cab: Optional[CabDesign] = None

    # Shared post-combination output gain -- see hybrid.design.HybridDesign's
    # matching fields for the full rationale.
    output_gain_mode: str = "auto"
    manual_output_gain_db: float = 0.0

    def to_dict(self) -> dict: return asdict(self)
    def write_json(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8"); return path
    @staticmethod
    def read_json(path: str | Path) -> "CharacterBlendDesign":
        data = json.loads(Path(path).read_text(encoding="utf-8"));
        # Designs written before causal donor selection have no version.  They
        # retain their original (v1) behaviour rather than silently changing a
        # previously generated target's meaning.
        data.setdefault("teacher_semantics_version", 1)
        if isinstance(data.get("cab"), dict): data["cab"] = CabDesign(**data["cab"])
        return CharacterBlendDesign(**data)


@dataclass
class CharacterBlendResult:
    blend: np.ndarray
    envelope_db: np.ndarray
    drive_weight_b: np.ndarray
    analysis_a: AmpCharacterAnalysis
    analysis_b: AmpCharacterAnalysis


# Fixed reference sweep for the low-level response sanity check (see
# docs/blend-mode-fixes.md, "Phase 4"). Deliberately highest-to-lowest so a
# reader (and the manifest/UI) sees it in the same order a player backing off
# their instrument would experience it.
DEFAULT_LOW_LEVEL_SWEEP_DB: tuple[float, ...] = (0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0)
# A natural amp/compression response can lose several dB of output for a
# given dB of input reduction; a HARD gate loses vastly more. This margin
# (added on top of the input step itself) is a conservative amount of extra
# attenuation-per-step no ordinary amplifier behaviour should produce.
DEFAULT_LOW_LEVEL_COLLAPSE_MARGIN_DB: float = 25.0
_LOW_LEVEL_FLOOR_DBFS: float = -90.0


@dataclass(frozen=True)
class LowLevelResponseCheck:
    ok: bool
    levels_db: list
    output_rms_dbfs: list
    max_step_error_db: float
    dead_zone_detected: bool
    warning: Optional[str] = None

    def to_dict(self) -> dict: return asdict(self)


def _rms_dbfs(audio: np.ndarray, floor_dbfs: float = _LOW_LEVEL_FLOOR_DBFS) -> float:
    return float(_shared_rms_dbfs(audio, floor_dbfs))


def evaluate_low_level_response(
    build_pair_at_gain, design: "CharacterBlendDesign", levels_db: tuple = DEFAULT_LOW_LEVEL_SWEEP_DB,
    *, collapse_margin_db: float = DEFAULT_LOW_LEVEL_COLLAPSE_MARGIN_DB, floor_dbfs: float = _LOW_LEVEL_FLOOR_DBFS,
    analysis_a: Optional[AmpCharacterAnalysis] = None, analysis_b: Optional[AmpCharacterAnalysis] = None,
) -> LowLevelResponseCheck:
    """Render+blend a fixed reference DI across `levels_db` relative input
    gains and verify Character Blend remains a responsive amplifier rather
    than developing a hard low-level gate (docs/blend-mode-fixes.md, Phases
    4-5). This is the same `build_character_blend()` used by preview and
    training-bundle generation (Phase 7) -- only the sweep of input gains
    driving it is new.

    `build_pair_at_gain(gain_db)` renders a pair-like object (`.dry`/
    `.amp_a`/`.amp_b`/`.sample_rate`) for one relative gain; injected rather
    than computed here so this stays usable with synthetic pairs in tests and
    with real NAM renders in the training-bundle gate alike.

    A dead zone is flagged either by an output level pinned at `floor_dbfs`
    (digital silence) or by a step from one level to the next losing far more
    output than the corresponding input change alone would explain -- e.g.
    "input drops 6 dB, output drops 50 dB" -- rather than one arbitrary
    absolute RMS floor, since a quiet-but-still-descending amp is healthy and
    a collapsed one is not.
    """
    output_rms = [
        _rms_dbfs(build_character_blend(build_pair_at_gain(gain_db), design, analysis_a=analysis_a, analysis_b=analysis_b).blend, floor_dbfs)
        for gain_db in levels_db
    ]
    step_errors = [
        abs(output_rms[i] - output_rms[i - 1]) - abs(levels_db[i] - levels_db[i - 1])
        for i in range(1, len(levels_db))
    ]
    max_step_error_db = float(max(step_errors)) if step_errors else 0.0
    dead_zone_detected = max_step_error_db > collapse_margin_db or any(rms <= floor_dbfs + 1e-6 for rms in output_rms)
    warning = None
    if dead_zone_detected:
        sweep = ", ".join(f"{lv:g}dB->{rms:.1f}dBFS" for lv, rms in zip(levels_db, output_rms))
        warning = (
            f"Character Blend low-level response collapsed (max step error {max_step_error_db:.1f} dB "
            f"exceeds the {collapse_margin_db:.1f} dB margin): {sweep}"
        )
    return LowLevelResponseCheck(not dead_zone_detected, list(levels_db), output_rms, max_step_error_db, dead_zone_detected, warning)


def _analysis_from_design(data: Optional[dict]) -> Optional[AmpCharacterAnalysis]:
    return AmpCharacterAnalysis.from_dict(data) if data else None


def _interp(levels: np.ndarray, values: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return np.interp(envelope, levels, values, left=values[0], right=values[-1])


def _adjacent_level_weights(levels: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Per-sample weights (n_levels, n_samples) over the analysis grid.

    Unlike an all-level triangular basis, every sample's weight comes from
    AT MOST its two neighbouring analysis levels, and clamps -- rather than
    zeroing out -- once the envelope moves outside the grid. This guarantees
    ``weights.sum(axis=0) == 1`` everywhere, including far below the lowest
    measured level, where a triangular kernel's support vanishes and would
    otherwise silently zero the teacher output (the low-level collapse bug
    this replaces).
    """
    n_levels = len(levels)
    lower_idx = np.clip(np.searchsorted(levels, envelope, side="right") - 1, 0, n_levels - 2)
    upper_idx = lower_idx + 1
    lo, hi = levels[lower_idx], levels[upper_idx]
    frac = np.clip((envelope - lo) / np.maximum(hi - lo, _EPS), 0.0, 1.0)
    below, above = envelope <= levels[0], envelope >= levels[-1]
    frac = np.where(below, 0.0, np.where(above, 1.0, frac))
    weights = np.zeros((n_levels, len(envelope)), dtype=np.float64)
    sample_idx = np.arange(len(envelope))
    weights[lower_idx, sample_idx] += 1.0 - frac
    weights[upper_idx, sample_idx] += frac
    assert np.allclose(weights.sum(axis=0), 1.0), "Character Blend interpolation weights must sum to 1"
    return weights


def _drive_curve(design: CharacterBlendDesign, envelope: np.ndarray, levels: np.ndarray) -> np.ndarray:
    if None not in (design.drive_low_mix_b, design.drive_mid_mix_b, design.drive_high_mix_b):
        points = np.array([_clamp(design.drive_low_mix_b), _clamp(design.drive_mid_mix_b), _clamp(design.drive_high_mix_b)])
        anchors = np.array([levels[0], levels[len(levels)//2], levels[-1]])
        return _interp(anchors, points, envelope)
    return np.full(len(envelope), _clamp(design.drive_mix_b), dtype=np.float64)


def _causal_donor_weight(drive_b: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return the causal Amp-B donor weight for the version-2 teacher.

    Below 50% requests Amp A; 50% and above requests Amp B.  A constant B
    request begins at B (there is no artificial A-to-B startup fade).  At a
    later change, the ramp starts at the current sample and from the current
    weight.  A reversal during that ramp consequently remains continuous and
    cannot use control values from the future.
    """
    requested = np.asarray(drive_b, dtype=np.float64) >= 0.5
    n = len(requested)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    ramp_samples = max(1, int(round(sample_rate * DONOR_TRANSITION_MS / 1000.0)))
    weights = np.empty(n, dtype=np.float64)
    weight = 1.0 if requested[0] else 0.0
    target = weight
    start_weight = weight
    ramp_remaining = 0
    for i, want_b in enumerate(requested):
        new_target = 1.0 if want_b else 0.0
        if new_target != target:
            # `weight` is the value actually emitted for the preceding sample;
            # starting here gives a causal, bounded transition.
            start_weight, target, ramp_remaining = weight, new_target, ramp_samples
        if ramp_remaining:
            step = ramp_samples - ramp_remaining + 1
            weight = start_weight + (target - start_weight) * (step / ramp_samples)
            ramp_remaining -= 1
        else:
            weight = target
        weights[i] = weight
    return np.clip(weights, 0.0, 1.0)


def _select_donor(a: np.ndarray, b: np.ndarray, drive_b: np.ndarray, sample_rate: int, *, semantics_version: int = CHARACTER_TEACHER_SEMANTICS_VERSION) -> np.ndarray:
    """Select the thresholded Drive donor with versioned transition semantics."""
    n = len(drive_b)
    if semantics_version >= CHARACTER_TEACHER_SEMANTICS_VERSION:
        weight_b = _causal_donor_weight(drive_b, sample_rate)
        return (a[:n] * (1.0 - weight_b) + b[:n] * weight_b).astype(np.float64)

    # Legacy bundles retain their historical centred transition.  Do not use
    # this path for newly frozen designs.
    donor = np.where(drive_b >= 0.5, b, a).astype(np.float64)
    changes = np.flatnonzero(np.diff((drive_b >= .5).astype(np.int8))) + 1
    fade = max(1, int(sample_rate * .01))
    for point in changes:
        start, end = max(0, point - fade), min(n, point + fade)
        w = np.linspace(0, 1, end - start)
        if drive_b[point] >= .5: donor[start:end] = a[start:end] * (1 - w) + b[start:end] * w
        else: donor[start:end] = b[start:end] * (1 - w) + a[start:end] * w
    return donor


def _smooth(value: np.ndarray, sample_rate: int, ms: float) -> np.ndarray:
    width = max(1, int(sample_rate * ms / 1000.0))
    if width == 1: return value
    return np.convolve(np.pad(value, (width - 1, 0), mode="edge"), np.ones(width) / width, mode="valid")


def _minimum_phase_correction(freqs: np.ndarray, correction_db: np.ndarray, sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2.0
    f = np.concatenate([[0.0], np.clip(freqs, 1.0, nyquist - 1.0), [nyquist]]) / nyquist
    mag = 10.0 ** (np.concatenate([[correction_db[0]], correction_db, [correction_db[-1]]]) / 20.0)
    linear = firwin2(129, f, mag)
    return minimum_phase(linear, half=True)


def build_character_blend(pair, design: CharacterBlendDesign, *, analysis_a: AmpCharacterAnalysis | None = None, analysis_b: AmpCharacterAnalysis | None = None) -> CharacterBlendResult:
    """Build the exact shared preview/training teacher from a rendered pair.

    The donor is selected once per sample from its dry envelope, then a
    level-derived gain curve and broad minimum-phase correction are applied.
    Raw A+B waveform summing is intentionally absent from this function.
    """
    n = min(len(pair.dry), len(pair.amp_a), len(pair.amp_b))
    dry, a, b = pair.dry[:n], pair.amp_a[:n], pair.amp_b[:n]
    envelope = bounded_causal_envelope_db(dry, pair.sample_rate)
    config = CharacterAnalysisConfig(**design.analysis_config) if design.analysis_config else CharacterAnalysisConfig()
    analysis_a = analysis_a or _analysis_from_design(design.analysis_a) or analyse_rendered_audio(dry, a, pair.sample_rate, config)
    analysis_b = analysis_b or _analysis_from_design(design.analysis_b) or analyse_rendered_audio(dry, b, pair.sample_rate, config)
    levels = np.array([x.input_gain_db for x in analysis_a.levels])
    drive_b = _smooth(_drive_curve(design, envelope, levels), pair.sample_rate, design.envelope_smoothing_ms)
    donor_weight_b = _causal_donor_weight(drive_b, pair.sample_rate) if design.teacher_semantics_version >= CHARACTER_TEACHER_SEMANTICS_VERSION else (drive_b >= 0.5).astype(np.float64)
    donor = _select_donor(a, b, drive_b, pair.sample_rate, semantics_version=design.teacher_semantics_version)
    gain_a = np.array([x.compression_gain_db for x in analysis_a.levels]); gain_b = np.array([x.compression_gain_db for x in analysis_b.levels])
    target_gain = gain_a * (1 - _clamp(design.feel_mix_b)) + gain_b * _clamp(design.feel_mix_b)
    # Apply the same transition weight to compensation: otherwise the donor
    # waveform would be smooth while its gain correction still jumped.
    donor_gain = _interp(levels, gain_a, envelope) * (1.0 - donor_weight_b) + _interp(levels, gain_b, envelope) * donor_weight_b
    gain = 10 ** (_smooth(_interp(levels, target_gain, envelope) - donor_gain, pair.sample_rate, design.envelope_smoothing_ms) / 20.0)
    corrected = donor * gain
    freqs = np.array(analysis_a.frequencies_hz)
    # Filter each measured-level correction and interpolate filtered donor
    # paths. This avoids zipper noise without changing a filter per sample.
    filtered = []
    for i, level in enumerate(levels):
        eq_a = np.array(analysis_a.levels[i].spectrum_db); eq_b = np.array(analysis_b.levels[i].spectrum_db)
        target = eq_a * (1 - _clamp(design.tone_mix_b)) + eq_b * _clamp(design.tone_mix_b)
        donor_eq = eq_b if _drive_curve(design, np.array([level]), levels)[0] >= .5 else eq_a
        correction = np.clip(target - donor_eq, -abs(design.eq_correction_limit_db), abs(design.eq_correction_limit_db))
        filtered.append(fftconvolve(corrected, _minimum_phase_correction(freqs, correction, pair.sample_rate), mode="full")[:n])
    weights = _adjacent_level_weights(levels, envelope)
    output = np.sum(np.vstack(filtered) * weights, axis=0).astype(np.float32)
    return CharacterBlendResult(output, envelope, drive_b, analysis_a, analysis_b)


def freeze_character_design(pair, result: CharacterBlendResult, amp_a_path: str, amp_b_path: str, **kwargs) -> CharacterBlendDesign:
    # JSON-normalise the nested analysis now, so the in-memory frozen design
    # has exactly the same representation that is later signed/written/read.
    frozen_a = json.loads(json.dumps(result.analysis_a.to_dict()))
    frozen_b = json.loads(json.dumps(result.analysis_b.to_dict()))
    config = json.loads(json.dumps(asdict(CharacterAnalysisConfig(
        levels_db=tuple(x.input_gain_db for x in result.analysis_a.levels), frequencies_hz=result.analysis_a.frequencies_hz,
    ))))
    return CharacterBlendDesign(amp_a_path=str(amp_a_path), amp_b_path=str(amp_b_path), analysis_a=frozen_a, analysis_b=frozen_b, analysis_config=config, instrument_type=pair.instrument_type, design_reference_profile_id=pair.input_profile_id, design_reference_profile_gain_db=pair.input_profile_gain_db, calibration_mode=pair.calibration_mode, reference_input_level_dbu=pair.reference_input_level_dbu, amp_a_input_level_dbu=pair.amp_a_model_input_level_dbu, amp_b_input_level_dbu=pair.amp_b_model_input_level_dbu, amp_a_calibration_gain_db=pair.amp_a_calibration_gain_db, amp_b_calibration_gain_db=pair.amp_b_calibration_gain_db, amp_a_input_gain_db=pair.amp_a_input_gain_db, amp_b_input_gain_db=pair.amp_b_input_gain_db, calibration_applied=pair.calibration_applied, calibration_effective_mode=pair.calibration_mode if pair.calibration_applied else "raw", calibration_warning=pair.calibration_warning, **kwargs)

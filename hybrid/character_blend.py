"""Character Blend teacher construction: one donor path plus broad correction."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve, firwin2, minimum_phase

from .cab_ir import CabDesign
from .calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU
from .character_analysis import AmpCharacterAnalysis, CharacterAnalysisConfig, analyse_rendered_audio
from .envelope import bounded_causal_envelope_db, bounded_envelope_max_history_ms

_EPS = 1e-10


def _clamp(value: float) -> float: return max(0.0, min(1.0, float(value)))


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
    calibration_applied: bool = False
    calibration_effective_mode: str = "raw"
    calibration_warning: Optional[str] = None
    amp_a_sha256: str = ""
    amp_b_sha256: str = ""
    design_di_file: Optional[str] = None
    mode: str = "character"
    cab: Optional[CabDesign] = None

    def to_dict(self) -> dict: return asdict(self)
    def write_json(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8"); return path
    @staticmethod
    def read_json(path: str | Path) -> "CharacterBlendDesign":
        data = json.loads(Path(path).read_text(encoding="utf-8"));
        if isinstance(data.get("cab"), dict): data["cab"] = CabDesign(**data["cab"])
        return CharacterBlendDesign(**data)


@dataclass
class CharacterBlendResult:
    blend: np.ndarray
    envelope_db: np.ndarray
    drive_weight_b: np.ndarray
    analysis_a: AmpCharacterAnalysis
    analysis_b: AmpCharacterAnalysis


def _analysis_from_design(data: Optional[dict]) -> Optional[AmpCharacterAnalysis]:
    return AmpCharacterAnalysis.from_dict(data) if data else None


def _interp(levels: np.ndarray, values: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return np.interp(envelope, levels, values, left=values[0], right=values[-1])


def _drive_curve(design: CharacterBlendDesign, envelope: np.ndarray, levels: np.ndarray) -> np.ndarray:
    if None not in (design.drive_low_mix_b, design.drive_mid_mix_b, design.drive_high_mix_b):
        points = np.array([_clamp(design.drive_low_mix_b), _clamp(design.drive_mid_mix_b), _clamp(design.drive_high_mix_b)])
        anchors = np.array([levels[0], levels[len(levels)//2], levels[-1]])
        return _interp(anchors, points, envelope)
    return np.full(len(envelope), _clamp(design.drive_mix_b), dtype=np.float64)


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
    # A block-continuous donor transition. It only occurs where the identity
    # changes; steady regions are exactly one source waveform.
    donor = np.where(drive_b >= 0.5, b, a).astype(np.float64)
    changes = np.flatnonzero(np.diff((drive_b >= .5).astype(np.int8))) + 1
    fade = max(1, int(pair.sample_rate * .01))
    for point in changes:
        start, end = max(0, point - fade), min(n, point + fade)
        w = np.linspace(0, 1, end - start)
        if drive_b[point] >= .5: donor[start:end] = a[start:end] * (1 - w) + b[start:end] * w
        else: donor[start:end] = b[start:end] * (1 - w) + a[start:end] * w
    gain_a = np.array([x.compression_gain_db for x in analysis_a.levels]); gain_b = np.array([x.compression_gain_db for x in analysis_b.levels])
    target_gain = gain_a * (1 - _clamp(design.feel_mix_b)) + gain_b * _clamp(design.feel_mix_b)
    donor_gain = np.where(drive_b >= .5, _interp(levels, gain_b, envelope), _interp(levels, gain_a, envelope))
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
    weights = np.vstack([np.maximum(0.0, 1.0 - np.abs(envelope - level) / max(1.0, np.diff(levels).mean())) for level in levels])
    weights /= np.maximum(weights.sum(axis=0), _EPS)
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
    return CharacterBlendDesign(amp_a_path=str(amp_a_path), amp_b_path=str(amp_b_path), analysis_a=frozen_a, analysis_b=frozen_b, analysis_config=config, instrument_type=pair.instrument_type, design_reference_profile_id=pair.input_profile_id, design_reference_profile_gain_db=pair.input_profile_gain_db, calibration_mode=pair.calibration_mode, reference_input_level_dbu=pair.reference_input_level_dbu, amp_a_input_level_dbu=pair.amp_a_model_input_level_dbu, amp_b_input_level_dbu=pair.amp_b_model_input_level_dbu, amp_a_calibration_gain_db=pair.amp_a_calibration_gain_db, amp_b_calibration_gain_db=pair.amp_b_calibration_gain_db, calibration_applied=pair.calibration_applied, calibration_effective_mode=pair.calibration_mode if pair.calibration_applied else "raw", calibration_warning=pair.calibration_warning, **kwargs)

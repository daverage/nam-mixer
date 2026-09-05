"""Orchestration: the two operations the UI actually needs, kept separate
because they have very different costs.

`render_pair` is EXPENSIVE (runs NAM inference twice) and only needs to be
called again when Amp A, Amp B, the DI, or the INPUT PROFILE/CALIBRATION
changes (an input profile changes the actual signal fed to both NAMs, so it
is a render-stage concern, not a blend-stage one -- see
docs/INPUT_PROFILE_RESEARCH.md). `build_hybrid` is CHEAP (pure numpy) and is
what should run on every crossover/transition/trim slider move -- see the
module docstrings of blend.py/level_match.py/align.py for the individual
steps this composes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .align import align_to_reference
from .blend import CrossoverConfig, blend
from .calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU, resolve_calibration
from .envelope import DEFAULT_BOUNDED_ENVELOPE_CONFIG, BoundedEnvelopeConfig, bounded_causal_envelope_db
from .input_profiles import db_to_amplitude
from .level_match import LevelMatchResult, compute_crossover_trim
from .nam_loader import NamModel
from .render import render


def _peak_dbfs(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return float("-inf")
    peak = float(np.max(np.abs(audio)))
    return 20.0 * np.log10(peak) if peak > 0 else float("-inf")


@dataclass
class RenderedPair:
    dry: np.ndarray                 # original source DI, unmodified
    profiled_dry: np.ndarray        # dry * input_profile_gain (before per-model calibration)
    amp_a: np.ndarray
    amp_b: np.ndarray
    envelope_db: np.ndarray         # crossover envelope, derived from profiled_dry
    source_envelope_db: np.ndarray  # envelope of the raw, un-profiled dry -- used by coverage analysis
    sample_rate: int

    instrument_type: str = "guitar"
    input_profile_id: str = "vintage_humbucker"
    input_profile_gain_db: float = 0.0

    calibration_mode: str = "raw"
    reference_input_level_dbu: float = DEFAULT_REFERENCE_INPUT_LEVEL_DBU
    calibration_applied: bool = False

    amp_a_model_input_level_dbu: Optional[float] = None
    amp_b_model_input_level_dbu: Optional[float] = None
    amp_a_calibration_gain_db: float = 0.0
    amp_b_calibration_gain_db: float = 0.0

    input_peak_dbfs: float = float("-inf")
    calibration_warning: Optional[str] = None


def render_pair(
    amp_a: NamModel,
    amp_b: NamModel,
    dry: np.ndarray,
    sample_rate: int,
    *,
    instrument_type: str = "guitar",
    input_profile_id: str = "vintage_humbucker",
    input_profile_gain_db: float = 0.0,
    calibration_mode: str = "auto",
    reference_input_level_dbu: float = DEFAULT_REFERENCE_INPUT_LEVEL_DBU,
    envelope_config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG,
) -> RenderedPair:
    """Render `dry` through both amp models. The expensive step -- call again
    whenever amp_a, amp_b, dry, the input profile, or calibration settings
    change; NOT on crossover/transition/trim slider moves.

    `input_profile_gain_db` is applied to `dry` BEFORE both NAM inference and
    crossover-envelope detection, so it represents a real change in how hard
    the (virtual) instrument is driving the signal chain -- unlike the
    deprecated test-only `dry_gain_db` on `build_hybrid`, which only shifted
    the envelope used for blending and never touched the actual audio.

    If `calibration_mode="auto"` and both models report a calibrated
    `input_level_dbu`, an additional PER-MODEL calibration gain (the official
    NAM plugin's `reference_input_level_dbu - model_input_level_dbu`
    formula) is applied to what each individual model actually receives, so
    two differently-calibrated `.nam` captures see the same virtual physical
    input level. The crossover envelope is always derived from the
    profile-adjusted signal BEFORE this per-model calibration split, so the
    crossover stays linked to one common virtual guitar level rather than
    either source model's own recording calibration.
    """
    dry = np.asarray(dry, dtype=np.float32)
    profiled_dry = (dry * db_to_amplitude(input_profile_gain_db)).astype(np.float32)

    calib = resolve_calibration(
        calibration_mode, reference_input_level_dbu, amp_a.input_level_dbu, amp_b.input_level_dbu
    )

    amp_a_input = (profiled_dry * db_to_amplitude(calib.amp_a_gain_db)).astype(np.float32)
    amp_b_input = (profiled_dry * db_to_amplitude(calib.amp_b_gain_db)).astype(np.float32)

    amp_a_render = render(amp_a, amp_a_input, sample_rate)
    amp_b_render = render(amp_b, amp_b_input, sample_rate)

    envelope_db = bounded_causal_envelope_db(profiled_dry, sample_rate, envelope_config)
    source_envelope_db = bounded_causal_envelope_db(dry, sample_rate, envelope_config)

    return RenderedPair(
        dry=dry,
        profiled_dry=profiled_dry,
        amp_a=amp_a_render,
        amp_b=amp_b_render,
        envelope_db=envelope_db,
        source_envelope_db=source_envelope_db,
        sample_rate=sample_rate,
        instrument_type=instrument_type,
        input_profile_id=input_profile_id,
        input_profile_gain_db=input_profile_gain_db,
        calibration_mode=calib.mode,
        reference_input_level_dbu=reference_input_level_dbu,
        calibration_applied=calib.applied,
        amp_a_model_input_level_dbu=calib.amp_a_model_input_level_dbu,
        amp_b_model_input_level_dbu=calib.amp_b_model_input_level_dbu,
        amp_a_calibration_gain_db=calib.amp_a_gain_db,
        amp_b_calibration_gain_db=calib.amp_b_gain_db,
        input_peak_dbfs=_peak_dbfs(profiled_dry),
        calibration_warning=calib.warning,
    )


@dataclass
class HybridResult:
    hybrid: np.ndarray
    blend_curve: np.ndarray
    envelope_db: np.ndarray
    auto_trim_db: float
    manual_trim_db: float
    effective_b_trim_db: float
    alignment_offset_samples: int
    level_match: LevelMatchResult | None


def build_hybrid(
    pair: RenderedPair,
    crossover_dbfs: float,
    transition_width_db: float,
    auto_level: bool = True,
    manual_b_trim_db: float = 0.0,
    align_enabled: bool = False,
    dry_gain_db: float = 0.0,
) -> HybridResult:
    """Blend an already-rendered amp pair. Cheap -- safe to call on every
    crossover/transition/trim slider move without re-running NAM inference.

    `manual_b_trim_db` is always applied, on top of the auto-match trim when
    `auto_level` is on -- auto-level gives a safe starting point, the manual
    trim is the user's tweak from there, and the two combine rather than one
    replacing the other.

    `dry_gain_db` is DEPRECATED and TEST-ONLY (kept for regression tests
    exercising the blend/threshold logic in isolation) -- it shifts only the
    envelope used for blending, by a constant (dB(x*g) = dB(x) + 20*log10(g),
    an exact O(n) shift), without touching the actual audio. The real,
    production input-level control is `input_profile_gain_db` on
    `render_pair`, which changes what both NAMs actually receive. Do not wire
    this parameter to a user-facing control.
    """
    envelope_db = pair.envelope_db + dry_gain_db

    amp_b_render, offset = align_to_reference(pair.amp_a, pair.amp_b, enabled=align_enabled)

    level_match_result: LevelMatchResult | None = None
    auto_trim_db = 0.0
    if auto_level:
        level_match_result = compute_crossover_trim(
            pair.envelope_db, pair.amp_a, amp_b_render, crossover_dbfs, transition_width_db
        )
        auto_trim_db = level_match_result.suggested_b_trim_db

    effective_b_trim_db = auto_trim_db + manual_b_trim_db

    config = CrossoverConfig(
        crossover_dbfs=crossover_dbfs,
        transition_width_db=transition_width_db,
        amp_b_trim_db=effective_b_trim_db,
    )
    hybrid_audio, t_curve = blend(envelope_db, pair.amp_a, amp_b_render, config)

    return HybridResult(
        hybrid=hybrid_audio,
        blend_curve=t_curve,
        envelope_db=envelope_db,
        auto_trim_db=auto_trim_db,
        manual_trim_db=manual_b_trim_db,
        effective_b_trim_db=effective_b_trim_db,
        alignment_offset_samples=offset,
        level_match=level_match_result,
    )

"""Orchestration: the two operations the UI actually needs, kept separate
because they have very different costs.

`render_pair` is EXPENSIVE (runs NAM inference twice) and only needs to be
called again when Amp A, Amp B, or the DI changes. `build_hybrid` is CHEAP
(pure numpy) and is what should run on every crossover/transition/trim slider
move -- see the module docstrings of blend.py/level_match.py/align.py for the
individual steps this composes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import align_to_reference
from .blend import CrossoverConfig, blend
from .envelope import rms_envelope_db
from .level_match import LevelMatchResult, compute_crossover_trim
from .nam_loader import NamModel
from .render import render


@dataclass
class RenderedPair:
    dry: np.ndarray
    amp_a: np.ndarray
    amp_b: np.ndarray
    envelope_db: np.ndarray
    sample_rate: int


def render_pair(
    amp_a: NamModel,
    amp_b: NamModel,
    dry: np.ndarray,
    sample_rate: int,
) -> RenderedPair:
    """Render `dry` through both amp models. The expensive step -- call once
    per (amp_a, amp_b, dry) choice, then reuse the result across slider moves.

    Also computes the crossover envelope here (depends only on `dry` and
    `sample_rate`, not on crossover/transition/trim) so `build_hybrid` never
    has to recompute it on every slider move.
    """
    dry = np.asarray(dry, dtype=np.float32)
    amp_a_render = render(amp_a, dry, sample_rate)
    amp_b_render = render(amp_b, dry, sample_rate)
    envelope_db = rms_envelope_db(dry, sample_rate)
    return RenderedPair(
        dry=dry,
        amp_a=amp_a_render,
        amp_b=amp_b_render,
        envelope_db=envelope_db,
        sample_rate=sample_rate,
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

    `dry_gain_db` is a TEST-ONLY control: it shifts the envelope that drives
    the crossfade by a constant (dB(x * g) = dB(x) + 20*log10(g), so this is
    an exact, O(n) shift -- no need to re-run rms_envelope_db) to let you push
    the crossover trigger up/down without needing a louder/quieter DI take.
    It does NOT re-render either amp at a hotter input, so it's for
    exercising the blend/threshold logic, not for previewing how the amps
    would actually respond to a different input level.

    Deliberately NOT used for auto-level-match's region selection, only for
    the blend weight: `compute_crossover_trim` always measures loudness
    against `pair.envelope_db` (the real, un-shifted envelope), so the
    suggested trim reflects the amps' actual behavior at the real
    `crossover_dbfs` input level and stays stable while `dry_gain_db` is
    swept -- using the shifted envelope there would instead measure loudness
    from whatever coincidentally-quiet-or-loud stretch of the real recording
    the shift maps onto, which has nothing to do with how loud the amps
    genuinely are at that input level.
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

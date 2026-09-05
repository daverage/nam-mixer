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
    sample_rate: int


def render_pair(
    amp_a: NamModel,
    amp_b: NamModel,
    dry: np.ndarray,
    sample_rate: int,
) -> RenderedPair:
    """Render `dry` through both amp models. The expensive step -- call once
    per (amp_a, amp_b, dry) choice, then reuse the result across slider moves.
    """
    dry = np.asarray(dry, dtype=np.float32)
    amp_a_render = render(amp_a, dry, sample_rate)
    amp_b_render = render(amp_b, dry, sample_rate)
    return RenderedPair(dry=dry, amp_a=amp_a_render, amp_b=amp_b_render, sample_rate=sample_rate)


@dataclass
class HybridResult:
    hybrid: np.ndarray
    blend_curve: np.ndarray
    auto_trim_db: float
    level_match: LevelMatchResult | None


def build_hybrid(
    pair: RenderedPair,
    crossover_dbfs: float,
    transition_width_db: float,
    auto_level: bool = True,
    manual_b_trim_db: float = 0.0,
    align_enabled: bool = False,
) -> HybridResult:
    """Blend an already-rendered amp pair. Cheap -- safe to call on every
    crossover/transition/trim slider move without re-running NAM inference.
    """
    envelope_db = rms_envelope_db(pair.dry, pair.sample_rate)

    amp_b_render, _offset = align_to_reference(pair.amp_a, pair.amp_b, enabled=align_enabled)

    level_match_result: LevelMatchResult | None = None
    if auto_level:
        level_match_result = compute_crossover_trim(
            envelope_db, pair.amp_a, amp_b_render, crossover_dbfs, transition_width_db
        )
        b_trim_db = level_match_result.suggested_b_trim_db
    else:
        b_trim_db = manual_b_trim_db

    config = CrossoverConfig(
        crossover_dbfs=crossover_dbfs,
        transition_width_db=transition_width_db,
        amp_b_trim_db=b_trim_db,
    )
    hybrid_audio, t_curve = blend(envelope_db, pair.amp_a, amp_b_render, config)

    return HybridResult(
        hybrid=hybrid_audio,
        blend_curve=t_curve,
        auto_trim_db=b_trim_db,
        level_match=level_match_result,
    )

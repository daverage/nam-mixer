"""N-way ordered level-driven blend (Continuous Gain v3): generalises `hybrid.blend` from two amps to an ordered
chain of captures (e.g. one amp at Gain 1..10). Only ADJACENT captures ever mix. Reuses `blend.smoothstep_curve`.

Capture k owns input level `levels_db[k]` (ascending). The rendered audio for capture k is expected to have been
produced from the input rescaled by `reference_db - levels_db[k]`, so that where the (causal) input envelope sits at
`levels_db[k]`, capture k is being driven at the common `reference_db` playing level. A model trained on the result
therefore sounds like capture k when its input sits at that level, i.e. an ordinary input-gain control moves it along
the chain (designated gain for capture k = levels_db[k] - reference_db).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid.blend import smoothstep_curve


@dataclass(frozen=True)
class GainChain:
    labels: tuple[float, ...]
    levels_db: tuple[float, ...]
    reference_db: float

    def __post_init__(self):
        if len(self.labels) != len(self.levels_db) or len(self.labels) < 2:
            raise ValueError("chain needs >= 2 captures with one level each")
        if any(b <= a for a, b in zip(self.levels_db, self.levels_db[1:])):
            raise ValueError("levels_db must be strictly ascending")

    def input_scale_db(self, k: int) -> float:
        return self.reference_db - self.levels_db[k]

    def designated_gain_db(self, k: int) -> float:
        return self.levels_db[k] - self.reference_db


def chain_weights(envelope_db: np.ndarray, levels_db: tuple[float, ...]) -> np.ndarray:
    """(N, T) weights, columns sum to 1, nonzero only for the two captures bracketing the envelope."""
    env = np.asarray(envelope_db, dtype=np.float64)
    lv = np.asarray(levels_db, dtype=np.float64)
    n = len(lv)
    w = np.zeros((n, len(env)))
    pos = np.clip(env, lv[0], lv[-1])
    hi = np.clip(np.searchsorted(lv, pos, side="right"), 1, n - 1)
    lo = hi - 1
    t = smoothstep_curve(np.clip((pos - lv[lo]) / (lv[hi] - lv[lo]), 0.0, 1.0))
    idx = np.arange(len(env))
    w[lo, idx] = 1.0 - t
    w[hi, idx] += t
    return w


def multi_blend(envelope_db: np.ndarray, renders: list[np.ndarray], chain: GainChain) -> np.ndarray:
    n = min(len(envelope_db), *(len(r) for r in renders))
    w = chain_weights(envelope_db[:n], chain.levels_db)
    return sum(w[k] * renders[k][:n] for k in range(len(renders)))

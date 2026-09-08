"""Small, dependency-light audio measurements shared by blend modes."""
from __future__ import annotations

import numpy as np


def rms_dbfs(audio: np.ndarray, floor_dbfs: float | None = None) -> float:
    """Return RMS level in dBFS, with an optional lower reporting floor."""
    if len(audio) == 0:
        return -np.inf if floor_dbfs is None else floor_dbfs
    rms = np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))
    level = 20.0 * np.log10(max(rms, 1e-10))
    return max(level, floor_dbfs) if floor_dbfs is not None else level

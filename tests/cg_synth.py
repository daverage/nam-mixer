"""Synthetic amps/DIs for the Continuous Gain tests: no native renderer, no real captures."""
from __future__ import annotations

import numpy as np

SR = 48000


def synth_di(name: str = "x", seconds: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(int(seconds * 3)):
        f0 = 110 * 2 ** (rng.integers(0, 12) / 12)
        on = int(k / 3 * SR)
        seg = np.arange(n - on) / SR
        x[on:] += 0.3 * np.exp(-seg * 6) * (np.sin(2 * np.pi * f0 * seg) + 0.4 * np.sin(2 * np.pi * 2 * f0 * seg))
    return (x / max(np.max(np.abs(x)), 1e-9) * 0.5).astype(np.float32)


def amp_render(gain: float, delay: int = 0, noise_db: float | None = None):
    """Memoryless saturating amp: more `gain` -> more drive and level; optional constant delay / noise floor."""
    drive = 0.6 * 10 ** (gain / 8.0)

    def render(audio: np.ndarray) -> np.ndarray:
        y = np.tanh(drive * audio.astype(np.float64)) / max(np.tanh(drive), 1e-6) * 0.5
        if noise_db is not None:
            y = y + np.random.default_rng(1).standard_normal(len(y)) * 10 ** (noise_db / 20)
        if delay:
            y = np.concatenate([np.zeros(delay), y[:-delay]])
        return y.astype(np.float32)

    return render

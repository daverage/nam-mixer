"""A single, deterministic Continuous Gain training excitation: guitar-like playing whose LEVEL is swept across the whole playback range.

The frozen FC recipe covers the level range by repeating three real guitar DIs at eight fixed level offsets (24 clips). The target of a
Continuous Gain model is blended by INPUT LEVEL, so what the training input must do is visit every level in the chain, with realistic
note shapes, decays and spectra at each of them. `level_swept_excitation` produces that as ONE file: an endless-feeling pluck / chord /
palm-mute / sustained-note sequence (additive synthesis with a pick-noise transient; no external audio) under a slow triangular gain sweep
between `lo_db` and `hi_db`, plus per-note velocity variation. It is calibrated so that 0 dB matches the level of the bundled guitar DIs.
Everything is seeded: the same arguments always give the same samples.
"""
from __future__ import annotations

import numpy as np

SR = 48000
NOTE_HZ = np.array([82.4, 110.0, 146.8, 196.0, 246.9, 329.6, 392.0, 493.9, 587.3, 659.3])       # E2 ... E5 area, open-string-ish


def _note(rng: np.random.Generator, f0: float, dur: float, kind: str) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    harmonics = np.arange(1, 15)
    bright = {"pluck": 1.0, "mute": 0.35, "sustain": 0.7}[kind]
    amps = (1.0 / harmonics ** 1.7) * np.exp(-((harmonics - 1) / (2.5 + 4 * bright)) ** 1.5)          # spectral tilt: less top end for muted/soft playing
    decay = {"pluck": rng.uniform(1.2, 3.0), "mute": rng.uniform(9.0, 18.0), "sustain": rng.uniform(0.25, 0.7)}[kind]
    vib = 0.0 if kind != "sustain" else rng.uniform(0.0, 0.006)
    phase_mod = vib * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t) if vib else 0.0
    y = np.zeros(n)
    for h, a in zip(harmonics, amps):
        f = f0 * h * (1 + 4e-5 * h * h)                                                    # slight string stiffness
        if f > 12000:
            break
        y += a * np.exp(-t * decay * (0.6 + 0.12 * h)) * np.sin(2 * np.pi * f * t * (1 + phase_mod) + rng.uniform(0, 2 * np.pi))
    pick = int(0.012 * SR)                                                                   # pick-noise transient
    burst = np.convolve(rng.standard_normal(pick + 8), np.ones(9) / 9, mode="valid")             # smoothed: a dull 'thunk', not white hiss
    y[:pick] += burst[:pick] * np.hanning(2 * pick)[pick:] * 0.5 * bright
    y *= np.minimum(1.0, t / 0.003 + 0.05)                                                   # tiny attack ramp (no click)
    y *= np.minimum(1.0, (dur - t) / 0.05)                                                   # and release
    return y


def level_swept_excitation(seconds: float = 480.0, *, seed: int = 7, lo_db: float = -32.0, hi_db: float = 20.0,
                           period_s: float = 80.0, reference_rms: float = 0.04) -> np.ndarray:
    """Mono float32 at 48 kHz. `reference_rms` is the active-material RMS at 0 dB (use the guitar DIs' RMS); the gain then sweeps
    triangularly lo_db -> hi_db -> lo_db every `period_s`, with +-3 dB (1 sigma) per-note velocity variation on top."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros(n)
    t = 0.0
    while t < seconds:
        chord = rng.random() < 0.3
        kind = rng.choice(["pluck", "mute", "sustain"], p=[0.55, 0.25, 0.20])
        dur = {"pluck": rng.uniform(0.5, 2.5), "mute": rng.uniform(0.12, 0.4), "sustain": rng.uniform(0.8, 2.5)}[kind]
        voices = rng.choice(NOTE_HZ, size=rng.integers(2, 4) if chord else 1, replace=False)
        onset = int(t * SR)
        vel_db = rng.normal(0.0, 3.0)
        for f0 in voices:
            y = _note(rng, float(f0) * 2 ** rng.choice([0, 0, 1 / 12, 2 / 12, 3 / 12]), dur, str(kind)) * 10 ** (vel_db / 20)
            end = min(n, onset + len(y))
            x[onset:end] += y[: end - onset] * (0.8 if chord else 1.0)
        t += rng.choice([0.18, 0.25, 0.33, 0.5, 0.75]) * (1.0 if kind != "sustain" else 1.8)
    tt = np.arange(n) / SR
    tri = np.abs(((tt / period_s) % 1.0) * 2 - 1)                                             # 0..1..0 triangle
    sweep_db = lo_db + (hi_db - lo_db) * (1 - tri)
    # calibrate on the un-swept material so that 0 dB really is the reference level, then apply the sweep
    frame = SR // 10
    fr = x[: n // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    active = rms > 0.1 * np.max(rms)
    x *= reference_rms / max(float(np.sqrt(np.mean(rms[active] ** 2))), 1e-9)
    x *= 10 ** (sweep_db / 20)
    return x.astype(np.float32)


def active_rms(audio: np.ndarray) -> float:
    """RMS over the active (above 10% of peak frame RMS) 100 ms frames -- how the calibration and the DIs are compared."""
    frame = SR // 10
    fr = audio[: len(audio) // frame * frame].reshape(-1, frame).astype(np.float64)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    return float(np.sqrt(np.mean(rms[rms > 0.1 * np.max(rms)] ** 2)))

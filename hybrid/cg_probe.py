"""Continuous Gain capture probe bank (Phase 4A/4B measurement input).

Extracted unchanged in behaviour from `scripts/p4_common.py` / `scripts/cg_report.py` (archived)
(`probe`, `feats`, `harmonics`, `tone`): the same click/silence/tone/music probes,
the same features, so a profile built here matches the archived Phase 4 profiles.
Pure numpy; the caller supplies `render_fn(audio) -> audio` (a real capture's
NAMCore render) and a DI loader, so the probes are testable on synthetic amps.

Only the "fit" DIs are probed: held-out DIs are reserved for validation and are
never used to fit a profile, an audit or a selection (docs/history Phase 4 safeguards).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .envelope import bounded_causal_envelope_db

SR = 48000
PROBE_VERSION = 1          # bump when the probe bank or its features change: cached probes from another version are ignored
EQ_BANDS = {"sub": (20, 100), "low": (100, 250), "lowmid": (250, 800), "mid": (800, 2500), "presence": (2500, 6000), "air": (6000, 16000)}
FIT_DIS = ["clean_smooth", "moderate_hotrod", "high_thrash", "high_metalcore"]
SECONDS = 15
OFFSETS = [-12.0, -6.0, 0.0, 6.0]
TONE_F0 = (110.0, 440.0)
TONE_LEVELS = [-54.0, -42.0, -30.0, -18.0, -6.0, 0.0]


def _db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def crest_db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-9) / max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-9)))


def hf_ratio_db(x: np.ndarray) -> float:
    """Energy above 3 kHz relative to total, dB."""
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(10 * np.log10(max(spec[f > 3000].sum(), 1e-20) / max(spec.sum(), 1e-20)))


def features(y: np.ndarray) -> dict:
    y = y[SR // 2:].astype(np.float64)
    rms = float(np.sqrt(np.mean(y ** 2)))
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    tot = max(s.sum(), 1e-20)
    out = {"rms_db": 20 * np.log10(max(rms, 1e-9)), "peak_db": 20 * np.log10(max(np.max(np.abs(y)), 1e-9)),
           "crest_db": crest_db(y), "hf3k_db": hf_ratio_db(y)}
    for k, (lo, hi) in EQ_BANDS.items():
        out[f"eq_{k}_db"] = 10 * np.log10(max(s[(f >= lo) & (f < hi)].sum(), 1e-20) / tot)
    out["tilt_db"] = out["eq_presence_db"] - out["eq_low_db"]
    e = bounded_causal_envelope_db(y.astype(np.float32), SR)
    a = e[e > np.max(e) - 50]
    out["dyn_range_db"] = float(np.percentile(a, 95) - np.percentile(a, 10)) if len(a) else 0.0
    return out


def tone_signal(f0: float, level_db: float) -> np.ndarray:
    t = np.arange(3 * SR) / SR
    x = np.sin(2 * np.pi * f0 * t) * _db(level_db) * np.sqrt(2)
    x[: SR // 20] *= np.linspace(0, 1, SR // 20)
    return x.astype(np.float32)


def harmonics(y: np.ndarray, f0: float) -> dict:
    y = y[SR:].astype(np.float64)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    frq = np.fft.rfftfreq(len(y), 1 / SR)
    h = [sp[(frq > k * f0 - 6) & (frq < k * f0 + 6)].max() for k in range(1, 11)]
    thd = np.sqrt(sum(v ** 2 for v in h[1:])) / max(h[0], 1e-12)
    return {"thd_db": 20 * np.log10(max(thd, 1e-9)), "h2_db": 20 * np.log10(max(h[1] / h[0], 1e-9)),
            "h3_db": 20 * np.log10(max(h[2] / h[0], 1e-9)), "out_rms_db": 20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-9))}


def click(amp_level: float) -> np.ndarray:
    x = np.zeros(SR, np.float32)
    x[SR // 2] = amp_level
    return x


def click_stats(y: np.ndarray) -> dict:
    a = np.abs(y.astype(np.float64))
    pk = int(np.argmax(a))
    onset = int(np.argmax(a > 0.1 * a[pk]))
    return {"peak_idx": pk - SR // 2, "onset_idx": onset - SR // 2, "peak": float(a[pk])}


def silence_stats(y: np.ndarray) -> dict:
    y = y[SR // 4:].astype(np.float64)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f > 40) & (f < 400)
    hum = 10 * np.log10(max(sp[m].max(), 1e-30) / max(sp.sum(), 1e-30)) if sp.sum() > 0 else -300.0
    return {"rms_db": float(20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-9))), "dc": float(np.mean(y)), "hum_frac_db": float(hum)}


def music_extra(y: np.ndarray) -> dict:
    a = np.abs(y[SR // 2:].astype(np.float64))
    active = a > 10 ** (-60 / 20) * max(a.max(), 1e-9)
    nz = 0
    if active.any():
        idx = np.flatnonzero(active)
        seg = a[idx[0]:idx[-1] + 1]
        run = 0
        for v in (seg < 1e-6):
            run = run + 1 if v else 0
            nz = max(nz, run)
    return {"clipped_frac": float(np.mean(a >= 0.999)), "longest_zero_run": int(nz), "nan": bool(not np.isfinite(y).all())}


def probe_capture(render_fn: Callable[[np.ndarray], np.ndarray], load_di: Callable[[str], np.ndarray], *,
                  dis: list[str] | None = None, seconds: int = SECONDS, progress: Callable[[str], None] | None = None) -> dict:
    """Full probe bank through `render_fn` (a real capture, or a model behind an input gain)."""
    note = progress or (lambda _m: None)
    r = {"click_lo": click_stats(render_fn(click(0.05))), "click_hi": click_stats(render_fn(click(0.5))),
         "silence": silence_stats(render_fn(np.zeros(2 * SR, np.float32))), "tones": {}, "music": {}}
    for f0 in TONE_F0:
        for lv in TONE_LEVELS:
            y = render_fn(tone_signal(f0, lv))
            h = harmonics(y, f0)
            h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9)))
            r["tones"][f"{int(f0)}@{lv:g}"] = h
    note("tones done")
    for n in (dis or FIT_DIS):
        x = load_di(n)[: seconds * SR]
        for o in OFFSETS:
            y = render_fn((x * _db(o)).astype(np.float32))
            r["music"][f"{n}@{o:g}"] = {**features(y), **music_extra(y)}
        note(f"music {n} done")
    return r


def load_reference_di(name: str) -> np.ndarray:
    """Bundled preview DI (assets/di/<name>.wav) as mono float32 at 48 kHz."""
    from pathlib import Path

    import soundfile as sf

    x, sr = sf.read(Path(__file__).resolve().parent.parent / "assets" / "di" / f"{name}.wav", dtype="float32")
    if sr != SR:
        raise ValueError(f"DI {name} is {sr} Hz, expected {SR}")
    return x if x.ndim == 1 else x.mean(axis=1)

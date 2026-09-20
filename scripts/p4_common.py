"""Phase 4 shared probing/feature code (see docs/CONTINUOUS_GAIN_PHASE4.md). Import AFTER setting SINGLE_NAM_AMP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from single_nam_common import ALL_GAINS, INT_GAINS, REPO, SR, TEST_DIS, TRAIN_DIS, VAL_DIS, capture, capture_path, db, load_di
from hybrid.render import render
from cg_report import EQ_BANDS, feats, harmonics, tone

FIT_DIS = TRAIN_DIS + VAL_DIS            # used to fit profiles / mappings / selection
HELD_OUT_DIS = TEST_DIS                  # never used for any fitting; final validation only
SECONDS = 15
OFFSETS = [-12.0, -6.0, 0.0, 6.0]        # DI level offsets for music probes
TONE_F0 = (110.0, 440.0)
TONE_LEVELS = [-54.0, -42.0, -30.0, -18.0, -6.0, 0.0]   # sine RMS dBFS
P4 = REPO / "work" / "p4"


def out_dir(amp: str) -> Path:
    d = P4 / amp
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip(name: str) -> np.ndarray:
    return load_di(name)[: SECONDS * SR]


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
    return {"rms_db": float(20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-9))), "dc": float(np.mean(y)),
            "hum_frac_db": float(hum)}


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


def probe(render_fn, want_click: bool = True) -> dict:
    """Full probe bank through render_fn(audio)->audio (a real capture, or a model behind an input gain)."""
    r = {}
    if want_click:
        r["click_lo"] = click_stats(render_fn(click(0.05)))
        r["click_hi"] = click_stats(render_fn(click(0.5)))
        r["silence"] = silence_stats(render_fn(np.zeros(2 * SR, np.float32)))
    r["tones"] = {}
    for f0 in TONE_F0:
        for lv in TONE_LEVELS:
            y = render_fn(tone(f0, lv))
            h = harmonics(y, f0)
            h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9)))
            r["tones"][f"{int(f0)}@{lv:g}"] = h
    r["music"] = {}
    for n in FIT_DIS + HELD_OUT_DIS:
        x = clip(n)
        for o in OFFSETS:
            y = render_fn((x * db(o)).astype(np.float32))
            r["music"][f"{n}@{o:g}"] = {**feats(y), **music_extra(y)}
    return r


def load_json(path):
    return json.loads(Path(path).read_text())

"""Analyze the DI WAV library and (re)generate assets/di/_analysis.json.

Run from the repo root: python scripts/analyze_di.py
"""
import soundfile as sf
import numpy as np
import glob
import os
import json

DI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "di")

def dbfs(x):
    if x <= 0:
        return -np.inf
    return 20 * np.log10(x)

def analyze(path):
    data, sr = sf.read(path, always_2d=True)
    n_frames, n_ch = data.shape
    duration = n_frames / sr
    # use first channel (or mono) for analysis; note if multi-channel differs
    mono = data.mean(axis=1)

    peak = np.max(np.abs(mono))
    peak_dbfs = dbfs(peak)

    rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
    rms_dbfs = dbfs(rms)

    crest_factor_db = peak_dbfs - rms_dbfs if np.isfinite(peak_dbfs) and np.isfinite(rms_dbfs) else None

    # windowed RMS envelope for dynamic range estimate (50ms windows)
    win = max(1, int(sr * 0.05))
    n_win = n_frames // win
    if n_win > 0:
        trimmed = mono[: n_win * win].reshape(n_win, win)
        win_rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
        win_rms_nonzero = win_rms[win_rms > 1e-6]
        if len(win_rms_nonzero) > 5:
            win_db = 20 * np.log10(win_rms_nonzero)
            p10 = np.percentile(win_db, 10)
            p90 = np.percentile(win_db, 90)
            dyn_range_db = p90 - p10
        else:
            dyn_range_db = None
    else:
        dyn_range_db = None

    # silence detection at start/end (threshold -50 dBFS relative to peak)
    thresh = peak * 10 ** (-50 / 20) if peak > 0 else 0
    above = np.abs(mono) > thresh
    idx = np.nonzero(above)[0]
    if len(idx) > 0:
        lead_silence_s = idx[0] / sr
        trail_silence_s = (n_frames - 1 - idx[-1]) / sr
    else:
        lead_silence_s = duration
        trail_silence_s = 0.0

    return {
        "filename": os.path.basename(path),
        "sample_rate": sr,
        "channels": n_ch,
        "duration_s": round(duration, 2),
        "peak_dbfs": round(peak_dbfs, 2) if np.isfinite(peak_dbfs) else None,
        "rms_dbfs": round(rms_dbfs, 2) if np.isfinite(rms_dbfs) else None,
        "crest_factor_db": round(crest_factor_db, 2) if crest_factor_db is not None else None,
        "dynamic_range_db_p10_p90": round(dyn_range_db, 2) if dyn_range_db is not None else None,
        "lead_silence_s": round(lead_silence_s, 3),
        "trail_silence_s": round(trail_silence_s, 3),
    }

results = []
for path in sorted(glob.glob(os.path.join(DI_DIR, "*.wav"))):
    r = analyze(path)
    results.append(r)
    print(json.dumps(r, indent=2))

with open(os.path.join(DI_DIR, "_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

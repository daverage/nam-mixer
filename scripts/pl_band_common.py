"""Weight-band variant of the teacher (ANALYSIS ONLY; hybrid/multi_blend.py is not modified). Design and criteria: docs/phase4e/weight_band_criteria.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from hybrid.blend import smoothstep_curve
PHIS = [0.0, 0.5, 0.75, 0.9, 1.0]

def band_weights(envelope_db, levels_db, phi):
    """(N,T) weights. phi=0 == hybrid.multi_blend.chain_weights exactly; phi=1 is a hard switch at the midpoint between adjacent anchors."""
    env = np.asarray(envelope_db, dtype=np.float64); lv = np.asarray(levels_db, dtype=np.float64); n = len(lv)
    w = np.zeros((n, len(env))); pos = np.clip(env, lv[0], lv[-1])
    hi = np.clip(np.searchsorted(lv, pos, side="right"), 1, n - 1); lo = hi - 1
    p = np.clip((pos - lv[lo]) / (lv[hi] - lv[lo]), 0.0, 1.0)
    if phi >= 1.0: t = (p >= 0.5).astype(np.float64)
    else:
        wd = 1.0 - phi; t = smoothstep_curve(np.clip((p - (0.5 - wd / 2)) / wd, 0.0, 1.0))
    idx = np.arange(len(env)); w[lo, idx] = 1.0 - t; w[hi, idx] += t
    return w

def crossfade_stats(w, env, sr):
    """Durations (ms) of contiguous stretches where no capture holds >= 0.9 of the weight, and dominant-capture switches per second, over active frames."""
    act = env > env.max() - 40; dom = w.max(axis=0) >= 0.9; cross = (~dom) & act
    d = np.diff(np.concatenate([[0], cross.astype(int), [0]])); starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1); dur = (ends - starts) / sr * 1000.0
    am = w.argmax(axis=0)[act]; switches = int(np.sum(np.diff(am) != 0)); secs = act.sum() / sr
    return {"n_crossfades": int(len(dur)), "median_ms": float(np.median(dur)) if len(dur) else None, "p10_ms": float(np.percentile(dur, 10)) if len(dur) else None,
            "switches_per_s": float(switches / max(secs, 1e-9)), "frac_time_dominant": float(np.mean(dom[act])) if act.any() else 0.0}

def hf10k_db(y, sr):
    y = y[sr // 2:].astype(np.float64); s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2; f = np.fft.rfftfreq(len(y), 1 / sr)
    return float(10 * np.log10(max(s[f > 10000].sum(), 1e-20) / max(s.sum(), 1e-20)))

"""Small parametric post-NAM EQ (RBJ biquads, causal minimum-phase IIR) + exact per-bin |H|^2 for LTI evaluation."""
import numpy as np
from scipy.signal import sosfreqz, sosfilt
SR = 48000
def _norm(b, a): b, a = np.array(b, float), np.array(a, float); return np.r_[b / a[0], a / a[0]]
def peaking(f0, g_db, q):
    A = 10 ** (g_db / 40); w = 2 * np.pi * f0 / SR; al = np.sin(w) / (2 * q); c = np.cos(w)
    return _norm([1 + al * A, -2 * c, 1 - al * A], [1 + al / A, -2 * c, 1 - al / A])
def shelf(f0, g_db, s, high):
    A = 10 ** (g_db / 40); w = 2 * np.pi * f0 / SR; c = np.cos(w); al = np.sin(w) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2); r = 2 * np.sqrt(A) * al
    if high: return _norm([A * ((A + 1) + (A - 1) * c + r), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - r)], [(A + 1) - (A - 1) * c + r, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - r])
    return _norm([A * ((A + 1) - (A - 1) * c + r), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - r)], [(A + 1) + (A - 1) * c + r, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - r])
def build(params):
    """params: list of dicts {type: low_shelf|high_shelf|peak, f, gain_db, q (peak) or s (shelf)} -> sos array (n,6)"""
    rows = []
    for p in params:
        if p["type"] == "peak": rows.append(peaking(p["f"], p["gain_db"], p["q"]))
        else: rows.append(shelf(p["f"], p["gain_db"], p.get("s", 0.7), p["type"] == "high_shelf"))
    return np.array(rows)
def resp_db(sos, freqs): _, h = sosfreqz(sos, worN=np.asarray(freqs) * 2 * np.pi / SR); return 20 * np.log10(np.abs(h) + 1e-12)
def power(sos, freqs): return 10 ** (resp_db(sos, freqs) / 10)
def apply(sos, x): return sosfilt(sos, x.astype(np.float64)).astype(np.float32)

"""Tonal-refinement shared analysis: 1/3-octave band diffs from class-resolved power spectra (tr_measure.py). Signed convention everywhere: MODEL minus REAL, dB;
positive = the model has MORE energy there than the real capture."""
import json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent; TR = REPO / "work" / "tr"; SR = 48000; NP = 2048
FREQ = np.fft.rfftfreq(NP, 1 / SR)
FC_BANDS = 1000.0 * 2.0 ** (np.arange(-13, 12) / 3.0)            # 50 Hz ... 12.7 kHz
EDGE = np.concatenate([[FC_BANDS[0] * 2 ** (-1 / 6)], FC_BANDS * 2 ** (1 / 6)])
BIN_BAND = np.digitize(FREQ, EDGE) - 1                            # -1 or len(FC_BANDS) = outside
REFB = (FC_BANDS >= 200) & (FC_BANDS <= 6400)                    # bands used to remove overall level from the shape
BROAD = {"low 80-250": (80, 250), "low-mid 250-630": (250, 630), "mid 630-1.6k": (630, 1600), "upper-mid 1.6-4k": (1600, 4000), "high 4-10k": (4000, 10000)}
CLASSES = ["all", "attack", "sustain", "decay"]
REGIONS = {"jcm800": {"clean (G1-2)": [1, 2], "edge (G3-4)": [3, 4], "crunch (G5-8)": [5, 6, 7, 8], "saturated (G9-10)": [9, 10]},
           "vibrolux": {"clean (G1-2)": [1, 2], "edge (G3-4)": [3, 4], "crunch (G5-8)": [5, 6, 7, 8], "saturated (G9-10)": [9, 10]}}

def bandpow(P, H=None):
    """power spectrum (1025 bins) -> 1/3-octave band powers; H = optional |H(f)|^2 applied per bin (post-NAM EQ, exact for a LTI filter)."""
    if H is not None: P = P * H
    return np.array([P[BIN_BAND == i].sum() for i in range(len(FC_BANDS))]) + 1e-30

def shape_db(Pm, Pr, H=None):
    """band-wise signed diff (model - real, dB) with overall level removed over the reference bands; second value = the removed level (dB)."""
    d = 10 * np.log10(bandpow(Pm, H) / bandpow(Pr)); lv = float(d[REFB].mean()); return d - lv, lv

def broad_db(Pm, Pr, H=None):
    """5 broad bands, level-independent (same removed level as shape_db), plus HF>3k share diff and tilt (dB/oct, 200-8k)."""
    lv = shape_db(Pm, Pr, H)[1]; A = Pm * (H if H is not None else 1); out = {}
    for n, (lo, hi) in BROAD.items():
        m = (FREQ >= lo) & (FREQ < hi); out[n] = float(10 * np.log10(A[m].sum() / Pr[m].sum()) - lv)
    m3 = FREQ >= 3000; m0 = FREQ >= 60
    out["HF>3k share"] = float(10 * np.log10(A[m3].sum() / A[m0].sum()) - 10 * np.log10(Pr[m3].sum() / Pr[m0].sum()))
    d = shape_db(Pm, Pr, H)[0]; mk = (FC_BANDS >= 200) & (FC_BANDS <= 8000); out["tilt dB/oct"] = float(np.polyfit(np.log2(FC_BANDS[mk]), d[mk], 1)[0]); out["level dB"] = lv
    return out

class Spec:
    def __init__(self, amp, which):
        z = np.load(TR / amp / f"spec_{which}.npz"); self.z = {k: z[k] for k in z.files}; self.keys = [str(k) for k in self.z["keys"]]
        self.gains = sorted({float(k.split("|")[0]) for k in self.keys}); self.dis = sorted({k.split("|")[1] for k in self.keys}); self.offs = sorted({float(k.split("|")[2]) for k in self.keys})
    def P(self, g, d, o, model, cls="all"): return self.z[f"{g:g}|{d}|{o:g}|{model}|{cls}"]
    def scalar(self, g, d, o, model, what): return float(self.z[f"{g:g}|{d}|{o:g}|{model}|{what}"])
    def nframes(self, g, d, o, cls): return int(self.z[f"{g:g}|{d}|{o:g}|nframes|{cls}"])
    def T(self, g): return float(self.z[f"{g:g}|{self.dis[0]}|{self.offs[0]:g}|T"])

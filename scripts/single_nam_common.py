"""Shared helpers for the single-NAM multi-gain training experiment
(docs/CONTINUOUS_GAIN_SINGLE_NAM_TRAINING.md). No training or Flask deps."""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hybrid.audio_metrics import rms_dbfs, spectral_magnitude_correlation  # noqa: E402
from hybrid.nam_loader import load_nam  # noqa: E402
from hybrid.render import render  # noqa: E402
from hybrid.validation import compute_esr_metrics  # noqa: E402

SR = 48000
AMP_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Marshall JCM800 2203 - updated")
AMP = os.environ.get("SINGLE_NAM_AMP", "jcm800")
TWIN_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/FENDER 57 CUSTOM TWIN (MULTI GAIN)")
OUT = REPO / "work" / ("single_nam" if AMP == "jcm800" else f"single_nam_{AMP}")
OUT.mkdir(parents=True, exist_ok=True)

INT_GAINS = [float(g) for g in range(1, 11)]
HALF_GAINS = [g + 0.5 for g in range(1, 10)] if AMP == "jcm800" else []
ALL_GAINS = sorted(INT_GAINS + HALF_GAINS)

# Previously established calibrated-G5 table (docs/history/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md).
# Fitted per-target against real captures INCLUDING half-steps -> Baseline 2 only, never a control law.
CALIBRATED_G5_DB = {} if AMP != "jcm800" else {1.0: -24.0, 1.5: -19.0, 2.0: -15.0, 2.5: -10.6, 3.0: -6.8, 3.5: -4.2, 4.0: -2.4, 4.5: -1.2,
                    5.0: 0.0, 5.5: 1.4, 6.0: 2.0, 6.5: 5.8, 7.0: 9.0, 7.5: 10.8, 8.0: 13.0, 9.0: 15.2,
                    9.5: 14.8, 10.0: 15.2}

# DI split. Training material never overlaps final-test material.
TRAIN_DIS = ["clean_smooth", "moderate_hotrod", "high_thrash"]
VAL_DIS = ["high_metalcore"]
TEST_DIS = ["moderate_brit", "clean_mayer", "bass_rollin"]


SUPERSONIC_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/[500 Epochs] Fender Super-Sonic 60W Head mk.1 - Flat EQ - Complete Pack")


PEAVEY_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Peavy 5150 (Head Only)")


def capture_path(g: float) -> Path:
    if AMP == "peavey":
        return PEAVEY_DIR / f"AMP HEAD - 5150 Gain {int(g)}.nam"
    if AMP == "supersonic":
        return SUPERSONIC_DIR / f"Super-Sonic Bassman Ch T5 B5 V{int(g)}.nam"
    if AMP == "twin":
        return TWIN_DIR / f"57 CUSTOM TWIN -  CH 1 - VOL {int(g)}.nam"
    return AMP_DIR / ("jcm800-high-ga10-11.4dBu.nam" if g == 10.0 else f"jcm800-high-g{g:.1f}-11.4dBu.nam")


@lru_cache(maxsize=None)
def capture(g: float):
    return load_nam(capture_path(g))


def load_di(name: str) -> np.ndarray:
    x, sr = sf.read(REPO / "assets" / "di" / f"{name}.wav", dtype="float32")
    assert sr == SR
    return x if x.ndim == 1 else x.mean(axis=1)


def official_input() -> np.ndarray:
    x, sr = sf.read(REPO / "work" / "training_input" / "input.wav", dtype="float32")
    assert sr == SR
    return x if x.ndim == 1 else x.mean(axis=1)


def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


@lru_cache(maxsize=None)
def capture_lag(g: float) -> int:
    """Samples by which capture g lags the G5 capture (click test). Only applied for amps whose captures carry
    real, differing latencies (Super-Sonic); the shift is applied identically to targets, references and baselines."""
    if AMP not in ("supersonic", "peavey"):
        return 0
    click = np.zeros(SR, dtype=np.float32)
    click[SR // 2] = 0.05
    peak = lambda gg: int(np.argmax(np.abs(render(capture(gg), click, SR))))
    return peak(g) - peak(5.0)


def render_capture(g: float, audio: np.ndarray, input_db: float = 0.0) -> np.ndarray:
    y = render(capture(g), (audio * db(input_db)).astype(np.float32), SR)
    lag = capture_lag(g)
    if lag > 0:
        y = np.concatenate([y[lag:], np.zeros(lag, y.dtype)])
    elif lag < 0:
        y = np.concatenate([np.zeros(-lag, y.dtype), y[:lag]])
    return y


def render_file_nam(path: Path, audio: np.ndarray) -> np.ndarray:
    return render(load_nam(path), audio.astype(np.float32), SR)


def metrics(cand: np.ndarray, ref: np.ndarray, warm: int = SR // 2) -> dict:
    n = min(len(cand), len(ref))
    c, r = cand[warm:n], ref[warm:n]
    m = compute_esr_metrics(c, r)
    m["level_delta_db"] = float(rms_dbfs(c) - rms_dbfs(r))
    m["peak_delta_db"] = float(20 * np.log10(max(np.max(np.abs(c)), 1e-9) / max(np.max(np.abs(r)), 1e-9)))
    m["spectral_corr"] = float(spectral_magnitude_correlation(c, r))
    m["crest_delta_db"] = float(crest_db(c) - crest_db(r))
    m["hf_delta_db"] = float(hf_ratio_db(c) - hf_ratio_db(r))
    return m


def crest_db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-9) / max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-9)))


def hf_ratio_db(x: np.ndarray) -> float:
    """Energy above 3 kHz relative to total, dB -- a crude 'fizz/brightness of the distortion' measure."""
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(10 * np.log10(max(spec[f > 3000].sum(), 1e-20) / max(spec.sum(), 1e-20)))


def load_law() -> dict[float, float]:
    """Frozen control law {physical gain -> player input dB}, produced by step 1."""
    d = json.loads((OUT / "control_law.json").read_text())
    return {float(k): v for k, v in d["law_db"].items()}


def save_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2))

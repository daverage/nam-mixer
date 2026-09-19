"""Shared helpers for the single-NAM multi-gain training experiment
(docs/CONTINUOUS_GAIN_SINGLE_NAM_TRAINING.md). No training or Flask deps."""
from __future__ import annotations

import json
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
OUT = REPO / "work" / "single_nam"
OUT.mkdir(parents=True, exist_ok=True)

INT_GAINS = [float(g) for g in range(1, 11)]
HALF_GAINS = [g + 0.5 for g in range(1, 10)]
ALL_GAINS = sorted(INT_GAINS + HALF_GAINS)

# Previously established calibrated-G5 table (docs/history/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md).
# Fitted per-target against real captures INCLUDING half-steps -> Baseline 2 only, never a control law.
CALIBRATED_G5_DB = {1.0: -24.0, 1.5: -19.0, 2.0: -15.0, 2.5: -10.6, 3.0: -6.8, 3.5: -4.2, 4.0: -2.4, 4.5: -1.2,
                    5.0: 0.0, 5.5: 1.4, 6.0: 2.0, 6.5: 5.8, 7.0: 9.0, 7.5: 10.8, 8.0: 13.0, 9.0: 15.2,
                    9.5: 14.8, 10.0: 15.2}

# DI split. Training material never overlaps final-test material.
TRAIN_DIS = ["clean_smooth", "moderate_hotrod", "high_thrash"]
VAL_DIS = ["high_metalcore"]
TEST_DIS = ["moderate_brit", "clean_mayer", "bass_rollin"]


def capture_path(g: float) -> Path:
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


def render_capture(g: float, audio: np.ndarray, input_db: float = 0.0) -> np.ndarray:
    return render(capture(g), (audio * db(input_db)).astype(np.float32), SR)


def render_file_nam(path: Path, audio: np.ndarray) -> np.ndarray:
    return render(load_nam(path), audio.astype(np.float32), SR)


def metrics(cand: np.ndarray, ref: np.ndarray, warm: int = SR // 2) -> dict:
    n = min(len(cand), len(ref))
    c, r = cand[warm:n], ref[warm:n]
    m = compute_esr_metrics(c, r)
    m["level_delta_db"] = float(rms_dbfs(c) - rms_dbfs(r))
    m["peak_delta_db"] = float(20 * np.log10(max(np.max(np.abs(c)), 1e-9) / max(np.max(np.abs(r)), 1e-9)))
    m["spectral_corr"] = float(spectral_magnitude_correlation(c, r))
    return m


def load_law() -> dict[float, float]:
    """Frozen control law {physical gain -> player input dB}, produced by step 1."""
    d = json.loads((OUT / "control_law.json").read_text())
    return {float(k): v for k, v in d["law_db"].items()}


def save_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2))

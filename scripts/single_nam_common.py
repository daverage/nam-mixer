"""Capture-file locations and the official training input, as used by the
single-NAM multi-gain training experiment
(docs/history/Continuous Gain/CONTINUOUS_GAIN_SINGLE_NAM_TRAINING.md). Kept
only for scripts/cg_reproduce_fc.py's frozen-configuration reproduction."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent

SR = 48000
AMP_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Marshall JCM800 2203 - updated")
AMP = os.environ.get("SINGLE_NAM_AMP", "jcm800")
TWIN_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/FENDER 57 CUSTOM TWIN (MULTI GAIN)")


SUPERSONIC_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/[500 Epochs] Fender Super-Sonic 60W Head mk.1 - Flat EQ - Complete Pack")


PEAVEY_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Peavy 5150 (Head Only)")


P6505_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/6505+ Gain Range Pack (High Gain)")


MESA_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/MESA DUAL RECTIFIER (REV G 1998) - FULL GAIN PACK")
ORANGE_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Orange Dual Terror (0.5.2)")


def capture_path(g: float) -> Path:
    if AMP == "mesa":
        return MESA_DIR / f"MESADUAL - RED - MODERN - GAIN {'MAX' if g == 10 else int(g)}.nam"
    if AMP == "orange":
        return ORANGE_DIR / f"ORANGE - DUAL TERROR - FAT - G{int(g)}.nam"
    if AMP == "peavey6505":
        return P6505_DIR / f"APP-6505Plus-Scooped-Gain-{int(g):02d}.nam"
    if AMP == "peavey":
        return PEAVEY_DIR / f"AMP HEAD - 5150 Gain {int(g)}.nam"
    if AMP == "vibrolux":
        return SUPERSONIC_DIR / f"Super-Sonic Vibrolux Ch T5 B5 V{int(g)}.nam"
    if AMP == "supersonic":
        return SUPERSONIC_DIR / f"Super-Sonic Bassman Ch T5 B5 V{int(g)}.nam"
    if AMP == "twin":
        return TWIN_DIR / f"57 CUSTOM TWIN -  CH 1 - VOL {int(g)}.nam"
    return AMP_DIR / ("jcm800-high-ga10-11.4dBu.nam" if g == 10.0 else f"jcm800-high-g{g:.1f}-11.4dBu.nam")


def official_input() -> np.ndarray:
    x, sr = sf.read(REPO / "work" / "training_input" / "input.wav", dtype="float32")
    assert sr == SR
    return x if x.ndim == 1 else x.mean(axis=1)

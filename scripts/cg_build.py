"""Continuous Gain v3: build an N-capture chain training bundle (input.wav / target.wav / manifest.json).

Gain positions are DESIGNED, not fitted: capture Gn owns level L_n = LO + (n-1)*STEP dBFS of the causal input
envelope; reference playing level is REF. Designated player input gain for Gn = L_n - REF. Target = level-driven
blend (hybrid.multi_blend) of the real captures, each rendered on the input rescaled to REF. No per-capture level
matching; one fixed peak-ceiling gain for the whole target. Segments: official NAM input (all) + training DIs at
several level offsets; validation = a different DI, at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from single_nam_common import SR, TRAIN_DIS, VAL_DIS, capture, db, load_di, official_input, render_capture  # noqa: E402  (adds repo to sys.path)
from hybrid.envelope import bounded_causal_envelope_db
from hybrid.multi_blend import GainChain, multi_blend
from hybrid.safety import apply_peak_ceiling

LO, STEP, REF = -52.0, 4.0, -30.0
PAD = SR // 2
CEILING_DBFS = -0.2


def level_of(gain: float) -> float:
    return LO + (gain - 1.0) * STEP


def segment(x: np.ndarray, chain: GainChain):
    env = bounded_causal_envelope_db(x, SR)
    renders = [render_capture(g, (x * db(chain.input_scale_db(k))).astype(np.float32)) for k, g in enumerate(chain.labels)]
    return multi_blend(env, renders, chain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", type=float, nargs="+", default=[float(g) for g in range(1, 11)])
    ap.add_argument("--offsets", type=float, nargs="+", default=[-16.0, -8.0, 0.0, 8.0])
    ap.add_argument("--di-seconds", type=int, default=25)
    ap.add_argument("--val-seconds", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    chain = GainChain(tuple(a.gains), tuple(level_of(g) for g in a.gains), REF)
    srcs = [("official", 0.0, official_input())]
    srcs += [(n, o, load_di(n)[: a.di_seconds * SR]) for n in TRAIN_DIS for o in a.offsets]
    vsrc = [(n, 0.0, load_di(n)[: a.val_seconds * SR]) for n in VAL_DIS]
    X, Y, seg, pos, train_stop = [], [], [], 0, 0
    for split, group in (("train", srcs), ("val", vsrc)):
        for name, off, x in group:
            x = np.concatenate([x * db(off), np.zeros(PAD, np.float32)]).astype(np.float32)
            y = segment(x, chain)
            X.append(x); Y.append(y)
            seg.append({"split": split, "source": name, "offset_db": off, "start": pos, "stop": pos + len(x)})
            pos += len(x)
            print(split, name, off, f"{len(x)/SR:.0f}s", flush=True)
        if split == "train":
            train_stop = pos
    X, Y = np.concatenate(X), np.concatenate(Y)
    Y, red_db = apply_peak_ceiling(Y[:], CEILING_DBFS)
    a.out.mkdir(parents=True, exist_ok=True)
    sf.write(a.out / "input.wav", X, SR, subtype="FLOAT")
    sf.write(a.out / "target.wav", Y.astype(np.float32), SR, subtype="FLOAT")
    (a.out / "manifest.json").write_text(json.dumps({
        "gains": list(chain.labels), "levels_db": list(chain.levels_db), "reference_db": REF,
        "designated_gain_db": {str(g): chain.designated_gain_db(k) for k, g in enumerate(chain.labels)},
        "train_stop": train_stop, "total": pos, "output_scale_c": float(10 ** (-red_db / 20)) if red_db else 1.0,
        "peak_ceiling_gain_reduction_db": red_db, "segments": seg, "input_peak": float(np.max(np.abs(X))),
        "target_sha256": hashlib.sha256((a.out / "target.wav").read_bytes()).hexdigest()}, indent=2))
    print(f"train {train_stop/SR:.0f}s total {pos/SR:.0f}s reduction {red_db:.2f} dB input peak {np.max(np.abs(X)):.2f}")


if __name__ == "__main__":
    main()

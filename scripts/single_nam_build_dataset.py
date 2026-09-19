"""Build the Method A (direct multi-gain) training pair.

For each captured integer gain N in `--gains`: input = DI * law(N) dB, target = REAL capture N rendered on the
UNGAINED DI (output level progression preserved). Blocks are concatenated in gain order; validation blocks
(a different DI) sit at the end so the trainer's train/validation split is a single boundary.
One global constant c scales all targets to -18 dBFS RMS (train set); evaluation divides it back out.
"""
from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

from single_nam_common import INT_GAINS, OUT, SR, TRAIN_DIS, VAL_DIS, db, load_di, load_law, official_input, render_capture, save_json

PAD = SR // 2


def pieces(names, official_s, di_s):
    out = []
    if official_s:
        out.append(("official", official_input()[10 * SR:(10 + official_s) * SR]))
    for n in names:
        d = load_di(n)
        out.append((n, d[: di_s * SR] if di_s else d))
    return out


def block(gain, srcs, law):
    xs, ys, idx = [], [], []
    for name, a in srcs:
        a = np.concatenate([a, np.zeros(PAD, np.float32)])
        xs.append(a * db(law[gain]))
        ys.append(render_capture(gain, a))
        idx.append((name, len(a)))
    return np.concatenate(xs), np.concatenate(ys), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", type=float, nargs="+", default=INT_GAINS)
    ap.add_argument("--official-seconds", type=int, default=45)
    ap.add_argument("--di-seconds", type=int, default=25)
    ap.add_argument("--val-seconds", type=int, default=15)
    ap.add_argument("--name", default="method_a_10")
    a = ap.parse_args()
    law = load_law()
    train_src = pieces(TRAIN_DIS, a.official_seconds, a.di_seconds)
    val_src = pieces(VAL_DIS, 0, a.val_seconds)

    X, Y, seg = [], [], []
    pos = 0
    for split, srcs in (("train", train_src), ("val", val_src)):
        for g in a.gains:
            x, y, idx = block(g, srcs, law)
            for name, n in idx:
                seg.append({"split": split, "gain": g, "source": name, "start": pos, "stop": pos + n})
                pos += n
            X.append(x); Y.append(y)
            print(split, g, f"{len(x)/SR:.0f}s", flush=True)
        if split == "train":
            train_stop = pos
    X, Y = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)
    rms = float(np.sqrt(np.mean(Y[:train_stop].astype(np.float64) ** 2)))
    c = float(10 ** (-18 / 20) / rms)
    d = OUT / a.name
    d.mkdir(exist_ok=True)
    sf.write(d / "input.wav", X, SR, subtype="FLOAT")
    sf.write(d / "target.wav", (Y * c).astype(np.float32), SR, subtype="FLOAT")
    save_json(f"{a.name}/manifest.json", {"gains": a.gains, "train_stop": train_stop, "total": pos, "output_scale_c": c,
                                          "law_db": {str(k): law[k] for k in a.gains}, "segments": seg,
                                          "train_dis": ["official"] + TRAIN_DIS, "val_dis": VAL_DIS,
                                          "input_peak": float(np.max(np.abs(X)))})
    print(f"train {train_stop/SR:.0f}s  total {pos/SR:.0f}s  c={c:.3f}  input peak {np.max(np.abs(X)):.2f}")


if __name__ == "__main__":
    main()

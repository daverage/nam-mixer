"""Phase 4E bundle builder (new; v3 code untouched). Same recipe as scripts/cg_build.py (official input + training DIs at four level offsets,
validation DI, one fixed peak ceiling, no per-capture level matching) but with (a) EXPLICIT training anchors per selected capture and
(b) timing alignment taken from the verified Phase 4A audit corrections instead of click-based shifts.
Usage: p4e_build.py <amp> --gains 1 2 4 10 --anchors -22 -18 -10 14 --out DIR   (anchors = designated input gain in dB, level = anchor + reference)"""
import argparse
import hashlib
import json
import os
import sys

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from pathlib import Path  # noqa: E402

from single_nam_common import SR, TRAIN_DIS, VAL_DIS, capture, db, load_di, official_input, REPO  # noqa: E402
from hybrid.envelope import bounded_causal_envelope_db  # noqa: E402
from hybrid.multi_blend import GainChain, multi_blend  # noqa: E402
from hybrid.render import render  # noqa: E402
from hybrid.safety import apply_peak_ceiling  # noqa: E402

REF = -30.0
PAD = SR // 2
CEILING_DBFS = -0.2
amp = sys.argv[1]
AUD = json.loads((REPO / "work" / "p4" / amp / "audit.json").read_text())["captures"]

def align_shift(g: float) -> int:
    c = AUD[f"{g:g}"]["correction"]
    return -int(c["samples"]) if c and "samples" in c else 0     # music-derived delay (samples) verified in Phase 4A

def render_aligned(g, x):
    y = render(capture(g), x, SR); sh = align_shift(g)
    if sh > 0: return np.concatenate([y[sh:], np.zeros(sh, y.dtype)])
    if sh < 0: return np.concatenate([np.zeros(-sh, y.dtype), y[:sh]])
    return y

def segment(x, chain):
    env = bounded_causal_envelope_db(x, SR)
    renders = [render_aligned(g, (x * db(chain.input_scale_db(k))).astype(np.float32)) for k, g in enumerate(chain.labels)]
    return multi_blend(env, renders, chain)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("amp")
    ap.add_argument("--gains", type=float, nargs="+", required=True); ap.add_argument("--anchors", type=float, nargs="+", required=True)
    ap.add_argument("--offsets", type=float, nargs="+", default=[-16.0, -8.0, 0.0, 8.0]); ap.add_argument("--di-seconds", type=int, default=25)
    ap.add_argument("--val-seconds", type=int, default=12); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert len(a.gains) == len(a.anchors)
    chain = GainChain(tuple(a.gains), tuple(t + REF for t in a.anchors), REF)
    srcs = [("official", 0.0, official_input())] + [(n, o, load_di(n)[: a.di_seconds * SR]) for n in TRAIN_DIS for o in a.offsets]
    vsrc = [(n, 0.0, load_di(n)[: a.val_seconds * SR]) for n in VAL_DIS]
    X, Y, seg, pos, train_stop = [], [], [], 0, 0
    for split, group in (("train", srcs), ("val", vsrc)):
        for name, off, x in group:
            x = np.concatenate([x * db(off), np.zeros(PAD, np.float32)]).astype(np.float32)
            X.append(x); Y.append(segment(x, chain))
            seg.append({"split": split, "source": name, "offset_db": off, "start": pos, "stop": pos + len(x)}); pos += len(x)
        if split == "train": train_stop = pos
    X, Y = np.concatenate(X), np.concatenate(Y)
    Y, red_db = apply_peak_ceiling(Y[:], CEILING_DBFS)
    a.out.mkdir(parents=True, exist_ok=True)
    sf.write(a.out / "input.wav", X, SR, subtype="FLOAT"); sf.write(a.out / "target.wav", Y.astype(np.float32), SR, subtype="FLOAT")
    (a.out / "manifest.json").write_text(json.dumps({
        "amp": amp, "gains": list(chain.labels), "anchors_designated_gain_db": list(a.anchors), "levels_db": list(chain.levels_db), "reference_db": REF,
        "alignment_shifts_samples": {f"{g:g}": align_shift(g) for g in a.gains}, "train_stop": train_stop, "total": pos,
        "output_scale_c": float(10 ** (-red_db / 20)) if red_db else 1.0, "peak_ceiling_gain_reduction_db": red_db, "segments": seg,
        "input_peak": float(np.max(np.abs(X))), "train_seconds": train_stop / SR, "val_seconds": (pos - train_stop) / SR,
        "target_audio_sha256": hashlib.sha256(Y.astype(np.float32).tobytes()).hexdigest(), "input_audio_sha256": hashlib.sha256(X.astype(np.float32).tobytes()).hexdigest(),
        "note": "audio hashes are over the float32 sample bytes; WAV files themselves carry a libsndfile timestamp so file hashes differ between identical builds"}, indent=2))
    print(f"train {train_stop/SR:.0f}s total {pos/SR:.0f}s reduction {red_db:.2f} dB target-audio {hashlib.sha256(Y.astype(np.float32).tobytes()).hexdigest()[:16]}")

main()

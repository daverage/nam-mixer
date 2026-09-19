"""Step 1: capture latency check, frozen control law (integers only), conflict analysis."""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from single_nam_common import (ALL_GAINS, CALIBRATED_G5_DB, INT_GAINS, OUT, SR, TRAIN_DIS, capture, db, load_di,
                               metrics, render_capture, save_json)


def click_latency() -> dict:
    click = np.zeros(SR, dtype=np.float32)
    click[SR // 2] = 0.05
    res = {}
    for g in ALL_GAINS:
        y = render_capture(g, click)
        res[g] = int(np.argmax(np.abs(y)) - SR // 2)
    return res


def cal_clip() -> np.ndarray:
    """Calibration audio: 3 s from each TRAINING-side DI (never validation/test material)."""
    return np.concatenate([load_di(n)[SR * 8: SR * 11] for n in TRAIN_DIS])


def best_gain(anchor: float, ref: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    def esr(d):
        y = render_capture(anchor, x, d)
        return metrics(y, ref)["raw_esr"]
    coarse = {d: esr(d) for d in np.arange(-36, 24.1, 2.0)}
    c = min(coarse, key=coarse.get)
    fine = {d: esr(d) for d in np.arange(c - 2, c + 2.01, 0.2)}
    best = min({**coarse, **fine}, key=lambda d: {**coarse, **fine}[d])
    return float(round(best, 2)), float({**coarse, **fine}[best])


def main():
    t = time.time()
    lat = click_latency()
    med = float(np.median(list(lat.values())))
    anomalies = {g: l for g, l in lat.items() if abs(l - med) > 20}
    print("click latency (samples):", lat, "\nanomalies:", anomalies, f"({time.time()-t:.0f}s)")

    x = cal_clip()
    law, fit_esr = {}, {}
    for g in INT_GAINS:
        ref = render_capture(g, x)
        law[g], fit_esr[g] = best_gain(5.0, ref, x)
        print(f"G{g:g}: law {law[g]:+.1f} dB  ESR {fit_esr[g]:.4f}  ({time.time()-t:.0f}s)", flush=True)
    law[5.0] = 0.0
    p = PchipInterpolator(INT_GAINS, [law[g] for g in INT_GAINS])
    full = {g: round(float(p(g)), 2) if g not in law else law[g] for g in ALL_GAINS}
    save_json("control_law.json", {
        "definition": "G5 NAM anchor; per integer physical Gain, input dB minimising raw ESR vs the real capture on "
                      "3s clips of the TRAINING DIs (integers only; no half-step or test data). Half-steps: PCHIP "
                      "interpolation of the integer law. Frozen before any training/eval.",
        "law_db": full, "integer_fit_raw_esr": fit_esr, "click_latency_samples": lat, "latency_anomalies": anomalies,
        "old_calibrated_g5_db_baseline2": CALIBRATED_G5_DB})

    # Conflict analysis: pairs of integer gains whose model inputs are ~identical up to a level offset
    ys = {g: render_capture(g, x) for g in INT_GAINS}
    pairs = []
    for i, a in enumerate(INT_GAINS):
        for b in INT_GAINS[i + 1:]:
            d = abs(full[b] - full[a])
            m = metrics(ys[a], ys[b])
            pairs.append({"a": a, "b": b, "input_db_gap": round(d, 2), "target_raw_esr": m["raw_esr"],
                          "target_level_gap_db": m["level_delta_db"], "target_gain_norm_esr": m["gain_normalized_esr"]})
    adj = [p for p in pairs if p["b"] - p["a"] == 1]
    print("\nAdjacent-gain conflict table (input dB gap vs target divergence):")
    for p in adj:
        print(f"G{p['a']:g}->G{p['b']:g}: input gap {p['input_db_gap']:5.1f} dB | target raw ESR {p['target_raw_esr']:.3f} "
              f"| level gap {p['target_level_gap_db']:+.1f} dB | shape ESR {p['target_gain_norm_esr']:.3f}")
    save_json("conflict_pairs.json", {"pairs": pairs, "note": "adjacent-pair input_db_gap small + target ESR large => "
              "inseparable-by-level; the network must average them"})


if __name__ == "__main__":
    main()

"""Continuous Gain Stage 4: automated validation of a trained model, plus audition material.

Everything reported here is an EXISTING measurement (`cg_probe.features`, level-matched ESR via
`hybrid.training.validation.compute_esr_metrics`) taken on the HELD-OUT DIs (never used to fit a profile, audit or
selection). No perceptual quality score is computed and nothing here gates export: the checks report, the user
decides. The model is compared with the real captures at the SAME Input gain the plan maps each position to,
using ONE global output constant (the training scale), never a per-position correction.

  compatibility  ordinary standard .nam structure, Full and Lite renders finite and the right length
  safety         raw and scaled output peaks over the intended Input-gain range; recommended output gain
  progression    model vs real capture per position (training anchors AND omitted reference captures),
                 direction reversals of the model's progression against the real amp's
  coverage       how well the selected captures reproduced the omitted ones (from the plan, Phase 4D measures)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .audit import apply_alignment_shift as _shift
from .parallel import pmap
from .probe import SR, features, load_reference_di
from ..core.input_profiles import db_to_amplitude as _db
from ..core.nam_loader import load_nam
from ..core.render import SLIM_FULL, SLIM_LITE, NamRenderError, render
from ..training.validation import compute_esr_metrics

HELD_OUT_DIS = ("moderate_brit", "clean_mayer", "bass_rollin")
CLIP_SECONDS = 12
SWEEP_GAINS_DB = tuple(range(-20, 15, 2))
REVERSAL_MIN_DB = 0.3
EQ_KEYS = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]


def _peak_db(y: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.max(np.abs(y))), 1e-9)))


def check_compatibility(nam_path: Path, validation_input: np.ndarray) -> dict:
    raw = json.loads(Path(nam_path).read_text(encoding="utf-8"))
    out = {"architecture": raw.get("architecture"), "version": raw.get("version"), "sample_rate": raw.get("sample_rate"),
           "top_level_keys": sorted(raw), "metadata_keys": sorted((raw.get("metadata") or {}).keys())}
    model = load_nam(nam_path)
    for label, slim in (("full", SLIM_FULL), ("lite", SLIM_LITE)):
        try:
            y = render(model, validation_input, SR, slim=slim)
            ok = y.ndim == 1 and len(y) == len(validation_input) and bool(np.all(np.isfinite(y)))
            out[label] = {"rendered_ok": ok, "peak_dbfs": _peak_db(y) if ok else None}
        except NamRenderError as exc:
            out[label] = {"rendered_ok": False, "error": str(exc)}
    out["standard_nam"] = bool(raw.get("architecture") and raw.get("weights") is not None and raw.get("sample_rate") == SR) and out["full"]["rendered_ok"]
    return out


def check_safety(model, scale_c: float, x: np.ndarray, gains_db=SWEEP_GAINS_DB) -> dict:
    def one(g):
        y = render(model, (x * _db(g)).astype(np.float32), SR)
        return {"input_gain_db": g, "raw_peak_dbfs": _peak_db(y), "scaled_peak_dbfs": _peak_db(y / scale_c)}

    rows = pmap(one, list(gains_db))
    rec = -20 * np.log10(scale_c) if scale_c > 0 else 0.0
    over = [r["input_gain_db"] for r in rows if r["scaled_peak_dbfs"] > 0.0]
    return {"sweep": rows, "recommended_output_gain_db": float(rec), "finite": all(np.isfinite(r["raw_peak_dbfs"]) for r in rows),
            "scaled_peak_over_0dbfs_at_input_gain_db": over,
            "note": "The model output is scaled by the training constant c (one peak-ceiling gain for the whole target). Set the player's Output gain to the recommended value to restore the real amp's level; peaks above 0 dBFS at high Input gain with hard playing are reported, not corrected."}


def _delta(m: dict, r: dict) -> dict:
    return {"level_db": m["rms_db"] - r["rms_db"], "hf_db": m["hf3k_db"] - r["hf3k_db"], "crest_db": m["crest_db"] - r["crest_db"],
            "dyn_range_db": m["dyn_range_db"] - r["dyn_range_db"], "eq_max_db": max(abs(m[k] - r[k]) for k in EQ_KEYS)}


def check_progression(model, scale_c: float, capture_models: dict, shifts: dict, mapping: list[dict], training: set,
                      load_di: Callable[[str], np.ndarray] = load_reference_di, progress: Callable[[str], None] | None = None) -> dict:
    note = progress or (lambda _m: None)
    ordered = sorted(mapping, key=lambda r: r["position"])
    dis = {di: load_di(di)[: CLIP_SECONDS * SR] for di in HELD_OUT_DIS}

    def compare(row):                                    # one position: independent renders, so positions run in parallel
        p, T = row["position"], row["input_gain_db"]
        ds, esrs = [], []
        for di in HELD_OUT_DIS:
            x = dis[di]
            ym = render(model, (x * _db(T)).astype(np.float32), SR) / scale_c
            yr = _shift(render(capture_models[p], x, SR), shifts.get(p, 0))
            ds.append(_delta(features(ym), features(yr)))
            n = min(len(ym), len(yr)); w = SR // 2
            c, r = ym[w:n].astype(np.float64), yr[w:n].astype(np.float64)
            cl = c * np.sqrt(np.mean(r ** 2) / max(np.mean(c ** 2), 1e-20))
            esrs.append(float(compute_esr_metrics(cl, r)["raw_esr"]))
        agg = {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}
        note(f"compared position {p:g}")
        return {"position": p, "input_gain_db": T, "role": "training" if p in training else "reference", **agg, "lm_esr": float(np.mean(esrs))}

    rows = pmap(compare, ordered)
    di0 = dis[HELD_OUT_DIS[0]]

    def curve(row):
        return (features(render(model, (di0 * _db(row["input_gain_db"])).astype(np.float32), SR) / scale_c),
                features(_shift(render(capture_models[row["position"]], di0, SR), shifts.get(row["position"], 0))))

    fm, fr = zip(*pmap(curve, rows)) if rows else ((), ())
    # direction reversals of the model's progression vs the real amp's (level, HF, crest on the first held-out DI)
    rev = []
    for key, name in (("rms_db", "level"), ("hf3k_db", "HF"), ("crest_db", "crest")):
        for i in range(len(rows) - 1):
            dm, dr = fm[i + 1][key] - fm[i][key], fr[i + 1][key] - fr[i][key]
            if abs(dm) > REVERSAL_MIN_DB and abs(dr) > REVERSAL_MIN_DB and dm * dr < 0:
                rev.append({"measure": name, "between": [rows[i]["position"], rows[i + 1]["position"]], "model_step_db": dm, "real_step_db": dr})
    by = lambda role, k: [abs(r[k]) for r in rows if r["role"] == role]
    summ = {}
    for role in ("training", "reference"):
        if by(role, "level_db"):
            summ[role] = {k: {"mean_abs": float(np.mean(by(role, k))), "max_abs": float(np.max(by(role, k)))} for k in ("level_db", "hf_db", "crest_db", "dyn_range_db", "eq_max_db")}
            summ[role]["lm_esr_mean"] = float(np.mean([r["lm_esr"] for r in rows if r["role"] == role]))
    return {"positions": rows, "reversals": rev, "summary": summ, "held_out_dis": list(HELD_OUT_DIS),
            "note": "Differences are model minus real capture, at the plan's Input gain per position, on held-out DIs, with one global output scale. 'reference' positions were not training anchors, so they are independent evidence of the interpolation. These are measurements, not a listening result."}


def write_audition(out_dir: Path, model, scale_c: float, capture_models: dict, shifts: dict, mapping: list[dict],
                   load_di: Callable[[str], np.ndarray] = load_reference_di, di: str = "moderate_brit", seconds: int = 6) -> dict:
    """Short Input-gain sweep + per-position original-capture comparison clips (WAV, 24-bit). Level-safe: a clip whose
    scaled peak would exceed -1 dBFS is attenuated as a whole and flagged, never limited."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = load_di(di)[: seconds * SR]
    seg = pmap(lambda g: render(model, (x * _db(g)).astype(np.float32), SR) / scale_c, list(SWEEP_GAINS_DB))
    sweep = np.concatenate(seg)
    att = 0.0
    pk = _peak_db(sweep)
    if pk > -1.0:
        att = pk + 1.0
        sweep = sweep * _db(-att)
    sf.write(out_dir / "sweep.wav", sweep.astype(np.float32), SR, subtype="PCM_24")
    sweep_info = {"file": "sweep.wav", "di": di, "seconds_per_step": seconds, "input_gains_db": list(SWEEP_GAINS_DB), "attenuated_db": att}
    ordered = sorted(mapping, key=lambda r: r["position"])

    def clip_pair(row):
        p, T = row["position"], row["input_gain_db"]
        ym = render(model, (x * _db(T)).astype(np.float32), SR) / scale_c
        yr = _shift(render(capture_models[p], x, SR), shifts.get(p, 0))
        return ym, yr

    files = []
    for row, (ym, yr) in zip(ordered, pmap(clip_pair, ordered)):
        p, T = row["position"], row["input_gain_db"]
        a = max(0.0, max(_peak_db(ym), _peak_db(yr)) + 1.0)
        sf.write(out_dir / f"pos_{p:g}_model.wav", (ym * _db(-a)).astype(np.float32), SR, subtype="PCM_24")
        sf.write(out_dir / f"pos_{p:g}_original.wav", (yr * _db(-a)).astype(np.float32), SR, subtype="PCM_24")
        files.append({"position": p, "input_gain_db": T, "model": f"pos_{p:g}_model.wav", "original": f"pos_{p:g}_original.wav", "attenuated_db": a})
    return {"sweep": sweep_info, "comparisons": files}

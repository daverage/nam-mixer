"""Phase 4A capture audit for one amp -> work/p4/<amp>/audit.json (+ corrected render copies).
Statuses: VALID / CORRECTED (verified technical fix) / SUSPECT (unexplained, needs investigation) / INVALID (confirmed unrecoverable).
A capture is never invalidated merely for breaking an assumed smooth or monotonic curve: curve deviations only ever raise SUSPECT.
Usage: p4_audit.py <amp>"""
import json
import os
import sys

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

from p4_common import ALL_GAINS, FIT_DIS, SR, capture, clip, db, load_json, out_dir, render  # noqa: E402
from hybrid.validation import compute_esr_metrics  # noqa: E402

TIMING_TOL = 4          # samples vs set median onset
VERIFY_TOL = 3          # samples: a second, different DI must reproduce the music lag
AGREE_TOL = 30          # samples: click-onset offset and music lag must agree (click onset is only approximate for a nonlinear model)
NOISE_SPIKE_DB = 10.0    # over the local median of neighbours
NOISE_ABS_NOTE_DB = -50.0
LEVEL_TOL_DB = 3.0
LOUD_TOL_DB = 6.0
LATDEP_TOL = 20
CURVE_RATIO = 3.0
FEATS = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db", "hf3k_db", "crest_db", "dyn_range_db"]

amp = sys.argv[1]
D = out_dir(amp)
RANK = {'VALID': 0, 'CORRECTED': 1, 'SUSPECT': 2, 'INVALID': 3}
worse = lambda cur, new: new if RANK[new] > RANK[cur] else cur
P = load_json(D / "captures.json")["captures"]
gains = [float(k) for k in P]
key = lambda g: f"{g:g}"


def music_mean(g, feat, off="0"):
    return float(np.mean([P[key(g)]["probe"]["music"][f"{n}@{off}"][feat] for n in FIT_DIS]))


onset = np.array([P[key(g)]["probe"]["click_hi"]["onset_idx"] for g in gains], float)
onset_lo = np.array([P[key(g)]["probe"]["click_lo"]["onset_idx"] for g in gains], float)
med_on = float(np.median(onset))
noise = np.array([P[key(g)]["probe"]["silence"]["rms_db"] for g in gains])
rms = np.array([music_mean(g, "rms_db") for g in gains])
loud = np.array([P[key(g)]["metadata"]["loudness"] if P[key(g)]["metadata"]["loudness"] is not None else np.nan for g in gains], float)
F = np.array([[music_mean(g, f) for f in FEATS] for g in gains])


def neighbours(i, k=2):
    idx = [j for j in range(max(0, i - k), min(len(gains), i + k + 1)) if j != i]
    return idx


def lowpass(x):
    return sosfilt(butter(4, 400, "low", fs=SR, output="sos"), x)


def music_lag(a, b):
    n = min(len(a), len(b)) // 2 * 2
    fa, fb = lowpass(a[SR:n]), lowpass(b[SR:n])
    fa, fb = fa[: SR * 8], fb[: SR * 8]
    c = np.fft.irfft(np.fft.rfft(fa) * np.conj(np.fft.rfft(fb)), n=len(fa))
    c = np.concatenate([c[-1000:], c[:1001]])
    return int(np.argmax(c) - 1000)


x_ver = clip("moderate_hotrod")[: 10 * SR]
renders = {}
def r_music(g):
    if g not in renders:
        renders[g] = render(capture(g), x_ver, SR)
    return renders[g]


step = np.abs(np.diff(F, axis=0))
scale = np.maximum(np.median(step, axis=0), 0.15)
Z = F / scale

sets = lambda k: sorted({P[key(g)]["metadata"][k] for g in gains}, key=str)
meta_issues = {k: sets(k) for k in ("gear_make", "gear_model") if len(sets(k)) > 1}
sr_set = sorted({P[key(g)]["sample_rate"] for g in gains}, key=str)
in_set = sorted({P[key(g)]["input_level_dbu"] for g in gains}, key=lambda v: (v is None, v))
loud_off = rms - loud   # measured RMS at ref minus metadata loudness; should be roughly constant across a set
loud_ref = float(np.nanmedian(loud_off))

res = {"amp": amp, "set": {"n": len(gains), "median_click_onset": med_on, "sample_rates": sr_set, "input_level_dbu_values": in_set,
                            "metadata_inconsistent_fields": meta_issues, "metadata_loudness_minus_measured_median_db": loud_ref,
                            "repeat_captures": "none available: physical repeatability not measurable; model-determinism proxy only"},
       "captures": {}}

for i, g in enumerate(gains):
    ev, status, corr = [], "VALID", None
    p = P[key(g)]["probe"]
    # 1 timing
    d_on = onset[i] - med_on
    lvl_dep = onset[i] - onset_lo[i]
    if abs(d_on) > TIMING_TOL:
        cands = [k for k in range(len(gains)) if k != i]
        j = min(cands, key=lambda k: (abs(onset[k] - med_on) > TIMING_TOL, abs(gains[k] - g)))
        lag = music_lag(r_music(g), r_music(gains[j]))
        expected = float(onset[i] - onset[j])
        d_music = lag + (onset[j] - med_on)
        ev.append(f"timing: click onset {onset[i]:+.0f} vs set median {med_on:+.0f} ({d_on:+.0f} samples); music xcorr vs G{gains[j]:g} = {lag:+d} samples (click predicts {expected:+.0f}) -> music-derived offset {d_music:+.0f}")
        agree = abs(lag - expected) <= AGREE_TOL
        x2 = clip("clean_smooth")[: 10 * SR]
        lag_b = music_lag(render(capture(g), x2, SR), render(capture(gains[j]), x2, SR))
        repro = abs(lag_b - lag) <= VERIFY_TOL
        if agree and repro and abs(d_music) <= TIMING_TOL:
            ev.append(f"click onset flagged {d_on:+.0f} samples but the music is aligned to the set within {TIMING_TOL} samples (offset {d_music:+.0f}): false alarm from the click probe, no correction needed")
        elif agree and repro:
            sh = int(round(d_music))
            status = "CORRECTED"
            corr = {"type": "sample shift", "samples": -sh, "basis": "music cross-correlation offset",
                    "verified_by": f"click and music agree within {AGREE_TOL} samples; a second DI reproduces the lag within {VERIFY_TOL} samples ({lag_b:+d} vs {lag:+d})"}
            y = r_music(g)
            y2 = np.concatenate([y[sh:], np.zeros(sh, y.dtype)]) if sh > 0 else np.concatenate([np.zeros(-sh, y.dtype), y[:sh]])
            cd = D / "corrected"; cd.mkdir(exist_ok=True)
            sf.write(cd / f"G{g:g}_original_render.wav", y.astype(np.float32), SR)
            sf.write(cd / f"G{g:g}_corrected_render.wav", y2.astype(np.float32), SR)
            corr["after_correction_music_lag_vs_reference"] = music_lag(y2, r_music(gains[j]))
            corr["expected_after_correction"] = float(med_on - onset[j])
        else:
            status = "SUSPECT"
            ev.append(f"timing offset not confirmed (click/music agree: {agree}; second-DI reproducible: {repro})")
    if abs(lvl_dep) > LATDEP_TOL:
        if corr:
            ev.append(f"info: click 0.5 vs 0.05 onset differ by {lvl_dep:+.0f} samples (residual timing uncertainty after the verified correction)")
        else:
            ev.append(f"level-dependent latency: click 0.5 vs 0.05 onset differ by {lvl_dep:+.0f} samples")
            status = worse(status, "SUSPECT")
    # 2 clipping / dropouts / nan
    m0 = [p["music"][f"{n}@0"] for n in FIT_DIS]
    clipped = max(v["clipped_frac"] for v in m0)
    if clipped > 0:
        ev.append(f"output reaches full scale on {clipped*100:.3f}% of samples at native DI level")
    if any(v["nan"] for v in p["music"].values()):
        ev.append("NaN/Inf in output"); status = "INVALID"
    if max(v["longest_zero_run"] for v in m0) > SR // 20:
        ev.append(f"dropout: exact-zero run of {max(v['longest_zero_run'] for v in m0)} samples inside active playing")
        status = worse(status, "SUSPECT")
    # 3 noise / hum
    nb = neighbours(i)
    local = float(np.median(noise[nb]))
    if noise[i] > local + NOISE_SPIKE_DB and noise[i] > -55:
        ev.append(f"noise spike: silence renders at {noise[i]:.0f} dBFS vs {local:.0f} dBFS typical for neighbours")
        status = worse(status, "SUSPECT")
    elif noise[i] > NOISE_ABS_NOTE_DB:
        ev.append(f"high noise floor {noise[i]:.0f} dBFS (in line with neighbours; may be genuine amp hiss)")
    if abs(p["silence"]["dc"]) > 0.01:
        ev.append(f"DC offset {p['silence']['dc']:.3f} on silence")
        status = worse(status, "SUSPECT")
    # 4 level trend and metadata loudness
    if 0 < i < len(gains) - 1:
        tt = (gains[i] - gains[i - 1]) / (gains[i + 1] - gains[i - 1])
        interp = lambda arr: arr[i - 1] * (1 - tt) + arr[i + 1] * tt
        lvl_res = float(rms[i] - interp(rms))
        meta_res = float(loud[i] - interp(loud)) if np.isfinite(loud[[i - 1, i, i + 1]]).all() else float("nan")
        hi = [f"440@{lv:g}" for lv in (-30, -18, -6, 0)]
        tone_lv = lambda k: float(np.median([P[key(gains[k])]["probe"]["tones"][h]["out_rms_db"] for h in hi]))
        sine_res = float(tone_lv(i) - (tone_lv(i - 1) * (1 - tt) + tone_lv(i + 1) * tt))
        big = [abs(v) > LEVEL_TOL_DB for v in (lvl_res, sine_res)]
        if any(big):
            corro = np.isfinite(meta_res) and abs(meta_res) > LEVEL_TOL_DB and np.sign(meta_res) == np.sign(lvl_res)
            ev.append(f"level: music RMS {lvl_res:+.1f} dB, sustained-sine output {sine_res:+.1f} dB, metadata loudness {meta_res:+.1f} dB vs the interpolation of the neighbouring captures"
                      + ("; the same signed offset appears in the file's own metadata and at every sustained input level, consistent with a capture-chain level error (a constant drop at saturation while both neighbours are equal is not typical amp behaviour)" if corro
                         else "; real amps can have genuine level steps, so this needs investigation, not exclusion"))
            if corro:
                ev.append(f"suggested level correction (ESTIMATE, not applied): {-float(np.mean([lvl_res, sine_res, meta_res])):+.1f} dB")
                corr = corr or {}
                corr["suggested_level_correction_db_estimate"] = -float(np.mean([lvl_res, sine_res, meta_res]))
            status = worse(status, "SUSPECT")
    if np.isfinite(loud_off[i]) and abs(loud_off[i] - loud_ref) > LOUD_TOL_DB:
        ev.append(f"info: metadata loudness ({loud[i]:.1f}, measured on the training signal) vs RMS at the reference level ({rms[i]:.1f}) differ by {loud_off[i]-loud_ref:+.1f} dB vs the rest of the set (expected for low-gain, linear captures; not a fault by itself)")
    # 5 metadata consistency
    if P[key(g)]["sample_rate"] not in (None, 48000):
        ev.append(f"unusual sample rate {P[key(g)]['sample_rate']}"); status = worse(status, "SUSPECT")
    # 6 curve deviation (never INVALID; SUSPECT only, and only when strong)
    if 0 < i < len(gains) - 1:
        t = (gains[i] - gains[i - 1]) / (gains[i + 1] - gains[i - 1])
        pred = Z[i - 1] * (1 - t) + Z[i + 1] * t
        dev = float(np.sqrt(np.mean((Z[i] - pred) ** 2)))
        loc = float(np.linalg.norm(Z[i + 1] - Z[i - 1]) / np.sqrt(len(FEATS)) / 2)
        typical = float(np.median([np.linalg.norm(Z[k + 1] - Z[k - 1]) / np.sqrt(len(FEATS)) / 2 for k in range(1, len(gains) - 1)]))
        ratio = dev / max(loc, 0.5 * typical)
        if ratio > CURVE_RATIO:
            ev.append(f"tone/dynamics deviate from the neighbours' interpolation ({ratio:.1f}x local slope); may be genuine amp behaviour, not a fault")
            status = worse(status, "SUSPECT")
    # 7 repeatability proxy
    x = clip("clean_smooth")[: 5 * SR]
    y1 = render(capture(g), x, SR)
    y2 = render(capture(g), (x + 1e-5 * np.random.default_rng(0).standard_normal(len(x))).astype(np.float32), SR)
    esr = float(compute_esr_metrics(y2[SR // 2:], y1[SR // 2:])["raw_esr"])
    if esr > 1e-3:
        ev.append(f"model sensitivity: -100 dB dither changes the render by ESR {esr:.4f}")
    res["captures"][key(g)] = {"status": status, "evidence": ev, "correction": corr,
        "original": {"onset": onset[i], "noise_db": noise[i], "music_rms_db": rms[i], "metadata_loudness": None if np.isnan(loud[i]) else loud[i]},
        "corrected": ({"onset": med_on, "noise_db": noise[i], "music_rms_db": rms[i]} if corr else None), "dither_esr": esr}
    print(f"{amp} G{g:g}: {status}  {'; '.join(ev)[:150]}", flush=True)
(D / "audit.json").write_text(json.dumps(res, indent=1, default=float))

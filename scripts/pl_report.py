"""Render docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md (+ docs/phase4e/playability/*.png). Analysis only; no training. Findings text is read from docs/phase4e/playability_findings.md."""
import json, os, hashlib
os.environ["SINGLE_NAM_AMP"] = "vibrolux"
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pl_load as L
from pl_apparent import apparent
from pl_common import teacher, config, REF
import p4e_common as C
from hybrid.envelope import bounded_causal_envelope_db
from hybrid.multi_blend import chain_weights

ROOT = L.ROOT; PL = L.PL; DOC = ROOT / "docs"; IMG = DOC / "phase4e" / "playability"; IMG.mkdir(parents=True, exist_ok=True)
AMPS = {"jcm800": "Marshall JCM800 2203 (High)", "vibrolux": "Fender Super-Sonic Vibrolux"}
OFFS = L.OFFS; CASES = {a: L.load_cases(a) for a in AMPS}
PATHV = json.loads((PL / "path_verification.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def student_err(cs, g, o, keys, f):
    v = []
    for k in keys:
        v.append(np.mean([c["student"][k][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o]))
    return v
def teacher_err(cs, g, o, t, f): return float(np.mean([c["teacher"][t][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o]))
def eq_abs(c_feats, real):
    return np.mean([abs(c_feats[b] - real[b]) for b in C.EQ])
def student_eq(cs, g, o, keys):
    return [np.mean([eq_abs(c["student"][k], c["real"]) for c in cs if c["g"] == g and c["offset"] == o]) for k in keys]
def teacher_eq(cs, g, o, t): return float(np.mean([eq_abs(c["teacher"][t], c["real"]) for c in cs if c["g"] == g and c["offset"] == o]))
FAM = {"B": ["B_s0", "B_s1"], "A": ["A_s0", "A_s1"], "C3": ["v3_C3"]}
def cell(vals, fmt="{:+.1f}"):
    m = np.mean(vals); return fmt.format(m) if len(vals) == 1 else f"{fmt.format(m)} ({min(vals):+.1f}..{max(vals):+.1f})".replace("(+", "(+").replace("..+", "..")

out = ["# Fixed-virtual-gain playability of the existing continuous-gain NAMs\n",
"**Scope.** Diagnostic of the EXISTING Phase 4E models; nothing was trained, no source captures or anchors changed, no Phase 4E artifact modified (new files only: `work/p4e/playability/`, `docs/phase4e/playability/`, `scripts/pl_*.py`). Human listening remains **pending**.\n",
"**Definitions.** *Virtual gain* = the ordinary NAM player's Input-gain setting. *Musical input* = the DI waveform before that control (picking, guitar volume, pickup). The NAM sees only their product (both are dB offsets on the same waveform). Playback path: musical DI -> Input gain -> ONE standard NAM -> Output gain.\n"]

# ------------------------------------------------------------------ 1 review
pv = PATHV["envelope_shift_vs_gain"]; tr = PATHV["teacher_reconstruction_B"]; ev = PATHV["evaluation_path"]
out += ["## 1. Implementation review (verified from the code and data, not from comments)\n",
"Reviewed: `hybrid/multi_blend.py`, `hybrid/envelope.py`, `scripts/cg_build.py`, `scripts/p4e_build.py`, `scripts/single_nam_train.py`, `scripts/p4_model_sweep.py`, `scripts/p4e_common.py`, `scripts/p4e_real.py`, `scripts/p4e_eval.py`, `docs/CONTINUOUS_GAIN_PHASE4E_RESULTS.md`, `docs/phase4e/manifest_frozen.json`. Numerical checks: `scripts/pl_verify_path.py` -> `work/p4e/playability/path_verification.json`.\n",
"1. **Which signal drives the envelope.** `p4e_build.segment(x, chain)` computes `env = bounded_causal_envelope_db(x, SR)` on `x`, the very waveform written to `input.wav`, i.e. the waveform that reaches the NAM (DI after the level-offset augmentation). It is a causal, bounded-history (about 80 ms: 20 ms RMS, 5 ms attack average, 55 ms release window) envelope in dBFS. It is NOT computed from either amp's output. Verification: my analysis-only teacher renderer reproduces the frozen Phase 4E training targets for both configurations to " + f"{tr['max_abs_diff_vs_frozen_target']:.1e} (12 s compared, target RMS {tr['rms_target']:.3f}), so the signal path below is the builder's path.\n",
"2. **Effect of the player's Input gain on the envelope.** A gain of g dB shifts the envelope by exactly g dB (median shift and worst deviation from g over active samples: " + "; ".join(f"{k}: {v['median_shift_db']:+.3f} dB / {v['max_abs_deviation_from_gain_db']:.1e} dB" for k, v in pv.items()) + "). Input gain therefore moves the envelope up or down the anchor-level scale one-for-one.\n",
"3. **Effect of musical-input level at fixed Input gain.** Identical to item 2: a louder performance is the same dB offset on the same waveform (envelope difference between 'offset -6 then gain -8.4' and 'offset -14.4': " + f"{PATHV['player_gain_and_musical_offset_commute']['max_abs_envelope_difference_db (offset -6 then gain -8.4 vs offset -14.4)']:.1e} dB). **By construction the teacher (and so any student trained on it) cannot distinguish virtual gain from musical intensity: the blend depends only on their sum.**\n",
"4. **How blend weights are applied.** `hybrid.multi_blend.chain_weights(env, levels_db)` gives, per sample, weight only to the two anchors whose level brackets the envelope, with a smoothstep across each gap (`smoothstep_curve`); the envelope is clipped to the outermost anchors. Capture k is rendered from `x * db(REF - L_k)` (`GainChain.input_scale_db`), i.e. the input rescaled so that when the envelope sits at the anchor level `L_k = anchor + REF` capture k is being driven at the -30 dBFS reference playing level; `multi_blend` then sums `w_k(t) * render_k(t)` in linear amplitude, per sample. The whole result is scaled by one fixed peak-ceiling constant `c` (1.0 for the Vibrolux, 0.581 for the JCM800).\n",
"5. **Does the evaluation reproduce the ordinary player's path?** Yes. `p4e_eval.py` feeds the model `DI * db(offset + T)` and divides the output by the single global constant `c`; the real reference gets `DI * db(offset)` only (Phase 4A alignment applied via `p4e_real.py`). Numerically `(DI*offset)*gain` and `DI*(offset+gain)` differ by " + f"{ev['model input = (DI*offset)*gain vs DI*(offset+gain): max abs diff']:.1e} (float precision). No per-gain or per-level output correction is applied. Caveats: the DIs are normalised guitar recordings whose median level is about -29 dBFS (reference level -30), and the official plugin's nominal Input range (about -20 dB minimum) is NOT applied here: the full intended mapping is used through NAMCore, including -22 dB.\n",
"**Consequence for the hypothesis.** Because the anchors sit at input levels, even at nominal playing the envelope of a real performance spans roughly 10 dB between its 10th and 90th percentiles (`clean_mayer` 11.2, `moderate_brit` 10.6, `bass_rollin` 5.0 dB block-level spread), so the teacher is a time-varying mixture of neighbouring captures around the selected virtual gain, and shifts systematically when the whole performance is louder or softer (section 4C).\n"]

# ------------------------------------------------------------------ 3 design
out += ["## 2. Test design\n",
"**Models (existing exports, unchanged):** Phase 4E B seeds 0 and 1 (primary), Phase 4E A seeds 0 and 1, and v3 C3 (G1, G5, G10) for each amp. Files and hashes: " + "; ".join(f"`{C.model_path(a, k)[0].name}` sha256 {sha(C.model_path(a, k)[0])}" for a in AMPS for k in ("B_s0", "B_s1", "A_s0", "A_s1", "v3_C3")) + ".\n" if False else "**Models (existing exports, unchanged):** Phase 4E B seeds 0 and 1 (primary), Phase 4E A seeds 0 and 1, and v3 C3 (G1, G5, G10) for each amp; identities and hashes are in `docs/phase4e/manifest_frozen.json` and `docs/phase4e/models/`.\n",
"**Virtual-gain settings (intended mappings, dB):** B uses the response-distance mapping, A and v3 C3 the fixed rule `-22+4(N-1)`. JCM800 positions: G1 (-22.0), G2 (-10.5), G3, G4 (-2.6), G5, G6.5 (genuine half-step), G8, G9, G10 (+14.0); anchors are G1, G2, G4, G10, so G3, G5, G6.5, G8, G9 are omitted from training. Vibrolux positions: G1, G2, G3, G4, G5, G7, G8, G10; anchors are G1, G2, G3, G4, G7, G10 (-22.0, -17.4, -8.4, -3.1, +5.3, +14.0), so G5 and G8 are omitted.\n",
"**Musical input:** held-out DIs `moderate_brit`, `clean_mayer`, `bass_rollin` (first 15 s, never used in training or fitting), each at -12, -6, 0, +6 dB. At every case the Input gain is fixed at that position's intended value, the reference is the SAME physical capture, and no compensation is applied. Errors are model minus real (signed) unless stated, mean over the three DIs; seed pairs are shown as `mean (min..max)`.\n",
"**References:** the validated physical captures with Phase 4A alignment (no correction was needed for any position used). No interpolated references. **Within-recording soft/hard:** the DIs contain enough natural dynamics for a block-level comparison (250 ms blocks; 10th-90th percentile spread 5 to 11 dB), reported for output level only (section 4B); other measures are global per case. Files: `scripts/pl_common.py`, `pl_cases.py`, `pl_apparent.py`, `pl_report.py`; raw data `work/p4e/playability/cases_*.json`.\n"]

def err_table(amp, f, title, fmt="{:+.1f}", eqmode=False):
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs}); rows = [f"**{title}** (model - real; columns = musical input offset in dB)\n",
        "| Position | B -12 | B -6 | B 0 | B +6 | A -12 | A -6 | A 0 | A +6 | v3 C3 -12 | -6 | 0 | +6 |", "|---|" + "---:|" * 12]
    for g in gains:
        cells = []
        for fam in ("B", "A", "C3"):
            for o in OFFS:
                cells.append(cell(list(map(float, student_eq(cs, g, o, FAM[fam]))) if eqmode else list(map(float, student_err(cs, g, o, FAM[fam], f))), fmt))
        rows.append(f"| G{g:g} | " + " | ".join(cells) + " |")
    return rows + [""]

# ------------------------------------------------------------------ 4B primary
out += ["## 3. Outcome B: fixed-virtual-gain playability (primary test)\n",
"With the Input gain fixed at each position's intended value, the same held-out DI is played quieter or louder and compared with the SAME physical capture at the SAME musical level. A real amp also saturates and compresses more when played harder; the failure is the DIFFERENCE from the real amp at the same setting.\n"]
for amp, nm in AMPS.items():
    out += [f"### {nm}\n"]
    out += err_table(amp, "rms_db", "Native output level error (dB)")
    out += err_table(amp, None, "Spectral balance: mean absolute EQ-band error over six bands (dB)", fmt="{:.2f}", eqmode=True)
    out += err_table(amp, "hf3k_db", "HF >3 kHz energy error (dB; negative = darker than the real amp)")
    out += err_table(amp, "tilt_db", "Spectral tilt error (dB)")
    out += err_table(amp, "crest_db", "Crest factor error (dB)")
    out += err_table(amp, "dyn_range_db", "Dynamic-range error (dB)")
    out += err_table(amp, "rise_p95_db", "Transient rise (p95 of positive 5 ms envelope steps) error (dB)", fmt="{:+.2f}")
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs})
    out += ["**Output-level change for +18 dB more musical input (level at +6 minus level at -12; real amp is the target):**\n", "| Position | real | teacher B | student B s0 | B s1 | A s0 | A s1 | v3 C3 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    dl = lambda sel, f_: np.mean([c[f_[0]][f_[1]]["rms_db"] if len(f_) == 2 else c[f_[0]]["rms_db"] for c in sel])
    for g in gains:
        def d(get):
            hi = np.mean([get(c) for c in cs if c["g"] == g and c["offset"] == 6.0]); lo = np.mean([get(c) for c in cs if c["g"] == g and c["offset"] == -12.0]); return hi - lo
        out.append(f"| G{g:g} | {d(lambda c: c['real']['rms_db']):.1f} | {d(lambda c: c['teacher']['B']['rms_db']):.1f} | " + " | ".join(f"{d(lambda c, k=k: c['student'][k]['rms_db']):.1f}" for k in ("B_s0", "B_s1", "A_s0", "A_s1", "v3_C3")) + " |")
    out.append("")
    # sine probes from Phase 4E eval (reuse)
    real = json.loads((ROOT / "work" / "p4e" / amp / "real_ref.json").read_text())
    def evj(k): return json.loads((ROOT / "work" / "p4e" / amp / f"eval_{k}.json").read_text())
    E = {k: evj(k) for k in ("B_s0", "B_s1", "A_s0", "A_s1", "v3_C3")}
    def tones(k): r = E[k]["results"]; return (r["intended"] if k.startswith("B") else r["fixed"])["tones"]
    out += ["**Saturation and harmonics from the existing sine-probe path (110/440 Hz sine at the four musical-input levels; THD error in dB, floor -80 dB; B mean of seeds; no music-derived THD is claimed):**\n", "| Position | THD err @-42 | @-30 | @-18 | @-6 | H2/H3 err @-18 | IO slope err (-54..-30 / -30..0) | A THD mean | v3 C3 THD mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fl = lambda v: max(v, -80.0)
    for g in gains:
        def terr(k, lv): T_ = tones(k); return fl(T_[f"{g:g}|440|{lv}"]["thd_db"]) - fl(real["tones"][f"{g:g}|440|{lv}"]["thd_db"])
        b = [np.mean([terr(k, lv) for k in ("B_s0", "B_s1")]) for lv in (-42, -30, -18, -6)]
        h = np.mean([np.mean([abs(fl(tones(k)[f"{g:g}|440|-18"][x]) - fl(real["tones"][f"{g:g}|440|-18"][x])) for x in ("h2_db", "h3_db")]) for k in ("B_s0", "B_s1")])
        io = lambda t: (t[f"{g:g}|440|-30"]["out_rms_db"] - t[f"{g:g}|440|-54"]["out_rms_db"] - 24, t[f"{g:g}|440|0"]["out_rms_db"] - t[f"{g:g}|440|-30"]["out_rms_db"] - 30)
        ior = io(real["tones"]); iob = np.mean([np.array(io(tones(k))) - np.array(ior) for k in ("B_s0", "B_s1")], axis=0)
        am = np.mean([np.mean([terr(k, lv) for lv in (-42, -30, -18, -6)]) for k in ("A_s0", "A_s1")]); cm = np.mean([terr("v3_C3", lv) for lv in (-42, -30, -18, -6)])
        out.append(f"| G{g:g} | " + " | ".join(f"{x:+.1f}" for x in b) + f" | {h:.1f} | {iob[0]:+.1f} / {iob[1]:+.1f} | {am:+.1f} | {cm:+.1f} |")
    out.append("")
    # within-recording
    out += ["**Within-recording soft versus hard playing (native DI dynamics, output level only):** the model-minus-real block-level error in the softest, middle and loudest third of each recording's blocks (250 ms), and the error slope in dB per dB of musical-input block level (0 = the model follows the real amp's level response within the performance). Mean of three DIs.\n", "| Position | teacher B slope [soft/mid/hard] | student B s0 | A s0 | v3 C3 |", "|---|---|---|---|---|"]
    csb = [c for c in cs if c["offset"] == 0.0]
    for g in gains:
        row = []
        for who in ("T_B", "B_s0", "A_s0", "v3_C3"):
            ter, sl = [], []
            for c in csb:
                if c["g"] != g: continue
                b_ = np.array(c["blocks"]["di_in_db"]); r_ = np.array(c["blocks"]["real"]); m_ = np.array(c["blocks"][who]); mk = b_ > b_.max() - 45
                e = (m_ - r_)[mk]; x = b_[mk]; q = np.percentile(x, [33.3, 66.7])
                ter.append([e[x <= q[0]].mean(), e[(x > q[0]) & (x <= q[1])].mean(), e[x > q[1]].mean()]); sl.append(np.polyfit(x, e, 1)[0])
            t = np.mean(ter, axis=0); row.append(f"{np.mean(sl):+.2f} [{t[0]:+.1f}/{t[1]:+.1f}/{t[2]:+.1f}]")
        out.append(f"| G{g:g} | " + " | ".join(row) + " |")
    out.append("")

# ------------------------------------------------------------------ mismatch attribution
THR = {"level": ("rms_db", 1.0), "HF": ("hf3k_db", 1.0), "crest": ("crest_db", 1.0), "dyn range": ("dyn_range_db", 1.5)}
rowsm = []
for amp, nm in AMPS.items():
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs})
    for g in gains:
        for lab, (f, thr) in THR.items():
            st = [np.mean([np.mean(student_err(cs, g, o, FAM["B"], f)) for o in OFFS]) for _ in (0,)][0]
            se = {o: float(np.mean(student_err(cs, g, o, FAM["B"], f))) for o in OFFS}; te = {o: teacher_err(cs, g, o, "B", f) for o in OFFS}
            fixed_s, fixed_t = np.mean(list(se.values())), np.mean(list(te.values())); swing_s, swing_t = se[6.0] - se[-12.0], te[6.0] - te[-12.0]
            worst_o = max(OFFS, key=lambda o: abs(se[o]))
            if max(abs(fixed_s), abs(swing_s)) >= thr:
                kind = "dynamic (grows with musical input)" if abs(swing_s) >= max(thr, abs(fixed_s)) else "fixed offset (present at every input level)"
                if kind.startswith("dynamic"): who = "already in the teacher" if abs(swing_t) >= 0.6 * abs(swing_s) else ("student only" if abs(swing_s - swing_t) >= 0.6 * abs(swing_s) else "both")
                else: who = "already in the teacher" if abs(fixed_t) >= 0.6 * abs(fixed_s) else ("student only" if abs(fixed_s - fixed_t) >= 0.6 * abs(fixed_s) else "both")
                rowsm.append((max(abs(fixed_s), abs(swing_s)) / thr, f"| {nm.split(' (')[0]} | G{g:g} | {lab} | {fixed_s:+.1f} | {swing_s:+.1f} | {se[worst_o]:+.1f} at {worst_o:+g} dB | {fixed_t:+.1f} / {swing_t:+.1f} | {kind}; {who} |"))
cnt = {"already in the teacher": 0, "student only": 0, "both": 0}; cnt_amp = {}
for _, r in rowsm:
    for k_ in cnt:
        if r.rstrip(" |").endswith("; " + k_): cnt[k_] += 1
    a_ = "Vibrolux" if "Vibrolux" in r else "JCM800"; cnt_amp[a_] = cnt_amp.get(a_, 0) + 1
out += [f"**Flagged amp/position/measure combinations: {len(rowsm)} ({cnt_amp.get('Vibrolux', 0)} Vibrolux, {cnt_amp.get('JCM800', 0)} JCM800). Attribution: already in the teacher {cnt['already in the teacher']}, student only {cnt['student only']}, both {cnt['both']}.**\n"]
out += ["### Where the important mismatches are (student B, mean of seeds; working flags: level 1.0, HF 1.0, crest 1.0, dynamic range 1.5 dB)\n",
        "*fixed* = mean signed error over the four musical-input levels; *swing* = error at +6 minus error at -12 (change over 18 dB of playing intensity). Attribution compares the student's fixed/swing with the teacher's (a measured decomposition, not proof of cause).\n",
        "| Amp | Virtual gain | Measure | fixed (dB) | swing (dB) | worst single case | teacher fixed / swing | type; where present |", "|---|---|---|---:|---:|---|---|---|"] + [r for _, r in sorted(rowsm, reverse=True)] + [""]

# ------------------------------------------------------------------ 4C teacher weights
out += ["## 4. Outcome C: training-target behaviour (the envelope-driven teacher)\n",
"For the same DI, the fixed intended Input gain and each musical-input offset, `hybrid.multi_blend.chain_weights` on the causal envelope of the waveform reaching the NAM. Statistics use ACTIVE segments only (envelope within 30 dB of its maximum) so silence and tails do not dominate. This describes the TRAINING TARGET, not an internal state of the exported NAM.\n"]
for amp, nm in AMPS.items():
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs}); an = cs[0]["weights"]["B"]["gains"]
    out += [f"### {nm}: teacher B (anchors G{', G'.join(f'{g:g}' for g in an)})\n", "Mean blend weight on the intended position's nearest anchor, and the weight-averaged (centre-of-mass) physical position, by musical-input offset. A perfectly fixed-gain target would keep the centre of mass at the intended position at every offset.\n", "| Virtual gain | nearest-anchor weight -12 / -6 / 0 / +6 | centre-of-mass position -12 / -6 / 0 / +6 | swing over 18 dB (positions) |", "|---|---|---|---:|"]
    for g in gains:
        nw, cm = [], []
        for o in OFFS:
            sel = [c for c in cs if c["g"] == g and c["offset"] == o]
            w = np.mean([c["weights"]["B"]["mean_weight_active"] for c in sel], axis=0); nw.append(np.mean([c["weights"]["B"]["nearest_anchor_weight_active"] for c in sel])); cm.append(float(np.dot(w, an)))
        out.append(f"| G{g:g} | " + " / ".join(f"{x:.2f}" for x in nw) + " | " + " / ".join(f"{x:.1f}" for x in cm) + f" | {cm[-1]-cm[0]:.1f} |")
    out.append("")
    # which neighbours contribute
    out += ["Which neighbouring captures carry the weight as musical input rises (mean weights on each anchor, teacher B):\n", "| Virtual gain | offset | " + " | ".join(f"G{g:g}" for g in an) + " |", "|---|---:|" + "---:|" * len(an)]
    for g in (gains[1], gains[len(gains) // 2], gains[-2]):
        for o in (-12.0, 0.0, 6.0):
            sel = [c for c in cs if c["g"] == g and c["offset"] == o]; w = np.mean([c["weights"]["B"]["mean_weight_active"] for c in sel], axis=0)
            out.append(f"| G{g:g} | {o:+g} | " + " | ".join(f"{x:.2f}" for x in w) + " |")
    out.append("")
    ap, agains = apparent(amp)
    out += ["**Apparent physical position** (supporting evidence; nearest real capture in tone + saturation + dynamics feature space at the same musical input, level excluded; median of three DIs; coarse where neighbouring captures sound alike): intended position -> apparent position at -12 / -6 / 0 / +6.\n", "| Virtual gain | teacher B | student B s0 | student B s1 | student A s0 | v3 C3 |", "|---|---|---|---|---|---|"]
    for g in gains:
        out.append(f"| G{g:g} | " + " | ".join("/".join(f"{np.median([x[0] for x in ap[(g, o, who)]]):.1f}" for o in OFFS) for who in ("teacher_B", "B_s0", "B_s1", "A_s0", "v3_C3")) + " |")
    out.append("")

# weight plot
def weights_plot(amp, g, di, fname):
    gains_, levels_, c_ = config(amp, "B_s0"); T_ = C.intended_T(amp, "B", [g])[0]; os.environ["SINGLE_NAM_AMP"] = amp
    fig, axs = plt.subplots(2, 3, figsize=(17, 7)); x = C.clip(di)[2 * C.SR: 10 * C.SR]; t = np.arange(len(x)) / C.SR
    for j, o in enumerate((-12.0, 0.0, 6.0)):
        x_in = (x * C.db(o) * C.db(T_)).astype(np.float32); env = bounded_causal_envelope_db(x_in, C.SR); w = chain_weights(env, tuple(levels_))
        ax = axs[0, j]; ax.plot(t, env, "k", lw=.8, label="envelope of the waveform reaching the NAM")
        for gg, lv in zip(gains_, levels_): ax.axhline(lv, ls="--", lw=.8, color="tab:red" if gg == g else "gray", alpha=.8); ax.text(t[-1], lv, f" G{gg:g}", fontsize=7, va="center")
        ax.set_title(f"musical input {o:+g} dB, virtual gain G{g:g} ({T_:+.1f} dB)", fontsize=9); ax.set_ylim(-75, -5); ax.set_ylabel("dBFS"); ax.legend(fontsize=6)
        ax = axs[1, j]; ax.stackplot(t, [w[k] for k in range(len(gains_))], labels=[f"G{gg:g}" + (" (intended)" if gg == g else "") for gg in gains_], alpha=.85); ax.set_ylim(0, 1); ax.set_ylabel("teacher weight"); ax.set_xlabel("s"); ax.legend(fontsize=6, ncol=3, loc="upper right")
    fig.suptitle(f"{AMPS[amp]}: teacher blend weights at FIXED virtual gain G{g:g}, {di}, as the musical input changes (physical reference for the fixed setting: G{g:g})", fontsize=10); fig.tight_layout(); fig.savefig(IMG / fname, dpi=100); plt.close(fig)
weights_plot("vibrolux", 3.0, "clean_mayer", "weights_vibrolux_G3.png"); weights_plot("vibrolux", 7.0, "moderate_brit", "weights_vibrolux_G7.png"); weights_plot("jcm800", 4.0, "moderate_brit", "weights_jcm800_G4.png")
out += ["![Vibrolux G3 weights](phase4e/playability/weights_vibrolux_G3.png)\n", "![Vibrolux G7 weights](phase4e/playability/weights_vibrolux_G7.png)\n", "![JCM800 G4 weights](phase4e/playability/weights_jcm800_G4.png)\n"]

# centre-of-mass plot + error swing plot
fig, axs = plt.subplots(2, 4, figsize=(19, 8))
for i, (amp, nm) in enumerate(AMPS.items()):
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs}); an = cs[0]["weights"]["B"]["gains"]; cmap = plt.cm.viridis(np.linspace(0, 1, len(gains)))
    ax = axs[i, 0]
    for gg, col in zip(gains, cmap):
        cm = [float(np.dot(np.mean([c["weights"]["B"]["mean_weight_active"] for c in cs if c["g"] == gg and c["offset"] == o], axis=0), an)) for o in OFFS]; ax.plot(OFFS, cm, "-o", ms=3, color=col, label=f"G{gg:g}"); ax.axhline(gg, color=col, lw=.4, ls=":")
    ax.set_title(f"{nm}: teacher B effective position vs musical input (dotted = intended)", fontsize=8); ax.set_xlabel("musical input offset dB"); ax.set_ylabel("centre-of-mass position"); ax.legend(fontsize=5, ncol=2)
    for ax, (f, ttl) in zip(axs[i, 1:], (("rms_db", "level error (student B mean, dB)"), ("hf3k_db", "HF error (student B mean, dB)"), ("dyn_range_db", "dynamic-range error (student B mean, dB)"))):
        for gg, col in zip(gains, cmap): ax.plot(OFFS, [np.mean(student_err(cs, gg, o, FAM["B"], f)) for o in OFFS], "-o", ms=3, color=col, label=f"G{gg:g}")
        ax.axhline(0, color="k", lw=.5); ax.set_title(f"{nm}: {ttl}", fontsize=8); ax.set_xlabel("musical input offset dB"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(IMG / "drift_and_error_vs_musical_input.png", dpi=100); plt.close(fig)
out += ["![drift and error vs musical input](phase4e/playability/drift_and_error_vs_musical_input.png)\n"]

# dense anchors (teacher only)
import glob as _g
out += ["### Would more anchors reduce the teacher's drift? (teacher-only test with the existing v3 C10 chain; no training)\n",
        "The v3 C10 teacher uses all ten physical captures as anchors at the fixed rule (-22 + 4(N-1) dB), so every tested position is an exact anchor. Same cases, same DIs and offsets; swing = error at +6 minus error at -12 (teacher minus real).\n",
        "| Amp | measure | mean |swing| teacher B (6 or 4 anchors) | mean |swing| teacher C10 (10 anchors) | C10 centre-of-mass position at -12 / 0 / +6 for a mid position |", "|---|---|---:|---:|---|"]
for amp, nm in AMPS.items():
    dense = []
    for fpath in sorted(_g.glob(str(PL / f"dense_{amp}_*.json"))):
        j = json.loads(Path(fpath).read_text()); dense += [dict(c, g=j["g"]) for c in j["cases"]]
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs}); mid = 5.0 if 5.0 in gains else gains[len(gains) // 2]
    for lab, f in (("level", "rms_db"), ("HF", "hf3k_db"), ("crest", "crest_db"), ("dynamic range", "dyn_range_db")):
        sB, sC = [], []
        for g in gains:
            eB = lambda o: np.mean([c["teacher"]["B"][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o]); eC = lambda o: np.mean([c["teacher_C10"][f] - c["real"][f] for c in dense if c["g"] == g and c["offset"] == o])
            sB.append(abs(eB(6.0) - eB(-12.0))); sC.append(abs(eC(6.0) - eC(-12.0)))
        cmC = [float(np.dot(np.mean([c["mean_weight_active"] for c in dense if c["g"] == mid and c["offset"] == o], axis=0), list(range(1, 11)))) for o in (-12.0, 0.0, 6.0)]
        out.append(f"| {nm.split(' (')[0]} | {lab} | {np.mean(sB):.2f} | {np.mean(sC):.2f} | G{mid:g}: " + " / ".join(f"{v:.1f}" for v in cmC) + " |")
out.append("")

# ------------------------------------------------------------------ 4D teacher vs student vs reference
out += ["## 5. Outcome D: teacher versus student versus reference\n",
"Signal paths (identical for the three): musical input `x0 = DI * db(offset)`; player Input gain `x_in = x0 * db(T)`; REFERENCE = real capture on `x0` (no player gain, Phase 4A alignment); TEACHER = `teacher(x_in)` (section 1, raw target units); STUDENT = exported NAM on `x_in`, divided by the single global constant `c`. The player gain is applied once, to `x_in`, and the builder's own rescaling (`REF - L_k`) is applied inside the teacher only. Level-matched ESR compares waveforms after matching RMS to the second argument.\n"]
for amp, nm in AMPS.items():
    cs = CASES[amp]; gains = sorted({c["g"] for c in cs})
    out += [f"### {nm} (configuration B)\n", "| Position | offset | teacher-real level / HF / crest / dyn | student-real level / HF / crest / dyn | student-teacher level / HF / crest / dyn | ESR teacher-real | ESR student-real | ESR student-teacher |", "|---|---:|---|---|---|---:|---:|---:|"]
    fs = ("rms_db", "hf3k_db", "crest_db", "dyn_range_db")
    for g in gains:
        for o in (-12.0, 0.0, 6.0):
            sel = [c for c in cs if c["g"] == g and c["offset"] == o]
            tr_ = [np.mean([c["teacher"]["B"][f] - c["real"][f] for c in sel]) for f in fs]
            st_ = [np.mean([np.mean([c["student"][k][f] - c["real"][f] for k in FAM["B"]]) for c in sel]) for f in fs]
            sm = [np.mean([np.mean([c["student"][k][f] - c["teacher"]["B"][f] for k in FAM["B"]]) for c in sel]) for f in fs]
            e1 = np.mean([c["esr"]["teacher_vs_real_B"] for c in sel]); e2 = np.mean([np.mean([c["esr"][f"{k}_vs_real"] for k in FAM["B"]]) for c in sel]); e3 = np.mean([np.mean([c["esr"][f"{k}_vs_teacher"] for k in FAM["B"]]) for c in sel])
            out.append(f"| G{g:g} | {o:+g} | " + " / ".join(f"{v:+.1f}" for v in tr_) + " | " + " / ".join(f"{v:+.1f}" for v in st_) + " | " + " / ".join(f"{v:+.1f}" for v in sm) + f" | {e1:.3f} | {e2:.3f} | {e3:.3f} |")
    out.append("")

# ------------------------------------------------------------------ 4A virtual gain progression
out += ["## 6. Outcome A: virtual-gain progression (secondary test; existing results reused)\n",
"With musical input fixed (native DI level), the Phase 4E evaluation already sweeps only the intended Input gain across every physical position for all three DIs. Its numbers (position-by-position errors against the real captures) are in `docs/CONTINUOUS_GAIN_PHASE4E_RESULTS.md` and its curves in `docs/phase4e/curves_*.png`; the rows at offset 0 in the tables above are the same cases. In addition a continuous sweep of the Input gain through each exported NAM (Phase 4 sweep data, 3 dB steps, four fit DIs and three held-out DIs, offset 0) shows the progression and the real captures at their intended positions:\n"]
fig, axs = plt.subplots(2, 4, figsize=(19, 8))
for i, amp in enumerate(AMPS):
    real = json.loads((ROOT / "work" / "p4e" / amp / "real_ref.json").read_text()); gains_all = real["gains"]
    for ax, (f, ttl) in zip(axs[i], (("rms_db", "output level dBFS"), ("hf3k_db", "HF >3 kHz"), ("crest_db", "crest dB"), ("dyn_range_db", "dynamic range dB"))):
        for k, col in (("B_s0", "tab:red"), ("B_s1", "tab:orange"), ("A_s0", "tab:blue"), ("A_s1", "tab:cyan")):
            sw = json.loads((ROOT / "work" / "p4e" / amp / f"sweep_{k}.json").read_text()); T = np.array(sw["T"]); y = np.mean([[r[f] for r in sw["music"][d]] for d in C.HELD], axis=0); ax.plot(T, y, "-", color=col, lw=1, label=f"{k} continuous sweep")
        Tb = C.intended_T(amp, "B", gains_all); ax.plot(Tb, [np.mean([real["music"][f"{g:g}|{d}|0"][f] for d in C.HELD]) for g in gains_all], "ks", ms=4, label="real captures at B intended gain")
        Ta = [C.fixed_T(g) for g in gains_all]; ax.plot(Ta, [np.mean([real["music"][f"{g:g}|{d}|0"][f] for d in C.HELD]) for g in gains_all], "x", color="gray", ms=4, label="real captures at fixed-rule gain")
        ax.set_xlim(-30, 24); ax.set_title(f"{AMPS[amp]}: {ttl} vs player Input gain (native DI)", fontsize=8); ax.set_xlabel("Input gain dB"); ax.grid(alpha=.25); ax.legend(fontsize=5)
fig.tight_layout(); fig.savefig(IMG / "virtual_gain_sweep.png", dpi=100); plt.close(fig)
out += ["![virtual gain sweep](phase4e/playability/virtual_gain_sweep.png)\n"]
# progression check against the REAL amp's own progression (not an assumed direction)
mono = ["| Amp | model | feature | steps between adjacent physical positions where the model's change at the intended Input gains has the opposite sign to the real amp's change (real step >= 0.5 dB) | of |", "|---|---|---|---:|---:|"]
for amp in AMPS:
    real = json.loads((ROOT / "work" / "p4e" / amp / "real_ref.json").read_text()); gs = real["gains"]
    for k in ("B_s0", "B_s1", "A_s0", "A_s1"):
        sw = json.loads((ROOT / "work" / "p4e" / amp / f"sweep_{k}.json").read_text()); T = np.array(sw["T"]); Tg = C.intended_T(amp, "B", gs) if k.startswith("B") else [C.fixed_T(g) for g in gs]
        for f in ("rms_db", "hf3k_db", "crest_db", "dyn_range_db"):
            m = np.mean([[r[f] for r in sw["music"][d]] for d in C.HELD], axis=0); mv = np.interp(Tg, T, m); rv = np.array([np.mean([real["music"][f"{g:g}|{d}|0"][f] for d in C.HELD]) for g in gs])
            dm, dr = np.diff(mv), np.diff(rv); sel = np.abs(dr) >= 0.5
            mono.append(f"| {amp} | {k} | {f} | {int(((np.sign(dm) != np.sign(dr)) & sel & (np.abs(dm) >= 0.3)).sum())} | {int(sel.sum())} |")
out += ["Does the progression turn the same way as the real amp's? Continuous sweep through each exported NAM, sampled at each physical position's intended Input gain, compared step by step with the real captures (native DI, held-out DIs, offset 0; a model step smaller than 0.3 dB is not counted as a reversal):\n"] + mono + [""]

fp = DOC / "phase4e" / "playability_findings.md"
out.append(fp.read_text() if fp.exists() else "## 7. Findings\n\n(pending)\n")
(DOC / "CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md").write_text("\n".join(out)); print("written", len(out), "lines")

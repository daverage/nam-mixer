"""Tonal refinement, step 2: signed tonal-error analysis, conservative EQ candidates (fit on FIT DIs at the FC anchor positions only), validation on held-out DIs
at ALL positions (incl. intermediate positions not used for fitting). No training. Usage: tr_analyze.py -> work/tr/analysis.json, docs/tonal/*.png"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import group_delay, sosfreqz
from tr_common import *
import tr_eq as EQ
DOC = REPO / "docs" / "tonal"; DOC.mkdir(parents=True, exist_ok=True)
ANCH = {"jcm800": [1, 2, 4, 10], "vibrolux": [1, 2, 3, 4, 7, 10]}; SEEDS = ("FC_s0", "FC_s1"); NAME = {"jcm800": "Marshall JCM800 2203 High", "vibrolux": "Fender Super-Sonic Vibrolux"}
FITB = (FC_BANDS >= 80) & (FC_BANDS <= 12700)
# ---- limits (documented in the report)
LIM = {"B": {"gain": 3.0, "note": "each band +-3 dB"}, "C": {"gain": 4.0, "note": "each band +-4 dB, high-gain-weighted fit"}}
def mean_shape(S, gains, dis, offs, cls="all", H=None):
    v = [shape_db(S.P(g, d, o, m, cls), S.P(g, d, o, "real", cls), H)[0] for g in gains for d in dis for o in offs for m in SEEDS if S.nframes(g, d, o, cls) > 30]
    return np.mean(v, axis=0) if v else np.full(len(FC_BANDS), np.nan)
def fit_eq(target_db, lim, w):
    """target_db: desired correction (dB per band). 3 biquads: low shelf, one peak, high shelf. Fit up to a free constant, gains bounded, small gain penalty."""
    def mk(p): return EQ.build([{"type": "low_shelf", "f": p[0], "gain_db": p[1], "s": 0.7}, {"type": "peak", "f": p[2], "gain_db": p[3], "q": p[4]}, {"type": "high_shelf", "f": p[5], "gain_db": p[6], "s": 0.7}])
    def res(p):
        h = EQ.resp_db(mk(p), FC_BANDS); e = h - target_db; e = e - np.sum(w * e) / np.sum(w); return np.r_[np.sqrt(w) * e, 0.15 * p[[1, 3, 6]]]
    lo = [60, -lim, 300, -lim, 0.5, 2500, -lim]; hi = [400, lim, 4000, lim, 1.5, 9000, lim]; best = None
    for f1, f2, f3 in ((150, 800, 4000), (100, 1500, 6000), (250, 2500, 3000)):
        r = least_squares(res, [f1, 0, f2, 0, 0.8, f3, 0], bounds=(lo, hi))
        if best is None or r.cost < best.cost: best = r
    p = best.x; params = [{"type": "low_shelf", "f": float(p[0]), "gain_db": float(p[1]), "s": 0.7}, {"type": "peak", "f": float(p[2]), "gain_db": float(p[3]), "q": float(p[4])}, {"type": "high_shelf", "f": float(p[5]), "gain_db": float(p[6]), "s": 0.7}]
    return params
R = {}; 
for amp in ("jcm800", "vibrolux"):
    held, fit = Spec(amp, "held"), Spec(amp, "fit"); anch = ANCH[amp]; out = {"gains": held.gains, "T": {f"{g:g}": held.T(g) for g in held.gains}}
    # ---------- 1. signed error tables (held-out DIs, all offsets, both seeds)
    tab = {}
    for g in held.gains:
        tab[f"{g:g}"] = {}
        for c in CLASSES:
            cases = [(d, o, m) for d in held.dis for o in held.offs for m in SEEDS if held.nframes(g, d, o, c) > 30]
            if not cases: continue
            B = [broad_db(held.P(g, d, o, m, c), held.P(g, d, o, "real", c)) for d, o, m in cases]; tab[f"{g:g}"][c] = {k: float(np.mean([b[k] for b in B])) for k in B[0]} | {"n": len(cases)}
        tab[f"{g:g}"]["native_level_rms_db"] = float(np.mean([held.scalar(g, d, o, m, "rms") - held.scalar(g, d, o, "real", "rms") for d in held.dis for o in held.offs for m in SEEDS]))
        tab[f"{g:g}"]["seed_diff_hf_db"] = float(np.mean([abs(broad_db(held.P(g, d, o, "FC_s0"), held.P(g, d, o, "real"))["high 4-10k"] - broad_db(held.P(g, d, o, "FC_s1"), held.P(g, d, o, "real"))["high 4-10k"]) for d in held.dis for o in held.offs]))
        tab[f"{g:g}"]["by_offset_hf_share"] = {f"{o:g}": float(np.mean([broad_db(held.P(g, d, o, m), held.P(g, d, o, "real"))["HF>3k share"] for d in held.dis for m in SEEDS])) for o in held.offs}
    out["signed_table"] = tab
    # ---------- plots
    shp = np.array([mean_shape(held, [g], held.dis, held.offs) for g in held.gains])
    fig, ax = plt.subplots(figsize=(9, 4.5)); im = ax.imshow(shp.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4, extent=[-.5, len(held.gains) - .5, -.5, len(FC_BANDS) - .5])
    ax.set_xticks(range(len(held.gains))); ax.set_xticklabels([f"G{g:g}\n{held.T(g):+.0f}" for g in held.gains], fontsize=7); ax.set_yticks(range(0, len(FC_BANDS), 2)); ax.set_yticklabels([f"{int(round(f))}" for f in FC_BANDS[::2]], fontsize=7)
    plt.colorbar(im, label="signed error, FC - real (dB): red = FC has MORE energy, blue = LESS"); ax.set_ylabel("Hz"); ax.set_xlabel("position (and intended Input gain, dB)"); ax.set_title(f"{NAME[amp]}: level-independent spectral error, held-out DIs, native NAM"); fig.tight_layout(); fig.savefig(DOC / f"tonal_heatmap_{amp}.png", dpi=110); plt.close(fig)
    show = [1, 5, 10] if amp == "jcm800" else [1, 5, 10]; inter = [g for g in held.gains if g in ([2, 4, 8] if amp == "jcm800" else [3, 7, 8])]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for g, col in zip(show + inter, ["tab:blue", "tab:green", "tab:red", "gray", "tab:orange", "tab:purple"]):
        allv = np.array([shape_db(held.P(g, d, o, m), held.P(g, d, o, "real"))[0] for d in held.dis for o in held.offs for m in SEEDS]); axs[0].plot(FC_BANDS, allv.mean(0), color=col, label=f"G{g:g}"); axs[0].fill_between(FC_BANDS, allv.min(0), allv.max(0), color=col, alpha=.10)
    for c, col in zip(CLASSES, ["k", "tab:red", "tab:green", "tab:blue"]):
        v = mean_shape(held, [10], held.dis, held.offs, c); axs[1].plot(FC_BANDS, v, color=col, label=c)
    for o, col in zip(held.offs, ["tab:blue", "tab:green", "tab:orange", "tab:red"]): axs[2].plot(FC_BANDS, mean_shape(held, [10], held.dis, [o]), color=col, label=f"{o:+.0f} dB DI")
    for a, t in zip(axs, ("by position (band = min-max over DIs/levels/seeds)", "G10 by playing phase", "G10 by musical input level")): a.set_xscale("log"); a.axhline(0, color="k", lw=.5); a.set_ylim(-8, 4); a.grid(alpha=.25); a.legend(fontsize=7); a.set_title(f"{NAME[amp].split(' ')[0]} {t}", fontsize=9); a.set_xlabel("Hz"); a.set_ylabel("FC - real (dB)")
    fig.tight_layout(); fig.savefig(DOC / f"tonal_curves_{amp}.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8)); xs = held.gains
    for k in BROAD: axs[0].plot(xs, [tab[f"{g:g}"]["all"][k] for g in xs], marker="o", ms=3, label=k)
    axs[1].plot(xs, [tab[f"{g:g}"]["all"]["HF>3k share"] for g in xs], "o-", label="HF>3k share"); axs[1].plot(xs, [tab[f"{g:g}"]["all"]["tilt dB/oct"] for g in xs], "s-", label="tilt (dB/oct)")
    axs[2].plot(xs, [tab[f"{g:g}"]["native_level_rms_db"] for g in xs], "o-", color="k", label="native output level")
    for a, t in zip(axs, ("broad-band level-independent error", "HF share and tilt", "native output-level error (separate)")): a.axhline(0, color="k", lw=.5); a.grid(alpha=.25); a.legend(fontsize=6); a.set_title(f"{NAME[amp].split(' ')[0]}: {t}", fontsize=9); a.set_xlabel("position"); a.set_ylabel("FC - real (dB)")
    fig.tight_layout(); fig.savefig(DOC / f"tonal_broad_{amp}.png", dpi=110); plt.close(fig)
    # ---------- 2. EQ candidates, fit ONLY on the fit DIs at the FC anchor positions
    base = {g: mean_shape(fit, [g], fit.dis, fit.offs) for g in anch}
    wB = np.where(FITB, 1.0, 0.3)
    cB = -np.mean([base[g] for g in anch], axis=0)
    wg = {g: (3.0 if g >= max(anch) - 3 and g >= 7 else 1.0) for g in anch}; cC = -np.sum([wg[g] * base[g] for g in anch], axis=0) / sum(wg.values())
    cands = {"A_no_EQ": []}; cands["B"] = fit_eq(cB, LIM["B"]["gain"], wB); cands["C"] = fit_eq(cC, LIM["C"]["gain"], wB)
    out["fit_target_B"] = cB.tolist(); out["fit_target_C"] = cC.tolist(); out["fit_target_per_anchor"] = {f"{g}": (-base[g]).tolist() for g in anch}
    freqs_h = np.fft.rfftfreq(NP, 1 / SR)
    eqinfo = {}
    for n, p in cands.items():
        if not p: eqinfo[n] = {"params": [], "max_gain_db": 0.0, "max_cut_db": 0.0}; continue
        sos = EQ.build(p); h = EQ.resp_db(sos, freqs_h[1:]); w_, gd = group_delay((np.r_[sos[0][:3]], np.r_[sos[0][3:]]), w=8) if False else (None, None)
        gdm = 0.0
        for row in sos:
            wgd, g_ = group_delay((row[:3], row[3:]), w=np.linspace(0.005, np.pi * 0.5, 400)); gdm += g_
        eqinfo[n] = {"params": p, "sos": sos.tolist(), "max_gain_db": float(h.max()), "max_cut_db": float(h.min()), "resp_at_band_db": EQ.resp_db(sos, FC_BANDS).tolist(), "group_delay_max_ms_below_12k": float(np.max(np.abs(gdm)) / SR * 1000)}
    out["eq"] = eqinfo
    # ---------- 3. validation on held-out DIs at ALL positions, per region, EQ applied exactly in the frequency domain
    Hs = {n: (None if not p else EQ.power(EQ.build(p), freqs_h)) for n, p in cands.items()}
    def mabs(gains, cls, H, offs=None):
        v = [np.abs(shape_db(held.P(g, d, o, m, cls), held.P(g, d, o, "real", cls), H)[0][FITB]).mean() for g in gains for d in held.dis for o in (offs or held.offs) for m in SEEDS if held.nframes(g, d, o, cls) > 30]; return float(np.mean(v))
    def sgn(gains, cls, H, key, offs=None):
        v = [broad_db(held.P(g, d, o, m, cls), held.P(g, d, o, "real", cls), H)[key] for g in gains for d in held.dis for o in (offs or held.offs) for m in SEEDS if held.nframes(g, d, o, cls) > 30]; return float(np.mean(v))
    def lvl(gains, H):
        return float(np.mean([10 * np.log10(np.sum(held.P(g, d, o, m) * (H if H is not None else 1)) / np.sum(held.P(g, d, o, m))) for g in gains for d in held.dis for o in held.offs for m in SEEDS]))
    val = {}
    regs = dict(REGIONS[amp]); regs["intermediate positions (not fitted)"] = [g for g in held.gains if g not in anch]; regs["fitted anchor positions"] = [g for g in held.gains if g in anch]
    regs = {k: [g for g in v if g in held.gains] for k, v in regs.items()}
    for n, H in Hs.items():
        val[n] = {}
        for rn, gs in regs.items():
            if not gs: continue
            val[n][rn] = {"mean_abs_band_err_db": mabs(gs, "all", H), "attack_mean_abs_db": mabs(gs, "attack", H), "decay_mean_abs_db": mabs(gs, "decay", H), "tilt_db_oct": sgn(gs, "all", H, "tilt dB/oct"), "HF>3k share": sgn(gs, "all", H, "HF>3k share"), "high 4-10k": sgn(gs, "all", H, "high 4-10k"), "low 80-250": sgn(gs, "all", H, "low 80-250"), "upper-mid 1.6-4k": sgn(gs, "all", H, "upper-mid 1.6-4k"), "level_shift_from_EQ_db": lvl(gs, H)}
            val[n][rn]["by_offset_mean_abs"] = {f"{o:g}": mabs(gs, "all", H, [o]) for o in held.offs}
        val[n]["per_position_mean_abs"] = {f"{g:g}": mabs([g], "all", H) for g in held.gains}
        val[n]["per_position_level_shift_db"] = {f"{g:g}": lvl([g], H) for g in held.gains}
    out["validation"] = val; out["regions"] = regs
    # native level progression check: relative level step between adjacent positions after EQ (shift differences)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for n, col in (("A_no_EQ", "k"), ("B", "tab:blue"), ("C", "tab:red")): axs[0].plot(held.gains, [val[n]["per_position_mean_abs"][f"{g:g}"] for g in held.gains], "o-", color=col, label=n, ms=3)
    axs[0].set_title(f"{NAME[amp]}: mean |band error| 80 Hz-12.7 kHz by position (held-out DIs)", fontsize=9); axs[0].set_xlabel("position"); axs[0].set_ylabel("dB"); axs[0].legend(); axs[0].grid(alpha=.25)
    for n, col in (("B", "tab:blue"), ("C", "tab:red")): axs[1].semilogx(FC_BANDS, eqinfo[n]["resp_at_band_db"], color=col, label=f"EQ {n}")
    axs[1].semilogx(FC_BANDS, cB, "--", color="tab:blue", alpha=.5, label="target B (fit)"); axs[1].semilogx(FC_BANDS, cC, "--", color="tab:red", alpha=.5, label="target C (fit)"); axs[1].axhline(0, color="k", lw=.5); axs[1].legend(fontsize=7); axs[1].grid(alpha=.25); axs[1].set_title("EQ candidates and their fit targets", fontsize=9); axs[1].set_xlabel("Hz"); axs[1].set_ylabel("dB")
    fig.tight_layout(); fig.savefig(DOC / f"tonal_eq_{amp}.png", dpi=110); plt.close(fig)
    R[amp] = out
(REPO / "work" / "tr" / "analysis.json").write_text(json.dumps(R, indent=1, default=float)); print("ok")

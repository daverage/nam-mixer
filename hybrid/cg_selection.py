"""Continuous Gain capture selection (Phase 4D): which captures carry distinct information?

Behaviour-preserving extraction of `scripts/p4_select.py` (same groups, tolerances, objective, exhaustive
search, k* rule, response coordinate). No neural training and no imposed capture count. Question asked of
every candidate subset S: how well are the OTHER captures reproduced from S? Each omitted position is predicted
by shape-preserving (PCHIP) interpolation of the selected captures' MEASURED response along the knob axis
(held flat outside the selected span), separately for tone, saturation and compression. Level is reported,
not optimised. The tolerances are working values, not perceptual measurements.

`k_star` is the smallest count at which EVERY omitted position is within tolerance in EVERY physical group.
If no subset satisfies that coverage rule, `k_star` is None and `resolve_selection` says so and falls back to
"use all eligible captures" -- it never invents an optimum.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.interpolate import PchipInterpolator

from .cg_audit import ELIGIBLE_STATUSES

GROUPS = {
    "tone": ["EQ sub", "EQ low", "EQ lowmid", "EQ mid", "EQ presence", "EQ air", "tilt (presence-low)"],
    "saturation": ["HF >3 kHz", "THD 440Hz @-30", "THD 440Hz @-18", "THD 440Hz @-6", "H2 440Hz @-18", "H3 440Hz @-18", "crest @0"],
    "compression": ["IO gain -54->-30", "IO gain -30->0", "dyn range @0", "dyn range @+6"],
    "level": ["music RMS @0 dB DI", "music RMS @-12 dB DI", "music RMS @+6 dB DI"],
}
SEL_DIMS = ["tone", "saturation", "compression"]
PHYS = {"tone EQ (dB)": (["EQ sub", "EQ low", "EQ lowmid", "EQ mid", "EQ presence", "EQ air"], 1.0), "HF>3k (dB)": (["HF >3 kHz"], 1.0),
        "crest (dB)": (["crest @0"], 1.0), "THD (dB)": (["THD 440Hz @-30", "THD 440Hz @-18", "THD 440Hz @-6"], 5.0),
        "IO slope (dB)": (["IO gain -54->-30", "IO gain -30->0"], 2.0), "dyn range (dB)": (["dyn range @0", "dyn range @+6"], 1.5),
        "level (dB)": (["music RMS @0 dB DI", "music RMS @-12 dB DI", "music RMS @+6 dB DI"], None)}
MAX_EXHAUSTIVE_ELIGIBLE = 12       # 2^12 = 4096 subsets; beyond that an exhaustive search is not attempted (see resolve_selection)


class SelectionAnalysis:
    def __init__(self, profile: dict, audit: dict, candidate_positions: list[float] | None = None):
        self.gains = [float(g) for g in profile["gains"]]
        self.G = np.array(self.gains)
        self.S = profile["series"]
        self.audit = audit["captures"]
        key = lambda g: f"{g:g}"
        if candidate_positions is None:
            ints = [i for i, g in enumerate(self.gains) if g == int(g)]
            self.ints = ints if len(ints) >= 2 else list(range(len(self.gains)))
        else:
            want = {float(p) for p in candidate_positions}
            self.ints = [i for i, g in enumerate(self.gains) if g in want]
        self.half = [i for i in range(len(self.gains)) if i not in self.ints]
        self.eligible = [i for i in self.ints if self.audit[key(self.gains[i])]["status"] in ELIGIBLE_STATUSES]
        self.level_q = {i for i, g in enumerate(self.gains) if any(e.startswith("level:") for e in self.audit[key(g)]["evidence"])}
        self._floored = [n for n in self.S if n.startswith(("THD", "H2", "H3"))]
        self._rng: dict[str, float] = {}

    def val(self, n):
        v = np.array(self.S[n]["values"], float)
        return np.maximum(v, -80.0) if n in self._floored else v

    def rng(self, n):
        if n not in self._rng:
            self._rng[n] = max(float(np.ptp(self.val(n))), 1e-9)
        return self._rng[n]

    def predict(self, sel, names, positions):
        sel = sorted(sel)
        xs = self.G[sel]
        out = np.zeros((len(positions), len(names)))
        for j, n in enumerate(names):
            y = self.val(n)[sel]
            f = PchipInterpolator(xs, y) if len(xs) > 1 else (lambda x, y=y: np.full_like(x, y[0], dtype=float))
            out[:, j] = f(np.clip(self.G[positions], xs.min(), xs.max()))
        return out

    def dim_errors(self, sel, positions, dim):
        names = GROUPS[dim]
        act = np.stack([self.val(n)[positions] for n in names], axis=1)
        e = np.abs(self.predict(sel, names, positions) - act) / np.array([self.rng(n) for n in names])
        return e.mean(axis=1)

    def J(self, sel) -> float:
        omit = [i for i in self.ints if i not in sel]
        if not omit:
            return 0.0
        tot = 0.0
        for d in SEL_DIMS:
            e = self.dim_errors(sel, omit, d)
            tot += 0.5 * (e.sum() / len(self.ints)) + 0.5 * e.max()
        return tot

    def phys(self, sel, positions):
        out = {}
        for g, (names, tol) in PHYS.items():
            act = np.stack([self.val(n)[positions] for n in names], axis=1)
            e = np.abs(self.predict(sel, names, positions) - act).mean(axis=1)
            keep = [k for k, p in enumerate(positions) if not (g.startswith("level") and p in self.level_q)]
            e = e[keep] if keep else e
            out[g] = {"mean": float(e.mean()), "max": float(e.max()), "tol": tol}
        return out

    def within_tolerance(self, sel) -> bool:
        omit = [i for i in self.ints if i not in sel]
        if not omit:
            return True
        return all(v["max"] <= v["tol"] for v in self.phys(list(sel), omit).values() if v["tol"] is not None)

    def response_coordinate(self) -> dict:
        names = [n for d in SEL_DIMS for n in GROUPS[d]]
        Z = np.stack([self.val(n) / self.rng(n) for n in names], axis=1)
        order = np.argsort(self.G)
        seg = np.linalg.norm(np.diff(Z[order], axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        return {"gains": self.G[order].tolist(), "arc": (s / s[-1]).tolist(), "total": float(s[-1])}


def select_captures(profile: dict, audit: dict, candidate_positions: list[float] | None = None) -> dict:
    """The full Phase 4D analysis (selection.json content, minus the research-only v3 baseline rows)."""
    a = SelectionAnalysis(profile, audit, candidate_positions)
    gains, key = a.gains, (lambda g: f"{g:g}")
    res = {"eligible": [gains[i] for i in a.eligible],
           "ineligible": {key(gains[i]): a.audit[key(gains[i])]["status"] for i in a.ints if i not in a.eligible},
           "by_k": {}, "exhaustive": len(a.eligible) <= MAX_EXHAUSTIVE_ELIGIBLE, "max_exhaustive_eligible": MAX_EXHAUSTIVE_ELIGIBLE,
           "candidate_positions": [gains[i] for i in a.ints], "k_star_all_within_tolerance": None}
    allc = {}
    if res["exhaustive"]:
        for k in range(2, len(a.eligible) + 1):
            best = min(itertools.combinations(a.eligible, k), key=lambda c: a.J(list(c)))
            allc[k] = best
            omit = [i for i in a.ints if i not in best]
            res["by_k"][str(k)] = {"best": [gains[i] for i in best], "J": a.J(list(best)), "phys": a.phys(list(best), omit) if omit else {}}
        for k in sorted(allc):
            ph = res["by_k"][str(k)]["phys"]
            if ph and all(v["max"] <= v["tol"] for v in ph.values() if v["tol"] is not None):
                res["k_star_all_within_tolerance"] = k
                break
        if a.half:
            res["half_step_validation"] = {str(k): a.phys(list(b), a.half) for k, b in allc.items()}
    greedy, cur = [], []
    if res["exhaustive"]:
        for _ in range(len(a.eligible)):
            nxt = min((i for i in a.eligible if i not in cur), key=lambda i: a.J(cur + [i]))
            cur.append(nxt)
            greedy.append({"add": gains[nxt], "J_after": a.J(cur)})
    res["greedy_order"] = greedy
    res["loo_normalised_error"] = {key(gains[i]): {d: float(a.dim_errors([j for j in a.ints if j != i], [i], d)[0]) for d in SEL_DIMS}
                                   for i in a.ints if i not in (a.ints[0], a.ints[-1])}
    res["endpoint_omitted_normalised_error"] = {key(gains[i]): {d: float(a.dim_errors([j for j in a.ints if j != i], [i], d)[0]) for d in SEL_DIMS}
                                                for i in (a.ints[0], a.ints[-1])}
    res["response_coordinate"] = a.response_coordinate()
    return res


def evaluate_set(profile: dict, audit: dict, selected: list[float], candidate_positions: list[float] | None = None) -> dict:
    """Measured coverage of an arbitrary chosen set (used by Custom / Use-all): J, per-group errors on the omitted
    candidate positions against the working tolerances, and a factual reason for every candidate capture."""
    a = SelectionAnalysis(profile, audit, candidate_positions)
    gains, key = a.gains, (lambda g: f"{g:g}")
    idx = sorted(a.gains.index(float(g)) for g in selected)
    omit = [i for i in a.ints if i not in idx]
    phys = a.phys(idx, omit) if omit else {}
    within = a.within_tolerance(idx)
    base_j = a.J(idx)
    reasons = {}
    for i in a.ints:
        g = gains[i]
        st = a.audit[key(g)]["status"]
        if i in idx:
            span = (i == idx[0] or i == idx[-1])
            j_without = a.J([j for j in idx if j != i]) if len(idx) > 2 else None
            reasons[key(g)] = {"role": "selected", "status": st,
                               "reason": (f"{'Lowest' if i == idx[0] else 'Highest'} selected position: omitting it makes the rest extrapolate from a shorter span" if span
                                          else "Interior selected position: its measured response is used directly where its neighbours would otherwise be interpolated")
                                         + (f" (coverage objective J rises {base_j:.3f} -> {j_without:.3f} without it)" if j_without is not None else "")}
        elif st not in ELIGIBLE_STATUSES:
            reasons[key(g)] = {"role": "needs_review", "status": st, "reason": f"Capture audit status {st}: analysis-only, never an automatic training anchor. " + "; ".join(a.audit[key(g)]["evidence"])[:300]}
        else:
            errs = {d: float(a.dim_errors(idx, [i], d)[0]) for d in SEL_DIMS}
            reasons[key(g)] = {"role": "omitted", "status": st, "predicted_from_selection_normalised_error": errs,
                               "reason": "Omitted: reproduced from the selected captures' interpolated measurements (normalised error tone {:.3f}, saturation {:.3f}, compression {:.3f}); retained as a reference capture for validation".format(errs["tone"], errs["saturation"], errs["compression"])}
    return {"selected": [gains[i] for i in idx], "J": base_j, "phys": phys, "all_within_tolerance": within, "reasons": reasons}


def resolve_selection(analysis: dict, profile: dict, audit: dict, mode: str, custom: list[float] | None = None,
                      candidate_positions: list[float] | None = None) -> dict:
    """Automatic / Use all / Custom -> the training capture set, with an honest statement of what was done."""
    eligible = list(analysis["eligible"])
    notes: list[str] = []
    fallback = False
    if mode == "automatic":
        kstar = analysis.get("k_star_all_within_tolerance")
        if not analysis.get("exhaustive", True):
            selected, fallback = eligible, True
            notes.append(f"{len(eligible)} eligible captures exceeds the exhaustive-search limit ({analysis['max_exhaustive_eligible']}); no optimum was searched, so all eligible captures are used. Choose Custom to pick a subset.")
        elif kstar is None:
            selected, fallback = eligible, True
            notes.append("No subset of the eligible captures reproduces every omitted position within the working tolerances (the Phase 4D coverage rule). No optimum is claimed; falling back to all eligible captures. Choose Custom to try a subset.")
        else:
            selected = list(analysis["by_k"][str(kstar)]["best"])
            notes.append(f"Smallest subset meeting the coverage rule (k*={kstar}): every omitted position is within the working tolerances in every physical group.")
    elif mode == "use_all":
        selected = eligible
        notes.append("All eligible (VALID/CORRECTED) captures are used.")
    elif mode == "custom":
        if not custom or len(set(custom)) < 2:
            raise ValueError("Custom selection needs at least two captures")
        unknown = [g for g in custom if float(g) not in {float(x) for x in profile["gains"]}]
        if unknown:
            raise ValueError(f"unknown capture positions: {unknown}")
        selected = sorted({float(g) for g in custom})
        bad = [g for g in selected if audit["captures"][f"{g:g}"]["status"] not in ELIGIBLE_STATUSES]
        if bad:
            notes.append(f"WARNING: positions {bad} are not VALID/CORRECTED in the capture audit; they are used only because you chose them.")
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    ev = evaluate_set(profile, audit, selected, candidate_positions)
    if not ev["all_within_tolerance"] and mode != "automatic":
        notes.append("The chosen set does not reproduce every omitted position within the working tolerances (see coverage).")
    return {"mode": mode, "selected": ev["selected"], "fallback": fallback, "coverage": ev, "notes": notes,
            "k_star": analysis.get("k_star_all_within_tolerance")}

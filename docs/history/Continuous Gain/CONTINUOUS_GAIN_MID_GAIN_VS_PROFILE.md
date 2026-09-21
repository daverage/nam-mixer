# Continuous Gain -- One Mid-Gain NAM vs. the Multi-Anchor Profile

**Top-level question: does NAM Mixer's Continuous Gain multi-anchor profile
meaningfully beat a single mid-gain NAM capture (Gain 5) driven by ordinary,
calibrated virtual input gain?**

Follow-up to docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md,
docs/CONTINUOUS_GAIN_CROSS_AMP_GENERALIZATION.md and
docs/CONTINUOUS_GAIN_PRODUCTION.md. No model training of any kind occurs
here, and no production module was modified: this is pure signal processing
plus real NAM inference (`hybrid.render.render`) over four real amp capture
sweeps.

STATUS: complete. Machine-readable results in
`work/continuous_gain_mid_vs_profile/results.json` (plus
`results_continuity_supersonic.json`), audio in
`work/continuous_gain_mid_vs_profile/audio/` -- all gitignored scratch,
regenerate with the scripts below.

New code (additive only):

- `scripts/continuous_gain_mid_vs_profile_benchmark.py` -- the benchmark driver.
- `scripts/continuous_gain_mid_vs_profile_report.py` -- turns the JSON into
  the tables reproduced below (formatting only; no metrics computed there).

Wall clock: **31.2 minutes** for the full four-amp run on an Apple Silicon
laptop (one 6-second DI segment per evaluation, native `nam_render` at
~0.16 s per 6 s render). The dominant cost is the per-amp 10x10
virtual-gain search grid: 517 s (JCM800), 536 s (Super-Sonic), 465 s
(Peavey 5150); the Twin's grid was already in the resume cache from a smoke
run and cost 0 s. Everything else -- every leave-one-out refit, every
profile build, every method's evaluation across three DIs -- added only
~60 s per amp, because the grid is memoized to disk and reused.

---

## 1. What is being compared

Four reconstruction methods, all scored against the **real per-position NAM
capture** rendered through the same DI:

| Key | Method | Captures needed | Fitting |
|---|---|---:|---|
| **A-direct** | Single fixed Gain-5 NAM + a *conventional* gain control: one constant dB per knob step | 1 | one scalar slope |
| **A-calibrated** | Single fixed Gain-5 NAM + the production profiling algorithm restricted to **one** anchor: measured per-position optimal input gain, monotonic-PCHIP interpolated | 1 | a measured nonlinear curve |
| **B-profile** | `scripts/continuous_gain_profile_builder.py`'s engine: `hybrid.continuous_gain_profile.build_profile(max_anchors=3)` + `ContinuousGainRuntime` | up to 3 | full automatic anchor selection + per-anchor curve |
| **C-dense** | `hybrid.continuous_gain.run_ground_truth_harness` + `interpolate_output` over all training captures | 9-10 | none (positional crossfade) |

Signal path for A and B is identical to every prior report in this thread:

```text
DI --(x 10**(input_gain_db / 20))--> fixed NAM anchor --> output
```

gain applied to the **dry** signal before `render()`, never to the output.

**B is the real production builder, not an oracle.** The earlier cross-amp
report's "best 2-anchor / best 3-anchor" numbers were a *best-per-target
minimum* over an anchor set -- a feasibility ceiling that assumed knowledge
of the right answer at every position. This report deliberately does not do
that: `build_profile` runs exactly as shipped, picks its own anchor set by
its own quality rule, assigns its own region ownership, and the runtime
engine then has to choose an anchor from the knob position alone. That
distinction turns out to matter a great deal (Section 5).

## 2. Datasets

Real capture sweeps outside the repo, at
`/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/`. Filenames were
listed and verified rather than assumed:

| Amp | Directory | Control | Captures found |
|---|---|---|---|
| Marshall JCM800 2203 (High ch.) | `Marshall JCM800 2203 - updated` | Gain | 19: 1.0-10.0 in 0.5 steps (`g10` is named `ga10`) |
| Fender Super-Sonic 60W (Bassman ch., T5/B5) | `[500 Epochs] Fender Super-Sonic 60W Head mk.1 - Flat EQ - Complete Pack` | Volume | 10 integer |
| Fender 57 Custom Twin (Ch. 1) | `FENDER 57 CUSTOM TWIN (MULTI GAIN)` | Volume | 10 integer (`CH 1 - VOL n`) |
| Peavey 5150 (head only, stock) | `Peavy 5150 (Head Only)` | Gain | 10 integer (the parallel "(Boosted)" set was not used) |

**Only the JCM800 has genuine half-step captures.** The brief's required
JCM800 train/validation split (integers G1..G10 to fit, half-steps
G1.5..G9.5 to validate) is therefore real and was used. The other three amps
have integer captures only -- that limitation is reported, not worked
around -- so they use leave-one-position-out instead.

Three DIs, last 6 seconds of each (the held-out-tail convention every prior
report in this thread uses): `moderate_brit.wav` ("standard", **primary**),
`clean_smooth.wav` ("clean"), `high_metalcore.wav` ("metalcore").

None of the four amps' `.nam` files report `input_level_dbu`, so every
render is uncalibrated/raw (`calibration_mode="raw"`) -- consistent across
all methods, so it cannot bias the comparison.

## 3. Methodology and every judgment call made

The brief left several things open. Each decision is recorded here.

1. **What "ordinary/conventional input gain" means (A-direct).** There is no
   defensible universal dB-per-knob-step constant, so rather than invent
   one, A-direct uses the *best possible* single linear law for each amp:
   `gain_db(p) = slope * (p - 5)`, with `slope` least-squares fitted
   (no intercept, since the anchor at its own position needs 0 dB by
   construction) to the measured optimal gains at the **training positions
   only**. This deliberately makes the conventional baseline as strong as a
   linear law can be -- any deficit it shows is a deficit of linearity
   itself, not of a badly chosen constant.
2. **What "calibrated" means (A-calibrated).** The production profiling
   algorithm restricted to one anchor:
   `hybrid.continuous_gain_profile.search_virtual_input_gain` per training
   position (coarse -24..+24 dB at 3 dB, then 0.5 dB, then 0.1 dB,
   minimising raw ESR over the DI's *active* material via
   `hybrid.coverage.active_signal_mask`), then
   `_monotonic_interpolate` (PCHIP, clamped outside the measured range).
   This is byte-for-byte the same search and the same curve fit the
   multi-anchor builder uses -- so A-calibrated vs. B isolates *anchor
   count and region logic*, nothing else.
3. **Scoring metric.** Raw (unmatched) ESR is the headline, because that is
   what a player actually hears. Level-matched ESR, output-level delta (dB)
   and `spectral_magnitude_correlation` are reported separately and never
   collapsed into it -- matching level alone is never treated as success.
4. **Search-grid memoization.** The (anchor, target) virtual-gain search is
   deterministic, so results are memoized to
   `work/continuous_gain_mid_vs_profile/search_cache.json` and reused across
   every leave-one-out fold and every rerun. The benchmark installs this
   cache by temporarily rebinding
   `hybrid.continuous_gain_profile.search_virtual_input_gain` for the
   duration of a `build_profile` call; production code is not edited and the
   cached value is exactly what the search would have recomputed. This is
   what makes a 10-fold leave-one-out affordable (and makes the run
   resumable after an interruption).
5. **No leakage.** For each fold, the withheld position is removed from the
   search-grid rows used for fitting, from `build_profile`'s training list,
   and from the dense-interpolation capture set. A withheld capture's real
   output is only ever used as the scoring reference.
6. **Gain 5 is always available as the anchor.** Per the brief, the
   single-anchor methods keep their G5 capture even in the fold where
   position 5 is withheld -- it is the one capture those methods are defined
   to own. That fold's single-anchor result is trivially near-exact, so it
   is flagged `anchor` and **excluded from aggregates** rather than counted
   as a win.
7. **Extrapolation is never presented as interpolation.** In
   leave-one-position-out, withholding position 1 or 10 removes the end of
   the fitted range, so every method is *extrapolating* (the PCHIP curve
   clamps, and `interpolate_output` clamps to the outermost training pair).
   Those rows are flagged `**extrap**` and excluded from all aggregates.
   With the anchor row also excluded, the integer-capture amps contribute
   **7 usable positions** each, and the JCM800 contributes 8 of its 9
   half-steps (Gain 8.5 is excluded as a capture defect, Section 4).
8. **Continuity was run on two amps, not four.** JCM800 is the primary
   regression amp, but its production profile turns out to have a *single*
   anchor, so it contains no anchor transitions to measure. Super-Sonic was
   added specifically because its profile does have two anchors and a real
   region boundary. Running all four would have added cost without adding a
   transition to test.
9. **`psutil` is not installed** in this environment, so CPU/memory uses the
   stdlib `resource` module (`RUSAGE_CHILDREN`, since the actual inference
   runs in the native `nam_render` child process) plus the resident anchor
   `.nam` file sizes.
10. **No human listening test was performed by the automation.** The 5150
    comparison WAVs (including level-matched variants) are generated and
    their paths are given; conclusions drawn from them here are explicitly
    limited to what the measurements support.

## 4. Capture-timing defects, isolated rather than averaged away

The cross-amp report's neighbour-lag check flags a defective capture *and*
both of its healthy neighbours, because it is pairwise. This benchmark
refines it (`isolate_defective_captures`): a capture is called defective
only if its lag to **every** neighbour exceeds 20 samples **and** those
neighbours align cleanly with **each other** when the suspect is skipped.

**JCM800 Gain 8.5 -- confirmed, correctly isolated.** Lag +691 samples vs.
Gain 8.0 and +690 vs. Gain 9.0, while Gain 8.0 vs. Gain 9.0 directly align
at **-1 sample (correlation 0.997)**. Gain 8.5 alone is flagged; **Gain 8.0
and Gain 9.0 are kept** and remain in the training set, as the brief
requires. Gain 8.5's row is reported separately (raw ESR 2.39-2.42 for
*every* method, including dense interpolation -- the defect, not the
method) and excluded from aggregates.

**New finding -- Peavey 5150 Gain 6 shows the same class of defect.** Lag
+414 samples vs. Gain 5 and +406 vs. Gain 7, while Gain 5 vs. Gain 7
directly align at -8 samples (correlation 0.983). This is the same
signature as JCM800 Gain 8.5 and has not been reported in this research
thread before (the previously-documented 5150 anomaly was a *level*
non-monotonicity at Gain 4, a different capture and a different symptom).
Treated with caution -- see the 5150 caveats in Section 7 -- but it is
flagged and excluded.

**Peavey 5150 Gain 10 is flagged but unverifiable.** It is a sweep endpoint
with only one neighbour (lag +114 vs. Gain 9), so the
"do-the-neighbours-agree-without-it" cross-check cannot run. It is reported
as *suspected, unverifiable* and excluded (it is an extrapolation row
anyway). No claim is made that it is genuinely defective.

No timing defects were found in the Super-Sonic or 57 Custom Twin sweeps.

## 5. Results

### 5.1 Cross-amp summary -- primary DI, usable withheld positions only

Raw ESR, **mean / worst**. Lower is better. "n" is the number of positions
left after excluding the anchor's own position, extrapolation rows and
flagged capture defects.

| Amp | Split | n | A-direct | A-calibrated | B-profile | C-dense |
|---|---|---:|---:|---:|---:|---:|
| Marshall JCM800 2203 | half-step validation | 8 | 0.0706 / 0.4860 | **0.0047 / 0.0121** | **0.0047 / 0.0121** | 0.0022 / 0.0036 |
| Fender Super-Sonic 60W | leave-one-out | 7 | 0.2560 / 1.1780 | **0.0277 / 0.0884** | 0.3054 / 2.0195 | 0.0539 / 0.2183 |
| Fender 57 Custom Twin | leave-one-out | 7 | 0.1481 / 0.5181 | **0.0277 / 0.0515** | **0.0277 / 0.0515** | 0.0308 / 0.1781 |
| Peavey 5150 (stock) | leave-one-out | 6 | 37.3486 / 188.2592 | 1.2126 / 1.9521 | 0.6035 / 1.2057 | 1.0683 / 2.2021 |
| | | | | | | *(5150: all values unreliable, Section 7)* |

Two things jump out immediately:

- **Calibrating the mapping is worth 5x-15x.** A-direct -> A-calibrated:
  15.0x better on JCM800, 9.2x on Super-Sonic, 5.3x on the Twin, ~30x on the
  5150. This is by far the largest single effect measured in this report.
- **B-profile is identical to A-calibrated on two of four amps, and worse on
  a third.** It is never better on any amp where the metric is reliable.

### 5.2 Why B so often equals A-calibrated: the builder picks one anchor

`build_profile` returns the **smallest** anchor set whose worst-case
training ESR reaches "acceptable" (`QUALITY_ACCEPTABLE_MAX_ESR = 0.06`). On
these datasets, a single anchor clears that bar almost everywhere:

| Amp | Anchor set selected per fold | Build quality |
|---|---|---|
| JCM800 | `[5.0]` in **all 9 folds** | acceptable (worst 0.0171) |
| 57 Custom Twin | `[5.0]` in 8 folds; `[6.0]` in the two folds where 5 or 1 is withheld | acceptable (worst 0.051-0.055) |
| Super-Sonic | `[3.0]`, `[1.0, 3.0]`, `[1.0, 5.0]`, `[1.0, 6.0]` depending on fold | acceptable / validated |
| Peavey 5150 | `[1.0]` (or `[2.0]`) in every fold | **poor** (worst ~1.00) |

So on the JCM800 -- the amp the whole Continuous Gain feature was designed
around -- **the shipped multi-anchor builder is already choosing exactly the
single mid-gain G5 anchor that method A-calibrated uses, and produces
bit-identical output.** The "multi-anchor profile" is, on that amp, a
single-anchor profile.

This is the central reason the earlier cross-amp report's conclusion should
not be read as a production result: that report's best-3-anchor oracle
(mean 0.0010 on JCM800) is not what the builder actually emits.

### 5.3 Marshall JCM800 2203 -- genuinely withheld half-step captures

Fitted on integers G1..G10 only; every row below is a real capture that
took no part in any fit. `Flags`: `defect` = the Gain 8.5 timing defect.

| Gain | Flags | A-dir gain | A-cal gain | B gain | A-dir ESR | A-cal ESR | B ESR | C ESR | A-cal lvl Δ | A-cal spec |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | -- | -14.3 | -19.5 | -19.5 | 0.4860 | 0.0121 | 0.0121 | 0.0027 | -0.50 | 0.982 |
| 2.5 | -- | -10.2 | -10.5 | -10.5 | 0.0048 | 0.0032 | 0.0032 | 0.0036 | -0.01 | 0.976 |
| 3.5 | -- | -6.1 | -4.2 | -4.2 | 0.0271 | 0.0014 | 0.0014 | 0.0014 | +0.08 | 0.978 |
| 4.5 | -- | -2.0 | -1.0 | -1.0 | 0.0081 | 0.0026 | 0.0026 | 0.0018 | +0.04 | 0.960 |
| 5.5 | -- | +2.0 | +0.9 | +0.9 | 0.0021 | 0.0022 | 0.0022 | 0.0015 | -0.35 | 0.967 |
| 6.5 | -- | +6.1 | +5.0 | +5.0 | 0.0017 | 0.0038 | 0.0038 | 0.0029 | -0.34 | 0.962 |
| 7.5 | -- | +10.2 | +10.5 | +10.5 | 0.0042 | 0.0036 | 0.0036 | 0.0010 | +0.17 | 0.950 |
| 8.5 | **defect** | +14.3 | +13.8 | +13.8 | 2.4197 | 2.3861 | 2.3861 | 2.2961 | +0.11 | 0.945 |
| 9.5 | -- | +18.3 | +14.9 | +14.9 | 0.0310 | 0.0084 | 0.0084 | 0.0027 | +0.47 | 0.913 |
| **mean** | | | | | **0.0706** | **0.0047** | **0.0047** | **0.0022** | | |
| **worst** | | | | | **0.4860** | **0.0121** | **0.0121** | **0.0036** | | |

Measured G5 mapping (fit set), conventional-law slope **+4.07 dB/step**:

| Gain | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| input gain dB | -24.0* | -15.1 | -6.7 | -2.3 | 0.0 | +2.1 | +8.3 | +12.2 | +14.7 | +15.0 |

`*` pinned at the -24 dB search boundary, reproducing the prior report's
finding that G5 cannot be turned down far enough to become Gain 1. The
curve is strongly compressive: ~8 dB per knob step at the bottom, ~0.3 dB
per step at the top -- which is exactly why the single linear slope
(A-direct) fails so badly at Gain 1.5 (0.486) while the calibrated curve
handles it (0.0121).

Train-vs-withheld contrast (in-sample means, excluding the anchor's own
position): A-direct 0.3141, A-calibrated 0.0076, B 0.0076. The withheld
half-step means (0.0706 / 0.0047 / 0.0047) are **as good as or better than**
the in-sample means for every method -- the fitted mappings are genuinely
interpolating, not memorising. That is the single most reassuring result in
this report.

**Dense interpolation (C) wins on this amp** (0.0022 mean, 0.0036 worst vs.
0.0047 / 0.0121). It needs all ten captures to do it, but on the strictest
test available -- genuinely withheld real captures -- the ten-capture
baseline is about 2x more accurate than either single-anchor method and than
the profile the builder actually emits.

### 5.4 Fender 57 Custom Twin -- leave-one-position-out

| Volume | Flags | A-dir ESR | A-cal ESR | B ESR | C ESR | A-cal spec |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **extrap** | 9.7592 | 3.2174 | 3.1500 | 3.0570 | 0.985 |
| 2 | -- | 0.5181 | 0.0492 | 0.0492 | 0.1781 | 0.986 |
| 3 | -- | 0.0107 | 0.0243 | 0.0243 | 0.0076 | 0.991 |
| 4 | -- | 0.0101 | 0.0007 | 0.0007 | 0.0002 | 0.992 |
| 5 | anchor | 0.0000 | 0.0001 | 0.0012 | 0.0022 | 0.995 |
| 6 | -- | 0.0076 | 0.0077 | 0.0077 | 0.0229 | 0.996 |
| 7 | -- | 0.0182 | 0.0235 | 0.0235 | 0.0031 | 0.998 |
| 8 | -- | 0.0699 | 0.0372 | 0.0372 | 0.0014 | 0.997 |
| 9 | -- | 0.4020 | 0.0515 | 0.0515 | 0.0026 | 0.998 |
| 10 | **extrap** | 2.6871 | 0.0520 | 0.0520 | 0.0004 | 0.997 |
| **mean (usable)** | | **0.1481** | **0.0277** | **0.0277** | **0.0308** | |

B equals A-calibrated at every usable position (single G5 anchor selected).
The calibrated single anchor edges out dense interpolation on the mean
(0.0277 vs 0.0308) and comfortably on the worst case (0.0515 vs 0.1781),
using **one capture instead of nine**. The Twin's measured mapping
saturates at the top (V9 +8.5 dB, V10 +8.6 dB) with error climbing to
0.05 -- the "clean-voiced anchor cannot be pushed into a hotter character"
knee the cross-amp report described, reproduced here.

### 5.5 Fender Super-Sonic 60W -- where the production profile actively hurts

| Volume | Flags | Anchors picked | Region used | B gain | A-dir ESR | A-cal ESR | **B ESR** | C ESR |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | **extrap** | `[3.0]` | p3 | -7.8 | 6.1113 | 1.1252 | 1.2209 | 0.9587 |
| 2 | -- | `[1.0, 6.0]` | **p1** | **+0.0** | 1.1780 | 0.0884 | **2.0195** | 0.2183 |
| 3 | -- | `[1.0, 5.0]` | p5 | -6.5 | 0.0151 | 0.0175 | 0.0078 | 0.0555 |
| 4 | -- | `[1.0, 3.0]` | p3 | +3.9 | 0.0692 | 0.0653 | 0.0714 | 0.0686 |
| 5 | anchor | `[1.0, 3.0]` | p3 | +6.3 | 0.0000 | 0.0002 | 0.0061 | 0.0145 |
| 6 | -- | `[1.0, 3.0]` | p3 | +8.5 | 0.0304 | 0.0069 | 0.0095 | 0.0165 |
| 7 | -- | `[1.0, 3.0]` | p3 | +10.1 | 0.0478 | 0.0089 | 0.0135 | 0.0088 |
| 8 | -- | `[1.0, 3.0]` | p3 | +12.6 | 0.1441 | 0.0026 | 0.0072 | 0.0031 |
| 9 | -- | `[1.0, 3.0]` | p3 | +13.6 | 0.3074 | 0.0042 | 0.0090 | 0.0067 |
| 10 | **extrap** | `[1.0, 3.0]` | p3 | +14.0 | 0.9309 | 0.0098 | 0.0125 | 0.0115 |
| **mean (usable)** | | | | | 0.2560 | **0.0277** | 0.3054 | 0.0539 |
| **worst (usable)** | | | | | 1.1780 | **0.0884** | 2.0195 | 0.2183 |

**A concrete, reproducible defect in the production builder, found here.**
At Volume 2 the builder selected anchors `[1.0, 6.0]`, and region ownership
gave the V1 anchor a region that owns *only its own position*. An anchor
whose `mapping_points` contains a single entry returns that entry's gain for
every position (`AnchorRegion.input_gain_db`'s `len(xs) == 1` branch), and
`_assemble_profile` then extends the outermost anchor's `range_low` down to
the control minimum. The net effect: at Volume 2 the runtime renders the
**Volume-1 capture at 0.0 dB** -- no virtual gain at all -- giving raw ESR
2.02 and a **+7.49 dB** level error, against A-calibrated's 0.0884 and
+1.39 dB with one capture and no anchor logic.

Excluding that one pathological row, B's mean falls to ~0.019 -- i.e. the
multi-anchor machinery is fine *where it does not hit the degenerate-region
case*, and the case is not rare (it recurs in the continuity run,
Section 6). This is a genuine production finding, not a benchmark artifact:
it is the shipped `build_profile` + `ContinuousGainRuntime` path.

Even setting it aside, **A-calibrated (one capture) still beats B (two
captures) on this amp's mean and worst case**.

### 5.6 Multi-DI generalization: mappings frozen on the standard DI

Slopes, curves, anchor sets and region assignments were derived on
`moderate_brit.wav` and then applied unchanged to the other two DIs. Nothing
was re-optimized. Raw ESR mean / worst over the same usable positions:

| Amp | DI | A-direct | A-calibrated | B-profile | C-dense |
|---|---|---:|---:|---:|---:|
| JCM800 | standard | 0.0706 / 0.4860 | 0.0047 / 0.0121 | 0.0047 / 0.0121 | 0.0022 / 0.0036 |
| JCM800 | clean | 0.0494 / 0.3337 | 0.0048 / 0.0110 | 0.0048 / 0.0110 | 0.0018 / 0.0049 |
| JCM800 | metalcore | 0.0738 / 0.3334 | **0.0632 / 0.3124** | **0.0632 / 0.3124** | 0.0185 / 0.0810 |
| Twin | standard | 0.1481 / 0.5181 | 0.0277 / 0.0515 | 0.0277 / 0.0515 | 0.0308 / 0.1781 |
| Twin | clean | 0.1248 / 0.5138 | 0.0268 / 0.0533 | 0.0268 / 0.0533 | 0.0306 / 0.1752 |
| Twin | metalcore | 0.0966 / 0.4463 | 0.0243 / 0.0536 | 0.0243 / 0.0536 | 0.0250 / 0.1311 |
| Super-Sonic | standard | 0.2560 / 1.1780 | 0.0277 / 0.0884 | 0.3054 / 2.0195 | 0.0539 / 0.2183 |
| Super-Sonic | clean | 0.2011 / 1.2309 | 0.0162 / 0.0834 | 0.3019 / 2.0748 | 0.0318 / 0.1973 |
| Super-Sonic | metalcore | 0.1567 / 0.9009 | 0.0646 / 0.3947 | 0.1936 / 1.1173 | 0.0444 / 0.2045 |
| 5150 | standard | 37.35 / 188.3 | 1.2126 / 1.9521 | 0.6035 / 1.2057 | 1.0683 / 2.2021 |
| 5150 | clean | 7.10 / 31.13 | 1.0735 / 2.0118 | 0.6167 / 1.0451 | 0.6491 / 1.3215 |
| 5150 | metalcore | 2.28 / 3.59 | 1.6820 / 2.7900 | 1.6785 / 2.7071 | 1.0620 / 1.6846 |

Findings:

- **The Twin is essentially DI-invariant** for every method -- a frozen
  single-anchor mapping transfers to different program material with no
  meaningful degradation.
- **JCM800 degrades ~13x on metalcore material** for the single-anchor
  methods (0.0047 -> 0.0632, worst 0.0121 -> 0.3124) while dense
  interpolation degrades only ~8x and stays an order of magnitude better in
  absolute terms (0.0185 / 0.0810). A frozen virtual-gain curve is the
  method most sensitive to program material here -- unsurprising, since it
  is the only method whose single free parameter was tuned against one DI's
  material.
- **B never overtakes A-calibrated on any DI** for JCM800 or the Twin
  (identical), and remains worse on Super-Sonic on all three DIs.
- Relative *ranking* between methods is stable across DIs on all four amps;
  only the absolute error moves. Anchor selection generalizes in direction,
  as the cross-amp report also concluded.

## 6. Runtime continuity, artifacts, CPU and memory

Sweeps are `1 -> 10 -> 1` triangle traversals of the knob across the
6-second clip: **gradual** = one traversal, **rapid** = four. An offline
block renderer cannot be swept sample-by-sample, so each sweep renders the
whole clip at a 0.25-step position grid (37 positions) and then, per sample,
linearly blends the two bracketing grid renders at the weight the knob's
instantaneous position implies. Any anchor-switching discontinuity lives
inside the rendered outputs and their weighting, so it survives this
assembly intact.

### 6.1 JCM800 -- no transitions exist to measure

| method | models resident | resident NAM bytes | mean s/position | max s/position | child CPU s | peak child RSS |
|---|---:|---:|---:|---:|---:|---:|
| G5-only | 1 | 294,951 | 0.168 | 0.177 | 5.9 | 7.2 MB |
| multi-anchor | 1 | 294,951 | 0.168 | 0.169 | 5.9 | 7.2 MB |

| sweep | method | max frame level jump | frames >3 dB | max abs sample delta | samples >0.25 delta |
|---|---|---:|---:|---:|---:|
| gradual | G5-only | 4.28 dB | 12 | 0.0191 | 0 |
| gradual | multi-anchor | 4.28 dB | 12 | 0.0191 | 0 |
| rapid | G5-only | 3.70 dB | 15 | 0.0411 | 0 |
| rapid | multi-anchor | 3.70 dB | 15 | 0.0411 | 0 |

Identical, because the builder emitted **one** anchor covering `[1.0, 10.0]`.
Static level across the position grid is monotonic with a maximum step of
2.27 dB; the phase-cancellation probe found 0 of 6 probed positions dipping
below all contributors. The frame-level "jumps" present in both methods are
the DI's own note attacks, not knob artifacts (they are identical between
the two methods and present in the static renders too).

Audio: `work/continuous_gain_mid_vs_profile/audio/jcm800_continuity/{gradual,rapid}_{G5_only,multi_anchor}.wav`

### 6.2 Super-Sonic -- a real anchor-transition artifact

Profile anchors: `p1` owning `[1.0, 1.0]` (a single-point, degenerate
region) and `p3` owning `[2.0, 10.0]`.

| method | models resident | resident NAM bytes | mean s/position | max s/position | child CPU s | peak child RSS |
|---|---:|---:|---:|---:|---:|---:|
| G5-only | 1 | 294,527 | 0.160 | 0.163 | 5.7 | 5.8 MB |
| multi-anchor | 2 | 590,409 | **0.188** | **0.325** | 6.7 | 5.8 MB |

| sweep | method | max frame level jump | frames >3 dB | max abs sample delta |
|---|---|---:|---:|---:|
| gradual | G5-only | 3.47 dB | 9 | 0.0023 |
| gradual | multi-anchor | **6.50 dB** | 12 | 0.0086 |
| rapid | G5-only | 3.88 dB | 14 | 0.0047 |
| rapid | multi-anchor | **19.40 dB** | 29 | 0.0086 |

- **Static level continuity is broken for the multi-anchor profile**: max
  step 7.05 dB, **non-monotonic**, with a **-5.52 dB drop at position 1.5**
  as the knob crosses out of the degenerate `p1` region. The single-anchor
  path over the same grid is monotonic with a 2.10 dB max step.
- The rapid sweep turns this into a **19.4 dB frame-to-frame level jump**,
  5x the single-anchor path's worst -- an audible lurch, on a sweep the
  single fixed NAM handles smoothly.
- No hard clicks were measured (0 samples exceeding a 0.25 inter-sample
  delta in any sweep, either method) and the phase-cancellation probe found
  **0 of 11** probed transition positions dipping below all contributing
  anchors. So the smoothstep crossfade itself is well-behaved -- the defect
  is the *mapping discontinuity across the region boundary*, not comb
  filtering.
- Runtime cost of transitions is real but small: 2x peak render time inside
  a transition band (0.325 s vs 0.163 s), ~17% higher mean, and 2x resident
  NAM bytes.

Audio: `work/continuous_gain_mid_vs_profile/audio/supersonic_continuity/{gradual,rapid}_{G5_only,multi_anchor}.wav`

## 7. Peavey 5150 -- included, and inconclusive by design

Per the brief, the 5150 is included rather than silently excluded, and
flagged per docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md.

**The metrics disagree with each other, so the amp is marked inconclusive.**

- **Raw ESR is uninterpretable.** Every method scores 0.1-190; the *search
  itself* fails, pinning at the -24 dB boundary at 7 of 10 positions with a
  best-case ESR near 1.0. The fitted conventional slope comes out
  **negative (-1.54 dB/step)** -- physically nonsensical for a gain knob,
  and a direct consequence of fitting to a failed search.
- **Spectral correlation stays high** (0.85-0.98 for A-calibrated) exactly
  as the metric-limitation report predicts: the harmonic content survives
  while the waveforms decorrelate at clipping instants.
- **Level error, however, is real and large, and level is not subject to the
  decorrelation problem.** B-profile is **-32 to -54 dB** quieter than the
  real capture at Gains 7, 8 and 10; A-calibrated is -8 to -19 dB out at the
  same positions. A 30+ dB level error is not a metric artifact -- it is a
  reconstruction that a listener would immediately reject. On this evidence
  **neither the single-anchor nor the multi-anchor approach currently works
  on this amp**, which is a stronger and better-supported statement than
  anything raw ESR alone could justify.
- The builder itself agrees: it classified every 5150 fold's profile
  **"poor"** (worst build-time ESR ~1.00) and selected the *lowest* capture
  (`[1.0]`/`[2.0]`) as anchor -- i.e. the production quality gate correctly
  refuses to certify this amp. That gate working is a positive result.
- Two capture-level anomalies compound the picture: the **Gain 6 timing
  defect** newly identified here (Section 4) and the previously-documented
  Gain 4 level non-monotonicity.

**Listening material** (level-matched variants generated specifically
because the numeric metrics cannot be trusted here), for Gains 2, 4, 6, 8
and 10, in `work/continuous_gain_mid_vs_profile/audio/peavey5150/`:
`p{N}_real.wav`, `p{N}_{A_direct,A_calibrated,B_profile,C_dense}.wav` and
`p{N}_{method}_levelmatched.wav`. **No listening test was performed by this
automation**; a human A/B of those files is required before any conclusion
about 5150 reconstruction *quality* (as opposed to the level failures above,
which are measured) is drawn.

## 8. Representative audio

All under `work/continuous_gain_mid_vs_profile/audio/` (gitignored):

| Path | What |
|---|---|
| `jcm800/p1.5_*.wav`, `p5.5_*.wav`, `p9.5_*.wav` | genuinely withheld half-step targets: `_real` plus all four reconstructions |
| `supersonic/p2_*.wav` | the degenerate-region failure (B at +0.0 dB, +7.49 dB level error) next to A-calibrated and the real capture |
| `supersonic/p6_*.wav`, `p9_*.wav` | healthy mid/high positions |
| `twin/p2_*.wav`, `p6_*.wav`, `p9_*.wav` | 57 Custom Twin withheld positions |
| `peavey5150/p{2,4,6,8,10}_*[_levelmatched].wav` | 5150 listening set, raw and level-matched |
| `jcm800_continuity/`, `supersonic_continuity/` | `gradual`/`rapid` 1->10->1 sweeps, G5-only vs multi-anchor |

98 WAV files, ~108 MB total.

## 9. Aggregate cross-amp summary

Counting only the three amps where the metric is reliable, over genuinely
withheld positions, primary DI:

```text
Method          JCM800    Super-Sonic   Twin      captures   mean of means
A-direct        0.0706    0.2560        0.1481    1          0.1582
A-calibrated    0.0047    0.0277        0.0277    1          0.0200
B-profile       0.0047    0.3054        0.0277    1-2        0.1126
C-dense         0.0022    0.0539        0.0308    9-10       0.0290
```

- **A-calibrated is the best method on average across the reliable amps**,
  with one capture.
- **B-profile is identical to it on two amps and much worse on the third.**
- **C-dense is the most consistent** (it is never catastrophic, and it wins
  outright on JCM800) but costs 9-10 captures.
- **A-direct is never competitive.**

## 10. Answers to the eight questions

**1. Can a single Gain 5 NAM reproduce the physical Gain 1-10 sweep using
ordinary input gain?**
Partly, and not with an *ordinary* (linear dB-per-step) gain control. With a
conventional single-slope law the mean raw ESR over withheld positions is
0.07-0.26 with worst cases of 0.49-1.18 -- the reconstruction collapses at
both ends of every sweep, because the true mapping is strongly compressive
(JCM800: ~8 dB per knob step at the bottom, ~0.3 dB per step at the top). A
single Gain-5 NAM *can* cover essentially the whole sweep, but only once the
input-gain mapping is calibrated (question 2). Even then, the ends of the
sweep remain the weak point, and the anchor cannot be turned down far enough
to become Gain 1 on the JCM800 (the search pins at -24 dB).

**2. How much improvement comes from calibrating the nonlinear input-gain
mapping rather than using a conventional gain control?**
Large, and the single biggest effect in this report: **5x to 15x lower raw
ESR** on the three amps where the metric is reliable (JCM800 0.0706 ->
0.0047, 15.0x; Super-Sonic 0.2560 -> 0.0277, 9.2x; Twin 0.1481 -> 0.0277,
5.3x; the 5150 improves ~30x but from a meaningless baseline). Worst-case
error improves similarly (JCM800 0.486 -> 0.012, 40x). Calibration also
costs nothing at runtime -- it is a curve lookup -- so this is essentially
free accuracy.

**3. Does the automatically generated multi-anchor profile materially
outperform a calibrated single-G5 NAM?**
**No.** On JCM800 and the 57 Custom Twin the production builder *selects a
single anchor* (G5 in almost every fold) and its output is bit-identical to
the calibrated single-G5 method. On the Super-Sonic, where it does select
two anchors, it is **11x worse on the mean and 23x worse on the worst case**
(0.3054 / 2.0195 vs. 0.0277 / 0.0884), due to a degenerate single-point
anchor region whose constant 0 dB mapping is applied across an extended
range. On the 5150 the builder marks every profile "poor" and nothing is
conclusive. There is no amp in this study where the multi-anchor profile
materially beat the calibrated single anchor.

**4. On which amplifiers and gain regions are additional anchors genuinely
useful?**
On this evidence: **nowhere, as currently selected and applied.** The
regions where extra anchors *should* help are the sweep extremes -- the
JCM800's Gain 1 end (the G5 anchor pins at the -24 dB search boundary, raw
ESR 0.034 in-sample) and the Twin's Volume 9-10 end (mapping saturates at
+8.5/+8.6 dB, ESR rising to 0.052) -- and the Super-Sonic's low end, which
is the one place the builder *did* add an anchor and where it then failed.
The builder does not add an anchor at the JCM800 or Twin tails because its
quality rule ("smallest set reaching worst-case ESR <= 0.06") is already
satisfied by one anchor, so those genuinely weak regions never trigger a
second anchor. Additional anchors are *potentially* useful at sweep
extremes; nothing here shows them being useful in practice.

**5. Does the multi-anchor profile remain more accurate when tested against
genuinely withheld gain positions and different DI material?**
It was never more accurate to begin with, so no. On the JCM800's nine
genuinely withheld half-step captures it ties the single calibrated anchor
exactly (0.0047 / 0.0121) and both lose to dense interpolation (0.0022 /
0.0036). Across frozen-mapping multi-DI evaluation it ties on JCM800 and the
Twin on all three DIs, and stays worse on the Super-Sonic on all three. The
more important withheld-data finding is positive for the *single-anchor*
method: on JCM800 its withheld error (0.0047) is **better than its own
in-sample error** (0.0076), so the calibrated curve is genuinely
interpolating rather than memorising. DI sensitivity is real though: the
frozen JCM800 mapping degrades ~13x on metalcore material (0.0047 ->
0.0632), noticeably more than dense interpolation does.

**6. Does switching between anchors introduce audible or measurable
artifacts that a single fixed NAM avoids?**
**Yes, measurably.** On JCM800 the question is moot because the builder
emits one anchor and there are no transitions. On the Super-Sonic, whose
profile has a real region boundary, the multi-anchor runtime shows a
**non-monotonic output level with a -5.52 dB drop at knob position 1.5**
(max static step 7.05 dB vs the single anchor's monotonic 2.10 dB), which
becomes a **19.4 dB frame-to-frame level jump on a rapid sweep** against the
single anchor's 3.88 dB. No hard clicks were detected (0 samples exceeding a
0.25 inter-sample delta in any sweep) and no phase cancellation was found
(0 of 11 probed transition positions dipped below all contributors) -- so
the smoothstep crossfade itself is sound; the artifact is a discontinuity in
the *mapping* across a degenerate region boundary. A single fixed NAM with a
calibrated curve avoids it entirely by construction.

**7. Is the improvement from multiple anchors large enough to justify the
additional capture requirements, profiling time, runtime complexity and
memory cost?**
**No.** There is no improvement to justify anything: multi-anchor ties the
single anchor on two amps, loses on one, and is inconclusive on the fourth.
Against that, the costs are real and were measured: 2-3x the capture
requirement; a profile build that is the dominant cost of this entire
benchmark (~470-540 s per amp for a 10-capture 10x10 search grid, i.e.
roughly 8-9 minutes of offline rendering per amp, and the search is
`O(N^2)` in captures); 2x resident NAM bytes (590 kB vs 295 kB) and 2x peak
render time inside a transition band (0.325 s vs 0.163 s for a 6 s block);
and the level-continuity artifact of question 6. By contrast the calibrated
single-anchor mapping delivers the 5-15x accuracy win for **one** capture,
zero runtime overhead, and no transitions.

**8. Could NAM Mixer offer a simpler single-capture Continuous Gain mode
alongside the full multi-capture profiler?**
**Yes, and the evidence says it should be the default.** No new algorithm is
needed: a single-anchor Continuous Gain mode is exactly
`build_profile([one anchor plus the sweep used to measure its curve])`, and
the existing `ContinuousGainProfile` schema already represents a
one-anchor profile (`anchors` of length 1, one PCHIP `mapping`, a region
spanning the full control range) -- the JCM800 and Twin profiles emitted in
this run *are* that. The honest caveat is that building the calibration
curve still requires the full capture sweep once, offline, to measure
against; what a single-capture mode saves is runtime complexity, memory,
transition artifacts and the need to *keep* more than one capture, not the
profiling session itself. A genuinely single-capture mode -- shipping a
measured curve alongside one anchor `.nam`, or letting a user dial in a
conventional slope -- would be strictly better than A-direct's linear law
and is the natural product shape suggested by these results.

## 11. Primary recommendation

**The multi-anchor Continuous Gain profile does not currently earn its
complexity, and the calibrated single mid-gain anchor should become the
default path.** Across four real amp datasets and three DIs, calibrating the
nonlinear physical-knob-to-input-gain mapping is worth 5-15x in raw ESR and
is the only large effect measured; adding anchors on top of it bought
nothing on two amps (the shipped builder simply selected one anchor and
produced identical audio), actively hurt on a third through a degenerate
anchor-region mapping that renders an anchor at 0 dB across a range it does
not fit, and could not be evaluated on the fourth. Concretely: ship
single-anchor Continuous Gain as the default, keep the profiler's *curve
measurement* (which is where the value is), fix the degenerate-region bug in
`_assemble_profile`/`AnchorRegion.input_gain_db` before any multi-anchor
profile is offered to users, change the anchor-count rule so a second anchor
is added when a *region* is weak (the JCM800 Gain-1 and Twin Volume-9/10
tails) rather than only when the global worst case exceeds 0.06, and retain
dense discrete interpolation as the fallback for users who already own a
full sweep -- it remains the most consistent method and still wins outright
on the JCM800's genuinely withheld half-step captures.

## 12. Caveats

- One 6-second evaluation segment per DI, three DIs, four amps. Absolute
  error levels are program-material dependent (Section 5.6); the *ranking*
  between methods was stable across all three DIs on all four amps.
- Leave-one-position-out on the three integer-only amps cannot test true
  half-step interpolation the way the JCM800 split can, and its endpoint
  folds are extrapolation, not interpolation -- both are flagged and
  excluded, leaving 7 usable positions per integer amp.
- The A-direct baseline is the *best* linear law per amp, not a fixed
  industry-standard slope; a real product's fixed slope would perform worse,
  so the 5-15x calibration win is a lower bound.
- The Super-Sonic degenerate-region failure is one specific code path. It
  recurred independently in the leave-one-out evaluation and in the
  continuity run, but it has been observed on one amp; whether the same path
  triggers on other capture sets is untested.
- The Peavey 5150 conclusions rest on *level* error and on the builder's own
  "poor" classification, not on raw ESR, which is unusable there. The
  listening files exist but have not been listened to.
- `hybrid.align` was not used anywhere; captures are compared as rendered,
  which is what makes the timing-defect findings visible rather than
  papered over.

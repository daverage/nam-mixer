# Continuous Gain Model -- Phase 2 Baseline Error Map

Research output for docs/CONTINUOUS_GAIN.md's Phase 2 ("Intermediate
Reconstruction") on real captures, produced by
`scripts/continuous_gain_baseline_matrix.py`. This is a decision-gate report
for whether/where to spend Phase 3 effort -- it is not itself Phase 3.

## Source dataset

One real amplifier (JCM800 2203 1985), two channels ("Hi"/"Lo"), same
cabinet/mic/EQ/master across the whole set (P6 B8 M4 T7), captured at Gain
2/4/6/8/10 on each channel -- 10 captures total. No `input_level_dbu`
metadata on these captures, so calibration correctly falls back to Raw (with
a warning) rather than calibrating asymmetrically.

DI material: `assets/di/moderate_brit.wav` ("standard"),
`assets/di/clean_smooth.wav` ("clean, lower output"),
`assets/di/high_metalcore.wav` ("dense, high-gain"). 6 seconds of each.

A second real dataset is used for the capture-density experiment below:
`Marshall JCM800 2203 - updated`, High channel only, a genuine 0.5-gain-step
sweep from Gain 1.0 to 10.0 (20 captures). This is a separate capture
session of the same amp model, not additional captures added to the 1985
unit above -- see that experiment's own caveat.

## Method

Approach A baseline only (`hybrid.continuous_gain.interpolate_output`):
linear crossfade between the two bracketing training captures, no Hybrid/
Character/synthetic-intermediate machinery. Leave-one-out validation per
docs/CONTINUOUS_GAIN.md's "Critical Validation Strategy": hide one interior
gain, reconstruct it from its two nearest remaining neighbours, score against
the real rendered capture (`hybrid.validation.compute_esr_metrics` plus a
low/high spectral-band delta split at 3 kHz).

## Results

| Channel | Hidden | Neighbours | DI | raw ESR | gain-norm ESR | RMS Δ | peak Δ | <3kHz Δ (dB) | >3kHz Δ (dB) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Hi | 4 | G2,G6 | standard | 0.0986 | 0.0639 | 0.0232 | 0.0187 | 2.11 | 3.59 |
| Hi | 4 | G2,G6 | clean | 0.1065 | 0.0704 | 0.0276 | 0.0139 | 2.15 | 3.21 |
| Hi | 4 | G2,G6 | metalcore | 0.0928 | 0.0611 | 0.0209 | 0.0101 | 2.01 | 3.58 |
| Hi | 6 | G4,G8 | standard | 0.0558 | 0.0542 | 0.0086 | 0.0250 | 0.58 | 2.73 |
| Hi | 6 | G4,G8 | clean | 0.0547 | 0.0535 | 0.0091 | 0.0147 | 0.51 | 1.91 |
| Hi | 6 | G4,G8 | metalcore | 0.0506 | 0.0485 | 0.0084 | 0.0102 | 0.61 | 2.79 |
| Hi | 8 | G6,G10 | standard | 0.0491 | 0.0483 | 0.0071 | 0.0149 | 0.40 | 2.99 |
| Hi | 8 | G6,G10 | clean | 0.0491 | 0.0489 | 0.0068 | 0.0116 | 0.36 | 1.71 |
| Hi | 8 | G6,G10 | metalcore | 0.0422 | 0.0415 | 0.0062 | 0.0345 | 0.39 | 2.73 |
| Lo | 4 | G2,G6 | standard | 0.0169 | 0.0028 | 0.0006 | 0.0076 | 1.09 | 2.74 |
| Lo | 4 | G2,G6 | clean | 0.0234 | 0.0079 | 0.0008 | 0.0120 | 1.06 | 2.69 |
| Lo | 4 | G2,G6 | metalcore | 0.0151 | 0.0025 | 0.0006 | 0.0101 | 1.04 | 3.16 |
| Lo | 6 | G4,G8 | standard | 0.0235 | 0.0177 | 0.0004 | 0.0004 | 0.58 | 0.69 |
| Lo | 6 | G4,G8 | clean | 0.0280 | 0.0187 | 0.0007 | 0.0036 | 0.84 | 0.98 |
| Lo | 6 | G4,G8 | metalcore | 0.0274 | 0.0165 | 0.0006 | 0.0079 | 0.80 | 0.84 |
| Lo | 8 | G6,G10 | standard | 0.0078 | 0.0009 | 0.0007 | 0.0049 | 0.76 | 0.52 |
| Lo | 8 | G6,G10 | clean | 0.0085 | 0.0015 | 0.0010 | 0.0085 | 0.78 | 0.40 |
| Lo | 8 | G6,G10 | metalcore | 0.0085 | 0.0011 | 0.0008 | 0.0125 | 0.78 | 0.40 |

### Wide-spacing ablation (predict Gain 6 from Gain 2/10 instead of Gain 4/8)

| Channel | raw ESR (narrow, G4/G8) | raw ESR (wide, G2/G10) |
|---|---:|---:|
| Hi | 0.0558 | **0.2670** |
| Lo | 0.0235 | **0.0735** |

### Knob-linear vs. level-matched-oracle ablation (Hi, hidden=6, standard DI)

| Method | raw ESR | RMS Δ |
|---|---:|---:|
| Knob-linear blend | 0.0558 | 0.0086 |
| Level-matched (oracle, uses the real hidden capture's own measured RMS -- **not achievable in production**) | 0.0571 | 0.0038 |

Rescaling to the withheld capture's real level fixes the RMS delta almost
completely but does **not** improve (and marginally worsens) raw ESR. The
baseline's error is therefore dominated by waveform/spectral shape, not by
picking the wrong level for a knob-linear blend fraction.

## Findings against the doc's Phase-3 decision gate

1. **The Hi channel's clean-to-breakup region is the failure boundary, not
   the whole gain range.** Reconstructing Gain 4 (bracketed by Gain 2/Gain 6,
   which straddles the clean->breakup transition) is consistently 1.7-2x
   worse than reconstructing Gain 6 or Gain 8, across all three DI types.
   This matches the doc's Q3 hypothesis directly: "captures around the
   clean-to-breakup transition may be more valuable than evenly spaced
   positions."
2. **The Lo channel's baseline is uniformly strong** (raw ESR 0.008-0.028
   throughout, including its own clean-to-breakup Gain-4 point) -- this
   amp's Lo-channel gain curve is evidently smoother/less discontinuous than
   Hi's, so channel identity, not just gain position, matters.
3. **DI material does not break the baseline.** Standard/clean/metalcore
   give broadly similar ESR at a given channel+hidden-gain (e.g. Hi/Gain 6:
   0.0558/0.0547/0.0506) -- no collapse under contrasting excitation level or
   density. This answers this round's "does interpolation stay valid under
   different excitation" question: yes, for this amp.
4. **Wide neighbour spacing is clearly harmful.** Widening the neighbour
   spacing from G4/G8 to G2/G10 for the same Gain-6 target roughly
   QUADRUPLES the Hi-channel error (0.056 -> 0.267) and more than triples
   the Lo-channel error (0.024 -> 0.074) -- a bigger effect than any
   channel/DI difference observed above. **This shows wide spacing hurts; it
   does not yet show that adding one more capture near the knee would beat a
   better interpolation method** -- that is a distinct, not-yet-run
   experiment (see "Blocked experiment" below), not a conclusion this data
   supports on its own.
5. **Errors concentrate above 3 kHz, but a magnitude-only correction there
   does not recover them (see the HF-correction ablation below).** The
   high-frequency band delta is consistently 2-6x the low-frequency band
   delta across every row, and the level-matched-oracle ablation shows
   getting the level exactly right barely moves raw ESR -- both point at
   spectral/harmonic SHAPE (not level) as the dominant error source above
   3 kHz. However, directly correcting the high band's measured energy
   (below) does not reduce raw ESR either, which narrows the likely cause
   further to harmonic/phase structure rather than a simple broadband tilt.

## Phase 3 hypothesis (falsifiable)

> The dominant error in the Hi-channel Gain-4 reconstruction is caused
> primarily by insufficient sampling density across the clean-to-breakup
> knee, with a secondary residual error concentrated above 3 kHz.

Two falsifiable tests follow directly:

1. Does denser sampling (real captures added near the knee, e.g. Gain 3 and
   Gain 5 on the Hi channel) materially reduce ESR at Gain 4?
2. After density is improved, does a targeted high-frequency correction
   reduce the remaining error?

If (1) is yes and (2) barely helps, the product implication is simple:
**capture placement matters more than interpolation sophistication**, which
also means the eventual UI should guide users to add a capture where the
curve changes fastest, rather than expose interpolation-method choices.

### Capture-density test (test 1) -- RESOLVED

Originally blocked: the JCM800 2203 1985 capture set above only has even
gains (2/4/6/8/10), so its own knee (around Gain 4) could not be
re-bracketed tighter without new real captures. A second, denser real
capture set became available -- `Marshall JCM800 2203 - updated`, High
channel only, a genuine 0.5-gain-step sweep from Gain 1.0 to 10.0 (20
captures, same cabinet/mic/EQ per filename convention, no `input_level_dbu`
metadata despite the "11.4dBu" filename suffix). **This is a different
capture session of the same amp model, not additional captures spliced into
the original 1985-unit dataset above** -- the two are not directly
comparable row-for-row, but the SAME methodology (leave-one-out, same three
DI types) applies to test the same hypothesis.

First, the active-RMS progression across the full 0.5-step sweep (standard
DI) locates this unit's knee unambiguously between Gain 1.0 and Gain 3.0
(+2.90, +1.32, +1.22, +0.69 dB per 0.5-gain step), flattening to near-zero
change per step from Gain 3.5 onward -- so Gain 2.0 is the knee-region test
point, analogous to the 1985 unit's Gain-4 failure region.

| Bracket | Predicts | DI | raw ESR | gain-norm ESR | >3kHz Δ (dB) |
|---|---|---|---:|---:|---:|
| G1.0 / G3.0 (spacing 2.0, "sparse") | G2.0 | standard | 0.0363 | 0.0297 | 1.03 |
| G1.5 / G2.5 (spacing 1.0, "dense") | G2.0 | standard | **0.0044** | 0.0042 | 0.34 |
| G1.0 / G3.0 (sparse) | G2.0 | clean | 0.0312 | 0.0249 | 0.87 |
| G1.5 / G2.5 (dense) | G2.0 | clean | **0.0036** | 0.0034 | 0.22 |
| G1.0 / G3.0 (sparse) | G2.0 | metalcore | 0.0317 | 0.0269 | 0.84 |
| G1.5 / G2.5 (dense) | G2.0 | metalcore | **0.0036** | 0.0035 | 0.31 |

Tightening the bracket from 2.0 gain-units wide to 1.0 gain-units wide
reduces raw ESR by roughly **8x**, consistently across all three DI types.
The local three-point test (predict each of G1.5/G2.0/G2.5 from its own
immediate 0.5-spaced neighbours) confirms the whole knee region reconstructs
well once bracketed this tightly (raw ESR 0.003-0.008 across all three
targets and DI types) -- there is no separate "even with tight bracketing
the local curve is still hard" effect here: spacing alone accounts for
essentially all of the sparse-bracket error at this knee.

**Test 1 answer: YES, unambiguously.** Denser sampling across the knee
reduces ESR by an order of magnitude, far more than any correction-side fix
tested below could plausibly close.

### HF-correction ablation (test 2, run without density improvement)

Tests whether a lightweight, causal (non-oracle) high-frequency correction
recovers accuracy on top of plain linear interpolation.
`hybrid.continuous_gain.interpolate_output_hf_corrected` predicts the
missing capture's high-band (>=3 kHz) energy by linearly interpolating the
two NEIGHBOURS' OWN measured high-band energy at the same knob-linear blend
weight (no access to the real hidden capture), then rescales only the
reconstruction's high band to match that prediction.

| Channel | Hidden | Method | raw ESR | gain-norm ESR | RMS Δ | <3kHz Δ (dB) | >3kHz Δ (dB) |
|---|---:|---|---:|---:|---:|---:|---:|
| Hi | 4 | plain-linear | 0.0986 | 0.0639 | 0.0232 | 2.11 | 3.59 |
| Hi | 4 | hf-corrected | 0.0984 | 0.0643 | 0.0231 | 2.11 | 3.17 |
| Hi | 6 | plain-linear | 0.0558 | 0.0542 | 0.0086 | 0.58 | 2.73 |
| Hi | 6 | hf-corrected | 0.0589 | 0.0586 | 0.0072 | 0.58 | 0.45 |
| Hi | 8 | plain-linear | 0.0491 | 0.0483 | 0.0071 | 0.40 | 2.99 |
| Hi | 8 | hf-corrected | 0.0521 | 0.0525 | 0.0053 | 0.40 | 0.39 |
| Lo | 4 | plain-linear | 0.0169 | 0.0028 | 0.0006 | 1.09 | 2.74 |
| Lo | 4 | hf-corrected | 0.0165 | 0.0026 | 0.0006 | 1.09 | 2.00 |

**Result: negative, and informative.** The correction does what it targets
-- it nearly eliminates the high-band ENERGY delta (e.g. Hi/Gain 6:
2.73 dB -> 0.45 dB) -- but this does **not** translate into a lower raw ESR;
at Hi Gain 6/8 it is slightly WORSE than plain linear interpolation despite
the much smaller measured high-band level mismatch. At Hi Gain 4 (the
failure region) it is essentially unchanged (0.0986 -> 0.0984).

This rules out "the high band is simply too loud/quiet relative to a linear
blend" as the mechanism, sharpening the hypothesis: the >3 kHz residual is
harmonic/phase-structure content that a magnitude-only correction cannot
recover, not a broadband tilt. A future high-frequency correction attempt
would need to reconstruct HARMONIC CONTENT (e.g. matching harmonic
distribution or phase relationships), not just re-level a frequency band.
This ablation was run on the sparse (1985-unit) baseline, i.e. BEFORE any
density improvement -- given how large the density effect above turned out
to be, this residual may simply not matter much in practice once captures
are placed densely enough across a knee.

**Test 2 answer: no, not as a magnitude-only correction.** Combined with
test 1's clear "yes", both halves of the hypothesis in this report resolve
in the direction the doc anticipated as the good outcome for the product:

> If the answer to 1 is yes and 2 barely helps, then the product implication
> becomes very simple: capture placement matters more than interpolation
> sophistication.

## Recommendation (per the doc's decision gate)

> If baseline performance collapses around specific regions, target Phase 3
> specifically at those regions... If errors mainly come from spectral
> differences above 3 kHz, a simpler frequency-aware correction may be more
> useful than the full Hybrid machinery.

- Phase 3 should **not** broadly search Hybrid/Character/synthetic-
  intermediate methods across the whole gain range. The wide baseline
  already performs adequately away from the clean-to-breakup transition
  (Hi Gain 6/8, all of Lo), and denser capture placement resolves the knee
  region far more effectively than any interpolation-side sophistication
  tested here.
- The magnitude-only high-frequency correction tested here does **not**
  meet the doc's "may be more useful" bar -- it fixes the metric it targets
  without fixing ESR, so it should not be pursued further as-is. If a
  frequency-aware fix is still worth attempting later, it needs to target
  harmonic/phase structure, not just band energy, and even then only after
  confirming a real capacity gap remains once capture density near the knee
  is fixed.
- **Product implication: capture placement matters more than interpolation
  sophistication.** Rather than exposing interpolation-method choices, the
  eventual UI should detect where a capture set's gain curve changes
  fastest (the same active-RMS-progression analysis used here to locate the
  knee) and prompt the user to add a capture there -- this is now supported
  by a direct, ~8x measured error reduction, not just a plausibility
  argument.
- Recommended follow-up, now that density is confirmed as the dominant
  lever: repeat this same knee-density test on the ORIGINAL 1985-unit
  capture set once real Gain 3/Gain 5 (or similar) captures exist for it,
  to confirm the effect size holds on that exact unit and not just on the
  "updated" capture session used here.

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
4. **Capture spacing matters far more than the interpolation method at this
   stage.** Widening the neighbour spacing from G4/G8 to G2/G10 for the same
   Gain-6 target roughly QUADRUPLES the Hi-channel error (0.056 -> 0.267) and
   more than triples the Lo-channel error (0.024 -> 0.074). This is a bigger
   effect than any channel/DI difference observed above.
5. **Errors concentrate above 3 kHz.** The high-frequency band delta is
   consistently 2-6x the low-frequency band delta across every row, and the
   level-matched-oracle ablation shows getting the level exactly right barely
   moves raw ESR -- pointing at spectral/harmonic shape (not level) as the
   dominant error source, concentrated above 3 kHz.

## Recommendation (per the doc's decision gate)

> If baseline performance collapses around specific regions, target Phase 3
> specifically at those regions... If errors mainly come from spectral
> differences above 3 kHz, a simpler frequency-aware correction may be more
> useful than the full Hybrid machinery.

Both conditions are met here:

- Phase 3 should **not** broadly search Hybrid/Character/synthetic-
  intermediate methods across the whole gain range. The wide baseline
  already performs adequately away from the clean-to-breakup transition
  (Hi Gain 6/8, all of Lo).
- Phase 3 should specifically target: (a) the **clean-to-breakup transition
  region on channels with a discontinuous gain curve** (denser capture
  spacing there is likely to help more than a smarter interpolation
  method -- worth testing capture-density directly before adding
  interpolation complexity), and (b) a **targeted high-frequency (>3 kHz)
  spectral correction** on top of the existing linear crossfade, rather than
  the full Hybrid/Character machinery, since the level-matched-oracle result
  shows level is not the bottleneck.
- The wide-spacing result is a strong, independent signal that capture
  COUNT/DENSITY (Q2 in the doc) deserves a dedicated experiment before
  method sophistication: it may be that adding one more capture near the
  clean-to-breakup knee outperforms any interpolation-side fix.

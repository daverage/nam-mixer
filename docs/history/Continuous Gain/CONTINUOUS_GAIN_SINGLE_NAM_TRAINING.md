# Continuous Gain: training one standard NAM from a physical gain sweep

Phase 1 results for the plan in `CONTINUOUS_GAIN_V2.md`. Scripts: `scripts/single_nam_*.py`. Raw results, logs,
listening WAVs and the trained models are in `work/single_nam/` (gitignored). Amp: Marshall JCM800 2203 High channel.

## Status

Done: ten-capture training (Method A, direct), capture-count reduction (5/3/2), baselines, half-step and
input-level validation, latency-anomaly detection, conflict analysis.

NOT done: Method B (Hybrid-assisted) and Method C (Blended-assisted) targets; Hybrid/Blended code review write-up;
multiple seeds; blind listening (WAVs are generated, no human listening has happened); Fender Super-Sonic
and Peavey 5150 (the 57 Twin is covered in Phase 2 below); alternative three-capture subsets; equal-compute vs converged comparison. Nothing here should be
read as covering those.

## Objective and method

One newly trained standard A2 (PackedWaveNet, 22,783 parameters, official `nam.train.core.train`) `.nam` that
reproduces the JCM800 Gain 1-10 sweep when only ordinary player input gain changes. Playback path is
`DI -> input gain -> one .nam`. No conditioning input, no runtime profile.

Training data (Method A): for each captured integer gain N, input = DI x law(N) dB, target = the REAL capture N
rendered on the ungained DI (output-level progression preserved). Per gain, 115 s of training audio (45 s of the
official NAM input plus clean_smooth, moderate_hotrod, high_thrash), 16 s validation from a different DI
(high_metalcore). A single global constant c (about 1.16) scales all targets to -18 dBFS; evaluation divides it
back out. It is one constant for all gains, so the level progression is untouched.

Held-out test DIs, never used in training or validation: moderate_brit, clean_mayer, bass_rollin (first 20 s each).
Half-steps G1.5-G9.5 were never used for training, the law, or checkpoint selection.

Only the trainer's data split, input-version detection, latency analysis and check hooks are patched (custom input
file); architecture, optimiser, checkpointing and export are stock. Exports load and render in NAMCore
(`hybrid.render`), which is the path every number below uses.

## Control law (frozen before training)

G5 NAM anchor. For each integer gain, the input dB minimising raw ESR against the real capture on 3 s clips of the
TRAINING DIs. Half-steps: PCHIP over the integer law. Values in `work/single_nam/control_law.json`; G1 -23.6,
G2 -14.2, G3 -6.6, G4 -2.6, G5 0, G6 +2.6, G7 +10.4, G8 +13.0, G9 +17.4, G10 +17.4 dB.

Disclosure: the law uses all ten integer captures, and every reduced-capture model received this same law.
The smaller models therefore had calibration information from captures they were not trained on. This is the
"common fixed mapping" condition; a per-configuration mapping was not run.

Baselines: (1) G5 NAM with the same law; (2) G5 NAM with the previously published per-target calibrated table,
which was fitted against captures including half-steps. On these clips they perform almost identically.

## Feasibility and conflicts

G8.5's `.nam` has a latency defect: a click through it peaks at 701 samples versus 10-11 for the other 18 captures
(detected automatically, `control_law.json`). G8.5 is excluded from aggregates. With the 690-sample offset removed
in analysis only, the ten-capture model scores raw ESR 0.072 there, in line with its neighbours. Source files were
not modified.

Conflict check on adjacent integer gains (input dB gap versus target raw ESR): the law is flat between G9 and G10
(gap 0.0 dB), but their targets differ by only 0.006 ESR, because the amp is saturated there, so the two are
effectively the same target for a level-driven model. The steepest target changes (G1->G2 0.277, G7->G8 0.149,
G8->G9 0.168) all have input gaps of 2.6-9.4 dB, so level does separate them. Hence no severe waveform-level
contradiction exists at natural DI level. The level-generalisation test below is where the representational
limit shows up.

## Results (raw ESR, mean of 3 held-out DIs, `*` = in training set, G8.5 excluded)

| Gain | 10 cap | 5 cap | 3 cap | 2 cap | G5 same law |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.0238* | 0.0143* | 0.0085* | 0.0056* | 0.0657 |
| 1.5 | 0.0132 | 0.0088 | 0.0099 | 0.0255 | 0.0481 |
| 2.0 | 0.0087* | 0.0094 | 0.0163 | 0.0569 | 0.0396 |
| 2.5 | 0.0096 | 0.0103 | 0.0248 | 0.0812 | 0.0372 |
| 3.0 | 0.0122* | 0.0133* | 0.0342 | 0.0964 | 0.0341 |
| 3.5 | 0.0088 | 0.0139 | 0.0248 | 0.1425 | 0.0149 |
| 4.0 | 0.0116* | 0.0170 | 0.0193 | 0.1696 | 0.0057 |
| 4.5 | 0.0124 | 0.0155 | 0.0185 | 0.1658 | 0.0038 |
| 5.0 | 0.0161* | 0.0181* | 0.0156* | 0.1620 | 0.0000 |
| 5.5 | 0.0215 | 0.0189 | 0.0192 | 0.1424 | 0.0044 |
| 6.0 | 0.0229* | 0.0187 | 0.0196 | 0.1292 | 0.0052 |
| 6.5 | 0.0542 | 0.0330 | 0.0504 | 0.1271 | 0.0309 |
| 7.0 | 0.0859* | 0.0442* | 0.0902 | 0.1479 | 0.0635 |
| 7.5 | 0.0509 | 0.0767 | 0.0734 | 0.1233 | 0.1023 |
| 8.0 | 0.0497* | 0.0925 | 0.0664 | 0.0952 | 0.1682 |
| 9.0 | 0.0331* | 0.0574 | 0.0254 | 0.0168 | 0.3351 |
| 9.5 | 0.1188 | 0.1651 | 0.1328 | 0.1525 | 0.3628 |
| 10.0 | 0.0365* | 0.0639* | 0.0284* | 0.0180* | 0.3496 |
| mean | 0.0328 | 0.0384 | 0.0377 | 0.1032 | 0.0928 |
| worst | 0.1188 | 0.1651 | 0.1328 | 0.1696 | 0.3628 |

Per-DI, level-delta, spectral-correlation and peak numbers are in `eval_*.json`. Level error of the ten-capture
model is within about 1 dB everywhere (worst -1.1 dB at G1), against up to -2 dB for the G5 baseline.

Exact G5 values at G5 are 0 by construction (the anchor), so a comparison at G5 says nothing.

## What the results show

- **A single standard NAM does learn the sweep.** The exported ten-capture `.nam` reproduces real captures across
  Gain 1-10 with a mean raw ESR of 0.033 against 0.093 for an unchanged G5 NAM under the same law. Its clear wins are
  at the ends: G8-G10 (0.033-0.050 versus 0.17-0.35) and G1-G3.
- **It is not better everywhere.** From G3.5 to G7 the unchanged G5 NAM is better (for example G4.5 0.004 versus
  0.012), because that region is where G5 is nearly exact. The trained model trades a small, roughly uniform error
  (about 0.01-0.02) for large gains at the ends.
- **The weak zone is G6.5-G7.5 and G9.5.** G7 is the worst trained position (0.086) and does worse than the G5 baseline
  at G6.5-G7. G9.5 (0.12) is poor even though G9 and G10 are good; not investigated, it could be a capture
  characteristic or a law artefact.
- **Half-steps are held-out and behave like their neighbours**, with the exceptions above, so the response is
  smooth between trained positions.
- **Capture count.** Three captures (G1, G5, G10) reach mean 0.038 versus 0.033 for ten, with omitted-position
  worst case 0.133 versus 0.119. Five captures were not better than three (0.038), so the extra two gave no
  measured benefit here. Two captures (G1, G10) fail in the middle (0.12-0.17 from G3.5 to G6) because no
  training example ever shows the mid-gain response. Single seed, so differences of a few thousandths between
  10/5/3 are not established. These are direct-training results; no synthetic intermediate targets were used.
- **Input level is a real limitation.** On moderate_brit at +6 dB DI level, ten-capture ESR rises at G7 to 0.288 and
  G8 to 0.180 (0.135 and 0.073 at 0 dB), while the G5 baseline is worse still at G8-G10. At -12 dB the trained model
  is worse than the G5 baseline at G4-G7. The same waveform cannot be attributed to different gain settings, so
  playing hard at low gain is not equivalent to playing softly at high gain.
- **Training cost.** 60 epochs, about 44 minutes for ten captures on MPS; final validation ESR 0.055 (10), 0.064 (5),
  0.046 (2). Validation ESR is pooled over the trained gains and does not track the per-gain quality above.

## What this does and does not support

Supports: continuing towards a single-NAM Continuous Gain product; a three-capture set is a credible starting point.

Does not yet establish: that three captures match ten perceptually (no listening test), that Hybrid or Blended
targets help, that the method transfers to other amps, or robustness across seeds. Listening WAVs (real, new, G5
same law, at G3/G6/G9, actual level) are in `work/single_nam/listening/`. Level-matched variants were not generated.

## Recommended next steps

1. Listen to the WAVs at G6-G7 and G9.5, where numbers are weakest.
2. Add Method B/C intermediate targets built ONLY from the retained captures (for the 3-capture set), and
   measure target error against real half-steps first.
3. Repeat 3- and 10-capture training with multiple seeds; try an alternative three-capture set that includes G7.
4. Add DI level variants to training data and re-run the level test.
5. Then the other amps.

## Reproduce

```
cd scripts
../.venv-a2/bin/python single_nam_step1_analysis.py            # latency, law, conflicts
../.venv-a2/bin/python single_nam_build_dataset.py --gains 1 5 10 --name method_a_3
../.venv-a2/bin/python single_nam_train.py method_a_3 --epochs 60 --name JCM800_ContinuousGain_3Captures
../.venv-a2/bin/python single_nam_evaluate.py <model.nam> --manifest <manifest.json> --label X --trained-gains 1 5 10
../.venv-a2/bin/python single_nam_summary.py
```

## Phase 2: Fender 57 Custom Twin (Channel 1, Volume 1-10)

Same architecture, procedure, DIs, epochs (60) and law method as the JCM800 (`SINGLE_NAM_AMP=twin`; outputs in
`work/single_nam_twin/`). Models: `Twin_ContinuousGain_10Captures.nam`, `Twin_ContinuousGain_3Captures.nam`
(V1, V5, V10). Caveats specific to this amp: the dataset has integer volumes only, so there are no half-step
captures; the 10-capture model has no unseen positions, and the 3-capture model's omitted integers are the only
held-out positions. No latency anomalies (click latency 4-6 samples on all ten). Single seed. No 5- or 2-capture
runs, and the old calibrated-G5 baseline does not exist for this amp.

Frozen law (dB): V1 -21.6, V2 -12.8, V3 -5.0, V4 -2.0, V5 0, V6 +2.2, V7 +5.6, V8 +7.0, V9 +7.8, V10 +7.8.

Raw ESR, mean of 3 held-out DIs (`*` = in that model's training set):

| Volume | 10 cap | 3 cap | V5 same law |
|---:|---:|---:|---:|
| 1 | 0.0262* | 0.0320* | 0.0096 |
| 2 | 0.0215* | 0.0255 | 0.0062 |
| 3 | 0.0166* | 0.0205 | 0.0022 |
| 4 | 0.0135* | 0.0182 | 0.0007 |
| 5 | 0.0106* | 0.0156* | 0.0000 |
| 6 | 0.0080* | 0.0129 | 0.0013 |
| 7 | 0.0071* | 0.0051 | 0.0329 |
| 8 | 0.0134* | 0.0080 | 0.0583 |
| 9 | 0.0201* | 0.0126 | 0.0776 |
| 10 | 0.0210* | 0.0136* | 0.0793 |
| mean | 0.0158 | 0.0164 | 0.0268 |
| worst | 0.0262 | 0.0320 | 0.0793 |

- Same pattern as the JCM800: the trained models have a small, fairly uniform error floor and win clearly at the
  high end (V8-V10 about 0.01-0.02 versus 0.06-0.08), while the unchanged V5 NAM is better at V1-V6, where it is
  nearly exact. Level error is within 0.2 dB everywhere.
- Ten and three captures are equivalent overall (mean 0.0158 versus 0.0164). The three-capture model is better at
  V7-V10 (for example V9 0.013 versus 0.020) and worse at V1-V6 (V4 0.018 versus 0.014). One seed, so these
  differences are not established.
- The high end shows no JCM800-style breakdown: the trained models' worst high-end value is 0.021 (10 cap).
  Input-level tests are flat too: at V8-V10 raw ESR stays 0.008-0.015 for the 3-capture model and 0.012-0.021 for
  the 10-capture model from -12 dB to +6 dB DI, versus 0.03-0.08 for V5 under the same law. At V1-V6 the 3-capture
  model degrades with quiet playing (V1 0.046 at -12 dB).
- The Twin is an easy test for the baseline: V5 plus input gain already reproduces V1-V6 to ESR under 0.01.

### High-gain character metrics (both amps; new, crude)

Mean over 3 DIs of crest-factor difference and above-3 kHz energy difference against the real capture, dB
(0 = matches). These are simple aggregates, not perception.

| Model | Gain | Crest (dB) | HF >3 kHz (dB) |
|---|---|---:|---:|
| JCM800 10 cap | 8 / 9 / 10 | -0.08 / +0.03 / +0.08 | -0.62 / -0.72 / -0.77 |
| JCM800 3 cap | 8 / 9 / 10 | +0.57 / +0.60 / +0.66 | -0.25 / -0.22 / -0.26 |
| JCM800 G5 same law | 8 / 9 / 10 | -0.28 / +0.38 / +0.44 | +0.13 / -0.01 / -0.06 |
| Twin 10 cap | 8 / 9 / 10 | -0.35 / -0.28 / -0.36 | -0.20 / -0.15 / -0.13 |
| Twin 3 cap | 8 / 9 / 10 | -0.42 / -0.37 / -0.45 | -0.32 / -0.27 / -0.25 |

At high gain the trained models are slightly LESS bright than the real captures on both amps (up to 0.8 dB
less energy above 3 kHz), with crest factor within about 0.6 dB. On these metrics they are not more distorted than
the real amp; if anything they are a little darker. The unchanged G5 NAM's high-band energy is closer to the real
capture on the JCM800 but its waveform error is far larger. The same sign on both amps suggests a property of
the training rather than of the JCM800, but that hypothesis (for example a smoothing bias from the loss) has not been
tested. Whether the real G10 character is reproduced still needs listening; the numbers cannot answer it.

### What the Twin adds

It supports the method generalising to a second amp: both trained models reproduce the full sweep at level
error under 0.2 dB and mean ESR 0.016, with no high-end failure. The JCM800's G7 and G9.5 weakness and its strong
level sensitivity are not repeated here, which points to those being amp- or dataset-specific (the JCM800's
sharp knee around G6-G7) rather than a general limit of input-level control. Multi-gain training does not
beat the unchanged V5 NAM at low volumes on this amp, only at V7-V10. Listening WAVs (V3, V6, V8, V9, V10) are in
`work/single_nam_twin/listening/`. Still not done: Super-Sonic, 5150, Hybrid/Blended targets, seeds, listening.

## Phase 3: Fender Super-Sonic 60W (Bassman channel, T5/B5, Volume 1-10)

Same procedure, DIs and 60 epochs (`SINGLE_NAM_AMP=supersonic`, `work/single_nam_supersonic/`). Models
`SuperSonic_ContinuousGain_10Captures.nam` and `..._3Captures.nam` (V1, V5, V10). Single seed, integer volumes only
(no half-steps), no 5/2-capture runs.

**Latency alignment (deviation, applies to this amp and the Peavey only).** A click test shows the captures'
latencies differ: V1 9 samples, V2 17, V3 26, V4-V10 30. Raw ESR is very sensitive to that, and it ruined the first
law fit (V1 ESR 0.95). Every capture is therefore advanced/delayed to the V5 capture's latency (V1 -21, V2 -13, V3 -4,
V7/V8 +1 samples) inside `render_capture`, identically for training targets, evaluation references, the law fit
and the V5 baseline. The trained models thus learn latency-aligned targets.

Frozen law (dB): V1 -20.0, V2 -14.4, V3 -6.4, V4 -2.6, V5 0, V6 +1.8, V7 +3.0, V8 +4.8, V9 +7.2, V10 +7.4. The V5 NAM
fits V1-V3 poorly at any input gain (fit ESR 0.10-0.13): this amp changes tone at low volume, not just level.

Raw ESR, mean of 3 held-out DIs (`*` = in that model's training set):

| Volume | 10 cap | 3 cap | V5 same law |
|---:|---:|---:|---:|
| 1 | 0.0155* | 0.0264* | 0.2127 |
| 2 | 0.0218* | 0.0381 | 0.2373 |
| 3 | 0.0414* | 0.0976 | 0.1723 |
| 4 | 0.0566* | 0.0328 | 0.0093 |
| 5 | 0.0310* | 0.0243* | 0.0000 |
| 6 | 0.0272* | 0.0269 | 0.0056 |
| 7 | 0.0315* | 0.0241 | 0.0340 |
| 8 | 0.0305* | 0.0263 | 0.0442 |
| 9 | 0.0165* | 0.0186 | 0.0180 |
| 10 | 0.0186* | 0.0167* | 0.0334 |
| mean | 0.0291 | 0.0332 | 0.0767 |
| worst | 0.0566 | 0.0976 | 0.2373 |

- The trained models win most clearly at V1-V3 (0.016-0.098 versus 0.17-0.24), where the V5 NAM cannot follow the
  low-volume tone; the V5 NAM is better at V4-V6 and about equal or worse at V7-V10.
- Ten and three captures are again close on average (0.029 versus 0.033), but the 3-capture model's worst omitted
  position is V3 (0.098), which the 10-capture model gets right (0.041). V4 is the 10-capture model's worst (0.057).
- Level error is within 0.3 dB. Character metrics: high-band (>3 kHz) energy is 0.4-2.0 dB LOW (10 cap: -0.9 to -2.0;
  3 cap: -0.4 to -2.0, largest at V3) and crest factor is 0.1-0.9 dB high, so again slightly darker, larger than on
  the JCM800/Twin. Not verified by listening.
- **Input-level weakness.** At a quiet DI (-12 dB) the 10-capture model degrades at V4-V8 (raw ESR 0.19-0.26 versus 0.00-0.05
  for the V5 NAM); the 3-capture model degrades less (0.12-0.19). At normal and +6 dB levels it is fine except V3
  (0.07 / 0.23 for 10 cap). So the model is less trustworthy with a quiet guitar at mid volumes on this amp.

## Phase 4: Peavey 5150 (Gain 1-10, unboosted) - stress test, inconclusive as a method test

`SINGLE_NAM_AMP=peavey SINGLE_NAM_LAW=level`, `work/single_nam_peavey/`. Models
`Peavey5150_ContinuousGain_10Captures.nam` and `..._3Captures.nam` (G1, G5, G10). Single seed, integers only, latency
alignment as for the Super-Sonic (all captures click-latency 73 samples, so effectively a no-op).

**The dataset does not provide a usable input-gain law.** With the same ESR-based law fit as the other amps, the
search hit its boundary (G1 -38 dB) and was non-monotonic (G3 -20.8, G4 -21.4), with fit ESR 0.13-0.64 at G1-G4
(`control_law_esr_search_rejected.json` kept). Switching to matching output RMS level (with a monotone constraint)
did not help, because the real captures' loudness is not monotone in the Gain setting: measured output RMS on
moderate_brit is -18.3, -16.2, -15.3, -18.6, -14.0, -13.9, -14.0, -14.1, -14.4, -14.5 dBFS for G1-G10. Loudness is
flat from G5 to G10 and G4 is quieter than G3. The captures may have been level-normalised or made at differing
settings; not investigated. The resulting law (G1 -34.4, G2 -27.4, G3 -20.6, G4 -20.6, G5 0, G6-G10 all +5.0 dB) has
G3=G4 and G6=G7=G8=G9=G10, i.e. input level cannot separate them, and G2/G3 targets are nearly uncorrelated
(raw ESR 0.97). So this run tests the identifiability limit, not the method under a fair control law.

Raw ESR, mean of 3 held-out DIs (`*` = trained; baseline = G5 NAM, same law):

| Gain | 10 cap | 3 cap | G5 same law |
|---:|---:|---:|---:|
| 1 | 0.0892* | 0.0137* | 0.8468 |
| 2 | 0.3124* | 0.0450 | 0.8447 |
| 3 | 0.1599* | 1.0734 | 0.1560 |
| 4 | 0.2419* | 1.8524 | 0.3674 |
| 5 | 0.0498* | 0.0782* | 0.0000 |
| 6 | 0.0871* | 0.1383 | 0.0324 |
| 7 | 0.0744* | 0.1233 | 0.0327 |
| 8 | 0.0493* | 0.0542 | 0.1128 |
| 9 | 0.0552* | 0.0449 | 0.1462 |
| 10 | 0.0750* | 0.0489* | 0.1814 |
| mean | 0.1194 | 0.3472 | 0.2720 |
| worst | 0.3124 | 1.8524 | 0.8468 |

- Both trained models beat the G5 baseline at G1-G2 and G8-G10 (0.05-0.09 versus 0.11-0.85), and lose to it at G6-G7.
- G3-G4 fail: the 10-capture model is poor but bounded (0.16-0.24); the 3-capture model never saw those positions
  and is worse than outputting silence (ESR 1.07, 1.85). Because the law gives G3 and G4 the same input level and
  the 3-capture set has no example between G1 and G5, this is a data/law problem and not evidence about the method.
- The 3-capture model's mean (0.347) is dominated by G3-G4 and is worse than the baseline mean; excluding those two it
  is comparable to the 10-capture model. Level error stays within 1.6 dB, except a +1.2 dB at G3 for the 3-capture model.
- High-band energy is 0.6-2.6 dB low (darker), as on the other amps. Quiet-DI behaviour at G3-G4 is worse still
  (ESR 0.8-2.6 at -12 dB).
- Waveform ESR on saturated 5150 captures was previously flagged as an unreliable metric; these numbers are
  additionally undermined by the degenerate law. Nothing here should be treated as a conclusion about capture count.

## Cross-amp summary (raw ESR, mean over gains, single seed, three held-out DIs)

| Amp | Gains | Baseline (mid NAM, same law) | 3 cap | 10 cap | Notes |
|---|---:|---:|---:|---:|---|
| JCM800 | 18 (G8.5 excl.) | 0.0928 | 0.0377 | 0.0328 | weak at G6.5-G7.5, G9.5 |
| Twin | 10 | 0.0268 | 0.0164 | 0.0158 | baseline already good at V1-V6 |
| Super-Sonic | 10 | 0.0767 | 0.0332 | 0.0291 | mid-volume, quiet-DI weakness |
| Peavey 5150 | 10 | 0.2720 | 0.3472 | 0.1194 | degenerate law; inconclusive |

On the three amps with a usable law, one standard NAM trained on the sweep beats the unchanged mid-gain NAM on average
and three captures come close to ten. On all four, the trained models win at the extremes of the sweep and the
mid-gain NAM wins near its own position, and trained models are slightly darker at high gain. Still not done:
Hybrid/Blended-assisted targets, multiple seeds, any human listening.

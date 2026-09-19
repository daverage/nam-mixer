# Continuous Gain: training one standard NAM from a physical gain sweep

Phase 1 results for the plan in `CONTINUOUS_GAIN_V2.md`. Scripts: `scripts/single_nam_*.py`. Raw results, logs,
listening WAVs and the trained models are in `work/single_nam/` (gitignored). Amp: Marshall JCM800 2203 High channel.

## Status

Done: ten-capture training (Method A, direct), capture-count reduction (5/3/2), baselines, half-step and
input-level validation, latency-anomaly detection, conflict analysis.

NOT done: Method B (Hybrid-assisted) and Method C (Blended-assisted) targets; Hybrid/Blended code review write-up;
multiple seeds; blind listening (WAVs are generated, no human listening has happened); Fender Super-Sonic, 57 Twin
and Peavey 5150; alternative three-capture subsets; equal-compute vs converged comparison. Nothing here should be
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

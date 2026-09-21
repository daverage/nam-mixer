# Continuous Gain v3: results (Marshall JCM800 2203, High channel)

Guide: `CONTINUOUS_GAIN_v3.md`. Code: `hybrid/multi_blend.py`, `scripts/cg_build.py`, `scripts/cg_eval.py`,
`scripts/run_cg.sh` (trains with the existing official-trainer wrapper `scripts/single_nam_train.py`, .venv-a2).
Artifacts (gitignored): `work/cg/jcm800/` -- datasets `cg_{10,5,3,2}`, models
`cg_N/JCM800_ContinuousGain_NCaptures/*.nam`, `eval_capN.json`, listening WAVs in `listening/`.

## Status

Done: JCM800 with 10, 5, 3 and 2 captures, evaluated on held-out DIs. NOT done: other amps, multiple seeds,
human listening (WAVs are generated, nobody has listened), Hybrid/Character-style teacher variants, extra checks in
sections 3/5/13 of the guide (feasibility/conflict analysis beyond the level-separation design, DI-level
generalisation sweep, continuous sweep through the exported NAM). Nothing below covers those.

## Method

- Gain positions are designed, not fitted: capture Gn owns causal input-envelope level `-52 + 4*(n-1)` dBFS
  (G1 -52 ... G10 -16); reference playing level -30 dBFS (median of the bundled DIs). Designated player input gain
  for Gn = `L_n - (-30)`: G1 -22 dB, G5 -6, G7 +2, G10 +14. No G5 anchor, no fitting to any capture. The same
  coordinate is used for every capture set, so reduced sets get no extra information.
- Target = `hybrid.multi_blend` (N-way generalisation of `hybrid.blend`; only adjacent captures mix, smoothstep) of
  real capture renders, each rendered on the input rescaled to the reference level, so that when the input sits at
  L_n the model is trained to give what capture n does when driven by the DI at its normal level. No per-capture
  level matching. One fixed peak-ceiling gain (`hybrid.safety.apply_peak_ceiling`, -0.2 dBFS; 4.7 dB).
- Training audio 467 s: the official NAM input plus clean_smooth, moderate_hotrod, high_thrash (25 s each) at -16,
  -8, 0, +8 dB; validation high_metalcore. A2 (22,783 params), official trainer, 60 epochs, seed 0, ~18 min each.
- Evaluation: held-out DIs (moderate_brit, clean_mayer, bass_rollin, first 20 s), frozen mapping, exported `.nam`
  through NAMCore only. Primary metric is level-matched ESR ("lmESR") since captures are normalised; also
  1/3-octave spectral-balance error (dB), crest and HF-energy deltas. Baseline (never a goal): real G5 driven by
  the SAME designed input gains. G8.5 is excluded (known capture latency defect, ~690 samples).

## Results (mean of 3 DIs; `*` trained position; integer gains only)

| Gain | 10 cap | 5 cap | 3 cap | 2 cap | G5, same gains |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.080* | 0.081* | 0.035* | 0.005* | 0.070 |
| 2 | 0.039* | 0.032 | 0.028 | 0.066 | 0.024 |
| 3 | 0.034* | 0.028* | 0.039 | 0.131 | 0.055 |
| 4 | 0.022* | 0.025 | 0.025 | 0.137 | 0.063 |
| 5 | 0.020* | 0.030* | 0.011* | 0.097 | 0.050 |
| 6 | 0.048* | 0.055 | 0.017 | 0.074 | 0.049 |
| 7 | 0.099* | 0.040* | 0.134 | 0.181 | 0.228 |
| 8 | 0.042* | 0.100 | 0.103 | 0.132 | 0.263 |
| 9 | 0.079* | 0.117 | 0.140 | 0.157 | 0.416 |
| 10 | 0.057* | 0.041* | 0.026* | 0.031* | 0.372 |
| mean | 0.052 | 0.055 | 0.056 | 0.101 | 0.159 |

Mean spectral-balance error (dB): 0.57 / 0.63 / 0.56 / 1.21 vs 0.67 (G5). Output level within about 1 dB of the
real capture at every gain for 10/5 captures (worst -1.1 dB), up to -2.1 dB for 3 captures at G2-G3. Genuine
half-steps (never trained) sit next to their neighbours (10 cap: 0.03-0.06 at G1.5-G5.5, 0.14 at G9.5).

## Findings

- **One standard NAM does follow the sweep.** The 10-capture `.nam` beats G5-pushed-around by ~3x on average and by
  5-7x at G8-G10, where G5 cannot reach the saturation. At G6.5-G10 the G5 baseline is far worse.
- **The weak zones are the low end and G7-G9.5.** Around G1-G2 and G3-G6 pushed G5 is as good or better (it is nearly
  exact there); the trained models trade that for the high end. G1 for the 10/5-cap models is weak (0.08, spectral
  error 2.2 dB, crest -2.3 dB): quiet material is amplified by 22 dB there and the target is noisy. Not investigated.
- **Capture count.** Three captures (G1, G5, G10) are about as good as ten on these metrics (0.056 vs 0.052; better
  at G1, G5, G6, worse at G7-G9). Five was not better than three. Two captures fail across G2-G9 (0.07-0.18, band
  error 1.2 dB), as no example shows mid-gain. So the smallest useful set here is 3, with G7-G9 the region where more
  captures help most (10 cap 0.04-0.10 vs 3 cap 0.10-0.14). One seed: small differences are not established.
- **Does not yet establish:** perceptual equivalence, DI-level robustness, seed variance, other amps.

## Known limitations

A stateless level control cannot separate a soft note at high gain from a hard note at low gain; not measured here
beyond the DI set (which sits at nearly the same level). Test with clean / hot DI levels before relying on it.

## Next justified steps

1. Listen to `work/cg/jcm800/listening/` (G1, G3, G5, G7, G10; real / new / G5, actual and level-matched).
2. DI level sweep (-12/+6 dB) and a continuous input-gain sweep through the exported `.nam`.
3. Investigate G1 and G7-G9 (extra captures placed there; seeds).
4. Repeat on the other amps.

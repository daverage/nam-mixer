# Phase 4C: can a better input-gain mapping make an existing NAM behave more like the amp? (no retraining)

Uses the existing v3 models (C3 = G1/G5/G10, C5 = G1/G3/G5/G7/G10, C10 = all) unchanged and changes ONLY the playback input-gain mapping. Code: `scripts/p4_model_sweep.py` (renders each model over -48..+36 dB input gain plus a sine input/output sweep), `scripts/p4_mapping.py` (fit + evaluation), `scripts/p4_mapping_report.py`; data `work/p4/<amp>/mapping.json`.

## Method

- **Fixed** = the v3 design: gain N at input level `-22 + 4(N-1)` dB (G1 -22 ... G10 +14) for every amp.
- **Measured** = one ordered (non-decreasing) mapping per amp and model, fitted on the fit DIs only (clean_smooth, moderate_hotrod, high_thrash, high_metalcore) by minimising, per real capture, a scale-free distance in **tone** (six EQ bands), **saturation** (HF>3 kHz, crest, sine THD) and **compression** (two input/output slopes, dynamic range). **Output level is deliberately excluded from the fit** and reported separately; it is not an RMS match. One mapping for all DIs and levels: nothing is tuned per recording or per metric.
- **Held-out validation** uses moderate_brit, clean_mayer and bass_rollin, which were never used for any fit or mapping. On the JCM800 the mapping is fitted on the 10 integer positions only, so its 9 genuine half-step captures are untouched reference positions (their mapping values are PCHIP-interpolated).
- **Quarantine:** captures flagged SUSPECT for level (Peavey G4, Orange G2) are excluded from the level column; timing does not affect these measurements.
- **Trade-off mappings:** the same fit using only one dimension, and an RMS-only fit (the naive approach the brief warns against), are shown so trade-offs are visible.
- **Oracle (upper bound):** best per-capture input gain per dimension chosen ON the held-out DIs, unordered. It is a ceiling that shows how much any mapping could ever fix; it is **not** a valid mapping and is never reported as a result.
- Thresholds below (`material` = smallest change worth noting; `high` = error still likely audible) are **working values, not perceptual measurements**: tone 0.15/1.0 dB, HF 0.2/1.0, crest 0.2/1.0, THD 1.5/5, IO slope 0.5/2, dynamic range 0.3/1.5, level 0.4/1.5.
- Single seed models; small differences are not established.

## Findings (mapping-only; existing v3 models; held-out DIs)

1. **Compression (input/output slope) is the dimension a measured mapping helps most, and the one it cannot finish.** Static input/output slope error (dB, mean of two slopes) for fixed / measured / oracle-ceiling:

| Amp | C3 fixed | C3 measured | C3 oracle UB | C10 fixed | C10 measured | C10 oracle UB |
|---|---:|---:|---:|---:|---:|---:|
| JCM800 | 4.42 | 3.11 | 2.84 | 4.83 | 2.88 | 2.70 |
| Twin | 4.36 | 2.15 | 2.18 | 5.34 | 3.77 | 3.58 |
| Super-Sonic Bassman | 3.00 | 3.56 | 1.25 | 3.61 | 1.33 | 0.81 |
| Super-Sonic Vibrolux | 3.67 | 2.55 | 2.49 | 3.59 | 1.84 | 1.72 |
| Peavey 5150 | 3.80 | 2.22 | 1.46 | 4.25 | 2.51 | 2.45 |
| Peavey 6505+ | 2.23 | 1.09 | 0.85 | 3.43 | 1.62 | 1.25 |
| Mesa Dual Rectifier | 2.95 | 3.13 | 1.63 | 5.77 | 4.52 | 4.35 |
| Orange Dual Terror | 7.12 | 6.56 | 5.21 | 6.26 | 4.89 | 2.97 |

   The measured mapping lowers C3 slope error clearly on five amps (JCM800, Twin, Vibrolux, Peavey 5150, 6505+), marginally on Orange, and not on Mesa or Super-Sonic Bassman (Bassman's oracle ceiling, 1.25, is well below the fitted 3.56, so headroom exists there that my joint tone+saturation+compression fit did not find; a compression-weighted fit might). What remains after even the oracle ceiling (about 1-5 dB) cannot be reached by any input-gain mapping of that model, and C10 does not consistently reduce it (Twin, Mesa and Peavey 5150 are worse at C10). That points to the level-driven design itself (a louder input is read as a higher gain setting, so the model compresses differently from a fixed-gain amp), not to mapping or to capture coverage.
2. **Tone is mostly reachable with either mapping.** Fixed-mapping tone error is already 0.4-1.0 dB; the measured mapping helps on Bassman, Mesa and Orange and worsens nothing at C3. Tone is not where the mapping matters.
3. **Saturation and dynamics gains are amp-specific.** The measured mapping helps crest/dynamic range on JCM800, Peavey 5150 (but HF worsens, 0.64 to 1.03 dB), Mesa and Orange, and HF on Bassman, Mesa and Orange; on the clean Fenders (Twin, Vibrolux) it worsens crest (and HF/THD on the Twin).
4. **Character versus output level is a genuine conflict on the clean Fenders.** Level is excluded from the fit; the measured mapping raises level error on Twin (1.47 to 2.91 dB) and Vibrolux (0.78 to 3.80 dB) at C3 while improving compression. A single mapping cannot give both the real amp's character progression and its level progression there. This must not be reported as a general improvement.
5. **The measured mapping is compressive at the top.** On the JCM800 G5-G10 spans about 8 dB of input gain instead of the fixed 20 dB, consistent with the real amp's plateau (see the 4B profile): equal 4 dB steps over-spend range where the real amp has stopped changing.
6. **Independent check (JCM800 half-steps, never used in the fit):** C3 with the measured mapping is better than fixed in all seven columns (tone 0.35 to 0.29, HF 0.46 to 0.40, crest 0.49 to 0.32, THD 1.23 to 0.70, IO slope 4.43 to 3.20, dyn range 0.91 to 0.46, level 0.74 to 0.55). C10 is mixed (IO slope and dyn range better; THD 0.57 to 0.92 and tone 0.33 to 0.36 slightly worse).
7. **Not shown here:** these are technical metrics on level-matched-to-native playback with a single training seed per model; no audible improvement has been demonstrated, and a model TRAINED with the measured mapping (Phase 4E) may behave differently from the same mapping applied to a model trained on fixed spacing.

## Cross-amp verdict for the mapping-only baseline (C3, held-out DIs, DI level 0)

`helps` / `worsens` = measured mapping vs fixed 4 dB, beyond the working thresholds; `-` = no material change. Add `*` where the error is still above the `high` threshold under the best mapping (measured or fixed) and the oracle ceiling is no better than 70% of it, i.e. mapping alone cannot fix it for this model.

| Amp | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---|---|---|---|---|---|---|
| JCM800 | - | - | helps | - | helps* | helps | - |
| Twin | - | worsens | worsens* | worsens* | helps* | -* | worsens |
| Super-Sonic Bassman | helps | helps | - | - | - | helps | - |
| Super-Sonic Vibrolux | - | - | worsens* | - | helps* | helps | worsens |
| Peavey 5150 | - | worsens | helps | - | helps | helps | - |
| Peavey 6505+ | - | helps | - | - | helps | - | - |
| Mesa Dual Rectifier | helps | helps | helps | - | - | helps | helps |
| Orange Dual Terror | helps | helps | - | - | -* | helps | - |

### Trade-offs (improving one characteristic worsens another)

- **Peavey 5150, C10:** measured mapping improves crest (dB) (1.03 to 0.65), IO slope (dB) (4.25 to 2.51), dyn range (dB) (1.60 to 0.96) but worsens HF>3k (dB) (0.42 to 0.73).
- **Peavey 5150, C3:** measured mapping improves crest (dB) (0.67 to 0.44), IO slope (dB) (3.80 to 2.22), dyn range (dB) (1.43 to 0.67) but worsens HF>3k (dB) (0.64 to 1.03).
- **Peavey 6505+, C10:** measured mapping improves IO slope (dB) (3.43 to 1.62) but worsens tone (EQ, dB) (0.37 to 0.57).
- **Super-Sonic Vibrolux, C10:** measured mapping improves IO slope (dB) (3.59 to 1.84) but worsens crest (dB) (1.37 to 1.79), level (dB) (0.86 to 1.46).
- **Super-Sonic Vibrolux, C3:** measured mapping improves IO slope (dB) (3.67 to 2.55), dyn range (dB) (2.06 to 1.12) but worsens crest (dB) (1.38 to 2.40), level (dB) (0.78 to 3.80).
- **Twin, C10:** measured mapping improves IO slope (dB) (5.34 to 3.77), dyn range (dB) (1.64 to 0.82) but worsens tone (EQ, dB) (0.76 to 1.09), HF>3k (dB) (1.55 to 2.13), crest (dB) (0.99 to 2.00), level (dB) (1.35 to 3.25).
- **Twin, C3:** measured mapping improves IO slope (dB) (4.36 to 2.15) but worsens HF>3k (dB) (1.33 to 1.75), crest (dB) (1.47 to 1.85), THD@-30 (dB) (13.28 to 16.89), level (dB) (1.47 to 2.91).

## Coverage check: does a richer model (C10) reach what C3 cannot, under the same measured mapping?

Where C10 is much better than C3 with the SAME method (error < 60% and materially lower), the limit is capture coverage, not mapping. Otherwise it is not fixed by more captures either.

| Amp | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---|---|---|---|---|---|---|
| JCM800 | - | - | - | - | - | - | - |
| Twin | - | - | - | - | - | coverage | - |
| Super-Sonic Bassman | - | - | - | - | coverage | - | - |
| Super-Sonic Vibrolux | - | - | - | - | - | - | coverage |
| Peavey 5150 | - | - | - | - | - | - | - |
| Peavey 6505+ | - | - | - | - | - | - | - |
| Mesa Dual Rectifier | - | - | - | - | - | - | - |
| Orange Dual Terror | - | - | - | - | - | - | - |

## Per-amp results (mean absolute error vs the real capture, held-out DIs, DI level 0; lower is better)

### JCM800

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.38 | 0.51 | 0.61 | 1.22 | 4.42 | 0.87 | 0.69 |
| measured_all | 0.32 | 0.44 | 0.36 | 0.70 | 3.11 | 0.41 | 0.58 |
| rms_only | 0.38 | 0.48 | 0.63 | 1.31 | 4.09 | 0.54 | 0.25 |
| oracle (UB) | 0.20 | 0.32 | 0.37 | 0.44 | 2.84 | 0.46 | 0.21 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.40 | 0.59 | 0.50 | 0.57 | 4.39 | 1.13 | 0.31 |
| measured_all | 0.38 | 0.59 | 0.29 | 1.05 | 2.66 | 0.81 | 0.46 |
| rms_only | 0.36 | 0.55 | 0.58 | 0.99 | 4.21 | 0.77 | 0.19 |
| oracle (UB) | 0.29 | 0.48 | 0.34 | 0.36 | 2.19 | 0.78 | 0.14 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.38 | 0.64 | 0.43 | 0.85 | 4.83 | 1.23 | 0.32 |
| measured_all | 0.39 | 0.54 | 0.31 | 0.94 | 2.88 | 0.96 | 0.47 |
| rms_only | 0.46 | 0.72 | 0.52 | 1.10 | 4.36 | 0.82 | 0.18 |
| oracle (UB) | 0.30 | 0.39 | 0.35 | 0.56 | 2.70 | 0.75 | 0.13 |

**Independent half-step captures only (untouched by the fit)**, C3 and C10:

C3:

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.35 | 0.46 | 0.49 | 1.23 | 4.43 | 0.91 | 0.74 |
| measured_all | 0.29 | 0.40 | 0.32 | 0.70 | 3.20 | 0.46 | 0.55 |
| rms_only | 0.35 | 0.44 | 0.54 | 1.22 | 4.16 | 0.52 | 0.25 |

C10:

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.33 | 0.56 | 0.32 | 0.57 | 4.80 | 1.30 | 0.35 |
| measured_all | 0.36 | 0.51 | 0.31 | 0.92 | 2.98 | 0.98 | 0.39 |
| rms_only | 0.42 | 0.64 | 0.46 | 0.75 | 4.31 | 0.90 | 0.20 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -24 | -16 | -10 | -5 | -3 | -2 | +2 | +3 | +5 | +5 |
| C3 rms_only | -22 | -13 | -4 | -4 | -3 | +0 | +5 | +13 | +14 | +14 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -26 | -17 | -10 | -1 | -1 | +0 | +3 | +5 | +5 | +6 |
| C10 rms_only | -22 | -16 | -8 | -8 | -7 | -5 | -2 | +6 | +8 | +8 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 0.76 | 0.76 | 1.34 | 2.39 | 1.50 |
| -12 | measured_all | 0.48 | 0.45 | 0.89 | 1.71 | 0.76 |
| -6 | fixed_4dB | 0.51 | 0.57 | 0.85 | 1.63 | 0.90 |
| -6 | measured_all | 0.29 | 0.35 | 0.53 | 0.91 | 0.56 |
| 0 | fixed_4dB | 0.38 | 0.51 | 0.61 | 0.87 | 0.69 |
| 0 | measured_all | 0.32 | 0.44 | 0.36 | 0.41 | 0.58 |
| 6 | fixed_4dB | 0.41 | 0.56 | 0.85 | 0.61 | 0.74 |
| 6 | measured_all | 0.40 | 0.49 | 0.45 | 0.39 | 0.71 |

### Twin

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.74 | 1.33 | 1.47 | 13.28 | 4.36 | 2.19 | 1.47 |
| measured_all | 0.91 | 1.75 | 1.85 | 16.89 | 2.15 | 2.12 | 2.91 |
| rms_only | 0.52 | 0.96 | 0.88 | 13.16 | 5.35 | 2.16 | 0.29 |
| oracle (UB) | 0.40 | 0.35 | 1.42 | 14.00 | 2.18 | 2.92 | 0.24 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.82 | 1.42 | 1.02 | 12.25 | 4.08 | 1.61 | 1.41 |
| measured_all | 1.10 | 1.88 | 1.32 | 13.13 | 2.43 | 0.71 | 2.42 |
| rms_only | 0.82 | 1.48 | 0.74 | 15.13 | 3.95 | 1.79 | 0.26 |
| oracle (UB) | 0.55 | 0.52 | 1.53 | 12.58 | 3.43 | 2.43 | 0.15 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.76 | 1.55 | 0.99 | 11.52 | 5.34 | 1.64 | 1.35 |
| measured_all | 1.09 | 2.13 | 2.00 | 13.49 | 3.77 | 0.82 | 3.25 |
| rms_only | 0.79 | 1.69 | 0.58 | 12.34 | 4.88 | 1.80 | 0.21 |
| oracle (UB) | 0.51 | 0.61 | 2.05 | 10.87 | 3.58 | 2.08 | 0.15 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -13 | -13 | -13 | -7 | -5 | +2 | +2 | +2 | +3 | +3 |
| C3 rms_only | -24 | -17 | -10 | -7 | -5 | +0 | +8 | +12 | +13 | +13 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -14 | -14 | -10 | -9 | -9 | -8 | -1 | +0 | +0 | +0 |
| C10 rms_only | -29 | -20 | -12 | -8 | -5 | -3 | +3 | +6 | +8 | +9 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 0.69 | 0.76 | 1.30 | 2.42 | 3.21 |
| -12 | measured_all | 0.49 | 0.41 | 1.02 | 1.59 | 2.92 |
| -6 | fixed_4dB | 0.44 | 0.72 | 1.09 | 2.35 | 1.85 |
| -6 | measured_all | 0.49 | 0.68 | 0.97 | 1.94 | 2.49 |
| 0 | fixed_4dB | 0.74 | 1.33 | 1.47 | 2.19 | 1.47 |
| 0 | measured_all | 0.91 | 1.75 | 1.85 | 2.12 | 2.91 |
| 6 | fixed_4dB | 1.14 | 2.30 | 1.85 | 1.78 | 1.51 |
| 6 | measured_all | 1.64 | 2.73 | 2.47 | 1.50 | 3.08 |

### Super-Sonic Bassman

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.76 | 1.54 | 0.58 | 2.83 | 3.00 | 1.47 | 0.56 |
| measured_all | 0.60 | 1.15 | 0.43 | 1.59 | 3.56 | 1.04 | 0.28 |
| rms_only | 0.60 | 1.18 | 0.42 | 1.80 | 3.90 | 1.14 | 0.22 |
| oracle (UB) | 0.58 | 0.68 | 0.76 | 0.38 | 1.25 | 0.77 | 0.18 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.83 | 1.58 | 0.87 | 2.00 | 3.63 | 1.40 | 0.65 |
| measured_all | 0.81 | 1.56 | 0.55 | 1.62 | 1.33 | 1.51 | 0.39 |
| rms_only | 0.74 | 1.37 | 0.61 | 1.12 | 4.49 | 1.42 | 0.19 |
| oracle (UB) | 0.67 | 0.61 | 1.01 | 2.11 | 1.37 | 1.16 | 0.15 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.40 | 0.65 | 0.71 | 1.53 | 3.61 | 1.22 | 0.62 |
| measured_all | 0.41 | 0.82 | 0.36 | 1.70 | 1.33 | 1.11 | 0.20 |
| rms_only | 0.38 | 0.73 | 0.36 | 1.33 | 2.57 | 1.20 | 0.13 |
| oracle (UB) | 0.32 | 0.60 | 0.47 | 1.01 | 0.81 | 0.91 | 0.12 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -23 | -17 | -11 | -7 | -5 | -4 | +10 | +11 | +12 | +12 |
| C3 rms_only | -23 | -18 | -11 | -7 | -4 | +0 | +8 | +12 | +13 | +14 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -25 | -20 | -12 | -8 | -5 | -3 | +0 | +3 | +3 | +3 |
| C10 rms_only | -25 | -20 | -13 | -8 | -6 | -3 | +1 | +6 | +8 | +9 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 0.90 | 1.15 | 1.72 | 2.16 | 1.86 |
| -12 | measured_all | 0.92 | 1.28 | 1.40 | 3.04 | 1.33 |
| -6 | fixed_4dB | 0.79 | 1.53 | 0.97 | 1.91 | 0.88 |
| -6 | measured_all | 0.68 | 1.27 | 0.47 | 2.00 | 0.36 |
| 0 | fixed_4dB | 0.76 | 1.54 | 0.58 | 1.47 | 0.56 |
| 0 | measured_all | 0.60 | 1.15 | 0.43 | 1.04 | 0.28 |
| 6 | fixed_4dB | 0.74 | 1.44 | 1.06 | 0.82 | 0.64 |
| 6 | measured_all | 0.66 | 1.30 | 1.13 | 0.49 | 0.46 |

### Super-Sonic Vibrolux

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.80 | 1.42 | 1.38 | 4.18 | 3.67 | 2.06 | 0.78 |
| measured_all | 0.76 | 1.36 | 2.40 | 3.08 | 2.55 | 1.12 | 3.80 |
| rms_only | 0.80 | 1.36 | 0.98 | 3.89 | 4.27 | 1.93 | 0.35 |
| oracle (UB) | 0.51 | 0.94 | 1.19 | 1.60 | 2.49 | 2.45 | 0.29 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.69 | 1.31 | 1.50 | 2.90 | 2.98 | 2.13 | 0.96 |
| measured_all | 0.84 | 1.64 | 2.14 | 3.72 | 1.49 | 2.07 | 1.76 |
| rms_only | 0.66 | 1.26 | 0.95 | 3.12 | 3.28 | 2.19 | 0.30 |
| oracle (UB) | 0.45 | 0.92 | 1.16 | 2.11 | 1.47 | 3.00 | 0.22 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.81 | 1.56 | 1.37 | 2.95 | 3.59 | 2.48 | 0.86 |
| measured_all | 0.89 | 1.73 | 1.79 | 3.59 | 1.84 | 2.54 | 1.46 |
| rms_only | 0.76 | 1.48 | 1.02 | 2.73 | 3.36 | 2.38 | 0.25 |
| oracle (UB) | 0.57 | 1.16 | 2.05 | 0.22 | 1.72 | 3.37 | 0.18 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -38 | -32 | -11 | -8 | -5 | -4 | +2 | +3 | +4 | +4 |
| C3 rms_only | -23 | -19 | -12 | -7 | -4 | -1 | +4 | +10 | +13 | +13 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -20 | -18 | -16 | -10 | -7 | +1 | +2 | +3 | +4 | +5 |
| C10 rms_only | -24 | -21 | -14 | -8 | -5 | -3 | +1 | +5 | +9 | +10 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 0.93 | 1.56 | 1.96 | 2.60 | 2.55 |
| -12 | measured_all | 0.90 | 1.69 | 1.25 | 1.27 | 4.13 |
| -6 | fixed_4dB | 0.62 | 0.90 | 0.98 | 2.63 | 1.23 |
| -6 | measured_all | 0.56 | 0.95 | 1.38 | 1.73 | 3.56 |
| 0 | fixed_4dB | 0.80 | 1.42 | 1.38 | 2.06 | 0.78 |
| 0 | measured_all | 0.76 | 1.36 | 2.40 | 1.12 | 3.80 |
| 6 | fixed_4dB | 1.06 | 1.83 | 1.94 | 1.09 | 1.19 |
| 6 | measured_all | 0.88 | 1.37 | 3.39 | 0.91 | 3.97 |

### Peavey 5150

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.65 | 0.64 | 0.67 | 0.94 | 3.80 | 1.43 | 0.51 |
| measured_all | 0.62 | 1.03 | 0.44 | 0.61 | 2.22 | 0.67 | 0.75 |
| rms_only | 0.72 | 0.52 | 2.68 | 1.47 | 7.69 | 1.82 | 0.43 |
| oracle (UB) | 0.49 | 1.09 | 0.31 | 0.23 | 1.46 | 0.50 | 0.36 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.47 | 0.71 | 0.45 | 0.47 | 3.61 | 1.22 | 0.35 |
| measured_all | 0.47 | 0.64 | 0.27 | 0.43 | 1.96 | 0.73 | 0.61 |
| rms_only | 0.55 | 0.81 | 0.49 | 0.73 | 2.91 | 1.12 | 0.35 |
| oracle (UB) | 0.35 | 0.51 | 0.27 | 0.24 | 1.01 | 0.41 | 0.31 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.37 | 0.42 | 1.03 | 0.15 | 4.25 | 1.60 | 0.60 |
| measured_all | 0.51 | 0.73 | 0.65 | 0.70 | 2.51 | 0.96 | 0.54 |
| rms_only | 0.57 | 0.85 | 0.89 | 1.15 | 3.25 | 1.07 | 0.41 |
| oracle (UB) | 0.28 | 0.72 | 0.51 | 0.55 | 2.45 | 0.71 | 0.37 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -25 | -17 | -11 | -4 | +1 | +4 | +7 | +7 | +7 | +7 |
| C3 rms_only | -22 | -17 | -13 | -13 | +28 | +28 | +28 | +28 | +28 | +28 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -23 | -18 | -8 | +1 | +2 | +2 | +3 | +3 | +3 | +3 |
| C10 rms_only | -21 | -14 | -10 | -10 | +5 | +6 | +6 | +6 | +10 | +14 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 1.08 | 1.80 | 1.11 | 3.52 | 1.76 |
| -12 | measured_all | 0.65 | 1.04 | 0.60 | 2.28 | 1.08 |
| -6 | fixed_4dB | 0.52 | 0.90 | 0.93 | 2.31 | 0.90 |
| -6 | measured_all | 0.50 | 0.80 | 0.51 | 1.17 | 0.78 |
| 0 | fixed_4dB | 0.65 | 0.64 | 0.67 | 1.43 | 0.51 |
| 0 | measured_all | 0.62 | 1.03 | 0.44 | 0.67 | 0.75 |
| 6 | fixed_4dB | 0.74 | 1.22 | 0.48 | 0.70 | 0.58 |
| 6 | measured_all | 0.67 | 1.03 | 0.32 | 0.43 | 0.71 |

### Peavey 6505+

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.69 | 1.31 | 0.63 | 1.09 | 2.23 | 0.58 | 0.71 |
| measured_all | 0.66 | 0.94 | 0.47 | 0.86 | 1.09 | 0.37 | 0.60 |
| rms_only | 0.71 | 0.69 | 1.28 | 0.60 | 4.30 | 0.56 | 0.31 |
| oracle (UB) | 0.45 | 0.83 | 0.41 | 0.13 | 0.85 | 0.39 | 0.23 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.53 | 0.90 | 0.44 | 0.37 | 2.91 | 0.55 | 0.34 |
| measured_all | 0.58 | 0.84 | 0.37 | 0.29 | 1.42 | 0.27 | 0.35 |
| rms_only | 0.69 | 0.86 | 1.22 | 0.44 | 6.07 | 0.48 | 0.29 |
| oracle (UB) | 0.50 | 0.75 | 0.28 | 0.35 | 1.04 | 0.27 | 0.19 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.37 | 0.52 | 0.38 | 0.23 | 3.43 | 0.44 | 0.39 |
| measured_all | 0.57 | 0.61 | 0.32 | 0.27 | 1.62 | 0.29 | 0.47 |
| rms_only | 0.57 | 0.63 | 2.00 | 0.32 | 6.18 | 0.70 | 0.23 |
| oracle (UB) | 0.35 | 0.54 | 0.24 | 0.20 | 1.25 | 0.37 | 0.17 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -23 | -13 | -9 | -4 | -3 | -3 | -3 | +1 | +1 | +1 |
| C3 rms_only | -20 | -13 | -9 | +10 | +16 | +19 | +19 | +19 | +19 | +19 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -22 | -15 | -11 | -4 | -4 | -4 | -3 | -3 | -3 | -3 |
| C10 rms_only | -23 | -17 | -12 | -2 | +16 | +18 | +18 | +18 | +18 | +18 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 0.95 | 1.38 | 0.45 | 1.41 | 1.22 |
| -12 | measured_all | 0.93 | 1.44 | 0.66 | 1.30 | 1.01 |
| -6 | fixed_4dB | 0.70 | 1.35 | 0.52 | 0.79 | 0.91 |
| -6 | measured_all | 0.67 | 0.79 | 0.47 | 0.68 | 0.55 |
| 0 | fixed_4dB | 0.69 | 1.31 | 0.63 | 0.58 | 0.71 |
| 0 | measured_all | 0.66 | 0.94 | 0.47 | 0.37 | 0.60 |
| 6 | fixed_4dB | 0.73 | 1.08 | 0.82 | 0.45 | 0.56 |
| 6 | measured_all | 0.97 | 1.39 | 0.59 | 0.35 | 0.70 |

### Mesa Dual Rectifier

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 1.02 | 1.44 | 0.70 | 1.60 | 2.95 | 1.29 | 1.27 |
| measured_all | 0.81 | 1.04 | 0.48 | 0.43 | 3.13 | 0.61 | 0.80 |
| rms_only | 1.79 | 2.17 | 0.82 | 0.37 | 3.61 | 1.04 | 0.66 |
| oracle (UB) | 0.68 | 0.65 | 0.36 | 0.34 | 1.63 | 0.49 | 0.54 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.74 | 1.15 | 0.40 | 0.96 | 4.42 | 1.54 | 0.61 |
| measured_all | 0.68 | 0.84 | 0.42 | 0.71 | 3.43 | 0.87 | 0.63 |
| rms_only | 1.15 | 1.67 | 0.59 | 0.90 | 6.17 | 1.10 | 0.39 |
| oracle (UB) | 0.54 | 0.90 | 0.20 | 0.54 | 3.15 | 0.68 | 0.32 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.80 | 1.26 | 0.54 | 0.57 | 5.77 | 1.18 | 0.61 |
| measured_all | 0.63 | 0.70 | 0.40 | 0.32 | 4.52 | 0.86 | 0.63 |
| rms_only | 1.20 | 1.83 | 1.02 | 0.98 | 7.51 | 1.25 | 0.47 |
| oracle (UB) | 0.52 | 0.71 | 0.33 | 0.19 | 4.35 | 0.86 | 0.32 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -23 | -12 | -8 | -6 | -5 | -3 | -3 | +2 | +12 | +12 |
| C3 rms_only | -22 | -10 | +0 | +0 | +8 | +12 | +14 | +18 | +21 | +24 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -23 | -16 | -10 | -5 | -5 | -4 | -3 | -3 | +4 | +5 |
| C10 rms_only | -25 | -15 | -8 | -8 | -3 | +4 | +12 | +18 | +19 | +20 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 1.72 | 1.82 | 0.84 | 2.36 | 2.32 |
| -12 | measured_all | 1.11 | 1.17 | 0.61 | 2.24 | 1.69 |
| -6 | fixed_4dB | 1.02 | 1.14 | 0.75 | 1.66 | 1.71 |
| -6 | measured_all | 0.80 | 0.57 | 0.54 | 1.37 | 0.88 |
| 0 | fixed_4dB | 1.02 | 1.44 | 0.70 | 1.29 | 1.27 |
| 0 | measured_all | 0.81 | 1.04 | 0.48 | 0.61 | 0.80 |
| 6 | fixed_4dB | 1.44 | 1.99 | 0.64 | 0.99 | 1.04 |
| 6 | measured_all | 1.58 | 2.07 | 0.70 | 0.70 | 1.02 |

### Orange Dual Terror

**C3** (trained on G1, G5, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.90 | 1.44 | 1.12 | 2.25 | 7.12 | 3.23 | 0.93 |
| measured_all | 0.65 | 0.75 | 1.27 | 2.27 | 6.56 | 1.31 | 0.85 |
| rms_only | 0.82 | 0.94 | 2.01 | 2.94 | 7.29 | 1.42 | 0.27 |
| oracle (UB) | 0.50 | 0.88 | 0.72 | 1.97 | 5.21 | 1.28 | 0.19 |

**C5** (trained on G1, G3, G5, G7, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.95 | 1.56 | 1.91 | 3.40 | 7.00 | 2.89 | 1.16 |
| measured_all | 0.66 | 0.95 | 1.14 | 2.13 | 5.81 | 1.85 | 1.26 |
| rms_only | 0.78 | 0.88 | 1.88 | 2.70 | 7.26 | 2.09 | 0.31 |
| oracle (UB) | 0.54 | 0.97 | 0.82 | 1.10 | 4.13 | 1.95 | 0.25 |

**C10** (trained on G1, G2, G3, G4, G5, G6, G7, G8, G9, G10)

| Mapping | tone (EQ, dB) | HF>3k (dB) | crest (dB) | THD@-30 (dB) | IO slope (dB) | dyn range (dB) | level (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_4dB | 0.68 | 1.04 | 1.77 | 2.16 | 6.26 | 2.55 | 1.15 |
| measured_all | 0.64 | 0.81 | 1.00 | 2.49 | 4.89 | 1.73 | 0.74 |
| rms_only | 0.77 | 0.86 | 1.68 | 2.70 | 5.84 | 1.80 | 0.28 |
| oracle (UB) | 0.46 | 0.82 | 0.68 | 1.56 | 2.97 | 1.87 | 0.22 |

Input-gain mapping (dB) by physical gain position:

| Mapping | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C3 measured_all | -28 | -22 | -7 | -4 | -2 | +2 | +7 | +11 | +12 | +12 |
| C3 rms_only | -24 | -24 | -8 | -4 | +0 | +0 | +3 | +17 | +17 | +17 |
| C10 fixed_4dB | -22 | -18 | -14 | -10 | -6 | -2 | +2 | +6 | +10 | +14 |
| C10 measured_all | -29 | -20 | -11 | -7 | -5 | -1 | +6 | +9 | +11 | +11 |
| C10 rms_only | -27 | -27 | -11 | -4 | -1 | -1 | +1 | +13 | +13 | +13 |

Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):

| DI offset | mapping | tone | HF>3k | crest | dyn range | level |
|---:|---|---:|---:|---:|---:|---:|
| -12 | fixed_4dB | 1.31 | 1.79 | 2.04 | 5.54 | 3.42 |
| -12 | measured_all | 1.29 | 2.21 | 0.75 | 4.78 | 1.34 |
| -6 | fixed_4dB | 1.04 | 1.55 | 1.31 | 4.22 | 1.61 |
| -6 | measured_all | 0.78 | 1.22 | 0.87 | 2.25 | 0.92 |
| 0 | fixed_4dB | 0.90 | 1.44 | 1.12 | 3.23 | 0.93 |
| 0 | measured_all | 0.65 | 0.75 | 1.27 | 1.31 | 0.85 |
| 6 | fixed_4dB | 0.89 | 1.13 | 1.76 | 1.85 | 1.01 |
| 6 | measured_all | 0.89 | 1.05 | 1.91 | 1.88 | 0.69 |

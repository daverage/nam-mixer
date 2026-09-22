# Phase 4A: capture-quality audit (all eight amp configurations)

Reproducible: `scripts/p4_probe_captures.py <amp>` (raw, uncorrected probe bank) then `scripts/p4_audit.py <amp>` (checks, statuses); data in `work/p4/<amp>/` (`captures.json`, `audit.json`, `corrected/`). Source `.nam` files are never modified: a correction is a documented render-time shift, with original and corrected renders saved side by side. No missing capture is ever replaced by an interpolated one.

Statuses: **VALID** no identified problem; **CORRECTED** a verified technical fault was corrected and the correction passed validation; **SUSPECT** unexplained anomaly, needs investigation (quarantine, never silently accepted or deleted); **INVALID** confirmed unrecoverable. A capture is never marked down merely for departing from a smooth or monotonic gain curve.

Checks: timing (click onset vs set median, then confirmed on real music by cross-correlation against a neighbouring capture and reproduced on a second DI); clipping, dropouts, NaN, DC; noise floor and hum against neighbours; output-level trend, cross-checked against sine outputs and the file's own `loudness` metadata; metadata consistency; tone/dynamics deviation from neighbouring captures (raises SUSPECT only); and a model-determinism proxy. **Repeatability:** no repeat physical captures exist, so real repeatability is not measurable; the determinism proxy only shows the trained model is stable, not that the capture is.

## Summary

| Amp | VALID | CORRECTED | SUSPECT | INVALID | Not VALID |
|---|---:|---:|---:|---:|---|
| Marshall JCM800 2203 (High) | 18 | 1 | 0 | 0 | G8.5 corrected |
| Fender 57 Custom Twin | 10 | 0 | 0 | 0 | - |
| Fender Super-Sonic, Bassman channel | 5 | 2 | 3 | 0 | G1 corrected, G2 corrected, G4 suspect, G7 suspect, G10 suspect |
| Fender Super-Sonic, Vibrolux channel | 10 | 0 | 0 | 0 | - |
| Peavey 5150 | 1 | 7 | 2 | 0 | G1 corrected, G2 corrected, G3 corrected, G4 suspect, G6 corrected, G7 corrected, G8 corrected, G9 corrected, G10 suspect |
| Peavey 6505+ Scooped | 10 | 0 | 0 | 0 | - |
| Mesa Dual Rectifier | 10 | 0 | 0 | 0 | - |
| Orange Dual Terror | 5 | 3 | 2 | 0 | G2 suspect, G7 suspect, G8 corrected, G9 corrected, G10 corrected |

## Cases named in the brief

- **JCM800 G8.5 timing:** CORRECTED. Click onset is +674 samples vs the set median (about +6); music cross-correlation gives a 692-sample delay (corrected by shifting it earlier) (click and music agree within 30 samples; a second DI reproduces the lag within 3 samples (+691 vs +692)). After the shift the residual music lag against G8 is 0. Only the half-step G8.5 is affected; no integer JCM800 capture was, so the v3 training set (integers only) was not.
- **Peavey 5150 G4 output level:** SUSPECT, and a level problem, not a timing one. G4 is 4.1 dB quieter than the interpolation of G3 and G5 at every sustained input level (sine and music), and the file's own `loudness` metadata (-16.4 vs about -12 for the neighbours) shows the same drop, while G4 has more low-level gain than G3. A constant fixed drop at saturation with both neighbours equal is not typical amp behaviour and is consistent with a capture-chain level error, but the physical amp is not available to confirm. **Suggested level correction +4.1 dB: an ESTIMATE, not applied.** Level-progression analysis should quarantine G4; its tone data are probably usable.
- **Peavey timing offsets:** confirmed real in the music, not only in the click. Corrections relative to the set median: G1 +38, G2 +26, G3 +52, G4 +46, G6 -406, G7 -10, G8 -22, G9 -410, G10 -524 samples (G6, G9, G10 are 400-520 samples; the rest 10-50). Click-onset and music offsets differ by up to 25 samples, so music cross-correlation is the better basis. All corrections reproduce on a second DI within 3 samples.
- **Fender Super-Sonic, Bassman channel G10 noise:** SUSPECT. noise spike: silence renders at -49 dBFS vs -63 dBFS typical for neighbours The hiss may be genuine amp noise at high gain, so this is a flag to investigate, not a fault.
- **Orange Dual Terror G7 noise:** SUSPECT. noise spike: silence renders at -44 dBFS vs -61 dBFS typical for neighbours The hiss may be genuine amp noise at high gain, so this is a flag to investigate, not a fault.
- **Super-Sonic Bassman G4, G7:** SUSPECT for the same noise-spike test (-51 and -52 dBFS on silence against about -66 for neighbours).

## Consequence for the v3 results

The v3 training used click-onset offsets (via `capture_lag`) for Super-Sonic, Vibrolux, Peavey 5150, 6505+, Mesa and Orange, and none for the JCM800 and Twin. The music-derived offsets here differ from the click offsets by up to about 25 samples (Peavey G3 -52 vs -50, G5 +2 vs -10, G6 +406 vs +416; Orange G9 +17 vs +9, G10 +18 vs +6), so some v3 training targets were misaligned by that much. That is small against the multi-hundred-sample offsets that were corrected, but it is the same order as A2 timing detail, so it should not be ignored when reading v3 numbers for the Peavey and Orange. Peavey G4's level error was also included in the v3 Peavey training data unmodified.

## Per-capture detail

### Marshall JCM800 2203 (High)

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | VALID | - | - |
| 1.5 | VALID | - | - |
| 2 | VALID | - | - |
| 2.5 | VALID | - | - |
| 3 | VALID | - | - |
| 3.5 | VALID | - | - |
| 4 | VALID | - | - |
| 4.5 | VALID | - | - |
| 5 | VALID | - | - |
| 5.5 | VALID | - | - |
| 6 | VALID | - | - |
| 6.5 | VALID | - | - |
| 7 | VALID | - | - |
| 7.5 | VALID | - | - |
| 8 | VALID | - | - |
| 8.5 | CORRECTED | timing: click onset +674 vs set median +6 (+668 samples); music xcorr vs G8 = +692 samples (click predicts +668) -> music-derived offset +692 | shift -692 samples (music cross-correlation offset) |
| 9 | VALID | - | - |
| 9.5 | VALID | - | - |
| 10 | VALID | - | - |

### Fender 57 Custom Twin

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | VALID | - | - |
| 2 | VALID | - | - |
| 3 | VALID | - | - |
| 4 | VALID | - | - |
| 5 | VALID | - | - |
| 6 | VALID | - | - |
| 7 | VALID | - | - |
| 8 | VALID | - | - |
| 9 | VALID | - | - |
| 10 | VALID | - | - |

### Fender Super-Sonic, Bassman channel

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | CORRECTED | timing: click onset +8 vs set median +24 (-16 samples); music xcorr vs G3 = -20 samples (click predicts -16) -> music-derived offset -20 | shift +20 samples (music cross-correlation offset) |
| 2 | CORRECTED | timing: click onset +16 vs set median +24 (-8 samples); music xcorr vs G3 = -10 samples (click predicts -8) -> music-derived offset -10 | shift +10 samples (music cross-correlation offset) |
| 3 | VALID | - | - |
| 4 | SUSPECT | noise spike: silence renders at -51 dBFS vs -67 dBFS typical for neighbours | - |
| 5 | VALID | - | - |
| 6 | VALID | - | - |
| 7 | SUSPECT | noise spike: silence renders at -52 dBFS vs -66 dBFS typical for neighbours | - |
| 8 | VALID | - | - |
| 9 | VALID | - | - |
| 10 | SUSPECT | noise spike: silence renders at -49 dBFS vs -63 dBFS typical for neighbours | - |

### Fender Super-Sonic, Vibrolux channel

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | VALID | - | - |
| 2 | VALID | - | - |
| 3 | VALID | timing: click onset +16 vs set median +21 (-5 samples); music xcorr vs G2 = +2 samples (click predicts -3) -> music-derived offset +0; click onset flagged -5 samples but the music is aligned to the set within 4 samples (offset +0): false alarm from the click probe, no correction needed | - |
| 4 | VALID | - | - |
| 5 | VALID | - | - |
| 6 | VALID | - | - |
| 7 | VALID | - | - |
| 8 | VALID | - | - |
| 9 | VALID | - | - |
| 10 | VALID | - | - |

### Peavey 5150

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | CORRECTED | timing: click onset +24 vs set median +62 (-38 samples); music xcorr vs G2 = -12 samples (click predicts -13) -> music-derived offset -38 | shift +38 samples (music cross-correlation offset) |
| 2 | CORRECTED | timing: click onset +37 vs set median +62 (-26 samples); music xcorr vs G1 = +12 samples (click predicts +13) -> music-derived offset -26 | shift +26 samples (music cross-correlation offset) |
| 3 | CORRECTED | timing: click onset +13 vs set median +62 (-50 samples); music xcorr vs G2 = -27 samples (click predicts -24) -> music-derived offset -52 | shift +52 samples (music cross-correlation offset) |
| 4 | SUSPECT | timing: click onset +15 vs set median +62 (-48 samples); music xcorr vs G3 = +3 samples (click predicts +2) -> music-derived offset -46; level: music RMS -3.8 dB, sustained-sine output -4.1 dB, metadata loudness -4.3 dB vs the interpolation of the neighbouring captures; the same signed offset appears in the file's own metadata and at every sustained input level, consistent with a capture-chain lev | shift +46 samples (music cross-correlation offset); suggested level +4.1 dB (estimate, not applied) |
| 5 | VALID | timing: click onset +53 vs set median +62 (-10 samples); music xcorr vs G4 = +49 samples (click predicts +38) -> music-derived offset +2; click onset flagged -10 samples but the music is aligned to the set within 4 samples (offset +2): false alarm from the click probe, no correction needed | - |
| 6 | CORRECTED | timing: click onset +478 vs set median +62 (+416 samples); music xcorr vs G5 = +415 samples (click predicts +425) -> music-derived offset +406 | shift -406 samples (music cross-correlation offset) |
| 7 | CORRECTED | timing: click onset +72 vs set median +62 (+10 samples); music xcorr vs G6 = -405 samples (click predicts -406) -> music-derived offset +10 | shift -10 samples (music cross-correlation offset) |
| 8 | CORRECTED | timing: click onset +82 vs set median +62 (+20 samples); music xcorr vs G7 = +12 samples (click predicts +10) -> music-derived offset +22 | shift -22 samples (music cross-correlation offset) |
| 9 | CORRECTED | timing: click onset +471 vs set median +62 (+408 samples); music xcorr vs G8 = +390 samples (click predicts +389) -> music-derived offset +410 | shift -410 samples (music cross-correlation offset) |
| 10 | SUSPECT | timing: click onset +586 vs set median +62 (+524 samples); music xcorr vs G9 = +115 samples (click predicts +115) -> music-derived offset +524; noise spike: silence renders at -53 dBFS vs -65 dBFS typical for neighbours | shift -524 samples (music cross-correlation offset) |

### Peavey 6505+ Scooped

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | VALID | - | - |
| 2 | VALID | - | - |
| 3 | VALID | - | - |
| 4 | VALID | - | - |
| 5 | VALID | - | - |
| 6 | VALID | - | - |
| 7 | VALID | - | - |
| 8 | VALID | - | - |
| 9 | VALID | - | - |
| 10 | VALID | - | - |

### Mesa Dual Rectifier

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | VALID | - | - |
| 2 | VALID | - | - |
| 3 | VALID | - | - |
| 4 | VALID | - | - |
| 5 | VALID | - | - |
| 6 | VALID | - | - |
| 7 | VALID | - | - |
| 8 | VALID | - | - |
| 9 | VALID | - | - |
| 10 | VALID | - | - |

### Orange Dual Terror

| Gain | Status | Evidence | Correction |
|---:|---|---|---|
| 1 | VALID | - | - |
| 2 | SUSPECT | level: music RMS +3.2 dB, sustained-sine output +3.3 dB, metadata loudness +3.2 dB vs the interpolation of the neighbouring captures; the same signed offset appears in the file's own metadata and at every sustained input level, consistent with a capture-chain level error (a constant drop at saturation while both neighbours are equal is not typical amp behaviour); suggested level correction (ESTIMA | ; suggested level -3.2 dB (estimate, not applied) |
| 3 | VALID | - | - |
| 4 | VALID | - | - |
| 5 | VALID | - | - |
| 6 | VALID | - | - |
| 7 | SUSPECT | noise spike: silence renders at -44 dBFS vs -61 dBFS typical for neighbours | - |
| 8 | CORRECTED | timing: click onset +15 vs set median +5 (+10 samples); music xcorr vs G7 = +11 samples (click predicts +10) -> music-derived offset +11 | shift -11 samples (music cross-correlation offset) |
| 9 | CORRECTED | timing: click onset +14 vs set median +5 (+9 samples); music xcorr vs G7 = +17 samples (click predicts +9) -> music-derived offset +17 | shift -17 samples (music cross-correlation offset) |
| 10 | CORRECTED | timing: click onset +11 vs set median +5 (+6 samples); music xcorr vs G7 = +18 samples (click predicts +6) -> music-derived offset +18 | shift -18 samples (music cross-correlation offset) |

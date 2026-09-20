# Automated preview of the fixed-gain listening set (NOT a perceptual result)

**Read this AFTER you listen if you want your ratings unbiased; read it BEFORE if you only want to know where to spend your time.** Human listening is still pending; nothing here is an audible claim.

For every clip these are objective distances between each candidate and the REAL reference, computed from the same audio files you will hear (`scripts/pl_listening_auto.py`). Lower = closer to the real amp. They are proxies: **MR-LSD** is a multi-resolution log-spectral distance on the level-matched versions (tone, saturation and dynamics without loudness); **EQ**, **HF>3k**, **crest** and **dyn range** are level-independent feature differences (candidate minus real, dB); **level** is the native-level difference; **lm-ESR** is a waveform error after level matching (a weak guide for saturated material). To keep the blind test blind this page reports by amp, virtual gain and playing intensity only, with no item numbers or A-D letters. The hidden real amp in each item is not scored (it would be zero).

## 1. Overall (mean over all single clips, MR-LSD dB, lower is closer)

| Model | MR-LSD | EQ err | |HF| | |crest| | |dyn range| | |level| | lm-ESR |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 C3 | 5.79 | 0.64 | 1.10 | 1.10 | 2.06 | 1.27 | 0.080 |
| B seed 0 | 6.02 | 0.67 | 1.15 | 0.64 | 1.25 | 0.69 | 0.070 |
| B seed 1 | 5.71 | 0.62 | 1.06 | 1.01 | 1.28 | 0.64 | 0.065 |

## 2. By amp and playing intensity (mean MR-LSD; the closest candidate is in bold)

| Amp | intensity | v3 C3 | B seed 0 | B seed 1 | B (mean of seeds) vs v3 C3 |
|---|---|---:|---:|---:|---|
| Vibrolux | soft | 7.42 | 8.17 | **7.15** | B is 3% FARTHER |
| Vibrolux | normal | 6.89 | 6.66 | **6.20** | B is 7% closer |
| Vibrolux | hard | 6.72 | 6.50 | **6.22** | B is 5% closer |
| JCM800 | soft | **5.71** | 6.52 | 6.62 | B is 15% FARTHER |
| JCM800 | normal | 3.89 | 3.80 | **3.72** | B is 3% closer |
| JCM800 | hard | **4.12** | 4.48 | 4.36 | B is 7% FARTHER |

## 3. Per setting (single clips; MR-LSD, closest bold; B is the mean of its two seeds)

| Amp | virtual gain | intensity | v3 C3 | B | predicted closest | B seeds agree? | trained on this setting |
|---|---:|---|---:|---:|---|---|---|
| Vibrolux | 3 | soft | 4.58 | 5.00 | v3 C3 | yes | B |
| Vibrolux | 3 | normal | 6.23 | 5.49 | B | yes | B |
| Vibrolux | 3 | hard | 7.06 | 6.72 | B | yes | B |
| Vibrolux | 5 | soft | 5.66 | 7.64 | v3 C3 | yes | v3 C3 |
| Vibrolux | 5 | normal | 4.65 | 6.69 | v3 C3 | yes | v3 C3 |
| Vibrolux | 5 | hard | 5.19 | 6.63 | v3 C3 | yes | v3 C3 |
| Vibrolux | 7 | soft | 7.54 | 9.24 | v3 C3 | yes | B |
| Vibrolux | 7 | normal | 8.23 | 7.00 | B | yes | B |
| Vibrolux | 7 | hard | 8.48 | 6.40 | B | yes | B |
| Vibrolux | 10 | soft | 11.88 | 8.76 | B | yes | v3 C3, B |
| Vibrolux | 10 | normal | 8.44 | 6.55 | B | yes | v3 C3, B |
| Vibrolux | 10 | hard | 6.14 | 5.69 | B | yes | v3 C3, B |
| JCM800 | 2 | soft | 5.41 | 4.79 | B | yes | B |
| JCM800 | 2 | normal | 5.39 | 4.37 | B | yes | B |
| JCM800 | 2 | hard | 5.70 | 6.25 | v3 C3 | yes | B |
| JCM800 | 5 | soft | 8.80 | 10.01 | v3 C3 | yes | v3 C3 |
| JCM800 | 5 | normal | 2.33 | 3.71 | v3 C3 | yes | v3 C3 |
| JCM800 | 5 | hard | 2.36 | 3.34 | v3 C3 | yes | v3 C3 |
| JCM800 | 8 | soft | 4.27 | 7.15 | v3 C3 | yes | neither |
| JCM800 | 8 | normal | 4.99 | 4.44 | B | yes | neither |
| JCM800 | 8 | hard | 5.21 | 4.95 | B | yes | neither |
| JCM800 | 10 | soft | 4.37 | 4.34 | B | yes | v3 C3, B |
| JCM800 | 10 | normal | 2.85 | 2.52 | B | yes | v3 C3, B |
| JCM800 | 10 | hard | 3.19 | 3.13 | B | yes | v3 C3, B |

*Trained on this setting* marks which model had this virtual gain as a training anchor. A model is naturally favoured where it was trained (v3 C3 at G5 and G10; B at G2, G4, G10 on the JCM800 and G3, G7, G10 on the Vibrolux), so the most informative rows are the ones where the model was NOT trained on the setting.

Predicted closest (single clips): Vibrolux: B 7, v3 C3 5; JCM800: B 7, v3 C3 5.

## 4. Soft-to-hard sequences: does the candidate change character the way the real amp does?

Candidate minus real for the CHANGE between the soft and the hard part of each sequence (0 = behaves like the real amp). Real level rise is the real amp's own soft-to-hard loudness increase.

| Amp | virtual gain | real level rise (dB) | model | level rise error | HF change error | crest change error | dyn-range change error |
|---|---:|---:|---|---:|---:|---:|---:|
| Vibrolux | 3 | 13.7 | v3 C3 | +3.5 | -3.0 | -1.9 | +3.8 |
| Vibrolux | 3 | 13.7 | B seed 0 | -1.3 | -3.5 | +1.5 | -1.6 |
| Vibrolux | 3 | 13.7 | B seed 1 | -0.9 | -3.0 | +0.9 | -1.2 |
| Vibrolux | 5 | 11.3 | v3 C3 | +0.1 | -4.2 | +0.2 | -1.9 |
| Vibrolux | 5 | 11.3 | B seed 0 | -2.1 | -3.6 | +0.6 | -2.8 |
| Vibrolux | 5 | 11.3 | B seed 1 | -1.9 | -3.1 | +2.4 | -2.2 |
| Vibrolux | 7 | 9.2 | v3 C3 | -3.5 | -3.2 | +2.4 | -5.7 |
| Vibrolux | 7 | 9.2 | B seed 0 | -2.4 | -2.5 | +0.8 | -1.1 |
| Vibrolux | 7 | 9.2 | B seed 1 | -2.3 | -2.0 | +3.8 | -1.0 |
| Vibrolux | 10 | 6.4 | v3 C3 | -4.0 | +1.1 | +5.5 | +6.8 |
| Vibrolux | 10 | 6.4 | B seed 0 | -3.2 | -0.1 | +2.3 | +3.5 |
| Vibrolux | 10 | 6.4 | B seed 1 | -3.2 | +0.7 | +4.7 | +3.3 |
| JCM800 | 2 | 10.5 | v3 C3 | +3.1 | +1.5 | -2.0 | +0.7 |
| JCM800 | 2 | 10.5 | B seed 0 | -2.3 | +1.9 | -0.6 | -0.9 |
| JCM800 | 2 | 10.5 | B seed 1 | -1.8 | +2.1 | -0.1 | -0.7 |
| JCM800 | 5 | 3.1 | v3 C3 | +2.0 | +3.6 | -1.0 | -5.4 |
| JCM800 | 5 | 3.1 | B seed 0 | +0.1 | +3.5 | -0.3 | -1.8 |
| JCM800 | 5 | 3.1 | B seed 1 | +0.2 | +3.7 | -0.2 | -2.2 |
| JCM800 | 8 | 1.5 | v3 C3 | -0.7 | +0.3 | +0.3 | -0.1 |
| JCM800 | 8 | 1.5 | B seed 0 | -0.8 | +1.6 | +0.2 | -0.1 |
| JCM800 | 8 | 1.5 | B seed 1 | -0.8 | +1.7 | +0.2 | -0.1 |
| JCM800 | 10 | 1.5 | v3 C3 | -0.6 | -1.0 | +0.9 | +0.3 |
| JCM800 | 10 | 1.5 | B seed 0 | -1.0 | +0.3 | +0.9 | +0.3 |
| JCM800 | 10 | 1.5 | B seed 1 | -1.0 | +0.6 | +0.4 | +0.3 |

Mean absolute soft-to-hard behaviour error over the eight sequences and four measures: v3 C3 2.33 dB, B seed 0 1.54 dB, B seed 1 1.65 dB.

## 5. Where to listen first

Single clips with the largest predicted B-versus-v3-C3 difference, EXCLUDING cases where the predicted winner is the only model trained on that setting (those are expected), plus any case where the two B seeds disagree:

| Amp | virtual gain | intensity | B MR-LSD | v3 C3 MR-LSD | predicted closest | B seeds agree? | trained on this setting |
|---|---:|---|---:|---:|---|---|---|
| JCM800 | 8 | soft | 7.15 | 4.27 | v3 C3 | yes | neither |
| Vibrolux | 10 | soft | 8.76 | 11.88 | B | yes | v3 C3, B |
| Vibrolux | 7 | soft | 9.24 | 7.54 | v3 C3 | yes | B |
| Vibrolux | 10 | normal | 6.55 | 8.44 | B | yes | v3 C3, B |
| JCM800 | 10 | normal | 2.52 | 2.85 | B | yes | v3 C3, B |
| JCM800 | 8 | normal | 4.44 | 4.99 | B | yes | neither |
| JCM800 | 2 | hard | 6.25 | 5.70 | v3 C3 | yes | B |
| Vibrolux | 3 | soft | 5.00 | 4.58 | v3 C3 | yes | B |

Soft-to-hard sequences where v3 C3 and B differ most in behaviour (positive = B keeps the real amp's soft-to-hard behaviour better): Vibrolux G10 (+1.7 dB); Vibrolux G7 (+1.7 dB); JCM800 G5 (+1.5 dB); smallest or reversed: JCM800 G8 (-0.3 dB); Vibrolux G5 (-0.7 dB).


## 6. Reading this honestly

- These numbers can agree or disagree with what you hear. A log-spectral distance treats a 2 dB tilt and a 2 dB level-independent HF error alike, ignores playing feel, and does not weight what guitarists notice.
- A large predicted difference is a reason to listen to that item first, not a verdict. Items where B and v3 C3 are predicted to be similar are the ones where your ears add the most information.
- The earlier objective diagnostics (`docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md`) measured the same clips' behaviour differently (feature errors against the real capture across musical-input levels); this preview is the per-clip view of the same effects.
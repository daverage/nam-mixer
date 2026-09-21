## 1. Starting point (verified)

Branch `research/continuous-gain-model`. Inputs reviewed: Phase 4E results and frozen manifest, the fixed-virtual-gain playability diagnostic, the weight-band test, the automated listening preview, the first human listening (one listener, speakers, ten shortlist items, `docs/phase4e/listening_fixed_gain/RESULTS_first_listening.md`), the teacher-versus-student audio, and the exported Phase 4E B and v3 C3 models (unchanged; `docs/phase4e/models/`). Phase 4E configuration B is the primary starting point for both amps; v3 C3 is the second baseline. Nothing already answered was re-run: everything in section 2 is computed from existing evaluation data (`scripts/fc_gaps.py`, `fc_coverage.py`); the only new renders are the teacher-only candidate comparison (section 3) and the new models.

## 2. Assessment of the existing B models (parts A-F)

Evidence: real captures, B (both seeds) and C3 evaluated at native musical level and at four playing intensities; continuous Input-gain sweeps; the soft-to-hard playability cases; the teacher/student audio and the first listening. The listening is used as evidence about specific weaknesses, not as an overall ranking.

### Marshall JCM800

- **A. Reproduced convincingly.** Everything from G3 to G10 at native level: the ten physical positions and the nine genuine half-steps are within about 1 dB in level, EQ, HF and crest, the only flagged positions (working flags: level, EQ, HF, crest 1.0 dB, dynamic range 1.5 dB) being G1, G1.5 and G2. The continuous Input-gain sweep tracks the real progression within 0.8-0.9 dB (level, HF) and 1.1 dB (crest). Hard picking and the G5 setting were rated at the real amp's level by ear (JCM800 G5 hard 91/90 for the B seeds against 92 for the hidden real amp).
- **B. Weak.** The low-gain end (G1-G2): dynamic range 1.5-2.4 dB off and G1 crest/level. Ear: JCM800 G8 normal, B 71/61 against C3 90, although every measured feature there is within 0.5 dB, so this is not explained by the measurements; G8 soft was not rated for B.
- **C. Sweep transitions.** One: dynamic range deviates from the real progression by up to 2.4 dB between Input gains of about -18 and -12 dB (virtual G2-G3). No opposite-direction steps, no level or HF gap.
- **D. Soft-to-hard.** 3 flagged combinations of 36: G1 and G2 level swings (-1.6 and -1.9 dB) and G10 crest (+1.2 dB swing, +1.5 dB too peaky at +6 dB). The G4-G10 range stays within about 0.5 dB level, 1 dB HF, 1.3 dB dynamic range at every musical level.
- **E. Teacher or student.** Two of the three flags are already in the teacher (the G1/G2 level swings); the G10 crest departure is student-only. Student-vs-teacher level-matched ESR is 0.005-0.007 for total levels up to +8 dB above native, then 0.011 (+8..+14), 0.026 (+14..+20), 0.056 (above +20).
- **F. Omitted captures.** No demonstrated hole. Teacher errors at the omitted positions (G3, G5, G6.5, G8, G9) are no larger than at the anchors (nominal level 0.25 vs 0.21, HF 0.33 vs 0.15, crest 0.66 vs 0.43, dynamic range 0.12 vs 0.67 dB), and swings are smaller (level 0.38 vs 1.11 dB). The real captures from G4 to G10 sound alike (Phase 4B plateau; G4 to G10 differ by 0.1-0.7 dB in every measure, `docs/CONTINUOUS_GAIN_WEIGHT_BAND_TEST.md`).

### Fender Super-Sonic Vibrolux

- **A. Reproduced convincingly.** The output-level progression (level errors 0.3-0.9 dB across all ten positions), transient behaviour, and ear-wise several settings: G5 normal for B seed 1 (91 against 95 for C3), G7 hard (85 and 50-unrated), G10 soft (B 40/57 against C3 27).
- **B. Weak.** (1) HF is 1-2.6 dB too dark from G1 to G7 for B seed 0 and G1 to G5 for seed 1 (an audible "dark clean" tendency; seed-dependent). (2) Dynamic range 1.5-3 dB too low at G5-G10 (over-compressed at high gain). (3) By ear: normal picking at the top gain and G5 (B seed 0), and the soft-to-hard sequence at G7: G10 normal B 35/35 against C3 90, G5 normal B seed 0 40 against 95, G7 sequence B 25/32 against C3 74; soft playing is poor for every model (about 57-59).
- **C. Sweep transitions.** HF deviates from the real progression by more than 1.5 dB across Input gains of about -15 to -3 dB for seed 0 and -21 to 0 dB for seed 1 (virtual G2-G5, the clean-to-breakup zone), and dynamic range by 2.2 dB near +3 dB (G6-G7). Level never deviates by more than 0.9 dB. No opposite-direction step.
- **D. Soft-to-hard.** 22 of 32 combinations flagged: HF swings of -3.1 to -4.4 dB across G3-G7, level swings -2.5 to -3.0 dB at G7, G8, G10, G10 dynamic range (-3.0 fixed, +4.3 swing) and crest (+5.2 swing).
- **E. Teacher or student.** 16 teacher, 3 student only, 3 both. The teacher carries the drift (the envelope-driven blend moves 1.2 to 5.5 physical positions over 18 dB of playing). Student-only: HF 1.9-2.0 dB too dark at G1-G2, the top-gain hard-playing crest. The student's departure from its teacher is 0.006-0.007 up to +8 dB total level and rises to 0.013, 0.022 and 0.049 in the higher bins; the clip you rated 35/35 (G10 normal, total +14 dB) has a teacher that is close to the real amp (mean weights G10 0.56, G7 0.34; distance 5.10) and a student that is not (6.67).
- **F. Omitted captures.** Weak association: teacher errors at G5 and G8 are similar to the anchors for level, HF and crest (0.38/0.51/0.24 against 0.32/0.72/0.27 dB) and larger only for dynamic range (1.69 against 1.12 dB); swings are larger at the omitted positions (level 2.24 against 1.57, HF 3.54 against 2.83, crest 1.23 against 0.93) but the anchors drift as well, so the drift is structural (anchor gaps against the playing range), not a missing capture.

### What the assessment says about action

- No sound is demonstrably missing on either amp: the teacher (built only from the real captures) tracks each physical position within about 1 dB, and adding captures does not change its errors (section 3).
- The playing-intensity drift on the Vibrolux is structural and cannot be removed by capture choice or crossfade shape (weight-band test, ambiguity table); it is accepted and documented.
- One weakness IS specific, checkable and fixable: the trained model departs from its own target at high total levels, where the guitar training material barely reaches. That is the change made below.

## 3. Capture sets and anchor mapping (teacher-only comparison)

Method: for each candidate capture set the envelope-driven teacher (the training target, built from real captures only) is rendered at every physical position, the held-out DIs and the four playing intensities, plus a continuous Input-gain sweep, with anchors on the response-distance rule mapped onto the plugin-compatible range -20 to +14 dB. A teacher-only result is not proof that a trained NAM improves (`scripts/fc_teacher_eval.py`, `fc_candidates.py`).

| Amp | Set | mean nominal error level / HF / crest / dyn (dB) | flagged positions level/HF/crest/dyn/EQ | mean swing level / HF / crest / dyn | worst sweep step ratio |
|---|---|---|---|---|---:|
| Vibrolux | B: G1 G2 G3 G4 G7 G10 | 0.27 / 0.63 / 0.26 / 1.10 | 0/2/0/5/0 | 1.47 / 3.04 / 0.98 / 1.53 | 2.0 |
| Vibrolux | B + G5 | 0.25 / 0.61 / 0.34 / 1.21 | 0/1/0/4/0 | 1.59 / 2.91 / 1.08 / 1.52 | 2.2 |
| Vibrolux | B + G6 | 0.28 / 0.59 / 0.27 / 1.21 | 0/2/0/5/0 | 1.61 / 2.88 / 1.06 / 1.80 | 2.0 |
| Vibrolux | B + G8 | 0.24 / 0.65 / 0.23 / 1.14 | 0/2/0/5/0 | 1.50 / 3.03 / 1.01 / 1.59 | 2.0 |
| JCM800 | B: G1 G2 G4 G10 | 0.21 / 0.28 / 0.59 / 0.29 | 0/0/0/0/0 | 0.56 / 0.43 / 0.76 / 1.05 | 4.8 |
| JCM800 | B + G5 | 0.23 / 0.27 / 0.60 / 0.31 | 0/0/0/0/0 | 0.60 / 0.39 / 0.59 / 0.82 | 4.8 |
| JCM800 | B + G6 | 0.21 / 0.23 / 0.58 / 0.28 | 0/0/0/0/0 | 0.51 / 0.42 / 0.71 / 1.02 | 3.4 |
| JCM800 | B + G7 | 0.22 / 0.24 / 0.62 / 0.29 | 0/0/0/0/0 | 0.53 / 0.44 / 0.74 / 1.07 | 4.9 |
| JCM800 | B + G8 | 0.21 / 0.24 / 0.56 / 0.28 | 0/0/0/0/0 | 0.57 / 0.48 / 0.77 / 1.07 | 4.2 |

**Capture sets: unchanged.** For the Vibrolux, G5, G6 and G8 change no error by more than about 0.1 dB except crest (worse with G5, 0.26 to 0.34) and dynamic range (worse with G5 and G6, 1.10 to 1.21) and leave the swing and the sweep steps unchanged: none contributes a sound the current set lacks, and G8 in particular does not (level 0.24 against 0.27, HF 0.65 against 0.63, dynamic range 1.14 against 1.10; not selected on the earlier plateau-compression metric alone, as instructed). For the JCM800 the B teacher has no flagged position at all and an intermediate capture (G5, G6, G7 or G8) moves nothing beyond ±0.05 dB nominal and ±0.2 dB swing: the large gap between G4 and G10 does not hide a missing crunch sound, and G5 is not added merely because v3 C3 was trained on it. This agrees with the earlier C10 teacher test.

**Anchor mapping: plugin-compatible, otherwise the response-distance rule.** The Phase 4E B mapping starts at -22 dB, below a player's -20 dB minimum. Teacher-only comparison of the B set on three ranges:

| Amp | Range (dB) | nominal dyn error | mean swing level / HF / crest / dyn | flagged dyn positions |
|---|---|---:|---|---:|
| Vibrolux | -22..+14 (frozen B) | 1.40 | 1.89 / 2.93 / 1.18 / 1.75 | 6 |
| Vibrolux | **-20..+14 (chosen)** | **1.10** | **1.47 / 3.04 / 0.98 / 1.53** | **5** |
| Vibrolux | -20..+20 | 1.97 | 2.66 / 2.76 / 1.81 / 2.35 | 9 |
| JCM800 | -22..+14 (frozen B) | 0.32 | 0.62 / 0.32 / 0.70 / 0.74 | 1 |
| JCM800 | **-20..+14 (chosen)** | **0.29** | **0.56 / 0.43 / 0.76 / 1.05** | **0** |
| JCM800 | -20..+20 | 0.43 | 0.92 / 0.23 / 0.61 / 0.49 | 1 |

Moving only the bottom anchor to -20 dB is at least as good as the frozen range (better on the Vibrolux, comparable on the JCM800) and removes the inaccessible endpoint; stretching the top to +20 dB is worse on the Vibrolux and mixed on the JCM800, so the top stays at +14. Resulting anchors: JCM800 G1 -20.0, G2 -9.2, G4 -1.7, G10 +14.0 dB; Vibrolux G1 -20.0, G2 -15.6, G3 -7.1, G4 -2.2, G7 +5.7, G10 +14.0 dB. No mapping was optimised per DI, playing level or metric.

## 4. Training coverage: the specific, checkable weakness

The training input is the official NAM signal plus guitar DIs at level offsets of -16 to +8 dB. Playback at the higher virtual gains puts the median level far above that: with the native `clean_mayer` DI at -28.8 dBFS median, G10 (Input +14 dB) sits at -14.8 dBFS for normal playing and -8.8 dBFS for hard playing. In the training input only 3% of the guitar audio's active envelope lies between -16 and -10 dBFS and none above -10 dBFS; that range is otherwise reached only by the synthetic official input. Consistent with this, the trained models' departure from their own teacher (level-matched ESR) is 0.006-0.007 for total levels (Input gain plus playing offset, dB re native) between -8 and +8 dB, then 0.013 (+8..+14), 0.022-0.026 (+14..+20) and 0.049-0.056 (above +20) on both amps and for both configurations A and B; on the Vibrolux it also rises to 0.032 below -16 dB, where the guitar material also stops.

## 5. Decision: Option 2, one targeted configuration per amp

For each amp the existing B is NOT the strongest justified candidate, because a specific, evidenced training-coverage gap and an inaccessible endpoint can be addressed at low cost, and no capture change is justified. The single new configuration per amp ("FC") keeps the B capture set and the standard A2 architecture and export, and changes only: (1) the anchor mapping (bottom anchor -20 dB, above), and (2) the guitar training material, now at offsets of -32, -24, -16, -8, 0, +8, +14, +20 dB (20 s per DI and offset; validation at -24, 0, +8, +20 dB), covering the playback range of both Input gain and playing intensity. Frozen before training (`docs/final/manifest_frozen.json`, commit `99ca9b6`): captures and hashes, alignment (none needed), anchors, data, architecture, 60 epochs and optimiser, seeds 0 and 1, evaluation cases, six success criteria and the selection rule. The models train on 1.42x more audio than B (about 14,600 optimiser updates against 10,260); that confound is disclosed.

# Phase 4E - Train and validate the best practical single continuous-gain NAM

## 1. Objective

Develop and validate a workflow that turns multiple fixed-gain captures of **one amplifier** into **one ordinary standard `.nam`**. The guitarist controls its usable clean-to-saturated progression through the normal input-gain control of a conventional NAM player. The generated model should approximate the source amp's gain and saturation, compression and dynamics, spectral balance and EQ, harmonics, transient behaviour, output-level progression, and response to different DI levels. An exact correspondence between a physical G1–G10 marking and an input-gain percentage is *not* required.

The objective is the best practical **single playable NAM**, not the lowest capture count, the smallest average ESR, or a particular compression improvement. Training time and capture count matter as practical costs. A model with four, six or ten useful training captures may be preferable to a three-capture model. Maintain standard NAM compatibility; do not add independent gain conditioning or unreported host-side processing.

## 2. Review and preserve the existing research

Work on `research/continuous-gain-model`. Before modifying training code, read and inspect the code and underlying artifacts for:

- `docs/CONTINUOUS_GAIN_v3_COMPARISON.md`
- `docs/CONTINUOUS_GAIN_PHASE4_QA.md`
- `docs/CONTINUOUS_GAIN_PHASE4_PROFILES.md`
- `docs/CONTINUOUS_GAIN_PHASE4_MAPPING.md`
- `docs/CONTINUOUS_GAIN_PHASE4_SELECTION.md`
- `docs/CONTINUOUS_GAIN_PHASE4E_PLAN.md`

Preserve all original captures, v3 models, research outputs, and reports. Store new Phase 4E results under new paths. Verify whether the proposed training spacing can actually be implemented in the existing data-preparation/training pipeline; do not assume that changing playback gain changes neural training.

## 3. Pilot amplifiers and capture sets

Phase 4D proposed the following **primary** sets. These are response-profile candidates, not yet demonstrated neural-training winners:

| Amplifier | Primary training captures |
|---|---|
| Marshall JCM800 2203, High | G1, G2, G4, G10 (4 captures) |
| Fender Super-Sonic, Vibrolux channel | G1, G2, G3, G4, G7, G10 (6 captures) |

The JCM800 has genuine half-step references for additional validation. The Vibrolux tests the clean-Fender trade-off in Phase 4C: a playback mapping can improve compression while worsening native output level and crest factor. The nearby JCM800 set G1/G2/G4/G9 is an **optional follow-up**, not an additional primary run.

## 4. Freeze the primary experiment before training

Create a machine-readable manifest and record its repo commit before the first training run. It must specify amplifier/channel, original capture identities and hashes, QA status and verified alignment corrections, selected positions, training anchors, architecture, DI/training audio, augmentation levels, optimiser and training budget, seeds, output paths, evaluation material, playback mappings, metrics, and comparison rules.

Use only capture audio eligible under Phase 4A. Keep original audio intact; use documented corrected renders. Do not apply an *estimated* level correction to a suspect capture as if it were verified. Changes forced by a discovered implementation defect must be documented as amendments, with exploratory results distinguished from the frozen comparison. Do not revise the selection or success criteria after viewing results and present them as predetermined.

## 5. Train the two primary configurations per amplifier

**Configuration A - selected captures, original fixed physical-position spacing**

| Amplifier | Training anchors (dB) |
|---|---|
| JCM800 G1/G2/G4/G10 | G1 −22, G2 −18, G4 −10, G10 +14 |
| Vibrolux G1/G2/G3/G4/G7/G10 | G1 −22, G2 −18, G3 −14, G4 −10, G7 +2, G10 +14 |

**Configuration B - same captures, response-distance training spacing**

| Amplifier | Training anchors (dB) |
|---|---|
| JCM800 G1/G2/G4/G10 | G1 −22.0, G2 −10.5, G4 −2.6, G10 +14.0 |
| Vibrolux G1/G2/G3/G4/G7/G10 | G1 −22.0, G2 −17.4, G3 −8.4, G4 −3.1, G7 +5.3, G10 +14.0 |

The response-distance anchors are a **hypothesis**, not an established mapping between the physical gain knob and input drive. Configuration A versus the old v3 C3 baseline investigates a new capture set under the original training-anchor rule, although count and placement both change; do not claim it isolates placement alone. Configuration B versus A isolates the training-anchor rule **within each new selected set**, provided all other training settings remain comparable.

Retain the existing v3 C3 (G1/G5/G10), C5 and C10 models as baselines; do not retrain them unless there is a documented incompatibility. For each new A and B configuration use two predeclared seeds. Keep the standard A2 architecture and the existing trainer/data recipe unless fixing a verified defect. No gain-conditioned architecture, post-EQ or added processing in this experiment.

## 6. Training comparability and timing

Keep architecture, sample rate, alignment, DI material, augmentation, optimiser, batch size, learning-rate schedule, and effective optimisation budget controlled between A and B. Record **total optimiser updates, effective exposure per physical capture, total audio duration and actual wall-clock time**. The existing v3 C3/C5/C10 models may have different optimisation exposure because of capture count; explicitly disclose that confound rather than credit all differences to subset selection.

Measure user-relevant generation time including QA/profile analysis, selection, data preparation, training, export, and verification. Do not run an exhaustive capture-count or seed search.

## 7. Verify standard NAM playback

Export each new model as a conventional standard A2 `.nam` and verify that it loads, runs, and produces valid non-clipped/finite audio across the intended input-gain range in a conventional inference/player implementation. The reported NAM-only response must be reproducible without NAM Mixer-specific EQ, dynamics, gain-conditioning metadata or hidden DSP.

A separate table mapping an intuitive virtual knob position to the conventional player's input-gain setting may be provided, but the standard `.nam` must not be described as having a native independently controlled physical gain knob. Report any host-assisted result separately from ordinary player behaviour.

## 8. Common evaluation protocol

Evaluate the existing baselines and new models with identical, reliable reference captures and code. Use held-out DIs `moderate_brit`, `clean_mayer`, `bass_rollin`, each at −12, −6, 0 and +6 dB input offsets. Evaluate real references over low, transition, middle, high and plateau regions, **including positions omitted from neural training**. For JCM800 include the nine genuine half-step captures as an additional independent check. Vibrolux omitted integer positions were already used in subset selection; distinguish neural-training holdout from fully independent research validation.

Report, separately, at each gain and input level:

- **Level/gain:** native output RMS, peaks, progression and signed level error.
- **Spectrum/EQ:** EQ-band errors, HF>3k energy, tilt, and the direction of coloration.
- **Saturation:** THD over multiple input levels, harmonic distribution and progression.
- **Compression:** input/output slopes and the response to changing DI intensity.
- **Dynamics:** crest, dynamic range and transient response.
- **Waveform:** level-matched ESR as a supporting measure only.

Report mean and worst-position errors, low/transition/high-region performance, and regressions by DI input level; include real-amp versus model response-curve plots. Avoid hiding a severe local failure behind a good mean or a composite score. Keep technical evidence distinct from audible claims.

## 9. Separate the effect of training anchors from playback mapping

Train A and B as specified above, then evaluate each with: (i) its intended anchor interpretation, (ii) fixed physical-position spacing where applicable, and (iii) **one ordered model-specific playback mapping fitted only on the fit DIs** and held unchanged on held-out DIs. Do not fit a different playback gain for each held-out recording, input intensity, or metric. Clearly state which numeric input-gain settings were applied and do not mistake equal dB settings for equal physical sounds when A and B were trained on different anchors.

Report training/model effects separately from remapping-only improvements. Include native output level even if playback-map fitting excludes level, because Phase 4C identified clean-Fender compression-versus-level/crest trade-offs. Do not assume the Phase 4C playback mapping is the correct training-anchor rule for new models.

## 10. Revised decision criteria: better overall single-amp experience

The central test: **Does the model offer a convincing, usable progression from cleaner to saturated sounds that better represents the physical amplifier than the prior single NAM?**

Consider (a) coverage of the amp's characteristic regimes and responses, (b) improvements and regressions in *each* measured dimension across all tested DI levels and worst regions, (c) repeatability across seeds, (d) audible assessment where available, and (e) total generation time.

Do **not** require an arbitrary ≥15% compression improvement or declare victory from one improved metric. Better compression accompanied by much worse level, tone or transient response is a trade-off, not a general gain. Likewise, do not reject an otherwise much more useful model solely because compression did not improve by a predetermined percentage. Phase 4C working thresholds can flag differences for examination; they are not validated perceptual thresholds. Differences within seed-to-seed variation are inconclusive.

## 11. Listening package

Render a reproducible, preferably blinded and randomised listening package using held-out guitar DIs: real reference, old v3 C3, new A and new B, at cleaner, transition and saturated settings, plus soft/hard picking examples where the source material allows. Include (i) level-matched comparisons for tone and dynamics and (ii) separate native-level comparisons for loudness progression. Provide a listening worksheet recording which responses sound closer and which have undesirable behaviour.

Do not claim an audible win without listening evidence. If the agent cannot arrange human listening, generate the files and worksheet and mark perceptual evaluation pending.

## 12. Stop points and optional follow-ups

The primary budget is **four new training configurations** (A/B × two amplifiers), **two seeds each**. Do not automatically retrain adaptive C3, C5, C10, neighbouring subsets, new architectures or additional amps. If the first pilot identifies a concrete failure, report one specifically justified follow-up (for example JCM800 G9 instead of G10; a smaller adaptive set; or a larger set to cover a demonstrably missing regime). Request a decision before running it.

The research may reveal that a single input-gain-controlled standard NAM cannot independently follow certain physical-gain-related changes in tone, compression and level. Report that limitation rather than disguising it through unreported host processing or declaring a fundamental impossibility from one unsuccessful configuration.

## 13. Deliverables

Create `docs/CONTINUOUS_GAIN_PHASE4E_RESULTS.md` and new, clearly labelled Phase 4E output directories containing:

1. Frozen experiment manifest, commit, input identities and QA corrections.
2. Training configurations, seeds, optimisation budgets and measured durations.
3. New `.nam` files and conventional-player export/inference verification.
4. Common evaluation against v3 C3/C5/C10, including JCM800 half-step validation.
5. A versus B training-anchor comparison; separate playback-mapping comparison.
6. Real-reference/model curves for saturation, EQ/spectrum, compression, dynamics and level across DI intensities.
7. Mean, worst-position and per-region error tables with uncertainty and regression reporting.
8. A reproducible listening package and worksheet, with listening status clearly labelled.
9. A direct finding about whether selected captures and response-based training anchors make the resulting **single ordinary NAM** more useful, plus what remains mismatched and the next justified experiment.

**Complete the primary Phase 4E pilot and report the evidence. Do not implement the final production workflow or automatically expand the search.**

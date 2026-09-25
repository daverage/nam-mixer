# NAM Mixer - Train One Standard NAM From a Physical Gain Sweep

## Mission

Return the Continuous Gain project to its original objective.

**Train one new, conventional NAM model using captures of the same amplifier at multiple physical Gain settings, so that changing the input gain in an ordinary NAM player produces a response resembling the real amplifier as its physical Gain knob is changed.**

The finished result must be one newly trained, standard `.nam` file-not a collection of NAM anchors, a runtime interpolation system, an external calibration profile or a custom player.

Begin with the complete ten-position JCM800 gain sweep. Establish whether the objective is achievable using all ten captures before investigating how few captures are needed.

Once the ten-capture approach has demonstrated useful results, compare models trained from ten, five, three and two captures under controlled conditions.

The central research questions are:

1. Can one newly trained standard NAM reproduce the real amplifier's changing response across physical Gain 1–10 when controlled solely through ordinary player input gain?
2. Does training across the physical gain range improve on using the original G5 NAM with input gain?
3. Is the existing NAM Mixer Hybrid or Blended approach useful for constructing the training data?
4. How much accuracy is gained by using all ten physical-gain captures rather than a smaller subset?
5. Could three captures produce a model that sounds effectively the same as the ten-capture model at the relevant gain settings?
6. What fundamental limitations prevent a conventional NAM from reproducing an independent physical Gain control?

Do not replace these questions with another investigation of automatic anchor selection or calibrated playback profiles.

---

## 1. Establish the exact technical requirements

The new model must:

* Be newly trained from multi-gain information.
* Use a conventional NAM-compatible architecture and file format.
* Load through the existing standard NAM inference engine.
* Require no changes to an ordinary NAM player's model format.
* Require no simultaneous loading of multiple NAM captures.
* Require no runtime Hybrid or Blended processing.
* Require no custom physical-gain parameter.
* Be controlled using the NAM player's normal pre-model input-gain control.

The expected playback path is:

```text
Guitar / DI
     ↓
Ordinary NAM player input gain
     ↓
ONE newly trained NAM
     ↓
Audio output
```

The `.nam` file must contain the learned behaviour.

An external physical-Gain-to-input-dB mapping may be used for research, training design and evaluation, but the finished model must not depend on a NAM Mixer-specific runtime profile.

Do not claim success merely because one existing G5 NAM can approximate the other captures using a calibrated input-gain curve.

That is an established baseline, not the objective.

---

## 2. Review the existing implementation before designing the experiment

Inspect the current NAM Mixer codebase and identify the relevant implementations of:

* Hybrid model generation.
* Blended model generation.
* Two-capture training and target generation.
* NAM inference and rendering.
* NAM training.
* Model architecture and export.
* Training-data generation.
* Continuous Gain research.
* Existing reference-render comparison tools.

Read the relevant research documents, including:

```text
docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md
docs/CONTINUOUS_GAIN_GENERALIZATION.md
docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md
docs/CONTINUOUS_GAIN_CONDITIONAL_MODEL.md
docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md
docs/CONTINUOUS_GAIN_CROSS_AMP_GENERALIZATION.md
docs/CONTINUOUS_GAIN_MID_GAIN_VS_PROFILE.md
```

Use the actual filenames present in the repository if any differ.

The current Continuous Gain profile builder is not the proposed output of this experiment.

It can be reused for measured baselines, gain-response analysis and evaluation, but do not use it as a substitute for training a new NAM.

### Questions to answer from the code

Determine precisely what Hybrid and Blended currently do.

For each approach, identify:

* Whether it blends rendered audio, intermediate model representations, model weights or training targets.
* Whether it creates a newly trained NAM.
* Whether its output is a conventional `.nam` file.
* Whether it can be applied repeatedly across adjacent gain-capture pairs.
* Whether chaining or generalising it to three or more captures introduces additional model error.
* Whether it preserves endpoint responses.
* Whether it can generate training targets for one final model.

Do not assume Hybrid or Blended can directly produce a multi-gain NAM merely because they can combine two NAMs.

Document what the code actually supports.

---

## 3. Fundamental feasibility check: can input amplitude represent physical Gain?

Before expensive training, explicitly examine the identifiability problem.

A conventional NAM receives an audio waveform. It does not receive the physical amplifier's Gain-knob position as an independent input.

The same waveform entering the new NAM must produce the same model response regardless of how that waveform was produced.

For example:

```text
Quiet DI + large player input gain
                 ↓
          Same input waveform
                 ↑
Hot DI + smaller player input gain
```

A conventional NAM cannot know which player input-gain setting created that waveform.

A real amplifier with its physical Gain knob at two different positions may produce different outputs for the same waveform.

Therefore, an exact emulation of an independent physical Gain control may not be representable by a conventional, single-input NAM.

Do not ignore this limitation.

The experiment must establish whether a useful approximation is achievable over a defined range of input signals and gain settings.

### Required compatibility analysis

Inspect the proposed training inputs and targets for contradictions.

Identify cases in which:

* Two different gain settings produce identical or nearly identical model-input waveforms.
* The required training targets are substantially different.
* The network would therefore receive incompatible input/output examples.

Measure how frequently such conflicts occur under the proposed training scheme.

Do not simply combine all physical-gain capture datasets without addressing this problem.

If the training objective is mathematically inconsistent or clearly unlearnable for a conventional NAM, document the issue before starting expensive training.

If a workable operating range exists, define it explicitly and proceed.

---

## 4. Primary dataset: complete JCM800 physical gain sweep

Use the existing Marshall JCM800 2203 updated High-channel sweep.

The primary training set is:

```text
G1
G2
G3
G4
G5
G6
G7
G8
G9
G10
```

These must be the original real NAM captures taken at the corresponding physical Gain settings.

The dataset also contains half-step captures for independent validation.

Use the real half-step captures at:

```text
G1.5
G2.5
G3.5
G4.5
G5.5
G6.5
G7.5
G8.5
G9.5
```

Do not use the half-step captures to train the primary ten-capture model.

Do not use them to select training hyperparameters, tune the physical-gain mapping or choose the best model checkpoint unless a separate validation protocol explicitly permits it and retains an untouched final test set.

### Known capture anomaly

The existing research identified a latency offset in the G8.5 reference capture.

Detect and report it automatically.

Do not allow it to dominate aggregate waveform metrics.

Do not incorrectly exclude healthy G8 and G9 captures simply because they neighbour G8.5.

Report G8.5 separately, including any analysis-only alignment diagnostics.

Do not silently modify the source `.nam` files.

---

## 5. Define a common input-gain control law

All newly trained models must be evaluated using the same clearly defined relationship between player input gain and intended physical Gain.

Initially establish:

```text
Physical Gain 5
       ↔
Player input gain 0 dB
```

For every other physical Gain setting, define a corresponding player input-gain value.

The mapping may be nonlinear.

However, it must be established before final evaluation and must not be independently optimised for every newly trained model against its withheld reference outputs.

Otherwise, the experiment would conflate improved neural-network training with better per-model gain calibration.

### Separate three questions

The benchmark must distinguish:

**A. Can the network learn a gain-dependent response?**

Test whether the trained model can reproduce the defined multi-gain training targets.

**B. Does that response generalise?**

Test the frozen model at unseen physical-gain positions and on unseen DI material.

**C. Does a better input-gain mapping explain the improvement?**

Compare against the original G5 NAM using the same control law and against the previously established calibrated-G5 baseline.

Do not claim that multi-gain training has improved the model if the measured improvement is caused solely by changing the gain mapping.

### Mapping fairness

For the capture-count experiment, maintain two explicitly labelled conditions if necessary:

1. A common, fixed mapping shared by all models.
2. A mapping derived using only the captures available to the corresponding training configuration.

Do not allow a three-capture model to benefit silently from a mapping calibrated using all ten reference captures.

If the all-ten mapping is used for every model as a controlled research variable, disclose that the smaller model has received additional calibration information.

The user-facing model must still work with ordinary player input gain and no external runtime profile.

---

## 6. Generate training data from the real gain sweep

Use the same source DI audio through each real physical-gain NAM capture.

For each desired physical Gain position:

```text
Original DI
     ↓
Real NAM captured at physical Gain N
     ↓
Reference target audio
```

Construct corresponding training input audio with the predefined virtual input gain:

```text
Original DI
     ↓
Defined virtual input gain for physical Gain N
     ↓
New model's training input
```

The objective is to train one standard NAM that maps the gain-adjusted input to the corresponding real amplifier response.

Do not apply virtual input gain to the reference target merely to make its level match.

Preserve the real amplifier's measured output-level progression.

### Training-data controls

Use:

* Identical source audio across gain positions.
* Consistent sample rate.
* Consistent waveform trim and processing.
* Correct handling of model latency.
* Existing NAM calibration metadata where available.
* Multiple source-input levels.
* Multiple playing dynamics.
* Representative clean and distorted DI material.

Maintain distinct training and held-out DI material.

Do not train and evaluate exclusively on the same six-second DI clips used by the previous benchmarks.

Use existing longer DI recordings or generate additional suitable training material where the repository permits it.

Ensure the training dataset is sufficiently varied to test the intended gain-response behaviour.

---

## 7. Evaluate Hybrid and Blended as training-target generators

The existing Hybrid and Blended implementations are candidates for producing intermediate responses between the real gain captures.

They are not assumed to be necessary.

Test the following approaches where technically supported:

### Method A - Direct multi-gain training

Use the original real capture responses at G1–G10 as the training targets.

No Hybrid or Blended processing is applied.

This establishes whether the conventional NAM architecture can learn the requested response directly from the available physical-gain examples.

### Method B - Hybrid-assisted training

Use the existing Hybrid method to create intermediate training targets between adjacent physical-gain captures.

For the ten-capture sweep:

```text
G1 ↔ G2
G2 ↔ G3
G3 ↔ G4
...
G9 ↔ G10
```

If the existing implementation supports only two inputs, reuse it across adjacent pairs.

Do not invent a new three-way blending formula unless necessary and justified by measured results.

Require continuity at the shared endpoints.

The resulting targets may be used to train one final NAM, but Hybrid must not remain part of the playback signal path.

### Method C - Blended-assisted training

Evaluate the existing Blended approach in the same controlled manner, if its implementation can produce suitable training targets.

Document how its behaviour differs from Hybrid.

Do not assume either method must outperform direct training.

### Critical target-quality check

Before training on generated intermediate targets, compare them against real captures at corresponding intermediate positions wherever genuine reference captures exist.

On the JCM800, use the half-step captures for this diagnostic.

However, **do not use half-step reference outputs to fit the training targets or tune Hybrid interpolation if those same positions are reserved for final evaluation.**

If intermediate-target generation itself is inaccurate, report that separately from the final neural-model training error.

Distinguish:

```text
Error introduced by Hybrid / Blended target generation
```

from:

```text
Error introduced by training the final NAM
```

A final model cannot be expected to reproduce the real amplifier more accurately merely because it has learned a flawed synthetic training target very closely.

---

## 8. First training experiment: all ten captures

Train one new standard NAM using all ten integer physical-gain captures.

Start with the most direct technically feasible training method.

Use an existing production-compatible NAM architecture, not the earlier small custom conditional WaveNet.

The final model must export to a conventional `.nam` and load in ordinary NAM inference.

Use the existing training pipeline and appropriate architecture-specific training procedures.

Do not introduce an independent gain-conditioning input or nonstandard model format.

### Training diagnostics

Record:

* Training configuration.
* Model architecture.
* Parameter count.
* Training duration.
* Training and validation losses.
* Per-gain training error.
* Per-gain held-out-DI error.
* Output-level progression.
* Spectral and dynamic-response progression.
* Gradient or optimisation instability.
* Evidence of conflicting training targets.
* Whether the model collapses towards one gain setting or an averaged response.

Do not assess success from aggregate training loss alone.

The model must be evaluated separately at every physical-gain position.

If the newly trained model cannot reproduce its training gain positions, investigate this before reducing the number of captures.

Do not proceed directly to the capture-count optimisation while the ten-capture result is unresolved.

---

## 9. Test the resulting `.nam` in a real NAM inference path

Export the newly trained NAM.

Reload it from disk using the standard NAM inference engine.

Use the actual exported file for every final comparison.

Do not evaluate only an in-memory training model or a special research renderer that accepts information unavailable to an ordinary NAM player.

For each physical Gain position:

```text
Held-out DI
     ↓
Real G1–G10 NAM capture
     ↓
Reference output
```

Compare with:

```text
Same held-out DI
     ↓
Predefined player input gain
     ↓
Newly trained standard NAM
     ↓
Reconstructed output
```

Do not switch NAM models during playback.

Do not apply Hybrid processing at runtime.

Do not optimise the virtual input gain against each evaluation target.

Do not independently level-normalise the reconstructed outputs before computing the primary raw-output comparison.

The only intended control changing between test positions is the ordinary input gain applied before the new NAM.

---

## 10. Required baselines

Compare the newly trained ten-capture model against:

**Baseline 1 - Original G5 NAM with ordinary input gain**

Use the original G5 capture and the same predefined input-gain control law.

This tests whether training across the gain sweep adds value beyond simply driving the existing G5 capture harder or softer.

**Baseline 2 - Original G5 NAM with its calibrated input-gain mapping**

Use the best previously established calibrated-G5 approach.

Label it clearly as a separately calibrated baseline.

This establishes whether the newly trained model improves on the shortcut identified in the previous research.

**Baseline 3 - Original real G1–G10 NAM captures**

These are the ground-truth references.

**Optional diagnostic - Direct Hybrid / Blended rendering**

Use only to identify errors introduced by the training-target generation method.

Do not present direct Hybrid playback as the finished solution.

---

## 11. Sound-quality evaluation

The objective is to reproduce the real amplifier's sound and playing response, not merely its output level.

For every physical Gain position, evaluate:

* Raw waveform error where the metric is reliable.
* Level-matched waveform error as a separate diagnostic.
* Output RMS and peak-level differences.
* Spectral differences.
* Distortion and harmonic character.
* Attack and transient response.
* Sustain and compression behaviour.
* Response to different DI input levels.
* Response to clean notes, chords and palm-muted passages.

Keep level and spectral measurements separate.

Do not define one composite score that hides poor behaviour in an individual gain region.

### Listening material

Generate representative WAV comparisons using:

1. The real physical-gain capture.
2. The newly trained ten-capture NAM.
3. The original G5 NAM with ordinary input gain.
4. The calibrated-G5 baseline.

Provide both actual-output-level and level-matched listening comparisons.

Use actual-output-level samples to evaluate whether the model reproduces the physical amplifier's gain and loudness progression.

Use level-matched samples to make differences in distortion character, tone and playing dynamics easier to hear.

Label the files without revealing the method in the blind-listening set.

Do not claim perceptual equivalence solely from ESR or spectral correlation.

Document that human listening is still required if it has not been performed.

---

## 12. Independent half-step validation

For the JCM800, evaluate the trained ten-capture model at the genuine half-step physical-gain positions.

These are particularly important because the model was trained using only integer positions.

Use the predefined physical-control-to-input-gain mapping to select the corresponding player input gain.

Do not optimise independently at each half-step.

Compare against the genuine real NAM captures at those half-step positions.

Report every position individually.

Exclude the known G8.5 timing defect from aggregate error statistics while retaining it as a separately documented diagnostic.

Success at G1–G10 is not sufficient if the model behaves poorly between them.

A continuous-gain model must have a smooth and useful response throughout the range, not merely at its training positions.

---

## 13. Input-level generalisation

A single standard NAM may respond differently when the player changes input gain versus when the guitar itself produces a hotter signal.

Explicitly test this limitation.

Use several fixed input-level conditions:

```text
Quiet guitar signal
Normal guitar signal
Hot guitar signal
```

For each condition, evaluate the complete physical-gain sweep using the same frozen player-gain mapping.

Check whether the trained model reproduces the appropriate real physical-gain capture for the corresponding DI level.

Do not recalibrate the mapping separately for each guitar-input level.

Identify cases where the model's response collapses or becomes substantially different from the reference.

Include tests where two different combinations of source level and player input gain produce similar model-input waveforms but correspond to different desired physical-gain responses.

Document the resulting representational limitation rather than treating it as a training bug.

---

## 14. Stop/go decision after the ten-capture experiment

Before reducing capture count, answer:

1. Did the training pipeline produce a valid standard `.nam` file?
2. Does that file load and render correctly through ordinary NAM inference?
3. Does it reproduce the original gain-capture responses at the defined input-gain settings?
4. Does it perform better than an unchanged G5 NAM using the same input-gain mapping?
5. Does it improve on the calibrated-G5 baseline?
6. Does its response generalise across held-out DI material and input levels?
7. Does it reproduce the real intermediate gain positions?
8. Does the response change smoothly as player input gain changes?
9. Is any remaining mismatch caused by training quality, synthetic-target generation, calibration, or the inherent limitations of a standard NAM?

If the all-ten model fails to demonstrate useful gain-range behaviour, do not automatically proceed to smaller capture subsets.

Investigate whether the approach is limited by:

* Incompatible multi-gain training targets.
* Insufficient network capacity.
* Unsuitable training-data scaling.
* Incorrect signal-path calibration.
* Hybrid / Blended target errors.
* Optimisation or training-budget problems.
* The lack of an independent physical Gain-control input.

Do not assume that adding more training time will overcome a structurally unidentifiable input/output relationship.

Report the blocker and the smallest justified next experiment.

---

## 15. Capture-count reduction experiment

Only after obtaining a useful ten-capture model, investigate the minimum number of physical-gain captures required.

Train separate new standard NAMs using controlled subsets.

Suggested configurations:

| Configuration  | Physical-gain captures |
| -------------- | ---------------------- |
| Full reference | G1–G10                 |
| Five captures  | G1, G3, G5, G7, G10    |
| Three captures | G1, G5, G10            |
| Two captures   | G1, G10                |

The subset positions are starting configurations, not assumptions that evenly spaced or endpoint captures are always optimal.

If the results justify it, compare alternative three-capture sets-for example, one targeting the amplifier's most nonlinear response region.

Do not optimise subset positions against the final held-out reference set and then claim that set remains independent.

### Strict comparison conditions

Keep constant wherever possible:

* Model architecture and parameter count.
* Training-data duration per physical-gain position.
* Optimiser and training procedure.
* Core training-budget policy.
* Training and validation DI splits.
* Control-mapping policy.
* Final evaluation DI material.
* Final evaluation gain positions.
* Export and NAM inference path.

Report both equal-compute and appropriately converged comparisons if the different dataset sizes make one training-budget policy misleading.

Use multiple training seeds where practical to distinguish a genuine capture-count effect from random optimisation variation.

### Synthetic training targets

For a three-capture model, Hybrid or Blended may generate intermediate training targets using G1, G5 and G10.

It must not use the omitted G2, G3, G4, G6, G7, G8 or G9 reference captures to generate those targets.

Those omitted captures are evaluation references.

The same principle applies to the two- and five-capture configurations.

A reduced-capture model must not receive information from omitted captures indirectly through:

* Calibration curves.
* Precomputed research search matrices.
* Previously generated ten-capture Hybrid targets.
* Fine-tuning from the ten-capture model.
* Model selection against omitted reference outputs.

Any experiment deliberately sharing such information must be labelled separately and must not be presented as a true reduced-capture result.

---

## 16. Compare every trained model against all ten original captures

For every newly trained model, evaluate every original physical-gain setting:

```text
G1
G2
G3
G4
G5
G6
G7
G8
G9
G10
```

This includes positions omitted from the smaller model's training set.

Also evaluate the JCM800 half-step positions.

For example:

| Physical Gain | Real reference | 10-capture trained NAM | 5-capture trained NAM | 3-capture trained NAM | 2-capture trained NAM | Original G5 baseline |
| ------------- | -------------- | ---------------------- | --------------------- | --------------------- | --------------------- | -------------------- |
| G1            | Real G1        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G2            | Real G2        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G3            | Real G3        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G4            | Real G4        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G5            | Real G5        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G6            | Real G6        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G7            | Real G7        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G8            | Real G8        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G9            | Real G9        | Test                   | Test                  | Test                  | Test                  | Test                 |
| G10           | Real G10       | Test                   | Test                  | Test                  | Test                  | Test                 |

Every method must use the same held-out source DI for a given comparison.

Mark which gain positions were included in each model's training set.

Do not combine in-sample and omitted-position results without identifying them.

Report mean and worst-case quality, but also retain the complete per-position results.

A three-capture model must not be described as equivalent to the ten-capture model merely because their average ESR is similar.

Investigate differences in individual gain regions, playing dynamics and output-level progression.

---

## 17. Determine whether three captures are sufficient

The main capture-reduction question is:

**Can a NAM trained using G1, G5 and G10 reproduce the complete gain sweep comparably to a NAM trained using all ten original captures?**

Assess this using:

* Error against the real physical-gain captures.
* Error at gain positions omitted from three-capture training.
* Error at the genuinely withheld half-step positions.
* Output-level progression.
* Spectral and distortion-character progression.
* Dynamic response across different input levels.
* Smoothness of response between gain positions.
* Blind listening material.
* Training stability across repeated runs.

If three captures are comparable to ten, determine whether five captures provide a meaningful improvement or whether two captures are also sufficient.

If the three-capture result is substantially worse, identify which physical-gain regions need additional training information.

Do not automatically conclude that more equally spaced captures are the solution.

The highest-value additional capture may be near a nonlinear response knee rather than in the largest numerical gap.

The objective is the minimum capture count that produces a newly trained NAM whose complete gain response remains close to the real amplifier-not the minimum capture count needed to reproduce a synthetic Hybrid sweep.

---

## 18. Generalise to the remaining amplifiers

After the JCM800 training method has demonstrated useful results, apply the same procedure to:

* Fender Super-Sonic.
* Fender 57 Custom Twin.
* Peavey 5150.

Start with all available original physical-gain captures for each amp.

Do not assume that a method that works on the JCM800 will necessarily work on the other amplifier types.

Keep the architecture, evaluation methodology and export requirements consistent.

For each amp, assess the ten-capture reference model before running the smaller capture-count configurations.

### Peavey 5150

The previous research identified limitations in interpreting waveform ESR on heavily saturated captures.

It also identified a suspected capture-timing problem around Gain 6 and separate gain-response anomalies.

Inspect these independently.

Do not draw strong conclusions from raw ESR alone.

Prioritise output-level accuracy, dynamic behaviour, representative listening comparisons and multiple complementary measurements.

Flag genuinely inconclusive results.

Do not silently exclude the 5150 simply because it presents a more difficult training problem.

---

## 19. Deliverables

Create a focused research document:

`docs/CONTINUOUS_GAIN_SINGLE_NAM_TRAINING.md`

Include:

1. Exact restatement of the objective.
2. Existing Hybrid and Blended implementation review.
3. Conventional NAM compatibility constraints.
4. Input-level identifiability analysis.
5. Multi-gain training-data construction.
6. Physical-gain-to-player-input-gain mapping policy.
7. Direct versus Hybrid- versus Blended-assisted training results.
8. Ten-capture JCM800 training results.
9. Final exported `.nam` compatibility verification.
10. Per-gain comparison against original reference captures.
11. Independent half-step validation.
12. Multiple-DI and multiple-input-level validation.
13. Original G5 and calibrated-G5 baseline comparisons.
14. Capture-count reduction results.
15. Three-versus-ten-capture comparison.
16. Cross-amp results, where the JCM800 phase justifies proceeding.
17. Listening-test audio.
18. Known limitations and recommended next implementation step.

Save the actual newly trained NAM models as test artifacts with clear names indicating their training configuration.

For example:

```text
JCM800_ContinuousGain_10Captures.nam
JCM800_ContinuousGain_5Captures.nam
JCM800_ContinuousGain_3Captures.nam
JCM800_ContinuousGain_2Captures.nam
```

Do not produce placeholder `.nam` files or rename existing anchor captures to imply that they are newly trained models.

Save machine-readable benchmark results and reproducible scripts.

Do not make changes to the existing production Continuous Gain profile builder or UI during this experiment unless a narrowly scoped correction is required to run a valid comparison.

---

## 20. Final reporting requirements

At the end, answer these questions explicitly.

**Feasibility**

* Can one newly trained standard NAM reproduce a useful physical gain sweep through ordinary player input gain alone?
* Does the model genuinely learn behaviour from multiple physical-gain captures?
* Is the gain response more faithful to the real amplifier than an unchanged G5 NAM?

**Training method**

* Does direct multi-gain training work?
* Does Hybrid-assisted training improve the result?
* Does Blended-assisted training improve the result?
* Are any improvements attributable to the newly trained model rather than a better control mapping?

**Capture count**

* How does the ten-capture model perform?
* How does the three-capture model compare?
* What is the lowest demonstrated capture count that reproduces the desired gain response?
* Where do reduced-capture models lose fidelity?

**Compatibility**

* Is the output a newly trained, standard `.nam` file?
* Does it work with ordinary NAM inference?
* Can the player vary its response using only the normal input-gain control?
* Does it require any external runtime processing beyond ordinary NAM playback?

**Sound**

* Does it reproduce the real amplifier's distortion character, frequency balance, output level and playing dynamics throughout the gain range?
* Are the differences audible in the generated comparison samples?
* Does it remain accurate with different guitars and input levels?

**Decision**

State whether the evidence supports continuing towards production of a single newly trained Continuous Gain NAM.

If the conventional NAM architecture cannot represent the required physical-gain behaviour reliably, explain the limitation and compare the remaining technically viable options without silently changing the product objective.

The final acceptance criterion is:

**One genuinely newly trained, standard NAM that can be loaded into an ordinary NAM player and produces an amplifier-like response across the physical Gain 1–10 range when the player changes only its normal input-gain control, validated against the real G1–G10 captures and genuinely unseen guitar audio.**

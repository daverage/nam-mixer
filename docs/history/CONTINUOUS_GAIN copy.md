# Research Branch: Continuous Gain Range NAM

## Purpose

Create a new experimental model type for NAM Mixer that attempts to represent the gain range of a single amplifier using multiple NAM captures taken at different physical gain settings.

This is not another two-amp mixing mode.

The existing NAM Mixer modes combine or interpolate between different amplifier captures. This research branch should instead investigate whether several captures of the same amplifier, taken across its gain control range, can be used to train a single standard NAM whose response to input gain better approximates the behaviour of the real amplifier gain control.

The target remains a normal NAM model that works in existing NAM players and supported hardware without requiring changes to the NAM runtime.

The project must remain evidence-led. Do not assume the technique works. The branch should make it easy to prove, disprove, measure, and understand the idea.

---

# Core Problem

A conventional NAM capture represents one fixed amplifier state.

For example:

* Amp gain: 5
* Bass: 5
* Mid: 5
* Treble: 5
* Master: fixed
* Signal chain: fixed

Changing the Input Gain control in a NAM player does not change the virtual amplifier gain control.

It only changes the level of the signal entering the captured nonlinear model.

Therefore:

```text
Real amplifier:

audio + Gain knob position -> amplifier -> output
```

while a conventional NAM behaves approximately as:

```text
audio * input level -> fixed captured amplifier -> output
```

The research question is whether a model trained from multiple gain settings can produce a useful approximation:

```text
low NAM input level
        ↓
behaviour closer to low physical amp gain

medium NAM input level
        ↓
behaviour closer to medium physical amp gain

high NAM input level
        ↓
behaviour closer to high physical amp gain
```

This would still not make the NAM player Input Gain control equivalent to a true parameterised amp control.

The objective is only to determine whether it can behave more realistically across a useful input-gain range than a conventional single-setting capture.

---

# Fundamental Limitation

The research must explicitly recognise this limitation.

A standard NAM receives audio only.

It has no separate control input representing the physical amplifier Gain knob.

Therefore it cannot distinguish between:

```text
quiet guitar + increased NAM input gain
```

and:

```text
hotter guitar signal + unchanged NAM input gain
```

It also cannot inherently distinguish player dynamics from intentional virtual gain-control movement.

This ambiguity cannot be solved completely while producing a standard audio-only NAM.

Any solution in this branch is therefore an approximation.

Do not describe the output as:

* a true parametric NAM
* a real amp gain control
* recreation of the amplifier's actual gain knob
* exact interpolation of amplifier controls

Suitable terminology includes:

* Continuous Gain Model
* Gain Range Model
* Gain Range Capture
* Input-Responsive Gain Model
* Gain Sweep Model

Working name:

# Continuous Gain Model

---

# User Concept

Instead of selecting two different amps:

```text
Amp A
Amp B
```

the user creates a capture set from one amplifier:

```text
Same amp
Same cabinet
Same mic
Same EQ
Same master
Same signal chain

Gain 1
Gain 3
Gain 5
Gain 7
Gain 10
```

Only the amplifier gain control should change.

The system then analyses and reconstructs the progression between these known states.

---

# Minimum Capture Requirements

The system should support arbitrary capture positions rather than enforcing fixed values.

Minimum:

```text
3 captures
```

Recommended:

```text
5 captures
```

Example:

```text
0.0    Gain minimum
0.25   Gain 3
0.50   Gain 5
0.75   Gain 7
1.0    Gain maximum
```

Users should also be able to provide irregular positions:

```text
0
2
3.5
6
7
10
```

because many amplifier controls are strongly nonlinear and the most useful samples may cluster around breakup transitions.

Internally, control positions should be normalised to:

```text
0.0 -> 1.0
```

---

# Important Capture Rule

All captures in a gain set should represent the same amplifier configuration.

The UI should state clearly:

> Keep every amplifier, cabinet, microphone and signal-chain setting identical. Change only the amplifier gain control.

Potential variables that must remain fixed include:

* amplifier model
* channel
* EQ
* tone stack
* presence
* resonance
* master volume where practical
* cabinet
* microphone
* microphone position
* load box
* IR
* pedals
* interface gain
* output level methodology

The application should warn that changing other parameters means the resulting model represents a multi-parameter morph rather than a controlled gain sweep.

---

# New Top-Level Model Type

Do not implement this as another existing mixer mode.

Suggested structure:

```text
NAM Mixer

Model Type
├── Amp Mixer
│   ├── existing modes
│   ├── existing blend modes
│   └── existing hybrid modes
│
└── Continuous Gain Model
    ├── Capture Set
    ├── Gain Analysis
    ├── Interpolation
    ├── Input Mapping
    ├── Training
    └── Validation
```

Existing code may be reused internally, but the user-facing concept and data model should remain separate.

---

# Research Architecture

The research should separate three concepts.

## 1. Physical Gain Position

Represents the real amplifier knob position.

Example:

```python
control_position = 0.5
```

This represents a real setting such as Gain 5.

This is metadata associated with each source capture.

---

## 2. Intermediate Amplifier State

Represents the expected amplifier behaviour between known captures.

Example:

```text
known:

G3 ---------------- G5

requested:

G4
```

The system should construct or estimate the state between the neighbouring captures.

Initially investigate several approaches rather than assuming one is correct.

---

## 3. NAM Player Input Level

Represents the signal level entering the final ordinary NAM.

This eventually acts as a proxy for the intended gain position.

Example:

```text
-24 dB -> low gain region
-15 dB -> medium gain region
 -6 dB -> high gain region
```

These concepts must remain separate internally.

Do not directly equate physical gain position and signal amplitude throughout the implementation.

Only map them together during final compatibility-model generation.

---

# Intermediate-State Research

Do not assume direct output crossfade is sufficient.

Investigate at least the following approaches.

## Approach A: Output Interpolation

Neighbouring NAM outputs are blended:

```text
output =
    NAM_A(input) * (1 - blend)
  + NAM_B(input) * blend
```

This is the simplest baseline.

It should be implemented primarily as a control comparison.

---

## Approach B: Existing NAM Mixer Hybrid Techniques

Evaluate whether existing decomposition, dynamic hybrid, character interpolation or related NAM Mixer techniques create a better intermediate state.

Reuse existing code where sensible.

The goal is not code reuse for its own sake.

The goal is to determine whether existing NAM Mixer techniques preserve:

* gain structure
* harmonic progression
* compression
* transient response
* frequency response
* saturation character

better than simple output blending.

---

## Approach C: Synthetic Intermediate NAMs

Investigate generating intermediate NAMs between real capture points.

Example:

```text
Real:
G1      G3      G5      G7      G10

Synthetic:
    G2      G4      G6      G8      G9
```

This could produce a more densely sampled gain trajectory before final model training.

Do not assume synthetic intermediate models improve results.

Measure them against real withheld captures.

When generating synthetic intermediate NAMs, check whether the interpolation or generation method introduces temporal behaviour beyond that present in the source models.

If a synthetic-model technique introduces:

* additional convolution
* smoothing
* delayed state
* long envelope behaviour
* other temporal dependencies

evaluate it under the existing receptive-field policy before using the resulting material for student training.

A synthetic intermediate model is not automatically valid merely because its static spectral or ESR measurements look good.

---

# Piecewise Interpolation

Interpolation should normally occur only between neighbouring known states.

Example capture points:

```text
0.00 = Gain 1
0.25 = Gain 3
0.50 = Gain 5
0.75 = Gain 7
1.00 = Gain 10
```

Requested:

```text
0.42
```

Use:

```text
Gain 3
Gain 5
```

Calculate local blend:

```python
local_blend = (0.42 - 0.25) / (0.50 - 0.25)
```

Do not interpolate Gain 1 directly against Gain 10 unless testing it explicitly as a baseline.

---

# Input-to-Gain Mapping

The final standard NAM has no real gain parameter.

A mapping therefore has to be created between input level and the reconstructed gain range.

Start with a configurable function:

```python
gain_position = clamp(
    (input_db - low_threshold) /
    (high_threshold - low_threshold),
    0.0,
    1.0
)
```

Example:

```text
Low threshold:  -24 dB
High threshold:  -6 dB
```

Possible interpretation:

```text
<= -24 dB -> minimum captured amp gain

-15 dB -> approximately middle gain region

>= -6 dB -> maximum captured amp gain
```

This is only a starting hypothesis.

---

# Envelope Behaviour

Do not use raw instantaneous sample amplitude as the primary gain-position controller without testing alternatives.

That is likely to cause the effective amp setting to move with every waveform peak and pick transient.

Research:

* RMS
* peak envelope
* moving average
* attack/release envelope
* logarithmic envelope
* percentile-based level estimation
* slower control envelopes

Candidate starting ranges:

```text
Attack: 20-100 ms
Release: 100-500 ms
```

Also test much slower behaviour.

The objective is to prevent rapid playing transients from effectively moving the simulated gain control while still allowing NAM-player input gain changes to influence the model.

### Causality Requirement

Any envelope used for gain-position estimation must be causal.

The implementation must not use future samples to determine the gain position of the current sample.

Avoid techniques equivalent to centred smoothing such as:

```python
np.convolve(..., mode="same")
```

when they implicitly introduce look-ahead.

This follows the same constraint already applied to the repository's existing envelope/crossover processing.

If existing `envelope.py` primitives provide the required attack/release behaviour, reuse them rather than implementing a separate envelope follower.

Any new envelope implementation must be tested explicitly for causality.

A simple test should verify that changing audio after sample `N` cannot affect the control/envelope value at or before sample `N`.

This is particularly important because the resulting control trajectory may be baked into generated training targets.

---

# Receptive Field and Temporal-State Policy

This research must respect the repository's existing receptive-field policy.

Before introducing any processing with temporal extent, inspect and reuse the rules and utilities in:

```text
hybrid/receptive_field.py
```

and the corresponding CORE versus advisory RF policy documented for the project.

Potential sources of temporal extent include:

* envelope attack/release
* RMS windows
* moving averages
* smoothing filters
* analysis windows
* synthetic intermediate processing
* teacher-side state
* convolution
* latency compensation

For example, a 500 ms release at 48 kHz represents approximately:

```text
24,000 samples
```

of control-state history.

That does not automatically mean the NAM model itself requires a 24,000-sample receptive field, because the control process and the audio model are not necessarily equivalent.

However, it does mean the interaction must be understood rather than ignored.

For every teacher-side process with temporal state, document:

```text
process
state/history length
whether it affects generated audio targets
whether the final student must reproduce that state
whether existing RF gates consider it CORE or advisory
```

Do not bypass RF checks simply because the feature is experimental.

Do not expand CORE receptive-field requirements unless the generated target demonstrably requires that additional temporal dependency.

The research should explicitly distinguish:

```text
teacher control history
```

from:

```text
audio-model receptive field required by the student
```

These may not be the same.

---

# Important Dynamic Failure Mode

Explicitly test this scenario:

```text
NAM player gain unchanged.

Player picks softly:
model appears to behave like Gain 3.

Player picks hard:
model appears to behave like Gain 8.
```

A real amplifier whose Gain knob is physically at 5 does not change its circuit control position when the guitarist picks harder.

Its nonlinear response changes because the input changes, but the gain control itself remains fixed.

This is likely the largest conceptual weakness of the compatibility approach.

Measure how severe this artefact is.

Do not hide it.

---

# Potential Better Mapping

Investigate whether the final student can learn a behaviour where input amplitude influences both:

1. ordinary nonlinear amplifier response
2. gradual movement through the gain-state trajectory

without the second behaviour dominating normal playing dynamics.

This may require a deliberately compressed gain-position mapping.

For example:

```text
normal guitar dynamic range:
mainly operates within one local gain region

large NAM input gain changes:
move between gain regions
```

This may work better than mapping the entire instantaneous guitar dynamic range directly across Gain 1 to Gain 10.

This should be tested rather than assumed.

---

# Capture Normalisation

Gain captures will probably have different output levels.

Do not automatically normalise them without understanding the consequences.

Investigate at least:

### Raw behaviour

Preserve the actual output-level differences produced by the amplifier.

### Loudness-normalised behaviour

Match output levels before interpolation.

### Partially normalised behaviour

Preserve some of the physical level progression while preventing the final model from producing impractical volume differences.

Output level and distortion character should be treated separately where possible.

A gain sweep may otherwise become primarily a loudness sweep.

### Candidate Partial-Normalisation Method

Do not leave partial normalisation as an undefined concept.

Use the existing active-material level-matching logic in `hybrid/fixed_blend.py`, particularly `compute_active_trim`, as the first reference implementation for measuring output-level differences without allowing silence or low-information regions to dominate the calculation.

A candidate approach is:

1. Identify active material using the existing active-region methodology.
2. Measure RMS over the same active region for every gain capture.
3. Choose one capture or a robust aggregate as the reference level.
4. Calculate the full trim required to RMS-match each capture.
5. Apply only a configurable fraction of that trim.

Conceptually:

```python
full_trim_db = reference_rms_db - capture_rms_db
applied_trim_db = full_trim_db * normalisation_amount
```

Where:

```text
normalisation_amount = 0.0
```

preserves the physical output-level progression,

```text
normalisation_amount = 1.0
```

fully level-matches the captures,

and intermediate values preserve part of the real amplifier level progression.

Initially test at least:

```text
0.0
0.5
1.0
```

Do not assume 0.5 is perceptually or technically optimal.

The important distinction is that this processing should prevent the interpolation system from interpreting simple output-level differences as amplifier-character differences, while still allowing real gain-dependent level behaviour to be retained where useful.

Reuse existing level-analysis primitives where possible rather than introducing another unrelated RMS/activity implementation.

### Relationship to Existing NAM Input Calibration

Do not use output normalisation and NAM input-level calibration interchangeably.

The existing `hybrid/calibration.py` logic addresses differences in NAM input calibration, including metadata such as `input_level_dbu`, so that multiple NAM models can be driven consistently.

That solves a different problem from the output normalisation described here.

For Continuous Gain Model research there are therefore two separate calibration stages:

```text
NAM input calibration
        ↓
ensures each source NAM receives an equivalent physical input level

Output/gain-range normalisation
        ↓
controls how much of the amp's output-level progression is preserved
during interpolation
```

Existing calibration code should be reused where applicable for the first stage.

Do not modify, replace, or silently duplicate the existing calibration behaviour just because all captures represent the same amplifier.

A gain sweep may still contain NAM files with differing capture metadata or calibration assumptions.

---

# Capture Validation

Add analysis to detect obviously inconsistent capture sets.

Possible checks:

* sample rate
* NAM architecture compatibility
* capture metadata
* extreme frequency-response changes
* unexpected polarity differences
* unusual level jumps
* inconsistent latency
* suspicious spectral discontinuities
* non-monotonic distortion changes
* capture identity where metadata exists

Warnings should be advisory.

Do not reject captures simply because an amplifier behaves unusually.

---

# Gain Curve Analysis

The system should not assume that amplifier gain controls are linear.

Example real behaviour may resemble:

```text
1     2     3     4     5     6     7     8     9     10
|---clean---|--breakup--|----saturation plateau---------|
```

Analyse the differences between capture points.

Possible measurements:

* harmonic energy
* ESR between neighbouring models
* compression ratio
* crest factor
* spectral centroid
* low-frequency response
* high-frequency roll-off
* transient response
* output level
* saturation behaviour

Use this analysis to explore whether the interpolation curve should be:

* linear
* logarithmic
* spline-based
* piecewise
* learned from measurements

Avoid inventing precision that the source captures do not support.

---

# Critical Validation Strategy

The strongest validation method should be leave-one-out testing.

Example source captures:

```text
Gain 1
Gain 3
Gain 5
Gain 7
Gain 10
```

## Test 1

Hide Gain 5.

Build the gain trajectory from:

```text
Gain 1
Gain 3
Gain 7
Gain 10
```

Predict Gain 5.

Compare the synthetic Gain 5 behaviour against the real Gain 5 capture.

---

## Test 2

Hide Gain 3.

Predict Gain 3.

Compare against real Gain 3.

---

## Test 3

Hide Gain 7.

Predict Gain 7.

Compare against real Gain 7.

---

# Validation Metrics

Use objective measurements where useful, but do not treat a single ESR value as proof of perceptual correctness.

Evaluate:

* ESR
* waveform error
* spectral difference
* harmonic structure
* transient response
* compression behaviour
* output level
* response to changing input amplitude

Also generate audio comparisons for listening tests.

Test material should include:

* isolated notes
* low notes
* high notes
* open chords
* dense chords
* palm-muted playing
* soft picking
* hard picking
* sustained notes
* guitar-volume changes
* different pickup outputs where available

---

# Most Important Comparison

Compare the new model against an ordinary midpoint NAM.

Example:

```text
A:
Real amp captured at Gain 5

B:
Continuous Gain Model operating around its intended Gain 5 region
```

Then change NAM-player input gain.

Determine whether the Continuous Gain Model follows the behaviour of real physical gain changes more closely than the conventional Gain-5 snapshot.

That is the actual product hypothesis.

The test is not merely:

> Does the new model sound good?

The test is:

> Is input-gain behaviour measurably or perceptually closer to moving the real amplifier gain control than an ordinary fixed capture?

---

# Recommended Experimental Capture Set

Start with one amplifier only.

Do not generalise prematurely.

Recommended first dataset:

```text
Gain 1
Gain 3
Gain 5
Gain 7
Gain 10
```

Also capture additional validation-only positions:

```text
Gain 2
Gain 4
Gain 6
Gain 8
Gain 9
```

Do not expose these validation captures to the interpolation system.

This gives real ground truth at the intermediate points.

Ideal dataset:

```text
Training:
1, 3, 5, 7, 10

Hidden validation:
2, 4, 6, 8, 9
```

This is much stronger than validating only against training captures.

---

# Research Questions

The branch should attempt to answer these questions.

## Q1

Can neighbouring NAM captures be interpolated closely enough to approximate a real unseen physical gain position?

## Q2

How many source captures are required?

Compare:

```text
2
3
5
7+
```

captures.

## Q3

Does capture spacing matter more than capture count?

For example, captures around the clean-to-breakup transition may be more valuable than evenly spaced positions.

## Q4

Does simple interpolation work, or do NAM Mixer decomposition/hybrid techniques improve reconstruction?

## Q5

Can a final audio-only NAM learn the generated gain trajectory without unacceptable artefacts?

## Q6

Can NAM input gain act as a useful proxy for physical gain position?

## Q7

How much does playing dynamics incorrectly move the apparent amplifier gain state?

## Q8

Can smoothing or range compression reduce that problem?

## Q9

Does the final model behave better than simply capturing the amplifier at a useful midpoint and changing input gain conventionally?

## Q10

Does the technique generalise across:

* clean amps
* edge-of-breakup amps
* high-gain amps
* preamp-heavy amps
* master-volume amps

Only investigate generalisation after the first controlled test succeeds.

---

# UI Prototype

Create a separate research interface.

Example:

```text
Continuous Gain Model
────────────────────────────────────────

Capture Set

Gain Position          NAM Capture
0.0                    amp_gain_1.nam
0.25                   amp_gain_3.nam
0.50                   amp_gain_5.nam
0.75                   amp_gain_7.nam
1.0                    amp_gain_10.nam

[ Add Capture ]

────────────────────────────────────────

Interpolation

Method:
○ Output Blend
○ Dynamic Hybrid
○ Character / Decomposition
○ Experimental

Curve:
○ Linear
○ Automatic
○ Custom

────────────────────────────────────────

Input Mapping

Low Gain Threshold:     -24 dB
High Gain Threshold:     -6 dB

Envelope:
Attack                   50 ms
Release                 250 ms

[ Analyse ]

────────────────────────────────────────

Validation

☑ Leave-one-out validation
☑ Generate comparison audio
☑ Show objective metrics

[ Run Research Build ]
```

Do not spend significant time polishing this UI until the modelling approach demonstrates value.

---

# Internal Data Model

Suggested structure:

```python
GainCaptureSet:
    id
    name
    captures[]

GainCapture:
    model_path
    control_position
    metadata
    analysis

GainRangeConfig:
    interpolation_method
    interpolation_curve
    low_input_threshold
    high_input_threshold
    attack_ms
    release_ms
    output_normalisation
```

Keep the research model independent of existing two-source mixer assumptions.

Avoid structures such as:

```python
amp_a
amp_b
blend
```

for this feature.

Use arbitrary ordered capture collections.

---

# Architecture Principle

Reuse low-level NAM Mixer capabilities where appropriate:

* model loading
* inference
* analysis
* decomposition
* resampling
* training
* dataset generation
* NAM A2 output
* evaluation

But do not force the new feature through abstractions designed specifically around two models.

If necessary, introduce reusable primitives underneath both systems.

Example:

```text
NAM inference
Model analysis
Model interpolation
Teacher generation
Dataset synthesis
Student training
Validation
```

Then:

```text
Amp Mixer
```

and:

```text
Continuous Gain Model
```

can compose those primitives differently.

Before creating new infrastructure, inspect and preferentially reuse the following existing mechanisms where appropriate:

```text
hybrid/calibration.py
```

For NAM input-level calibration and reconciliation of source capture input assumptions.

```text
hybrid/fixed_blend.py
```

Particularly existing active-material level analysis such as `compute_active_trim`, as a candidate primitive for gain-capture output normalisation.

```text
hybrid/envelope.py
```

For causal envelope behaviour and existing attack/release conventions.

```text
hybrid/receptive_field.py
```

For existing receptive-field measurement, policy, gating and CORE/advisory distinctions.

Do not reuse these blindly.

For each one, determine:

```text
What existing problem does this code solve?

Is that problem actually equivalent to the Continuous Gain Model requirement?

Can the primitive be reused directly?

Should a lower-level reusable primitive be extracted?

Would reuse accidentally couple the new research model to assumptions from the two-amp mixer?
```

Prefer extraction of genuinely shared primitives over duplicating mature logic.

---

# Branch

Use:

```text
research/continuous-gain-model
```

Do not create alternative branch names.

Keep all experimental work isolated from the stable mixer implementation until the research passes the defined validation criteria.

---

# Development Order

## Phase 1: Ground Truth Harness

Before implementing the full model:

* load multiple captures
* associate each with a control position
* run identical input material through each
* calculate differences
* produce comparison plots/data
* support hidden validation captures

Before implementing new analysis logic, inspect existing repository implementations for:

```text
input calibration
active-material detection
level matching
envelope following
receptive-field analysis
```

The Phase 1 harness should reuse these primitives where they are semantically appropriate.

It should also record, for every source capture:

```text
control position
input calibration metadata
measured active RMS
output trim used
normalisation mode
RF-related metadata
```

This ensures later experiments can be reproduced and compared rather than depending on hidden preprocessing decisions.

No final NAM training yet.

---

## Phase 2: Intermediate Reconstruction

Implement:

* neighbouring capture lookup
* simple interpolation
* interpolation curves
* withheld-capture comparison

Determine whether intermediate reconstruction itself works.

Stop here if it clearly fails.

---

## Phase 3: Improved Reconstruction

Test:

* NAM Mixer hybrid methods
* character interpolation
* decomposition-based methods
* synthetic intermediate NAMs

Compare all methods objectively against hidden real captures.

Do not retain complexity that does not improve validation.

---

## Phase 4: Continuous Teacher

Construct:

```text
audio + requested gain position
        ↓
continuous teacher
        ↓
audio
```

At this stage the teacher may have an explicit gain-position input.

The teacher may have an explicit gain-position input and may use causal control-state processing.

At this stage it does not need to be compatible with standard NAM.

However:

* all control processing must remain causal
* temporal state must be documented
* interaction with existing receptive-field policy must be assessed
* teacher-only state must not silently become an impossible requirement for the final student

The purpose of the teacher is to represent the best supported gain trajectory available from the source captures, not to create an arbitrarily powerful system that an ordinary NAM could never approximate.

---

## Phase 5: Compatibility Mapping

Map input level onto teacher gain position.

Experiment with:

* thresholds
* RMS
* envelope smoothing
* mapping curves
* restricted dynamic range

Generate training data from this behaviour.

---

## Phase 6: Student Training

Train an ordinary NAM:

```text
audio -> final output
```

The resulting model must work in an unmodified standard NAM runtime.

---

## Phase 7: Final Validation

Compare:

```text
Real amplifier sweep
Continuous teacher
Final Continuous Gain NAM
Ordinary fixed-gain NAM
```

Determine exactly what was lost when compressing the explicitly controlled teacher into the standard audio-only NAM.

---

# Stop Conditions

This is important.

Do not keep adding complexity merely to rescue the concept.

Stop or reconsider the approach if:

1. Interpolated gain positions fail badly against withheld real captures.

2. The final standard NAM cannot retain the teacher's behaviour.

3. Playing dynamics cause severe unintended movement through gain states.

4. Results are not meaningfully better than a normal midpoint NAM.

5. The technique only works after requiring input calibration so strict that normal users cannot use it reliably.

6. The final model behaves unpredictably across different guitars or pickup outputs.

A negative result is useful research.

Document it.

---

# Success Criteria

The research is promising if all of the following are reasonably demonstrated:

1. Intermediate physical gain positions can be reconstructed from neighbouring captures with useful accuracy.

2. The final standard NAM preserves a meaningful part of that trajectory.

3. Increasing NAM-player input gain produces behaviour closer to increasing the physical amplifier gain control than happens with a conventional snapshot.

4. Normal playing dynamics do not cause unacceptable gain-state pumping.

5. Behaviour remains usable across realistic guitar output levels.

6. The model does not require a modified NAM runtime.

The result does not need to perfectly recreate a physical gain control.

It needs to demonstrate a clear improvement over the conventional fixed-capture behaviour.

---

# Research Output

At completion, produce a report containing:

## Source dataset

Which amp, capture positions and signal chain were used.

## Methods tested

All interpolation and mapping approaches.

## Validation results

Including withheld real captures.

## Failure cases

Especially player-dynamics and input-level sensitivity.

## Best-performing method

Based on measured evidence, not subjective preference alone.

## Comparison with conventional NAM

Determine whether the new approach actually improves gain-control behaviour.

## Recommendation

One of:

```text
Promising enough to develop further

Useful only under restricted conditions

Interesting research but unsuitable as a user feature

No meaningful advantage over conventional NAM
```

Do not bias the research toward the first outcome.

---

# Coding-Agent Instruction

Implement this as an isolated research branch.

First inspect the existing NAM Mixer architecture and identify reusable inference, model-analysis, teacher-generation, training and validation code.

Do not immediately modify the existing mixer modes.

Do not assume existing two-model abstractions are appropriate.

Build the smallest research harness required to answer the central question:

> Can several captures of the same amplifier at different physical gain settings be used to create a standard NAM whose response to the NAM player's input gain control more closely follows the real amplifier's gain-control behaviour than a conventional single-setting NAM?

Prioritise measurable experiments over UI work.

For every modelling shortcut or approximation, document:

* what it assumes
* why it was chosen
* how it can fail
* how it will be tested

Do not present plausible-sounding interpolation as successful modelling without validation against withheld real amplifier captures.

Prefer simple methods that demonstrate measurable improvement over complex methods that merely produce interesting results.

Keep all experimental work reversible and isolated from the stable NAM Mixer workflow.

Before writing implementation code, inspect:

```text
hybrid/calibration.py
hybrid/fixed_blend.py
hybrid/envelope.py
hybrid/receptive_field.py
CLAUDE.md
```

Identify the existing project rules around:

* NAM input-level calibration
* active-material level matching
* causal envelope processing
* receptive-field CORE/advisory policy

Treat these as existing architectural constraints unless the research produces evidence that they need to change.

Do not introduce:

* a second calibration system
* a second active-region detector
* a non-causal envelope follower
* an independent receptive-field policy

without a documented technical reason.

For output normalisation, begin by evaluating whether the existing active-material trim calculation can be reused or factored into a shared primitive.

For envelope-controlled gain mapping, explicitly verify causality.

For any teacher-side operation with meaningful temporal state, assess whether it affects the receptive-field requirements of the final student.

Document the distinction between:

```text
teacher state required to construct the target
```

and:

```text
temporal context the trained NAM must reproduce
```

Do not automatically equate the two.

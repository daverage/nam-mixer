# Continuous Gain - Fixed Virtual-Gain Playability Test

## Mission

We are developing a workflow that turns multiple fixed-gain captures of ONE
physical amplifier into ONE standard, portable .nam.

The guitarist loads that NAM into an ordinary NAM player and uses the player's
existing Input gain control as a virtual amplifier-gain control.

The musical input is a separate variable.

At a selected virtual-gain setting, the guitarist should be able to play
softly, play hard, change pickup output or use the guitar's volume knob,
while retaining a convincing approximation of the physical amplifier at
that selected gain setting.

The aim is not to reproduce exact physical knob markings. It is to provide
one useful, continuous progression through the amplifier's characteristic
clean, breakup and saturated sounds.

This task is a focused diagnostic of the EXISTING implementation and models.

DO NOT:
- Train or retrain models.
- Add Vibrolux G8.
- Change training anchors or capture sets.
- Change the existing blend algorithm.
- Introduce a conditioned NAM, custom player or external runtime DSP.
- Start another broad research phase.
- Modify the frozen Phase 4E results.

We need to establish whether the current models behave correctly for the
intended playing experience before deciding what to build next.


## 1. Review the relevant implementation

Work on research/continuous-gain-model.

Review the current versions of:

- hybrid/multi_blend.py
- hybrid/envelope.py
- scripts/cg_build.py
- scripts/p4e_build.py
- scripts/single_nam_train.py
- scripts/p4_model_sweep.py
- The Phase 4E model-rendering and evaluation code.
- docs/CONTINUOUS_GAIN_PHASE4E_RESULTS.md
- docs/phase4e/manifest_frozen.json

Locate and reuse existing rendering, DI loading, capture-reference,
alignment, feature measurement and reporting utilities.

The training target currently uses a causal input envelope to determine
the contribution of adjacent physical-gain captures.

Verify this from the actual code.

Document precisely:

1. Which signal is used to calculate the envelope.
2. How changing player Input gain affects the envelope.
3. How changing musical input level affects the envelope while the
   player's Input gain remains fixed.
4. How blend weights are applied to the individual capture renders.
5. Whether the current evaluation reproduces the ordinary NAM player's
   input-gain signal path.

Do not rely on the comments alone. Verify the implementation.

Do not rewrite the training pipeline as part of this task.


## 2. Define the two independent user controls

Use these definitions consistently throughout the diagnostic:

VIRTUAL GAIN:
The deliberate setting of the ordinary NAM player's Input gain control.
This is the control the guitarist uses to select a cleaner, breaking-up
or saturated amplifier sound.

MUSICAL INPUT:
The actual guitar/DI waveform presented to the player before its
Input gain control.

Musical input varies with picking intensity, guitar-volume setting,
pickup output and other aspects of the performance.

The conceptual playback signal path is:

    musical DI
        |
        v
    player Input gain [virtual amplifier-gain setting]
        |
        v
    ONE standard NAM
        |
        v
    player Output gain
        |
        v
    audio output

The effective waveform reaching NAM inference is the musical DI multiplied
by the player's input-gain factor.

The NAM does not receive the two variables as independent parameters.

That is a known constraint of a standard NAM, not a reason to abandon
the project.

Our practical objective is to determine how convincingly the current
trained model handles the intended combinations of these variables.


## 3. Use existing models and physical references

Primary models:

- JCM800 Phase 4E Configuration B, seeds 0 and 1.
- Vibrolux Phase 4E Configuration B, seeds 0 and 1.

Comparisons:

- Phase 4E Configuration A, seeds 0 and 1.
- Existing v3 C3 model for each amplifier.

Use existing standard .nam exports.

Use the validated physical-gain captures as reference amplifiers,
with Phase 4A alignment corrections where applicable.

Do not generate new source captures or substitute synthetic interpolated
references for actual captured positions.

Use the models' intended virtual-gain mappings.

Keep the same global output-scaling convention used in Phase 4E.
Do not fit a separate output correction for each gain position or DI level.

Record the exact model, capture and source-DI identities used.


## 4. Primary test: hold virtual gain fixed and vary musical input

This is the MOST IMPORTANT part of the task.

At each selected virtual-gain setting, keep the player's Input gain fixed.

Then vary ONLY the musical DI input.

Compare the resulting NAM output against the SAME physical amplifier
capture, with the physical gain knob held at the corresponding setting.

For example:

    Vibrolux virtual G3 selected
        |
        +-- quiet musical DI   -> ONE NAM at fixed Input gain
        +-- normal musical DI  -> ONE NAM at fixed Input gain
        +-- loud musical DI    -> ONE NAM at fixed Input gain

Reference:

    Physical Vibrolux G3 capture
        |
        +-- quiet musical DI   -> physical G3 response
        +-- normal musical DI  -> physical G3 response
        +-- loud musical DI    -> physical G3 response

The physical capture must NOT change between quiet, normal and loud tests.

The virtual-gain setting must NOT change between quiet, normal and loud tests.

The same DI waveform and level must be supplied to the model and
the corresponding physical reference, accounting only for the intended
player Input-gain mapping and the established reference-capture calibration.

Do not compensate for a change in musical input by adjusting the player's
Input gain.

Do not choose a new playback mapping for each DI level.

Do not compare a quiet performance against physical G2 and a loud
performance against physical G4 when the selected virtual setting is G3.

That would measure the wrong behaviour.


## 5. Choose representative gain settings

Use the exact existing training-anchor positions first.

JCM800 B:

    G1  -> -22.0 dB
    G2  -> -10.5 dB
    G4  ->  -2.6 dB
    G10 -> +14.0 dB

Vibrolux B:

    G1  -> -22.0 dB
    G2  -> -17.4 dB
    G3  ->  -8.4 dB
    G4  ->  -3.1 dB
    G7  ->  +5.3 dB
    G10 -> +14.0 dB

Include additional omitted physical positions only where the existing
intended mapping and real capture references are available.

For the JCM800, include representative genuine half-step references
where useful.

Do not rely only on the endpoints.

The diagnostic must include cleaner sounds, the clean-to-breakup
transition, medium gain and saturated sounds.

For this diagnostic, use the full intended mapping through NAMCore,
including -22 dB where necessary. Record separately that the official
plugin's nominal Input range may restrict access to that endpoint.

Do not make the plugin's -20 dB minimum a confounding variable in the
initial training-method diagnosis.


## 6. Musical-input test matrix

Use the existing held-out DIs:

- moderate_brit
- clean_mayer
- bass_rollin

Use the existing musical-input level offsets:

    -12 dB
     -6 dB
      0 dB
     +6 dB

At each virtual-gain setting:

1. Select the corresponding physical reference capture.
2. Fix the player Input gain.
3. Render each held-out DI at each musical-input offset.
4. Render the same DI and offset through the physical reference.
5. Compare the resulting outputs.

Also examine within-recording changes in intensity if the existing DIs
contain sufficiently distinct quiet and loud passages.

A global DI-level offset is useful but is not a complete substitute
for comparing actual soft and hard playing.

If the source material cannot support a reliable within-recording
soft/hard comparison, state that limitation instead of inventing
a performance-dynamics result.


## 7. Measure whether the selected amp character remains stable

At each FIXED virtual-gain setting, measure how the NAM's response
changes as musical input increases.

Compare those changes with the response of the corresponding fixed-gain
physical capture.

Report:

- Signed native output-level error.
- Output level as a function of musical-input level.
- Input/output compression slope.
- Spectral balance and EQ-band errors.
- High-frequency energy and spectral tilt.
- Saturation and harmonic behaviour where the source signal supports
  reliable measurement.
- Crest factor and transient behaviour.
- Dynamic-range response.
- Level-matched waveform error as a secondary diagnostic.

For harmonic and THD measurements requiring a controlled test signal,
reuse the existing sine-probe measurement path. Do not infer accurate THD
solely from arbitrary musical DI recordings.

Show the response across all four musical-input offsets.

Do not report only the mean error.

Identify the gain setting and musical-input level responsible for each
important mismatch.

Where possible, distinguish:

A. A fixed tonal offset that is present at every input level.

B. A dynamic-response error that grows as the guitarist plays harder
   or softer.

C. A change in character suggesting that the model is moving towards
   a neighbouring physical-gain capture despite the virtual-gain setting
   remaining fixed.

Avoid treating every ordinary increase in saturation with harder
playing as unwanted gain-position drift.

A real fixed-gain amplifier also changes saturation and compression
with input intensity.

The relevant failure is a difference from how the REAL amplifier behaves
at the SAME fixed physical-gain setting.


## 8. Diagnose possible gain-position drift

Investigate the hypothesis that the existing envelope-driven training
target changes its physical-capture blend weights when musical input
changes, even though the intended virtual-gain setting is fixed.

Use the existing hybrid/multi_blend.py and hybrid/envelope.py functions.

For the same test DI, virtual-gain setting and musical-input offsets:

1. Calculate the envelope used by the training-target construction.
2. Calculate the corresponding capture blend weights.
3. Summarise how the weights change with musical-input level.
4. Identify whether neighbouring physical-gain captures contribute
   more heavily as musical input increases or decreases.

Use active musical segments rather than letting silence or long tails
dominate the weight statistics.

Generate representative plots showing:

- Dry musical-input waveform or level envelope.
- The fixed intended virtual-gain setting.
- The training target's time-varying capture weights.
- The physical-gain capture that serves as the fixed reference.

This analyses the TRAINING TARGET, not a directly observable internal
state of the exported NAM.

Do not claim that the trained NAM literally switches captures or
contains the teacher's blend-weight mechanism.

The trained NAM is a neural approximation of that target and may
generalise differently.


## 9. Separate teacher-target error from trained-NAM error

This is essential to deciding what to do next.

For selected representative cases, evaluate THREE outputs using the
same musical DI and musical-input level:

REFERENCE:
The real fixed-gain amplifier capture.

TEACHER:
The existing envelope-driven blended target, constructed with the
frozen Phase 4E settings.

STUDENT:
The exported Phase 4E standard NAM at the fixed virtual-gain setting.

Compare:

    teacher vs reference
    student vs reference
    student vs teacher

Do not generate new training datasets or train new models.

Reuse the existing target-generation functions to render the teacher
for analysis only.

Ensure that the teacher evaluation uses the same effective input-gain
and output-scaling conventions as the existing builder.

Pay particular attention to how the builder rescales the waveform
before rendering individual source captures. Do not accidentally
apply the player Input gain twice or compare outputs driven by
different effective DI levels.

Document the exact signal path used for each comparison.

Interpretation:

- If the teacher already departs substantially from the fixed-gain
  physical reference when musical input varies, investigate the
  target-construction method before adding more captures.

- If the teacher tracks the fixed-gain reference but the student does
  not, investigate training coverage, model capacity, optimisation
  or generalisation.

- If both track the fixed-gain reference across musical-input levels,
  the current approach is behaving as intended in this tested regime.

- If failures are confined to omitted physical-gain positions, capture
  placement or interpolation may warrant investigation.

These are diagnostic interpretations, not automatic proof of root cause.


## 10. Secondary test: sweep virtual gain while musical input stays fixed

Use the same held-out DI at one fixed musical-input level.

Sweep ONLY the intended player Input gain through the available
virtual-gain range.

Compare the result with the validated physical amplifier captures.

The purpose is to confirm that the fixed-virtual-gain playability findings
have not obscured the principal product requirement:

One standard NAM should still provide a convincing clean-to-saturated
progression when the guitarist deliberately turns the player's Input knob.

Reuse the existing Phase 4E results wherever they already answer this
question.

Do not repeat the entire Phase 4E evaluation unnecessarily.


## 11. Report clear, separate outcomes

Create:

    docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md

Save any new plots and measurements under a dedicated diagnostic
directory without overwriting Phase 4E artifacts.

Report separately:

A. VIRTUAL-GAIN PROGRESSION

With musical input fixed, does turning the player's Input gain
produce the intended progression through amplifier sounds?

B. FIXED-VIRTUAL-GAIN PLAYABILITY

With the player's Input gain fixed, does the model preserve the
selected physical amplifier's response as musical input changes?

C. TRAINING-TARGET BEHAVIOUR

Does the envelope-driven teacher introduce changes in physical-capture
contribution that are inconsistent with a fixed physical-gain reference?

D. STUDENT GENERALISATION

Where does the exported NAM depart from its teacher, and where does
it depart from the physical reference?

Provide per-amplifier, per-gain-position and per-musical-input-level
findings.

Distinguish measured differences from hypotheses about their causes.

Do not use a composite score or declare success based on a single metric.

Human listening remains pending unless somebody actually completes
the listening test.


## 12. Final decision: stay aligned with the mission

End the report by answering:

1. Does the current approach meet the intended virtual-gain-control
   behaviour on the tested amplifiers?

2. Does varying musical input at a fixed virtual-gain setting cause
   unacceptable departures from the corresponding fixed-gain
   physical amplifier?

3. If there is a departure, is it already present in the teacher
   target, primarily introduced by student training, or unresolved?

4. Is there evidence that an additional physical capture would address
   the specific remaining problem?

5. What is the SMALLEST justified next change that moves us towards
   one convincing, portable, input-gain-controlled standard NAM?

Do not automatically recommend G8, a new model architecture,
conditional inference, external DSP or another broad research phase.

If the current models already provide the intended playing experience,
say so and identify the remaining validation needed before production.

If a foundational flaw is demonstrated, describe the smallest targeted
experiment that can test a remedy.

Do not implement that remedy or begin further training without
presenting the findings first.


## Deliverable

Complete the focused diagnostic and report its results.

No model training.
No production integration.
No changes to the frozen Phase 4E experiment.

The purpose of this task is to determine whether the existing
continuous-gain training method is genuinely delivering the musical
experience we set out to build.

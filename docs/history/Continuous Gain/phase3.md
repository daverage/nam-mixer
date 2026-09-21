Continue work in the existing repository:

`C:\Users\daver\Documents\GitHub\hybrid-nam-builder`

Do NOT create a new repository.

The current HEAD should include the completed input-profile/calibration/coverage milestone, currently around:

`72050a66e533c77766d6fa75e8d625e0410baf9f`

Before changing anything:

1. Pull/check the actual current HEAD.
2. Read the current README, CLAUDE.md, pipeline, envelope, blend, calibration, metadata, safety and tests.
3. Run the full existing test suite.
4. Do not discard or rewrite the working preview/input-profile functionality.

The goal of THIS phase is very specific:

# Produce the first genuinely usable NAM A2 hybrid model.

At the end of the phase there must be:

* a reproducible hybrid training target
* generated from the official/current NAM training excitation
* trained using the official/current A2 PackedWaveNet trainer
* exported as a valid `.nam`
* loadable by our existing native NeuralAmpModelerCore renderer
* compared against the original two-NAM hybrid
* with A2 Full and Lite behaviour validated where possible
* with enough provenance to reproduce the model later

Do NOT implement A1 in this phase.

Do NOT implement CLO conversion in this phase.

Do NOT modify the settled `NamtoClo---GP50` project.

Do NOT add new UI polish unrelated to generating/training the A2.

---

# 1. Important architectural distinction: pickup profiles are DESIGN / VALIDATION controls

The current input-profile system correctly simulates different pickup/output levels during preview by changing the actual signal sent to Amp A and Amp B.

Keep that behaviour for preview and validation.

However:

DO NOT apply the chosen pickup-profile gain to the official NAM training excitation when creating the final hybrid target.

This is critical.

For example, if the design was auditioned using:

```text
P90
+1.0 dB
```

that +1 dB helped us answer:

```text
How does a P90 reach the crossover?
How much of the playing becomes Amp A / transition / Amp B?
```

It must NOT mean:

```text
official NAM training input
    ↓
+1 dB
    ↓
train model
```

Otherwise pickup sensitivity starts becoming baked into the trained model itself.

Likewise, a hot humbucker +4.5 dB must not cause the final NAM to have +4.5 dB of hidden extra sensitivity.

The correct concept is:

```text
PICKUP PROFILE
     ↓
used to DESIGN and TEST the crossover

then

OFFICIAL NAM TRAINING INPUT
     ↓
unchanged pickup/reference level
     ↓
fixed approved crossover behaviour
     ↓
hybrid target
```

The final NAM should learn the level-dependent function across the levels contained in the official NAM excitation.

When a real hot pickup later sends a hotter signal into that NAM, it should naturally move farther through the learned crossover response.

---

# 2. Freeze a Hybrid Design before target generation

Introduce a clear immutable representation of the settings we are actually training.

Create something like:

`HybridDesign`

Preferably a dataclass in a new module such as:

`hybrid/design.py`

It should contain at least:

```python
amp_a_path
amp_b_path

crossover_dbfs
transition_width_db

auto_trim_db
manual_b_trim_db
effective_b_trim_db

blend_algorithm

alignment_enabled
alignment_offset_samples

instrument_type
design_reference_profile_id
design_reference_profile_gain_db

calibration_mode
reference_input_level_dbu

amp_a_input_level_dbu
amp_b_input_level_dbu
amp_a_calibration_gain_db
amp_b_calibration_gain_db
calibration_applied

envelope configuration
```

The pickup profile fields above are DESIGN CONTEXT only.

They are not automatically applied to the training excitation.

Once target generation begins, do not silently recompute crossover or trim against the NAM training input.

The design must be frozen.

---

# 3. Freeze the effective Amp B trim that was actually auditioned

This is also important.

Today the preview path can calculate:

```text
Auto match      -X.X dB
Manual tweak    +Y.Y dB
Effective trim  -Z.Z dB
```

The final hybrid target should use:

```text
effective_b_trim_db
```

as a fixed value.

DO NOT run automatic level matching again against the completely different official NAM training excitation.

Otherwise the trained target may not be the same hybrid the user auditioned.

Generation should effectively behave as:

```python
build_hybrid(
    training_pair,
    crossover_dbfs=locked_design.crossover_dbfs,
    transition_width_db=locked_design.transition_width_db,
    auto_level=False,
    manual_b_trim_db=locked_design.effective_b_trim_db,
)
```

Amp A remains the 0 dB reference.

Record:

```text
original auto trim
original manual trim
effective frozen trim
```

in metadata.

---

# 4. Replace the current infinite-history envelope with a finite-memory causal envelope

This must happen BEFORE training.

The current crossover envelope uses:

```text
20 ms causal RMS
5 ms attack
60 ms one-pole release
```

The problem is that the one-pole release has theoretically infinite memory.

A causal NAM model has finite receptive history.

For a synthetic target we should not deliberately create a control signal that depends on arbitrarily old audio.

## Requirement

Change the production crossover envelope so that its dependence on dry input is BOTH:

```text
strictly causal
AND
strictly bounded in time
```

Do not simply truncate the old output after the fact.

Implement a genuinely bounded causal envelope follower.

The precise DSP implementation is up to you, but it must meet these conditions:

1. No future samples.
2. No contribution from dry-input samples older than a documented maximum.
3. Smooth enough to avoid crossover chatter.
4. Similar enough to the existing envelope that the musical behaviour is not radically changed.
5. Deterministic.
6. Efficient enough for current preview use.

Examples of acceptable approaches include a finite causal exponential/FIR-style release or another bounded release detector.

Do not use another recursive filter with an infinite tail and merely describe it as finite.

---

# 5. Fit the control history safely inside the real A2 receptive field

Do not hard-code assumptions from this prompt.

The training environment should inspect the actual A2 configuration supplied by the pinned/current `neural-amp-modeler` package.

The current official simplified A2 trainer uses its packaged:

`nam.train._resources/config_model_packed.json`

Compute or obtain the minimum receptive field of the A2 submodels used for training.

Then assert that the COMPLETE dry-input dependency of our crossover detector fits comfortably inside that receptive field.

Target:

```text
maximum hybrid crossover-control history <= about 100 ms at 48 kHz
```

unless inspection of the actual A2 config gives a strong reason to choose slightly differently.

Leave useful margin.

Remember that if:

```text
20 ms causal RMS
+
80 ms finite release
```

is used, the oldest raw audio influencing the release can be roughly 100 ms old.

Count the complete dependency from ORIGINAL dry sample to current blend weight.

Do not count only the final smoother stage.

Add an automated test that proves bounded history.

For example:

Create two long signals which are completely different in their distant past but are identical for the last N samples.

At the comparison point:

```text
envelope(signal_a)[t]
==
envelope(signal_b)[t]
```

within numerical tolerance once all differing samples are older than the declared maximum history.

Also preserve the existing future-independence/causality test.

---

# 6. Regression-check the new envelope against the old behaviour

Before deleting the existing behaviour, compare both envelope implementations on the bundled DIs.

At minimum:

```text
moderate_brit.wav
clean_smooth.wav
high_thrash.wav
bass_rollin.wav
```

Report:

```text
old p10/p50/p90
new p10/p50/p90

old crossover coverage
new crossover coverage
```

using representative crossover settings.

The new envelope does not have to be numerically identical.

But if it radically changes the practical crossover behaviour, tune/document it before proceeding.

The goal is:

```text
same musical idea
bounded causal memory
```

not:

```text
completely new envelope behaviour
```

---

# 7. Official NAM training input

The genre DIs remain:

```text
preview
design
validation
held-out testing
```

They MUST NOT become the primary training excitation.

Use the current official NAM training input expected by the installed/pinned A2 trainer.

Do not invent our own excitation.

Do not concatenate the genre guitar clips and call it training data.

The official trainer currently expects 48 kHz training.

Provide a simple local mechanism for obtaining/selecting the official training input.

Possible implementation:

```text
assets/training/input.wav
```

or:

```text
work/training_input/input.wav
```

Do not commit a large downloaded training WAV unless licensing/repo policy clearly makes that appropriate.

If a stable official download mechanism exists, a helper script is acceptable.

Otherwise provide a clear file-selection/upload path.

The training script must ask the NAM package itself to validate/detect the input version.

Do not rely solely on filename.

Record:

```text
input path
sample rate
length
SHA-256
MD5 if needed by official NAM input detection
official detected input version
```

in the training manifest.

If the file is not recognized as an appropriate standard NAM input by the selected official trainer:

ABORT.

Do not bypass the check just to make training run.

---

# 8. Training-input signal path

When generating the final training pair, use:

```text
OFFICIAL NAM INPUT
       |
       +-----------------------------+
       |                             |
       v                             v
source-model calibration A     source-model calibration B
       |                             |
       v                             v
    AMP A NAM                     AMP B NAM
       |                             |
       +--------------+--------------+
                      |
          fixed approved B trim
                      |
official input ------>|
       |
       v
finite causal crossover envelope
       |
       v
fixed crossover / fixed transition
       |
       v
smoothstep linear blend
       |
       v
HYBRID TARGET
```

Important:

The crossover envelope sees the common official training input BEFORE per-model source calibration.

Exactly as the preview architecture treats the common virtual guitar level.

Do not derive the crossover from either processed amp.

Do not derive the crossover from the calibration-adjusted Amp A input.

Do not derive it from Amp B input.

---

# 9. Calibration behaviour for target generation

Reuse the current calibration rules.

If:

```text
calibration mode = Auto
AND
Amp A has input_level_dbu
AND
Amp B has input_level_dbu
```

then apply the already-implemented per-model input compensation:

```python
reference_input_level_dbu - model_input_level_dbu
```

to each source model.

The common crossover envelope still sees the uncalibrated common training input.

If only one source model has calibration metadata:

```text
fall back to Raw for BOTH
```

and record a warning.

Do not calibrate one model but not the other.

If both are uncalibrated:

```text
Raw
```

If final target generation used valid Auto calibration for both source models, the resulting final hybrid A2 may legitimately use:

```text
input_level_dbu = reference_input_level_dbu
```

in its NAM metadata, if the official export API supports setting it correctly.

If source calibration was unavailable and Raw fallback was used:

DO NOT invent an input_level_dbu for the final model.

Leave it absent/unset.

Likewise, do not invent `output_level_dbu`.

---

# 10. Generate the real target as FLOAT audio

Implement target generation independently of Flask.

Create something like:

`hybrid/training_target.py`

with a pure orchestration function such as:

```python
generate_training_bundle(
    design,
    official_input_path,
    output_directory,
)
```

Flask should call this function.

The function should be testable directly.

Outputs should include at least:

```text
input.wav
hybrid_target_raw.wav
hybrid_target.wav
hybrid.hybrid.json
training_manifest.json
```

`input.wav` can be copied into the self-contained bundle if practical.

The target WAV should be:

```text
48 kHz
mono
same exact sample count as input
32-bit float WAV
```

Do not write the training target as PCM16.

Do not clip.

Do not normalize individual sections.

---

# 11. Target safety

Keep the current important distinction:

```text
preview safety != training-target safety
```

NEVER run:

```python
preview_safety_limiter()
```

on the training target.

Run the normal safety checks:

```text
NaN
Inf
silence
peak
suspicious jumps
```

If the target exceeds the chosen safe target peak, use only:

```python
apply_peak_ceiling()
```

which applies ONE CONSTANT GAIN to the entire target.

No limiter.

No compression.

No per-section gain.

No automatic makeup gain.

Record:

```text
raw target peak
final target peak
global safety gain reduction
```

in the manifest.

Keep `hybrid_target_raw.wav` for debugging if disk use is reasonable.

The actual trainer uses `hybrid_target.wav`.

---

# 12. Expand metadata/provenance properly

The existing `HybridMetadata` is a good start but is not complete enough for real training.

Extend metadata to include at least:

```text
hybrid_builder_version / git commit

Amp A:
  filename
  SHA-256
  architecture
  sample rate
  input_level_dbu
  output_level_dbu

Amp B:
  filename
  SHA-256
  architecture
  sample rate
  input_level_dbu
  output_level_dbu

design:
  instrument type
  reference pickup profile
  reference pickup relative gain
  crossover dBFS
  transition width
  blend algorithm
  envelope algorithm
  envelope RMS window
  envelope attack behaviour
  envelope release behaviour
  maximum envelope memory
  original auto trim
  manual trim
  frozen effective B trim
  alignment enabled
  alignment offset

calibration:
  requested mode
  effective mode
  reference dBu
  Amp A compensation
  Amp B compensation
  warning if any

training input:
  filename/version
  sample rate
  frame count
  SHA-256

target:
  raw SHA-256
  final SHA-256
  peak before safety
  peak after safety
  global safety reduction

training:
  neural-amp-modeler version
  torch version
  Python version
  A2 config identifier/hash
  training settings
  device used
```

Use SHA-256 for project provenance.

Do not rely on filenames alone.

---

# 13. Pin the training implementation to official A2

Create/update:

`requirements-training.txt`

The current official NAM release we are targeting is 0.13.x, where the simplified trainer uses A2/PackedWaveNet.

For reproducibility, prefer pinning the exact NAM trainer version used for this experiment rather than a floating unbounded dependency.

For example, after verifying compatibility:

```text
neural-amp-modeler==0.13.0
```

Do not pin Torch blindly to a platform-incompatible build unless necessary.

Document the exact successfully installed Torch version in the manifest.

The normal Flask/runtime environment must remain independent of Torch.

The runtime app should still work without any training dependencies installed.

---

# 14. Separate A2 training environment

The current app environment may be Python 3.14 and may not be suitable for Torch.

Do NOT destabilize the working runtime environment to make training work.

Create/document a separate training environment.

On the current Windows development machine prefer Python 3.12 if available.

For example conceptually:

```text
.venv-a2
```

Do not hard-code one machine-specific Python installation path into source.

A helper such as:

```text
scripts/setup_a2_env.ps1
```

is acceptable.

But the actual training code must also be runnable directly from any correctly configured Python environment.

---

# 15. Use the official simplified A2 trainer, not a home-made A2 config

This is important.

Do NOT manually recreate A2 architecture JSON from memory.

Do NOT use an old A1 WaveNet config.

Do NOT assume that an example `wavenet_packed.json` from another part of the repo is identical to the current simplified A2 architecture.

Use the actual A2 configuration/resources supplied by the pinned official `neural-amp-modeler` package.

The current simplified trainer obtains the A2 model configuration from its packaged:

```text
nam.train._resources/config_model_packed.json
```

Prefer calling the current official simplified training API that the GUI itself uses.

Inspect the actual installed 0.13.x function signatures before writing the wrapper.

Do not guess an old `core.train()` signature.

Create something like:

`scripts/train_a2.py`

Its job is to:

1. load the generated training manifest
2. locate input.wav / hybrid_target.wav
3. verify hashes/sample rate/length
4. call the official current A2 training implementation
5. explicitly use synthetic latency = 0
6. use the official normal-quality defaults
7. suppress interactive plots where supported
8. export the packed A2 `.nam`
9. update training metadata with the exact environment/settings

Do not use a demo/very-low-epoch configuration as the final model.

A quick smoke-test training mode is fine for development.

But completion requires a normal usable training run unless the machine genuinely cannot support it.

---

# 16. Zero latency

Our target is synthetic.

There is no ADC/DAC/DAW round-trip delay.

Therefore:

```text
latency = 0 samples
```

is authoritative.

Do not auto-detect a fake latency from the synthetic target if the official API allows explicitly setting zero.

Input and target must remain exactly sample-aligned.

Add an automated sanity test demonstrating:

```text
training input frame count == target frame count
```

and no samples were prepended/deleted.

---

# 17. Do not bypass NAM's standard data checks

Because the target is deterministic synthetic audio, the standard NAM validation/repeat sections should behave very well.

Run the trainer's normal input/data validation.

If the official trainer reports that validation repeats do not match sufficiently:

STOP and investigate.

Likely causes could include:

```text
hybrid control history leaking across supposedly repeated sections
source-model state/memory
incorrect target alignment
incorrect input version
target generation bug
```

Do not simply disable/ignore the checks to get a model.

If an official current training-input version legitimately has reduced checks, document that rather than pretending a check passed.

---

# 18. A2 output expectations

One A2 packed training run should produce the current PackedWaveNet model containing its supported submodels.

Treat:

```text
A2 Full
```

as the authoritative quality result.

Treat:

```text
A2 Lite
```

as an important secondary result.

Do NOT train A1 in this phase.

Do not create separate fake “A2 Full.nam” and “A2 Lite.nam” files unless the official trainer/export format actually does so.

Preserve the official packed A2 structure.

---

# 19. Validate the exported .nam structurally

After training:

1. Confirm the file exists.
2. Parse it as NAM JSON.
3. Record version/architecture/config.
4. Confirm it is the expected current packed/A2 structure.
5. Load it using the existing native NeuralAmpModelerCore renderer.
6. Render real audio successfully.
7. Confirm:

   * mono output
   * correct frame count
   * finite values
   * correct sample rate

Do not call the phase complete merely because the Python trainer wrote a file.

NAMCore must actually load and render it.

---

# 20. Extend native rendering to test A2 Full and Lite if necessary

The underlying NAMCore render tool supports selecting the slim/submodel mode.

If our current Python wrapper does not expose that, add an OPTIONAL parameter such as:

```python
render(model, audio, sample_rate, slim=False)
```

Do not break the existing API.

Use:

```text
slim=False
```

for A2 Full/reference validation.

Where NAMCore semantics make it valid, use:

```text
slim=True
```

to exercise the Lite/slim submodel.

Verify the CLI/API arguments from the actual pinned NAMCore tool rather than assuming.

---

# 21. The first real model should be the INVERTED hybrid stress test

For the first end-to-end training validation use:

```text
LOW INPUT = HIGH GAIN
HIGH INPUT = CLEAN
```

Specifically:

```text
Amp A:
JCM800_2203_Modified_HighGain.nam

Amp B:
FenderSuperReverb1977_Clean.nam
```

This is deliberately backwards compared with normal amplifier behaviour.

It is diagnostically useful because a conventional static nonlinear amp cannot trivially fake:

```text
more input → cleaner
```

through ordinary saturation behaviour.

Use:

```text
moderate_brit.wav
```

for the design/listening/coverage phase because it has a wide useful dynamic range.

Initial settings:

```text
alignment OFF
auto level ON during design
manual B trim 0 unless listening requires otherwise
transition 8 dB
calibration Auto
reference calibration +12 dBu
```

For the actual current source pair, calibration may fall back to Raw because the existing Fender model does not provide input calibration metadata.

That is acceptable.

Do not invent the missing calibration.

---

# 22. Choosing the first crossover

Do not invent a crossover just because this prompt needs a number.

Use the current UI's active-envelope analysis and suggested crossover as the STARTING point.

The design should have meaningful coverage across realistic pickup levels.

At minimum inspect:

```text
Vintage single
Standard single
PAF humbucker
P90
Modern humbucker
Hot humbucker
Extreme passive
```

For this inverted hybrid the coverage labels still mean:

```text
Amp A = dirty JCM800
Amp B = clean Fender
```

A useful stress-test design should allow the reference/higher-output profiles to meaningfully traverse from A toward B.

Do not claim a setting is musically approved without human listening.

But select a technically meaningful first model that exercises both endpoints and the transition.

Record the exact crossover used.

---

# 23. Training-target generation ignores the pickup boost

Suppose the design was auditioned with:

```text
P90 +1 dB
```

The frozen manifest may say:

```text
design reference profile = P90
design reference profile gain = +1 dB
```

But target generation uses:

```text
official NAM input gain = 0 dB profile adjustment
```

The only changes before individual source NAMs are:

```text
per-model calibration compensation
```

if valid.

Add an explicit manifest field such as:

```text
pickup_profile_applied_to_training_input: false
```

and test it.

This distinction must never become ambiguous later.

---

# 24. Validation against the training target

After training the A2, render the SAME official input through the trained A2 Full model.

Compare:

```text
A2 output
vs
hybrid_target.wav
```

Report at least:

```text
raw ESR
gain-normalized ESR
RMS difference
peak difference
```

Use existing NAM metrics if readily available rather than inventing subtly incompatible versions.

A global level discrepancy and a shape/dynamics discrepancy should be distinguishable.

Do the same for Lite if possible.

Do not overfit the system further based only on one metric yet.

---

# 25. Held-out musical validation

The more important test is material the A2 did not train on.

Create a validation utility.

For at least:

```text
moderate_brit.wav
clean_smooth.wav
high_thrash.wav
```

and where useful a bass DI for architectural testing, compare:

```text
REFERENCE HYBRID
(two original source NAMs + locked hybrid design)

vs

TRAINED A2 FULL

vs

TRAINED A2 LITE
```

The reference hybrid must use exactly the locked design:

```text
same crossover
same transition
same fixed B trim
same calibration rules
same finite envelope
alignment OFF
```

For pickup/output-level testing, run held-out DI at several actual profile gains.

At minimum:

```text
-7.0 dB  vintage single
-3.0 dB  standard single
 0.0 dB  PAF
+1.0 dB  P90
+4.5 dB  hot humbucker
+6.0 dB  extreme passive
```

Unlike target generation, these gains ARE applied during validation because now we are testing how different real input levels behave.

For each gain:

1. Apply that level to the held-out dry DI.
2. Render the live two-NAM hybrid reference correctly.
3. Feed the same physical/reference signal to the trained hybrid A2.
4. Compare.

Do not use the deprecated envelope-only `dry_gain_db`.

Use real audio gain.

---

# 26. Calibration of the trained A2 during validation

If the final trained hybrid has valid `input_level_dbu` metadata because both source models were calibrated and training used the chosen reference calibration:

validate it using those semantics.

If training was Raw because source calibration was incomplete:

feed the trained model the same raw/profile-adjusted digital signal used by the reference comparison.

Do not apply invented compensation.

---

# 27. Generate listening files

Metrics are not enough.

Produce easy-to-audition validation WAVs.

For each selected held-out DI and selected input profile, create:

```text
reference_hybrid.wav
a2_full.wav
a2_lite.wav
```

Prefer a directory structure like:

```text
work/a2/<design_id>/validation/
    moderate_brit/
        vintage_single/
        paf/
        p90/
        hot_humbucker/
        extreme_passive/
```

Also generate a convenient sequential comparison if useful:

```text
Reference
A2 Full
A2 Lite
```

with sensible silence between sections.

Do not normalize each comparison independently.

If a common listening safety gain is required, use the SAME fixed gain for all compared files and document it.

---

# 28. Specifically test the inverted behaviour

This is the reason for using JCM800 → Fender first.

Build a validation clip or report across increasing real input level:

```text
-7 dB
-3 dB
 0 dB
+1 dB
+4.5 dB
+6 dB
```

The live reference should move progressively toward:

```text
dirty JCM800
     ↓
blend
     ↓
clean Fender
```

The trained A2 should reproduce that progression as closely as possible.

Do not infer “cleaner” solely from output RMS.

Generate the listening files.

Report objective similarity at each level.

Human listening remains the final judgement.

If increasing input level causes the trained A2 to behave like an ordinary amp and simply become more distorted, call that out clearly.

That would mean this synthetic behaviour is not being captured sufficiently.

---

# 29. Add a simple A2 creation workflow to the existing UI

Do NOT redesign the UI again.

Replace the stale:

```text
Generate Hybrid Target
```

placeholder behaviour.

Add a compact final card/section such as:

```text
5. CREATE A2

Locked design:
JCM800 → Fender
Crossover: -XX.X dBFS
Transition: 8 dB
Effective B trim: -X.X dB
Calibration: Raw / Auto

Official NAM training input:
[Ready / Missing]

[Generate Training Bundle]

Bundle:
work/a2/<id>/

A2 training:
[environment detected / not configured]

Command:
...

Result:
Hybrid_JCM800_to_Fender_A2.nam
```

Do not make the web UI responsible for complex environment management.

The essential requirement is that:

```text
Generate Training Bundle
```

works correctly.

If a configured A2 training Python executable exists, an optional:

```text
Train A2
```

button may invoke the training wrapper.

But do not build a queue/database/job scheduler just for this.

A clean CLI training command is acceptable for this phase.

---

# 30. `/api/generate` must become real

Replace the current stale 501 implementation.

`POST /api/generate` should:

1. Require an existing rendered/auditioned pair.
2. Capture/freeze the current HybridDesign.
3. Require a valid official NAM training input.
4. Generate the self-contained training bundle.
5. Return:

   * design id
   * bundle path
   * target path
   * manifest path
   * safety report
   * calibration summary
   * exact training command

It must NOT perform preview limiting.

It must NOT silently train using the genre DI.

It must NOT recompute auto trim from the official training input.

---

# 31. Training wrapper

Create:

`scripts/train_a2.py`

Suggested usage:

```text
python scripts/train_a2.py work/a2/<design_id>/training_manifest.json
```

Optional flags may include:

```text
--quick
--device auto
--output-dir ...
```

But default invocation should mean:

```text
train a real usable A2
```

not a smoke test.

The script should:

```text
validate environment
validate neural-amp-modeler version
validate input/target
run official A2 trainer
verify output .nam
render it through NAMCore
run basic target comparison
update manifest
exit non-zero on failure
```

Do not put Flask imports in this script.

Keep training reusable independently.

---

# 32. Training environment diagnostics

At start print/report:

```text
Python
neural-amp-modeler
Torch
PyTorch Lightning
CUDA available?
MPS available?
CPU only?
selected accelerator
```

On Windows/NVIDIA use GPU if the official trainer selects/supports it.

Do not add custom CUDA logic unless necessary.

Let official Torch/NAM facilities handle normal accelerator selection.

If no GPU exists, warn that CPU is being used but do not corrupt/change model architecture to compensate.

---

# 33. Model naming

Generate a descriptive final filename.

For this first experiment something like:

```text
Hybrid_JCM800_to_Fender_A2.nam
```

If necessary add a compact design id:

```text
Hybrid_JCM800_to_Fender_A2_<id>.nam
```

Do not call it Fender→JCM800 accidentally.

Remember:

```text
Amp A = LOW input = JCM800
Amp B = HIGH input = Fender
```

---

# 34. Verify the final model with our app

Once the model exists:

1. Upload/select it in the existing Hybrid Builder NAM loader.
2. Confirm metadata parses.
3. Render it using native NAMCore.
4. Confirm no errors.
5. Use it as an ordinary NAM outside the hybrid pair where practical.

The final output must be a normal usable `.nam`, not an internal checkpoint.

---

# 35. A1 is explicitly OUT OF SCOPE

Do NOT train A1 yet.

Do NOT add an A1 selector.

Do NOT spend time comparing architectures.

The reason for this phase is to answer:

```text
Can current official A2 learn our synthetic level-dependent hybrid?
```

A2 Full is the primary answer.

A2 Lite comes effectively with the packed architecture and is useful secondary evidence.

Only after the complete chain works should we consider an A1 control experiment.

---

# 36. CLO is explicitly OUT OF SCOPE

Do not modify or call the GP50 CLO converter yet.

The next phase after this one will be:

```text
live reference hybrid
    ↓
A2 hybrid
    ↓
settled NamtoClo---GP50
    ↓
GP50
```

We first need to prove the A2 itself.

---

# 37. Tests required

Keep all existing tests passing.

Add tests for at least:

### Finite envelope

* future independence
* exact bounded past dependence
* deterministic output
* expected length
* no NaN/Inf

### Design freezing

* current auto trim captured
* manual trim captured
* effective trim frozen
* generation does not recompute auto trim

### Pickup-profile separation

* preview profile gain still affects real Amp A/B render
* training target DOES NOT apply pickup-profile gain
* manifest says profile was design context only

### Training input

* 48 kHz required
* mono required
* input/target exact frame equality
* invalid training input rejected

### Calibration

* both calibrated → per-model compensation
* one calibrated → Raw both
* Raw → no compensation
* training envelope always uses common input before model compensation

### Target

* float WAV
* no preview limiter
* fixed peak reduction only
* raw/final hashes recorded
* no NaN/Inf

### Metadata

* source hashes
* git commit
* envelope parameters
* fixed trim
* training input hash
* target hash

### A2 wrapper

Mock the official trainer for normal unit tests.

Do not make the fast test suite require an actual Torch training run.

Have a separate integration/manual test path for real A2 training.

### NAMCore validation

If a trained test A2 exists, verify rendering.
Otherwise skip with a clear reason.

---

# 38. Real end-to-end validation required

Before declaring completion:

1. Generate a real bundle for the inverted JCM800 → Fender hybrid.
2. Use the official current NAM training input.
3. Confirm NAM data checks.
4. Train a real A2.
5. Produce a `.nam`.
6. Load it through NAMCore.
7. Render the official input.
8. Compare with the hybrid target.
9. Run held-out musical validation.
10. Produce Full/Lite listening files if possible.
11. Exercise the inverted input-level progression.

Do not stop at:

```text
"code implemented, training not run"
```

unless the machine genuinely cannot run the official trainer.

If a hard environment/hardware blocker exists, report the exact blocker and leave the generated training bundle and one-command training path ready.

But make every reasonable effort to finish with an actual `.nam`.

---

# 39. Completion criteria

This phase is complete only when:

1. Production crossover envelope has finite causal memory.
2. Its maximum history is verified to fit inside current A2 receptive history with margin.
3. Existing preview behaviour remains sensible.
4. HybridDesign can be frozen.
5. Effective B trim is frozen from the auditioned design.
6. Pickup-profile gain is NOT baked into training excitation.
7. Official NAM input is used.
8. Training target is generated at 48 kHz mono matching exact input length.
9. Source model calibration is handled correctly.
10. Training target is never limited/clipped.
11. Only fixed whole-file safety reduction is allowed.
12. Full provenance is written.
13. `/api/generate` works.
14. `scripts/train_a2.py` works.
15. Official A2 trainer is used.
16. No hand-written fake A2 configuration is used.
17. Synthetic latency is exactly zero.
18. NAM's normal data checks are not bypassed.
19. A real packed A2 `.nam` is produced.
20. NAMCore loads and renders it.
21. A2 Full is compared against the live hybrid.
22. A2 Lite is tested where supported.
23. Held-out genre DI comparisons exist.
24. Multi-level inverted-behaviour comparison exists.
25. A1 was not added.
26. CLO was not added.
27. All existing + new tests pass.
28. Changes are committed and pushed.

---

# 40. Final report

When finished, report:

## Repository

* final commit SHA
* files added
* files changed
* tests passed/skipped

## Envelope

* previous algorithm
* new finite algorithm
* exact maximum dry-input memory in samples and ms
* current A2 receptive field in samples and ms
* safety margin

## Design

* Amp A
* Amp B
* design DI
* reference input profile
* crossover
* transition
* auto trim
* manual trim
* frozen effective trim
* calibration requested/effective
* source input_level_dbu values

## Training input

* official detected NAM input version
* sample rate
* duration
* SHA-256

## Target

* raw target peak
* safety reduction
* final target peak
* SHA-256
* confirmation that pickup profile was NOT applied
* confirmation preview limiter was NOT used

## A2 environment

* Python version
* neural-amp-modeler version
* Torch version
* accelerator
* A2 config hash/identifier
* training settings

## Final NAM

* exact filename/path
* file SHA-256
* NAM file format version
* packed/A2 confirmation
* NAMCore load success
* Full render success
* Lite/slim render success if supported

## Quality

* training-input target vs A2 Full metrics
* target vs Lite metrics
* held-out DI metrics
* metrics by input gain:

  * -7 dB
  * -3 dB
  * 0 dB
  * +1 dB
  * +4.5 dB
  * +6 dB

## Inverted hybrid result

State clearly whether the trained A2 appears to preserve:

```text
LOW input  → dirty JCM800
HIGH input → clean Fender
```

Do not declare this subjectively successful purely from metrics.

List the generated listening files for human evaluation.

## Deviations

List any deviation from this prompt and explain why.

Finally:

commit and push the completed phase.

Do not automatically start A1 or CLO work after this commit.

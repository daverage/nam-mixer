Continue work in the existing repository:

`C:\Users\daver\Documents\GitHub\hybrid-nam-builder`

Do NOT create another repository.

The current repo already has:

* real NAM inference via the native NeuralAmpModelerCore v0.5.4 renderer
* Amp A / Amp B rendering and caching
* causal dry-input RMS envelope
* smoothstep A→B dynamic blend
* automatic crossover-region level matching
* manual Amp B trim on top of automatic matching
* alignment disabled by default
* real browser playback for Amp A / Hybrid / Amp B
* NAM file upload
* genre-specific guitar and bass DI files
* crossover and transition controls
* a test-only `dry_gain_db`
* a live “journey between amps” visualization
* a modern two-column UI
* tests passing at the current HEAD
* `/api/generate` intentionally not implemented yet

The next milestone is to replace the crude/test-only input-gain concept with a physically meaningful, research-grounded **instrument input profile system**, add proper NAM input calibration handling, and show whether each realistic input profile can actually traverse Amp A → transition → Amp B.

Do NOT implement A2 training yet.

Do NOT implement CLO conversion in this repo.

Do NOT add React, Node, a database, Docker, cloud services, or another framework.

Use the existing Flask + Python + native NAMCore + vanilla JS architecture.

---

# 1. Core conceptual change

The current `dry_gain_db` is explicitly test-only:

* it shifts the crossover envelope
* it does NOT change the signal actually sent through Amp A and Amp B

That was useful for debugging the crossover, but it is NOT how different pickup output levels behave physically.

Replace the main user-facing concept with a real:

`input_profile_gain_db`

A real input profile must affect the signal BEFORE BOTH:

1. NAM inference
2. crossover envelope detection

The correct conceptual path is:

```text
SOURCE DI
   |
   v
INPUT PROFILE GAIN
   |
   +----------------------+
   |                      |
   v                      v
COMMON DRY ENVELOPE      NAM INPUT CALIBRATION
                          |
                    +-----+-----+
                    |           |
                    v           v
                  AMP A       AMP B
                    |           |
                    +-----+-----+
                          |
                    LEVEL MATCH
                          |
                  DYNAMIC CROSSFADE
                          |
                       HYBRID
```

Changing input profile therefore IS an expensive operation and must invalidate/re-render the Amp A/B pair.

Changing crossover, transition width, or manual B trim must remain cheap and must NOT rerender the NAMs.

---

# 2. Research-grounded pickup/input profiles

Add a central module, preferably:

`hybrid/input_profiles.py`

Do not scatter these constants through Flask or JavaScript.

Define profile metadata with a dataclass or equivalent.

Suggested fields:

```python
id
instrument
label
gain_db
reference
research_range_db
description
confidence
requires_custom_gain
source_notes
```

The preset offsets below are not arbitrary guesses. They are rounded simulation values derived from manufacturer relative-output data.

## GUITAR reference family

Use a vintage/PAF-type passive humbucker as the guitar 0 dB reference.

Underlying research basis:

```text
Vintage / low single:
approximately 90–125 mV
relative to 250 mV = roughly -8.9 to -6.0 dB

Standard / hotter single:
approximately 160–200 mV
relative to 250 mV = roughly -3.9 to -1.9 dB

Vintage / PAF humbucker:
approximately 220–250 mV
relative to 250 mV = roughly -1.1 to 0 dB

P90:
approximately 270–287 mV
relative to 250 mV = roughly +0.7 to +1.2 dB

Medium / modern humbucker:
approximately 300–375 mV
relative to 250 mV = roughly +1.6 to +3.5 dB

Hot humbucker:
approximately 400–435 mV
relative to 250 mV = roughly +4.1 to +4.8 dB

Extreme passive:
approximately 510 mV
relative to 250 mV = roughly +6.2 dB
```

Use these rounded application presets:

```python
GUITAR_PROFILES = {
    "vintage_single": -7.0,
    "standard_single": -3.0,
    "vintage_humbucker": 0.0,
    "p90": +1.0,
    "modern_humbucker": +2.5,
    "hot_humbucker": +4.5,
    "extreme_passive": +6.0,
}
```

Suggested labels:

```text
Vintage / low-output single coil       -7.0 dB
Standard / hotter single coil          -3.0 dB
Vintage / PAF humbucker                 0.0 dB
P90                                     +1.0 dB
Medium / modern humbucker              +2.5 dB
Hot humbucker                          +4.5 dB
Extreme passive                       +6.0 dB
Active / buffered                     Custom
```

The 0 dB point is a REFERENCE for relative simulations. It does not mean every PAF produces the same absolute voltage.

---

# 3. Bass profiles

Bass MUST be supported as a first-class instrument family.

Do NOT derive bass offsets relative to the guitar 250 mV reference.

Use a separate passive-bass reference around a conventional J/P output level.

Manufacturer data used for this family includes approximately:

```text
standard Jazz/P bass:
150–163 mV

moderately hotter bass:
170–200 mV

high-output passive bass:
230–250 mV
```

Use:

```python
BASS_PROFILES = {
    "standard_jp": 0.0,
    "modern_passive_bass": +1.5,
    "hot_passive_bass": +3.5,
}
```

Labels:

```text
Standard Jazz / Precision bass          0.0 dB
Modern / hotter passive bass           +1.5 dB
Very high-output passive bass          +3.5 dB
Active / preamped bass                 Custom
```

Do NOT claim that active bass pickups are inherently hotter.

There is manufacturer evidence showing some active bass pickup elements are actually below stock passive output before the onboard preamp.

Active basses vary heavily according to:

* pickup
* onboard preamp
* EQ boost/cut
* supply voltage
* instrument design

Therefore active/preamped bass must use a custom relative gain.

---

# 4. Active pickups: deliberately no universal preset

DO NOT add:

```text
Active = +9 dB
```

or any similar universal value.

Research shows active pickup output varies too much.

Examples from one active manufacturer show roughly:

```text
active single-coil:
1.0 V "string" output

active high-output humbucker:
3.0 V "string" output
```

which is already about a 9.5 dB difference inside one manufacturer's active range.

On a different “strum” measurement the same pair is much closer.

Some Fishman Fluence pickups additionally provide a selectable 6 dB output reduction.

Therefore add:

```text
Active / Buffered — Custom
```

for guitar and:

```text
Active / Preamped Bass — Custom
```

for bass.

When selected, reveal a custom slider/input.

Suggested custom range:

```text
-12 dB to +12 dB
```

This slider range is an engineering adjustment range, NOT a claim that all active instruments live within exact endpoints.

UI wording should make this distinction clear.

For example:

```text
Active instruments vary too much for one trustworthy fixed output preset.
Set the relative level manually.
```

Do not overload the user with the entire research explanation in the main UI.

Put the detail in documentation/help text.

---

# 5. Document the research

Create:

`docs/INPUT_PROFILE_RESEARCH.md`

Explain:

* why relative output matters for this project
* why absolute guitar dBFS cannot be inferred from pickup type alone
* why selected DI files cannot tell us their original pickup output after normalization
* why passive guitar values use one consistent manufacturer's mV relative-output data
* why bass is treated as a separate reference family
* why active is custom
* the conversion formula:

```python
gain_db = 20 * log10(output / reference_output)
```

Document representative manufacturer examples, including:

Guitar:

* low/vintage single roughly 90–125 mV
* hotter single roughly 160–200 mV
* vintage PAF roughly 220–250 mV
* P90 roughly 270–287 mV
* medium humbucker roughly 300–375 mV
* hot humbucker roughly 400–435 mV
* extreme passive roughly 510 mV

Bass:

* standard J/P roughly 150–163 mV
* hotter models roughly 170–200 mV
* high-output bass pickups roughly 230–250 mV

Active:

* active products vary significantly
* active circuitry must not be equated automatically with higher output
* active/preamped basses are especially variable

Make clear these profiles are **research-based category simulations**, not laboratory calibration values for every pickup.

Pickup height, string type, string gauge, pick attack, pickup position and instrument wiring all affect real level.

---

# 6. Reference DI issue

The genre DI clips in this repository were found during Phase 1 to be mostly normalized around a common RMS level.

That means:

THEIR ORIGINAL REAL-WORLD PICKUP OUTPUT LEVEL IS NOT TRUSTWORTHY.

Do not pretend we can recover whether a source DI was originally recorded using a Strat, P90, humbucker, etc. from its waveform.

Instead treat the selected DI as a reference performance.

The input profiles simulate RELATIVE level differences around that reference performance.

For guitar DIs default the reference assumption to:

```text
Vintage / PAF humbucker = 0 dB
```

For bass DIs default to:

```text
Standard Jazz / Precision bass = 0 dB
```

If filenames beginning with `bass_` make automatic instrument selection trivial, use that only as an instrument-category default.

Do not infer actual pickup construction from waveform content.

Allow the user to override Guitar/Bass manually.

---

# 7. Real input-profile gain implementation

Add a helper such as:

```python
def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)
```

Then in the expensive render path:

```python
profiled_dry = source_dry * db_to_amplitude(input_profile_gain_db)
```

The envelope MUST be computed from:

```python
profiled_dry
```

NOT the original source DI.

Therefore:

```python
envelope_db = rms_envelope_db(profiled_dry, sample_rate)
```

Both Amp A and Amp B must receive this same physical/reference-profile-adjusted guitar signal, subject only to the model-specific NAM calibration compensation described below.

Do NOT simply shift `pair.envelope_db`.

The whole point of this feature is that the NAM models themselves must actually receive the hotter/quieter signal.

---

# 8. Retire the current test-only dry_gain_db from the main flow

The existing:

```python
dry_gain_db
```

was useful for testing crossover logic because it only shifted the blend envelope.

It should no longer be the user-facing input simulation.

Prefer to remove it from:

* main UI
* `/api/preview`
* `/api/blend_info`
* `/api/blend_curve`
* normal `build_hybrid()` calls

If retaining it internally is useful for regression tests, mark it clearly deprecated/test-only.

The normal production architecture should be:

```text
input gain happens in render_pair()
build_hybrid() receives a RenderedPair that already represents the selected input profile
```

Do not have two different user-facing concepts that both look like “input gain”.

---

# 9. NAM calibration — implement this in the same milestone

This is important.

Two `.nam` files may have different:

```text
input_level_dbu
```

metadata.

Feeding identical raw digital samples into two differently calibrated models does not necessarily represent feeding the same physical guitar voltage into both amps.

The official Neural Amp Modeler plugin uses:

```python
model_input_adjustment_db =
    user_input_calibration_dbu - model_input_level_dbu
```

Implement this behaviour.

Create something like:

`hybrid/calibration.py`

with a pure testable function:

```python
def input_calibration_gain_db(
    reference_input_level_dbu: float,
    model_input_level_dbu: float,
) -> float:
    return reference_input_level_dbu - model_input_level_dbu
```

The default virtual/reference input calibration should be:

```text
+12.0 dBu
```

because this is the default calibration reference used by the official NAM plugin.

This is NOT a pickup output value.

It describes the analogue level corresponding to digital full scale.

Keep these concepts separate in names and UI.

---

# 10. Calibration modes

Add:

```text
NAM Input Calibration:
Auto
Raw
```

Default:

```text
Auto
```

AUTO behaviour:

If BOTH Amp A and Amp B have valid `input_level_dbu`:

```text
apply model-specific input compensation to each render
```

For example:

```python
amp_a_gain_db =
    reference_input_level_dbu - amp_a.input_level_dbu

amp_b_gain_db =
    reference_input_level_dbu - amp_b.input_level_dbu
```

Then:

```python
amp_a_input =
    profiled_dry * db_to_amplitude(amp_a_gain_db)

amp_b_input =
    profiled_dry * db_to_amplitude(amp_b_gain_db)
```

The crossover detector MUST still see:

```text
profiled_dry
```

not either model-specific calibration-adjusted input.

That keeps the crossover linked to the common virtual guitar level rather than a source-model's recording calibration.

If one model is calibrated and the other is not:

DO NOT silently calibrate only one.

Use raw digital input for both and display a clear warning:

```text
Input calibration unavailable for this pair:
one or both source NAMs do not contain input_level_dbu.
Using raw digital level for both models.
```

If both are uncalibrated:

```text
Raw mode effectively applies.
```

RAW mode:

```text
same profiled_dry goes directly to both models
```

with no metadata compensation.

Do not invent calibration for uncalibrated models.

---

# 11. RenderedPair needs provenance

Expand `RenderedPair`.

It should retain enough state to know exactly what was rendered.

Suggested fields:

```python
source_dry
profiled_dry

amp_a
amp_b

envelope_db
sample_rate

instrument_type
input_profile_id
input_profile_gain_db

calibration_mode
reference_input_level_dbu
calibration_applied

amp_a_model_input_level_dbu
amp_b_model_input_level_dbu

amp_a_calibration_gain_db
amp_b_calibration_gain_db

input_peak_dbfs
```

Use sensible dataclasses or nested structures if this becomes unwieldy.

Do not duplicate audio arrays unnecessarily if aliases/references are sufficient.

---

# 12. Input peak/headroom warning

Applying a +4.5 or +6 dB profile may push a normalized DI above 0 dBFS.

Do NOT silently clip it.

Do NOT normalize it behind the user's back.

Calculate:

```text
profiled input peak dBFS
```

and expose it.

If peak is at or above 0 dBFS, show a warning such as:

```text
This simulated input exceeds 0 dBFS relative to the reference DI.
The floating-point renderer can process it, but a real ADC using this
reference gain would have clipped. Treat this profile as a stress test.
```

For this experimentation milestone it is acceptable to continue rendering in floating point after warning.

Do not add a limiter to NAM input.

---

# 13. User interface redesign for input profiles

The recent two-column UI is good.

Do not redesign the whole interface again.

Replace:

```text
TEST INPUT GAIN
```

with:

```text
INPUT PROFILE
```

Suggested controls:

```text
Instrument
[ Guitar ▼ ]

Input profile
[ P90 ▼ ]

Relative level
+1.0 dB

NAM calibration
[ Auto ▼ ]

Reference level
+12.0 dBu
```

Keep the +12 dBu field under an Advanced disclosure if possible.

For Guitar profile options:

```text
Vintage / low-output single coil
Standard / hotter single coil
Vintage / PAF humbucker
P90
Medium / modern humbucker
Hot humbucker
Extreme passive
Active / buffered — Custom
Custom
```

For Bass:

```text
Standard Jazz / Precision bass
Modern / hotter passive bass
Very high-output passive bass
Active / preamped — Custom
Custom
```

When Custom/Active is chosen, show:

```text
Relative input gain
[-12 -------- 0 -------- +12]
```

Make it visually obvious that choosing another input profile requires re-rendering the amps.

For example:

```text
Input profile changed — render amps again
```

Disable/mark stale previews until rerender completes.

Changing:

* crossover
* transition
* auto level
* B manual trim

must NOT invalidate the pair.

---

# 14. Coverage analysis across pickup profiles

This is one of the main reasons for doing this work.

Add a lightweight analysis that tells us whether realistic instruments actually reach the intended transition.

This calculation does NOT need to render every NAM profile.

For coverage only:

```text
source DI envelope
    +
profile relative gain
    ↓
blend curve for current crossover/transition
```

is sufficient.

Create something like:

```python
analyse_profile_coverage(
    source_envelope_db,
    profiles,
    crossover_dbfs,
    transition_width_db,
)
```

Use the SAME crossover weighting implementation as the audio path.

Do not duplicate the smoothstep maths in another module if it can be shared.

Report, for each profile:

```text
mostly Amp A
transition
mostly Amp B
```

Use explicit blend-weight boundaries such as:

```text
Amp A region: t <= 0.10
Transition:    0.10 < t < 0.90
Amp B region: t >= 0.90
```

Document these definitions.

Percentages should preferably be based on active playing rather than long stretches of silence.

Before inventing a new silence detector, inspect whether the repo already contains active-signal/silence analysis from the DI classification work and reuse it if suitable.

If a new rule is required, isolate it in one function and document/test it.

Do not let this UI statistic alter the actual audio path.

---

# 15. Show a profile coverage table

Add a compact table/card near the journey graph.

Example:

```text
EXPECTED CROSSOVER COVERAGE

Input profile                 Amp A   Transition   Amp B
-------------------------------------------------------
Vintage single                 78%       19%        3%
Standard single                62%       27%       11%
PAF humbucker                  48%       31%       21%
P90                            43%       32%       25%
Modern humbucker               32%       31%       37%
Hot humbucker                  20%       27%       53%
Extreme passive                12%       22%       66%
```

These numbers are illustrative only.

Compute the real values from the selected DI and current crossover settings.

For active/custom, show the result for the user's entered custom gain.

This table should update cheaply when:

* crossover changes
* transition changes
* custom relative gain changes

It should not require NAM inference.

---

# 16. Flag useless crossover settings

Add simple warnings.

Examples:

If every realistic guitar profile is >95% Amp A:

```text
Crossover is probably too high:
normal guitar output rarely reaches Amp B with this DI.
```

If every profile is >95% Amp B:

```text
Crossover is probably too low:
the hybrid spends almost no time in Amp A.
```

If a profile has at least meaningful representation of A, transition and B, optionally identify it as:

```text
Good crossover coverage
```

Do not claim this means it will sound good.

This is only a reachability/dynamic-range diagnostic.

---

# 17. Suggested crossover from the actual DI

Add a suggested crossover value, but do not call it “optimal”.

Use the active input-envelope distribution.

Prefer percentiles over whole-file RMS.

Expose:

```text
p10
p25
p50
p75
p90
```

for active samples.

A reasonable initial suggestion may be based around approximately p60/p65 of the reference profile's active envelope.

However:

* inspect the actual DI distribution first
* document the exact percentile selected
* keep the user free to override it
* label it “Suggested” not “Best” or “Optimal”

If an existing DI-analysis tool already computes these percentiles, reuse it.

---

# 18. Important distinction: reference profile versus selected profile

Because the bundled DIs were normalized, do not imply:

```text
this waveform really is a PAF
```

The UI should communicate:

```text
Reference assumption:
The selected guitar DI is treated as the 0 dB PAF-like reference.
Pickup profiles simulate relative output around that recording.
```

Likewise for bass:

```text
The selected bass DI is treated as a standard passive J/P reference.
```

This is a simulation convention necessitated by the original DI normalization.

Keep that language in Help/Info rather than cluttering the main controls.

---

# 19. API changes

Extend `/api/render_pair` to accept input-profile/calibration information.

Suggested request:

```json
{
  "amp_a_path": "...",
  "amp_b_path": "...",
  "di_file": "moderate_brit.wav",
  "instrument_type": "guitar",
  "input_profile_id": "p90",
  "custom_input_gain_db": null,
  "calibration_mode": "auto",
  "reference_input_level_dbu": 12.0
}
```

Return useful diagnostics:

```json
{
  "instrument_type": "guitar",
  "input_profile": "p90",
  "input_profile_gain_db": 1.0,

  "input_peak_dbfs": -3.4,

  "calibration_mode": "auto",
  "calibration_applied": true,
  "reference_input_level_dbu": 12.0,

  "amp_a_input_level_dbu": 8.5,
  "amp_b_input_level_dbu": 12.0,

  "amp_a_calibration_gain_db": 3.5,
  "amp_b_calibration_gain_db": 0.0,

  "warnings": []
}
```

Values above are examples only.

Use the actual model metadata.

Add an endpoint for profile definitions if useful:

```text
GET /api/input_profiles
```

and one for coverage if clean:

```text
POST /api/profile_coverage
```

Do not force everything through one giant endpoint.

---

# 20. build_hybrid cleanup

After real input profiles are applied in `render_pair`, `build_hybrid` should go back to doing only:

```text
already-rendered pair
    ↓
optional alignment
    ↓
auto level match
    ↓
manual B tweak
    ↓
smoothstep blend
```

Its envelope should already correspond to the actual selected input profile.

Do not apply another input-gain shift during blending.

This keeps the architecture understandable:

```text
INPUT LEVEL belongs to render stage
CROSSOVER SETTINGS belong to blend stage
```

---

# 21. Auto-level matching under real profile gain

With the new architecture, auto-level matching SHOULD use:

```text
pair.envelope_db
```

because that envelope now represents the actual input profile that was genuinely rendered through both NAMs.

This is different from the old test-only `dry_gain_db` situation.

Test explicitly that changing from:

```text
Vintage single -7 dB
```

to:

```text
Hot humbucker +4.5 dB
```

causes:

* NAM A to be rerendered
* NAM B to be rerendered
* envelope to change
* level-match calculation to use the new profile envelope
* blend coverage to shift naturally toward Amp B

---

# 22. Tests

Keep all existing tests passing.

Add comprehensive tests for:

### Profile definitions

Verify exact preset values:

```text
guitar vintage single = -7.0
guitar standard single = -3.0
guitar vintage humbucker = 0.0
guitar P90 = +1.0
guitar modern humbucker = +2.5
guitar hot humbucker = +4.5
guitar extreme passive = +6.0

bass standard J/P = 0.0
bass modern/hot passive = +1.5
bass high-output passive = +3.5
```

Verify active profiles require custom gain and do not contain a fake universal offset.

### dB conversion

Test:

```python
20 * log10(125 / 250) ~= -6.02
20 * log10(425 / 250) ~= +4.61
```

and amplitude conversion round-trips.

### Real gain application

A +6 dB profile should approximately double sample amplitude before NAM calibration.

A -6 dB profile should approximately halve it.

### Envelope

The envelope stored in `RenderedPair` must be derived from the profile-adjusted dry input.

### Calibration

Test the official formula.

Example:

```text
reference calibration = +12 dBu
model input_level_dbu = +8 dBu
expected model compensation = +4 dB
```

Test:

* both calibrated → compensation applied separately
* both uncalibrated → raw
* only one calibrated → raw for both + warning
* explicit Raw → no compensation

### No double gain

Ensure input-profile gain is not also applied again inside `build_hybrid`.

### Coverage monotonicity

On a synthetic DI:

```text
-7 dB profile
0 dB profile
+4.5 dB profile
+6 dB profile
```

must move monotonically toward higher Amp B coverage for the same crossover.

### Cache invalidation

Changing profile/instrument/calibration must invalidate the old rendered pair.

Changing only crossover/transition/manual trim must not rerun NAM inference.

### Peak warning

A profile that pushes the source DI over 0 dBFS must produce a warning but must not silently clamp or normalize the input.

---

# 23. Manual real validation

Use the real local models already used successfully:

```text
Amp A:
FenderSuperReverb1977_Clean.nam

Amp B:
JCM800_2203_Modified_HighGain.nam
```

Use:

```text
moderate_brit.wav
```

Start with:

```text
alignment OFF
auto level ON
manual trim 0
transition 8 dB
calibration AUTO
reference calibration +12 dBu
```

Test real rendering/listening for at least:

```text
Vintage single       -7.0 dB
PAF humbucker         0.0 dB
P90                   +1.0 dB
Hot humbucker         +4.5 dB
Extreme passive       +6.0 dB
```

For each report:

```text
input peak dBFS
source NAM input_level_dbu values
whether calibration was applied
per-model calibration gain
auto B trim
Amp A %
Transition %
Amp B %
render duration/performance
any clipping/headroom warning
```

Listen to Amp A / Hybrid / Amp B.

Do not claim the hybrid sounds musically successful without human listening.

What we want to establish technically is:

```text
lower-output pickup simulation
        ↓
spends more time in Fender

higher-output pickup simulation
        ↓
drives both NAMs harder
        AND
spends more time in Marshall
```

Both effects must occur.

---

# 24. Bass validation

Bass support is required even if no suitable pair of real bass NAMs is locally available.

At minimum use a bundled bass DI such as:

```text
bass_rollin.wav
```

and verify coverage calculations for:

```text
Standard J/P              0.0 dB
Modern/hot passive       +1.5 dB
High-output passive      +3.5 dB
Active/preamped custom
```

If suitable bass NAM models happen to be locally available, run one real pair.

Do not block completion if they are not.

Do not use guitar NAMs to claim bass tonal validation.

---

# 25. Do not add guitar-volume-knob presets yet

Earlier discussion considered values such as:

```text
volume 10
volume 7
volume 5
volume 3
```

Do NOT implement fixed dB values for these in this milestone.

We have not yet established sufficiently defensible pot-taper / loading / treble-bleed assumptions.

A guitar volume control changes more than pure level in many instruments.

If useful, note guitar-volume simulation as a future research item.

Do not guess the dB values.

---

# 26. Metadata preparation

Do not implement A2 target generation yet, but update the future metadata design so generated models can record:

```text
instrument family
reference input profile
profile gain dB
calibration mode
reference input level dBu
Amp A source input_level_dbu
Amp B source input_level_dbu
per-model calibration compensation
```

The final generated hybrid must be reproducible later.

---

# 27. Documentation

Update README and CLAUDE.md with the new conceptual distinction:

```text
Input profile:
changes the actual signal hitting both NAMs and therefore requires rerendering.

Crossover:
changes only how the already-rendered A/B responses are blended and is cheap.

NAM calibration:
makes differently captured NAMs see the same virtual physical input level when both models provide calibration metadata.
```

Document why active pickups use Custom.

Document why bass is a separate reference family.

Document why the normalized genre DIs cannot provide absolute pickup-output calibration.

---

# 28. Do not implement yet

Still do NOT implement:

* A2 training
* `/api/generate` final target generation
* CLO generation
* GP50 upload
* automatic pickup detection from waveform
* automatic active-pickup gain assumptions
* guitar volume-pot simulations
* phase alignment by default
* React
* Node
* database
* cloud services
* LLM features

The next line is:

**realistic instrument input profiles + NAM calibration + crossover reachability + trustworthy listening.**

---

# 29. Completion criteria

This milestone is complete only when:

1. Research-based passive guitar profiles exist centrally.
2. Bass profiles exist centrally.
3. Active guitar/bass correctly require custom gain rather than a fake universal level.
4. Input-profile gain alters the actual dry audio sent to both NAMs.
5. The crossover envelope uses the same profile-adjusted dry signal.
6. Profile change requires rerender.
7. Crossover/transition change does not rerender.
8. NAM input calibration uses the official compensation formula.
9. Calibration is applied only when both models provide valid metadata in Auto mode.
10. Raw mode remains available.
11. Input peak/headroom is reported.
12. Profile coverage A/Transition/B is shown.
13. Coverage changes sensibly across low→high pickup output.
14. Guitar and bass have separate reference families.
15. The normalized-DI limitation is documented.
16. Existing tests still pass.
17. New profile/calibration/coverage tests pass.
18. Fender→JCM800 is manually exercised across multiple real profile levels.
19. No A2 training is added yet.
20. Changes are committed and pushed.

---

# 30. Final report

When finished, report:

* commit SHA
* changed/new files
* total tests passed/skipped
* exact profile table implemented
* exact source/reference assumptions
* calibration formula implemented
* actual Fender input_level_dbu
* actual JCM800 input_level_dbu
* whether Auto calibration was applied
* per-model calibration compensation used
* crossover/transition used in manual validation
* coverage results for:

  * Vintage single
  * PAF
  * P90
  * Hot humbucker
  * Extreme passive
* bass coverage results
* any profile that caused >0 dBFS input
* any unexpected tonal/output behaviour
* any deviations from this prompt and why

At the end, commit and push the repository.

Do not start A2 training automatically after completing this milestone.

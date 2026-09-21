# UI Spec: Continuous Gain Tab

## Purpose

Add a new top-level tab to NAM Mixer for creating a single NAM from multiple captures of the same amplifier taken at different physical gain settings.

This is separate from the existing Mixer and Builder workflows.

The user is not mixing different amps.

They are providing several snapshots of one amp across its gain range so the application can analyse, interpolate and eventually build a Continuous Gain NAM.

---

# Navigation

Add a new primary tab immediately after Builder.

Example:

```text
Mixer | Builder | Continuous Gain | Analyse | Settings
```

Use the existing tab styling and navigation behaviour.

The new tab should visually feel like part of NAM Mixer, not a separate application.

Working tab label:

```text
Continuous Gain
```

Alternative shorter label if space becomes an issue:

```text
Gain Range
```

Prefer `Continuous Gain` initially because it communicates the purpose better.

---

# Overall Workflow

Follow the same broad flow as the current Builder:

```text
1. Add source material
2. Configure
3. Analyse
4. Review
5. Build
6. Export
```

For Continuous Gain this becomes:

```text
Gain Captures
      ↓
Range Analysis
      ↓
Interpolation / Mapping
      ↓
Validation
      ↓
Build NAM
      ↓
Result
```

The interface should progressively reveal complexity.

Do not present every research option immediately.

The normal workflow should remain simple.

---

# Page Header

Use the same heading hierarchy and spacing as Builder.

Example:

```text
Continuous Gain

Build a single NAM from multiple gain settings of the same amplifier.

Add captures of the same amp with only the physical Gain control changed.
```

Add a small `Experimental` or `Research` badge while the feature is not considered production-ready.

Example:

```text
Continuous Gain    [Experimental]
```

Do not use warning-heavy language.

---

# Section 1: Gain Captures

This replaces the two-source cards used by the existing mixer/builder workflow.

Do not show ten empty cards.

Start with two compact capture rows because two captures are the minimum technically required.

Example:

```text
GAIN CAPTURES                                      2 captures

Gain    NAM Capture
────────────────────────────────────────────────────────────
[ 2.0 ]  Marshall_JCM800_Hi_G2.nam        [Replace]  [×]
[ 8.0 ]  Marshall_JCM800_Hi_G8.nam        [Replace]  [×]

[ + Add Gain Capture ]     [ Add Multiple NAM Files ]
```

Maximum:

```text
10 captures
```

Minimum:

```text
2 captures
```

Recommended:

```text
3+
```

Do not visually present ten fixed slots.

---

# Empty State

On first opening the tab:

```text
GAIN CAPTURES

Add at least two captures of the same amplifier.

[ Choose NAM Files ]

or drag NAM files here

All amp, EQ, cabinet and signal-chain settings should remain the same.
Only change the physical Gain control.
```

Allow multiple file selection immediately.

This should be the easiest route for users who already have a full gain sweep.

---

# Capture Row

Each row contains:

```text
Gain position
NAM filename
optional metadata/status
Replace
Remove
```

Example:

```text
[ 4.0 ]    JCM800_Hi_G4.nam              ✓ Ready     [×]
```

Gain input should:

* accept decimal values
* default to a 0–10 or 1–10 style scale
* be editable directly
* support keyboard arrows
* prevent duplicate gain positions
* sort automatically after editing

Do not require captures to be added in gain order.

If the user enters:

```text
8
2
10
4
6
```

display them as:

```text
2
4
6
8
10
```

---

# Filename Gain Detection

When multiple NAMs are selected, attempt to infer gain positions from obvious filenames.

Examples:

```text
JCM800_G2.nam
JCM800_Gain4.nam
JCM800_G6.5.nam
```

Suggested result:

```text
2
4
6.5
```

Treat this only as a suggestion.

Never silently assume the detected number is correct.

Visually distinguish inferred values until confirmed if necessary.

Example:

```text
[ 4.0 ] Suggested from filename
```

Do not block progress if filename detection fails.

---

# Bulk Add

`Add Multiple NAM Files` should allow selection or drag/drop of several NAMs at once.

After loading:

```text
Gain    NAM Capture
────────────────────────────────────────────────
[   ]   JCM800_G2.nam
[   ]   JCM800_G4.nam
[   ]   JCM800_G6.nam
[   ]   JCM800_G8.nam
[   ]   JCM800_G10.nam
```

Populate gain values automatically where confidence is high.

Focus the first unresolved gain field where detection fails.

---

# Capture Limit

Maximum of 10 captures.

When 10 are loaded:

```text
10 captures · Maximum reached
```

Hide or disable:

```text
+ Add Gain Capture
```

Do not show empty placeholders for unused capacity.

---

# Capture Guidance

Directly beneath the capture list:

```text
Use captures from the same amp, channel, cabinet, mic and signal chain.
Change only the amplifier Gain control.
```

Optionally expandable:

```text
Why?
```

Expanded text can explain that changing EQ, channel or cabinet means the system is no longer learning a controlled gain sweep.

Keep this collapsed by default.

---

# Section 2: Gain Range Visualisation

Once at least two valid captures exist, display a compact range visual.

Example:

```text
GAIN RANGE

0     1     2     3     4     5     6     7     8     9     10
            ●           ●           ●           ●            ●
            G2          G4          G6          G8           G10
```

This should update immediately when gain values change.

Purpose:

* show capture spacing
* make missing areas obvious
* reinforce that these files form one continuum
* later show analysis results

Do not make it an editable slider initially.

Editing remains in the capture rows.

---

# Coverage Indicator

Below the range:

```text
5 captures · Full range covered
```

or:

```text
3 captures · Large gap between Gain 3 and Gain 9
```

Do not use simplistic scores such as:

```text
Excellent
72%
```

unless later supported by actual research.

Use descriptive information only.

Examples:

```text
Minimum capture count met
```

```text
Sparse coverage in upper gain range
```

```text
Capture positions are evenly distributed
```

---

# Section 3: Analyse Range

Primary action after capture setup:

```text
[ Analyse Gain Range ]
```

Disable until:

* at least two captures exist
* every capture has a valid gain position
* no duplicate positions exist
* all files load successfully

Analysis should use the same interaction pattern as Builder analysis where possible.

Show progress in the existing application style.

---

# Analysis Results

After analysis, show a concise summary rather than exposing every research metric immediately.

Example:

```text
GAIN RANGE ANALYSIS

5 captures analysed

Output progression
-23.6 dB ─────────────────────── -18.3 dB

Capture spacing
✓ Good coverage

Input calibration
⚠ No input_level_dbu metadata found
  Raw calibration will be used.

Neighbour consistency
✓ No major anomalies detected
```

Then provide:

```text
[ View Detailed Analysis ]
```

Detailed analysis may show:

* active RMS
* peaks
* input calibration metadata
* spectral differences
* neighbouring ESR
* receptive-field information
* detected anomalies

---

# Analysis Table

Expanded view:

```text
Gain    Active RMS    Peak      Calibration     Status
──────────────────────────────────────────────────────
2       -23.6 dB      ...       Raw             ✓
4       -21.4 dB      ...       Raw             ✓
6       -19.8 dB      ...       Raw             ✓
8       -18.8 dB      ...       Raw             ✓
10      -18.3 dB      ...       Raw             ✓
```

Keep technical detail behind the expandable analysis section.

---

# Section 4: Model Settings

Normal users should see a small number of choices.

Do not expose all research controls by default.

Suggested initial UI:

```text
MODEL SETTINGS

Interpolation
[ Automatic ▼ ]

Output Level
[ Preserve amp behaviour ▼ ]

Input Mapping
[ Automatic ▼ ]
```

Defaults should eventually be based on validated research.

During the research phase, Automatic may simply map to the current best baseline.

---

# Advanced Settings

Add:

```text
▸ Advanced
```

Expanded:

```text
Interpolation Method
○ Linear Output
○ Hybrid
○ Character
○ Experimental

Output Normalisation
○ Preserve
○ Partial
○ Full

Normalisation Amount
[────●────] 50%

Input Mapping Range
Low:   -24 dB
High:   -6 dB

Envelope
Attack:    50 ms
Release:  250 ms
```

Do not show these controls unless Advanced is expanded.

---

# Research Mode

Because the feature is initially being developed as research, optionally expose:

```text
☐ Research / Validation Mode
```

When enabled, reveal controls for:

```text
Leave-one-out validation
Hidden capture selection
Comparison metrics
Generate comparison WAVs
Alternative interpolation methods
```

This prevents research tooling from contaminating the normal future workflow.

---

# Section 5: Validation

After analysis, offer validation before building.

Example:

```text
VALIDATION

Test how accurately the gain range can reconstruct known captures.

[ Run Validation ]
```

For a five-capture set:

```text
Gain 4 reconstructed from Gain 2 + Gain 6
Gain 6 reconstructed from Gain 4 + Gain 8
Gain 8 reconstructed from Gain 6 + Gain 10
```

Results:

```text
Hidden Gain    Neighbours    ESR       Level Error
────────────────────────────────────────────────────
4              2 / 6         0.061     0.8 dB
6              4 / 8         0.056     0.7 dB
8              6 / 10        0.049     0.4 dB
```

Do not make validation mandatory once the technique is mature.

During the research phase it should be strongly encouraged.

---

# Validation Range Visual

Reuse the same gain-range graphic.

Example:

```text
2          4          6          8          10
●──────────●──────────●──────────●──────────●
           ↑          ↑          ↑
         tested     tested     tested
```

Clicking a tested point may show its validation details later.

---

# Section 6: Build

Once analysis is valid:

```text
BUILD CONTINUOUS GAIN NAM
```

Follow the same layout and workflow conventions as the existing Builder tab.

Where possible reuse:

* training configuration presentation
* output naming
* progress state
* logs
* cancel behaviour
* result card
* export/download workflow

Default output name could derive from the shared capture name:

```text
Marshall_JCM800_Hi_ContinuousGain.nam
```

Allow editing before build.

---

# Build Summary

Immediately above the Build button:

```text
BUILD SUMMARY

Amp range             Gain 2 → Gain 10
Source captures       5
Interpolation         Linear Output
Output behaviour      Preserved
Input mapping         Automatic
Validation mean ESR   0.055
```

Then:

```text
[ Build Continuous Gain NAM ]
```

This gives the user a final sanity check without making them revisit every section.

---

# Build Progress

Match Builder.

Example:

```text
Building Continuous Gain NAM

Preparing capture range       ✓
Generating training target    ✓
Training model                63%
Validating output             …
```

Avoid exposing implementation terminology unless useful.

---

# Result

On success:

```text
CONTINUOUS GAIN NAM CREATED

Marshall_JCM800_Hi_ContinuousGain.nam

Gain range:      2 → 10
Source captures: 5
Validation:      completed

[ Save NAM ]
[ Test Model ]
```

If NAM Mixer already has a natural route into another tab for auditioning/testing, include that action rather than creating another audio player.

---

# Warnings

Use inline warnings.

Examples:

### Duplicate gain position

```text
Gain 6
⚠ Another capture already uses this gain position.
```

### Failed capture

```text
JCM800_G6.nam
⚠ NAM could not be loaded.
```

### Metadata mismatch

```text
⚠ Capture metadata differs from neighbouring captures.
```

### Sparse range

```text
⚠ Large gap between Gain 2 and Gain 9.
Interpolation across this region may be less reliable.
```

Warnings should not block the workflow unless the input is technically invalid.

---

# Destructive Actions

Removing one capture should not require confirmation unless it would discard generated work.

Simple row action:

```text
×
```

If removing a capture invalidates analysis:

```text
Range analysis will need to be run again.
```

Mark analysis as stale automatically.

Do not show a modal confirmation for routine capture removal.

---

# State Invalidation

Changes to these values should invalidate previous analysis/validation:

* source NAM
* gain position
* capture addition/removal
* calibration-relevant configuration
* interpolation method
* normalisation method

Show:

```text
Analysis out of date
[ Re-analyse ]
```

Do not silently use stale results.

---

# Recommended Default Screen

The main screen after five files have been added should remain compact:

```text
CONTINUOUS GAIN                                      Experimental

Build one NAM from multiple gain settings of the same amplifier.


GAIN CAPTURES                                        5 captures

Gain    NAM Capture
──────────────────────────────────────────────────────────────
2.0     JCM800_Hi_G2.nam                          [×]
4.0     JCM800_Hi_G4.nam                          [×]
6.0     JCM800_Hi_G6.nam                          [×]
8.0     JCM800_Hi_G8.nam                          [×]
10.0    JCM800_Hi_G10.nam                         [×]

[ + Add Gain Capture ]     [ Add Multiple NAM Files ]


GAIN RANGE

0    1    2    3    4    5    6    7    8    9    10
          ●         ●         ●         ●          ●
          G2        G4        G6        G8         G10

5 captures · Full range covered

All captures should use the same amp and signal chain.
Only change the amplifier Gain control.

                                      [ Analyse Gain Range ]
```

After analysis:

```text
GAIN RANGE ANALYSIS

✓ 5 captures analysed
✓ Smooth gain progression detected
⚠ No input calibration metadata. Raw calibration used.

[ View Detailed Analysis ]


MODEL SETTINGS

Interpolation       Automatic
Output behaviour    Preserve amp behaviour
Input mapping       Automatic

▸ Advanced


VALIDATION

[ Run Validation ]


BUILD

Output name
[ Marshall_JCM800_Hi_ContinuousGain ]

                              [ Build Continuous Gain NAM ]
```

---

# UX Principle

The UI should communicate:

```text
These are measurements of one amplifier control
```

rather than:

```text
These are ten models being mixed together
```

That distinction should drive the layout.

Use:

* ordered compact rows
* one shared gain-range visual
* progressive disclosure
* automatic sorting
* multi-file upload
* filename gain suggestions

Avoid:

* ten cards
* ten permanent slots
* A/B terminology
* per-source blend controls
* large numbers of visible research parameters
* forcing users to understand interpolation mathematics

The normal user's mental model should simply be:

```text
Load captures → tell NAM Mixer where the Gain knob was → analyse → build.
```


Turn the current Continuous Gain research into a production-ready implementation for NAM Mixer.

Base the design on the strongest validated finding so far:

* A normal NAM-style pre-model input-gain control can reproduce much of a physical amp Gain sweep.
* A well-chosen middle anchor can cover a surprisingly wide range.
* A small anchor set such as G1/G5/G10 can match or outperform dense 10-capture interpolation on the tested JCM800 dataset.
* The physical Gain -> virtual input-gain relationship is strongly nonlinear.
* Extreme anchors can fail outside their useful range.
* The system therefore needs automatic anchor selection and per-anchor nonlinear mapping, not hard-coded assumptions.

Do not add more research-only complexity unless required to make the feature safe and reliable.

## Goal

Build a production architecture that can take a set of gain/volume NAM captures and automatically create a compact Continuous Gain profile.

Conceptually:

```text
multiple real NAM captures
        ↓
offline analysis
        ↓
select minimum useful anchor set
        ↓
derive physical-knob -> NAM-input-gain mappings
        ↓
validate reconstruction quality
        ↓
store Continuous Gain profile
        ↓
runtime Gain knob
        ↓
selected anchor NAM
        ↓
pre-model virtual input gain
        ↓
NAM inference
```

The expensive analysis happens during profile creation/import, not during realtime playback.

## Production requirements

### 1. Automated capture validation

Before building a profile:

* verify files load and render correctly
* verify sample rate / format compatibility
* detect duplicate or suspicious captures
* detect large unexpected latency offsets between neighbouring captures
* flag non-monotonic or anomalous captures but do not automatically reject unusual amp behaviour
* specifically prevent a latency defect like the previously discovered ~105-sample offset from corrupting model selection or validation

Where possible, estimate and compensate analysis-only alignment before computing sample-domain metrics.

Do not silently alter the original NAM files.

### 2. Determine capture ordering

Use explicit metadata/filenames/user-supplied gain values where available.

Never infer physical Gain position purely from output level.

Store the real physical/control position independently from virtual input gain.

### 3. Automated anchor selection

Do not assume G1/G5/G10.

Given all available captures, automatically test candidate anchor sets.

At minimum evaluate:

```text
best single anchor
best two-anchor combination
best three-anchor combination
```

Optionally continue to more anchors only if validation thresholds are not met.

For every candidate anchor:

* sweep NAM input gain before inference
* determine the best virtual input gain for each real target capture
* use coarse then fine optimisation
* record usable target range
* detect search-boundary failures
* calculate actual-level and level-matched quality metrics
* keep spectral and level metrics separate

Choose the SMALLEST anchor set that meets the required quality threshold across the sweep.

Avoid exhaustive combinatorial searches if they become expensive. Use a sensible optimisation/search strategy once the exact exhaustive version has been validated on small capture sets.

### 4. Derive the continuous control mapping

For each selected anchor, derive:

```text
physical knob position
    ->
virtual NAM input gain dB
```

Do not assume this mapping is linear.

Fit a smooth monotonic mapping only where supported by the measured data.

Prefer a stable interpolation such as monotonic PCHIP / monotonic spline or another simple bounded curve.

Do not allow the fitted curve to introduce large overshoot between known measurements.

Preserve the measured points so the production mapping can be validated against them.

### 5. Anchor regions

Determine which anchor owns each region of the physical Gain control based on measured reconstruction quality.

For example:

```text
low region    -> anchor A
middle region -> anchor B
high region   -> anchor C
```

Do not base boundaries purely on knob midpoint.

Store:

* anchor NAM
* anchor physical Gain position
* valid physical Gain range
* mapping curve
* confidence / validation metrics for that range

### 6. Anchor transitions

A production Gain knob cannot click or jump when switching anchors.

Implement and validate smooth transitions between anchor regions.

Start with the simplest safe method:

* short crossfade between the outgoing and incoming anchor outputs around the boundary
* both anchors receive their own correctly mapped virtual input gain during the transition

Test for:

* level jumps
* spectral discontinuities
* phase cancellation
* audible comb filtering
* transient artifacts
* CPU spikes

If simple audio crossfading produces unacceptable phase/interference problems, investigate a better transition mechanism, but do not redesign the whole system pre-emptively.

### 7. Runtime architecture

The realtime path must stay simple:

```text
physical Gain knob
        ↓
profile lookup
        ↓
anchor selection
        ↓
mapping lookup/interpolation
        ↓
pre-NAM input gain
        ↓
NAM inference
        ↓
optional anchor transition crossfade
        ↓
output
```

No optimisation searches, rendering comparisons, model training or heavy analysis at runtime.

### 8. Profile format

Create a stable serializable Continuous Gain profile.

Something conceptually like:

```json
{
  "version": 1,
  "control": {
    "name": "Gain",
    "min": 1.0,
    "max": 10.0
  },
  "anchors": [
    {
      "capture": "...",
      "physical_position": 5.0,
      "range": [3.2, 6.8],
      "mapping": [
        [3.5, -4.2],
        [4.0, -2.4],
        [4.5, -1.2],
        [5.0, 0.0],
        [5.5, 1.4],
        [6.0, 2.0],
        [6.5, 5.8]
      ]
    }
  ],
  "validation": {}
}
```

The actual schema should be designed cleanly rather than copying this literally.

Include schema versioning from the start.

### 9. Quality thresholds

Define explicit acceptance criteria.

Do not choose arbitrary thresholds without referencing the distributions observed in the existing research.

A profile should be classified as something like:

```text
validated
acceptable
poor / insufficient captures
```

based on measured reconstruction quality.

If the requested quality cannot be reached with the available captures:

* do not pretend the profile is accurate
* identify the physical Gain region with the highest error
* recommend an additional capture in that region

This links the production system back to the previous capture-density research.

### 10. Minimum-capture discovery

The builder should answer:

```text
How few NAM captures can reproduce this physical Gain sweep
within the chosen quality tolerance?
```

Workflow:

```text
try 1 anchor
    ↓
passes? -> use 1

otherwise try 2
    ↓
passes? -> use 2

otherwise try 3
    ↓
...

otherwise recommend additional captures
```

The goal is not to force three anchors.

The goal is the minimum validated set for that particular amplifier.

### 11. Validation

The production implementation must be tested against real withheld captures.

For the existing dense JCM800 dataset:

* build the profile using integer captures
* validate against half-step captures
* reproduce the existing virtual-gain benchmark as a regression test

Ensure the implementation retains approximately the same results as the research harness.

Then add the Fender Super-Sonic and Fender 57 Twin as generalization tests when practical.

Do not assume the JCM800 result generalizes automatically.

### 12. Automated regression tests

Add tests for:

* 0 dB input gain through an anchor equals direct anchor NAM rendering
* input gain is definitely applied before NAM inference
* mapping interpolation passes through known measured points
* anchor switching cannot exceed expected level discontinuity
* malformed profile rejection
* missing NAM file handling
* out-of-range physical Gain clamping
* latency-offset detection
* deterministic profile generation from identical inputs
* profile serialization/deserialization round trip
* regression against the known JCM800 benchmark

### 13. Performance

Measure:

* realtime CPU cost
* memory cost
* anchor-transition cost
* profile-generation time

Profile building can be relatively expensive.

Realtime use cannot be.

Avoid keeping unnecessary NAM models loaded simultaneously outside transition regions if this materially affects memory usage.

### 14. UX

Expose the feature initially as:

```text
Continuous Gain
Experimental
```

The user should provide/select a capture sweep and physical Gain values.

The analysis should then report:

* number of captures supplied
* selected anchors
* usable control range
* validation quality
* problem regions
* recommended additional captures, if required

Do not expose technical details such as ESR unless an advanced/debug view is enabled.

The normal UI should simply say whether the generated profile is high confidence, acceptable, or needs more captures.

### 15. Keep fallback compatibility

Do not remove the current discrete interpolation path.

Continuous Gain profiles should be able to use:

```text
virtual-gain anchor mode
```

or fall back to:

```text
discrete interpolation
```

when the virtual-gain approach cannot meet the quality threshold.

Design the interface so other strategies can be added later without rewriting the UI/runtime control layer.

### 16. Scope boundaries

Do NOT:

* restart the conditional-NAM architecture work
* assume physical knob position is electrically linear
* collapse spectral and level behaviour into one universal response axis
* hard-code G1/G5/G10
* hard-code JCM800-specific mappings
* train new neural models
* remove existing interpolation support
* treat one amp's success as universal proof

## Deliverables

Implement the production-oriented Continuous Gain foundation and document it.

Produce:

1. production architecture
2. profile schema
3. offline profile builder
4. automatic anchor selection
5. nonlinear gain-mapping generation
6. validation and anomaly detection
7. runtime anchor/mapping engine
8. safe anchor transitions
9. fallback to existing interpolation
10. automated tests
11. JCM800 regression benchmark
12. documentation explaining profile creation and runtime behaviour

Create/update an implementation document such as:

`docs/CONTINUOUS_GAIN_PRODUCTION.md`

At the end, report:

* what was implemented
* remaining experimental components
* current known limitations
* benchmark results
* whether the JCM800 production implementation reproduces the research result
* what must be validated on another amplifier before removing the Experimental label

The production principle is:

**Use the smallest number of real NAM anchors that can reproduce the measured physical Gain sweep within a validated quality threshold, using ordinary pre-model NAM input gain and a measured nonlinear control mapping.**

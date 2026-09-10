# NAM Mixer

**Blend two [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) captures into one new amp — playable live, and trainable into a single standalone `.nam` model.**

Ever wanted a clean Fender that opens up into a Marshall crunch the harder you
dig in, without switching presets? NAM Mixer renders two of your own `.nam`
captures through the same real NAMCore inference engine used by the official
plugin, blends the results using one of three design modes, and lets you
train the blend into a brand-new NAM model that stands on its own — no
switching, no source models required at inference time.

Everything runs locally as a small Flask app with no build step: your amp
captures, DI files, and generated training material never leave your
computer unless you explicitly send a training job to your own Kaggle
account for free GPU time.

**Status: Beta. It renders source NAMs through real NAMCore inference and
generates trainable A2 bundles end to end — listen to and validate every
generated model before you trust it.**

## Why this exists

Splitting your tone across two amp captures — a clean voice for verses, a
driven one for choruses — usually means a preset switch, a MIDI patch change,
or an A/B/Y box. NAM Mixer instead builds a **continuous, level-driven
transition** between two captures, so the amp itself responds to how hard you
play, the way a real tube amp's character shifts with picking dynamics. It
then lets you freeze that behavior into one trained model, so the hybrid
becomes something you can load anywhere a NAM runs.

## Features

| | |
| --- | --- |
| 🎚️ **Three design modes** | **Dynamic Hybrid** — level-driven crossfade between Amp A and Amp B. **Parallel Blend** — constant-ratio mix of both amps, independent of playing level. **Character Blend** — a deterministic teacher built from one drive-selected donor plus measured tone/feel corrections. |
| 🎧 **Real NAM inference** | Amps are rendered through the same [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) C++ engine the official plugin uses — not a reimplementation, not an approximation. |
| 🔊 **Live audition** | A/B the two source amps against the blended result over a bundled library of real guitar/bass performances, with input-profile simulation, test gain, coverage analysis, and an optional cabinet IR — all without re-running inference on every tweak. |
| 🧠 **Train a real model** | Generate a proper training bundle from the official NAM training excitation and train a new A2 model on it — locally, or with one click on a private Kaggle GPU. |
| ✅ **Built-in validation** | Every exported model is re-rendered and compared against its frozen teacher (ESR, RMS, peak, quiet-response) so you know exactly how close the trained model landed, not just that training finished. |
| 💾 **Sessions library** | Save, reload, import/export, and revisit designs and completed models without losing your place. |
| 🔒 **Local-first & private** | No cloud dependency, no telemetry, no account required. Kaggle training is fully opt-in and uses your own credentials and quota. |

The app never claims that a completed training run sounds identical to its
teacher. Always listen to and validate exported models.

## Contents

- [Why this exists](#why-this-exists) · [Features](#features)
- [What this is](#what-this-is) · [This is NOT model-weight merging](#this-is-not-model-weight-merging)
- [How the blend works](#why-the-dry-inputs-level-controls-the-transition):
  [dry-input level](#why-the-dry-inputs-level-controls-the-transition),
  [level matching](#why-automatic-level-matching-is-needed),
  [input profile vs. crossover vs. calibration](#input-profile-vs-crossover-vs-nam-calibration--three-separate-knobs)
- [Preview DIs vs. training material](#preview-dis-vs-nam-training-material--an-important-distinction)
- [Workflow](#workflow) · [Sessions](#sessions) · [Design modes & Cabinet IR](#design-modes-and-the-shared-cabinet-stage)
- [macOS quick start](#macos-quick-start) · [Running it](#running-it)
- [Current status and limitations](#current-status-and-limitations)
- [Safety: training target vs. live preview](#safety-training-target-vs-live-preview)
- [Project layout](#project-layout) · [NAM Tools](#nam-tools-output-volume-and-metadata)
- [License and attribution](#license-and-attribution)

## What this is

NAM Mixer is a small local tool for building a **dynamic transition,
parallel blend, or deterministic character blend** from two existing
[Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) (NAM)
amp captures — for example, moving from a Fender clean model toward a Marshall
crunch model as you play harder. It generates a trainable target, then can
train and validate a single resulting NAM A2 model that reproduces the chosen
design without requiring the two source models at inference time.

## This is NOT model-weight merging

We are **not** averaging, interpolating, or otherwise combining the weights of
two NAM neural networks. That's a different (and much harder, largely unsolved)
problem — two independently-trained networks don't share a latent space, so
blending their weights doesn't produce anything musically meaningful.

Instead, the approach is:

1. **Render** the same input audio through Amp A and through Amp B separately
   (two ordinary, unmodified NAM inference passes).
2. **Blend the resulting audio**, sample-by-sample, using a crossfade curve
   driven by the *level of the original dry input* — quiet input leans toward
   Amp A's render, loud input leans toward Amp B's render, with a smooth
   transition in between.
3. That blended audio becomes a **synthetic training target**.
4. **Eventually**, train a fresh NAM A2 model against that synthetic
   input/output pair, producing one model that reproduces the hybrid behavior
   directly — at that point the two source `.nam` files are no longer needed at
   inference time.

Step 4 is available through the UI for either Kaggle GPU or local training.
Every generated model still needs listening and validation; the tool does not
make a claim of perceptual equivalence merely because training completes.

## Why the *dry input's* level controls the transition

The crossover has to be driven by something that exists independently of which
amp is currently being auditioned — otherwise there's a chicken-and-egg problem
(you'd need to know the blend to compute the blend). The dry guitar signal
before either amp model touches it is the one signal that's the same regardless
of which amp(s) you render it through, so it's the natural, stable control
signal for "how hard is the player driving this right now."

Concretely: `level = envelope(dry_input)`, and that per-sample level (in dBFS)
is what decides the Amp A/Amp B mix weight, not the loudness of either
processed output. See `hybrid/envelope.py`.

## Why automatic level matching is needed

Two independently-trained/captured NAM models will almost never happen to sit
at the same loudness for a given input level. If Amp B is simply louder than
Amp A at the crossover point, the transition will sound like a volume jump
rather than a change in amp character — exactly the artifact we're trying to
avoid. Rather than normalizing the two amps' *overall* loudness (which doesn't
guarantee anything about how they compare specifically **at the crossover
point**, where it actually matters), this project measures each amp's loudness
using only the portion of the render that falls near the chosen crossover level
and computes a trim from that. See `hybrid/level_match.py`.

## Input profile vs. crossover vs. NAM calibration — three separate knobs

It's easy to conflate these; they operate at different stages and have very
different costs:

- **Input profile** (`hybrid/input_profiles.py`, `render_pair()`): simulates a
  different instrument/pickup driving the signal chain *before* it reaches
  either NAM. This changes the actual audio both amps receive, so changing it
  is EXPENSIVE — it requires re-running NAM inference for both amps. See
  `docs/INPUT_PROFILE_RESEARCH.md` for the research behind the presets and
  why active pickups deliberately have no fixed preset.
- **Crossover / transition width / manual trim** (`build_hybrid()`): changes
  only how the *already-rendered* Amp A/B responses are blended together.
  This is CHEAP — pure numpy, no NAM inference, safe to recompute on every
  slider move.
- **NAM input calibration** (`hybrid/calibration.py`): when both `.nam`
  captures report their own recording calibration (`input_level_dbu`),
  applies the official NAM plugin's per-model compensation formula so two
  differently-calibrated captures see the same virtual physical input level.
  This also happens at render time (it's folded into `render_pair()`
  alongside the input profile), but it's a distinct concept from "how hard is
  the (virtual) instrument driving the amps" — one is about the instrument,
  the other is about reconciling two amp captures' own assumptions about
  their input level.

## Why the genre/style DI files are included

`assets/di/` contains real recorded DI (direct input) guitar/bass performances,
copied from the [NAMtoClo](https://github.com/Goaltoday/NamtoClo) project (see
`assets/di/README.md` for exactly which files, where they came from, and their
measured characteristics). They exist so you can **audition** a hybrid
crossover on real playing dynamics — palm-muted riffing, clean chords, dynamic
picking, etc. — without needing your own guitar/interface set up, and so the
project has a consistent, versioned set of fixtures for regression testing the
blend engine itself.

## Preview DIs vs. NAM training material — an important distinction

```text
genre/style DI (assets/di/*.wav)
    ↓
preview, auditioning, crossover analysis,
automatic level-match testing, regression testing

official NAM training input (a proper calibrated reamp/DI signal, e.g. the
kind of stimulus NAMtoClo itself uses as nam_input_wav.wav — NOT included here)
    ↓
Amp A render
Amp B render
    ↓
dynamic blend
    ↓
synthetic training output
    ↓
A2 training
```

The genre DIs are **musical performances** — useful for judging how a hybrid
*sounds* and behaves, but not designed to exercise the amp's full frequency/
level response the way a proper reamping signal is. The **actual** synthetic
training pair used to eventually train an A2 model must come from a real NAM
training/reamping signal, not from `assets/di/`. This distinction is
deliberate and should not be blurred — see `assets/di/README.md` for more.

## Workflow

The interface is organised into four stages. You can move between them at any
time; the same sources, preview material, cabinet, output safety, and training
tools remain shared across all three design modes.

1. **Sources** — select Amp A and Amp B `.nam` files, a preview DI, an
   instrument/input profile, and (when needed) advanced NAM calibration or
   per-amp input trims. **Render Amps** runs the two expensive NAM inference
   passes.
2. **Shape** — choose Dynamic Hybrid crossover/transition, Parallel Blend mix,
   or Character Blend tone/feel/drive controls, then set level matching. These
   controls recombine the cached pair and do not re-run NAM inference.
3. **Listen** — compare Amp A, the current result, and Amp B against a chosen
   genre DI, using an input profile to simulate different pickups/output
   levels (DESIGN/preview context only — see "Input profile vs. crossover vs.
   NAM calibration" above). The genre DI is a convenience audition
   performance, not engineered to hit every level a real player might reach
   — use **Test gain** (an additional real gain on top of the profile,
   applied before both amps render) to deliberately push the level up/down
   and stress-test the crossfade beyond whatever that clip's own dynamics
   happen to cover. Test gain automatically re-renders the pair after you stop
   dragging. Cabinet and output-gain controls are optional finishing tools;
   Dynamic Hybrid analysis is available on demand.
4. **Create & train** — upload the official NAM training excitation, then "Generate
   Training Bundle" — this freezes the current design into an immutable
   `HybridDesign` (`hybrid/design.py`) and blends the *official*
   training input (not the preview DI, and not with the input-profile gain
   applied — the profile only shaped *design/preview*, never the actual
   training excitation) through Amp A/Amp B with that frozen design
   (`hybrid/training_target.py`). Produces `input.wav`, `hybrid_target.wav`
   (+ `hybrid_target_raw.wav` for comparison), `hybrid.hybrid.json`, and
   `training_manifest.json` under `work/a2/<design_id>/`.
   Then train a real A2 (PackedWaveNet) model on the bundle, either:
   - **Kaggle GPU** (recommended, one-click from the "Create A2" card after a
     one-time `pip install kaggle && kaggle auth login` — see
     `docs/kaggle_training.md`), which trains on a private Kaggle T4 GPU and
     downloads/verifies the result automatically, or
   - **Local**, with `scripts/train_a2.py` (needs a separate Torch/
     `neural-amp-modeler` environment — see `requirements-training.txt` /
     `scripts/setup_a2_env.ps1` — never the app's own Python environment).

   Both paths validate the exported `.nam` identically: loading and
   rendering its Full and Lite branches through the existing native NAMCore
   renderer, recording raw/gain-normalized ESR, and checking quiet response
   against an equivalent processed reference when one is available. Training
   completion and technical validation quality are reported separately.

## Sessions

The **Sessions** tab is a project library, not a popup. Use it to save the
current controls under a name, load or inspect an earlier design, export a
portable NAM Mixer JSON file, import one, or delete a session. Generated
training bundles are also saved as sessions automatically so their completed
NAM can be downloaded again or opened in **Tools**. Validation reports are
bound to the completed NAM's SHA-256; editing the NAM invalidates that report.
A completed bundle also offers a synchronized Teacher/Full/Lite comparison on
the selected musical DI at normal or quiet input level. It reconstructs the
teacher from the saved design, never from current controls.

A session restores the selected settings and app-managed NAM/cabinet file
references, but it deliberately does not render automatically. After loading,
use **Render Amps** to rebuild the pair and verify that the referenced files
are still available. Training manifests under `work/a2` are separate from
sessions and are not interchangeable with session JSON files.

## Design modes and the shared Cabinet stage

The workflow above describes **Dynamic Hybrid** mode, the original/default
mode. Two further modes are available from the same mode selector:

- **Dynamic Hybrid** (`hybrid/blend.py`, `hybrid/design.py`,
  `hybrid/training_target.py`): changes from Amp A toward Amp B according to
  playing level, via the crossover/transition envelope described above.
- **Parallel Blend** (`hybrid/fixed_blend.py`, `hybrid/blend_training_target.py`):
  always combines the two amp responses at one constant, user-chosen ratio
  (`result = A * (1 - mix_b) + B * mix_b`), independent of playing level —
  no crossover envelope at all. Its own auto level-match uses the DI's
  ACTIVE playing material (silence excluded) rather than a crossover band,
  since there's no crossover region to match around (see
  `hybrid.fixed_blend.compute_active_trim`).
- **Character Blend** (`hybrid/character_blend.py`): uses a level-selected
  nonlinear donor plus measured EQ and compression corrections to produce a
  deterministic teacher design. Tone, Feel, and Drive are not a simple
  parallel waveform mix; Drive selects one donor at a time and can optionally
  vary by input level.

All modes share Amp A/Amp B, the preview DI, input profile/calibration,
render, test gain, the Listen controls, the Cabinet IR stage, the official
training input, A2 quality, and training — switching tabs never re-runs NAM
inference; the already-rendered `RenderedPair` (`hybrid/pipeline.py`) is
reused by whichever mode you're auditioning.

A third, mode-independent stage — **Cabinet IR** (`hybrid/cab_ir.py`) — sits
AFTER the amp combination in any mode: an ordinary causal FIR convolution,
optionally auditioned in preview only, or "baked" into the generated A2
training target. Preview-only and baked processing always go through the
exact same `apply_cab_ir` function on the COMPLETE prepared IR (never a
shortened one), so what you hear in preview with "Use cab in preview"
checked is exactly what gets trained if you also check "Bake cab into A2".

**Receptive-field policy: one hard check, two advisory ones.** Amp A/Amp B
(+, for Hybrid and Character, the bounded crossover envelope) are the CORE
dependency — this must fit inside the destination A2's actual receptive
field, or generation/training is refused exactly as before. Two further
dependencies are always calculated and reported honestly, but never block
training by themselves, because in both cases the A2 is being trained to
*approximate* the rendered teacher rather than to compile its signal graph
exactly:

- **Character processing.** Character Blend's donor transitions, smoothing,
  and correction filters add their own formal temporal dependency
  (`formal_character_required_samples`). If it exceeds the A2's receptive
  field, training still proceeds as an approximation — the Full/Lite export
  and quiet-response checks stay authoritative for judging the result.
- **Baked cabinet.** A baked cabinet adds `len(ir) - 1` samples of *serial*
  temporal dependency on top of the core (see `hybrid/receptive_field.py`'s
  `combine_required_history`). A baked cab whose formal total exceeds the
  A2's receptive field does not block training either — the A2 learns an
  approximation of the post-cab response within its available capacity.

Either case prints an explicit "CABINET APPROXIMATION" or "CHARACTER
APPROXIMATION" notice from `scripts/train_a2.py`/the Kaggle cloud worker —
validate the result by listening and by checking the printed ESR/RMS
metrics against the baked target. The exact same full-length cabinet IR is
used for preview and for baking in either case; only the training-time
gating differs. Local and Kaggle training implement this policy
independently but are kept from silently diverging by
`tests/test_receptive_field_parity.py`.

Because raw WAV/FIR length is a poor proxy for how much of a captured IR is
actually audible signal, the Cabinet card also reports cumulative-energy
diagnostics (e.g. "99.9% energy by: 42.7 ms" for a nominally-500ms IR) —
purely informational, never used to shorten the actual convolution.

## macOS quick start

Requires Python 3.10+, Xcode Command Line Tools, CMake 3.18+, and internet
access the first time the native renderer downloads NAMCore.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cmake -B native/nam_render/build -S native/nam_render -DCMAKE_BUILD_TYPE=Release
cmake --build native/nam_render/build --config Release --target nam_render -j 4
python3 app.py
```

Open <http://127.0.0.1:5000>. The app discovers the resulting
`native/nam_render/build/nam_render` automatically. Local A2 training still
needs the separate training environment documented in `requirements-training.txt`;
Kaggle training is available from the UI after authenticating the Kaggle CLI.
If macOS has reserved port 5000 (for example, AirPlay Receiver), run
`PORT=5001 python3 app.py` and open <http://127.0.0.1:5001>.

## Current status and limitations

### Implemented

- **Real NAM inference:** `hybrid/render.py` calls the native `nam_render`
  C++ CLI in `native/nam_render/`, built against
  [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).
  This keeps inference independent of Python `torch`/`neural-amp-modeler` and
  lets NAMCore remain authoritative for `.nam` model loading. Build steps are
  in `native/nam_render/README.md`.
- **`.nam` loading and calibration:** `hybrid/nam_loader.py` reads model
  metadata, including `input_level_dbu` and `output_level_dbu` when present.
  Older or uncalibrated files remain usable and are reported as having
  unavailable calibration metadata.
- **The complete app workflow:** the Flask app and browser UI connect real
  rendering, preview, coverage analysis, all three design modes, cabinet IR,
  Sessions, target generation, local training, and optional private Kaggle
  training. The main routes include `/api/render_pair`, `/api/preview`,
  `/api/blend_info`, `/api/blend_curve`, `/api/input_profiles`,
  `/api/profile_coverage`, and `/api/generate`.
- **Renderer readiness:** the app performs a cheap `--help` launch check before
  preview inference, distinguishes missing from unusable executables, shows
  expandable setup help, and can retry after installation without a restart.
- **Mechanical safeguards:** envelope following, blend math, level matching,
  optional alignment, receptive-field checks, NaN/Inf and silence checks, and
  non-limiting training-target peak control are implemented and covered by
  automated tests against synthetic signals.

### Important validation boundaries

- The Phase 4 real-render harness passed with two distinct local source models,
  two musical DIs, all three design modes, cabinet on/off, and input levels of
  -12/0/+12 dB. It produced deterministic rerenders and exact frozen-teacher
  equivalence for that run. Auto-calibrated cases were unavailable because
  those source files did not declare input-level metadata. Most automated
  pipeline tests still use mocked renders. Other real source-model
  combinations can still expose latency, calibration, or musical problems;
  listen to every preview and validate every exported model against its target.
- **A/B alignment is deliberately off by default.**
  `hybrid/align.py` cross-correlates the two rendered signals, so a tonal or
  phase difference between dissimilar amps can look like latency. Enable it
  only when the timing behavior of the source models is known.
- Character Blend is a deterministic teacher design, not a perceptual-match
  guarantee. No automated system judges tone, feel, or musical quality; the
  checks can only catch mechanical problems such as clipping, discontinuities,
  silence, or non-finite samples.
- Local A2 training needs the separate environment described in
  `requirements-training.txt`. Kaggle training needs a configured Kaggle
  account and network access; neither is required to run the preview UI.

## Safety: training target vs. live preview

The generated hybrid **training target must never be run through a limiter** —
that would distort the very dynamic behavior we're trying to capture. If the
generated hybrid exceeds a target peak ceiling (default -3 dBFS), a single
fixed gain reduction is applied to the whole file instead
(`hybrid.safety.apply_peak_ceiling`). A limiter (`hybrid.safety.
preview_safety_limiter`) exists only as a speaker/headphone safety net on the
live preview/playback path and must never touch a file destined to become (or
derive) a training target.

## Project layout

```text
hybrid-nam-builder/
├── app.py                 -- Flask entry point
├── requirements.txt
├── hybrid/                -- core library (no Flask/UI dependencies)
│   ├── nam_loader.py       -- parse .nam files + calibration metadata
│   ├── render.py           -- NAM inference (shells out to native/nam_render)
│   ├── envelope.py         -- dry-input level/envelope extraction
│   ├── level_match.py      -- crossover-region auto level-match trim
│   ├── align.py            -- sample-offset detection/correction (optional, off by default)
│   ├── blend.py            -- the dynamic crossfade itself
│   ├── fixed_blend.py      -- Parallel Blend mode (fixed-ratio combination)
│   ├── character_blend.py  -- Character Blend teacher and low-level check
│   ├── cab_ir.py           -- shared cabinet IR convolution (preview + baked target)
│   ├── training_target.py  -- Dynamic Hybrid A2 training-target generation
│   ├── blend_training_target.py -- Parallel Blend A2 target generation
│   ├── character_training_target.py -- Character Blend A2 target generation
│   ├── receptive_field.py  -- mode/cab-aware temporal-dependency accounting
│   ├── validation.py        -- frozen-teacher/model rendering and shared metrics
│   ├── validation_report.py -- versioned Full/Lite/quiet quality reports
│   ├── kaggle_training.py  -- private Kaggle GPU job and local validation
│   ├── nam_tools.py        -- safe output-volume and metadata editing
│   ├── wizard.py           -- guided setup flow shared by the UI modes
│   ├── safety.py           -- NaN/clip checks, non-limiting peak ceiling
│   └── metadata.py         -- hybrid provenance metadata (JSON sidecar)
├── native/nam_render/      -- C++ NAM inference tool (NeuralAmpModelerCore), see its README
├── assets/nam_models/      -- user's own .nam amp captures (gitignored)
├── assets/di/              -- genre/style DI library + its own README
├── templates/, static/     -- minimal HTML/CSS/JS UI (no build step)
├── tests/                  -- unit tests for the hybrid/ modules
├── scripts/analyze_di.py   -- regenerates assets/di/_analysis.json
└── work/                   -- gitignored scratch output directory
```

## Running it

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000/
```

```bash
python3 -m pytest -q
# `pytest -q` works too; pytest.ini resolves the repository modules.
```

The test suite exercises `hybrid/envelope.py`, `hybrid/blend.py`,
`hybrid/level_match.py`, `hybrid/align.py`, `hybrid/safety.py`,
`hybrid/nam_loader.py`, `hybrid/input_profiles.py`, `hybrid/calibration.py`,
`hybrid/coverage.py`, `hybrid/pipeline.py`, `hybrid/design.py`,
`hybrid/training_target.py`, `hybrid/fixed_blend.py`,
`hybrid/blend_training_target.py`, `hybrid/character_blend.py`,
`hybrid/character_training_target.py`, `hybrid/cab_ir.py`,
`hybrid/receptive_field.py`, `hybrid/a2_training_settings.py`, and
`hybrid/kaggle_training.py` against synthetic signals and mocked renders —
none of it requires torch, `neural-amp-modeler`, or the native `nam_render`
tool to be built. `tests/test_render.py` exercises real NAM inference and is
skipped automatically unless both `native/nam_render` has been built (see its
README) and a real `.nam` file is present at
`assets/nam_models/FenderSuperReverb1977_Clean.nam`.
`tests/test_receptive_field_parity.py` loads the local and Kaggle trainers
side by side and fails the suite if their receptive-field policies ever
diverge, rather than letting that drift go unnoticed.

For repeatable opt-in real-render coverage across all three design modes, use
the explicit harness rather than relying on that personal-model test:

```bash
python3 scripts/real_render_regression.py \
  --amp-a assets/test_cabs/Clean_NoCab_Fender_Deluxe_Reverb_Head_2.nam \
  --amp-b 'assets/test_cabs/HighGain_NoCab_SLASH AFD#2 Head.nam' \
  --di assets/di/high_thrash.wav --di assets/di/clean_smooth.wav \
  --cab 'assets/nam_models/V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav' \
  --out-dir work/real-render-regression --release
```

It writes listenable stems and `real_render_report.json` beneath ignored
`work/`; `--release` fails if prerequisites are missing or a checked case is
invalid/silent. The Phase 4 implementation was exercised with this command in
release mode: 6 cases and 36 listenable artifacts passed, with no invariant or
determinism failures. Calibration combinations remain explicitly unavailable
when both source models lack `input_level_dbu` metadata.

`torch`/`neural-amp-modeler` are no longer in `requirements.txt` — inference
is handled entirely by the native `nam_render` tool now. They only matter for
local A2 training, and live in `requirements-training.txt`
instead (install into a separate, supported-Python-version environment when
you get to that step).

## Relationship to NAMtoClo

This project reuses the genre/style DI WAV files bundled in the
[NAMtoClo](https://github.com/Goaltoday/NamtoClo) fork this repository's author
also maintains, purely as audio fixtures (see `assets/di/README.md`). It shares
no code with NAMtoClo and solves a completely different problem — NAMtoClo
converts a single existing NAM model to Valeton hardware's CLO format; this
project builds new hybrid NAM training material out of two existing NAM
models. Nothing here depends on NAMtoClo at runtime.

## NAM Tools: output volume and metadata

The **Tools** tab can open a NAM from your computer or select the latest model
generated locally by this app. It always creates a new download and never
overwrites the source file.

The output-volume tool applies `10 ** (dB / 20)` to the recognised final audio
`head_scale`. `head_scale` is the output scale after the model has generated
its signal, so this changes output level without retraining or changing the
learned tone, distortion, dynamics, or input response. It also updates each
available `metadata.loudness` by the same dB amount. It deliberately does not
change `metadata.gain`: that describes separate metadata/calibration intent,
not the final output scale.

Modern A2 `SlimmableContainer` NAMs have one final audio model per submodel;
the tool edits only `config.submodels[*].model.config.head_scale`. Older
single-model files are supported only when their root `config.head_scale` is
present. Unknown layouts are refused rather than guessed. Before saving, a
recursive JSON diff must match exactly the approved output-scale and loudness
paths; weights and every other model field are therefore unchanged.

Examples: enter `+3`, `+6`, or `-6` in the Output volume field. A +6 dB change
uses a multiplier of approximately `1.995262`; -6 dB uses `0.501187`. Boosts
above +12 dB are allowed but may clip in a host or target hardware.

The same safe operation is available from a terminal:

```sh
python3 nam_volume.py Mesa_Boogie.nam +3
python3 nam_volume.py Mesa_Boogie.nam +6 --dry-run
python3 nam_volume.py Mesa_Boogie.nam -6 --output Mesa_Boogie_quieter.nam
```

The metadata editor loads and can safely change every official NAM A2
UserMetadata field: name, modeled-by, gear type/make/model, tone type, and
input/output dBu. Export date, trainer details, and measured loudness remain
untouched; use the Output volume slider for loudness. New NAMs identify their
creator as `NAM Mixer`; physical gear make/model and tone metadata are left
blank until the user can provide factual values.

## License and attribution

NAM Mixer is copyright © 2026 Andrzej Marczewski and is released under the
[MIT License](LICENSE). The bundled genre/style DI recordings are credited and
documented separately in [`assets/di/README.md`](assets/di/README.md); their
upstream terms continue to apply. Neural Amp Modeler, NAMCore, and other
third-party components retain their own copyrights and licenses.

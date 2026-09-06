# Hybrid NAM Builder

**Status: experimental proof-of-concept. Not a finished tool. NAM inference is not
yet wired in — see "Current limitations" below.**

## What this is

Hybrid NAM Builder is a small local tool for building a **dynamic transition
between two existing [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)
(NAM) amp captures** — e.g. a Fender clean model that gradually morphs into a
Marshall crunch model as you play harder, then eventually into a JCM800 at full
tilt. The goal is a single resulting NAM model that behaves like one amp with
a wide, dynamic gain range, built out of two (or more, eventually) amps that
already exist as separate `.nam` captures.

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

Step 4 (actually training the A2 model) is the long-term goal, not something
this proof-of-concept does yet. What exists today builds up to steps 1–3.

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

1. Select Amp A `.nam` and Amp B `.nam`.
2. Choose the crossover point (dBFS) and transition width (dB) — how loud the
   input needs to get before Amp A hands off to Amp B, and how gradual that
   handoff is.
3. Auto level-match trims Amp B to Amp A around the crossover region; override
   manually if desired.
4. Preview Amp A alone, Amp B alone, and the hybrid blend against a chosen
   genre DI, using an input profile to simulate different pickups/output
   levels (DESIGN/preview context only — see "Input profile vs. crossover vs.
   NAM calibration" above). The genre DI is a convenience audition
   performance, not engineered to hit every level a real player might reach
   — use **Test gain** (an additional real gain on top of the profile,
   applied before both amps render) to deliberately push the level up/down
   and stress-test the crossfade beyond whatever that clip's own dynamics
   happen to cover. Both require clicking Render Amps (real NAM inference).
5. **Create A2**: upload the official NAM training excitation, then "Generate
   Training Bundle" — this freezes everything from steps 2-4 into an
   immutable `HybridDesign` (`hybrid/design.py`) and blends the *official*
   training input (not the preview DI, and not with the input-profile gain
   applied — the profile only shaped *design/preview*, never the actual
   training excitation) through Amp A/Amp B with that frozen design
   (`hybrid/training_target.py`). Produces `input.wav`, `hybrid_target.wav`
   (+ `hybrid_target_raw.wav` for comparison), `hybrid.hybrid.json`, and
   `training_manifest.json` under `work/a2/<design_id>/`.
6. Train a real A2 (PackedWaveNet) model on the bundle, either:
   - **Kaggle GPU** (recommended, one-click from the "Create A2" card after a
     one-time `pip install kaggle && kaggle auth login` — see
     `docs/kaggle_training.md`), which trains on a private Kaggle T4 GPU and
     downloads/verifies the result automatically, or
   - **Local**, with `scripts/train_a2.py` (needs a separate Torch/
     `neural-amp-modeler` environment — see `requirements-training.txt` /
     `scripts/setup_a2_env.ps1` — never the app's own Python environment).

   Both paths validate the exported `.nam` identically: loading and
   rendering it (Full and Lite) through the existing native NAMCore renderer
   and comparing against the training target.

## Design modes: Dynamic Hybrid vs. Fixed Blend, and the shared Cabinet stage

The workflow above describes **Dynamic Hybrid** mode, the original/default
mode. A second mode, **Fixed Blend**, is available as a separate tab in the
UI:

- **Dynamic Hybrid** (`hybrid/blend.py`, `hybrid/design.py`,
  `hybrid/training_target.py`): changes from Amp A toward Amp B according to
  playing level, via the crossover/transition envelope described above.
- **Fixed Blend** (`hybrid/fixed_blend.py`, `hybrid/blend_training_target.py`):
  always combines the two amp responses at one constant, user-chosen ratio
  (`result = A * (1 - mix_b) + B * mix_b`), independent of playing level —
  no crossover envelope at all. Its own auto level-match uses the DI's
  ACTIVE playing material (silence excluded) rather than a crossover band,
  since there's no crossover region to match around (see
  `hybrid.fixed_blend.compute_active_trim`).

Both tabs share Amp A/Amp B, the preview DI, input profile/calibration,
render, test gain, the Listen controls, the Cabinet IR stage, the official
training input, A2 quality, and training — switching tabs never re-runs NAM
inference; the already-rendered `RenderedPair` (`hybrid/pipeline.py`) is
reused by whichever mode you're auditioning.

A third, mode-independent stage — **Cabinet IR** (`hybrid/cab_ir.py`) — sits
AFTER the amp combination in either mode: an ordinary causal FIR convolution,
optionally auditioned in preview only, or "baked" into the generated A2
training target. Preview-only and baked processing always go through the
exact same `apply_cab_ir` function on the COMPLETE prepared IR (never a
shortened one), so what you hear in preview with "Use cab in preview"
checked is exactly what gets trained if you also check "Bake cab into A2".

**Receptive-field policy: hard core check vs. advisory cabinet check.**
Amp A/Amp B (+, for Hybrid, the bounded crossover envelope) are the CORE
dependency — this must fit inside the destination A2's actual receptive
field, or generation/training is refused exactly as before. A baked cabinet
adds `len(ir) - 1` samples of *serial* temporal dependency ON TOP of that
core (see `hybrid/receptive_field.py`'s `combine_required_history`), and
this FORMAL total is always calculated and reported — but it is only
advisory: since we're training an A2 to *approximate* the rendered teacher
target rather than compiling its signal graph exactly, a baked cab whose
formal total exceeds the A2's receptive field does NOT block training. It
means the A2 will learn an approximation of the post-cab response within its
available temporal capacity, and `scripts/train_a2.py`/the Kaggle cloud
worker print a "CABINET APPROXIMATION" notice explaining exactly that —
validate the result by listening and by checking the printed ESR/RMS
metrics against the baked target. The exact same full-length IR is used for
preview and for baking in either case; only the training-time gating differs.

Because raw WAV/FIR length is a poor proxy for how much of a captured IR is
actually audible signal, the Cabinet card also reports cumulative-energy
diagnostics (e.g. "99.9% energy by: 42.7 ms" for a nominally-500ms IR) —
purely informational, never used to shorten the actual convolution.

## Current limitations / experimental status

- **NAM inference is implemented, via a native tool, not torch.**
  `hybrid/render.py` shells out to `nam_render` (`native/nam_render/`), a
  small C++ CLI built against
  [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
  (the same inference core the official NAM plugin uses). This avoids
  depending on the Python `neural-amp-modeler`/torch package for inference
  entirely, and avoids guessing at the `.nam` → model-class mapping, since
  NAMCore's own `nam::get_dsp()` loader is the reference implementation. See
  `native/nam_render/README.md` for how to build it. `hybrid/render.py` can
  now render real `.nam` files end-to-end; what's still missing is wiring
  that into `app.py`'s `/api/preview`/`/api/generate` routes (currently
  still return HTTP 501 — see below).
- `.nam` file **parsing and calibration metadata** (`input_level_dbu`/
  `output_level_dbu` where present) IS implemented (`hybrid/nam_loader.py`) —
  older/uncalibrated files are read fine, just reported as
  "Calibration metadata unavailable" rather than rejected.
- The envelope follower, blend/crossfade math, level-match trim calculation,
  alignment correction, and safety/peak-ceiling logic are all implemented and
  unit-tested (`hybrid/*.py`, `tests/`) against synthetic signals. `render.py`
  has been smoke-tested against real Fender/JCM800 `.nam` captures, but the
  full pipeline (render → level-match → blend) hasn't been exercised
  end-to-end with real renders yet.
- **A/B alignment (`hybrid/align.py`) is optional and disabled by default**
  (`align_to_reference(..., enabled=False)`). It cross-correlates Amp A's
  render directly against Amp B's render, which can misread a genuine
  tonal/phase difference between dissimilar amps (e.g. clean vs. heavily
  distorted) as latency and "correct" for something that isn't actually a
  timing offset. It stays available (`enabled=True`) for later use, but only
  once we've verified what latency guarantees, if any, the official NAM
  inference API actually makes — see that module's docstring.
- The Flask app (`app.py`) and UI (`templates/index.html`, `static/`) now
  wire up real rendering and preview end-to-end (`/api/render_pair`,
  `/api/preview`, `/api/blend_info`, `/api/blend_curve`, `/api/input_profiles`,
  `/api/profile_coverage`). `/api/generate` (final training-target generation)
  still intentionally returns HTTP 501 — see "Not implemented yet" below.
- No A2 training step exists yet at all — that remains future work once
  synthetic target generation is working end-to-end.
- No automated audio-quality/tone judgment is attempted anywhere in this
  project; only mechanical sanity checks (NaN/Inf, clipping, silence,
  alignment, discontinuities — see `hybrid/safety.py` and `tests/`).

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
│   ├── fixed_blend.py      -- Fixed Blend design mode (fixed-ratio combination)
│   ├── cab_ir.py           -- shared cabinet IR convolution (preview + baked target)
│   ├── training_target.py         -- Dynamic Hybrid A2 training-target generation
│   ├── blend_training_target.py   -- Fixed Blend A2 training-target generation
│   ├── receptive_field.py -- mode/cab-aware temporal-dependency accounting
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
python -m pytest tests/
```

The test suite covers `hybrid/envelope.py`, `hybrid/blend.py`,
`hybrid/level_match.py`, `hybrid/align.py`, `hybrid/safety.py`, and
`hybrid/nam_loader.py` against synthetic signals and does not require torch,
`neural-amp-modeler`, or the native `nam_render` tool to be built.
`tests/test_render.py` exercises real NAM inference and is skipped
automatically unless both `native/nam_render` has been built (see its
README) and a real `.nam` file is present at
`assets/nam_models/FenderSuperReverb1977_Clean.nam`.

`torch`/`neural-amp-modeler` are no longer in `requirements.txt` — inference
is handled entirely by the native `nam_render` tool now. They only matter for
the eventual A2 training step, and live in `requirements-training.txt`
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

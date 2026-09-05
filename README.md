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

## Workflow (target, once fully implemented)

1. Select Amp A `.nam` and Amp B `.nam`.
2. Choose the crossover point (dBFS) and transition width (dB) — how loud the
   input needs to get before Amp A hands off to Amp B, and how gradual that
   handoff is.
3. Auto level-match trims Amp B to Amp A around the crossover region; override
   manually if desired.
4. Preview Amp A alone, Amp B alone, and the hybrid blend against a chosen
   genre DI.
5. Generate the hybrid synthetic target (dry input → Amp A render → Amp B
   render → level-matched, blended output — with an *optional*, currently
   disabled-by-default alignment step, see below), with metadata describing
   exactly how it was built.
6. (Future) Train a NAM A2 model on the generated input/output pair.

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
- The Flask app (`app.py`) and UI (`templates/index.html`, `static/`) expose
  the intended controls and routes, but `/api/preview` and `/api/generate`
  currently return HTTP 501 until rendering is wired in.
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

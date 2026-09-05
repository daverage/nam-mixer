# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hybrid NAM Builder is an experimental proof-of-concept tool for building a dynamic,
input-level-driven crossfade between two existing Neural Amp Modeler (`.nam`) amp
captures (e.g. clean amp at low input level morphing into a crunch/lead amp at high
input level), with the eventual goal of training a fresh NAM model on the resulting
blended audio. See README.md for the full concept, rationale, and current
limitations — it is detailed and should be read before making architectural changes.

**Key thing to know before touching this repo:** NAM inference
(`hybrid/render.py`) is implemented via a native C++ tool, not the Python
`neural-amp-modeler`/torch package. `render()` shells out to `nam_render`
(built from `native/nam_render/`, which links
[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) —
see `native/nam_render/README.md` to build it), avoiding both the torch
dependency and any risk of guessing wrong at the `.nam` on-disk schema, since
NAMCore's own loader is authoritative. The full render → level-match → blend
pipeline IS wired into the Flask routes and browser UI (`/api/render_pair`,
`/api/preview`, `/api/blend_info`, `/api/blend_curve`, `/api/input_profiles`,
`/api/profile_coverage`) — only `/api/generate` (final A2 training-target
generation) still intentionally returns HTTP 501.

**Second key thing:** there are three separate, easily-conflated "level"
concepts, at two different costs:

- **Input profile** (`hybrid/input_profiles.py`) simulates a different
  instrument/pickup driving the signal BEFORE either NAM sees it. Changing it
  is EXPENSIVE (`render_pair()` re-runs NAM inference for both amps). See
  `docs/INPUT_PROFILE_RESEARCH.md`.
- **NAM input calibration** (`hybrid/calibration.py`) reconciles two `.nam`
  captures' own recording-calibration metadata (`input_level_dbu`) via the
  official NAM plugin's compensation formula, also applied inside
  `render_pair()` — a different concept from the input profile (instrument
  vs. amp-capture bookkeeping), applied at the same render stage.
- **Crossover / transition / manual trim** (`build_hybrid()`) only reblends
  the already-rendered Amp A/B audio. This is CHEAP (pure numpy) and must
  never re-invoke NAM inference.

A now-deprecated `dry_gain_db` parameter still exists on `build_hybrid()` for
regression tests only (it shifts just the blend envelope, not real audio) —
never wire it to a user-facing control; use `input_profile_gain_db` on
`render_pair()` instead for anything real.

## Commands

```bash
pip install -r requirements.txt   # flask, numpy, scipy, soundfile, torch, neural-amp-modeler, pytest
python app.py                     # runs Flask dev server on http://127.0.0.1:5000/
python -m pytest tests/           # full test suite
python -m pytest tests/test_blend.py           # single test file
python -m pytest tests/test_blend.py::test_name -v  # single test
```

The test suite exercises `hybrid/envelope.py`, `hybrid/blend.py`,
`hybrid/level_match.py`, `hybrid/align.py`, `hybrid/safety.py`,
`hybrid/nam_loader.py`, `hybrid/input_profiles.py`, `hybrid/calibration.py`,
`hybrid/coverage.py`, and `hybrid/pipeline.py` against synthetic signals only
(the pipeline tests fake out `render()` via monkeypatch) — no torch or built
native tool required. `tests/test_render.py` exercises real NAM inference and
auto-skips unless `native/nam_render` has been built AND a real `.nam` file
exists at `assets/nam_models/FenderSuperReverb1977_Clean.nam` (gitignored,
user-provided).

```bash
cmake -B native/nam_render/build -S native/nam_render
cmake --build native/nam_render/build --config Release --target nam_render
```
builds the native inference tool — see `native/nam_render/README.md`.

`scripts/analyze_di.py` regenerates `assets/di/_analysis.json` (measured
characteristics of the bundled DI fixtures).

## Architecture

`hybrid/` is a pure audio-processing library with no Flask/UI dependencies;
`app.py` + `templates/`/`static/` is a thin Flask UI layer over it. The intended
end-to-end pipeline (see README.md "Workflow" section for the full picture):

1. **`nam_loader.py`** parses `.nam` files into a `NamModel`, including
   `input_level_dbu`/`output_level_dbu` calibration metadata when present (older
   files without it are read fine, just report "unavailable" rather than erroring).
2. **`render.py`** is the seam between a parsed `NamModel` and actual audio —
   shells out to the native `nam_render` tool (see above). Contract: mono
   float32 in/out, same length, sample-rate-preserving; raises
   `NamRenderError` if the tool is missing or fails (e.g. sample-rate mismatch
   between `audio` and what the model expects — resampling is the caller's
   job, not `render()`'s).
3. **`envelope.py`** computes `level = envelope(dry_input)` — the per-sample dBFS
   envelope of the *original dry signal* (not either amp's output) that drives
   the crossfade. This is deliberate: the dry signal is the one thing that's the
   same regardless of which amp(s) are being auditioned, avoiding a
   chicken-and-egg dependency on the blend to compute the blend.
4. **`level_match.py`** computes an automatic gain trim for Amp B relative to Amp
   A, measured only from the portion of each render near the chosen crossover
   point (not overall loudness) — because what matters is that the two amps
   sound level-matched *at the transition*, not on average.
5. **`align.py`** detects/corrects sample-offset between the two amp renders
   before blending.
6. **`blend.py`** performs the actual crossfade (smoothstep by default) driven by
   the envelope output from (3), using the trim from (4) and alignment from (5).
7. **`safety.py`** has two distinct, non-interchangeable functions:
   - `apply_peak_ceiling` — a single fixed gain reduction applied to a whole file
     if it exceeds a peak ceiling (default -3 dBFS). This is what must be used on
     anything that becomes/derives a **training target**, because a limiter would
     distort the dynamic behavior the whole project exists to capture.
   - `preview_safety_limiter` — an actual limiter, but only for the **live
     preview/playback** path (speaker/headphone safety net). Never apply this to
     a training-target file.
8. **`metadata.py`** defines the JSON sidecar schema recording exactly how a
   given hybrid target was generated (amps used, crossover point, trims, etc.),
   for provenance.
9. **`input_profiles.py`** defines research-grounded relative-gain presets for
   guitar/bass pickup families (see docs/INPUT_PROFILE_RESEARCH.md) plus
   `db_to_amplitude`/`resolve_profile_gain_db`. Active/buffered pickups
   deliberately have NO fixed preset (`requires_custom_gain=True`) — manufacturer
   data shows too much spread for a defensible universal number.
10. **`calibration.py`** implements the official NAM plugin's per-model input
    compensation formula (`reference_input_level_dbu - model_input_level_dbu`)
    and the Auto/Raw mode selection logic (Auto only compensates when BOTH
    models in a pair report calibration metadata; one-sided calibration falls
    back to Raw with a warning rather than calibrating asymmetrically).
11. **`coverage.py`** is a render-free reachability analysis: given a DI's
    envelope and a candidate profile gain, reuses `blend.blend_weight` to
    report what fraction of ACTIVE playing time would land mostly-A/
    transition/mostly-B for the current crossover settings. No NAM inference,
    safe to call on every crossover/transition change.
12. **`pipeline.py`** ties it together as two costs: `render_pair()` (EXPENSIVE
    — NAM inference, applies input profile gain + calibration, computes the
    profile-adjusted envelope) and `build_hybrid()` (CHEAP — pure numpy
    reblend of an already-rendered `RenderedPair`).

`assets/di/` contains real recorded genre/style DI guitar/bass performances
(sourced from the NAMtoClo project — see `assets/di/README.md`) used for
auditioning and as regression-test fixtures. These are explicitly **not** the
same thing as a proper calibrated NAM training/reamping signal — do not use them
as the actual input for generating a real A2 training pair; see README.md's
"Preview DIs vs. NAM training material" section for the distinction.

`work/` is a gitignored scratch output directory.

`assets/nam_models/` holds the user's own `.nam` amp capture files (e.g. a
Fender clean + a JCM800 high-gain capture) used as Amp A/Amp B inputs. These
are gitignored (`assets/nam_models/*.nam`) — personal captures, not versioned
project fixtures like `assets/di/*.wav`.

## Known caveats in the current implementation

- `hybrid/envelope.py`'s `rms_envelope_db` is deliberately **causal** (a
  zero-padded windowed sum, not `np.convolve(..., mode="same")`) — the
  envelope becomes the crossover control signal baked into the synthetic
  training target, so it must never be influenced by samples after the
  current one, or a causal A2 model trained on the result would be asked to
  predict the future.
- `hybrid/align.py`'s `align_to_reference` cross-correlates Amp A's render
  directly against Amp B's render, which can misread a genuine tonal/phase
  difference between dissimilar amps (e.g. clean vs. heavily distorted) as
  latency. Pass `enabled=False` (skips correction, still length-matches) until
  real NAM inference is wired in and the official inference API's latency
  behavior is understood — see the module docstring.
- The bundled `assets/di/*.wav` genre clips were found to be mostly
  normalized around a common RMS level by their original source, so their
  waveform cannot tell you what pickup actually produced them. Input
  profiles simulate relative gain around the SELECTED DI treated as a
  reference performance (vintage/PAF humbucker for guitar, standard J/P bass
  for bass) — never a claim about the DI's real recording history. See
  docs/INPUT_PROFILE_RESEARCH.md.
- Guitar volume-knob simulation is deliberately NOT implemented — pot taper,
  loading, and treble-bleed circuits vary too much between instruments to
  responsibly guess fixed dB values yet.

At the end of each major change, commit and push the repo

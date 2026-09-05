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
NAMCore's own loader is authoritative. `app.py`'s `/api/preview` and
`/api/generate` still return HTTP 501 — `render()` itself works, but the full
pipeline (render both amps → level-match → blend) isn't wired into the Flask
routes yet.

## Commands

```bash
pip install -r requirements.txt   # flask, numpy, scipy, soundfile, torch, neural-amp-modeler, pytest
python app.py                     # runs Flask dev server on http://127.0.0.1:5000/
python -m pytest tests/           # full test suite
python -m pytest tests/test_blend.py           # single test file
python -m pytest tests/test_blend.py::test_name -v  # single test
```

The test suite exercises `hybrid/envelope.py`, `hybrid/blend.py`,
`hybrid/level_match.py`, `hybrid/align.py`, `hybrid/safety.py`, and
`hybrid/nam_loader.py` against synthetic signals only — no torch or built
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

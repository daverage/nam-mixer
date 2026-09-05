# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hybrid NAM Builder is an experimental proof-of-concept tool for building a dynamic,
input-level-driven crossfade between two existing Neural Amp Modeler (`.nam`) amp
captures (e.g. clean amp at low input level morphing into a crunch/lead amp at high
input level), with the eventual goal of training a fresh NAM model on the resulting
blended audio. See README.md for the full concept, rationale, and current
limitations — it is detailed and should be read before making architectural changes.

**Key thing to know before touching this repo:** NAM inference itself
(`hybrid/render.py`) is deliberately NOT implemented yet — see that module's
docstring for why (the `.nam` on-disk schema → `neural-amp-modeler` model class
mapping was not guessed at without a way to verify it). `/api/preview` and
`/api/generate` in `app.py` return HTTP 501 as a result. Do not silently "fill in"
`render()` with a guessed mapping; implementing it properly requires verifying
against a known-good reference render per the steps in the module docstring.

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
`hybrid/nam_loader.py` against synthetic signals only — it does **not** require
torch/neural-amp-modeler to be installed. Those two packages are only needed to
run `app.py` and (once implemented) actual NAM inference.

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
   currently a stub (see above). `blend.py` and `app.py` are written against its
   intended contract (mono float32 in/out, same length, sample-rate-preserving)
   so wiring in real inference should be a drop-in change.
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

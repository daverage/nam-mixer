# NAM Mixer: Continuous Gain v3

## 1. Goal

Create **one standard `.nam`** that behaves like the gain control of a real amp. We have real NAM captures of one
amp at ten gain settings (G1-G10). The new NAM, played through an ordinary NAM player where the only control is
input gain, should sound and feel as close as possible to those ten captures:

- Input gain at its lowest: sounds like the real G1 capture.
- Input gain at its highest: sounds like the real G10 capture.
- In between: tracks G2-G9 and moves smoothly between them.

Playback path:

```text
Guitar / DI -> ordinary NAM-player input gain -> ONE newly trained NAM -> output
```

The aim is a more usable single model. It does **not** have to match exactly. It should be meaningfully close in
tone, saturation, compression, body and playing feel at every gain position. No extra parameters, custom loader,
anchor switching, runtime blending, or external profile.

## 2. What is not the goal

- **G5 is only a baseline.** The G5 capture with its input gain pushed up or down is the thing to beat ("can a
  newly trained NAM do better than just changing the input on the G5 capture?"). It is never a training anchor,
  used to derive the gain mapping, or a target to improve.
- Not the earlier multi-anchor / calibrated-G5 playback work, and not a runtime blend controller.
- Not exact waveform reproduction. ESR is a diagnostic, not a success criterion (heavily saturated material).
- Output level does **not** need to match. Captures are likely normalised and their levels unbalanced, so training
  does not level-match targets, and evaluation compares level-matched.

The earlier `scripts/single_nam_*` experiment (ESR-fitted, G5-anchored gain law) is history, not a base to extend.
Its results are in `docs/history/CONTINUOUS_GAIN_SINGLE_NAM_TRAINING.md` for comparison only.

## 3. Approach: reuse the app's Dynamic Hybrid machinery

The main application already does level-driven blending and A2 training. Reuse it rather than building parallel
tooling.

Dynamic Hybrid today: two rendered amps crossfaded by a causal level envelope of the dry input
(`hybrid/envelope.py`, `hybrid/blend.py`), turned into an A2 training bundle from the official NAM training input
(`hybrid/training_target.py`), then trained (`scripts/train_a2.py`, Kaggle backend) and validated
(`hybrid/validation.py`). A model trained on that target answers quiet input like Amp A and loud input like Amp B,
so raising input gain moves it towards Amp B. That is the behaviour of a gain control.

Extension: an **ordered N-capture chain** (G1..G10, or a subset). Level thresholds are spread across the training
input's active level range, quietest region = lowest gain capture, loudest = highest. Only adjacent captures ever
mix. The target is a teacher signal built from real capture renders, then trained as a normal A2 bundle.

Reuse: `envelope.py`/`bounded_causal_envelope_db`, `blend_weight`/`CrossoverConfig` per adjacent pair, `render.py`,
official training-input validation, cab/safety/receptive-field helpers, manifest/bundle format, `train_a2.py` and
Kaggle, `validation.py`. New: an N-way blend sibling of `blend.py` and a bundle generator sharing
`generate_training_bundle`'s helpers. Keep production changes small and separate from existing modes.

## 4. Design rules

1. **Gain positions are designed, not fitted.** Thresholds are placed across the official input's measured active
   level range (about -56 dBFS at the 10th percentile to -11 dBFS at the 98th on the current input). No fitting
   against any capture, G5, or evaluation audio. The mapping is frozen before training and recorded.
2. **Preserve the real amp's behaviour.** No per-capture normalisation, EQ, compression, or saturation changes to
   targets. Level-matching lives in analysis only.
3. **Minimum separation.** Adjacent gains must be separable by level. Where two captures are near-identical (e.g.
   saturated G9/G10), record it rather than hide it.
4. **Only permitted captures.** A 5- or 3-capture model may use only its own captures. No cached targets, mappings
   or fine-tuning from a larger model. Any shared mapping is labelled a separate experiment.
5. **One conventional exported `.nam`**, reloaded through normal NAM inference (NAMCore) for every result.
6. **Frozen mapping for evaluation.** Never re-tune input gain per test DI, per capture, or against hidden
   references.

## 5. Known limitation to document, not hide

A stateless level control cannot tell a soft note at high gain from a hard note at lower gain. Playing dynamics
overlap with the gain setting. Design to make this rare (separation, varied training material), then measure and
report it: test clean, standard and hot DI levels and report where the model drifts.

## 6. Plan

1. **Review and reuse check.** Confirm what Hybrid/Blended/Character actually do (weights vs rendered output,
   targets, retraining, `.nam` output). Note what extends to N captures.
2. **Feasibility.** Check for conflicting input/output pairs across the level regions (target divergence versus
   input-level separation), on the real captures.
3. **Endpoint test (G1 + G10).** Build the two-endpoint target with the N-way tool, train a small representative
   model, and compare against real G1/G10 on training and unseen DIs, level-matched. Diagnose whether any failure is
   the target, the training, or generalisation before scaling up.
4. **10-capture model** on the JCM800 (Marshall 2203 High channel, G1-G10 plus genuine half-steps for held-out
   interpolation checks). Export, reload, evaluate.
5. **Reduce captures:** 5 (G1, G3, G5, G7, G10), 3 (G1, G5, G10), 2 (G1, G10). Each is its own newly trained model
   with a comparable budget. Find the smallest set that stays close to the 10-capture result; if the gaps are in
   specific gain regions, try captures placed there.
6. **Other amps** (Super-Sonic, 57 Twin, Peavey 5150, and any others): repeat per amp; do not assume mapping or
   capture count transfers.

## 7. Evaluation

Held-out DIs only (never in training or validation), including clean, standard and higher-output material. For
each gain: real capture on the DI versus the new NAM at that gain's designated input level, plus half-steps for the
JCM800, and a continuous input-gain sweep through the single exported NAM (no model switching).

Measure: level-matched spectral balance, harmonic/saturation character, crest factor and compression, attack and
dynamics, output-level progression (informational, since captures are normalised), and continuity between gains.
Raw and level-matched ESR are supporting diagnostics. Report per gain; never bury a poor endpoint in a mean.

Baselines (not goals): the G5 capture with the same designed input gains, and the G5 with its previously
calibrated mapping (`docs/history/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md`). Report plainly if the new model
does not beat them.

Listening: generate labelled reference sets (real vs new, original level and level-matched, at low/mid/high gain)
plus a blind A/B set. State explicitly that no human listening has happened until it has. Do not claim perceptual
equivalence from measurements alone.

## 8. Deliverables and success criterion

- Scripts/tools kept reproducible; production changes minimal.
- New `.nam` files named by configuration (e.g. `JCM800_ContinuousGain_10Captures.nam`), created only for
  configurations actually trained.
- A results write-up (`docs/CONTINUOUS_GAIN_SINGLE_NAM_TRAINING.md`) covering: designed mapping, target
  construction, feasibility and conflict analysis, endpoint result, per-gain results, half-steps, level
  generalisation, G5 baseline comparison, capture-count findings, listening file locations, limitations, next step.

**Success:** a single standard NAM, driven only by ordinary input gain, that is meaningfully close in tone,
saturation and feel to the real G1-G10 captures, and the smallest capture count that achieves it.

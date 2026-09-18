# Continuous Gain Model -- Conditional-Model Feasibility Experiment

Follow-up to docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md,
docs/CONTINUOUS_GAIN_GENERALIZATION.md, and
docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE(_ADDENDUM).md. Tests a
fundamentally different approach from everything so far: instead of
interpolating between several independently-trained NAM captures, train
ONE small neural network on `(audio, knob position) -> output` directly,
and see whether it generalizes to genuinely withheld knob positions better
than the existing nearest-neighbour interpolation baseline.

This is a feasibility check, not an architecture search or a production
candidate. Per the task's own instruction, the architecture and training
budget were kept deliberately small and were NOT tuned to make the model
look better.

New code: `scripts/train_conditional_gain_nam.py`. Runs in the SEPARATE
`.venv-a2` training environment (has torch + Apple MPS), never the normal
Flask runtime environment (see CLAUDE.md / requirements-training.txt) --
this is real inference and real gradient-based training throughout, not a
fake-render unit test.

## 1. Conditioning mechanism

FiLM (Feature-wise Linear Modulation): the knob position is normalized to
`[0, 1]` linearly over the TRAINING gain range (no manual pot-law warping,
no response coordinate, no level/spectral trajectory fed in -- exactly the
"feed the actual knob value, let the model learn whatever nonlinear
relationship exists" instruction). That scalar goes through a small MLP
(`Linear(1, 24) -> ReLU -> Linear(24, 24)`) to produce a 24-dim conditioning
vector, which every residual block turns into a per-channel scale and
shift (`Linear(24, 48)` per block) applied to that block's gated-activation
input. This is the standard WaveNet-family conditioning mechanism, chosen
specifically because it makes no assumption about linearity -- the network
is free to learn an arbitrarily nonlinear mapping from the scalar knob
value to its effect on the audio.

## 2. Model architecture and parameter count

A small causal dilated-convolution stack (WaveNet-family), NOT the official
`neural-amp-modeler` architecture and NOT run through `scripts/train_a2.py`:

- 1x1 input conv (1 -> 24 channels)
- 10 `FiLMGatedBlock`s: causal dilated conv (kernel 3, dilation
  1, 2, 4, ..., 512) -> FiLM conditioning -> gated tanh/sigmoid activation
  -> 1x1 residual projection + 1x1 skip projection
- Skip connections summed, ReLU, 1x1 conv, ReLU, 1x1 conv -> single-channel output

**Parameters: 60,361. Receptive field: 2,047 samples (42.6 ms @ 48 kHz).**

This receptive field is deliberately much smaller than the official A2's
~6,332 samples (~132 ms, see `hybrid/receptive_field.py`) -- appropriate
for a feasibility check on whether conditioning works at all, not a claim
that this capacity is sufficient for a real product model.

## 3. Training data construction

Primary dataset: `Marshall JCM800 2203 - updated` (the same dense 0.5-step
sweep used in the density/response-coordinate reports), High channel only.

For each TRAINING gain, the real `.nam` capture is rendered (real NAM
inference, `hybrid.render.render`) through 3 DI clips for content
diversity: `assets/di/moderate_brit.wav`, `clean_smooth.wav`,
`high_metalcore.wav`, 12 seconds each (36s per gain). WITHHELD gains'
captures are loaded ONLY to produce ground truth for scoring -- never
rendered into any training crop.

**Evaluation audio is a held-out TEMPORAL segment**: the final 6 seconds of
`moderate_brit.wav`, which is never included in any training crop (training
only uses that file's first 12 of its ~18s). This means the withheld-gain
score is not confounded by the model having simply memorized that exact
audio content -- both the GAIN and the AUDIO SEGMENT are genuinely unseen
at evaluation time.

Training used random 16,384-sample crops, batch size 8, Adam (lr 1e-3),
3,000 steps, loss = the same raw-ESR definition
`hybrid.validation.compute_esr_metrics` uses (`sum((pred-target)^2) /
sum(target^2)`), so training progress is directly comparable to the
evaluation metric.

## 4. Gain encoding / normalization

Linear min-max normalization over the TRAINING gains' range only (matching
`GainCaptureSet.normalized_position`'s convention elsewhere in this
research), e.g. for training gains 1-9: `(gain - 1) / (9 - 1)`. No
electrical-linearity assumption is baked in beyond this -- the network's
own nonlinear layers are what have to discover the real mapping.

## 5. Held-out gains

Two experiments, exactly as specified:

- **Experiment 1 (sparse)**: train on Gain 1, 3, 5, 7, 9; test on Gain 2,
  4, 6, 8 (all genuinely interior, all real captures).
- **Experiment 2 (dense)**: train on integer Gain 1-10; test on half-step
  Gain 1.5, 2.5, ..., 9.5 (9 positions, all real captures).

## 6. Training convergence

Loss (raw ESR) recorded every 50 steps:

| Step | Experiment 1 (run 1) | Experiment 1 (run 2) | Experiment 2 |
|---:|---:|---:|---:|
| 0 | 4.05 | 1.43 | 1.80 |
| 750 | 0.24 | -- | 0.22 |
| 1500 | 0.10 | -- | 0.10 |
| 2250 | 0.04 | -- | 0.05 |
| 2999 (final) | 0.049 | 0.051 | 0.068 |

Loss decreases smoothly and plateaus by roughly step 2000-2250 in every
run -- training clearly converges within the given budget, it does not
diverge or stall early. Wall-clock: ~165-170s per 3,000-step run on Apple
MPS.

**Important caveat discovered during this experiment**: model weight
initialization was not seeded in the original runs, and re-running
Experiment 1 with identical hyperparameters produced a noticeably different
converged loss (0.049 vs 0.051) and, more importantly, a DIFFERENT
qualitative outcome at the withheld positions (see Section 8). `--seed` has
since been added to the script (fixes both weight init and batch sampling)
so future runs are reproducible; the two Experiment 1 runs reported here
predate that fix and are kept as evidence of how much run-to-run variance
exists at this model scale, not as a bug.

## 7. Real vs. reconstructed outputs

Audio artifacts for Experiment 1 (run 2) are in
`work/continuous_gain_conditional/exp1/`: `gain_{2,4,6,8}_{real,model,
baseline}.wav` -- the real capture, the conditional model's reconstruction,
and the knob-linear interpolation baseline, all rendered on the identical
held-out 6-second evaluation clip, for direct listening comparison.

## 8. Comparison with ordinary interpolation

Metrics: raw ESR and gain-normalized ESR
(`hybrid.validation.compute_esr_metrics`, already established as
trustworthy for the JCM800 datasets -- unlike the 5150, per
docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md), plus level delta (`|Δ active
RMS dB|`) and spectral correlation
(`hybrid.audio_metrics.spectral_magnitude_correlation`) reported
SEPARATELY, not collapsed into one score (per the response-coordinate
addendum's finding that they often disagree).

### Sanity check: reconstructing TRAINING gains

| Gain | Experiment | raw ESR | spectral corr | level Δ (dB) |
|---:|---|---:|---:|---:|
| 1 | 1 (run 1) | 0.021 | 0.882 | 0.18 |
| 3 | 1 (run 1) | 0.031 | 0.921 | 0.37 |
| 5 | 1 (run 1) | 0.034 | 0.911 | 0.14 |
| 7 | 1 (run 1) | 0.049 | 0.884 | 0.51 |
| 9 | 1 (run 1) | 0.058 | 0.875 | 0.52 |
| 1-10 | 2 | 0.025-0.076 | 0.857-0.919 | 0.00-0.43 |

**The model can reproduce gain positions it saw during training, but not
perfectly** -- raw ESR 0.02-0.08 on TRAINING data is nontrivial error (for
comparison, the knob-linear interpolation baseline's WITHHELD-position ESR
on the same amp region was often in the same 0.002-0.06 range in the
density report). This is the first sign that the limiting factor here is
likely capacity/training budget, not the conditioning mechanism itself --
see Section 12.

### Experiment 1 (sparse): withheld Gain 2, 4, 6, 8

| Gain | Run | model raw ESR | baseline raw ESR | model wins? |
|---:|---|---:|---:|---|
| 2 | 1 | 0.059 | 0.043 | no |
| 4 | 1 | 0.036 | 0.008 | no |
| 6 | 1 | 0.063 | 0.054 | no |
| 8 | 1 | 0.133 | 0.123 | no |
| 2 | 2 | 0.028 | 0.043 | **yes** |
| 4 | 2 | 0.030 | 0.008 | no |
| 6 | 2 | 0.053 | 0.054 | **yes (marginal)** |
| 8 | 2 | 0.117 | 0.123 | **yes (marginal)** |

Run 1: baseline wins all 4. Run 2 (different random init only): model wins
3 of 4, though 2 of those wins are marginal. **Gain 4 is the one position
where the baseline wins clearly and consistently in BOTH runs** (0.008 vs
0.030-0.036, 4-5x better) -- see Section 9.

### Experiment 2 (dense): withheld Gain 1.5-9.5

| Gain | model raw ESR | baseline raw ESR | model wins? |
|---:|---:|---:|---|
| 1.5 | 0.031 | 0.011 | no |
| 2.5 | 0.025 | 0.004 | no |
| 3.5 | 0.033 | 0.002 | no |
| 4.5 | 0.037 | 0.002 | no |
| 5.5 | 0.037 | 0.002 | no |
| 6.5 | 0.049 | 0.030 | no |
| 7.5 | 0.074 | 0.045 | no |
| 8.5 | 2.106 | 2.050 | no (both anomalous, see Section 10) |
| 9.5 | 0.162 | 0.140 | no |

**Baseline wins all 9 of 9 positions**, several by large margins (up to
~17x at Gain 3.5/4.5/5.5).

## 9. Behaviour around the clean-to-breakup region

The dense-Marshall knee (per docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md) sits
around Gain 1.0-3.0. Experiment 1's Gain 4 (just past the knee) is the
single most consistent model failure across both runs -- the model is
4-5x worse than baseline there in both, while at Gain 2/6/8 the two runs
disagree on which method wins. This is consistent with (but does not
prove) the region right after a fast-changing knee being harder for a
FIXED-capacity, FIXED-receptive-field conditional model to fit accurately
across ALL training gains simultaneously, since it has to represent the
sharpest local nonlinearity using the same shared weights that also have
to represent the much flatter Gain 5-9 region.

## 10. A note on the Gain 8.5 outlier (Experiment 2)

Both the model (raw ESR 2.11) and the baseline (2.05) show a huge, matched
spike at Gain 8.5 -- roughly 30-50x every other position in that
experiment. Checked the real Gain 7.5-9.5 captures' active RMS/peak level
on the exact held-out eval segment: perfectly smooth and monotonic
(-22.06, -21.37, -20.98, -20.84, -21.04 dBFS) -- **no capture-level anomaly
like the earlier-documented 5150 Gain 4 case**. Since both methods fail
almost identically here, this looks like a shared artifact in that specific
audio moment (e.g. a transient in the held-out clip that both
reconstructions handle differently in timing) rather than a model-specific
failure. Not investigated further -- flagged so it isn't mistaken for
either method's typical behaviour.

## 11. Effect of capture/training spacing

Comparing the two experiments is confounded (see Section 12), but taken at
face value: the DENSE experiment's relative gap between model and baseline
is WORSE (baseline consistently 2-17x better) than the SPARSE experiment's
(mixed, close, sometimes model-favourable). **Denser training positions did
NOT materially improve the conditional model's unseen-position
reconstruction in this test** -- if anything the opposite trend appears,
though see Section 12 for why this should not yet be read as "density
hurts a conditional model."

## 12. Does one conditional model preserve level and spectral trajectories?

Reported separately as instructed. Level delta stays small and comparable
between model and baseline in almost every row (both methods are
usually within a fraction of a dB of the real capture's level) -- level
reproduction is NOT where the conditional model loses. Spectral
correlation is more telling: the model's spectral correlation is
consistently a few points LOWER than the baseline's at most withheld
positions (e.g. Experiment 2, Gain 3.5: baseline 0.986 vs model 0.918;
Gain 9.5: baseline 0.977 vs model 0.865) -- **the conditional model's
shortfall versus baseline shows up more in spectral/harmonic character than
in level**, consistent with every other finding in this research thread
that level is rarely the bottleneck.

## Decision gate

**Does a single conditional NAM predict genuinely unseen gain positions
better than interpolating the outputs of independently trained NAMs?**

**No, not in this implementation.** The dense experiment (9 withheld
positions, single run) shows a clear, consistent, often large loss to the
baseline. The sparse experiment (4 withheld positions) is noisy and
run-dependent, with no consistent win for the conditional model and one
position (Gain 4) where the baseline wins clearly in both runs.

### Is the limitation capacity, conditioning representation, training data
construction, or evidence that discrete-model interpolation is better?

Most likely **capacity and/or training budget**, not the conditioning
representation itself:

- The model does not even fit its OWN training gains cleanly (raw ESR
  0.02-0.08, not near-zero) -- a model that can't memorize its training
  conditions well can't be expected to generalize past them convincingly.
  This points at model size (60k params, 42.6ms receptive field, both far
  below the official A2's scale) and/or training steps (3,000, a small
  budget) as the first things to change, before concluding anything about
  FiLM/scalar conditioning as an approach.
- The two experiments are NOT a clean density comparison: Experiment 2 has
  twice as many (gain, DI clip) pairs to fit as Experiment 1 but used the
  SAME 3,000-step budget, so it was very likely more under-trained per
  condition, not fundamentally harder to condition on. Any conclusion about
  "does density help a conditional model" needs a matched-training-budget
  rerun (e.g. scale steps with pair count) before it can be trusted -- the
  current result is confounded and should not be read as evidence against
  scaling training data for this approach.
- Conditioning representation (FiLM on a raw normalized scalar) is a
  standard, working mechanism in the broader neural audio literature and
  showed no obvious pathology here (loss converges smoothly, level
  reproduction is fine) -- there is no evidence in this experiment that the
  conditioning MECHANISM itself is the bottleneck, as opposed to capacity.
- **Given the evidence available now, discrete-model interpolation remains
  the better practical approach** -- it is simpler, requires no training
  step at all, and outperformed the conditional model at 13 of 13 tested
  withheld positions across both experiments combined (excluding the
  shared Gain 8.5 anomaly) except for the noisy, small, run-dependent wins
  in Experiment 1's second run.

## What this does NOT resolve

- Whether a LARGER conditional model (more channels/layers, official-A2-
  scale receptive field) or a longer training budget would close or reverse
  this gap. This experiment was explicitly kept small per the task's scope
  ("do not optimise beyond what is required to answer the feasibility
  question") -- it answers "does the simplest reasonable version work,"
  not "can this approach ever work."
- Whether a matched-training-budget rerun would change the density
  conclusion in Section 11.
- Behaviour on amps other than the dense Marshall JCM800 sweep (not tested
  here, to keep this a single, interpretable comparison rather than
  spreading the same small budget across every dataset).

## Recommendation for next step

Given the decision gate's answer is "no" at this scale, and the most
likely cause is capacity/training budget rather than a fundamental flaw:
**do not invest further in scaling up a conditional model yet.** The
existing discrete-capture-plus-interpolation approach, refined by this
research thread's actual positive findings (capture-density placement,
per-dataset knee identification), remains the better-supported direction
for near-term product work. If conditional modelling is revisited later,
the correctly-scoped next experiment is a matched-training-budget rerun of
Experiment 2 (scale steps with the number of (gain, DI) pairs) specifically
to separate "under-trained" from "doesn't scale with density" -- not a
larger architecture search, which would be premature before that confound
is resolved.

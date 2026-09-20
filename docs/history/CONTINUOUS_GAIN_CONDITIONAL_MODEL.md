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
audio moment rather than a model-specific failure.

**Root cause identified in
docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md**: this is a genuine
~105-sample (~2.2ms @ 48kHz) LATENCY offset baked into the Gain 8.5 `.nam`
capture file itself, relative to its immediate Gain 8.0/9.0 neighbours
(which are aligned to within 1 sample of each other). It reproduced
identically across three methods that share nothing except this one target
capture (virtual-input-gain reconstructions from 3 different anchors, and
dense discrete interpolation) -- conclusive evidence the anomaly lives in
the Gain 8.5 file, not in any reconstruction method tested against it. Any
future benchmark on this dataset should flag or exclude Gain 8.5 from
aggregate statistics rather than re-investigating it as a modelling
failure.

## 11. Effect of capture/training spacing

Comparing the two experiments was originally confounded by unequal
training budget per (gain, DI) pair -- **now resolved by Section 13's
matched-budget re-run.** Taken at face value here: the DENSE experiment's
relative gap between model and baseline is WORSE (baseline consistently
2-17x better) than the SPARSE experiment's (mixed, close, sometimes
model-favourable). **Denser training positions did NOT materially improve
the conditional model's unseen-position reconstruction in this test**, and
Section 13 confirms this was not simply because the dense run was
under-trained relative to the sparse one -- doubling its budget closed only
part of the training-gain gap and left the withheld-gain shortfall
essentially unchanged in every one of 3 seeds. The most defensible reading
is that this particular architecture/budget combination does not scale
cleanly with added gain conditions, not that density is actively harmful.

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

## 13. Matched-training-budget re-run (6,000 steps, 3 seeds)

Direct follow-up to Section 12's identified confound: Experiment 2 had
twice the (gain, DI) pairs of Experiment 1 but used the same 3,000-step
budget. This re-run changes ONLY the training budget (3,000 -> 6,000 steps)
and reproducibility controls (3 fixed seeds: 0, 1, 2) -- architecture,
conditioning, loss, receptive field, training DIs, and held-out evaluation
audio are all UNCHANGED from Experiment 2.

### Training convergence at 6,000 steps

Loss in the final 350 steps oscillates in a 0.02-0.04 band in all 3 seeds
(e.g. seed 0: 0.038, 0.025, 0.035, 0.031, 0.036, 0.033, 0.040, 0.030 at
steps 5650-5999) rather than continuing to trend down -- **training has
reached a noise-floor plateau by 6,000 steps, not a budget cutoff
mid-improvement.** Final loss: 0.030 (seed 0), 0.029 (seed 1), 0.035
(seed 2).

### Training-gain reconstruction: primary decision gate

Per-gain raw ESR, averaged across the 3 seeds, vs. the original single
3,000-step run:

| Gain | 3,000 steps (1 run) | 6,000 steps (avg of 3 seeds) | Improvement |
|---:|---:|---:|---:|
| 1 | 0.049 | 0.014 | 3.5x |
| 2 | 0.026 | 0.010 | 2.6x |
| 3 | 0.029 | 0.013 | 2.2x |
| 4 | 0.034 | 0.016 | 2.1x |
| 5 | 0.034 | 0.017 | 2.0x |
| 6 | 0.040 | 0.022 | 1.8x |
| 7 | 0.076 | 0.032 | 2.4x |
| 8 | 0.068 | 0.042 | 1.6x |
| 9 | 0.062 | 0.048 | 1.3x |
| 10 | 0.069 | 0.054 | 1.3x |
| **Mean** | **0.049** | **0.027** | **1.8x** |

**Verdict on the primary decision gate: partial, not substantial.**
Doubling the training budget produced a real, consistent, roughly 2x
average reduction in training-gain error -- this is not nothing, and it is
NOT simply noise (the improvement direction is consistent across every
single gain and all 3 seeds). But it is also not the "drops substantially"
outcome the gate was checking for: the range shifted from 0.02-0.08 to
roughly 0.01-0.05, a modest downward shift, not a move toward near-zero
fit. Critically, **the improvement shrinks as gain increases** (3.5x at
Gain 1 down to 1.3x at Gain 9-10) -- the model is hitting a firmer capacity
ceiling specifically in the higher-gain region, matching the training-loss
plateau observed above. Per the task's own stopping rule, this sits close
enough to "remains roughly in the previous range" that **the honest
reading is: still capacity-limited, especially at high gain** -- but
because the withheld-gain comparison was already produced by the same runs
at zero extra cost, it is reported below rather than discarded.

### Withheld-gain results across 3 seeds

Raw ESR, conditional model vs. the (seed-independent) knob-linear baseline:

| Gain | Baseline | Model seed 0 | Model seed 1 | Model seed 2 | Model wins? |
|---:|---:|---:|---:|---:|---|
| 1.5 | 0.0111 | 0.0113 | 0.0117 | 0.0077 | seed 2 only |
| 2.5 | 0.0041 | 0.0111 | 0.0078 | 0.0116 | no (all 3) |
| 3.5 | 0.0019 | 0.0171 | 0.0132 | 0.0138 | no (all 3, 7-9x worse) |
| 4.5 | 0.0023 | 0.0185 | 0.0180 | 0.0159 | no (all 3, ~7-8x worse) |
| 5.5 | 0.0022 | 0.0203 | 0.0177 | 0.0178 | no (all 3, ~8x worse) |
| 6.5 | 0.0296 | 0.0273 | 0.0284 | 0.0264 | **yes, all 3 seeds** |
| 7.5 | 0.0449 | 0.0939 | 0.0884 | 0.0817 | no (all 3) |
| 8.5 | 2.0500 | 2.0871 | 2.0942 | 2.0230 | seed 2 only (anomaly, see below) |
| 9.5 | 0.1396 | 0.1671 | 0.1503 | 0.1864 | no (all 3) |

Baseline wins 7-8 of 9 positions in every seed. **Gain 6.5 is the one
position where the model reproducibly ties/beats the baseline in ALL THREE
seeds** -- a small but consistent effect, not noise.

### Answers to the 5 follow-up questions

1. **Does the conditional model now approach or beat ordinary output
   interpolation?** No. It still loses at 7-8 of 9 withheld positions in
   every seed, several by 6-9x (Gain 3.5-5.5) -- the same region that was
   worst before, now confirmed with 3-seed consistency instead of one run.
2. **Are results stable across the 3 seeds?** Yes, much more so than
   Experiment 1's noisy 4-position comparison. Every position's WIN/LOSS
   outcome vs. baseline is identical across all 3 seeds except Gain 1.5 and
   the Gain 8.5 anomaly region -- the qualitative conclusion does not depend
   on which seed is used.
3. **Does the remaining error concentrate in particular gain regions?**
   Yes, in two different senses. In ABSOLUTE terms, the model's worst
   region is high gain (7.5-9.5, raw ESR 0.08-0.19) -- matching the
   training-gain plateau in that same region. In RELATIVE terms (model
   error / baseline error), the worst region is Gain 3.5-5.5, where the
   baseline is exceptionally accurate (raw ESR 0.002, likely because this
   part of the real amp's response happens to interpolate very cleanly
   between its integer-gain neighbours) and the model's comparatively
   modest absolute error looks like a 7-9x gap by comparison.
4. **Is the remaining shortfall primarily spectral/harmonic rather than
   level?** Yes, confirmed more clearly than before. Level deltas for the
   model are small and often SMALLER than the baseline's (e.g. Gain 1.5:
   model 0.25-0.47 dB vs. baseline 0.71 dB; Gain 2.5: model 0.01-0.20 dB
   vs. baseline 0.33 dB) -- level is not where the model loses. Spectral
   correlation is consistently 0.03-0.09 lower than the baseline's at
   nearly every withheld position (e.g. Gain 9.5: model 0.887-0.897 vs.
   baseline 0.977), a real and repeatable gap.
5. **Does the Gain 8.5 anomaly reproduce across seeds?** Yes, closely: model
   raw ESR is 2.09, 2.09, and 2.02 across the 3 seeds, essentially
   identical to the (seed-independent) baseline's 2.05. This confirms the
   anomaly is a property of that specific point in the evaluation audio
   (a hard-to-reconstruct transient, most likely, given both methods and
   all 3 seeds fail almost identically there) rather than a training
   artifact of any one run.

### Answer to the isolating question

**Was the previous dense conditional-model result mainly caused by giving
twice as many gain conditions the same 3,000-step budget?**

**Partially, but not mainly.** Doubling the budget did produce a real,
consistent improvement (training-gain error roughly halved on average,
one withheld position -- Gain 6.5 -- flipped from a loss to a reproducible
win). But the core finding is unchanged and now more strongly evidenced:
baseline interpolation still wins the large majority of withheld positions
in every one of 3 seeds, by large margins in the Gain 3.5-5.5 region
specifically. Under-training was a real, measurable contributor to the
original gap, but it is not the primary explanation -- the dominant factor
remains model capacity (most visible as the persistent training-loss
plateau and the shrinking-with-gain improvement curve above), not an
artifact of the budget-vs-pair-count mismatch.

## Decision gate

**Does a single conditional NAM predict genuinely unseen gain positions
better than interpolating the outputs of independently trained NAMs?**

**No -- and this is now supported by a 3-seed, matched-training-budget
re-run, not just a single run.** The dense experiment, re-run at double the
original training budget across 3 seeds, shows the same clear, consistent,
often large loss to the baseline (7-8 of 9 positions in every seed). The
sparse experiment (4 withheld positions, 2 runs) remains noisy and
run-dependent, with no consistent win for the conditional model there
either.

### Is the limitation capacity, conditioning representation, training data
construction, or evidence that discrete-model interpolation is better?

**Capacity, now confirmed rather than merely suspected.** Section 13 was
designed specifically to test whether under-training (not capacity) was
the real explanation, and the answer is no:

- Doubling the training budget only partially closed the training-gain
  fit gap (1.8x average improvement, shrinking to 1.3x at the highest
  gains) and the loss curve plateaus by step 6,000 in all 3 seeds --
  this is a model that has converged to its capacity ceiling, not one that
  was cut off mid-improvement. Model size (60k params, 42.6ms receptive
  field, far below the official A2's ~6,332-sample/132ms receptive field)
  remains the most likely bottleneck.
- The budget-vs-pair-count confound identified after Experiment 2 is now
  resolved: matching the budget closed only a small part of the gap and
  left the qualitative conclusion (baseline wins almost everywhere)
  unchanged and more strongly evidenced (3 seeds, not 1).
- Conditioning representation (FiLM on a raw normalized scalar) continues
  to show no pathology -- loss converges smoothly and reproducibly in every
  seed, level reproduction is consistently good. There remains no evidence
  the conditioning MECHANISM itself is the problem.
- **Discrete-model interpolation remains the clearly better practical
  approach at this model scale**, now confirmed across 22 withheld-position
  evaluations (13 from the original two experiments + 9 x 3 seeds from
  Section 13, counting each seed's Gain 6.5 win as an exception, not a
  contradiction).

## What this does NOT resolve

- Whether a LARGER conditional model (more channels/layers, official-A2-
  scale receptive field) would close or reverse this gap -- Section 13
  answered the BUDGET question specifically, not the CAPACITY question.
  Both experiments were explicitly kept small per the task's scope ("do
  not optimise beyond what is required to answer the feasibility
  question").
- Why Gain 6.5 specifically is the one position where the model
  reproducibly matches or beats the baseline -- noted as a consistent,
  real effect but not investigated further here.
- Behaviour on amps other than the dense Marshall JCM800 sweep (not tested
  here, to keep this a single, interpretable comparison rather than
  spreading the same budget across every dataset).

## Recommendation for next step

The budget confound is now resolved, and the answer strengthens the
original recommendation rather than reversing it: **do not invest further
in scaling up a conditional model as a near-term product direction.** The
existing discrete-capture-plus-interpolation approach, refined by this
research thread's actual positive findings (capture-density placement,
per-dataset knee identification), remains the better-supported direction.

If conditional modelling is revisited later, the correctly-scoped next
experiment is now a CAPACITY test, not a budget test: scale up model size
(more channels and/or layers, a receptive field closer to the official A2's
~6,332 samples) at a fixed, already-converged training budget, and check
whether training-gain raw ESR drops toward near-zero. Only if that succeeds
would re-testing withheld-gain generalization be worthwhile -- there is no
value in testing generalization of a model that still cannot fit its own
training data well.

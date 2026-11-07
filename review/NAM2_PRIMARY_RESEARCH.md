# NAM A2: primary-source research for the HybridN review

Researched 2026-11-07. This note reviews upstream capabilities, not the local implementation or the user's actual run. Sources on `main` describe current upstream, which can differ from this application's pinned dependencies.

## What A2 does

A2 is the second generation of NAM's **snapshot modeling recipe**. It improves the accuracy/compute tradeoff for reproducing an input-to-output relationship. The official release points integrators to trainer v0.13.0, Core v0.5.2, and plugin v0.7.14. Merely writing a newer NAM file version does not make a trained model A2. [Official release](https://www.neuralampmodeler.com/post/a2-is-released), [architecture plan](https://www.neuralampmodeler.com/post/architecture-a2).

The author explicitly frames NAM as regression against reference audio and favors evaluation on musical DI material. ESR measures sample error normalized by target energy; it does not by itself establish that the target has the intended sound. **Inference:** a very accurate model of an unexpectedly clean hybrid target remains unexpectedly clean. Fixing target construction comes before replacing a network. [Architecture plan](https://www.neuralampmodeler.com/post/architecture-a2).

## Composition is not automatically morphing

`Sequential` runs each complete child in order, feeding each output into the next input. Thus two amp children mean `B(A(x))`, not `(1-w)A(x) + wB(x)`. Amp followed by linear cabinet filtering is a natural use. Replacing a parallel blend with two amps in Sequential would change the sound and drive staging. Core checks channel compatibility and sample rates. [Actual processing implementation, lines 119-131](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/main/NAM/sequential.cpp#L119).

Canonical Sequential files embed complete children, use empty wrapper weights, and require matching sample rates. Earlier exports containing bare child configs and concatenated wrapper weights are explicitly unsupported. A player supporting A2 should still be tested with the actual exported composition, because architecture support and file-version support are distinct concerns. [File specification](https://github.com/sdatkinson/neural-amp-modeler/blob/main/docs/source/model-file.rst#L90).

## Conditioning, packed training, and real gain controls

Upstream WaveNet can contain FiLM and condition DSP. However its standard conditioning input is the audio input; these features do not independently create a user-controlled gain or amp-morph parameter. The runtime API inspected processes audio channels, and its ordinary condition dimension equals the input channel count. [Core WaveNet declaration](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/main/NAM/wavenet/model.h#L82), [FiLM implementation](https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam/models/wavenet/_film.py).

Packed A2 training learns independent models of different sizes against the **same target**, then exports ordinary WaveNets in a `SlimmableContainer`. Its selection controls model size/compute, not clean-to-distorted tone. Current documented packed training excludes FiLM, condition DSP, and multichannel input. [Upstream packed training documentation](https://github.com/sdatkinson/neural-amp-modeler/blob/main/docs/source/tutorials/packed-training.rst).

A separate parametric fork explicitly trains against captures at multiple knob positions. Its ConcatWaveNet appends knob values; HyperWaveNet generates weights from them. Its documentation requires a corresponding parametric player; HyperWaveNet can additionally bake a fixed-setting standard WaveNet. This is evidence that genuine knob-conditioned NAM is possible, not evidence that stock A2 supplies such a control. Adopting it introduces a new training and player compatibility commitment. [Parametric fork README](https://github.com/DLC86/neural-amp-modeler-parametric).

## Practical implications to check locally

1. **Hear the teacher before training.** Render the exact hybrid target for a held-out DI at quiet, normal, and hard playing levels. Compare both source amps using output-only level matching. If the high-gain character is already missing, architecture or epochs cannot fix that target.
2. **Separate drive from loudness.** Preserve source input calibration where available. Output level matching should occur after amp rendering; input scaling changes distortion. Record unknown calibration as unknown. NAM's calibration tutorial specifically warns that insufficient training drive can leave strong-input distortion poorly learned. [Official calibration tutorial](https://neural-amp-modeler.readthedocs.io/en/stable/tutorials/calibration.html).
3. **Define the product behavior.** A fixed blend, a blend selected automatically by playing level, and a gain knob traversing clean to saturated are different contracts. A mono snapshot can learn a deterministic level-dependent blend, but a user-selectable blend needs an explicit supported control or separate snapshots. This is an engineering inference from the interfaces above.
4. **Validate gain behavior separately from global fit.** Compare source, teacher, and final model across an input-level sweep and held-out guitar passages. Measure per-level error, harmonic change, crest factor, and high-gain contribution alongside listening. The current file specification also defines `metadata.gain` as a compression heuristic; it is not a gain control or a sufficient sound-quality verdict. [File specification](https://github.com/sdatkinson/neural-amp-modeler/blob/main/docs/source/model-file.rst#L49).
5. **Upgrade only against a baseline.** Benchmark A2 on the unchanged target with the same held-out material and native renderer. Keep an upgrade if fidelity or practical CPU cost improves. Do not expect it to repair ambiguous blend semantics, wrong normalization, or missing high-gain examples.

These recommendations are candidates for local confirmation. None establishes the root cause of the user's AC30/Mesa export without its database settings, source models, teacher audio, and final-model measurements.

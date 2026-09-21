# Teacher-only weight-band test: predeclared design and criteria

Committed BEFORE any weight-band result was computed. Teacher-only: no NAM is trained, no exported model is used, `hybrid/multi_blend.py` is not modified (the variant weight function lives in analysis scripts and is verified to reproduce `chain_weights` exactly at plateau fraction 0).

**Question.** Can the existing training target be changed so that, at a fixed virtual gain, the captured amp stays close to the SAME physical capture as the guitarist plays harder or softer, without destroying the progression when the Input gain is turned?

**Variant.** For adjacent anchors a < b (levels L_a < L_b, gap d) the current weight uses `smoothstep((env-L_a)/d)`. The band variant holds each anchor's capture at full weight over a plateau and concentrates the transition in the middle of the gap: with plateau fraction phi, `t = smoothstep(clip((p - (0.5 - w/2)) / w, 0, 1))`, `p = (env-L_a)/d`, `w = 1 - phi`. phi = 0 is exactly the current teacher; phi = 1 is a hard switch at the midpoint. Levels tested: phi = 0, 0.5, 0.75, 0.9, 1.0. Configurations: the frozen Phase 4E anchor sets (B primary; A for comparison). Amps: JCM800 and Vibrolux. Cases: the same positions, held-out DIs and musical-input offsets (-12, -6, 0, +6 dB) as the playability diagnostic, and a continuous Input-gain sweep (-24..+16 dB, 2 dB steps) at nominal musical input for the progression.

**Measures (all teacher vs the real capture, signed; nothing composite).** Fixed-gain swing (error at +6 minus error at -12) and fixed offset for level, HF, crest, dynamic range, EQ, tilt; weight-averaged physical position swing and nearest-anchor weight at nominal input; level-matched ESR; progression cost = teacher error at NOMINAL musical input at every physical position, especially positions omitted from the anchor set, and roughness/steps along the continuous Input-gain sweep; switching side effects = crossfade duration statistics, dominant-capture switch rate, and HF energy above 10 kHz relative to the real capture (splatter from abrupt switching).

**A variant is called PROMISING only if, on BOTH amps and for the primary configuration (B), all of these hold (working thresholds, not perceptual):**
1. Mean absolute fixed-gain swing over the tested positions falls by at least 40% relative to phi = 0 for EACH of level, HF, crest and dynamic range.
2. Nominal-input progression cost is bounded: mean error over omitted positions rises by no more than 0.5 dB for each of level, HF, crest, dynamic range, and no single position's error rises by more than 1.5 dB.
3. No switching side effect: median crossfade duration of at least 5 ms, and HF>10 kHz energy excess over the real capture within 1.5 dB of phi = 0.
A variant that passes 1 but fails 2 or 3 is reported as a trade-off, not a remedy. If no variant passes, that is the finding.

**Not done in this test:** any training, any change to anchors or capture sets, G8, new architectures, and any audible claim.

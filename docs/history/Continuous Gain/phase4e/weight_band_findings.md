## Findings

### Result against the predeclared criteria: no variant is promising

- **Criterion 1 (fixed-gain swing falls by at least 40% for each of level, HF, crest and dynamic range) fails for every plateau fraction on both amps, and by a wide margin.** On the JCM800 (configuration B) the change relative to the current teacher at phi = 1 (hard switching) is -6% level, -14% HF, -13% crest, -3% dynamic range (negative = swing got larger); on the Vibrolux it is about -1% level, -1% HF, -7% crest, +1% dynamic range. Configuration A behaves the same (no reduction beyond +5%, and slightly worse for level and HF on both amps). The per-position table shows the same: swing at phi 0.75 and 1.0 is within a few tenths of a dB of phi 0 at every position.
- **Criterion 2 (progression cost at omitted positions) passes**, but only because nothing changed: the mean omitted-position error moves by 0.21 dB or less. That is not a virtue, since criterion 1 already fails. The continuous Input-gain sweep gets more stair-stepped with hard plateaus for the JCM800 in configuration B (worst max/median step ratio 3.9 at phi 0 to 8.7 at phi 1, dynamic-range step 1.38 to 1.53 dB per 2 dB) and barely changes for the Vibrolux (2.2 to 2.4), so a cost appears on one amp without any benefit on either.
- **Criterion 3 (no switching side effects) fails for the harder settings**: the median crossfade shortens to 2.7 ms at phi 0.9 on the JCM800 and 3.8 ms (phi 0.75) and 1.6 ms (phi 0.9) on the Vibrolux, below the 5 ms line, and HF energy above 10 kHz rises relative to phi 0 by up to +0.33 dB (JCM800) and +0.70 dB (Vibrolux) at phi 1, within the 1.5 dB allowance but in the wrong direction.
- **Nothing passes criteria 1-3 together, so the weight-band remedy is not supported.** That is the finding.

### Why reshaping the crossfades cannot work here

The centre-of-mass swing (how far the weighted physical position moves over 18 dB of playing intensity) is unchanged by the plateau fraction (Vibrolux 3.64 at phi 0 and 3.66 at phi 1). The reason is geometric: anchors sit 4.6 to 9.0 dB apart on the Vibrolux (7.9 to 16.6 dB on the JCM800), and playing harder or softer by 6 to 18 dB moves the whole envelope across several of them. A plateau can hold a capture over at most about half a gap on each side, far short of the playing range it would have to absorb. The dominant capture at any moment is the nearest anchor at every phi (switch rate 16.9 per second at all settings on the Vibrolux), so which capture leads is fixed by the envelope, and only the shape of the mix between two of them changes.

### Supplementary: the drift is the size of the real amps' own difference for identical input

No teacher or NAM is involved (real captures only; table above). For each pair of adjacent anchors, the waveform that reaches the NAM when capture a is played `gap` dB harder is the same waveform that reaches it when capture b is played normally, yet the real amps answer differently. On the Vibrolux (B anchors) those differences are 1.7 dB in HF for G2 to G3 and for G3 to G4, 2.2 dB HF plus 1.1 dB level plus 1.1 dB dynamic range for G4 to G7, and 1.3 dB level plus 1.9 dB dynamic range for G7 to G10; they accumulate over an 18 dB change in playing, which is the same order as the 3-4 dB HF and 2.5-3 dB level swings measured on the students. On the JCM800 the top zone is intrinsically unambiguous: G4 to G10 (a 16.6 dB gap) differ by only 0.7 dB in level, 0.3 dB in HF, 0.3 dB in crest and 0.1 dB in dynamic range. So the drift tracks how different neighbouring real captures sound at their spacing (a property of the amp and of the anchor gaps), not the shape of the teacher's crossfade. Any standard NAM trained to reproduce both captures for identical waveforms must compromise by roughly half of this difference.

### What this means for the current training target

The current envelope-driven teacher is not improved by any plateau setting tested, and the identical ambiguity would bind any other way of assigning targets from a single waveform. Two levers remain that change the size of the ambiguity: (1) larger gaps between virtual-gain zones relative to the playing dynamics (fewer zones, or a wider Input-gain span), and (2) choosing zones where neighbouring captures already sound alike (as the JCM800's G4 to G10 do). The smallest experiment that tests this needs no training: score candidate zone layouts built from the existing captures with the same ambiguity table and teacher-only swing test (a teacher-only anchor-layout test), and check the trade-off in progression fidelity at the settings between zones.

The decision that gates it is not only technical. It is whether the real amps' 2-4 dB differences on the Vibrolux are audible enough to matter to a guitarist when they play harder at a fixed setting. That is what the listening evaluation (still pending) can answer, and it is cheaper than any further build: a listening set built for this specific question (same virtual gain, soft/normal/hard playing, real amp versus B versus v3 C3) would show whether the drift is objectionable before anything else is changed.

### Recommendation (not implemented)

1. Do not adopt any weight-band variant.
2. Do not add G8 or train another model; the drift is not caused by too few captures (the C10 teacher shows this) and not by the crossfade shape.
3. The most informative next step is a listening check of the fixed-gain cases, using the existing models. If the drift is not objectionable, the current models already provide the intended experience apart from the top-end hard-playing crest issue and validation in the official plugin. If it is, the next teacher-only test is the anchor-layout test above, with the number of virtual-gain zones as a product decision.

### Limits of this test

Teacher-only; three DIs at four global offsets; measures are aggregate features, not audible judgements; the variant family is one parameterisation of the crossfade (a non-monotone or level-dependent weighting was not explored); the ambiguity table uses two-anchor gaps and does not capture multi-anchor accumulation exactly; NAMCore was not needed for the teacher and no student behaviour is inferred.

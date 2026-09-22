
## Decision, selection and packaging

### Frozen selection rule applied
| Amp | FC vs frozen criteria | Recommended |
|---|---|---|
| JCM800 | C1 pass (student-vs-teacher ESR at total level >= +8 dB: FC 0.009 vs B 0.022), C2 pass, C4 pass, C5 pass, C6 pass; **C3 FAIL** (flagged soft-to-hard combinations FC 8 / 7 vs B 6 / 2; limit B mean + 2 = 6) | **B seed 0** (`jcm800_P4E_B_s0.nam`) |
| Vibrolux | C1, C2, C3, C5, C6 pass; **C4 FAIL** (FC seeds differ in HF error by 0.42 dB, more than B's 0.26 dB) | **B seed 0** (`vibrolux_P4E_B_s0.nam`) |

The new configuration did what it was designed to do: at high total level (Input gain plus playing level above +8 dB) the trained model tracks its teacher 2.3-2.6x more closely, and at +20 dB or more the ESR is 4x lower. It did not improve the native-level, tone or soft-to-hard measurements beyond seed noise, and it added flagged JCM800 combinations. The frozen rule therefore keeps B. FC is not discarded: both FC seeds are packaged as alternatives (`alt_FC_s*`) because the Vibrolux FC seed 1 is the best model on most Vibrolux numbers, and the rule's C4 failure is a seed-reproducibility failure, not a quality one.

**Listening caveat (important).** The only human listening so far (10 items, one listener) rated the older v3 C3 model closer to the real amp than B, especially on the Vibrolux (see `docs/phase4e/listening_fixed_gain/RESULTS_first_listening.md`). The measurements in this document did not predict that result. The selection above follows the frozen, measurement-based rule; it is **not** a perceptual recommendation. v3 C3 is therefore packaged as a listening alternative, and a blind FC listening page (v3 C3 / B s0 / FC s0 / FC s1 / hidden real; 10 items) is ready at `work/p4e/final/listening/listening_tool_fc.html`. Its key must stay closed until it has been rated. Until then, no perceptual claim is made for FC or for the recommended B model.

### Player guide (ordinary NAM player, Input gain only)
Practical range **-20 to +14 dB** (player minimum -20 dB; the B anchors start at -22 dB, so -20 is 2 dB hotter than the G1 anchor, which the assessment showed is within tolerance). Output gain: JCM800 B **+4.7 dB**, Vibrolux B **0 dB** (JCM800 FC would need +2.8 dB). Levels are matched to the real amp at native output.

| Sound | JCM800 (B) Input gain | Vibrolux (B) Input gain |
|---|---|---|
| Clean | -20 to -14 | -20 to -14 |

(The knob ranges in this table are approximate readings of the anchor mapping, not measured boundaries.)

| Breakup | about -10 | -9 to -5 |
| Crunch | about -3 | about +5 |
| Saturated | +10 to +14 | +12 to +14 |

Training anchors (Input gain in dB at the source capture): JCM800 G1 -22.0, G2 -10.5, G4 -2.6, G10 +14.0; Vibrolux G1 -22.0, G2 -17.4, G3 -8.4, G4 -3.1, G7 +5.3, G10 +14.0. Sounds between anchors are interpolations that the model learned, not captures.

### Known weaknesses
- Soft playing is poor for every model in the earlier listening (all models), and hard picking at the extremes is not better.
- B seeds differ from each other perceptually by about 15 points per item in the first listening; seed choice matters.
- Above about +14 dB total level (a high Input gain combined with hard playing) B drifts from its own teacher (ESR up to 0.05); FC does not.
- The model cannot separate Input gain from playing intensity (both move the same envelope), so soft playing at a high Input gain is not the same as a low gain with hard picking.
- Vibrolux has 22 of 32 flagged soft-to-hard combinations for B (19 for FC): the clean Fender's dynamics are the weakest area.

### Validation status
Technical validation (this document): complete. Human listening: v3 C3 vs B done for 10 items (C3 preferred); **FC not heard**, recommended B models only partly heard. Answer to the stopping question: technically a single NAM covers the useful range with no opposite-direction sweep steps, but the listening result means the claim "switching between ten NAMs is no longer needed" is unproven until the FC/B/C3 listening is done.

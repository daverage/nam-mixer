# First listening result: fixed-virtual-gain shortlist (10 items, one listener)

Data: `results/fixed_gain_listening_results_2026-09-20.csv` (exported from `listening_tool_short.html`; listener on speakers, native-level mode only). Analysis: `scripts/pl_listening_analyze.py` (unblinded with the key after rating). **Untouched sliders are scored 50, as the listener confirmed** (the page originally omitted those rows from the export; fixed since: it now exports every candidate of a visited item with `touched=0/1`). Both versions are shown below so the effect of that assumption is visible. Scale: 0-100 closeness to the real amp at the same setting and playing intensity (for the sequence: how well the candidate keeps the real amp's character from soft to hard).

## Bottom line

**By ear, on this shortlist, the old v3 C3 was rated closer to the real amp than the new Phase 4E B model, most clearly on the Vibrolux and at normal picking.** This does not support the earlier conclusion (from aggregate feature measurements) that B is the more useful single NAM. It is one listener, ten items, so it is strong evidence about these particular clips and a warning about the measurements, not a final verdict.

## Results (missing = 50 | observed ratings only)

| Model | mean rating | (observed only) |
|---|---:|---:|
| v3 C3 (old) | 77.4 | 80.4 |
| Phase 4E B seed 0 | 61.6 | 62.9 |
| Phase 4E B seed 1 | 61.7 | 64.6 |
| hidden real amp (attention check) | 83.2 | 91.5 |

- Paired per item, B (mean of seeds) minus v3 C3: **-15.8 points; C3 better in 7 of 10 items, B better in 3.** Best candidate per item: C3 5.5, B seed 0 3.5, B seed 1 1.0.
- **By amp:** JCM800 C3 88.5, B 76.0 / 72.2 (hidden real 91.0; without the untouched sliders B is 84.7 / 79.7, i.e. close to C3); **Vibrolux C3 70.0, B 52.0 / 54.7** (hidden real 78.0).
- **By playing intensity (mean over items):**

| Model | soft | normal | hard | soft-to-hard sequence (1 item) |
|---|---:|---:|---:|---:|
| v3 C3 | 58.0 | 87.8 | 87.5 | 74.0 |
| B seed 0 | 59.0 | 59.5 | 88.0 | 25.0 |
| B seed 1 | 56.7 | 68.8 | 70.0 | 32.0 |

  Soft playing is poor for every model (about 57-59). At normal picking B is clearly farther than C3. At hard picking B seed 0 matches C3 (88.0 vs 87.5) but B seed 1 does not (70.0).
- **Per item** (`*` = untouched slider counted as 50):

| Item | hidden real | v3 C3 | B seed 0 | B seed 1 |
|---|---:|---:|---:|---:|
| Vibrolux G5 normal | 97 | 95 | 40 | 91 |
| Vibrolux G7 hard | 50* | 84 | 85 | 50* |
| Vibrolux G7 soft | 99 | 50* | 87 | 63 |
| Vibrolux G7 soft-then-normal-then-hard | 72 | 74 | 25 | 32 |
| Vibrolux G10 normal | 50* | 90 | 35 | 35 |
| Vibrolux G10 soft | 100 | 27 | 40 | 57 |
| JCM800 G5 hard | 92 | 91 | 91 | 90 |
| JCM800 G5 normal | 96 | 76 | 92 | 88 |
| JCM800 G8 normal | 89 | 90 | 71 | 61 |
| JCM800 G8 soft | 87 | 97 | 50* | 50* |

- **Issues ticked:** B seed 0 and seed 1: "too distorted / saturated" at Vibrolux G7 soft and G10 soft and normal, "boomy" at G10 normal; "not saturated enough / too clean" at Vibrolux G7 sequence and JCM800 G8 normal. v3 C3: "too distorted / saturated, boomy, loudness wrong" at Vibrolux G10 soft (where it scored 27, the only place it clearly lost); "loudness wrong" in the G7 sequence.

## How far to trust it

- **Noise floor.** The hidden real amp itself was rated 72 in the sequence item and (with untouched sliders scored 50) 50 in two others; it was rated best in 50% of items (62% observed only). So differences below roughly 10-20 points are within this listener's noise here. The large gaps are well outside it: Vibrolux G10 normal (C3 90, B 35 / 35), Vibrolux G5 normal (C3 95, B seed 0 40), the G7 sequence (C3 74, B 25 / 32).
- **Seed variation is large.** The two B seeds differ by 14.9 points on average per item and by 51 on Vibrolux G5 normal (40 vs 91). Perceived quality is not stable across seeds even where the aggregate measurements were.
- **Limits:** one listener, speakers, native-level only (loudness differences may have influenced ratings), ten items, one DI per amp, the sequence conclusion rests on a single item, and the untouched-slider assumption.

## Against the earlier measurements

- The automated preview (`AUTOMATED_PREVIEW.md`) only partly matched: it predicted C3 closer at Vibrolux G5 and JCM800 G8 soft (matched) and B closer at Vibrolux G10 soft (matched) but also B closer at Vibrolux G10 normal, where the listener rated B 35 against C3 90. The soft-to-hard behaviour errors that favoured B on paper did not survive the one sequence rated.
- A check of the rated clips (band-by-band level-independent differences against the real amp) does not explain the big gaps. At Vibrolux G10 normal all three candidates are within about 1.2 dB of the real amp in every band (B seed 1 differs mainly in crest factor, +2.3 dB, versus +0.3 dB for C3 and B seed 0), yet the ratings are 90 versus 35 / 35. So whatever is audible there is not captured by the aggregate spectral and dynamic features used so far.
- Hypotheses only (none tested): the trained model departs from its own target at the top gain (the playability diagnostic measured the student drifting from the teacher at G10); the envelope-driven target changes character during note decay, which a fixed-gain amp does not; and the distortion texture or seed-level training differences matter more than band energies.

## What this changes

The technical claim that the Phase 4E B configuration produces a more useful single NAM than v3 C3 is not supported by this listening on the Vibrolux and is at best neutral on the JCM800. Hard-picking and the JCM800 G5 items are the encouraging cases. The simpler v3 C3 is currently rated closer at normal picking, and soft playing is a weakness for both.

## Decisions and possible next steps (not started)

1. **Second pass, level-matched (about 10 minutes):** the same ten items with the "Level-matched" toggle, to see whether loudness differences drove the ratings.
2. **Teacher-versus-student listening (no training):** for the four failing clips (Vibrolux G10 normal, G5 normal seed 0, G7 sequence, JCM800 G8 normal) render the training target itself next to the real amp, C3 and B. If the target already sounds like the real amp and the trained model does not, the problem is in training/generalisation at that setting; if the target already sounds different, it is the target construction. This is analysis of existing renders, not a new experiment or model.
3. **Live guitar test (`LIVE_GUITAR_TEST_QUICK.md`)** for feel, which no clip can show.

# Live guitar test, QUICK version (about 15-20 minutes)

Same idea as `LIVE_GUITAR_TEST.md` (guitar into a normal NAM player, ONE standard `.nam`, Input knob FIXED, change only how you play) but restricted to the five settings the automated preview flagged (`AUTOMATED_PREVIEW.md`). Perceptual result: pending until you do it.

## Setup (once)

1. Two slots in your NAM player. **Reference:** the original capture for the setting (file names below), Input gain 0 dB. **Model under test:** swap between `docs/phase4e/models/vibrolux_P4E_B_s0.nam` (B) and `docs/phase4e/models/v3/vibrolux_ContinuousGain_3Captures.nam` (v3 C3); JCM800: `jcm800_P4E_B_s0.nam` and `v3/jcm800_ContinuousGain_3Captures.nam`.
2. Output level: the Vibrolux models need nothing; for the JCM800 models raise the player's Output gain by **+4.7 dB** (training constant). Match once, then don't touch outputs again.
3. Turn off any automatic level/input compensation from file metadata, or make sure it is identical for the reference and both models.
4. Optional but better: ask someone to load the two models under neutral names (X and Y) so you don't know which is which.

## The five settings (do them in this order)

| # | Amp | Virtual gain | Input gain: B (dB) | Input gain: v3 C3 (dB) | Reference capture | Why it is on the list | Play |
|---:|---|---:|---:|---:|---|---|---|
| 1 | Vibrolux | 10 | +14.0 | +14.0 | `Super-Sonic Vibrolux Ch T5 B5 V10.nam` | B is predicted much closer than v3 C3 when played soft or normal | soft and normal picking: which model stays closer to the reference at full gain? |
| 2 | Vibrolux | 7 | +5.3 | +2.0 | `Super-Sonic Vibrolux Ch T5 B5 V7.nam` | B is predicted better at normal and hard, worse when soft; the soft-to-hard sequence is where B should behave best | play the same phrase soft, then hard, without touching Input gain |
| 3 | JCM800 | 8 | +9.6 | +6.0 | `jcm800-high-g8.0-11.4dBu.nam` | the largest predicted problem for B: soft playing is much farther than v3 C3, and neither model was trained on G8 | soft picking, then normal |
| 4 | Vibrolux | 5 | -0.5 | -6.0 | `Super-Sonic Vibrolux Ch T5 B5 V5.nam` | v3 C3 is predicted closer (it was trained on G5, B was not): is that audible? | normal picking |
| 5 | JCM800 | 5 | -0.8 | -6.0 | `jcm800-high-g5.0-11.4dBu.nam` | same check on the JCM800 | normal and hard picking |

## For each setting (about 3 minutes)

1. Set the Input gain from the table and **leave it**. Pick one short phrase or riff and use the same one for all three sounds.
2. Play it soft, then normal, then hard on the REFERENCE. Note how the real amp changes as you dig in (it gets louder, brighter, more compressed).
3. Play the same three on X, then on Y. Do not touch the Input knob. Optionally repeat once with your guitar's volume knob rolled back a third.
4. Fill in one row of the sheet below.

## Scoring sheet (copy this table; scores 1 = not at all, 5 = indistinguishable from the reference)

| # | Setting | X: same amp when soft | X: normal | X: hard | X: keeps its character soft-to-hard | Y: soft | Y: normal | Y: hard | Y: keeps character | Feel (attack, bloom, squash): which is closer? | Clicks/jumps/artefacts? | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | Vibrolux 10 |  |  |  |  |  |  |  |  |  |  |  |
| 2 | Vibrolux 7 |  |  |  |  |  |  |  |  |  |  |  |
| 3 | JCM800 8 |  |  |  |  |  |  |  |  |  |  |  |
| 4 | Vibrolux 5 |  |  |  |  |  |  |  |  |  |  |  |
| 5 | JCM800 5 |  |  |  |  |  |  |  |  |  |  |  |

After the five, note which of X and Y was B and which was v3 C3 (if you used neutral names), and answer: **when you dig in or back off at a fixed Input setting, does it still feel like that amp?** A few sentences is enough.

## What to send back

The filled sheet or a paraphrase of your notes. If the results disagree with the automated preview (e.g. JCM800 G8 soft sounds fine), that is itself useful.

Limits: five settings, one guitar, your own touch; the plugin's nominal Input range may not reach very low settings but none of these five is below -6 dB.
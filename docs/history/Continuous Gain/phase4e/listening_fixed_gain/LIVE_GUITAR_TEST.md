# Live guitar test: does the NAM stay that amp when you change how you play?

The clips show tone and dynamics; they cannot show **playing feel**. The closest test of the experience we want to deliver is: guitar into an ordinary NAM player, ONE standard `.nam`, the Input knob fixed, and only your picking, guitar volume knob or pickup changing. Do this after (or alongside) the blind clips. Status of any result: pending until you do it.

## Setup

1. Load the model in your NAM player: `docs/phase4e/models/vibrolux_P4E_B_s0.nam` (and, for comparison, `docs/phase4e/models/v3/vibrolux_ContinuousGain_3Captures.nam`; the JCM800 equivalents are `jcm800_P4E_B_s0.nam` and `v3/jcm800_ContinuousGain_3Captures.nam`).
2. **Reference amp:** load the original capture for the setting you are testing in a second player instance/slot, with Input gain 0 dB (it IS that physical amp setting). File names are below.
3. Match output levels once, and do not touch them again. The Vibrolux models need no output compensation (training constant 1.0). The JCM800 models output about 4.7 dB quieter than the real amp: raise the player's Output gain by +4.7 dB for them.
4. Set the model's **Input gain** to the value in the table for the virtual setting, then LEAVE IT. Change only how you play.
5. If your player applies its own input/output calibration from the .nam metadata (auto level compensation), either turn it off or make sure it is applied identically to every model and to the reference, so that only the trained behaviour differs.

## Input gain per virtual setting (dB)

The two models use different mappings; each is the model's own intended mapping. The official plugin's nominal Input range may stop at about -20 dB, so virtual gain 1 (-22 dB) may not be reachable; use -20 and note it. **Physical reference capture files** are listed for each setting.

### Marshall JCM800

| Virtual gain | Input gain: Phase 4E B (dB) | Input gain: v3 C3 (dB) | Reference capture file |
|---:|---:|---:|---|
| 1 | -22.0 | -22.0 | `jcm800-high-g1.0-11.4dBu.nam` |
| 2 | -10.5 | -18.0 | `jcm800-high-g2.0-11.4dBu.nam` |
| 3 | -5.4 | -14.0 | `jcm800-high-g3.0-11.4dBu.nam` |
| 4 | -2.6 | -10.0 | `jcm800-high-g4.0-11.4dBu.nam` |
| 5 | -0.8 | -6.0 | `jcm800-high-g5.0-11.4dBu.nam` |
| 6 | +1.2 | -2.0 | `jcm800-high-g6.0-11.4dBu.nam` |
| 7 | +4.8 | +2.0 | `jcm800-high-g7.0-11.4dBu.nam` |
| 8 | +9.6 | +6.0 | `jcm800-high-g8.0-11.4dBu.nam` |
| 9 | +12.1 | +10.0 | `jcm800-high-g9.0-11.4dBu.nam` |
| 10 | +14.0 | +14.0 | `jcm800-high-ga10-11.4dBu.nam` |

### Fender Super-Sonic Vibrolux

| Virtual gain | Input gain: Phase 4E B (dB) | Input gain: v3 C3 (dB) | Reference capture file |
|---:|---:|---:|---|
| 1 | -22.0 | -22.0 | `Super-Sonic Vibrolux Ch T5 B5 V1.nam` |
| 2 | -17.4 | -18.0 | `Super-Sonic Vibrolux Ch T5 B5 V2.nam` |
| 3 | -8.4 | -14.0 | `Super-Sonic Vibrolux Ch T5 B5 V3.nam` |
| 4 | -3.1 | -10.0 | `Super-Sonic Vibrolux Ch T5 B5 V4.nam` |
| 5 | -0.5 | -6.0 | `Super-Sonic Vibrolux Ch T5 B5 V5.nam` |
| 6 | +2.9 | -2.0 | `Super-Sonic Vibrolux Ch T5 B5 V6.nam` |
| 7 | +5.3 | +2.0 | `Super-Sonic Vibrolux Ch T5 B5 V7.nam` |
| 8 | +8.6 | +6.0 | `Super-Sonic Vibrolux Ch T5 B5 V8.nam` |
| 9 | +11.5 | +10.0 | `Super-Sonic Vibrolux Ch T5 B5 V9.nam` |
| 10 | +14.0 | +14.0 | `Super-Sonic Vibrolux Ch T5 B5 V10.nam` |

## What to do, per setting (start with Vibrolux 3, 5, 7 and JCM800 2, 8)

1. Play the same phrase or riff **softly**, at your **normal** touch, then **hard**. First on the reference capture, then on the B model, then on v3 C3, without touching the Input knob.
2. Repeat with your guitar's **volume knob** rolled back a third and then a half (the same idea: quieter musical input at the same Input gain), and with a hotter and a weaker pickup if you have them.
3. Notes to make for each: Does it still sound like the SAME amp as you dig in, or does it drift toward a cleaner/dirtier one? Does the **attack** and **bloom** feel like the reference? Does hard picking compress or fizz differently from the reference? Does softer playing clean up the way the reference does? Any clicks or sudden character jumps while dynamics change (crossfade artefacts)?
4. Then turn the Input knob deliberately and check the progression from clean to saturated still feels continuous and useful.
5. Record a few takes if you can (same phrase, soft/normal/hard) so they can be compared later or added to the blind set.

## Known limits to keep in mind

The models are 48 kHz A2 NAMs trained on the level-driven target described in `docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md`; the measured drift is largest on the Vibrolux mid range (G3-G8) and small on the JCM800 from G4 up. Real playing feel also depends on latency, your guitar and the player's Input-gain implementation, none of which the clips capture.

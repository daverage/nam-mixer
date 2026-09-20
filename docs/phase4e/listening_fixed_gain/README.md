# Fixed-virtual-gain listening: does it still sound like that amp when you play softer or harder?

**Status: PERCEPTUAL EVALUATION PENDING. No human has listened to this set.** Built from the existing Phase 4E models; nothing was trained for it.

## What you are judging
A guitarist selects a virtual amp-gain setting (the player's Input gain), leaves it there, and then plays softly or hard. Does the NAM still sound and feel like THAT amp? Each item fixes the Input gain and compares, against the real amp capture at the same setting, three candidate models (old v3 C3, the new Phase 4E B seed 0, B seed 1) plus a hidden copy of the real amp (a check on attention). Everything is blind: the candidates are lettered A-D at random, differently for every item, and the key is in a separate file that you should not open until you have finished.

- **Single items (24):** the same 5-second phrase played soft (-12 dB), normal, or hard (+6 dB) at the same Input gain. Rate how close each candidate is to the reference.
- **Sequence items (8):** one clip with soft, then normal, then hard playing. Listen to how the REFERENCE changes as the player digs in, then rate each candidate on whether it keeps that same behaviour (a real amp also gets more saturated and compressed when played harder; the failure is a DIFFERENCE from the real amp).
- **Amps and settings:** Vibrolux (clean guitar) at virtual gains 3, 5, 7, 10; JCM800 (driven guitar) at 2, 5, 8, 10. The measured problems were largest on the Vibrolux mid range and small on the JCM800 from G4 upward, so both are included.
- **Two versions of every clip:** *native level* keeps the real loudness differences (judge loudness progression); *level-matched* removes them (judge tone, saturation and dynamics). Use the toggle in the page; ideally hear each item both ways.

## How to run it
1. Open `work/p4e/listening_fixed_gain/listening_tool.html` in a browser (double-click). Audio files are in the same folder. Use good headphones at a fixed volume.
2. Go through the items in order (the order is randomised). Rate each of A-D 0-100, tick anything that sounds wrong, add notes.
3. Click **Export results (CSV)**. Ratings are also kept in your browser between sessions.
4. Analyse with `python scripts/pl_listening_analyze.py <your_csv>`; it opens the key and prints results per candidate, playing intensity and amp, and whether the hidden reference was recognised.
Alternatively fill in `listening_worksheet.csv` by hand.

## Where the audio came from (reproducible)
`scripts/pl_listen_stage.py <amp>` then `scripts/pl_listen_finalize.py` (seeded). Held-out DIs only: `clean_mayer` for the Vibrolux, `moderate_brit` for the JCM800; 5 s single clips and 13 s sequences chosen deterministically as the windows with the widest natural dynamics. Each model is played at its OWN intended mapping: B at the response-distance Input gain, v3 C3 at the fixed rule (see `LIVE_GUITAR_TEST.md` for the numbers). One common gain per item keeps relative levels; PCM 16-bit, 48 kHz, mono.

## Limits
One DI per amp; three playing intensities (-12, 0, +6 dB) are global level changes of a recorded performance, not a real player changing touch; clips show tone, saturation and dynamics but **cannot establish playing feel** (see the live-guitar test). Raw model output for the JCM800 is scaled by a training constant (see the live guide); the clips already undo it.

## Shortlist version (10 items)
`python scripts/pl_listen_short.py` builds `work/p4e/listening_fixed_gain/listening_tool_short.html`: ten of the 32 items chosen by the automated preview (Vibrolux 10 soft/normal, Vibrolux 7 soft/hard/sequence, JCM800 8 soft/normal, Vibrolux 5 normal, JCM800 5 normal/hard), in that order. Same audio, same blind labels and key, same CSV format, so `pl_listening_analyze.py` works on either page's export; ratings are stored separately from the full page. About 15 minutes.

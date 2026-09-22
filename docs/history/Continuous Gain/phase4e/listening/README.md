# Phase 4E listening package (STATUS: PERCEPTUAL EVALUATION PENDING - no human listening has happened)

- `<amp>/reference/`: the real capture (labelled REF) for every stimulus. `<amp>/blind_native/` and `<amp>/blind_levelmatched/`: three candidates per stimulus with randomised labels X1-X3 (old v3 C3, new A seed 0, new B seed 0; key in `listening_KEY_do_not_open_before_listening.csv`).
- Native-level files preserve relative loudness between reference and candidates (one common gain per stimulus to avoid clipping); level-matched files match each candidate's RMS to the reference so tone, saturation and dynamics can be judged without loudness dominating.
- Settings: cleaner (G2), transition (G4), saturated (G7), top (G10); soft (-12 dB) and hard (+6 dB) picking at transition and saturated; DIs clean_mayer and moderate_brit (held-out, never in training). Each candidate is played at the intended playback mapping of its own configuration (v3 C3 and A: fixed rule; B: response-distance rule).
- Fill in `listening_worksheet.csv` (which candidate is closest to the reference, what is undesirable, loudness progression). Open the key only afterwards.

# Official NAM v3.0.0 training input

`official_nam_v3_input.wav` is the official Neural Amp Modeler reamp/training
excitation signal, version 3.0.0 -- the same file the real
`neural-amp-modeler` trainer (`nam.train.core._detect_input_version`)
recognizes by exact MD5 match against its own table of known official inputs
(v1.0.0, v1.1.1, v2.0.0, v3.0.0, Proteus). See
[sdatkinson/neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler)
for the project that publishes it.

- MD5: `36cd1af62985c2fac3e654333e36431e` (matches
  `hybrid/training_target.py`'s `OFFICIAL_V3_INPUT_MD5` exactly -- that
  constant is the actual verification this project performs; this README is
  documentation, not the check itself).
- Sample rate: 48 kHz, mono.

## Why it's bundled

`app.py` seeds a fresh `work/training_input/input.wav` from this file on
first run if nothing has been uploaded yet (a plain local file copy, never a
network fetch -- see the comment next to `TRAINING_INPUT_PATH` in `app.py`),
so generating a training bundle works immediately without a manual upload
first. Uploading a different file through the UI still overwrites it, same
as before this existed.

## What this is NOT

This is **not** the same thing as `assets/di/*.wav` (see that directory's
README and the main project README's "Preview DIs vs. NAM training
material"). The DI library is genre/style musical performance clips for
auditioning; this file is the calibrated reamp signal actually used to
generate a real A2 training target.

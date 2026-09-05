"""Hybrid NAM Builder core package.

See the top-level README for the concept. Modules:
- nam_loader: parse .nam files and their calibration metadata.
- render: run audio through a loaded NAM model (NOT YET IMPLEMENTED, see module).
- envelope: dry-input level/envelope extraction that drives the crossover.
- level_match: automatic Amp A/B trim calculation focused on the crossover region.
- align: sample-offset detection/correction between two amp renders.
- blend: the actual dynamic crossfade (smoothstep by default, replaceable).
- safety: NaN/clip checks and non-limiting peak-ceiling gain staging.
- metadata: the JSON sidecar schema describing how a hybrid target was generated.
"""

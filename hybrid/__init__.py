"""NAM Mixer core package.

See the top-level README for the concept. Subpackages:
- core: NAM I/O (nam_loader, render) and shared signal processing (envelope,
  level_match, align, safety, cab_ir, receptive_field, pipeline, ...).
- modes: the three design modes -- Dynamic Hybrid (blend, design,
  training_target), Parallel Blend (fixed_blend, blend_training_target) and
  Character Blend (character_*).
- continuous_gain: the Continuous Gain tab (one amp, N captures -> one .nam).
- training: A2 training settings, local/Kaggle backends, export packaging,
  and validation.
- services: settings/.env, the local recipe assistant, research lookups,
  update checks.
`paths` stays at this top level because it derives the repo root from its
own location.
"""

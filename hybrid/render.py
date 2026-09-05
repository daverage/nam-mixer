"""Rendering audio through a loaded NAM model.

STATUS: NOT YET IMPLEMENTED. This module defines the interface the rest of the
pipeline (blend.py, app.py) is written against, so that wiring in real inference
later is a drop-in change rather than a redesign.

Why it's not implemented yet: turning a NamModel's raw `architecture`/`config`/
`weights` dicts (see nam_loader.py) into a runnable model requires instantiating
the correct architecture class from the `neural-amp-modeler` package (WaveNet,
LSTM, ConvNet, ...) and loading its weights, then running the network in torch.
That mapping is not a small, obviously-correct guess -- the on-disk `.nam` schema
has changed across `neural-amp-modeler` versions, and getting it wrong would
silently produce plausible-looking but wrong audio, which is worse than an
explicit "not implemented" error. Since torch and neural-amp-modeler are not
installed in this environment, that mapping has not been verified against a real
.nam file and a known-good render, so it has deliberately not been guessed at here.

To implement this:
1. `pip install torch neural-amp-modeler` (already in requirements.txt).
2. Import the model classes from the `nam` package (e.g. `nam.models`) and find
   the loader NAM's own trainer/plugin code uses to go from a parsed .nam JSON
   dict to a runnable `nam.models.base.BaseNet` instance.
3. Implement `render()` below using that loader, keeping the same signature.
4. Verify against a known .nam + reference render (e.g. one of the NAMtoClo
   test_assets renders, if available) before trusting the output.
"""
from __future__ import annotations

import numpy as np

from .nam_loader import NamModel


class RenderNotImplementedError(NotImplementedError):
    """Raised by render() until real NAM inference is wired in. See module docstring."""


def render(model: NamModel, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Render `audio` (mono float32, at `sample_rate`) through `model`.

    Intended contract once implemented:
    - Input and output are both mono float32 numpy arrays of the same length
      (NAM models are sample-rate-preserving, causal, no added latency assumed).
    - Resampling `audio` to `model.sample_rate` (if known and different) is the
      caller's responsibility, not this function's -- keeps this function a pure
      "run the network" call.
    - Should process in chunks internally if needed for memory; not exposed here.

    Currently always raises RenderNotImplementedError -- see module docstring.
    """
    raise RenderNotImplementedError(
        f"NAM inference for architecture={model.architecture!r} is not implemented yet. "
        "See hybrid/render.py module docstring for what's needed to wire it in."
    )

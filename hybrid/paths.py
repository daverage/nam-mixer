"""The repository root, for modules that locate files relative to the checkout.

This file must stay directly inside `hybrid/` (not in a subpackage), because
the root is derived from its own location. The packaged desktop app copies
`hybrid/` with the same layout (see packaging/backend/nam_mixer_backend.spec),
so there REPO_ROOT resolves to the bundle root instead.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

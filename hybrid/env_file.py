"""Shared, minimal `.env` reader used by the optional local-AI helpers.

Only ever consulted as a fallback for names the caller allow-lists -- never
overrides an already-set environment variable, and never used for anything
outside this repo's own opt-in local-LLM/TONE3000 settings.
"""
from __future__ import annotations

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def read_env_value(name: str) -> str:
    """Return `name` from the process environment, falling back to `.env`."""
    import os

    # An explicitly empty inherited variable must not mask a valid `.env`
    # fallback. Launchers commonly pass optional variables as empty strings.
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if not ENV_FILE.is_file():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return ""

"""Shared, minimal `.env` reader/writer used by the optional local-AI helpers
and the standalone desktop app's Settings page.

Only ever consulted as a fallback for names the caller allow-lists -- never
overrides an already-set environment variable, and never used for anything
outside this repo's own opt-in local-LLM/TONE3000/render-path settings.

The file location defaults to the repo root, but can be overridden via
`NAM_MIXER_ENV_FILE` -- the standalone desktop app (desktop/main.py) points
this at a per-user config directory instead, since a frozen app's install
directory (e.g. Program Files) is often not writable.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_file() -> Path:
    override = os.environ.get("NAM_MIXER_ENV_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / ".env"


def read_env_value(name: str) -> str:
    """Return `name` from the process environment, falling back to `.env`."""
    # An explicitly empty inherited variable must not mask a valid `.env`
    # fallback. Launchers commonly pass optional variables as empty strings.
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_file = _env_file()
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return ""


def read_env_values(names: "set[str]") -> "dict[str, str]":
    """Batch form of read_env_value, reading the .env file only once."""
    result = {name: os.environ.get(name, "").strip() for name in names}
    remaining = {name for name, value in result.items() if not value}
    env_file = _env_file()
    if remaining and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, raw_value = stripped.partition("=")
            key = key.strip()
            if key in remaining:
                result[key] = raw_value.strip().strip('"').strip("'")
    return result


def write_env_values(values: "dict[str, str]") -> Path:
    """Merge `values` into the `.env` file, updating existing keys in place.

    A value of "" (or containing only whitespace) removes that key from the
    file rather than writing an empty assignment, so clearing a setting in
    the UI actually falls back to any real environment variable/default
    again instead of pinning an empty string.

    Also updates `os.environ` for keys removed this way and returns the
    env-file path written to, so the caller can report where it lives.
    """
    env_file = _env_file()
    existing_lines: list[str] = []
    if env_file.is_file():
        existing_lines = env_file.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    output_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in values:
            seen.add(key)
            new_value = values[key].strip()
            if new_value:
                output_lines.append(f"{key}={new_value}")
            # else: drop the line entirely (key cleared)
        else:
            output_lines.append(line)

    for key, value in values.items():
        if key in seen:
            continue
        value = value.strip()
        if value:
            output_lines.append(f"{key}={value}")

    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(output_lines) + "\n" if output_lines else "", encoding="utf-8")

    for key, value in values.items():
        value = value.strip()
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

    return env_file

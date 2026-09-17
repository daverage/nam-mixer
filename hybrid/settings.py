"""Registry of user-configurable environment-variable settings.

This exists mainly for the standalone desktop app (see desktop/main.py):
someone running a double-clicked app has no shell to `export` an env var
into, so anything that would normally be set that way needs a Settings page
instead. The web/`python app.py` workflow gets the same page for free.

Every setting here is backed by the existing `.env` fallback mechanism in
hybrid/env_file.py -- this module adds no new storage, just describes what's
safe to expose in the UI and how. Saving a value writes it to `.env` (or
`NAM_MIXER_ENV_FILE`, see env_file.py) and also updates the current
process's os.environ so most settings take effect immediately, without a
restart -- the exceptions are flagged via `restart_required` below (anything
read only once at process startup, e.g. the server PORT).
"""
from __future__ import annotations

from dataclasses import dataclass

from hybrid.env_file import read_env_values, write_env_values


@dataclass(frozen=True)
class SettingField:
    name: str
    label: str
    description: str
    group: str
    kind: str = "text"  # "text" | "path" | "number" | "secret"
    placeholder: str = ""
    restart_required: bool = False


SETTINGS: tuple[SettingField, ...] = (
    SettingField(
        name="NAM_RENDER_EXE",
        label="NAM render executable",
        description="Path to the native nam_render binary used for NAM inference. "
                     "Leave blank to use the bundled/auto-detected one.",
        group="Rendering",
        kind="path",
        placeholder="/path/to/nam_render",
    ),
    SettingField(
        name="NAM_RENDER_SEQUENTIAL_EXE",
        label="NAM render executable (Sequential compatibility gate)",
        description="Optional second nam_render build pinned to the experimental "
                     "Sequential/Linear NAMCore commit -- see native/nam_render/README.md. "
                     "Leave blank unless you are testing that gate.",
        group="Rendering",
        kind="path",
    ),
    SettingField(
        name="PORT",
        label="Server port",
        description="Local port the app listens on. Only used by `python app.py` directly "
                     "(the standalone desktop app always picks a free port automatically).",
        group="Server",
        kind="number",
        placeholder="5001",
        restart_required=True,
    ),
    SettingField(
        name="NAM_MIXER_LOCAL_LLM_BASE_URL",
        label="Local AI assistant: base URL",
        description="OpenAI-compatible endpoint for the optional local recipe assistant "
                     "(e.g. Ollama, LM Studio). Must be http:// and point at localhost -- "
                     "never a remote host.",
        group="AI Assistant",
        placeholder="http://127.0.0.1:11434/v1",
    ),
    SettingField(
        name="NAM_MIXER_LOCAL_LLM_MODEL",
        label="Local AI assistant: model name",
        description="Model name as known to your local server. Leave blank to disable "
                     "the AI Assistant tab entirely.",
        group="AI Assistant",
    ),
    SettingField(
        name="NAM_MIXER_LOCAL_LLM_TEMPERATURE",
        label="Local AI assistant: temperature",
        description="Sampling temperature (0.0-2.0).",
        group="AI Assistant",
        kind="number",
        placeholder="0.2",
    ),
    SettingField(
        name="NAM_MIXER_LOCAL_LLM_TIMEOUT_SECONDS",
        label="Local AI assistant: request timeout (seconds)",
        description="How long to wait for the local model before giving up.",
        group="AI Assistant",
        kind="number",
        placeholder="60",
    ),
)

_KNOWN_NAMES = {field.name for field in SETTINGS}


def get_settings() -> list[dict]:
    """Return every registered setting's current value plus its UI metadata."""
    values = read_env_values(_KNOWN_NAMES)
    return [
        {
            "name": field.name,
            "label": field.label,
            "description": field.description,
            "group": field.group,
            "kind": field.kind,
            "placeholder": field.placeholder,
            "restart_required": field.restart_required,
            "value": values.get(field.name, ""),
        }
        for field in SETTINGS
    ]


def save_settings(values: dict) -> dict:
    """Persist `{name: value}` for known setting names only; silently ignores the rest."""
    filtered = {name: str(value) for name, value in values.items() if name in _KNOWN_NAMES}
    env_file = write_env_values(filtered)
    return {"saved": sorted(filtered), "env_file": str(env_file)}

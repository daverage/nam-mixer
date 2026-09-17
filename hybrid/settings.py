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
        description="Any local LLM host that exposes an OpenAI-compatible /v1 API will work here "
                     "-- Ollama, LM Studio, llama.cpp server, etc. -- this app has no preference. "
                     "Must be http:// and point at localhost -- never a remote host. Nothing needs "
                     "to be running for the rest of the app to work; only the AI Assistant tab uses this.",
        group="AI Assistant",
        placeholder="http://127.0.0.1:11434/v1",
    ),
    SettingField(
        name="NAM_MIXER_LOCAL_LLM_MODEL",
        label="Local AI assistant: model name",
        description="Model name as known to your local server. We recommend Google's Gemma "
                     "(gemma3:4b) as a good balance of speed and quality for this app's recipe "
                     "suggestions, but any chat-capable model your host serves will work. Leave "
                     "blank to disable the AI Assistant tab entirely.",
        group="AI Assistant",
        placeholder="gemma3:4b",
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
    SettingField(
        name="TONE3000_API_KEY",
        label="TONE3000 API key",
        description="Server-side TONE3000 Secret Key (t3k_cs_...) used for the TONE3000 tab's "
                     "capture search. Leave blank to disable that tab. Never logged or returned "
                     "by the API once saved -- see hybrid/research.py's _require_tone3000_api_key.",
        group="TONE3000",
        kind="secret",
        placeholder="t3k_cs_...",
    ),
)

_KNOWN_NAMES = {field.name for field in SETTINGS}
_FIELDS_BY_NAME = {field.name: field for field in SETTINGS}


def get_settings() -> list[dict]:
    """Return every registered setting's current value plus its UI metadata.

    A `kind="secret"` field's real value is never returned -- only whether
    one is currently set (`has_value`) -- so an already-saved API key never
    round-trips back out over the API or onto a screen someone might share.
    """
    values = read_env_values(_KNOWN_NAMES)
    result = []
    for field in SETTINGS:
        raw_value = values.get(field.name, "")
        entry = {
            "name": field.name,
            "label": field.label,
            "description": field.description,
            "group": field.group,
            "kind": field.kind,
            "placeholder": field.placeholder,
            "restart_required": field.restart_required,
        }
        if field.kind == "secret":
            entry["value"] = ""
            entry["has_value"] = bool(raw_value)
        else:
            entry["value"] = raw_value
        result.append(entry)
    return result


def save_settings(values: dict) -> dict:
    """Persist `{name: value}` for known setting names only; silently ignores the rest.

    A blank submitted value for a `kind="secret"` field means "leave it
    unchanged" (the UI never shows the real value to re-submit) rather than
    "clear it" -- clearing a secret requires editing the .env file directly.
    """
    filtered = {}
    for name, value in values.items():
        field = _FIELDS_BY_NAME.get(name)
        if field is None:
            continue
        value = str(value)
        if field.kind == "secret" and not value.strip():
            continue
        filtered[name] = value
    env_file = write_env_values(filtered)
    return {"saved": sorted(filtered), "env_file": str(env_file)}

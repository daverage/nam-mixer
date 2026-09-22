"""Registry of user-configurable environment-variable settings.

This exists so anything normally set via `export FOO=bar` before `python
app.py` can instead be set from the Settings page in the browser, no shell
required.

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

from hybrid.env_file import read_saved_env_values, read_env_values, write_env_values


@dataclass(frozen=True)
class SettingField:
    name: str
    label: str
    description: str
    group: str
    kind: str = "text"  # "text" | "path" | "number" | "secret" | "checkbox"
    placeholder: str = ""
    restart_required: bool = False
    options: tuple[tuple[str, str], ...] = ()
    providers: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()


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
    # NAM_RENDER_SEQUENTIAL_EXE (hybrid/render.py's find_sequential_nam_render_exe)
    # is deliberately NOT exposed here. Since the switch to a single render
    # engine, it already falls back to the same NAM_RENDER_EXE binary by
    # default -- it exists only as a developer/CI override for the
    # experimental NAMCore compatibility gate (tests/test_namcore_sequential_gate.py,
    # native/nam_render/README.md's "Experimental Sequential compatibility
    # gate"), comparing two separately-built renderers against a pinned
    # commit. No end user needs to set this.
    SettingField(
        name="PORT",
        label="Server port",
        description="Local port the app listens on.",
        group="Server",
        kind="number",
        placeholder="5001",
        restart_required=True,
    ),
    SettingField(
        name="NAM_MIXER_AI_PROVIDER",
        label="Provider",
        description="Choose Local, Cloudflare Workers AI, or another OpenAI-compatible HTTPS provider.",
        group="AI Assistant",
        kind="select",
        options=(("local", "Local"), ("cloudflare", "Cloudflare Workers AI"), ("custom", "Custom OpenAI-compatible")),
    ),
    SettingField(
        name="NAM_MIXER_AI_BASE_URL",
        label="Base URL",
        description="Local URLs may use HTTP only on localhost. Custom remote endpoints must use HTTPS and public addresses.",
        group="AI Assistant",
        placeholder="http://127.0.0.1:11434/v1",
        providers=("local", "custom"),
    ),
    SettingField(
        name="NAM_MIXER_AI_ACCOUNT_ID",
        label="Cloudflare Account ID",
        description="Copy this from Cloudflare's Workers AI → Use REST API page. It must be 32 hexadecimal characters; NAM Mixer constructs the fixed Workers AI URL for you.",
        group="AI Assistant", providers=("cloudflare",),
    ),
    SettingField(
        name="NAM_MIXER_AI_MODEL",
        label="Model",
        description="Model name exposed by your selected provider. For Cloudflare JSON recipes, start with Llama 3.3 70B for general quality or DeepSeek R1 Distill Qwen 32B for stronger reasoning; all suggestions support Workers AI JSON Mode. Leave blank to disable the AI Assistant.",
        group="AI Assistant", placeholder="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        suggestions=(
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "@hf/nousresearch/hermes-2-pro-mistral-7b",
            "@cf/meta/llama-3-8b-instruct",
            "@cf/meta/llama-3.1-8b-instruct",
            "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
        ),
    ),
    SettingField(
        name="NAM_MIXER_AI_API_KEY",
        label="API token",
        description="Create it from Workers AI → Use REST API (or grant a manual token Account > Workers AI > Read). Stored server-side only; tests and requests can consume quota or incur billing.",
        group="AI Assistant", kind="secret", placeholder="Bearer token", providers=("cloudflare", "custom"),
    ),
    SettingField(
        name="NAM_MIXER_AI_TEMPERATURE",
        label="Temperature",
        description="Sampling temperature (0.0-2.0).",
        group="AI Assistant",
        kind="number",
        placeholder="0.2",
    ),
    SettingField(
        name="NAM_MIXER_AI_TIMEOUT_SECONDS",
        label="AI request timeout (seconds)",
        description="How long to wait for an AI provider before giving up.",
        group="AI Assistant",
        kind="number",
        placeholder="60",
    ),
    SettingField(
        name="TONE3000_API_KEY",
        label="TONE3000 API key",
        description="Server-side TONE3000 Secret Key (t3k_cs_...) used for the TONE3000 tab's "
                     "capture search. Leave blank to disable that tab. Never logged or returned "
                     "by the API once saved.",
        group="TONE3000",
        kind="secret",
        placeholder="t3k_cs_...",
    ),
    SettingField(
        name="NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES",
        label="Enable experimental NAM architectures",
        description="Shows Sequential Embedded (a valid NAM Sequential model: trained amp "
                     "followed by a separate Linear/FIR cabinet stage) as a cabinet export "
                     "choice. This is a real, valid NAM structure, but A2-only players may "
                     "reject it -- see README.md. Off by default; Baked In remains the "
                     "standard, broadly compatible way to include a cabinet.",
        group="Advanced",
        kind="checkbox",
    ),
)

_KNOWN_NAMES = {field.name for field in SETTINGS}
_FIELDS_BY_NAME = {field.name: field for field in SETTINGS}


def get_settings() -> list[dict]:
    """Return every registered setting's current value plus its UI metadata.

    A `kind="secret"` field's real value is never returned -- only whether
    one is currently set (`has_value`) -- so an already-saved API key never
    round-trips back out over the API or onto a screen someone might share.

    Uses read_saved_env_values() instead of read_env_values() so that Settings
    page shows only what the user explicitly saved to .env, not stale values
    that might be inherited from the shell environment (e.g., a stale
    TONE3000_API_KEY exported before a correct one was saved to .env).
    """
    values = read_saved_env_values(_KNOWN_NAMES)
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
            "options": [{"value": value, "label": label} for value, label in field.options],
            "providers": list(field.providers),
            "suggestions": list(field.suggestions),
        }
        if field.kind == "secret":
            entry["value"] = ""
            entry["has_value"] = bool(raw_value)
        elif field.kind == "checkbox":
            entry["value"] = raw_value.strip().lower() in ("1", "true", "yes", "on")
        else:
            entry["value"] = raw_value
        result.append(entry)
    return result


def save_settings(values: dict, clear_secrets: list[str] | None = None) -> dict:
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
        if field.kind == "checkbox":
            value = "true" if (value is True or str(value).strip().lower() in ("1", "true", "yes", "on")) else "false"
        else:
            value = str(value)
        if field.kind == "secret" and not value.strip():
            continue
        filtered[name] = value
    for name in clear_secrets or []:
        field = _FIELDS_BY_NAME.get(str(name))
        if field and field.kind == "secret":
            filtered[field.name] = ""
    env_file = write_env_values(filtered)
    return {"saved": sorted(filtered), "env_file": str(env_file)}


def experimental_architectures_enabled() -> bool:
    """Whether Sequential Embedded (and any future experimental NAM
    architecture) may be selected. Off by default -- see the
    NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES SettingField above. Callers
    that produce or serve a Sequential-embedded artifact MUST check this
    server-side; the UI hiding the option is not itself the gate."""
    raw = read_env_values({"NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES"}).get(
        "NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", ""
    )
    return raw.strip().lower() in ("1", "true", "yes", "on")

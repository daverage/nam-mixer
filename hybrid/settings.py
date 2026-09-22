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

from hybrid.env_file import read_saved_env_values, write_env_values
from hybrid.local_llm import RECOMMENDED_LOCAL_MODEL


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
    required_prefix: str = ""
    validation_message: str = ""


class SettingsValidationError(ValueError):
    """A submitted setting is syntactically invalid and must not be saved."""


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
        # Overridden per-provider by _MODEL_FIELD_TEXT in get_settings() below --
        # this is only the fallback before a provider has ever been selected.
        description="Model name exposed by your selected provider. If replies with research notes attached fail "
                     "to come back, raise the AI response token limit under Advanced. Leave blank to disable the "
                     "AI Assistant.",
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
        name="NAM_MIXER_AI_MAX_TOKENS",
        label="AI response token limit",
        description="Max tokens the AI provider may generate per reply. Raise this if replies with "
                     "research notes attached fail with \"the local model could not incorporate it\" -- "
                     "that usually means the response was cut off before valid JSON completed. "
                     "Leave blank to use the automatic default.",
        group="Advanced",
        kind="number",
        placeholder="3000",
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
        required_prefix="t3k_cs_",
        validation_message="Use the TONE3000 Secret Key beginning t3k_cs_. The t3k_pub_ key is for OAuth and cannot be used for direct API access.",
    ),
)

_KNOWN_NAMES = {field.name for field in SETTINGS}
_FIELDS_BY_NAME = {field.name: field for field in SETTINGS}

# Connection details belong to one provider, even though the browser uses the
# same small set of field names for whichever provider is selected.  Keeping
# the storage keys separate prevents (for example) an Ollama model name and a
# Cloudflare token from being carried into a custom endpoint when switching.
_PROVIDER_STORAGE: dict[str, dict[str, str]] = {
    "local": {
        "NAM_MIXER_AI_BASE_URL": "NAM_MIXER_AI_LOCAL_BASE_URL",
        "NAM_MIXER_AI_MODEL": "NAM_MIXER_AI_LOCAL_MODEL",
    },
    "cloudflare": {
        "NAM_MIXER_AI_ACCOUNT_ID": "NAM_MIXER_AI_CLOUDFLARE_ACCOUNT_ID",
        "NAM_MIXER_AI_MODEL": "NAM_MIXER_AI_CLOUDFLARE_MODEL",
        "NAM_MIXER_AI_API_KEY": "NAM_MIXER_AI_CLOUDFLARE_API_KEY",
    },
    "custom": {
        "NAM_MIXER_AI_BASE_URL": "NAM_MIXER_AI_CUSTOM_BASE_URL",
        "NAM_MIXER_AI_MODEL": "NAM_MIXER_AI_CUSTOM_MODEL",
        "NAM_MIXER_AI_API_KEY": "NAM_MIXER_AI_CUSTOM_API_KEY",
    },
}
_PROVIDER_STORAGE_NAMES = {
    storage_name
    for provider_fields in _PROVIDER_STORAGE.values()
    for storage_name in provider_fields.values()
}
_ALL_STORAGE_NAMES = _KNOWN_NAMES | _PROVIDER_STORAGE_NAMES

_TOKEN_LIMIT_HINT = (
    "If replies with research notes attached fail to come back, raise the AI response token limit under Advanced."
)

# NAM_MIXER_AI_MODEL's description/suggestions depend on which provider is
# currently selected, so this is applied in get_settings() rather than baked
# into the static SettingField above (the field list itself is provider-agnostic).
_MODEL_FIELD_TEXT: dict[str, dict[str, object]] = {
    "local": {
        "description": f"Model name exposed by your local OpenAI-compatible host (e.g. Ollama). We recommend "
                        f"{RECOMMENDED_LOCAL_MODEL} -- see the \"Pull {RECOMMENDED_LOCAL_MODEL}\" button below. "
                        f"Installed models are loaded from the host when it supports model discovery. "
                        f"{_TOKEN_LIMIT_HINT} Leave blank to disable the AI Assistant.",
        "suggestions": (RECOMMENDED_LOCAL_MODEL, "qwen3:8b", "llama3.1:8b", "gemma3:4b"),
        "placeholder": RECOMMENDED_LOCAL_MODEL,
    },
    "cloudflare": {
        "description": f"Model name exposed by Cloudflare Workers AI. For Cloudflare JSON recipes, start with "
                        f"Llama 3.3 70B for general quality or DeepSeek R1 Distill Qwen 32B for stronger "
                        f"reasoning; all suggestions support Workers AI JSON Mode. {_TOKEN_LIMIT_HINT} Leave "
                        f"blank to disable the AI Assistant.",
        "suggestions": (
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "@hf/nousresearch/hermes-2-pro-mistral-7b",
            "@cf/meta/llama-3-8b-instruct",
            "@cf/meta/llama-3.1-8b-instruct",
            "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
        ),
        "placeholder": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    },
    "custom": {
        "description": f"Model name exposed by your custom OpenAI-compatible endpoint. If the endpoint supports "
                        f"GET /models, its models are loaded into the searchable list. {_TOKEN_LIMIT_HINT} Leave "
                        f"blank to disable the AI Assistant.",
        "suggestions": (),
        "placeholder": "",
    },
}


def _saved_provider_value(values: dict[str, str], provider: str, name: str) -> str:
    """Return a provider slot, falling back to the pre-slot shared key.

    The fallback makes existing .env files upgrade in place.  The next save
    copies the legacy value into the active provider's dedicated slot.
    """
    storage_name = _PROVIDER_STORAGE.get(provider, {}).get(name)
    if storage_name and values.get(storage_name):
        return values[storage_name]
    # Shared keys are a legacy layout. Once any scoped slot exists, an empty
    # slot means this provider genuinely has no saved value.
    if any(values.get(candidate) for candidate in _PROVIDER_STORAGE_NAMES):
        return ""
    return values.get(name, "")


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
    values = read_saved_env_values(_ALL_STORAGE_NAMES)
    provider = values.get("NAM_MIXER_AI_PROVIDER", "").strip() or "local"
    result = []
    for field in SETTINGS:
        raw_value = _saved_provider_value(values, provider, field.name)
        description, suggestions, placeholder = field.description, field.suggestions, field.placeholder
        if field.name == "NAM_MIXER_AI_MODEL":
            model_text = _MODEL_FIELD_TEXT.get(provider, _MODEL_FIELD_TEXT["local"])
            description, suggestions, placeholder = model_text["description"], model_text["suggestions"], model_text["placeholder"]
        entry = {
            "name": field.name,
            "label": field.label,
            "description": description,
            "group": field.group,
            "kind": field.kind,
            "placeholder": placeholder,
            "restart_required": field.restart_required,
            "options": [{"value": value, "label": label} for value, label in field.options],
            "providers": list(field.providers),
            "suggestions": list(suggestions),
            "required_prefix": field.required_prefix,
            "validation_message": field.validation_message,
        }
        if field.kind == "secret":
            entry["value"] = ""
            entry["has_value"] = bool(raw_value)
            entry["is_valid"] = not raw_value or _validation_error(field, raw_value) is None
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
    "clear it" -- clearing is an explicit `clear_secrets` operation.
    """
    saved_values = read_saved_env_values(_ALL_STORAGE_NAMES)
    previous_provider = saved_values.get("NAM_MIXER_AI_PROVIDER", "").strip() or "local"
    requested_provider = str(values.get("NAM_MIXER_AI_PROVIDER", previous_provider)).strip().lower()
    if requested_provider not in _PROVIDER_STORAGE:
        raise SettingsValidationError("AI provider must be Local, Cloudflare Workers AI, or Custom OpenAI-compatible")

    filtered: dict[str, str] = {}

    # On the first save after upgrading, preserve the old shared connection
    # values in the provider that owned them before changing provider.
    for generic_name, storage_name in _PROVIDER_STORAGE[previous_provider].items():
        legacy_value = saved_values.get(generic_name, "")
        if legacy_value and not saved_values.get(storage_name):
            filtered[storage_name] = legacy_value

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
        validation_error = _validation_error(field, value)
        if validation_error:
            raise SettingsValidationError(validation_error)
        storage_name = _PROVIDER_STORAGE.get(requested_provider, {}).get(name, name)
        # Provider-specific fields that do not apply to the selected provider
        # can still be present in a browser form while hidden. Ignore them.
        if name.startswith("NAM_MIXER_AI_") and name in {
            "NAM_MIXER_AI_BASE_URL", "NAM_MIXER_AI_ACCOUNT_ID",
            "NAM_MIXER_AI_MODEL", "NAM_MIXER_AI_API_KEY",
        } and name not in _PROVIDER_STORAGE[requested_provider]:
            continue
        filtered[storage_name] = value
    for name in clear_secrets or []:
        field = _FIELDS_BY_NAME.get(str(name))
        if field and field.kind == "secret":
            filtered[_PROVIDER_STORAGE.get(requested_provider, {}).get(field.name, field.name)] = ""
    env_file = write_env_values(filtered)
    return {"saved": sorted(filtered), "env_file": str(env_file)}


def _validation_error(field: SettingField, value: str) -> str | None:
    value = str(value).strip()
    if not value or not field.required_prefix:
        return None
    if value.startswith(field.required_prefix) and len(value) > len(field.required_prefix):
        return None
    return field.validation_message or f"{field.label} must start with {field.required_prefix}."

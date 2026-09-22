"""Tests for hybrid/settings.py and the .env read/write helpers it relies on
(hybrid/env_file.py) -- these back the browser Settings page, which lets
someone configure the app without a shell to `export` env vars into.
"""
import os

import pytest

from hybrid import env_file, settings


@pytest.fixture
def isolated_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(env_path))
    for field in settings.SETTINGS:
        monkeypatch.delenv(field.name, raising=False)
    for name in settings._PROVIDER_STORAGE_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield env_path
    # write_env_values() sets os.environ directly (by design -- so a saved
    # setting takes effect immediately, see hybrid/env_file.py), which
    # monkeypatch doesn't track since it wasn't set via monkeypatch.setenv;
    # clean up explicitly so a value saved in one test can't leak into others.
    for field in settings.SETTINGS:
        os.environ.pop(field.name, None)
    for name in settings._PROVIDER_STORAGE_NAMES:
        os.environ.pop(name, None)


def test_write_then_read_round_trip(isolated_env_file):
    env_file.write_env_values({"NAM_RENDER_EXE": "/opt/nam_render"})
    assert env_file.read_env_value("NAM_RENDER_EXE") == "/opt/nam_render"
    assert isolated_env_file.read_text().strip() == "NAM_RENDER_EXE=/opt/nam_render"


def test_write_updates_existing_key_in_place(isolated_env_file):
    env_file.write_env_values({"NAM_RENDER_EXE": "/opt/one", "PORT": "5001"})
    env_file.write_env_values({"NAM_RENDER_EXE": "/opt/two"})
    lines = isolated_env_file.read_text().splitlines()
    assert "NAM_RENDER_EXE=/opt/two" in lines
    assert "PORT=5001" in lines
    assert len(lines) == 2


def test_write_empty_value_clears_key(isolated_env_file):
    env_file.write_env_values({"NAM_RENDER_EXE": "/opt/one"})
    env_file.write_env_values({"NAM_RENDER_EXE": ""})
    assert env_file.read_env_value("NAM_RENDER_EXE") == ""
    assert "NAM_RENDER_EXE" not in isolated_env_file.read_text()


def test_write_sets_os_environ_immediately(isolated_env_file):
    env_file.write_env_values({"PORT": "6001"})
    assert os.environ["PORT"] == "6001"


def test_shell_env_var_takes_priority_over_env_file(isolated_env_file):
    env_file.write_env_values({"NAM_RENDER_EXE": "/from/dotenv"})
    # write_env_values() also sets os.environ (by design); overwrite it
    # directly here to simulate a real shell-exported value taking priority,
    # since monkeypatch.setenv's teardown would otherwise restore whatever
    # write_env_values just set rather than the pre-test empty state (the
    # isolated_env_file fixture's own teardown pops it afterwards regardless).
    os.environ["NAM_RENDER_EXE"] = "/from/shell"
    assert env_file.read_env_value("NAM_RENDER_EXE") == "/from/shell"


def test_read_saved_values_ignores_shell_environment(isolated_env_file, monkeypatch):
    env_file.write_env_values({"TONE3000_API_KEY": "t3k_cs_saved"})
    monkeypatch.setenv("TONE3000_API_KEY", "stale_shell_value")

    assert env_file.read_saved_env_values({"TONE3000_API_KEY", "NOT_ALLOWED"}) == {
        "TONE3000_API_KEY": "t3k_cs_saved",
    }


def test_get_settings_lists_all_registered_fields(isolated_env_file):
    result = settings.get_settings()
    names = {field["name"] for field in result}
    assert names == {field.name for field in settings.SETTINGS}
    assert all(
        field["value"] == ("" if field["kind"] != "checkbox" else False)
        for field in result
    )


def test_save_settings_ignores_unknown_names(isolated_env_file):
    result = settings.save_settings({"NAM_RENDER_EXE": "/opt/nam_render", "NOT_A_REAL_SETTING": "x"})
    assert result["saved"] == ["NAM_RENDER_EXE"]
    assert env_file.read_env_value("NOT_A_REAL_SETTING") == ""
    assert env_file.read_env_value("NAM_RENDER_EXE") == "/opt/nam_render"


def test_ai_provider_selection_round_trips(isolated_env_file):
    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "cloudflare"})
    values = {field["name"]: field for field in settings.get_settings()}
    assert values["NAM_MIXER_AI_PROVIDER"]["value"] == "cloudflare"
    assert isolated_env_file.read_text().strip() == "NAM_MIXER_AI_PROVIDER=cloudflare"


def test_secret_field_value_never_returned_by_get_settings(isolated_env_file):
    settings.save_settings({"TONE3000_API_KEY": "t3k_cs_realsecret"})
    result = {field["name"]: field for field in settings.get_settings()}
    assert result["TONE3000_API_KEY"]["value"] == ""
    assert result["TONE3000_API_KEY"]["has_value"] is True
    assert env_file.read_env_value("TONE3000_API_KEY") == "t3k_cs_realsecret"


def test_blank_secret_on_save_means_unchanged_not_cleared(isolated_env_file):
    settings.save_settings({"TONE3000_API_KEY": "t3k_cs_realsecret"})
    result = settings.save_settings({"TONE3000_API_KEY": ""})
    assert result["saved"] == []
    assert env_file.read_env_value("TONE3000_API_KEY") == "t3k_cs_realsecret"


def test_ai_api_token_is_secret_and_can_be_explicitly_cleared(isolated_env_file):
    settings.save_settings({
        "NAM_MIXER_AI_PROVIDER": "cloudflare",
        "NAM_MIXER_AI_API_KEY": "cloudflare-secret",
    })
    field = {field["name"]: field for field in settings.get_settings()}["NAM_MIXER_AI_API_KEY"]
    assert field["value"] == ""
    assert field["has_value"] is True
    settings.save_settings({}, clear_secrets=["NAM_MIXER_AI_API_KEY"])
    assert env_file.read_env_value("NAM_MIXER_AI_CLOUDFLARE_API_KEY") == ""
    assert "NAM_MIXER_AI_CLOUDFLARE_API_KEY" not in os.environ


def test_ai_connection_settings_are_kept_separately_per_provider(isolated_env_file):
    settings.save_settings({
        "NAM_MIXER_AI_PROVIDER": "cloudflare",
        "NAM_MIXER_AI_ACCOUNT_ID": "a" * 32,
        "NAM_MIXER_AI_MODEL": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "NAM_MIXER_AI_API_KEY": "cloudflare-secret",
    })
    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "local"})
    settings.save_settings({
        "NAM_MIXER_AI_MODEL": "gemma4:e4b",
        "NAM_MIXER_AI_BASE_URL": "http://127.0.0.1:11434/v1",
    })
    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "custom"})
    settings.save_settings({
        "NAM_MIXER_AI_MODEL": "custom-chat",
        "NAM_MIXER_AI_BASE_URL": "https://models.example.com/v1",
        "NAM_MIXER_AI_API_KEY": "custom-secret",
    })

    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "cloudflare"})
    cloudflare = {field["name"]: field for field in settings.get_settings()}
    assert cloudflare["NAM_MIXER_AI_MODEL"]["value"] == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert cloudflare["NAM_MIXER_AI_ACCOUNT_ID"]["value"] == "a" * 32
    assert cloudflare["NAM_MIXER_AI_API_KEY"]["has_value"] is True

    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "local"})
    local = {field["name"]: field for field in settings.get_settings()}
    assert local["NAM_MIXER_AI_MODEL"]["value"] == "gemma4:e4b"
    assert local["NAM_MIXER_AI_BASE_URL"]["value"] == "http://127.0.0.1:11434/v1"
    assert local["NAM_MIXER_AI_API_KEY"]["has_value"] is False

    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "custom"})
    custom = {field["name"]: field for field in settings.get_settings()}
    assert custom["NAM_MIXER_AI_MODEL"]["value"] == "custom-chat"
    assert custom["NAM_MIXER_AI_BASE_URL"]["value"] == "https://models.example.com/v1"
    assert custom["NAM_MIXER_AI_API_KEY"]["has_value"] is True


def test_switching_provider_migrates_legacy_shared_values_to_their_owner(isolated_env_file):
    env_file.write_env_values({
        "NAM_MIXER_AI_PROVIDER": "local",
        "NAM_MIXER_AI_MODEL": "legacy-local-model",
        "NAM_MIXER_AI_BASE_URL": "http://localhost:9999/v1",
    })

    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "cloudflare"})

    assert env_file.read_env_value("NAM_MIXER_AI_LOCAL_MODEL") == "legacy-local-model"
    assert env_file.read_env_value("NAM_MIXER_AI_LOCAL_BASE_URL") == "http://localhost:9999/v1"
    current = {field["name"]: field for field in settings.get_settings()}
    assert current["NAM_MIXER_AI_MODEL"]["value"] == ""


def test_provider_and_unrelated_fields_survive_once_any_provider_slot_exists(isolated_env_file):
    """Regression test: once ANY provider gains a scoped connection slot (i.e.
    after the very first provider-aware save), fields that are NOT
    provider-scoped -- NAM_MIXER_AI_PROVIDER itself, PORT, etc. -- must keep
    reporting their real saved value, not silently go blank. A blank
    NAM_MIXER_AI_PROVIDER makes the Settings page's provider selector appear
    to forget the user's choice on every reload."""
    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "local", "NAM_MIXER_AI_MODEL": "gemma4:e4b", "PORT": "5001"})
    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "cloudflare"})

    current = {field["name"]: field for field in settings.get_settings()}
    assert current["NAM_MIXER_AI_PROVIDER"]["value"] == "cloudflare"
    assert current["PORT"]["value"] == "5001"

    settings.save_settings({"NAM_MIXER_AI_PROVIDER": "local"})
    current = {field["name"]: field for field in settings.get_settings()}
    assert current["NAM_MIXER_AI_PROVIDER"]["value"] == "local"
    assert current["PORT"]["value"] == "5001"
    assert current["NAM_MIXER_AI_MODEL"]["value"] == "gemma4:e4b"


def test_secret_field_not_set_reports_no_value(isolated_env_file):
    result = {field["name"]: field for field in settings.get_settings()}
    assert result["TONE3000_API_KEY"]["has_value"] is False
    assert result["TONE3000_API_KEY"]["value"] == ""


def test_tone3000_secret_key_rejects_publishable_key_without_saving(isolated_env_file):
    with pytest.raises(settings.SettingsValidationError, match="t3k_cs_"):
        settings.save_settings({"TONE3000_API_KEY": "t3k_pub_not-a-secret"})

    assert not isolated_env_file.exists()


def test_tone3000_saved_invalid_key_is_reported_without_exposing_it(isolated_env_file):
    env_file.write_env_values({"TONE3000_API_KEY": "legacy-invalid-value"})

    field = {field["name"]: field for field in settings.get_settings()}["TONE3000_API_KEY"]

    assert field["has_value"] is True
    assert field["is_valid"] is False
    assert field["value"] == ""
    assert "legacy-invalid-value" not in str(field)

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
    yield env_path
    # write_env_values() sets os.environ directly (by design -- so a saved
    # setting takes effect immediately, see hybrid/env_file.py), which
    # monkeypatch doesn't track since it wasn't set via monkeypatch.setenv;
    # clean up explicitly so a value saved in one test can't leak into others.
    for field in settings.SETTINGS:
        os.environ.pop(field.name, None)


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
    assert all(field["value"] == "" for field in result)


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
    settings.save_settings({"NAM_MIXER_AI_API_KEY": "cloudflare-secret"})
    field = {field["name"]: field for field in settings.get_settings()}["NAM_MIXER_AI_API_KEY"]
    assert field["value"] == ""
    assert field["has_value"] is True
    settings.save_settings({}, clear_secrets=["NAM_MIXER_AI_API_KEY"])
    assert env_file.read_env_value("NAM_MIXER_AI_API_KEY") == ""
    assert "NAM_MIXER_AI_API_KEY" not in os.environ


def test_secret_field_not_set_reports_no_value(isolated_env_file):
    result = {field["name"]: field for field in settings.get_settings()}
    assert result["TONE3000_API_KEY"]["has_value"] is False
    assert result["TONE3000_API_KEY"]["value"] == ""

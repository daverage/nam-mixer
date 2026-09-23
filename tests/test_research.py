import pytest

from hybrid.services.research import _rank_tone3000_metadata, _require_tone3000_api_key


def test_tone3000_credential_uses_saved_value_over_stale_shell(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TONE3000_API_KEY=t3k_cs_saved\n", encoding="utf-8")
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(env_path))
    monkeypatch.setenv("TONE3000_API_KEY", "stale-shell-value")

    assert _require_tone3000_api_key(for_action="search") == "t3k_cs_saved"


def test_tone3000_invalid_saved_credential_points_to_settings_not_shell(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TONE3000_API_KEY=invalid-saved-value\n", encoding="utf-8")
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(env_path))
    monkeypatch.setenv("TONE3000_API_KEY", "t3k_cs_valid_but_ignored_shell_value")

    with pytest.raises(RuntimeError) as error:
        _require_tone3000_api_key(for_action="search")

    assert "Replace or clear the saved key in Settings" in str(error.value)
    assert "stale" not in str(error.value).lower()


def test_tone3000_metadata_score_prioritizes_title_matches():
    query = "Marshall JCM800 high gain amp"
    title_match = {"title": "Marshall JCM800", "description": "High gain capture."}
    description_match = {"title": "British Head", "description": "Marshall JCM800 high gain capture."}

    assert _rank_tone3000_metadata(query, title_match)[0] == 75
    assert _rank_tone3000_metadata(query, description_match)[0] == 50


def test_tone3000_metadata_score_ignores_generic_query_words():
    result = {"title": "Clean Combo", "description": "A bright clean capture."}

    assert _rank_tone3000_metadata("I need a guitar amp with this tone", result)[0] == 0


def test_tone3000_search_ranks_the_complete_catalogue_page_before_limiting(monkeypatch):
    import hybrid.services.research as research

    monkeypatch.setattr(research, "_require_tone3000_api_key", lambda **_kwargs: "t3k_cs_test")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return __import__("json").dumps({"data": [
                {"id": number, "title": f"Unrelated {number}", "description": ""}
                for number in range(8)
            ] + [{"id": 99, "title": "Vox AC30 Top Boost", "description": "Clean Vox capture"}]}).encode()

    ranked = research.tone3000_search(
        "Vox AC30", rig_scope="anything", rank_query="clean Vox AC30", opener=lambda *_args, **_kwargs: Response()
    )

    assert ranked[0]["title"] == "Vox AC30 Top Boost"
    assert len(ranked) == 8

import pytest

import hybrid.services.research as research
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


# ---- SSRF guard for pages linked from third-party search results

def _resolving_to(monkeypatch, *addresses, error=None):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        if error:
            raise error
        return [(None, None, None, "", (a, 0)) for a in addresses]
    monkeypatch.setattr(research.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.mark.parametrize("addresses, safe", [
    (("93.184.216.34",), True),                        # public IPv4
    (("2606:2800:220:1:248:1893:25c8:1946",), True),   # public IPv6
    (("127.0.0.1",), False),                           # loopback
    (("::1",), False),
    (("10.0.0.5",), False),                            # private ranges
    (("172.16.3.4",), False),
    (("192.168.1.1",), False),
    (("169.254.169.254",), False),                     # link-local, incl. cloud metadata endpoints
    (("fe80::1",), False),
    (("0.0.0.0",), False),                             # unspecified
    (("224.0.0.1",), False),                           # multicast
    (("93.184.216.34", "10.0.0.5"), False),            # any private answer makes the host unsafe (DNS rebinding)
])
def test_is_safe_public_host_rejects_every_non_public_address(monkeypatch, addresses, safe):
    _resolving_to(monkeypatch, *addresses)
    assert research._is_safe_public_host("example.com") is safe


def test_is_safe_public_host_rejects_unresolvable_or_empty_hosts(monkeypatch):
    _resolving_to(monkeypatch, error=OSError("no such host"))
    assert research._is_safe_public_host("nowhere.invalid") is False
    assert research._is_safe_public_host("") is False
    _resolving_to(monkeypatch, "not-an-address")
    assert research._is_safe_public_host("example.com") is False


@pytest.mark.parametrize("href", [
    "http://127.0.0.1:5001/api/settings",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
    "ftp://example.com/x",
])
def test_page_evidence_never_opens_a_connection_to_an_unsafe_target(monkeypatch, href):
    host = research.urlparse(href).hostname or ""
    _resolving_to(monkeypatch, host if host[:1].isdigit() else "93.184.216.34")

    def must_not_open(*_args, **_kwargs):
        raise AssertionError(f"opened a connection for {href}")
    monkeypatch.setattr(research, "build_opener", must_not_open)
    assert research._page_evidence(href, "marshall amp") == ""


def test_redirects_from_search_result_pages_are_refused():
    assert research._NoRedirectHandler().redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/") is None

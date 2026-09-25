"""Tests for hybrid/services/update_check.py -- the Settings page's manual "Check for updates" button's backend, mocked at
the urlopen boundary (same style as tests/test_render_bootstrap.py) so these never hit the real network.
"""
import json
from io import BytesIO
from unittest.mock import patch

import pytest

from hybrid.services import update_check


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _releases(*tags_and_flags, asset_names_for_last=()):
    """Build a fake GitHub releases-list payload. Each entry is (tag, draft, prerelease)."""
    releases = []
    for tag, draft, prerelease in tags_and_flags:
        releases.append({
            "tag_name": tag, "draft": draft, "prerelease": prerelease,
            "html_url": f"https://github.com/daverage/nam-mixer/releases/tag/{tag}",
            "assets": [{"name": n, "browser_download_url": f"https://example.invalid/{tag}/{n}"} for n in asset_names_for_last],
        })
    return releases


def _mock_releases(payload):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))
    return patch.object(update_check, "urlopen", fake_urlopen)


def test_reports_an_available_update():
    payload = _releases(("v0.3.2", False, False), ("v0.3.1", False, False))
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.1", platform="linux")
    assert result.update_available is True
    assert result.latest_version == "v0.3.2" and result.current_version == "v0.3.1"
    assert result.release_url.endswith("/v0.3.2")


def test_reports_up_to_date_when_current_is_newest():
    payload = _releases(("v0.3.1", False, False))
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.2", platform="linux")
    assert result.update_available is False
    assert result.latest_version == "v0.3.1"


def test_up_to_date_is_exact_match_not_just_not_older():
    payload = _releases(("v0.3.2", False, False))
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.2", platform="linux")
    assert result.update_available is False


def test_draft_and_prerelease_releases_are_ignored():
    payload = _releases(("v9.9.9", True, False), ("v8.8.8", False, True), ("v0.3.2", False, False))
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.1", platform="linux")
    assert result.latest_version == "v0.3.2"


def test_non_version_tags_are_ignored_eg_the_native_renderers_own_release_series():
    payload = _releases(("nam-render-v1.2.3", False, False), ("v0.3.2", False, False))
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.1", platform="linux")
    assert result.latest_version == "v0.3.2"


def test_version_comparison_is_numeric_not_lexical():
    # v0.3.10 must be treated as newer than v0.3.9, unlike a plain string comparison.
    payload = _releases(("v0.3.10", False, False), ("v0.3.9", False, False))
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.9", platform="linux")
    assert result.update_available is True and result.latest_version == "v0.3.10"


def test_no_version_release_at_all_raises():
    payload = _releases(("nam-render-v1.0.0", False, False))
    with _mock_releases(payload):
        with pytest.raises(update_check.UpdateCheckError, match="No published app release"):
            update_check.check_for_update("v0.3.1", platform="linux")


def test_network_failure_raises_a_ui_safe_error():
    def fake_urlopen(request, timeout=None):
        raise OSError("no route to host")
    with patch.object(update_check, "urlopen", fake_urlopen):
        with pytest.raises(update_check.UpdateCheckError, match="Could not reach GitHub"):
            update_check.check_for_update("v0.3.1", platform="linux")


def test_unparseable_current_version_raises():
    with pytest.raises(update_check.UpdateCheckError, match="Could not parse"):
        update_check.check_for_update("not-a-version")


ALL_ASSETS = [
    "NAM-Mixer-macOS-arm64.dmg", "NAM-Mixer-Linux-x64.AppImage", "NAM-Mixer-Linux-x64.deb",
    "NAM-Mixer-Windows-x64-setup.exe", "NAM-Mixer-Windows-x64.msi",
]


@pytest.mark.parametrize("platform,machine,package,expected_asset", [
    ("darwin", "arm64", None, "NAM-Mixer-macOS-arm64.dmg"),
    ("win32", "x86_64", None, "NAM-Mixer-Windows-x64-setup.exe"),
    ("win32", "arm64", None, "NAM-Mixer-Windows-x64-setup.exe"),     # x64 runs under emulation
    ("linux", "x86_64", "appimage", "NAM-Mixer-Linux-x64.AppImage"),
    ("linux", "x86_64", "deb", "NAM-Mixer-Linux-x64.deb"),            # a .deb install gets the .deb
])
def test_asset_url_is_the_installer_for_this_machine(platform, machine, package, expected_asset):
    payload = _releases(("v0.3.2", False, False), asset_names_for_last=ALL_ASSETS)
    with _mock_releases(payload):
        result = update_check.check_for_update("v0.3.1", platform=platform, machine=machine, linux_package=package)
    assert result.asset_url == f"https://example.invalid/v0.3.2/{expected_asset}"


@pytest.mark.parametrize("platform,machine", [
    ("darwin", "x86_64"),   # no Intel-Mac build: the arm64 DMG would not run
    ("linux", "arm64"),     # no Linux-ARM build
    ("sunos5", "x86_64"),
])
def test_no_installer_is_offered_where_no_build_runs(platform, machine):
    payload = _releases(("v0.3.2", False, False), asset_names_for_last=ALL_ASSETS)
    with _mock_releases(payload):
        assert update_check.check_for_update("v0.3.1", platform=platform, machine=machine).asset_url is None


def test_asset_url_is_none_for_a_release_missing_it():
    payload = _releases(("v0.3.2", False, False), asset_names_for_last=["NAM-Mixer-macOS-arm64.dmg"])
    with _mock_releases(payload):
        assert update_check.check_for_update("v0.3.1", platform="win32", machine="x86_64").asset_url is None


def test_linux_package_detection(monkeypatch):
    monkeypatch.setenv("APPIMAGE", "/home/u/NAM-Mixer-Linux-x64.AppImage")
    assert update_check._linux_package() == "appimage"
    monkeypatch.delenv("APPIMAGE")
    monkeypatch.setattr(update_check.sys, "frozen", True, raising=False)
    monkeypatch.setattr(update_check.sys, "executable", "/usr/lib/NAM Mixer/nam-mixer-backend/nam-mixer-backend")
    assert update_check._linux_package() == "deb"
    monkeypatch.setattr(update_check.sys, "executable", "/home/u/src/.venv/bin/python")
    assert update_check._linux_package() == "appimage"


def test_requests_a_full_page_of_releases():
    """App releases share the repo with nam-render-v* releases; the default
    30-entry page can miss the newest app release."""
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        return _FakeResponse(json.dumps(_releases(("v0.3.5", False, False))).encode("utf-8"))

    with patch.object(update_check, "urlopen", fake_urlopen):
        update_check.check_for_update("v0.3.5")
    assert seen["url"].endswith("/releases?per_page=100")


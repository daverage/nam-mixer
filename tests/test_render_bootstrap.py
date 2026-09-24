"""Tests for hybrid/core/render_bootstrap.py -- the in-app "download nam_render"
button's backend (Settings page), mocked at the urlopen boundary so these
never hit the real network.
"""
import json
import sys
from io import BytesIO
from unittest.mock import patch

import pytest

from hybrid.core import render_bootstrap


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_unsupported_platform_raises_clear_error(monkeypatch):
    monkeypatch.setattr(sys, "platform", "sunos5", raising=False)
    with pytest.raises(render_bootstrap.NamRenderDownloadError, match="No prebuilt nam_render"):
        render_bootstrap.download_prebuilt_nam_render()


def test_download_writes_executable_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    monkeypatch.setattr(render_bootstrap, "_machine", lambda: "x86_64")
    releases_payload = json.dumps([{"tag_name": "nam-render-v1.2.3"}]).encode("utf-8")
    binary_payload = b"\x7fELF-fake-binary"

    responses = [_FakeResponse(releases_payload), _FakeResponse(binary_payload)]

    def fake_urlopen(request, timeout=None):
        return responses.pop(0)

    with patch.object(render_bootstrap, "urlopen", fake_urlopen):
        dest = tmp_path / "nam_render"
        result = render_bootstrap.download_prebuilt_nam_render(dest_path=dest)

    assert result == dest
    assert dest.read_bytes() == binary_payload
    assert dest.stat().st_mode & 0o111  # executable bit set
    assert not (tmp_path / "nam_render.download").exists()


def test_download_replaces_existing_binary_with_a_new_file(tmp_path, monkeypatch):
    """Swap in a new inode rather than rewriting the old binary in place
    (macOS kills a signed executable modified in place)."""
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    monkeypatch.setattr(render_bootstrap, "_machine", lambda: "x86_64")
    dest = tmp_path / "nam_render"
    dest.write_bytes(b"old")
    old_inode = dest.stat().st_ino
    responses = [_FakeResponse(json.dumps([{"tag_name": "nam-render-v1"}]).encode()), _FakeResponse(b"new")]
    with patch.object(render_bootstrap, "urlopen", lambda request, timeout=None: responses.pop(0)):
        render_bootstrap.download_prebuilt_nam_render(dest_path=dest)
    assert dest.read_bytes() == b"new"
    assert dest.stat().st_ino != old_inode


def test_intel_mac_has_no_prebuilt_binary(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(render_bootstrap, "_machine", lambda: "x86_64")
    with pytest.raises(render_bootstrap.NamRenderDownloadError, match="darwin, x86_64"):
        render_bootstrap.download_prebuilt_nam_render()


def test_no_matching_release_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(render_bootstrap, "_machine", lambda: "arm64")
    releases_payload = json.dumps([{"tag_name": "v1.0.0"}]).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(releases_payload)

    with patch.object(render_bootstrap, "urlopen", fake_urlopen):
        with pytest.raises(render_bootstrap.NamRenderDownloadError, match="No nam_render release"):
            render_bootstrap.download_prebuilt_nam_render()


def test_default_dest_path_matches_render_module_candidate():
    from hybrid.core.render import _NAM_RENDER_EXE_CANDIDATES

    dest = render_bootstrap.default_dest_path()
    assert dest in _NAM_RENDER_EXE_CANDIDATES

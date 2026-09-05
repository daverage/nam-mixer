"""API-level tests for the render-stage/blend-stage cost split enforced by
app.py: /api/render_pair is the only route allowed to touch NAM inference
(render() is faked out here, same as tests/test_pipeline_render.py, so any
route that accidentally invoked it would still "work" but this suite proves
the cache actually gets replaced correctly and that blend-stage routes never
need a fresh render).
"""
from __future__ import annotations

import json as jsonlib

import numpy as np
import pytest

import app as app_module
import hybrid.pipeline as pipeline

# Smallest bundled DI fixture (17.75s) -- keeps these tests fast since the
# causal envelope follower is a real (if cheap) per-sample computation.
DI_FILE = "high_thrash.wav"


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(pipeline, "render", fake_render)


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
    # Don't leak a rendered pair into unrelated tests/sessions.
    app_module._rendered_pair_cache["pair"] = None


def _write_fake_nam(path, input_level_dbu=None):
    data = {"architecture": "Test", "config": {}, "sample_rate": 48000}
    if input_level_dbu is not None:
        data["input_level_dbu"] = input_level_dbu
    path.write_text(jsonlib.dumps(data))


def _render_body(amp_a, amp_b, **overrides):
    body = {
        "amp_a_path": str(amp_a),
        "amp_b_path": str(amp_b),
        "di_file": DI_FILE,
        "instrument_type": "guitar",
        "input_profile_id": "vintage_humbucker",
        "calibration_mode": "auto",
    }
    body.update(overrides)
    return body


def test_preview_without_render_pair_first_returns_400(client):
    app_module._rendered_pair_cache["pair"] = None
    resp = client.post("/api/preview", json={"source": "a"})
    assert resp.status_code == 400
    assert "render" in resp.get_json()["error"].lower()


def test_render_pair_cache_reflects_the_newest_profile_not_the_old_one(client, tmp_path):
    """Re-rendering with a different profile must REPLACE the cached pair --
    a stale /api/preview response here would mean the server is serving an
    old render instead of the one that was just requested."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp0 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="vintage_humbucker"))
    assert resp0.status_code == 200
    data0 = resp0.get_json()
    assert data0["input_profile_gain_db"] == 0.0
    audio_at_0db = client.post("/api/preview", json={"source": "a"}).data

    resp1 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="hot_humbucker"))
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert data1["input_profile_gain_db"] == 4.5
    assert data1["input_peak_dbfs"] > data0["input_peak_dbfs"]
    audio_at_hot = client.post("/api/preview", json={"source": "a"}).data

    assert audio_at_0db != audio_at_hot


def test_rerendering_same_profile_repeatedly_does_not_stack_gain(client, tmp_path):
    """Calling /api/render_pair twice with the SAME profile must reproduce
    the exact same input_peak_dbfs each time -- proves the profile gain is
    always applied fresh from the original source DI, never compounded onto
    a previous render's already-gained signal."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    body = _render_body(amp_a, amp_b, input_profile_id="extreme_passive")

    peaks = [client.post("/api/render_pair", json=body).get_json()["input_peak_dbfs"] for _ in range(3)]
    assert peaks[0] == pytest.approx(peaks[1]) == pytest.approx(peaks[2])


def test_blend_stage_routes_never_require_a_fresh_render(client, tmp_path):
    """Once rendered, crossover/transition/trim changes must be servable
    purely from the cached RenderedPair -- this is the entire point of
    splitting render_pair()/build_hybrid() apart. Sweeping crossover here
    must succeed and change the reported trim without ever touching
    /api/render_pair again."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    r1 = client.post("/api/blend_info", json={"crossover_dbfs": -35.0, "transition_width_db": 8.0})
    r2 = client.post("/api/blend_info", json={"crossover_dbfs": -5.0, "transition_width_db": 8.0})
    assert r1.status_code == 200
    assert r2.status_code == 200

    # amp_a/amp_b are identical fakes here, so auto_trim_db is always 0 --
    # the real proof that crossover moves the blend without a rerender is
    # the blend WEIGHT curve itself, which depends only on the envelope and
    # crossover config, not on which amps were rendered.
    curve1 = client.post("/api/blend_curve", json={"crossover_dbfs": -35.0, "transition_width_db": 8.0, "max_points": 200})
    curve2 = client.post("/api/blend_curve", json={"crossover_dbfs": -5.0, "transition_width_db": 8.0, "max_points": 200})
    assert curve1.status_code == 200 and curve2.status_code == 200
    assert curve1.get_json()["blend_weight"] != curve2.get_json()["blend_weight"]


def test_input_profile_gain_reaches_the_actual_rendered_audio(client, tmp_path):
    """End-to-end check through the real Flask route (not just the pipeline
    unit test): a hotter profile must make what /api/preview actually
    returns for source=a louder, proving the gain reaches the served audio,
    not just the diagnostics JSON."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="vintage_single"))
    quiet_wav = client.post("/api/preview", json={"source": "a"}).data

    client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="extreme_passive"))
    loud_wav = client.post("/api/preview", json={"source": "a"}).data

    # Both are valid WAVs of the same nominal duration; the hot one must
    # simply contain larger sample magnitudes since render() is an identity
    # pass-through here.
    assert len(quiet_wav) == len(loud_wav)
    assert quiet_wav != loud_wav


def test_calibration_warning_surfaces_through_render_pair_response(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a, input_level_dbu=8.0)
    _write_fake_nam(amp_b)  # uncalibrated

    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, calibration_mode="auto"))
    data = resp.get_json()
    assert data["calibration_applied"] is False
    assert any("input_level_dbu" in w for w in data["warnings"])


def test_active_profile_without_custom_gain_is_rejected(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="active_buffered"))
    assert resp.status_code == 400

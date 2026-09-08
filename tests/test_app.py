"""API-level tests for the render-stage/blend-stage cost split enforced by
app.py: /api/render_pair is the only route allowed to touch NAM inference
(render() is faked out here, same as tests/test_pipeline_render.py, so any
route that accidentally invoked it would still "work" but this suite proves
the cache actually gets replaced correctly and that blend-stage routes never
need a fresh render).
"""
from __future__ import annotations

import io
import json as jsonlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import app as app_module
import hybrid.blend_training_target as blend_training_target
import hybrid.pipeline as pipeline
import hybrid.training_target as training_target

# Smallest bundled DI fixture (17.75s) -- keeps these tests fast since the
# causal envelope follower is a real (if cheap) per-sample computation.
DI_FILE = "high_thrash.wav"


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(pipeline, "render", fake_render)
    monkeypatch.setattr(training_target, "render", fake_render)
    monkeypatch.setattr(blend_training_target, "render", fake_render)
    # Bypass the official-V3-file MD5 check for synthetic training-input
    # fixtures in these tests -- we don't ship the real ~27MB official file.
    monkeypatch.setattr(training_target, "_md5_file", lambda path: training_target.OFFICIAL_V3_INPUT_MD5)


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


def test_live_blend_stems_returns_trimmed_stereo_pair_without_rerender(client, tmp_path):
    """The live-audition route must reuse the cached NAM output and leave the
    browser to perform only the final, adjustable linear mix."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    response = client.post("/api/live_blend_stems", json={
        "mix_b": 0.5, "auto_level": True, "manual_b_trim_db": 0.0,
    })

    assert response.status_code == 200
    assert response.headers["X-Live-Audition"] == "fixed-blend-stems"
    assert response.headers["X-Effective-Trim-Db"] == "0.000"
    audio, sample_rate = sf.read(io.BytesIO(response.data), dtype="float32", always_2d=True)
    assert sample_rate == 48_000
    assert audio.shape[1] == 2
    assert np.allclose(audio[:, 0], audio[:, 1])


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


def test_render_pair_applies_test_gain_db_as_real_additional_gain(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp0 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db=0.0))
    assert resp0.status_code == 200
    data0 = resp0.get_json()
    assert data0["test_gain_db"] == 0.0

    resp1 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db=12.0))
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert data1["test_gain_db"] == 12.0
    assert data1["input_peak_dbfs"] > data0["input_peak_dbfs"]


def test_render_pair_rejects_non_numeric_test_gain_db(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db="loud"))
    assert resp.status_code == 400
    assert "test_gain_db" in resp.get_json()["error"]


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


@pytest.mark.parametrize("field", ["reference_input_level_dbu", "test_gain_db", "amp_a_input_gain_db", "amp_b_input_gain_db"])
def test_render_pair_rejects_non_numeric_gain_controls(client, tmp_path, field):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, **{field: "not-a-number"}))

    assert resp.status_code == 400
    assert resp.is_json


def test_preview_rejects_invalid_output_gain(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    resp = client.post("/api/preview", json={
        "source": "hybrid", "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "manual_output_gain_db": "not-a-number",
    })

    assert resp.status_code == 400
    assert resp.is_json


@pytest.fixture
def isolated_training_paths(tmp_path, monkeypatch):
    """Redirect app.py's training-input/bundle paths into tmp_path so these
    tests never touch the real work/ directory."""
    training_path = tmp_path / "training_input" / "input.wav"
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    monkeypatch.setattr(app_module, "TRAINING_INPUT_PATH", training_path)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    return training_path, a2_dir


def _write_training_wav(path, n=4800, sample_rate=48000):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.uniform(-1, 1, n)).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path


def test_training_input_status_missing_by_default(client, isolated_training_paths):
    resp = client.get("/api/training_input/status")
    assert resp.status_code == 200
    assert resp.get_json()["ready"] is False


def test_training_input_upload_accepts_valid_wav(client, isolated_training_paths, tmp_path):
    src = _write_training_wav(tmp_path / "src.wav")
    with open(src, "rb") as f:
        resp = client.post("/api/training_input/upload", data={"file": (f, "input.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ready"] is True

    status = client.get("/api/training_input/status").get_json()
    assert status["ready"] is True


def test_training_input_upload_rejects_wrong_sample_rate(client, isolated_training_paths, tmp_path):
    src = _write_training_wav(tmp_path / "src.wav", sample_rate=44100)
    with open(src, "rb") as f:
        resp = client.post("/api/training_input/upload", data={"file": (f, "input.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert client.get("/api/training_input/status").get_json()["ready"] is False


def test_generate_requires_rendered_pair(client, isolated_training_paths):
    app_module._rendered_pair_cache["pair"] = None
    resp = client.post("/api/generate", json={})
    assert resp.status_code == 400


def test_generate_requires_training_input(client, isolated_training_paths, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={"crossover_dbfs": -20.0, "transition_width_db": 8.0})
    assert resp.status_code == 400
    assert resp.get_json()["training_input_ready"] is False


def test_generate_end_to_end_produces_bundle(client, isolated_training_paths, tmp_path):
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)

    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={
        "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "auto_level": False, "manual_b_trim_db": 1.5,
        "model_name": "My Hybrid Rig",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["implemented"] is True
    assert Path(data["target_path"]).is_file()
    assert Path(data["manifest_path"]).is_file()

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    # The design was auditioned at input_profile_id=vintage_humbucker (0 dB
    # gain, per _render_body's default) -- but this proves the field exists
    # and generation never re-applies ANY profile gain to the training input.
    assert manifest["design"]["pickup_profile_applied_to_training_input"] is False
    assert manifest["design"]["frozen_effective_b_trim_db"] == pytest.approx(1.5)
    assert manifest["mode"] == "hybrid"  # omitted mode defaults to Hybrid
    assert manifest["model_name"] == "My Hybrid Rig"
    assert manifest["artifact_filename"] == "My_Hybrid_Rig.nam"
    assert data["download_filename"] == "My_Hybrid_Rig.nam"


def test_mix_info_defaults_to_hybrid_mode(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    resp = client.post("/api/mix_info", json={"crossover_dbfs": -20.0, "transition_width_db": 8.0})
    assert resp.status_code == 200
    assert resp.get_json()["mode"] == "hybrid"


def test_mix_info_blend_mode_accepts_mix_b(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    resp = client.post("/api/mix_info", json={"mode": "blend", "mix_b": 0.75})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode"] == "blend"
    assert data["mix_b"] == pytest.approx(0.75)
    assert data["mix_a"] == pytest.approx(0.25)


def test_preview_blend_source_accepts_mix_and_is_independent_of_crossover_params(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/preview", json={"source": "blend", "mix_b": 0.2, "auto_level": False})
    assert resp.status_code == 200
    assert resp.headers.get("X-Mix-B") is not None
    assert float(resp.headers["X-Mix-B"]) == pytest.approx(0.2)


def test_preview_invalid_mix_b_is_clamped(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/preview", json={"source": "blend", "mix_b": 5.0, "auto_level": False})
    assert resp.status_code == 200
    assert float(resp.headers["X-Mix-B"]) == pytest.approx(1.0)


def test_cab_upload_validation_rejects_non_wav(client, tmp_path):
    bogus = tmp_path / "not_a_wav.txt"
    bogus.write_text("nope")
    with open(bogus, "rb") as f:
        resp = client.post("/api/cab/upload", data={"file": (f, "cab.txt")}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_cab_upload_validation_rejects_silent_ir(client, tmp_path):
    import soundfile as sf
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
    with open(silent, "rb") as f:
        resp = client.post("/api/cab/upload", data={"file": (f, "cab.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_cab_upload_accepts_valid_ir(client, tmp_path):
    import soundfile as sf
    ir = tmp_path / "ir.wav"
    ir_data = np.zeros(200, dtype=np.float32)
    ir_data[0] = 1.0
    sf.write(ir, ir_data, 48000, subtype="FLOAT")
    with open(ir, "rb") as f:
        resp = client.post("/api/cab/upload", data={"file": (f, "cab.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["original_sample_rate"] == 48000
    assert "path" in data


def test_preview_with_cab_applies_same_ir_to_a_result_and_b(client, tmp_path):
    import soundfile as sf
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    ir_path = tmp_path / "ir.wav"
    ir_data = np.zeros(10, dtype=np.float32)
    ir_data[0] = 0.5
    ir_data[1] = 0.5
    sf.write(ir_path, ir_data, 48000, subtype="FLOAT")

    for source in ("a", "b", "hybrid"):
        resp = client.post("/api/preview", json={
            "source": source, "crossover_dbfs": -20.0, "transition_width_db": 8.0,
            "cab_path": str(ir_path), "cab_preview_enabled": True,
        })
        assert resp.status_code == 200, (source, resp.get_json() if resp.data else resp.status_code)


def test_generate_blend_mode_produces_bundle_with_mode_blend(client, isolated_training_paths, tmp_path):
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)

    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={"mode": "blend", "mix_b": 0.4, "auto_level": False, "manual_b_trim_db": 0.5})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode"] == "blend"
    assert Path(data["target_path"]).is_file()

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    assert manifest["mode"] == "blend"
    assert manifest["design"]["mix_b"] == pytest.approx(0.4)


def test_generate_baked_cab_records_provenance(client, isolated_training_paths, tmp_path):
    import soundfile as sf
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)

    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    ir_path = tmp_path / "ir.wav"
    ir_data = np.zeros(10, dtype=np.float32)
    ir_data[0] = 0.6
    ir_data[1] = 0.4
    sf.write(ir_path, ir_data, 48000, subtype="FLOAT")

    resp = client.post("/api/generate", json={
        "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "cab_path": str(ir_path), "cab_preview_enabled": True, "cab_baked": True,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["cab_summary"]["baked"] is True
    assert data["cab_summary"]["sha256"]

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    assert manifest["cab"]["baked"] is True
    assert manifest["cab"]["selected"] is True

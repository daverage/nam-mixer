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
import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

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
    app_module._rendered_pair_cache["snapshot"] = None
    app_module._comparison_cache.clear()


def _write_fake_nam(path, input_level_dbu=None):
    data = {"architecture": "Test", "config": {}, "sample_rate": 48000}
    if input_level_dbu is not None:
        data["input_level_dbu"] = input_level_dbu
    path.write_text(jsonlib.dumps(data))


def _write_tool_nam(path):
    path.write_text(jsonlib.dumps({
        "architecture": "SlimmableContainer",
        "config": {"submodels": [{"model": {"config": {"head_scale": 0.0051461088670930214, "weights": [1]}, "metadata": {"loudness": -22.8}}}]},
        "metadata": {"loudness": -22.7, "gain": 3.0},
    }))


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


def _current_render_id():
    return app_module._rendered_pair_cache["snapshot"]["render_id"]


def test_preview_without_render_pair_first_returns_400(client):
    app_module._rendered_pair_cache["pair"] = None
    resp = client.post("/api/preview", json={"source": "a"})
    assert resp.status_code == 400
    assert "render" in resp.get_json()["error"].lower()


def test_renderer_readiness_reports_missing_binary_without_attempting_inference(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_nam_render_exe", lambda: (_ for _ in ()).throw(app_module.NamRenderError("not found")))
    response = client.get("/api/renderer/readiness")
    assert response.status_code == 200
    assert response.get_json() == {"found": False, "verified": False, "error": "not found"}


def test_renderer_readiness_accepts_namcore_usage_exit(client, monkeypatch):
    class Result:
        returncode, stdout, stderr = 1, "Usage: render <model.nam> <input.wav>", ""
    monkeypatch.setattr(app_module, "find_nam_render_exe", lambda: Path("/tmp/nam_render"))
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: Result())
    assert client.get("/api/renderer/readiness").get_json() == {
        "found": True, "verified": True, "path": "/tmp/nam_render",
    }


def test_renderer_readiness_distinguishes_found_but_unusable_binary(client, monkeypatch):
    class Result:
        returncode, stdout, stderr = 126, "", "permission denied"
    monkeypatch.setattr(app_module, "find_nam_render_exe", lambda: Path("/tmp/broken-renderer"))
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: Result())
    assert client.get("/api/renderer/readiness").get_json() == {
        "found": True, "verified": False, "path": "/tmp/broken-renderer", "error": "permission denied",
    }


def test_renderer_readiness_can_recover_on_retry_without_server_restart(client, monkeypatch):
    calls = 0
    def find_renderer():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise app_module.NamRenderError("not installed")
        return Path("/tmp/nam_render")
    class Result:
        returncode, stdout, stderr = 1, "Usage: render <model.nam> <input.wav>", ""
    monkeypatch.setattr(app_module, "find_nam_render_exe", find_renderer)
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: Result())

    assert client.get("/api/renderer/readiness").get_json()["verified"] is False
    recovered = client.get("/api/renderer/readiness").get_json()
    assert recovered["verified"] is True
    assert recovered["path"] == "/tmp/nam_render"


def test_oversized_upload_returns_a_clear_json_error(client, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 1)
    response = client.post(
        "/api/nam/upload",
        data={"file": (io.BytesIO(b"{}"), "too-large.nam")},
    )
    assert response.status_code == 413
    assert response.get_json()["error"] == "upload exceeds the 256 MiB limit"


def test_file_backed_session_embeds_nam_for_download_and_tools(client, tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    model_dir = session_dir / "models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    nam_bytes = jsonlib.dumps({"architecture": "WaveNet", "config": {"head_scale": 1.0}}).encode()
    session = {
        "type": "nam-mixer-session", "version": 1,
        "id": "saved-session", "name": "Saved session", "savedAt": "2026-09-09T10:00:00Z",
        "settings": {"mode": "hybrid"},
        "artifact": {"filename": "saved.nam", "nam_base64": base64.b64encode(nam_bytes).decode()},
    }
    saved = client.post("/api/sessions", json=session)
    assert saved.status_code == 201
    listed = client.get("/api/sessions").get_json()
    assert listed[0]["artifact"]["downloadUrl"] == "/api/sessions/saved-session/nam/download"
    assert listed[0]["artifact"]["toolPath"] == str(model_dir / "saved-session.nam")
    assert listed[0]["artifact"]["sha256"] == hashlib.sha256(nam_bytes).hexdigest()
    assert "nam_base64" not in listed[0]["artifact"]
    assert client.get("/api/sessions/saved-session/nam/download").data == nam_bytes
    tool_inspect = client.post("/api/nam/tools/inspect", json={"path": listed[0]["artifact"]["toolPath"]})
    assert tool_inspect.status_code == 200
    assert client.delete("/api/sessions/saved-session").status_code == 204
    assert not (model_dir / "saved-session.nam").exists()


def test_session_validation_report_must_match_embedded_nam(client, tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    model_dir = session_dir / "models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    nam_bytes = b'{"architecture":"WaveNet","config":{}}'
    sha256 = hashlib.sha256(nam_bytes).hexdigest()
    base = {
        "type": "nam-mixer-session", "version": 1, "id": "bound-report",
        "name": "Bound report", "savedAt": "2026-09-10T10:00:00Z",
        "settings": {"mode": "hybrid"},
        "artifact": {"filename": "model.nam", "nam_base64": base64.b64encode(nam_bytes).decode()},
    }
    mismatched = {
        **base,
        "validationReport": {"schema_version": 2, "model_sha256": "0" * 64, "state": "passed"},
    }
    rejected = client.post("/api/sessions", json=mismatched)
    assert rejected.status_code == 400
    assert "different NAM artifact" in rejected.get_json()["error"]

    matching = {
        **base,
        "validationReport": {"schema_version": 2, "model_sha256": sha256, "state": "passed"},
    }
    saved = client.post("/api/sessions", json=matching)
    assert saved.status_code == 201
    restored = client.get("/api/sessions").get_json()[0]
    assert restored["artifact"]["sha256"] == sha256
    assert restored["validationReport"]["model_sha256"] == sha256


def test_session_rejects_declared_artifact_hash_mismatch(client, tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    model_dir = session_dir / "models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    session = {
        "type": "nam-mixer-session", "version": 1, "id": "bad-hash",
        "name": "Bad hash", "savedAt": "2026-09-10T10:00:00Z",
        "settings": {},
        "artifact": {"filename": "model.nam", "nam_base64": base64.b64encode(b"model").decode(), "sha256": "f" * 64},
    }
    response = client.post("/api/sessions", json=session)
    assert response.status_code == 400
    assert not (model_dir / "bad-hash.nam").exists()


def test_generated_session_is_stored_in_and_deletes_its_bundle(client, tmp_path, monkeypatch):
    a2_dir = tmp_path / "a2"
    bundle = a2_dir / "demo"
    bundle.mkdir(parents=True)
    (bundle / "training_manifest.json").write_text(jsonlib.dumps({"mode": "hybrid", "model_name": "Demo", "amp_a": {}, "amp_b": {}, "design": {}}))
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    listed = client.get("/api/sessions").get_json()
    assert listed[0]["generated"] is True
    assert (bundle / "nam-mixer-session.json").is_file()
    assert client.delete(f"/api/sessions/{listed[0]['id']}").status_code == 204
    assert not bundle.exists()


def test_rejected_training_start_preserves_running_bundle_protection(client, tmp_path, monkeypatch):
    from hybrid.local_training import LocalTrainingManager

    a2_dir = tmp_path / "a2"
    for design in ("running-A", "other-B"):
        bundle = a2_dir / design
        bundle.mkdir(parents=True)
        (bundle / "training_manifest.json").write_text(jsonlib.dumps({"mode": "hybrid", "amp_a": {}, "amp_b": {}, "design": {}}))
    manager = LocalTrainingManager(tmp_path, a2_dir)
    manager.manifest_path = a2_dir / "running-A" / "training_manifest.json"
    manager.process = SimpleNamespace(poll=lambda: None)
    manager.state = "training"
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module, "_local_training_manager", manager)
    sessions = client.get("/api/sessions").get_json()
    running_session = next(item for item in sessions if item["designId"] == "running-A")

    response = client.post("/api/local_training/start", json={"design_id": "other-B"})
    assert response.status_code == 400
    assert manager.design_id == "running-A"
    response = client.delete(f"/api/sessions/{running_session['id']}")
    assert response.status_code == 409
    assert manager.manifest_path.is_file()


def test_nam_volume_tool_writes_only_a_new_validated_file(client, tmp_path):
    source = tmp_path / "Mesa.nam"
    _write_tool_nam(source)
    uploaded = client.post("/api/nam/upload", data={"file": (io.BytesIO(source.read_bytes()), "Mesa.nam")}).get_json()
    response = client.post("/api/nam/tools/volume", json={"path": uploaded["path"], "db_change": 6})
    assert response.status_code == 200
    data = response.get_json()
    assert data["changed_paths"] == [
        "config.submodels[0].model.config.head_scale",
        "config.submodels[0].model.metadata.loudness",
        "metadata.loudness",
    ]
    assert data["filename"] == "Mesa_+6dB.nam"
    downloaded = client.get(data["download_url"])
    assert downloaded.status_code == 200
    edited = jsonlib.loads(downloaded.data)
    assert edited["metadata"]["gain"] == 3.0
    assert edited["config"]["submodels"][0]["model"]["config"]["weights"] == [1]


def test_nam_metadata_tool_edits_descriptive_fields_only(client, tmp_path):
    source = tmp_path / "Meta.nam"
    _write_tool_nam(source)
    uploaded = client.post("/api/nam/upload", data={"file": (io.BytesIO(source.read_bytes()), "Meta.nam")}).get_json()
    response = client.post("/api/nam/tools/metadata", json={"path": uploaded["path"], "metadata": {"name": "Battery", "modeled_by": "Test", "gear_model": "Mark IIC+"}})
    assert response.status_code == 200
    edited = jsonlib.loads(client.get(response.get_json()["download_url"]).data)
    assert edited["metadata"]["name"] == "Battery"
    assert edited["metadata"]["modeled_by"] == "Test"
    assert edited["metadata"]["gear_model"] == "Mark IIC+"
    assert edited["metadata"]["gain"] == 3.0


def test_wizard_insight_requires_a_rendered_pair(client):
    app_module._rendered_pair_cache["pair"] = None
    resp = client.post("/api/wizard/insight", json={})
    assert resp.status_code == 400
    assert "render" in resp.get_json()["error"].lower()


def test_wizard_insight_describes_a_rendered_pair(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    resp = client.post("/api/wizard/insight", json={"render_id": _current_render_id()})
    assert resp.status_code == 200
    data = resp.get_json()
    assert {"level_text", "tone_text", "feel_text", "amp_a", "amp_b"} <= data.keys()


def test_live_blend_stems_returns_trimmed_stereo_pair_without_rerender(client, tmp_path):
    """The live-audition route must reuse the cached NAM output and leave the
    browser to perform only the final, adjustable linear mix."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    response = client.post("/api/live_blend_stems", json={"render_id": _current_render_id(),
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
    audio_at_0db = client.post("/api/preview", json={"source": "a", "render_id": data0["render_id"]}).data

    resp1 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="hot_humbucker"))
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert data1["input_profile_gain_db"] == 4.5
    assert data1["input_peak_dbfs"] > data0["input_peak_dbfs"]
    audio_at_hot = client.post("/api/preview", json={"source": "a", "render_id": data1["render_id"]}).data

    assert audio_at_0db != audio_at_hot


def test_stale_render_id_cannot_preview_or_generate_a_newer_pair(client, tmp_path, isolated_training_paths):
    training_path, _ = isolated_training_paths
    _write_training_wav(training_path)
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    old = client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).get_json()
    new = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db=2.0)).get_json()
    assert old["render_id"] != new["render_id"]
    preview = client.post("/api/preview", json={"source": "a", "render_id": old["render_id"]})
    generate = client.post("/api/generate", json={"render_id": old["render_id"], "model_name": "must-not-exist"})
    assert preview.status_code == generate.status_code == 409
    assert preview.get_json()["code"] == generate.get_json()["code"] == "stale_render"


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


def test_render_and_generation_retain_identical_source_bytes(client, tmp_path, isolated_training_paths, monkeypatch):
    training_path, _ = isolated_training_paths
    _write_training_wav(training_path)
    monkeypatch.setattr(app_module, "WORK_DIR", tmp_path / "work")
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    original = amp_a.read_bytes()
    observed_paths = []

    def changing_original(model, audio, sample_rate):
        # Simulate replacement during native inference. The retained model
        # is already separate, and later generation must consume those bytes.
        amp_a.write_text("replaced during inference")
        assert model.path.read_bytes() == original
        observed_paths.append(model.path)
        return np.asarray(audio, dtype=np.float32).copy()

    monkeypatch.setattr(pipeline, "render", changing_original)
    monkeypatch.setattr(training_target, "render", changing_original)
    rendered = client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    assert rendered.status_code == 200
    identity = rendered.get_json()
    expected_hash = hashlib.sha256(original).hexdigest()
    assert identity["source_hashes"]["amp_a"] == expected_hash
    generated = client.post("/api/generate", json={
        "render_id": identity["render_id"], "model_name": "Frozen source regression",
    })
    assert generated.status_code == 200
    manifest = jsonlib.loads(Path(generated.get_json()["manifest_path"]).read_text())
    provenance = manifest["preview_render_provenance"]
    assert provenance["source_hashes"] == identity["source_hashes"]
    assert Path(provenance["source_paths"]["amp_a"]).read_bytes() == original
    assert len(observed_paths) == 4
    assert observed_paths[:2] == observed_paths[2:]


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

    r1 = client.post("/api/blend_info", json={"render_id": _current_render_id(), "crossover_dbfs": -35.0, "transition_width_db": 8.0})
    r2 = client.post("/api/blend_info", json={"render_id": _current_render_id(), "crossover_dbfs": -5.0, "transition_width_db": 8.0})
    assert r1.status_code == 200
    assert r2.status_code == 200

    # amp_a/amp_b are identical fakes here, so auto_trim_db is always 0 --
    # the real proof that crossover moves the blend without a rerender is
    # the blend WEIGHT curve itself, which depends only on the envelope and
    # crossover config, not on which amps were rendered.
    curve1 = client.post("/api/blend_curve", json={"render_id": _current_render_id(), "crossover_dbfs": -35.0, "transition_width_db": 8.0, "max_points": 200})
    curve2 = client.post("/api/blend_curve", json={"render_id": _current_render_id(), "crossover_dbfs": -5.0, "transition_width_db": 8.0, "max_points": 200})
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

    quiet_render = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="vintage_single")).get_json()
    quiet_wav = client.post("/api/preview", json={"source": "a", "render_id": quiet_render["render_id"]}).data

    loud_render = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="extreme_passive")).get_json()
    loud_wav = client.post("/api/preview", json={"source": "a", "render_id": loud_render["render_id"]}).data

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
        "source": "hybrid", "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
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


def _comparison_bundle(tmp_path, monkeypatch):
    a2_dir, di_dir = tmp_path / "a2", tmp_path / "di"
    bundle = a2_dir / "design-1"
    bundle.mkdir(parents=True)
    di_dir.mkdir()
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module, "DI_DIR", di_dir)
    _write_training_wav(di_dir / "held-out.wav", n=1000)
    amp_a, amp_b, model = bundle / "a.nam", bundle / "b.nam", bundle / "model.nam"
    for path in (amp_a, amp_b, model):
        _write_fake_nam(path)
    from hybrid.design import HybridDesign
    HybridDesign(
        str(amp_a), str(amp_b), crossover_dbfs=-20.0, calibration_mode="raw",
    ).write_json(bundle / "hybrid_design.json")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "mode": "hybrid",
        "amp_a": {"path": str(amp_a), "sha256": sha(amp_a)},
        "amp_b": {"path": str(amp_b), "sha256": sha(amp_b)},
        "cab": {"selected": False},
        "output_gain": {"applied_gain_db": 0.0},
        "target": {"global_safety_gain_reduction_db": 0.0},
        "training": {"output_nam_path": str(model), "output_nam_sha256": sha(model)},
    }
    (bundle / "training_manifest.json").write_text(jsonlib.dumps(manifest))
    return bundle, amp_a, amp_b, model


def test_comparison_builds_synchronised_teacher_full_lite_stems(client, tmp_path, monkeypatch):
    _bundle, _a, _b, model = _comparison_bundle(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(app_module, "render_processed_reference", lambda design, manifest, dry, sr: SimpleNamespace(hybrid=dry * 2.0))
    def fake_model(path, dry, sr, slim=None):
        seen.append((Path(path), slim, float(dry[0])))
        return dry * (2.0 if slim == 0.0 else 1.5)
    monkeypatch.setattr(app_module, "render_trained_a2", fake_model)

    response = client.post("/api/comparison", json={
        "design_id": "design-1", "di_file": "held-out.wav", "input_gain_db": -6,
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["actual_output_levels"] is True
    assert [variant["id"] for variant in data["variants"]] == ["teacher", "full", "lite"]
    assert data["variants"][1]["metrics"]["raw_esr"] == pytest.approx(0.0)
    assert [entry[1] for entry in seen] == [0.0, 1.0]
    assert all(entry[0] == model for entry in seen)
    audio_response = client.get(data["audio_url"])
    audio, sample_rate = sf.read(io.BytesIO(audio_response.data), dtype="float32", always_2d=True)
    assert sample_rate == 48000
    assert audio.shape == (1000, 3)
    np.testing.assert_allclose(audio[:, 0], audio[:, 1], atol=1e-6)


def test_comparison_missing_teacher_source_keeps_model_available(client, tmp_path, monkeypatch):
    _bundle, amp_a, _b, _model = _comparison_bundle(tmp_path, monkeypatch)
    amp_a.unlink()
    called = False
    def should_not_render(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("inference must not start")
    monkeypatch.setattr(app_module, "render_trained_a2", should_not_render)

    response = client.post("/api/comparison", json={"design_id": "design-1", "di_file": "held-out.wav"})

    assert response.status_code == 409
    assert response.get_json()["code"] == "teacher_sources_unavailable"
    assert response.get_json()["model_available"] is True
    assert called is False


def test_imported_model_without_original_bundle_explains_teacher_is_unavailable(client, tmp_path, monkeypatch):
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    model = a2_dir / "embedded-copy.nam"
    _write_fake_nam(model)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)

    response = client.post("/api/comparison", json={
        "design_id": "missing-design", "model_path": str(model), "di_file": "anything.wav",
    })

    assert response.status_code == 409
    data = response.get_json()
    assert data["code"] == "teacher_sources_unavailable"
    assert data["model_available"] is True
    assert "embedded NAM remains usable" in data["error"]


def test_comparison_cache_identity_changes_with_gain_and_lite_can_be_unavailable(client, tmp_path, monkeypatch):
    _comparison_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "render_processed_reference", lambda design, manifest, dry, sr: SimpleNamespace(hybrid=dry))
    def fake_model(path, dry, sr, slim=None):
        if slim == 1.0:
            raise RuntimeError("export has no Lite branch")
        return dry
    monkeypatch.setattr(app_module, "render_trained_a2", fake_model)
    body = {"design_id": "design-1", "di_file": "held-out.wav", "input_gain_db": 0}

    first = client.post("/api/comparison", json=body).get_json()
    second = client.post("/api/comparison", json=body).get_json()
    quieter = client.post("/api/comparison", json={**body, "input_gain_db": -24}).get_json()

    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert first["comparison_id"] == second["comparison_id"]
    assert quieter["identity"] != first["identity"]
    assert first["variants"][2]["state"] == "unavailable"
    assert first["variants"][1]["metrics"]["raw_esr"] == pytest.approx(0.0)


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
    app_module._rendered_pair_cache["snapshot"] = None
    resp = client.post("/api/generate", json={})
    assert resp.status_code == 400


def test_generate_requires_training_input(client, isolated_training_paths, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={"render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0})
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
        "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
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
    resp = client.post("/api/mix_info", json={"render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0})
    assert resp.status_code == 200
    assert resp.get_json()["mode"] == "hybrid"


def test_mix_info_blend_mode_accepts_mix_b(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    resp = client.post("/api/mix_info", json={"render_id": _current_render_id(), "mode": "blend", "mix_b": 0.75})
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

    resp = client.post("/api/preview", json={"source": "blend", "render_id": _current_render_id(), "mix_b": 0.2, "auto_level": False})
    assert resp.status_code == 200
    assert resp.headers.get("X-Mix-B") is not None
    assert float(resp.headers["X-Mix-B"]) == pytest.approx(0.2)


def test_preview_invalid_mix_b_is_clamped(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/preview", json={"source": "blend", "render_id": _current_render_id(), "mix_b": 5.0, "auto_level": False})
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
                "source": source, "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
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

    resp = client.post("/api/generate", json={"render_id": _current_render_id(), "mode": "blend", "mix_b": 0.4, "auto_level": False, "manual_b_trim_db": 0.5})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode"] == "blend"
    assert Path(data["target_path"]).is_file()

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    assert manifest["mode"] == "blend"
    assert manifest["design"]["mix_b"] == pytest.approx(0.4)
    assert manifest["artifact_filename"] == "a_and_b_Parallel_Blend.nam"
    assert data["download_filename"] == "a_and_b_Parallel_Blend.nam"


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
        "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
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

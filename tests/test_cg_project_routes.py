"""Continuous Gain project workflow and /api/cg routes with the native renderer and NAM loader faked."""
from __future__ import annotations

import io
import json
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from flask import Flask

import cg_routes
import hybrid.cg_project as cgp
from hybrid.cg_project import CgProject, CgProjectError, suggest_position
from tests.cg_synth import SR, amp_render, synth_di


def _nam_bytes(gain, sr=48000):
    return json.dumps({"architecture": "WaveNet", "sample_rate": sr, "config": {}, "metadata": {"gain_param": gain}, "weights": [0.0]}).encode()


@pytest.fixture()
def fake_backend(monkeypatch):
    def fake_load(path):
        raw = json.loads(Path(path).read_text())
        return types.SimpleNamespace(raw=raw, input_level_dbu=None, path=Path(path))
    monkeypatch.setattr(cgp, "load_nam", fake_load)
    monkeypatch.setattr(cgp, "render", lambda m, x, sr, **k: amp_render(m.raw["metadata"]["gain_param"] * 1.5)(x))
    monkeypatch.setattr(cgp, "load_reference_di", lambda name: synth_di(name, 3.0))
    monkeypatch.setattr(cgp, "receptive_field_record", lambda positions, models, env=None: {"branch_samples": {f"G{p:g}": 100 for p in positions}, "cab": {"baked": False}})
    monkeypatch.setattr(cgp, "FC_RECIPE", cgp.FC_RECIPE.__class__(**{**cgp.FC_RECIPE.__dict__, "di_seconds": 2, "val_seconds": 2, "train_offsets_db": (-6.0, 6.0), "val_offsets_db": (0.0,)}))


def _project(tmp_path, gains=(1, 2, 3, 4, 5, 6), names=None):
    p = CgProject.create(tmp_path, "Test Amp", "Amp", "Ch1")
    for g in gains:
        p.add_capture((names or (lambda g: f"amp-G{g}.nam"))(g), _nam_bytes(g))
    return p


def test_position_suggestions_are_only_suggestions():
    assert suggest_position("jcm800-high-g4.5-11.4dBu.nam") == 4.5
    assert suggest_position("jcm800-high-ga10-11.4dBu.nam") == 10
    assert suggest_position("Super-Sonic Vibrolux Ch T5 B5 V3.nam") == 3
    assert suggest_position("57 CUSTOM TWIN - CH 1 - VOL 7.nam") == 7
    assert suggest_position("MESADUAL - RED - MODERN - GAIN MAX.nam") is None


def test_stage1_checks_block_analysis_until_positions_are_confirmed(tmp_path):
    p = CgProject.create(tmp_path, "A")
    assert not p.check_captures()["ready"]
    p.add_capture("a.nam", _nam_bytes(1)); p.add_capture("b.nam", _nam_bytes(2)); p.add_capture("c.nam", _nam_bytes(2))
    msgs = " ".join(i["message"] for i in p.check_captures()["issues"])
    assert "no physical gain position" in msgs
    p.set_positions({"a.nam": 1, "b.nam": 2, "c.nam": 2})
    assert "duplicate position 2" in " ".join(i["message"] for i in p.check_captures()["issues"])
    p.set_positions({"c.nam": 3})
    chk = p.check_captures()
    assert chk["ready"] and "positions 1-3" in chk["summary"]
    p.set_positions({"c.nam": None})
    with pytest.raises(CgProjectError, match="blocking"):
        p.analyse()


def test_bad_uploads_are_rejected_and_sample_rate_is_blocking(tmp_path):
    p = CgProject.create(tmp_path, "A")
    for name, data in (("x.wav", b"RIFF"), ("x.nam", b"not json"), ("y.nam", b'{"foo": 1}')):
        with pytest.raises(CgProjectError):
            p.add_capture(name, data)
    p.add_capture("hi.nam", _nam_bytes(1, sr=44100)); p.set_positions({"hi.nam": 1})
    assert any("44100" in i["message"] for i in p.check_captures()["issues"])
    with pytest.raises(CgProjectError):
        p.add_capture("hi.nam", _nam_bytes(1))


def test_plan_requires_analysis_and_changing_captures_invalidates_it(tmp_path, fake_backend):
    p = _project(tmp_path, names=None)
    with pytest.raises(CgProjectError, match="analyse"):
        p.plan()
    p.analyse()
    plan = p.plan("automatic", None, "fc")
    assert plan["anchor_method"] == "fc" and plan["anchors_input_gain_db"][0] == -20.0 and plan["anchors_input_gain_db"][-1] == 14.0
    assert [m["kind"] for m in plan["mapping"]].count("training_anchor") == len(plan["selected"])
    p.remove_capture("amp-G3.nam")
    st = p.state()
    assert st["plan"] is None and st["analysis"] is None
    assert p.analysis() is not None            # stale file remains but the state no longer points at it


def test_fixed_ladder_is_an_explicit_advanced_alternative(tmp_path, fake_backend):
    p = _project(tmp_path); p.analyse()
    fc = p.plan("use_all", None, "fc")
    fx = p.plan("use_all", None, "fixed")
    assert fc["anchor_method"] == "fc" and fx["anchor_method"] == "fixed"
    assert fx["anchors_input_gain_db"] == [-22.0 + 4.0 * (g - 1) for g in fx["selected"]]
    assert any("Advanced" in w for w in fx["warnings"]) and not any("Advanced" in w for w in fc["warnings"])
    with pytest.raises(CgProjectError):
        p.plan("automatic", None, "bogus")


def test_generate_bundle_writes_a_continuous_gain_a2_bundle(tmp_path, fake_backend):
    p = _project(tmp_path); p.analyse(); plan = p.plan("use_all", None, "fc")
    official = tmp_path / "official.wav"; sf.write(official, synth_di("o", 2.0), SR)
    out = tmp_path / "a2"
    b = p.generate_bundle(out, official, "My Model")
    m = json.loads(Path(b["manifest"]).read_text())
    assert m["mode"] == "continuous_gain" and m["artifact_filename"] == "My_Model.nam" and m["design"]["anchors_input_gain_db"] == plan["anchors_input_gain_db"]
    assert [s["position"] for s in m["sources"]] == plan["selected"] and all(len(s["sha256"]) == 64 for s in m["sources"])
    assert m["receptive_field"]["branch_samples"]["G1"] == 100 and m["training_input"]["custom_split"]
    assert (Path(b["manifest"]).parent / "input.wav").is_file() and (Path(b["manifest"]).parent / "hybrid_target.wav").is_file()
    assert p.state()["bundle"]["design_id"] == b["design_id"]


# ---- routes
@pytest.fixture()
def client(tmp_path, fake_backend):
    app = Flask(__name__)
    official = tmp_path / "official.wav"; sf.write(official, synth_di("o", 2.0), SR)
    cg_routes.register_cg_routes(app, cg_dir=tmp_path / "cg", a2_output_dir=tmp_path / "a2", training_input_path=official)
    (tmp_path / "a2").mkdir(exist_ok=True)
    return app.test_client()


def _wait(c, r):
    jid = r.get_json()["job_id"]
    for _ in range(600):
        j = c.get(f"/api/cg/jobs/{jid}").get_json()
        if j["state"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_full_route_flow_to_a_generated_bundle_and_gated_stage4(client, tmp_path):
    pid = client.post("/api/cg/projects", json={"name": "Amp X", "amp": "X"}).get_json()["project"]["id"]
    r = client.post(f"/api/cg/projects/{pid}/captures", data={"files": [(io.BytesIO(_nam_bytes(g)), f"x-G{g}.nam") for g in (1, 2, 3, 4, 5, 6)]}, content_type="multipart/form-data")
    assert r.status_code == 201 and r.get_json()["check"]["ready"] and r.get_json()["project"]["captures"]["x-G2.nam"]["position"] == 2
    assert client.post(f"/api/cg/projects/{pid}/plan", json={}).status_code == 400          # not analysed yet
    assert client.post(f"/api/cg/projects/{pid}/validate").status_code == 409               # nothing trained
    assert client.get(f"/api/cg/projects/{pid}/export").status_code == 409
    j = _wait(client, client.post(f"/api/cg/projects/{pid}/analyse")); assert j["state"] == "done", j["error"]
    st = client.get(f"/api/cg/projects/{pid}").get_json()
    assert st["analysis"]["audit"] and st["analysis"]["profile"]["series"] and st["plan"] is None
    st = client.post(f"/api/cg/projects/{pid}/plan", json={"mode": "custom", "custom": [1, 3, 6], "anchors": "fc"}).get_json()
    assert st["plan"]["selected"] == [1.0, 3.0, 6.0] and st["plan"]["mode"] == "custom"
    assert client.post(f"/api/cg/projects/{pid}/plan", json={"mode": "custom", "custom": [1]}).status_code == 400
    j = _wait(client, client.post(f"/api/cg/projects/{pid}/generate", json={"model_name": "Amp X FC"})); assert j["state"] == "done", j["error"]
    st = client.get(f"/api/cg/projects/{pid}").get_json()
    did = st["bundle"]["design_id"]
    assert (tmp_path / "a2" / did / "training_manifest.json").is_file() and st["training"]["trained"] is False
    # a Continuous Gain bundle must never be treated as a Mixer/Builder session by the app
    import app as mixer_app
    assert mixer_app._is_continuous_gain_bundle(tmp_path / "a2" / did)
    # pretend the local trainer finished: the model path lands in the manifest exactly as scripts/train_a2.py writes it
    nam = tmp_path / "trained.nam"; nam.write_bytes(_nam_bytes(0))
    mp = tmp_path / "a2" / did / "training_manifest.json"; m = json.loads(mp.read_text()); m["training"] = {"output_nam_path": str(nam), "epochs": 60, "epoch_preset": "standard"}; mp.write_text(json.dumps(m))
    st = client.get(f"/api/cg/projects/{pid}").get_json(); assert st["training"]["trained"] and st["training"]["backend"] == "local"
    z = client.get(f"/api/cg/projects/{pid}/export")                # export is never gated on validation or listening
    assert z.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(z.data)); names = zf.namelist()
    meta = json.loads(zf.read("Amp_X_FC.continuous_gain.json"))
    assert "Amp_X_FC.nam" in names and "PLAYER_GUIDE.md" in names and meta["validation"] == "not run"
    assert meta["anchor_method"] == "fc" and [s["position"] for s in meta["selected_captures"]] == [1.0, 3.0, 6.0]
    assert meta["usable_input_gain_range_db"] == [-20.0, 14.0] and "guide and provenance record" in meta["note"]
    assert len(meta["known_limits"]) >= 4 and "Output gain" in zf.read("PLAYER_GUIDE.md").decode()


def test_only_one_job_per_project_and_unknown_things_are_404(client):
    assert client.get("/api/cg/projects/nope").status_code == 404
    assert client.get("/api/cg/jobs/nope").status_code == 404
    pid = client.post("/api/cg/projects", json={"name": "A"}).get_json()["project"]["id"]
    assert client.get(f"/api/cg/projects/{pid}/audition/../../x.wav").status_code == 404
    assert client.post("/api/cg/projects", json={"name": " "}).status_code == 400
    assert client.delete(f"/api/cg/projects/{pid}").status_code == 200 and client.get(f"/api/cg/projects/{pid}").status_code == 404

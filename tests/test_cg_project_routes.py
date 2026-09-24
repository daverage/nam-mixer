"""Continuous Gain project workflow and /api/cg routes with the native renderer and NAM loader faked."""
from __future__ import annotations

import base64
import hashlib
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

from routes import continuous_gain as cg_routes
import hybrid.continuous_gain.project as cgp
from hybrid.continuous_gain.project import CgProject, CgProjectError, suggest_position
from hybrid.core.cab_ir import cab_design_from_prepared, get_prepared_cab_ir
from tests.cg_synth import SR, amp_render, synth_di


def _nam_bytes(gain, sr=48000):
    return json.dumps({"architecture": "WaveNet", "sample_rate": sr, "config": {}, "metadata": {"gain_param": gain}, "weights": [0.0]}).encode()


@pytest.fixture()
def fake_backend(monkeypatch):
    def fake_load(path):
        raw = json.loads(Path(path).read_text())
        return types.SimpleNamespace(raw=raw, input_level_dbu=None, path=Path(path),
                                     gear_type=(raw.get("metadata") or {}).get("gear_type"))   # as NamModel.gear_type
    monkeypatch.setattr(cgp, "load_nam", fake_load)
    monkeypatch.setattr(cgp, "render", lambda m, x, sr, **k: amp_render(m.raw["metadata"]["gain_param"] * 1.5)(x))
    monkeypatch.setattr(cgp, "load_reference_di", lambda name: synth_di(name, 3.0))
    monkeypatch.setattr(cgp, "receptive_field_record", lambda positions, models, env=None, cab=None: {"branch_samples": {f"G{p:g}": 100 for p in positions}, "cab": {"baked": False}})
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


def test_continuous_gain_cab_is_a_second_artifact_not_part_of_training_target(tmp_path, fake_backend):
    p = _project(tmp_path); p.analyse(); p.plan("use_all", None, "fc")
    official = tmp_path / "official.wav"; sf.write(official, synth_di("o", 2.0), SR)
    plain = p.generate_bundle(tmp_path / "a2", official, "Plain")
    plain_target, _ = sf.read(Path(plain["manifest"]).parent / "hybrid_target.wav", dtype="float32")

    ir_path = tmp_path / "cab.wav"
    sf.write(ir_path, np.array([1.0, 0.5, -0.25], dtype=np.float32), SR, subtype="FLOAT")
    prepared = get_prepared_cab_ir(ir_path, SR)
    cab = cab_design_from_prepared(prepared, ir_path.name, preview_enabled=False, export_mode="embedded", display_name="Test Cab")
    with_cab = p.generate_bundle(tmp_path / "a2", official, "With Cab", cab=cab)
    cab_target, _ = sf.read(Path(with_cab["manifest"]).parent / "hybrid_target.wav", dtype="float32")
    manifest = json.loads(Path(with_cab["manifest"]).read_text())

    assert np.array_equal(cab_target, plain_target)
    assert manifest["cab"]["export_mode"] == "embedded" and manifest["cab"]["baked"] is False
    assert manifest["output_gain"]["embedded_final"]["final_linear_scalar"] > 0



def test_continuous_gain_learned_cab_is_convolved_into_each_training_segment(tmp_path, fake_backend):
    from hybrid.core.cab_ir import apply_cab_ir
    from hybrid.training.nam_provenance import export_gear_type, export_model_name

    p = _project(tmp_path); p.analyse(); p.plan("use_all", None, "fc")
    official = tmp_path / "official.wav"; sf.write(official, synth_di("o", 2.0), SR)
    plain = p.generate_bundle(tmp_path / "a2", official, "Plain")
    plain_manifest = json.loads(Path(plain["manifest"]).read_text())
    plain_target, _ = sf.read(Path(plain["manifest"]).parent / "hybrid_target.wav", dtype="float32")

    ir_path = tmp_path / "cab.wav"
    sf.write(ir_path, np.array([1.0, 0.5, -0.25], dtype=np.float32), SR, subtype="FLOAT")
    prepared = get_prepared_cab_ir(ir_path, SR)
    cab = cab_design_from_prepared(prepared, ir_path.name, preview_enabled=False, export_mode="learned", display_name="Test Cab")
    learned = p.generate_bundle(tmp_path / "a2", official, "Learned", cab=cab)
    manifest = json.loads(Path(learned["manifest"]).read_text())
    learned_target, _ = sf.read(Path(learned["manifest"]).parent / "hybrid_target.wav", dtype="float32")

    # Each segment is the cab-free target through the cabinet, then one peak-ceiling gain for the whole file.
    c_plain, c_learned = plain_manifest["target"]["output_scale_c"], manifest["target"]["output_scale_c"]
    for seg in manifest["segments"]:
        a, b = seg["start"], seg["stop"]
        expected = apply_cab_ir(plain_target[a:b] / c_plain, prepared)
        np.testing.assert_allclose(learned_target[a:b] / c_learned, expected, atol=1e-5)
    assert manifest["cab"]["export_mode"] == "learned" and manifest["cab"]["baked"] is True
    assert "embedded_final" not in manifest["output_gain"]
    assert export_model_name(manifest) == "Learned + Test Cab [Learned Cab]"
    assert export_gear_type(manifest) == "amp_cab"


def test_receptive_field_record_marks_a_learned_cab_as_baked(tmp_path):
    from hybrid.continuous_gain.bundle import receptive_field_record

    ir_path = tmp_path / "cab.wav"
    sf.write(ir_path, np.array([1.0, 0.5, -0.25], dtype=np.float32), SR, subtype="FLOAT")
    prepared = get_prepared_cab_ir(ir_path, SR)
    for mode, baked in (("learned", True), ("embedded", False)):
        cab = cab_design_from_prepared(prepared, ir_path.name, preview_enabled=False, export_mode=mode)
        record = receptive_field_record([], {}, None, cab=cab)["cab"]
        assert record["baked"] is baked and record["fir_history_samples"] == 2


def test_learned_cab_validation_compares_the_captures_through_the_same_cabinet():
    from hybrid.continuous_gain import validation as cgv

    calls = []
    original = cgv.render
    try:
        cgv.render = lambda model, x, sr, **k: np.asarray(x, dtype=np.float32) * 2.0
        ref = cgv._reference_render({1.0: object()}, {1.0: 0}, lambda y: calls.append(len(y)) or y + 1.0)
        out = ref(1.0, np.ones(4, dtype=np.float32))
    finally:
        cgv.render = original
    assert calls == [4] and np.allclose(out, 3.0)


# ---- routes
@pytest.fixture()
def client(tmp_path, fake_backend):
    app = Flask(__name__)
    official = tmp_path / "official.wav"; sf.write(official, synth_di("o", 2.0), SR)
    records = []
    cg_routes.register_cg_routes(app, cg_dir=tmp_path / "cg", a2_output_dir=tmp_path / "a2", training_input_path=official,
                                 store_session=lambda rec: records.append(json.loads(json.dumps(rec))))
    (tmp_path / "a2").mkdir(exist_ok=True)
    c = app.test_client()
    c.session_records = records
    return c


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
    assert client.post(f"/api/cg/projects/{pid}/plan", json={"mode": "custom", "custom": "136"}).status_code == 400   # not positions 1, 3, 6
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
    # ---- the Sessions record: what it is, how far it got, and the finished NAM (+ validation report only if it belongs to that NAM)
    last = [r for r in client.session_records if r["id"] == pid][-1]
    assert last["type"] == "nam-mixer-session" and last["version"] == 1 and last["name"] == "Amp X" and last["settings"]["mode"] == "continuous_gain"
    cgs = last["settings"]["continuousGain"]
    assert cgs["projectId"] == pid and cgs["captures"] == 6 and cgs["selected"] == [1.0, 3.0, 6.0] and cgs["stage"] == "trained" and cgs["anchorMethod"] == "fc"
    assert last["designId"] == did and base64.b64decode(last["artifact"]["nam_base64"]) == nam.read_bytes() and "validationReport" not in last
    stages = [r["settings"]["continuousGain"]["stage"] for r in client.session_records if r["id"] == pid]
    assert stages[0] == "captures" and stages.index("analysed") < stages.index("planned") < stages.index("files") < stages.index("trained")
    good = {"model_sha256": hashlib.sha256(nam.read_bytes()).hexdigest(), "state": "ok"}
    m = json.loads(mp.read_text()); m["training"]["validation_report"] = good; mp.write_text(json.dumps(m))
    client.get(f"/api/cg/projects/{pid}")
    assert [r for r in client.session_records if r["id"] == pid][-1]["validationReport"] == good
    n_before = len(client.session_records); client.get(f"/api/cg/projects/{pid}"); client.get(f"/api/cg/projects/{pid}")
    assert len(client.session_records) == n_before                    # unchanged state does not rewrite the record
    hashed = []
    real_sha256 = hashlib.sha256
    def counting_sha256(data=b"", *a, **k):
        if len(data) == len(nam.read_bytes()):
            hashed.append(1)
        return real_sha256(data, *a, **k)
    import routes.continuous_gain as cg_mod
    cg_mod.hashlib.sha256 = counting_sha256
    try:
        for _ in range(3):
            client.get(f"/api/cg/projects/{pid}")
    finally:
        cg_mod.hashlib.sha256 = real_sha256
    assert hashed == []                                                # the unchanged NAM is not re-read and re-hashed per request
    m["training"]["validation_report"] = {"model_sha256": "0" * 64}; mp.write_text(json.dumps(m)); client.get(f"/api/cg/projects/{pid}")
    assert "validationReport" not in [r for r in client.session_records if r["id"] == pid][-1]        # a report for a different NAM is never attached
    z = client.get(f"/api/cg/projects/{pid}/export")                # export is never gated on validation or listening
    assert z.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(z.data)); names = zf.namelist()
    meta = json.loads(zf.read("Amp_X_FC.continuous_gain.json"))
    assert "Amp_X_FC.nam" in names and "PLAYER_GUIDE.md" in names and meta["validation"] == "not run"
    assert meta["anchor_method"] == "fc" and [s["position"] for s in meta["selected_captures"]] == [1.0, 3.0, 6.0]
    assert meta["usable_input_gain_range_db"] == [-20.0, 14.0] and "guide and provenance record" in meta["note"]
    assert len(meta["known_limits"]) >= 4 and "Output gain" in zf.read("PLAYER_GUIDE.md").decode()
    # ---- a validation report is only ever shown/synced/exported for the model it measured
    vf = tmp_path / "cg" / pid / "validation.json"
    report = {"design_id": did, "compatibility": {"standard_nam": True}, "progression": {"reversals": []}}
    vf.write_text(json.dumps({**report, "model": {"sha256": "0" * 64}}))              # an earlier model's report
    assert client.get(f"/api/cg/projects/{pid}").get_json()["validation"] is None
    assert json.loads(zipfile.ZipFile(io.BytesIO(client.get(f"/api/cg/projects/{pid}/export").data))
                      .read("Amp_X_FC.continuous_gain.json"))["validation"] == "not run"
    assert [r for r in client.session_records if r["id"] == pid][-1]["settings"]["continuousGain"]["stage"] == "trained"
    vf.write_text(json.dumps({**report, "model": {"sha256": hashlib.sha256(nam.read_bytes()).hexdigest()}}))
    assert client.get(f"/api/cg/projects/{pid}").get_json()["validation"]["design_id"] == did


def test_only_one_job_per_project_and_unknown_things_are_404(client):
    pid0 = client.post("/api/cg/projects", json={"name": "B"}).get_json()["project"]["id"]
    bad = client.post(f"/api/cg/projects/{pid0}/captures", data={"files": [(io.BytesIO(b"not a nam"), "notes.nam")]}, content_type="multipart/form-data")
    assert bad.status_code == 400 and "not a readable .nam" in bad.get_json()["error"]          # the UI shows this text
    assert client.get("/api/cg/projects/nope").status_code == 404
    assert client.get("/api/cg/jobs/nope").status_code == 404
    pid = client.post("/api/cg/projects", json={"name": "A"}).get_json()["project"]["id"]
    assert client.get(f"/api/cg/projects/{pid}/audition/../../x.wav").status_code == 404
    assert client.post("/api/cg/projects", json={"name": " "}).status_code == 400


# ---- Sessions integration (app.py): listed and labelled like every other project, deleted with its working files
def test_continuous_gain_sessions_are_listed_labelled_and_deleted_with_their_files(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "SESSION_DIR", tmp_path / "sessions"); (tmp_path / "sessions").mkdir()
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", tmp_path / "sessions" / "models"); (tmp_path / "sessions" / "models").mkdir()
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path / "a2")
    monkeypatch.setattr(app_module, "CG_PROJECT_DIR", tmp_path / "cg")
    proj = CgProject.create(tmp_path / "cg", "Delete me", "Amp")
    pid = proj.root.name
    design = tmp_path / "a2" / f"{pid}-1"; design.mkdir(parents=True)
    (design / "training_manifest.json").write_text(json.dumps({"mode": "continuous_gain"}))
    nam = _nam_bytes(1)
    rec = {"type": "nam-mixer-session", "version": 1, "id": pid, "name": "Delete me", "savedAt": "2026-09-21T00:00:00+00:00",
           "settings": {"mode": "continuous_gain", "continuousGain": {"projectId": pid, "stage": "trained", "captures": 3}}, "designId": design.name,
           "artifact": {"filename": "m.nam", "nam_base64": base64.b64encode(nam).decode()}}
    app_module._store_session_record(rec)
    c = app_module.app.test_client()
    listed = [s for s in c.get("/api/sessions").get_json() if s["id"] == pid]
    assert len(listed) == 1 and listed[0]["settings"]["mode"] == "continuous_gain" and listed[0]["artifact"]["downloadUrl"].endswith("/nam/download")
    assert c.get(f"/api/sessions/{pid}/nam/download").data == nam                    # the same "Download NAM" the other sessions use
    assert not [s for s in c.get("/api/sessions").get_json() if s["id"] == design.name]   # the bundle itself is not a second session
    assert c.delete(f"/api/sessions/{pid}").status_code == 204
    assert not proj.root.exists() and not design.exists() and not (tmp_path / "sessions" / f"{pid}.nam-mixer.json").exists()


def test_analysis_is_identical_serial_or_parallel_and_probes_are_cached_by_file_hash(tmp_path, fake_backend, monkeypatch):
    calls = []
    real_probe = cgp.probe_capture
    monkeypatch.setattr(cgp, "probe_capture", lambda *a, **k: (calls.append(1), real_probe(*a, **k))[1])
    def analysed(workers):
        monkeypatch.setenv("NAM_MIXER_CG_WORKERS", str(workers))
        p = _project(tmp_path / f"w{workers}", ); p.analyse()
        return p, json.loads(p.analysis_file.read_text())
    (tmp_path / "w1").mkdir(); (tmp_path / "w4").mkdir()
    n0 = len(calls); p1, a1 = analysed(1); n1 = len(calls); p4, a4 = analysed(4)
    assert n1 - n0 == 6 and len(calls) - n1 == 6
    for k in ("audit", "profile", "selection", "probes"):
        assert a1[k] == a4[k]                                         # parallel probing changes nothing but the wall time
    before = len(calls)
    p4.analyse()                                                      # nothing changed: every probe comes from the cache
    assert len(calls) == before and json.loads(p4.analysis_file.read_text())["probes"] == a4["probes"]
    p4.add_capture("amp-G7.nam", _nam_bytes(7)); p4.set_positions({"amp-G7.nam": 7}); p4.analyse()
    assert len(calls) == before + 1                                   # only the new capture was probed
    p4.remove_capture("amp-G7.nam"); p4.analyse()
    assert len(calls) == before + 1                                   # and removing one needs no probe at all


@pytest.fixture()
def cab_client(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "test.env"))
    app = Flask(__name__)
    official = tmp_path / "official.wav"; sf.write(official, synth_di("o", 2.0), SR)
    cab_dir = tmp_path / "cabs"; cab_dir.mkdir()
    sf.write(cab_dir / "cab.wav", np.array([1.0, 0.5, -0.25], dtype=np.float32), SR, subtype="FLOAT")
    cg_routes.register_cg_routes(app, cg_dir=tmp_path / "cg", a2_output_dir=tmp_path / "a2", training_input_path=official,
                                 store_session=lambda rec: None, cab_upload_dir=cab_dir)
    (tmp_path / "a2").mkdir(exist_ok=True)
    c = app.test_client()
    c.cab_path = str(cab_dir / "cab.wav")
    return c


@pytest.mark.parametrize("enabled", [False, True])
def test_cg_embedded_cab_needs_experimental_architectures_but_learned_does_not(cab_client, monkeypatch, enabled):
    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "true" if enabled else "false")
    pid = cab_client.post("/api/cg/projects", json={"name": "Amp X", "amp": "X"}).get_json()["project"]["id"]
    assert cab_client.get(f"/api/cg/projects/{pid}").get_json()["experimental_architectures"] is enabled

    embedded = cab_client.post(f"/api/cg/projects/{pid}/generate", json={"cab_path": cab_client.cab_path, "cab_export_mode": "embedded"})
    if enabled:
        assert "job_id" in embedded.get_json()
    else:
        assert embedded.status_code == 409 and "experimental" in embedded.get_json()["error"]
    for jid in [embedded.get_json().get("job_id")] if enabled else []:
        _wait(cab_client, types.SimpleNamespace(get_json=lambda jid=jid: {"job_id": jid}))

    learned = cab_client.post(f"/api/cg/projects/{pid}/generate", json={"cab_path": cab_client.cab_path, "cab_export_mode": "learned"})
    assert "job_id" in learned.get_json(), learned.get_json()
    _wait(cab_client, learned)


# ---- Stage 4: validation and downloads through the routes

def _trained_project(c, tmp_path, generate_payload=None):
    """Captures -> analysis -> plan -> training files -> a 'trained' NAM recorded in the manifest (as the local trainer does)."""
    pid = c.post("/api/cg/projects", json={"name": "Amp X", "amp": "X"}).get_json()["project"]["id"]
    c.post(f"/api/cg/projects/{pid}/captures", data={"files": [(io.BytesIO(_nam_bytes(g)), f"x-G{g}.nam") for g in (1, 2, 3, 4, 5, 6)]},
           content_type="multipart/form-data")
    assert _wait(c, c.post(f"/api/cg/projects/{pid}/analyse"))["state"] == "done"
    c.post(f"/api/cg/projects/{pid}/plan", json={"mode": "custom", "custom": [1, 3, 6], "anchors": "fc"})
    j = _wait(c, c.post(f"/api/cg/projects/{pid}/generate", json={"model_name": "Amp X FC", **(generate_payload or {})}))
    assert j["state"] == "done", j["error"]
    did = c.get(f"/api/cg/projects/{pid}").get_json()["bundle"]["design_id"]
    nam = tmp_path / "trained.nam"; nam.write_bytes(_nam_bytes(3))
    mp = tmp_path / "a2" / did / "training_manifest.json"
    m = json.loads(mp.read_text()); m["training"] = {"output_nam_path": str(nam)}; mp.write_text(json.dumps(m))
    return pid, mp, nam


@pytest.fixture()
def fake_validation(monkeypatch):
    import hybrid.continuous_gain.validation as cgv

    def fake_load(path):
        raw = json.loads(Path(path).read_text())
        return types.SimpleNamespace(raw=raw, input_level_dbu=None, path=Path(path), gear_type=(raw.get("metadata") or {}).get("gear_type"))
    fake_render = lambda m, x, sr, **k: amp_render(m.raw["metadata"]["gain_param"] * 1.5)(x)  # noqa: E731
    for module in (cg_routes, cgv):
        monkeypatch.setattr(module, "load_nam", fake_load)
    monkeypatch.setattr(cgv, "render", fake_render)
    monkeypatch.setattr(cg_routes, "load_reference_di", lambda name: synth_di(name, 3.0))


@pytest.mark.parametrize("learned_cab", [False, True])
def test_validation_route_runs_every_check_and_reports_a_learned_cab(cab_client, tmp_path, monkeypatch, fake_validation, learned_cab):
    payload = {"cab_path": cab_client.cab_path, "cab_export_mode": "learned", "cab_display_name": "Test Cab"} if learned_cab else {}
    pid, mp, nam = _trained_project(cab_client, tmp_path, payload)
    assert json.loads(mp.read_text())["cab"]["baked"] is learned_cab

    j = _wait(cab_client, cab_client.post(f"/api/cg/projects/{pid}/validate"))
    assert j["state"] == "done", j["error"]
    v = cab_client.get(f"/api/cg/projects/{pid}").get_json()["validation"]
    assert v["model"]["sha256"] == hashlib.sha256(nam.read_bytes()).hexdigest()
    assert v["compatibility"]["standard_nam"] is True and v["safety"]["finite"] is True
    assert {r["position"] for r in v["progression"]["positions"]} == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    assert {r["role"] for r in v["progression"]["positions"]} == {"training", "reference"}
    assert v["blocks_export"] is False and v["learned_cab"] is learned_cab and ("cab_note" in v) is learned_cab
    assert cab_client.get(f"/api/cg/projects/{pid}/audition/sweep.wav").status_code == 200
    assert cab_client.get(f"/api/cg/projects/{pid}/audition/..%2Fproject.json").status_code == 404


def test_nam_download_route_serves_the_tested_nam_and_gates_the_embedded_one(client, tmp_path, monkeypatch):
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "test.env"))
    pid, mp, nam = _trained_project(client, tmp_path)
    stem = json.loads(mp.read_text())["artifact_stem"]

    head = client.get(f"/api/cg/projects/{pid}/nam/download")
    assert head.status_code == 200 and head.data == nam.read_bytes()
    assert f'filename={stem}.nam' in head.headers["Content-Disposition"]
    assert client.get(f"/api/cg/projects/{pid}/nam/download?artifact=other").status_code == 400

    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "false")
    refused = client.get(f"/api/cg/projects/{pid}/nam/download?artifact=cab")
    assert refused.status_code == 409 and "experimental" in refused.get_json()["error"]

    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "true")
    assert client.get(f"/api/cg/projects/{pid}/nam/download?artifact=cab").status_code == 409     # nothing validated yet
    sequential = tmp_path / "with-cab.nam"
    m = json.loads(mp.read_text())
    m["training"]["embedded_artifact"] = {"state": "validated", "artifacts": {"sequential_nam_path": str(sequential)}}
    mp.write_text(json.dumps(m))
    assert client.get(f"/api/cg/projects/{pid}/nam/download?artifact=cab").status_code == 404     # recorded but missing on disk
    sequential.write_text("{}")
    cab = client.get(f"/api/cg/projects/{pid}/nam/download?artifact=cab")
    assert cab.status_code == 200 and f"filename={stem}-with-cab.nam" in cab.headers["Content-Disposition"]

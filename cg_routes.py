"""Flask routes for the Continuous Gain tab (`/api/cg/*`): a thin layer over `hybrid.cg_project` / `hybrid.cg_validation`.

Training is NOT implemented here. `POST /api/cg/projects/<id>/generate` writes an ordinary A2 bundle into the app's
existing A2 output directory (mode "continuous_gain"), and the UI then starts it through the EXISTING
`/api/local_training/*` and `/api/kaggle/*` routes with the returned `design_id`.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

from flask import jsonify, request, send_file
from werkzeug.utils import secure_filename

from hybrid.a2_training_settings import A2_EPOCH_PRESETS
from hybrid.cg_project import ANCHOR_METHODS, SELECTION_MODES, CgProject, CgProjectError
from hybrid.cg_validation import (HELD_OUT_DIS, SWEEP_GAINS_DB, check_compatibility, check_progression, check_safety, write_audition)
from hybrid.cg_audit import alignment_shift
from hybrid.kaggle_training import find_active_job
from hybrid.cg_probe import SR, load_reference_di
from hybrid.nam_loader import load_nam

_JOBS: dict[str, dict] = {}
_JOB_LOCK = threading.Lock()
_PROFILE_SERIES_FOR_UI = ["music RMS @0 dB DI", "EQ presence", "EQ low", "HF >3 kHz", "THD 440Hz @-18", "crest @0", "IO gain -30->0", "dyn range @0"]

KNOWN_LIMITS = [
    "Continuous Gain is one standard NAM driven only by the player's Input gain; it approximates the amplifier's gain range and does not reproduce a physical knob position exactly.",
    "Sounds between the training anchors are interpolations the model learned, not captures.",
    "The model cannot separate Input gain from playing intensity: soft playing at a high Input gain is not the same as a lower gain played harder.",
    "The recommended Output gain is a constant (the training peak-ceiling scale), not hidden DSP.",
    "Validation figures are measurements against the real captures on held-out DIs; no perceptual quality score is produced, and listening remains the final acceptance test.",
]


def _job_start(project_id: str, kind: str, fn) -> dict:
    with _JOB_LOCK:
        for j in _JOBS.values():
            if j["project_id"] == project_id and j["state"] == "running":
                raise CgProjectError(f"a {j['kind']} job is already running for this project")
        jid = uuid.uuid4().hex[:12]
        job = {"job_id": jid, "project_id": project_id, "kind": kind, "state": "running", "message": "starting", "log": [], "started": time.time(), "result": None, "error": None}
        _JOBS[jid] = job

    def note(msg: str) -> None:
        job["message"] = msg
        job["log"].append(msg)
        del job["log"][:-200]

    def run() -> None:
        try:
            job["result"] = fn(note)
            job["state"] = "done"
            job["message"] = "done"
        except Exception as exc:  # noqa: BLE001 -- surfaced to the UI verbatim
            job["state"] = "error"
            job["error"] = str(exc)
            job["message"] = f"failed: {exc}"

    threading.Thread(target=run, daemon=True, name=f"cg-{kind}").start()
    return job


def register_cg_routes(app, *, cg_dir: Path, a2_output_dir: Path, training_input_path: Path) -> None:
    cg_dir.mkdir(parents=True, exist_ok=True)

    def project_or_404(pid: str) -> CgProject:
        safe = secure_filename(pid)
        p = CgProject(cg_dir / safe)
        if safe != pid or not p.file.is_file():
            raise CgProjectError("project not found")
        return p

    def err(exc: Exception, code: int = 400):
        return jsonify({"error": str(exc)}), (404 if str(exc) == "project not found" else code)

    def model_path_for(design_id: str, manifest: dict) -> tuple[str | None, str | None]:
        """The trained model of a design: the local trainer records it in the manifest, the Kaggle backend on its job."""
        local = (manifest.get("training") or {}).get("output_nam_path")
        if local and Path(local).is_file():
            return local, "local"
        try:
            job = find_active_job(a2_output_dir, design_id)
        except (OSError, ValueError, json.JSONDecodeError):
            job = None
        if job is not None and job.state == "complete" and job.output_nam_path and Path(job.output_nam_path).is_file():
            return job.output_nam_path, "kaggle"
        return None, None

    def training_record(st: dict) -> dict | None:
        b = st.get("bundle")
        if not b:
            return None
        mp = a2_output_dir / b["design_id"] / "training_manifest.json"
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"design_id": b["design_id"], "manifest_missing": True}
        t = m.get("training") or {}
        path, backend = model_path_for(b["design_id"], m)
        epochs, preset = t.get("epochs"), t.get("epoch_preset")
        if backend == "kaggle":       # the cloud worker records what it actually trained with on the job, not in the manifest
            try:
                tr = (find_active_job(a2_output_dir, b["design_id"]).training_result) or {}
                epochs, preset = tr.get("epochs", epochs), tr.get("epoch_preset", preset)
            except (AttributeError, OSError, ValueError, json.JSONDecodeError):
                pass
        return {"design_id": b["design_id"], "output_nam_path": path, "backend": backend, "epochs": epochs, "epoch_preset": preset,
                "quick_mode": t.get("quick_mode"), "full_metrics_vs_target": t.get("full_metrics_vs_target"),
                "trained": path is not None, "core": m.get("core"), "receptive_field_check": m.get("receptive_field_check")}

    def public_state(p: CgProject) -> dict:
        st = p.state()
        an = p.analysis()
        summary = None
        if an:
            sel, prof, aud = an["selection"], an["profile"], an["audit"]
            summary = {
                "made": an["made"], "k_star": sel["k_star_all_within_tolerance"], "eligible": sel["eligible"], "ineligible": sel["ineligible"],
                "exhaustive": sel["exhaustive"], "candidate_positions": sel["candidate_positions"],
                "by_k": {k: {"best": v["best"], "J": v["J"], "all_within_tolerance": bool(v["phys"]) and all(x["max"] <= x["tol"] for x in v["phys"].values() if x["tol"] is not None)} for k, v in sel["by_k"].items()},
                "greedy_order": sel["greedy_order"], "response_coordinate": sel["response_coordinate"],
                "audit": {k: {"status": v["status"], "evidence": v["evidence"], "correction": v["correction"]} for k, v in aud["captures"].items()},
                "audit_set": aud["set"], "capture_files": an["capture_files"],
                "profile": {"gains": prof["gains"], "regions": prof["regions"], "quarantine": prof["quarantine"],
                            "series": {n: {k: prof["series"][n][k] for k in ("dimension", "unit", "values", "reliable")} for n in _PROFILE_SERIES_FOR_UI if n in prof["series"]}},
            }
        val = json.loads(p.validation_file.read_text(encoding="utf-8")) if p.validation_file.is_file() else None
        return {"project": st, "check": p.check_captures(), "analysis": summary, "plan": st.get("plan"), "bundle": st.get("bundle"),
                "training": training_record(st), "validation": val, "epoch_presets": A2_EPOCH_PRESETS, "selection_modes": list(SELECTION_MODES),
                "anchor_methods": list(ANCHOR_METHODS), "held_out_dis": list(HELD_OUT_DIS)}

    @app.get("/api/cg/projects")
    def api_cg_list():
        out = []
        for f in sorted(cg_dir.glob("cg-*/project.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                st = json.loads(f.read_text(encoding="utf-8"))
                out.append({"id": st["id"], "name": st["name"], "amp": st.get("amp"), "channel": st.get("channel"), "captures": len(st["captures"]),
                            "updated": st.get("updated"), "analysed": bool(st.get("analysis")), "planned": bool(st.get("plan")), "bundle": bool(st.get("bundle"))})
            except (OSError, ValueError, KeyError):
                continue
        return jsonify(out)

    @app.post("/api/cg/projects")
    def api_cg_create():
        d = request.get_json(force=True, silent=True) or {}
        try:
            p = CgProject.create(cg_dir, str(d.get("name") or ""), str(d.get("amp") or ""), str(d.get("channel") or ""), str(d.get("notes") or ""))
        except CgProjectError as exc:
            return err(exc)
        return jsonify(public_state(p)), 201

    @app.get("/api/cg/projects/<pid>")
    def api_cg_get(pid):
        try:
            return jsonify(public_state(project_or_404(pid)))
        except CgProjectError as exc:
            return err(exc)

    @app.patch("/api/cg/projects/<pid>")
    def api_cg_meta(pid):
        try:
            p = project_or_404(pid)
            d = request.get_json(force=True, silent=True) or {}
            p.update_meta(**{k: d.get(k) for k in ("name", "amp", "channel", "notes")})
            return jsonify(public_state(p))
        except CgProjectError as exc:
            return err(exc)

    @app.delete("/api/cg/projects/<pid>")
    def api_cg_delete(pid):
        try:
            p = project_or_404(pid)
        except CgProjectError as exc:
            return err(exc)
        with _JOB_LOCK:
            if any(j["project_id"] == p.root.name and j["state"] == "running" for j in _JOBS.values()):
                return jsonify({"error": "a job is still running for this project"}), 409
        shutil.rmtree(p.root)
        return jsonify({"deleted": pid})

    @app.post("/api/cg/projects/<pid>/captures")
    def api_cg_add_captures(pid):
        try:
            p = project_or_404(pid)
            added, problems = [], []
            for f in request.files.getlist("files"):
                try:
                    p.add_capture(f.filename or "", f.read())
                    added.append(Path(f.filename).name)
                except CgProjectError as exc:
                    problems.append(str(exc))
            st = public_state(p)
            st["added"], st["problems"] = added, problems
            if not added:
                st["error"] = "; ".join(problems) or "no files were uploaded"
            return jsonify(st), (201 if added else 400)
        except CgProjectError as exc:
            return err(exc)

    @app.delete("/api/cg/projects/<pid>/captures/<path:filename>")
    def api_cg_remove_capture(pid, filename):
        try:
            p = project_or_404(pid)
            p.remove_capture(filename)
            return jsonify(public_state(p))
        except CgProjectError as exc:
            return err(exc)

    @app.patch("/api/cg/projects/<pid>/positions")
    def api_cg_positions(pid):
        try:
            p = project_or_404(pid)
            pos = (request.get_json(force=True, silent=True) or {}).get("positions") or {}
            p.set_positions({k: (None if v in (None, "") else float(v)) for k, v in pos.items()})
            return jsonify(public_state(p))
        except (CgProjectError, ValueError, TypeError) as exc:
            return err(exc)

    @app.post("/api/cg/projects/<pid>/analyse")
    def api_cg_analyse(pid):
        try:
            p = project_or_404(pid)
            job = _job_start(p.root.name, "analyse", lambda note: p.analyse(progress=note))
        except CgProjectError as exc:
            return err(exc, 409)
        return jsonify({"job_id": job["job_id"]}), 202

    @app.post("/api/cg/projects/<pid>/plan")
    def api_cg_plan(pid):
        try:
            p = project_or_404(pid)
            d = request.get_json(force=True, silent=True) or {}
            custom = [float(x) for x in d["custom"]] if d.get("custom") else None
            p.plan(str(d.get("mode") or "automatic"), custom, str(d.get("anchors") or "fc"))
            return jsonify(public_state(p))
        except (CgProjectError, ValueError, TypeError) as exc:
            return err(exc)

    @app.post("/api/cg/projects/<pid>/generate")
    def api_cg_generate(pid):
        try:
            p = project_or_404(pid)
            d = request.get_json(force=True, silent=True) or {}
            if not training_input_path.is_file():
                return jsonify({"error": "the official NAM training input is not available (see the Builder's training input status)"}), 409
            name = str(d.get("model_name") or "").strip() or None
            if name and len(name) > 100:
                return jsonify({"error": "model name must be 100 characters or fewer"}), 400
            job = _job_start(p.root.name, "generate", lambda note: p.generate_bundle(a2_output_dir, training_input_path, name, progress=note))
        except CgProjectError as exc:
            return err(exc, 409)
        return jsonify({"job_id": job["job_id"]}), 202

    @app.get("/api/cg/jobs/<jid>")
    def api_cg_job(jid):
        j = _JOBS.get(jid)
        if not j:
            return jsonify({"error": "unknown job"}), 404
        return jsonify({k: j[k] for k in ("job_id", "project_id", "kind", "state", "message", "error", "result")} | {"log": j["log"][-40:], "elapsed": time.time() - j["started"]})

    def trained_model(p: CgProject) -> tuple[dict, Path, dict]:
        st = p.state()
        b = st.get("bundle")
        if not b:
            raise CgProjectError("generate the training files first")
        manifest = json.loads((a2_output_dir / b["design_id"] / "training_manifest.json").read_text(encoding="utf-8"))
        path, _backend = model_path_for(b["design_id"], manifest)
        if not path:
            try:
                job = find_active_job(a2_output_dir, b["design_id"])
            except (OSError, ValueError, json.JSONDecodeError):
                job = None
            detail = f" (the Kaggle job is '{job.state}': wait until it is complete and its download has been validated)" if job is not None and job.state != "complete" else ""
            raise CgProjectError("this design has not produced a trained model yet" + detail)
        return st, Path(path), manifest

    def run_validation(p: CgProject, note) -> dict:
        st, nam_path, manifest = trained_model(p)
        an = p.analysis()
        plan = manifest["design"]["input_gain_mapping"]
        training = set(manifest["design"]["positions"])
        c = float(manifest["target"]["output_scale_c"])
        model = load_nam(nam_path)
        paths = p.capture_paths()
        caps = {row["position"]: load_nam(paths[row["position"]]) for row in plan}
        shifts = {row["position"]: alignment_shift(an["audit"]["captures"][f"{row['position']:g}"]) for row in plan}
        import soundfile as sf
        vin, _ = sf.read(str(a2_output_dir / st["bundle"]["design_id"] / "input.wav"), dtype="float32")
        vstop = manifest["training_input"]["train_stop_samples"]
        note("checking standard NAM compatibility")
        compat = check_compatibility(nam_path, vin[vstop:vstop + 20 * SR])
        note("checking output safety over the Input-gain range")
        x = load_reference_di(HELD_OUT_DIS[0])[: 12 * SR]
        safety = check_safety(model, c, x)
        note("comparing with the original captures")
        prog = check_progression(model, c, caps, shifts, plan, training, progress=note)
        note("rendering the audition sweep")
        aud = write_audition(p.root / "audition", model, c, caps, shifts, plan)
        report = {"made": time.time(), "design_id": st["bundle"]["design_id"], "model": {"path": str(nam_path), "sha256": __import__("hashlib").sha256(nam_path.read_bytes()).hexdigest()},
                  "compatibility": compat, "safety": safety, "progression": prog, "coverage": (manifest["design"].get("selection") or {}).get("coverage"),
                  "audition": aud, "blocks_export": False,
                  "statement": "These are measurements. Export is never gated on them or on listening."}
        p.validation_file.write_text(json.dumps(report, default=float), encoding="utf-8")
        return {"ok": True}

    @app.post("/api/cg/projects/<pid>/validate")
    def api_cg_validate(pid):
        try:
            p = project_or_404(pid)
            trained_model(p)
            job = _job_start(p.root.name, "validate", lambda note: run_validation(p, note))
        except (CgProjectError, OSError, json.JSONDecodeError) as exc:
            return err(exc, 409)
        return jsonify({"job_id": job["job_id"]}), 202

    @app.get("/api/cg/projects/<pid>/audition/<path:filename>")
    def api_cg_audition(pid, filename):
        try:
            p = project_or_404(pid)
        except CgProjectError as exc:
            return err(exc)
        safe = secure_filename(filename)
        f = p.root / "audition" / safe
        if safe != filename or not f.is_file() or f.suffix != ".wav":
            return jsonify({"error": "not found"}), 404
        return send_file(f, mimetype="audio/wav")

    @app.get("/api/cg/projects/<pid>/export")
    def api_cg_export(pid):
        """One ordinary .nam plus the human-readable/JSON guide. Never gated on validation or listening."""
        try:
            p = project_or_404(pid)
            st, nam_path, manifest = trained_model(p)
        except (CgProjectError, OSError, json.JSONDecodeError) as exc:
            return err(exc, 409)
        val = json.loads(p.validation_file.read_text(encoding="utf-8")) if p.validation_file.is_file() else None
        design, core = manifest["design"], manifest["core"]
        t = training_record(st) or {}
        mapping = design["input_gain_mapping"]
        rec_out = float(design.get("output_gain_recommendation_db") or 0.0)
        usable = [min(m["input_gain_db"] for m in mapping), max(m["input_gain_db"] for m in mapping)]
        stem = manifest.get("artifact_stem") or "continuous_gain"
        meta = {
            "kind": "continuous_gain", "amplifier": st["amp"], "channel": st["channel"], "model_name": manifest.get("model_name"),
            "model_file": f"{stem}.nam", "model_sha256": __import__("hashlib").sha256(nam_path.read_bytes()).hexdigest(),
            "selected_captures": manifest["sources"], "selection": design.get("selection"), "anchor_method": design["anchor_method"],
            "input_gain_mapping": mapping, "usable_input_gain_range_db": usable, "recommended_constant_output_gain_db": rec_out,
            "training": {"epochs": t.get("epochs"), "epoch_preset": t.get("epoch_preset"), "recipe": design["recipe"], "input_audio_sha256": core["input_audio_sha256"],
                         "target_audio_sha256": core["target_audio_sha256"], "trainer": "existing NAM Mixer A2 trainer (local or Kaggle)"},
            "validation": ({"made": val["made"], "standard_nam": val["compatibility"]["standard_nam"], "reversals": val["progression"]["reversals"],
                            "summary": val["progression"]["summary"]} if val else "not run"),
            "known_limits": KNOWN_LIMITS,
            "note": "This file is a guide and provenance record. The .nam works in an ordinary NAM player without it; nothing here is runtime processing.",
        }
        rows = "\n".join(f"| {m['position']:g} | {m['input_gain_db']:+.1f} dB | {'training anchor' if m['kind'] == 'training_anchor' else 'interpolated'} |" for m in mapping)
        guide = (f"# {manifest.get('model_name')} - player guide\n\nLoad `{stem}.nam` in any NAM player. Set **Output gain** to **{rec_out:+.1f} dB** and keep it constant; "
                 f"move only **Input gain** (usable range {usable[0]:+.1f} to {usable[1]:+.1f} dB).\n\n| Original knob position | Input gain | Kind |\n|---|---|---|\n{rows}\n\n"
                 "Positions marked interpolated were not training captures; the model learned them from their neighbours.\n\n## Known limits\n" + "\n".join(f"- {k}" for k in KNOWN_LIMITS) + "\n")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(nam_path, f"{stem}.nam")
            z.writestr(f"{stem}.continuous_gain.json", json.dumps(meta, indent=2, default=float))
            z.writestr("PLAYER_GUIDE.md", guide)
            z.writestr("training_manifest.json", json.dumps({k: v for k, v in manifest.items() if k != "segments"}, indent=2, default=float))
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{stem}-continuous-gain.zip")

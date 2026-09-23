"""Continuous Gain project: file-backed state and orchestration for the four-stage workflow
(Add captures -> Analyse & select -> Train -> Test & export). Pure library, no Flask.

A project is ONE amplifier/channel and its fixed-gain captures. Layout (under `work/cg_projects/<id>/`):

  project.json    name, amp/channel, captures {filename: {position, sha256, size}}, plan, bundle
  captures/       the user's uploaded .nam files (never modified)
  analysis.json   audit + profile + selection analysis (+ the probe bank), written by `analyse`
  validation.json Stage-4 report (`hybrid.continuous_gain.validation`)

Only measurements that already exist are used (cg_probe/cg_audit/cg_profile/cg_selection); no perceptual score.
Training is NOT run here: `generate_bundle` writes an ordinary A2 bundle (input.wav / hybrid_target.wav /
training_manifest.json, mode "continuous_gain") which the existing local and Kaggle trainers consume.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from .anchors import REFERENCE_DB, fixed_ladder_anchors, mapping_table, response_anchors
from .audit import alignment_shift, audit_captures
from .bundle import (FC_RECIPE, build_training_audio, frozen_design_record, make_chain, receptive_field_record,
                        source_records, write_bundle)
from .parallel import pmap
from .probe import PROBE_VERSION, SR, load_reference_di, probe_capture
from .profile import build_profile
from .selection import resolve_selection, select_captures
from ..core.cab_ir import CabDesign
from ..core.envelope import bounded_envelope_max_history_samples
from ..core.nam_loader import load_nam
from ..core.render import render
from ..modes.training_target import embedded_final_scalar

ANCHOR_METHODS = ("fc", "fixed")           # "fc" = production default (frozen FC recipe); "fixed" = Advanced v3 ladder
SELECTION_MODES = ("automatic", "use_all", "custom")
_POSITION_RE = re.compile(r"(?:gain|ga|g|volume|vol|v|drive)[\s_\-]*(\d+(?:\.\d+)?)", re.IGNORECASE)


class CgProjectError(ValueError):
    pass


def suggest_position(filename: str) -> float | None:
    """A SUGGESTION only (the user confirms or edits): the number after gain/g/vol/v/drive in the file name."""
    m = _POSITION_RE.findall(Path(filename).stem)
    return float(m[-1]) if m else None


def _now() -> float:
    return time.time()


class CgProject:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.file = self.root / "project.json"
        self.captures_dir = self.root / "captures"
        self.analysis_file = self.root / "analysis.json"
        self.validation_file = self.root / "validation.json"

    # ---- lifecycle
    @classmethod
    def create(cls, base: Path, name: str, amp: str = "", channel: str = "", notes: str = "") -> "CgProject":
        name = (name or "").strip()
        if not name:
            raise CgProjectError("project name is required")
        pid = f"cg-{uuid.uuid4().hex[:10]}"
        p = cls(Path(base) / pid)
        p.captures_dir.mkdir(parents=True)
        p._write({"id": pid, "name": name, "amp": amp.strip(), "channel": channel.strip(), "notes": notes.strip(),
                  "created": _now(), "updated": _now(), "captures": {}, "plan": None, "bundle": None, "analysis": None})
        return p

    def state(self) -> dict:
        return json.loads(self.file.read_text(encoding="utf-8"))

    def _write(self, st: dict) -> None:
        st["updated"] = _now()
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2, default=float) + "\n", encoding="utf-8")
        tmp.replace(self.file)

    def update_meta(self, **kw) -> dict:
        st = self.state()
        for k in ("name", "amp", "channel", "notes"):
            if k in kw and kw[k] is not None:
                st[k] = str(kw[k]).strip()
        self._write(st)
        return st

    # ---- Stage 1: captures
    def add_capture(self, filename: str, data: bytes) -> dict:
        safe = Path(filename).name
        if not safe.lower().endswith(".nam"):
            raise CgProjectError(f"{safe}: only .nam files can be added")
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CgProjectError(f"{safe} is not a readable .nam file: {exc}") from exc
        if not isinstance(raw, dict) or "architecture" not in raw:
            raise CgProjectError(f"{safe} does not look like a NAM model (no architecture)")
        st = self.state()
        if safe in st["captures"]:
            raise CgProjectError(f"{safe} was already added")
        (self.captures_dir / safe).write_bytes(data)
        import hashlib
        st["captures"][safe] = {"position": suggest_position(safe), "position_suggested": suggest_position(safe) is not None,
                                "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                                "sample_rate": raw.get("sample_rate")}
        st["analysis"] = None       # any earlier analysis is stale
        st["plan"] = None
        self._write(st)
        return st["captures"][safe]

    def remove_capture(self, filename: str) -> None:
        st = self.state()
        if filename not in st["captures"]:
            raise CgProjectError(f"unknown capture {filename!r}")
        (self.captures_dir / Path(filename).name).unlink(missing_ok=True)
        del st["captures"][filename]
        st["analysis"], st["plan"] = None, None
        self._write(st)

    def set_positions(self, positions: dict[str, float | None]) -> dict:
        st = self.state()
        for fn, pos in positions.items():
            if fn not in st["captures"]:
                raise CgProjectError(f"unknown capture {fn!r}")
            st["captures"][fn]["position"] = None if pos is None else float(pos)
            st["captures"][fn]["position_suggested"] = False
        st["analysis"], st["plan"] = None, None
        self._write(st)
        return self.check_captures()

    def check_captures(self) -> dict:
        """Stage-1 consistency summary; `ready` is False while blocking issues remain."""
        st = self.state()
        caps = st["captures"]
        issues: list[dict] = []
        missing = [fn for fn, c in caps.items() if c["position"] is None]
        for fn in missing:
            issues.append({"level": "blocking", "file": fn, "message": "no physical gain position assigned"})
        seen: dict[float, str] = {}
        for fn, c in sorted(caps.items(), key=lambda kv: (kv[1]["position"] is None, kv[1]["position"] or 0)):
            p = c["position"]
            if p is None:
                continue
            if p in seen:
                issues.append({"level": "blocking", "file": fn, "message": f"duplicate position {p:g} (also {seen[p]})"})
            seen[p] = fn
            if c.get("sample_rate") not in (None, SR):
                issues.append({"level": "blocking", "file": fn, "message": f"sample rate {c['sample_rate']} Hz (48000 Hz required; resampling is never done silently)"})
        if len(caps) < 3:
            issues.append({"level": "blocking", "file": None, "message": "at least 3 captures are needed"})
        ps = sorted(seen)
        summary = f"{len(caps)} captures" + (f"; positions {ps[0]:g}-{ps[-1]:g}" if ps else "") + (f"; {len(missing)} need a position" if missing else "")
        return {"summary": summary, "issues": issues, "ready": not any(i["level"] == "blocking" for i in issues), "count": len(caps)}

    def ordered(self) -> list[tuple[float, str]]:
        return sorted((c["position"], fn) for fn, c in self.state()["captures"].items() if c["position"] is not None)

    def capture_paths(self) -> dict[float, Path]:
        return {p: self.captures_dir / fn for p, fn in self.ordered()}

    # ---- Stage 2: analysis
    def analyse(self, progress: Callable[[str], None] | None = None) -> dict:
        note = progress or (lambda _m: None)
        chk = self.check_captures()
        if not chk["ready"]:
            raise CgProjectError("fix the blocking capture issues first: " + "; ".join(i["message"] for i in chk["issues"] if i["level"] == "blocking"))
        paths = self.capture_paths()
        models = {p: load_nam(f) for p, f in paths.items()}
        renderers = {p: (lambda x, m=m: render(m, x, SR)) for p, m in models.items()}
        st = self.state()
        sha = {p: st["captures"][fn]["sha256"] for p, fn in self.ordered()}
        cache = self._probe_cache()
        todo = [p for p in sorted(paths) if sha[p] not in cache]
        note(f"probing {len(todo)} capture(s)" + (f" ({len(paths) - len(todo)} unchanged, reused)" if len(todo) < len(paths) else ""))

        def probe_one(p: float) -> dict:
            pr = probe_capture(renderers[p], load_reference_di)
            note(f"probed capture {p:g}")
            return pr

        for p, pr in zip(todo, pmap(probe_one, todo)):      # independent native renders, in parallel; results in order
            cache[sha[p]] = pr
        self._save_probe_cache(cache)
        probes, meta = {}, {}
        for p in sorted(paths):
            probes[p] = cache[sha[p]]
            md = models[p].raw.get("metadata") or {}
            loud = md.get("loudness")
            meta[p] = {"loudness": loud if isinstance(loud, (int, float)) else None, "gear_make": md.get("gear_make"), "gear_model": md.get("gear_model"),
                       "sample_rate": models[p].raw.get("sample_rate"), "input_level_dbu": models[p].input_level_dbu}
        note("auditing captures")
        audit = audit_captures(sorted(paths), probes, meta, lambda g, x: render(models[g], x, SR), load_reference_di)
        note("building response profile")
        profile = build_profile(probes, audit)
        note("searching capture subsets")
        selection = select_captures(profile, audit)
        result = {"audit": audit, "profile": profile, "selection": selection, "probes": {f"{p:g}": pr for p, pr in probes.items()},
                  "capture_files": {f"{p:g}": fn for p, fn in self.ordered()}, "made": _now()}
        self.analysis_file.write_text(json.dumps(result, default=float), encoding="utf-8")
        st = self.state()
        st["analysis"] = {"made": result["made"], "k_star": selection["k_star_all_within_tolerance"], "eligible": selection["eligible"],
                          "statuses": {k: v["status"] for k, v in audit["captures"].items()}}
        st["plan"] = None
        self._write(st)
        return st["analysis"]

    def _probe_cache(self) -> dict:
        """Probes are a pure function of the capture file and the probe bank, so they are kept by file hash: re-analysing after adding or
        removing a capture only probes what is new."""
        f = self.root / "probe_cache.json"
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            return d["probes"] if d.get("version") == PROBE_VERSION else {}
        except (OSError, ValueError, KeyError):
            return {}

    def _save_probe_cache(self, cache: dict) -> None:
        (self.root / "probe_cache.json").write_text(json.dumps({"version": PROBE_VERSION, "probes": cache}, default=float), encoding="utf-8")

    def analysis(self) -> dict | None:
        return json.loads(self.analysis_file.read_text(encoding="utf-8")) if self.analysis_file.is_file() else None

    # ---- planning (selection + anchors + mapping)
    def plan(self, mode: str = "automatic", custom: list[float] | None = None, anchors: str = "fc") -> dict:
        if mode not in SELECTION_MODES:
            raise CgProjectError(f"selection mode must be one of {SELECTION_MODES}")
        if anchors not in ANCHOR_METHODS:
            raise CgProjectError(f"anchor method must be one of {ANCHOR_METHODS}")
        an = self.analysis()
        if an is None:
            raise CgProjectError("analyse the captures first")
        try:
            sel = resolve_selection(an["selection"], an["profile"], an["audit"], mode, custom)
        except ValueError as exc:
            raise CgProjectError(str(exc)) from exc
        rc = an["selection"]["response_coordinate"]
        try:
            pos, anc = (response_anchors(rc, sel["selected"]) if anchors == "fc" else fixed_ladder_anchors(sel["selected"]))
        except ValueError as exc:
            raise CgProjectError(str(exc)) from exc
        warnings = list(sel["notes"])
        if anchors == "fixed":
            warnings.append("Advanced: fixed 4 dB-spacing anchors (v3). The production default is the FC response-distance anchors; this alternative is not a substitute for them.")
            if anc[-1] > 14.0 or anc[0] < -22.0:
                warnings.append(f"fixed-ladder anchors span {anc[0]:g} to {anc[-1]:g} dB, outside the plugin-compatible -20..+14 dB range")
        if anchors == "fc" and len(pos) > 9:
            warnings.append(f"{len(pos)} captures do not fit at 4 dB spacing inside -20..+14 dB; the anchor separation is reduced to fit")
        positions_all = [p for p, _ in self.ordered()]
        mapping = mapping_table(rc, pos, anc, positions_all) if anchors == "fc" else \
            [{"position": p, "input_gain_db": (dict(zip(pos, anc)).get(p, -22.0 + 4.0 * (p - 1.0))), "kind": "training_anchor" if p in pos else "fixed_rule"} for p in positions_all]
        plan = {"mode": mode, "custom": custom, "anchor_method": anchors, "selected": pos, "anchors_input_gain_db": anc,
                "levels_db": [a + REFERENCE_DB for a in anc], "mapping": mapping, "coverage": sel["coverage"], "fallback": sel["fallback"],
                "k_star": sel["k_star"], "warnings": warnings, "planned": _now()}
        st = self.state()
        st["plan"] = plan
        self._write(st)
        return plan

    # ---- Stage 3: freeze + generate bundle for the existing trainers
    def generate_bundle(self, out_root: Path, official_input_path: Path, model_name: str | None = None,
                        progress: Callable[[str], None] | None = None, *, recipe=None, load_di: Callable[[str], np.ndarray] | None = None,
                        official_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                        excitation: str = "FC recipe: official input + guitar DIs at level offsets",
                        cab: CabDesign | None = None) -> dict:
        """`recipe` / `load_di` / `official_transform` exist for training-material experiments (e.g. official input only, or one synthetic
        level-swept excitation); the defaults are the frozen FC recipe."""
        import soundfile as sf

        st = self.state()
        plan = st.get("plan")
        if not plan:
            raise CgProjectError("review the training plan first")
        an = self.analysis()
        if an is None:
            raise CgProjectError("analysis is missing; analyse the captures again")
        note = progress or (lambda _m: None)
        positions, anchors = plan["selected"], plan["anchors_input_gain_db"]
        paths = self.capture_paths()
        models = {p: load_nam(paths[p]) for p in positions}
        chain = make_chain(positions, anchors)
        shifts = {p: alignment_shift(an["audit"]["captures"][f"{p:g}"]) for p in positions}
        official, sr = sf.read(str(official_input_path), dtype="float32")
        if sr != SR:
            raise CgProjectError(f"official training input must be {SR} Hz")
        official = official if official.ndim == 1 else official.mean(axis=1)
        recipe = recipe or FC_RECIPE
        if official_transform is not None:
            official = official_transform(official).astype(np.float32)
        built = build_training_audio(chain, lambda g, x: render(models[g], x, SR), shifts, official, load_di or load_reference_di, recipe, progress=note)
        name = (model_name or st["name"]).strip()
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "continuous_gain"
        design_id = f"{st['id']}-{int(_now())}"
        out_dir = Path(out_root) / design_id
        design = frozen_design_record(positions, anchors, recipe, plan["anchor_method"],
                                      {"mode": plan["mode"], "k_star": plan["k_star"], "fallback": plan["fallback"], "coverage": plan["coverage"], "warnings": plan["warnings"]},
                                      plan["mapping"])
        design.update({"project_id": st["id"], "amp": st["amp"], "channel": st["channel"], "output_gain_recommendation_db": built.reduction_db, "training_material": excitation})
        srcs = source_records(positions, paths, anchors, chain, an["audit"])
        rf = receptive_field_record(positions, models, bounded_envelope_max_history_samples(SR), cab=cab)
        output_gain = {"mode": "continuous_gain_scale", "applied_gain_db": -built.reduction_db}
        if cab is not None:
            _, output_gain["embedded_final"] = embedded_final_scalar(built.target, cab, SR, recipe.ceiling_dbfs)
        mp = write_bundle(out_dir, built, chain, anchors, shifts, sources=srcs, model_name=name, artifact_stem=stem, design=design, receptive_field=rf,
                          cab=cab, output_gain=output_gain,
                          extra={"project": {"id": st["id"], "name": st["name"]},
                                 "capture_audit": {k: {"status": v["status"], "correction": v["correction"]} for k, v in an["audit"]["captures"].items()}})
        st = self.state()
        st["bundle"] = {"design_id": design_id, "manifest": str(mp), "made": _now(), "plan": plan,
                        "cab": cab.to_dict() if cab else None}
        self._write(st)
        return st["bundle"]

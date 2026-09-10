#!/usr/bin/env python3
"""Opt-in native NAMCore regression harness; never trains or uploads assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hybrid.render as render_module
from hybrid.blend import DEFAULT_TRANSITION_WIDTH_DB
from hybrid.character_blend import CharacterBlendDesign, build_character_blend, freeze_character_design
from hybrid.cab_ir import apply_cab_ir, get_prepared_cab_ir
from hybrid.design import freeze_design
from hybrid.fixed_blend import build_fixed_blend, freeze_blend_design
from hybrid.nam_loader import load_nam
from hybrid.pipeline import build_hybrid, render_pair
from hybrid.render import NamRenderError, find_nam_render_exe, render
from hybrid.validation import compute_esr_metrics, render_reference_hybrid

REPORT_SCHEMA_VERSION = 2


class PrerequisiteUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_di(path: Path, max_seconds: float) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.ndim != 1 or not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError(f"DI must be non-empty, mono-compatible, and finite: {path}")
    if max_seconds > 0:
        audio = audio[:max(1, int(round(sample_rate * max_seconds)))]
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _configure_renderer(explicit: Path | None) -> tuple[Path, dict]:
    if explicit is not None:
        if not explicit.is_file():
            raise PrerequisiteUnavailable(f"explicit renderer does not exist: {explicit}")
        render_module._NAM_RENDER_EXE_CANDIDATES = (explicit.resolve(),)
    try:
        executable = find_nam_render_exe().resolve()
    except NamRenderError as exc:
        raise PrerequisiteUnavailable(str(exc)) from exc
    probe = subprocess.run([str(executable), "--help"], capture_output=True, text=True, timeout=5)
    usage = (probe.stderr or probe.stdout or "").strip()
    if probe.returncode != 0 and "usage:" not in usage.lower():
        raise PrerequisiteUnavailable(f"renderer found but not verified: {usage or probe.returncode}")
    return executable, {
        "path": str(executable), "sha256": _sha256(executable),
        "size_bytes": executable.stat().st_size, "mtime_ns": executable.stat().st_mtime_ns,
        "probe_returncode": probe.returncode, "probe_output": usage[:500],
    }


def _audio_check(audio: np.ndarray, frames: int) -> dict:
    finite = bool(np.all(np.isfinite(audio)))
    peak = float(np.max(np.abs(audio))) if len(audio) and finite else None
    valid = len(audio) == frames and finite and peak is not None and peak >= 1e-7
    return {"frame_count": len(audio), "length_ok": len(audio) == frames, "finite": finite,
            "peak": peak, "silent": peak is None or peak < 1e-7, "valid": valid}


def _git_identity() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
    return {"commit": head.stdout.strip() if head.returncode == 0 else None, "dirty": dirty}


def run(
    amp_a_path: Path, amp_b_path: Path, di_paths: list[Path], out_dir: Path,
    cab_path: Path | None = None, *, renderer_path: Path | None = None,
    levels_db: tuple[float, ...] = (-12.0, 0.0, 12.0), max_seconds: float = 6.0,
    export_paths: tuple[Path, ...] = (), export_mode: str | None = None,
) -> dict:
    assets = [amp_a_path, amp_b_path, *di_paths, *([cab_path] if cab_path else []), *export_paths]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise PrerequisiteUnavailable("missing required assets: " + ", ".join(missing))
    if export_paths and export_mode not in ("hybrid", "blend", "character"):
        raise PrerequisiteUnavailable("--export-mode is required with --export-nam")
    _executable, renderer = _configure_renderer(renderer_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    amp_a, amp_b = load_nam(amp_a_path), load_nam(amp_b_path)
    calibration_modes = ["raw"]
    unavailable = []
    if amp_a.input_level_dbu is not None and amp_b.input_level_dbu is not None:
        calibration_modes.append("auto")
    else:
        unavailable.append({"combination": "calibration=auto",
                            "reason": "both source NAMs must record input_level_dbu; raw was exercised"})
    report = {
        "schema_version": REPORT_SCHEMA_VERSION, "state": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "git": _git_identity(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "renderer": renderer,
        "source_hashes": {"amp_a": _sha256(amp_a_path), "amp_b": _sha256(amp_b_path)},
        "source_metadata": {"amp_a": amp_a.summary(), "amp_b": amp_b.summary(),
                            "cab_sha256": _sha256(cab_path) if cab_path else None},
        "settings": {"levels_db": list(levels_db), "calibration_modes": calibration_modes,
                     "max_seconds": max_seconds, "alignment_enabled": False,
                     "hybrid": {"crossover_dbfs": -20.0, "transition_width_db": DEFAULT_TRANSITION_WIDTH_DB},
                     "blend": {"mix_b": .5}, "character": {"tone": .5, "feel": .5, "drive": .5}},
        "unavailable_combinations": unavailable, "cases": [], "exports": [],
    }
    baseline = {}
    for di_path in di_paths:
        source_dry, sample_rate = _load_di(di_path, max_seconds)
        for calibration_mode in calibration_modes:
            for level_db in levels_db:
                started = time.monotonic()
                dry = (source_dry * 10.0 ** (level_db / 20.0)).astype(np.float32)
                render_started = time.monotonic()
                pair = render_pair(amp_a, amp_b, dry, sample_rate, calibration_mode=calibration_mode)
                render_seconds = time.monotonic() - render_started
                hybrid = build_hybrid(pair, -20.0, DEFAULT_TRANSITION_WIDTH_DB,
                                      auto_level=False, align_enabled=False)
                blend = build_fixed_blend(pair, .5, auto_level=False, align_enabled=False)
                character_design = CharacterBlendDesign(str(amp_a_path), str(amp_b_path),
                                                        calibration_mode=calibration_mode)
                character = build_character_blend(pair, character_design)
                frozen_hybrid = freeze_design(
                    pair, hybrid, str(amp_a_path), str(amp_b_path), -20.0,
                    DEFAULT_TRANSITION_WIDTH_DB, False, design_di_file=str(di_path),
                    output_gain_mode="manual", manual_output_gain_db=0.0)
                frozen_blend = freeze_blend_design(
                    pair, blend, str(amp_a_path), str(amp_b_path), False,
                    design_di_file=str(di_path), output_gain_mode="manual", manual_output_gain_db=0.0)
                frozen_character = freeze_character_design(
                    pair, character, str(amp_a_path), str(amp_b_path),
                    tone_mix_b=.5, feel_mix_b=.5, drive_mix_b=.5,
                    output_gain_mode="manual", manual_output_gain_db=0.0)
                modes = {"hybrid": hybrid.hybrid, "blend": blend.blend, "character": character.blend}
                invariants = {
                    "hybrid": bool(np.all((hybrid.blend_curve >= 0) & (hybrid.blend_curve <= 1)) and hybrid.alignment_offset_samples == 0),
                    "blend": bool(np.allclose(blend.blend, .5 * pair.amp_a + .5 * pair.amp_b,
                                              rtol=1e-6, atol=1e-7) and blend.alignment_offset_samples == 0),
                    "character": bool(np.all((character.drive_weight_b >= 0) & (character.drive_weight_b <= 1))),
                }
                # One frozen-source native rerender supplies equivalent stems
                # for all modes; calibration/input gain are not double-applied.
                reference_hybrid = render_reference_hybrid(frozen_hybrid, dry, sample_rate)
                reference_pair = SimpleNamespace(dry=dry, amp_a=reference_hybrid.amp_a,
                                                 amp_b=reference_hybrid.amp_b,
                                                 envelope_db=reference_hybrid.envelope_db,
                                                 sample_rate=sample_rate)
                references = {
                    "hybrid": reference_hybrid.hybrid,
                    "blend": build_fixed_blend(reference_pair, frozen_blend.mix_b, auto_level=False,
                                               manual_b_trim_db=frozen_blend.effective_b_trim_db).blend,
                    "character": build_character_blend(reference_pair, frozen_character).blend,
                }
                equivalence = {name: {
                    "max_abs_error": float(np.max(np.abs(audio - references[name]))),
                    "allclose": bool(np.allclose(audio, references[name], rtol=2e-6, atol=2e-7)),
                } for name, audio in modes.items()}
                prepared_cab = get_prepared_cab_ir(cab_path, sample_rate) if cab_path else None
                outputs = dict(modes)
                if prepared_cab:
                    outputs.update({f"{name}_cab": apply_cab_ir(audio, prepared_cab)
                                    for name, audio in modes.items()})
                slug = f"{di_path.stem}_{calibration_mode}_{level_db:+g}dB".replace("+", "plus").replace("-", "minus")
                mode_report = {}
                for name, audio in outputs.items():
                    result = _audio_check(audio, len(dry))
                    result["invariant_ok"] = invariants[name.removesuffix("_cab")]
                    result["artifact"] = f"{slug}_{name}.wav"
                    sf.write(out_dir / result["artifact"], audio.astype(np.float32), sample_rate, subtype="FLOAT")
                    mode_report[name] = result
                deterministic = None
                if calibration_mode == "raw" and level_db == 0.0:
                    repeated = render_pair(amp_a, amp_b, dry, sample_rate, calibration_mode="raw")
                    deterministic = {"amp_a_exact": bool(np.array_equal(pair.amp_a, repeated.amp_a)),
                                     "amp_b_exact": bool(np.array_equal(pair.amp_b, repeated.amp_b)),
                                     "amp_a_max_abs_error": float(np.max(np.abs(pair.amp_a - repeated.amp_a))),
                                     "amp_b_max_abs_error": float(np.max(np.abs(pair.amp_b - repeated.amp_b)))}
                    baseline[(str(di_path), export_mode or "hybrid")] = (dry, sample_rate, modes[export_mode or "hybrid"])
                report["cases"].append({"di": str(di_path), "di_sha256": _sha256(di_path),
                    "sample_rate": sample_rate, "source_frames": len(source_dry), "level_db": level_db,
                    "calibration_requested": calibration_mode, "calibration_applied": pair.calibration_applied,
                    "render_seconds": render_seconds, "elapsed_seconds": time.monotonic() - started,
                    "determinism": deterministic, "frozen_teacher_equivalence": equivalence,
                    "modes": mode_report})
    if export_paths:
        dry, sample_rate, reference = baseline[(str(di_paths[0]), export_mode)]
        for export_path in export_paths:
            model = load_nam(export_path)
            variants = {}
            for label, slim in (("full", 0.0), ("lite", 1.0)):
                try:
                    candidate = render(model, dry, sample_rate, slim=slim)
                    variants[label] = {"audio": _audio_check(candidate, len(dry)),
                                       "metrics": compute_esr_metrics(candidate, reference)}
                    sf.write(out_dir / f"{export_path.stem}_{label}.wav", candidate, sample_rate, subtype="FLOAT")
                except NamRenderError as exc:
                    variants[label] = {"audio": {"valid": False}, "error": str(exc)}
            report["exports"].append({"path": str(export_path), "sha256": _sha256(export_path),
                                      "reference_mode": export_mode, "variants": variants})
    failures = []
    for index, case in enumerate(report["cases"]):
        for name, result in case["modes"].items():
            if not result["valid"] or not result["invariant_ok"]:
                failures.append(f"case {index} mode {name} failed audio/invariant checks")
        if not all(item["allclose"] for item in case["frozen_teacher_equivalence"].values()):
            failures.append(f"case {index} preview/frozen-teacher equivalence failed")
        if case["determinism"] and not (case["determinism"]["amp_a_exact"] and case["determinism"]["amp_b_exact"]):
            failures.append(f"case {index} native render was not deterministic")
    for exported in report["exports"]:
        for label, result in exported["variants"].items():
            if not result["audio"]["valid"]:
                failures.append(f"export {exported['path']} {label} render invalid")
    report["failures"] = failures
    report["state"] = "failed" if failures else "passed"
    report["elapsed_seconds"] = sum(case["elapsed_seconds"] for case in report["cases"])
    (out_dir / "real_render_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp-a", type=Path, required=True)
    parser.add_argument("--amp-b", type=Path, required=True)
    parser.add_argument("--di", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, help="explicit nam_render executable")
    parser.add_argument("--cab", type=Path)
    parser.add_argument("--level-db", type=float, action="append",
                        help="external DI level; repeat (default -12, 0, +12 dB)")
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--export-nam", type=Path, action="append", default=[])
    parser.add_argument("--export-mode", choices=("hybrid", "blend", "character"))
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.amp_a, args.amp_b, args.di, args.out_dir, args.cab,
                     renderer_path=args.renderer,
                     levels_db=tuple(args.level_db) if args.level_db else (-12.0, 0.0, 12.0),
                     max_seconds=args.max_seconds, export_paths=tuple(args.export_nam),
                     export_mode=args.export_mode)
    except PrerequisiteUnavailable as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": REPORT_SCHEMA_VERSION, "state": "unavailable", "reason": str(exc)}
        (args.out_dir / "real_render_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Real render unavailable: {exc}", file=sys.stderr)
        return 2 if args.release else 0
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": REPORT_SCHEMA_VERSION, "state": "failed",
                   "reason": f"{type(exc).__name__}: {exc}"}
        (args.out_dir / "real_render_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Real render failed: {exc}", file=sys.stderr)
        return 1
    print(f"Real render report: {args.out_dir / 'real_render_report.json'}")
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

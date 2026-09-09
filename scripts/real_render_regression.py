#!/usr/bin/env python3
"""Opt-in native NAMCore regression harness for two user-provided models.

It never starts training or uploads assets.  Missing prerequisites are a
recorded skip for normal developer runs and a non-zero failure with
``--release`` so an all-skipped release check cannot look successful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.blend import DEFAULT_TRANSITION_WIDTH_DB
from hybrid.character_blend import CharacterBlendDesign, build_character_blend
from hybrid.cab_ir import apply_cab_ir, get_prepared_cab_ir
from hybrid.fixed_blend import build_fixed_blend
from hybrid.nam_loader import load_nam
from hybrid.pipeline import build_hybrid, render_pair
from hybrid.render import NamRenderError, find_nam_render_exe


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_di(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return (audio[:, 0] if audio.ndim > 1 else audio), sr


def run(amp_a_path: Path, amp_b_path: Path, di_paths: list[Path], out_dir: Path, cab_path: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    amp_a, amp_b = load_nam(amp_a_path), load_nam(amp_b_path)
    report = {"schema_version": 1, "source_hashes": {"amp_a": _sha256(amp_a_path), "amp_b": _sha256(amp_b_path)}, "cab_sha256": _sha256(cab_path) if cab_path else None, "renderer": str(find_nam_render_exe()), "cases": []}
    for di_path in di_paths:
        dry, sr = _load_di(di_path)
        started = time.monotonic()
        pair = render_pair(amp_a, amp_b, dry, sr)
        modes = {
            "hybrid": build_hybrid(pair, crossover_dbfs=-20.0, transition_width_db=DEFAULT_TRANSITION_WIDTH_DB).hybrid,
            "blend": build_fixed_blend(pair, mix_b=.5).blend,
            "character": build_character_blend(pair, CharacterBlendDesign(str(amp_a_path), str(amp_b_path))).blend,
        }
        if cab_path:
            cab = get_prepared_cab_ir(cab_path, sr)
            modes.update({f"{mode}_cab": apply_cab_ir(audio, cab) for mode, audio in list(modes.items())})
        case = {"di": str(di_path), "sha256": _sha256(di_path), "sample_rate": sr, "duration_s": len(dry) / sr, "elapsed_s": time.monotonic() - started, "modes": {}}
        for mode, audio in modes.items():
            valid = len(audio) == len(dry) and bool(np.all(np.isfinite(audio)))
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            case["modes"][mode] = {"valid": valid, "silent": peak < 1e-7, "peak": peak}
            sf.write(out_dir / f"{di_path.stem}_{mode}.wav", audio.astype(np.float32), sr, subtype="FLOAT")
        report["cases"].append(case)
    (out_dir / "real_render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp-a", type=Path, required=True)
    parser.add_argument("--amp-b", type=Path, required=True)
    parser.add_argument("--di", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cab", type=Path, help="optional cabinet IR; exercises cab-on cases alongside dry cases")
    parser.add_argument("--release", action="store_true", help="fail instead of recording a skipped developer run")
    args = parser.parse_args(argv)
    try:
        report = run(args.amp_a, args.amp_b, args.di, args.out_dir, args.cab)
    except (OSError, ValueError, NamRenderError) as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "real_render_report.json").write_text(json.dumps({"schema_version": 1, "state": "blocked", "reason": str(exc)}, indent=2), encoding="utf-8")
        print(f"Real render unavailable: {exc}", file=sys.stderr)
        return 2 if args.release else 0
    print(f"Real render report: {args.out_dir / 'real_render_report.json'}")
    return 0 if all(mode["valid"] and not mode["silent"] for case in report["cases"] for mode in case["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

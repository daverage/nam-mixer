#!/usr/bin/env python3
"""Held-out musical validation of a trained A2 export against the LIVE
reference hybrid -- docs/phase3.md sections 24-28.

Runs entirely on the native NAMCore renderer (`hybrid.render.render`), no
torch/neural-amp-modeler required -- safe to run in the normal app
environment, independently of scripts/train_a2.py's training venv.

Usage:
    python scripts/validate_a2.py <hybrid_design.json> <trained_a2.nam> \
        --di assets/di/moderate_brit.wav --di assets/di/clean_smooth.wav \
        --gain -7.0:vintage_single --gain 0.0:paf --gain 4.5:hot_humbucker \
        --out-dir work/a2/<design_id>/validation

For each (DI, gain label) pair, generates:
    <out>/<di_stem>/<gain_label>/reference_hybrid.wav
    <out>/<di_stem>/<gain_label>/a2_full.wav
    <out>/<di_stem>/<gain_label>/a2_lite.wav
    <out>/<di_stem>/<gain_label>/sequence.wav   (reference, full, lite, with silence between)
and a single validation_report.json with objective metrics for every
combination.

Unlike training-target generation, input-profile gains ARE applied here as
real audio gain (never the deprecated envelope-only `dry_gain_db`) --
that's the whole point of this validation: testing how different real
input levels behave (docs/phase3.md section 25).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hybrid.design import HybridDesign  # noqa: E402
from hybrid.fixed_blend import BlendDesign  # noqa: E402
from hybrid.character_blend import CharacterBlendDesign  # noqa: E402
from hybrid.input_profiles import db_to_amplitude  # noqa: E402
from hybrid.safety import check_audio  # noqa: E402
from hybrid.validation import compute_esr_metrics, render_reference_hybrid, render_reference_blend, render_reference_character, render_trained_a2  # noqa: E402

# A single fixed listening-safety gain applied identically to every file in a
# comparison, if any of them would clip -- never per-file, never a limiter,
# per docs/phase3.md section 27 ("use the SAME fixed gain for all compared
# files and document it").
LISTENING_TARGET_PEAK_DBFS = -3.0
SILENCE_GAP_S = 1.0


def _load_di(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def _parse_gain_arg(value: str) -> tuple[float, str]:
    """'-7.0:vintage_single' -> (-7.0, 'vintage_single')."""
    gain_str, _, label = value.partition(":")
    gain_db = float(gain_str)
    label = label or gain_str.replace(".", "p").replace("-", "neg")
    return gain_db, label


def run_validation(
    design,
    a2_nam_path: Path,
    di_paths: list[Path],
    gains: list[tuple[float, str]],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_teacher = {"hybrid": render_reference_hybrid, "blend": render_reference_blend, "character": render_reference_character}[design.mode]
    report: dict = {"mode": design.mode, "design": str(design.amp_a_path) + " -> " + str(design.amp_b_path), "results": []}

    for di_path in di_paths:
        dry, sr = _load_di(di_path)
        di_stem = di_path.stem
        for gain_db, label in gains:
            gained = (dry * db_to_amplitude(gain_db)).astype(np.float32)

            ref = render_teacher(design, gained, sr)
            full = render_trained_a2(a2_nam_path, gained, sr, slim=0.0)
            try:
                lite = render_trained_a2(a2_nam_path, gained, sr, slim=1.0)
                lite_ok = True
            except Exception as exc:  # noqa: BLE001 -- Lite may not be supported by every export
                print(f"WARNING: Lite render failed for {di_stem}/{label}: {exc}")
                lite = np.zeros_like(ref.hybrid)
                lite_ok = False

            full_metrics = compute_esr_metrics(full, ref.hybrid)
            lite_metrics = compute_esr_metrics(lite, ref.hybrid) if lite_ok else None

            result = {
                "di": di_path.name, "gain_db": gain_db, "profile_label": label,
                "full_vs_reference": full_metrics,
                "lite_vs_reference": lite_metrics,
            }
            report["results"].append(result)
            print(f"{di_stem}/{label} (gain={gain_db:+.1f}dB): "
                  f"Full ESR={full_metrics['raw_esr']:.4f}  "
                  f"Lite ESR={lite_metrics['raw_esr']:.4f}" if lite_ok else
                  f"{di_stem}/{label} (gain={gain_db:+.1f}dB): Full ESR={full_metrics['raw_esr']:.4f}  Lite unavailable")

            case_dir = out_dir / di_stem / label
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_listening_files(case_dir, ref.hybrid, full, lite if lite_ok else None, sr)

    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report: {report_path}")
    return report


def _write_listening_files(case_dir: Path, reference: np.ndarray, full: np.ndarray, lite, sample_rate: int) -> None:
    """Apply ONE shared fixed gain (if any of the three would clip) to all
    three files identically -- never normalize each independently
    (docs/phase3.md section 27)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(reference), len(full), len(lite) if lite is not None else len(full))
    clips = [reference[:n], full[:n]] + ([lite[:n]] if lite is not None else [])
    peak_dbfs = max(check_audio(c).peak_dbfs for c in clips)

    shared_gain_db = 0.0
    if np.isfinite(peak_dbfs) and peak_dbfs > LISTENING_TARGET_PEAK_DBFS:
        shared_gain_db = LISTENING_TARGET_PEAK_DBFS - peak_dbfs
    shared_gain = db_to_amplitude(shared_gain_db)

    names_and_clips = [("reference_hybrid.wav", reference[:n]), ("a2_full.wav", full[:n])]
    if lite is not None:
        names_and_clips.append(("a2_lite.wav", lite[:n]))

    for name, clip in names_and_clips:
        sf.write(case_dir / name, (clip * shared_gain).astype(np.float32), sample_rate, subtype="FLOAT")

    silence = np.zeros(int(SILENCE_GAP_S * sample_rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for i, (_name, clip) in enumerate(names_and_clips):
        if i > 0:
            pieces.append(silence)
        pieces.append(clip * shared_gain)
    sequence = np.concatenate(pieces).astype(np.float32)
    sf.write(case_dir / "sequence.wav", sequence, sample_rate, subtype="FLOAT")

    with open(case_dir / "listening_gain.json", "w", encoding="utf-8") as f:
        json.dump({"shared_gain_db": shared_gain_db, "peak_before_dbfs": peak_dbfs}, f, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path, help="path to hybrid_design.json")
    parser.add_argument("a2_nam", type=Path, help="path to the trained/exported A2 .nam file")
    parser.add_argument("--di", action="append", type=Path, required=True, help="held-out DI wav (repeatable)")
    parser.add_argument("--gain", action="append", default=["0.0:paf"],
                         help="'<gain_db>:<label>', repeatable (default: 0.0:paf)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    raw_design = json.loads(args.design.read_text(encoding="utf-8"))
    design = {"hybrid": HybridDesign, "blend": BlendDesign, "character": CharacterBlendDesign}.get(raw_design.get("mode", "hybrid"), HybridDesign).read_json(args.design)
    gains = [_parse_gain_arg(g) for g in args.gain]
    run_validation(design, args.a2_nam, args.di, gains, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Opt-in coverage for the native real-render regression harness."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hybrid.render import NamRenderError, find_nam_render_exe

ROOT = Path(__file__).resolve().parent.parent
AMP_A = ROOT / "assets/test_cabs/Clean_NoCab_Fender_Deluxe_Reverb_Head_2.nam"
AMP_B = ROOT / "assets/test_cabs/HighGain_NoCab_SLASH AFD#2 Head.nam"
CAB = ROOT / "assets/nam_models/V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav"
DI = ROOT / "assets/di/high_thrash.wav"


def _available() -> bool:
    try:
        find_nam_render_exe()
    except NamRenderError:
        return False
    return all(path.is_file() for path in (AMP_A, AMP_B, CAB, DI))


@pytest.mark.skipif(not _available(), reason="requires built native renderer and repository real-render fixtures")
def test_real_render_harness_exercises_every_mode_and_cab(tmp_path):
    output = tmp_path / "render-report"
    completed = subprocess.run(
        [sys.executable, "scripts/real_render_regression.py", "--amp-a", str(AMP_A), "--amp-b", str(AMP_B), "--di", str(DI), "--cab", str(CAB), "--out-dir", str(output), "--release"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "real_render_report.json").read_text())
    modes = report["cases"][0]["modes"]
    assert set(modes) == {"hybrid", "blend", "character", "hybrid_cab", "blend_cab", "character_cab"}
    assert all(result["valid"] and not result["silent"] for result in modes.values())

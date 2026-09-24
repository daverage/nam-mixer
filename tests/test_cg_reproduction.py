"""The frozen JCM800 and Vibrolux FC configurations, reproduced through the hybrid/continuous_gain/ backend.

Reads the archived Phase 4 measurements (work/p4/<amp>/{profile,audit}.json, restored from
~/Documents/hybrid-nam-builder-archive/research_work_dirs_*.tar.gz) and the frozen FC record
(docs/history/Continuous Gain/final/manifest_frozen.json), so it skips on a machine without them.
The expensive half -- rebuilding the training audio from the user's real captures and comparing its SHA-256 with
the frozen bundle manifest -- runs only with CG_REPRODUCE_AUDIO=1 (scripts/cg_reproduce_fc.py does the same
from the command line); it was verified bit-for-bit for both amps when the backend was written, and again
on 2026-09-24 after the repo review.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hybrid.continuous_gain.anchors import response_anchors
from hybrid.continuous_gain.audit import alignment_shift
from hybrid.continuous_gain.bundle import make_chain
from hybrid.continuous_gain.selection import resolve_selection, select_captures

REPO = Path(__file__).resolve().parent.parent
AMPS = ("jcm800", "vibrolux")


FROZEN_MANIFEST = REPO / "docs" / "history" / "Continuous Gain" / "final" / "manifest_frozen.json"


def _frozen() -> dict | None:
    return json.loads(FROZEN_MANIFEST.read_text()) if FROZEN_MANIFEST.is_file() else None


def _archive(amp: str):
    d = REPO / "work" / "p4" / amp
    b = REPO / "work" / "p4e" / "final" / amp / "FC_bundle" / "manifest.json"
    if not all(f.is_file() for f in (d / "profile.json", d / "audit.json", b)):
        pytest.skip("archived Phase 4 measurements not restored under work/")
    return json.loads((d / "profile.json").read_text()), json.loads((d / "audit.json").read_text()), json.loads(b.read_text())


@pytest.mark.parametrize("amp", AMPS)
def test_automatic_selection_and_fc_anchors_reproduce_the_frozen_configuration(amp):
    frozen = _frozen()
    if frozen is None:
        pytest.skip("frozen FC manifest not available")
    profile, audit, bundle = _archive(amp)
    analysis = select_captures(profile, audit)
    sel = resolve_selection(analysis, profile, audit, "automatic")
    assert sel["fallback"] is False and sel["k_star"] == len(sel["selected"])
    assert sel["selected"] == [float(g) for g in bundle["gains"]]
    assert [f"G{g:g}" for g in sel["selected"]] == list(frozen["amps"][amp]["captures"])
    pos, anchors = response_anchors(analysis["response_coordinate"], sel["selected"])
    assert anchors == [float(t) for t in bundle["anchors_designated_gain_db"]]
    assert anchors == list(map(float, frozen["amps"][amp]["anchors_designated_input_gain_db"].values()))
    chain = make_chain(pos, anchors)
    assert list(chain.levels_db) == bundle["levels_db"] and chain.reference_db == bundle["reference_db"]
    shifts = {f"{g:g}": alignment_shift(audit["captures"][f"{g:g}"]) for g in pos}
    assert shifts == bundle["alignment_shifts_samples"]


@pytest.mark.parametrize("amp", AMPS)
def test_frozen_manifest_recipe_matches_the_default_recipe(amp):
    frozen = _frozen()
    if frozen is None:
        pytest.skip("frozen FC manifest not available")
    _, _, bundle = _archive(amp)
    from hybrid.continuous_gain.bundle import FC_RECIPE
    assert tuple(bundle["train_offsets_db"]) == FC_RECIPE.train_offsets_db and tuple(bundle["val_offsets_db"]) == FC_RECIPE.val_offsets_db
    assert bundle["reference_db"] == FC_RECIPE.reference_db
    assert frozen["training"]["epochs"] == 60 and frozen["training"]["batch_size"] == 16 and frozen["training"]["ny"] == 8192


@pytest.mark.skipif(os.environ.get("CG_REPRODUCE_AUDIO") != "1", reason="set CG_REPRODUCE_AUDIO=1 (renders the 11.6-minute training audio through every selected capture of each amp; about a minute in total on an Apple-silicon Mac)")
@pytest.mark.parametrize("amp", AMPS)
def test_training_audio_and_output_scale_reproduce_the_frozen_bundle_bit_for_bit(amp):
    import subprocess as sp
    r = sp.run([str(REPO / ".venv" / "bin" / "python"), str(REPO / "scripts" / "cg_reproduce_fc.py"), amp], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

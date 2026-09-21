"""Continuous Gain FC training bundle: one standard NAM from a chain of fixed-gain captures of ONE amp.

Extracted from the frozen Final-Candidate builder (`scripts/fc_build.py` + `scripts/p4e_build.py`, now archived), whose
Phase 4E arguments it reproduces exactly (audio SHA-256 checked in tests/test_cg_reproduction.py against
docs/final/manifest_frozen.json). The recipe:

  * target = level-driven blend of the REAL captures (`hybrid.multi_blend`): capture k owns chain level
    `anchor_k + REFERENCE_DB` and is rendered on the input rescaled to the common reference playing level;
  * training input = the official NAM input plus guitar DIs at level offsets covering the whole playback range;
    validation input = a held-out DI at several offsets, at the end;
  * verified Phase 4A timing corrections are applied to the renders (never unverified ones);
  * ONE fixed peak-ceiling gain reduction for the whole target (never a limiter, never per-capture level matching);
  * `output_scale_c` records that constant: in a player it is the output-level setting, not hidden DSP.

The trainers receive a bundle in the same shape as every other A2 bundle (`input.wav`, `hybrid_target.wav`,
`training_manifest.json`, mode "continuous_gain"); the custom (non-official) input is declared in the manifest
as `training_input.custom_split` with the train/validation boundary, and both trainers apply the same
data-config patch the FC models were trained with (see hybrid/a2_training_settings.py).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .cg_anchors import REFERENCE_DB
from .cg_audit import alignment_shift
from .character_analysis import sha256_file      # shared file-hash helper
from .envelope import bounded_causal_envelope_db
from .multi_blend import GainChain, multi_blend
from .safety import apply_peak_ceiling

SR = 48000
PAD = SR // 2
CEILING_DBFS = -0.2


@dataclass(frozen=True)
class FcRecipe:
    """Training-material recipe of the frozen FC configuration (docs/final/manifest_frozen.json `data.recipe`)."""
    train_dis: tuple[str, ...] = ("clean_smooth", "moderate_hotrod", "high_thrash")
    val_dis: tuple[str, ...] = ("high_metalcore",)
    train_offsets_db: tuple[float, ...] = (-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 14.0, 20.0)
    val_offsets_db: tuple[float, ...] = (-24.0, 0.0, 8.0, 20.0)
    di_seconds: int = 20
    val_seconds: int = 8
    reference_db: float = REFERENCE_DB
    ceiling_dbfs: float = CEILING_DBFS


FC_RECIPE = FcRecipe()


def _db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def _shifted(y: np.ndarray, sh: int) -> np.ndarray:
    if sh > 0:
        return np.concatenate([y[sh:], np.zeros(sh, y.dtype)])
    if sh < 0:
        return np.concatenate([np.zeros(-sh, y.dtype), y[:sh]])
    return y


def sha256_f32(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.float32).tobytes()).hexdigest()


@dataclass
class BuiltAudio:
    input: np.ndarray
    target: np.ndarray
    segments: list[dict]
    train_stop: int
    reduction_db: float
    output_scale_c: float
    train_offsets_db: list[float]
    val_offsets_db: list[float]


def build_training_audio(chain: GainChain, render_fn: Callable[[float, np.ndarray], np.ndarray], shifts: dict[float, int],
                         official_input: np.ndarray, load_di: Callable[[str], np.ndarray], recipe: FcRecipe = FC_RECIPE,
                         progress: Callable[[str], None] | None = None) -> BuiltAudio:
    """`render_fn(position, audio)` renders capture `position` raw; `shifts[position]` is its verified alignment."""
    note = progress or (lambda _m: None)

    def segment(x: np.ndarray) -> np.ndarray:
        env = bounded_causal_envelope_db(x, SR)
        renders = [_shifted(render_fn(g, (x * _db(chain.input_scale_db(k))).astype(np.float32)), shifts.get(g, 0))
                   for k, g in enumerate(chain.labels)]
        return multi_blend(env, renders, chain)

    srcs = [("official", 0.0, official_input)] + [(n, o, load_di(n)[: recipe.di_seconds * SR]) for n in recipe.train_dis for o in recipe.train_offsets_db]
    vsrc = [(n, o, load_di(n)[: recipe.val_seconds * SR]) for n in recipe.val_dis for o in recipe.val_offsets_db]
    X, Y, seg, pos, train_stop = [], [], [], 0, 0
    for split, group in (("train", srcs), ("val", vsrc)):
        for name, off, x in group:
            x = np.concatenate([x * _db(off), np.zeros(PAD, np.float32)]).astype(np.float32)
            X.append(x)
            Y.append(segment(x))
            seg.append({"split": split, "source": name, "offset_db": off, "start": pos, "stop": pos + len(x)})
            pos += len(x)
            note(f"{split} {name} {off:+g} dB")
        if split == "train":
            train_stop = pos
    Xc, Yc = np.concatenate(X), np.concatenate(Y)
    Yc, red_db = apply_peak_ceiling(Yc[:], recipe.ceiling_dbfs)
    return BuiltAudio(Xc, Yc.astype(np.float32), seg, train_stop, float(red_db), float(10 ** (-red_db / 20)) if red_db else 1.0,
                      list(recipe.train_offsets_db), list(recipe.val_offsets_db))


def make_chain(positions: list[float], anchors_db: list[float], reference_db: float = REFERENCE_DB) -> GainChain:
    return GainChain(tuple(float(p) for p in positions), tuple(float(a) + reference_db for a in anchors_db), reference_db)


def bundle_manifest_core(built: BuiltAudio, chain: GainChain, anchors_db: list[float], shifts: dict[float, int]) -> dict:
    """The fields the frozen FC bundle manifests record (compared field-for-field in the reproduction test)."""
    total = len(built.input)
    return {"gains": list(chain.labels), "anchors_designated_gain_db": list(anchors_db), "levels_db": list(chain.levels_db),
            "reference_db": chain.reference_db, "alignment_shifts_samples": {f"{g:g}": int(shifts.get(g, 0)) for g in chain.labels},
            "train_stop": built.train_stop, "total": total, "output_scale_c": built.output_scale_c,
            "peak_ceiling_gain_reduction_db": built.reduction_db, "train_seconds": built.train_stop / SR,
            "val_seconds": (total - built.train_stop) / SR, "train_offsets_db": built.train_offsets_db, "val_offsets_db": built.val_offsets_db,
            "target_audio_sha256": sha256_f32(built.target), "input_audio_sha256": sha256_f32(built.input)}


def write_bundle(out_dir: Path, built: BuiltAudio, chain: GainChain, anchors_db: list[float], shifts: dict[float, int], *,
                 sources: list[dict], model_name: str, artifact_stem: str, design: dict, receptive_field: dict,
                 extra: dict | None = None) -> Path:
    """Write input.wav / hybrid_target.wav / training_manifest.json (the shape every A2 trainer consumes)."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "input.wav", built.input, SR, subtype="FLOAT")
    sf.write(out_dir / "hybrid_target.wav", built.target, SR, subtype="FLOAT")
    core = bundle_manifest_core(built, chain, anchors_db, shifts)
    manifest = {
        "schema_version": 1, "mode": "continuous_gain", "model_name": model_name, "artifact_stem": artifact_stem,
        "artifact_filename": f"{artifact_stem}.nam",
        "training_input": {"sample_rate": SR, "sha256": sha256_file(out_dir / "input.wav"), "custom_split": True,
                           "train_stop_samples": built.train_stop,
                           "description": "official NAM input + DI segments at level offsets (train), held-out DI (validation, at the end)"},
        "target": {"final_sha256": sha256_file(out_dir / "hybrid_target.wav"), "output_scale_c": built.output_scale_c,
                   "peak_ceiling_gain_reduction_db": built.reduction_db, "ceiling_dbfs": CEILING_DBFS,
                   "combination": "level-driven chain blend of the real captures (hybrid.multi_blend), one fixed peak-ceiling gain, no limiter"},
        "design": design, "sources": sources, "receptive_field": receptive_field, "cab": {"baked": False, "export_mode": "none"},
        "segments": built.segments, "core": core,
        **(extra or {}),
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n", encoding="utf-8")
    return out_dir / "training_manifest.json"


def source_records(positions: list[float], paths: dict[float, Path], anchors_db: list[float], chain: GainChain, audit: dict) -> list[dict]:
    recs = []
    for k, p in enumerate(positions):
        a = audit["captures"][f"{p:g}"]
        recs.append({"position": p, "filename": Path(paths[p]).name, "path": str(paths[p]), "sha256": sha256_file(Path(paths[p])),
                     "anchor_input_gain_db": anchors_db[k], "chain_level_db": chain.levels_db[k], "audit_status": a["status"],
                     "alignment_shift_samples": alignment_shift(a)})
    return recs


@dataclass
class ReceptiveFieldRecord:
    branch_samples: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"branch_samples": self.branch_samples, "cab": {"baked": False}, "note": "core dependency = max over the source captures; the chain blend adds no temporal dependency beyond the bounded causal envelope"}


def receptive_field_record(positions: list[float], loaded_models: dict, envelope_max_history_samples: int | None = None) -> dict:
    """Core RF per source capture (hard gate) plus the bounded causal envelope's history."""
    from .receptive_field import compute_source_nam_receptive_field

    bs = {f"G{p:g}": int(compute_source_nam_receptive_field(loaded_models[p])) for p in positions}
    if envelope_max_history_samples is not None:
        bs["envelope"] = int(envelope_max_history_samples)
    return ReceptiveFieldRecord(bs).as_dict()


def frozen_design_record(positions, anchors_db, recipe: FcRecipe, mode: str, selection: dict | None, mapping: list[dict]) -> dict:
    return {"kind": "continuous_gain_fc", "anchor_method": mode, "positions": list(positions), "anchors_input_gain_db": list(anchors_db),
            "recipe": asdict(recipe), "selection": selection, "input_gain_mapping": mapping}

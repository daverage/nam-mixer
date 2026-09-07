"""Hybrid NAM Builder -- Flask app entry point.

Experimental proof-of-concept. See README.md for the overall concept and current
limitations. Run with:

    python app.py

then open http://127.0.0.1:5000/ in a browser.
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from hybrid.a2_training_settings import A2_EPOCH_PRESETS, DEFAULT_EPOCH_PRESET
from hybrid.blend import DEFAULT_TRANSITION_WIDTH_DB, TRANSITION_WIDTH_PRESETS_DB
from hybrid.blend_training_target import generate_blend_training_bundle
from hybrid.character_analysis import CharacterAnalysisConfig, analyse_rendered_audio, load_cached_analysis, sha256_file, store_cached_analysis
from hybrid.character_blend import CharacterBlendDesign, build_character_blend, evaluate_low_level_response, freeze_character_design
from hybrid.character_training_target import LOW_LEVEL_CHECK_REFERENCE_SECONDS, generate_character_training_bundle
from hybrid.cab_ir import CabIrError, cab_design_from_prepared, get_prepared_cab_ir
from hybrid.calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU
from hybrid.coverage import analyse_profile_coverage, envelope_percentiles, suggest_crossover_dbfs
from hybrid.design import freeze_design
from hybrid.fixed_blend import build_fixed_blend, freeze_blend_design
from hybrid.input_profiles import (
    PROFILE_ORDER_BY_INSTRUMENT,
    PROFILES_BY_INSTRUMENT,
    db_to_amplitude,
    get_profile,
    resolve_profile_gain_db,
)
from hybrid.kaggle_training import (
    KaggleJobManager,
    KaggleTrainingError,
    find_active_job,
    load_job,
)
from hybrid.metadata import suggested_nam_filename
from hybrid.local_training import LocalTrainingManager
from hybrid.nam_loader import load_nam
from hybrid.pipeline import RenderedPair, build_hybrid, render_pair
from hybrid.render import NamRenderError, render
from hybrid.safety import preview_safety_limiter
from hybrid.training_target import TrainingInputError, generate_training_bundle, validate_training_input

# Applying a hot profile to an already-normalized DI can push it over 0 dBFS.
# We warn rather than silently clip or normalize -- see docs/INPUT_PROFILE_RESEARCH.md.
PEAK_WARNING_THRESHOLD_DBFS = 0.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DI_DIR = BASE_DIR / "assets" / "di"
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)
NAM_UPLOAD_DIR = WORK_DIR / "uploaded_nam"
NAM_UPLOAD_DIR.mkdir(exist_ok=True)
CAB_UPLOAD_DIR = WORK_DIR / "uploaded_cab"
CAB_UPLOAD_DIR.mkdir(exist_ok=True)
TRAINING_INPUT_DIR = WORK_DIR / "training_input"
TRAINING_INPUT_DIR.mkdir(exist_ok=True)
TRAINING_INPUT_PATH = TRAINING_INPUT_DIR / "input.wav"
A2_OUTPUT_DIR = WORK_DIR / "a2"
A2_OUTPUT_DIR.mkdir(exist_ok=True)

_kaggle_manager = KaggleJobManager(A2_OUTPUT_DIR)
_local_training_manager = LocalTrainingManager(BASE_DIR, A2_OUTPUT_DIR)

app = Flask(__name__)

# Single-process, single-user local tool (see README) -- a module-level cache
# for the last-rendered amp pair is the whole point of splitting
# render_pair()/build_hybrid() apart: sliders should only ever hit
# build_hybrid() against this, never re-invoke NAM inference.
_rendered_pair_cache: dict = {
    "pair": None, "amp_a_summary": None, "amp_b_summary": None, "di_file": None,
    "amp_a_path": None, "amp_b_path": None,
}


def _profile_options(instrument_type: str) -> list[dict]:
    profiles = PROFILES_BY_INSTRUMENT[instrument_type]
    order = PROFILE_ORDER_BY_INSTRUMENT[instrument_type]
    return [
        {
            "id": profiles[pid].id,
            "label": profiles[pid].label,
            "gain_db": profiles[pid].gain_db,
            "requires_custom_gain": profiles[pid].requires_custom_gain,
            "description": profiles[pid].description,
        }
        for pid in order
    ]


@app.route("/")
def index():
    di_files = sorted(p.name for p in DI_DIR.glob("*.wav"))
    return render_template(
        "index.html",
        di_files=di_files,
        transition_presets=TRANSITION_WIDTH_PRESETS_DB,
        default_transition_width_db=DEFAULT_TRANSITION_WIDTH_DB,
        guitar_profiles=_profile_options("guitar"),
        bass_profiles=_profile_options("bass"),
        default_reference_input_level_dbu=DEFAULT_REFERENCE_INPUT_LEVEL_DBU,
    )


@app.route("/api/input_profiles", methods=["GET"])
def api_input_profiles():
    """Profile metadata for both instrument families, for clients that want
    it without a full page load (the main page gets it inlined via Jinja)."""
    return jsonify({
        "guitar": _profile_options("guitar"),
        "bass": _profile_options("bass"),
        "custom_gain_range_db": [-12.0, 12.0],
        "default_reference_input_level_dbu": DEFAULT_REFERENCE_INPUT_LEVEL_DBU,
    })


@app.route("/api/nam/upload", methods=["POST"])
def api_nam_upload():
    """Accept a .nam file picked in the browser, save it under work/uploaded_nam/,
    and return its parsed metadata plus the server-side path to use as
    amp_a_path/amp_b_path in /api/render_pair.

    Saved by original filename (sanitized) -- re-uploading the same filename
    overwrites the previous copy, which is fine since these are just a working
    copy of a file the user already has, not the source of truth.
    """
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "no file uploaded"}), 400
    filename = secure_filename(upload.filename)
    if not filename.lower().endswith(".nam"):
        return jsonify({"error": "expected a .nam file"}), 400

    dest = NAM_UPLOAD_DIR / filename
    upload.save(dest)
    try:
        model = load_nam(dest)
    except (OSError, ValueError) as exc:
        dest.unlink(missing_ok=True)
        return jsonify({"error": f"not a valid .nam file: {exc}"}), 400

    return jsonify({**model.summary(), "path": str(dest)})


@app.route("/api/nam/inspect", methods=["POST"])
def api_nam_inspect():
    """Given a server-side path to a .nam file, return its parsed metadata.

    Used internally after /api/nam/upload resolves a browser-picked file to a
    server-side path; can also be called directly if you'd rather point at a
    .nam file already sitting on the machine running the server.
    """
    data = request.get_json(force=True)
    nam_path = data.get("path", "")
    if not nam_path:
        return jsonify({"error": "missing 'path'"}), 400
    try:
        model = load_nam(nam_path)
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(model.summary())


@app.route("/api/cab/upload", methods=["POST"])
def api_cab_upload():
    """Accept a cabinet IR WAV picked in the browser, save it under
    work/uploaded_cab/, and return its parsed metadata plus the server-side
    path used by /api/preview and /api/generate's cab params -- see
    hybrid/cab_ir.py and docs/blend-mode.md "CAB UPLOAD / STORAGE".

    Prepares the IR against the currently-rendered pair's sample rate (if
    any) purely to report prepared/trimmed info back to the UI -- this is
    NOT what gets used for the actual official-input bake at generation
    time (hybrid.training_target.maybe_bake_cab re-prepares against the
    training input's own sample rate; see that function's docstring for why
    the tap count can differ).
    """
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "no file uploaded"}), 400
    filename = secure_filename(upload.filename)
    if not filename.lower().endswith(".wav"):
        return jsonify({"error": "expected a .wav file"}), 400

    dest = CAB_UPLOAD_DIR / filename
    upload.save(dest)

    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    preview_sample_rate = pair.sample_rate if pair is not None else None

    response = {"filename": filename, "path": str(dest)}
    try:
        if preview_sample_rate is not None:
            prepared = get_prepared_cab_ir(dest, preview_sample_rate)
        else:
            # No pair rendered yet -- prepare against the file's own native
            # rate just to validate/describe it; preview will re-prepare
            # against the real target rate once a pair exists.
            import soundfile as _sf

            info = _sf.info(str(dest))
            prepared = get_prepared_cab_ir(dest, info.samplerate)
        response.update({
            "original_sample_rate": prepared.original_sample_rate,
            "original_channels": prepared.original_channels,
            "original_frame_count": prepared.original_frame_count,
            "duration_s": prepared.original_frame_count / prepared.original_sample_rate,
            "sha256": prepared.sha256,
            "prepared_sample_rate": prepared.sample_rate,
            "prepared_frame_count": prepared.prepared_frame_count,
            "prepared_duration_ms": prepared.prepared_duration_ms,
            "leading_samples_trimmed": prepared.leading_samples_trimmed,
            # Diagnostic-only cabinet energy profile (docs/blend-mode.md
            # "CABINET ENERGY ANALYSIS") -- how much of this IR is actually
            # meaningful signal vs. raw WAV/FIR length. Never used to alter
            # the actual convolution.
            "energy_99_samples": prepared.energy_99_samples,
            "energy_99_ms": prepared.energy_99_ms,
            "energy_999_samples": prepared.energy_999_samples,
            "energy_999_ms": prepared.energy_999_ms,
            "energy_9999_samples": prepared.energy_9999_samples,
            "energy_9999_ms": prepared.energy_9999_ms,
        })
    except CabIrError as exc:
        dest.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400

    return jsonify(response)


def _parse_cab_params(data: dict, pair_sample_rate: int):
    """Shared cab-preview parsing for /api/preview (source=a/b/hybrid/blend).
    Returns a `PreparedCabIr` or None. Cab is a SHARED, mode-independent
    post-amp stage -- see docs/blend-mode.md "SHARED CABINET IR STAGE"."""
    cab_path = data.get("cab_path")
    cab_enabled = bool(data.get("cab_preview_enabled", False))
    if not cab_path or not cab_enabled:
        return None
    return get_prepared_cab_ir(cab_path, pair_sample_rate)


def _resolve_cab_design(data: dict, pair_sample_rate: int):
    """Build a `CabDesign` for provenance/freezing from generate-request
    params, or None if no cab is selected. Mirrors _parse_cab_params but
    also records `baked` -- see hybrid/cab_ir.py's CabDesign."""
    cab_path = data.get("cab_path")
    if not cab_path:
        return None
    preview_enabled = bool(data.get("cab_preview_enabled", False))
    baked = bool(data.get("cab_baked", False))
    if baked:
        # Baking without preview is never allowed (docs/blend-mode.md "CAB
        # UI": "If Bake cab into A2 is enabled, automatically ensure Use cab
        # in preview is also enabled") -- enforced server-side too, not just
        # in the UI, so provenance can never record a baked-but-unaudited cab.
        preview_enabled = True
    prepared = get_prepared_cab_ir(cab_path, pair_sample_rate)
    return cab_design_from_prepared(
        prepared, original_filename=Path(cab_path).name, preview_enabled=preview_enabled, baked=baked,
    )


def _load_di(di_file: str):
    """Validate `di_file` against the actual DI directory listing (prevents path
    traversal via a hand-crafted filename) and load it as mono float32."""
    allowed = {p.name for p in DI_DIR.glob("*.wav")}
    if di_file not in allowed:
        raise ValueError(f"unknown DI file: {di_file!r}")
    audio, sample_rate = sf.read(DI_DIR / di_file, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sample_rate


@app.route("/api/render_pair", methods=["POST"])
def api_render_pair():
    """Render both amps against the chosen DI clip and cache the result.

    This is the EXPENSIVE step (runs NAM inference twice) -- the UI should
    call this only when Amp A, Amp B, the DI clip, or the INPUT PROFILE/
    CALIBRATION settings change (a profile changes the actual signal fed to
    both NAMs -- see hybrid/pipeline.py), never on a crossover/transition/
    trim slider move (that's /api/preview, against the cached RenderedPair
    below).
    """
    data = request.get_json(force=True)
    amp_a_path = data.get("amp_a_path", "")
    amp_b_path = data.get("amp_b_path", "")
    di_file = data.get("di_file", "")
    if not amp_a_path or not amp_b_path or not di_file:
        return jsonify({"error": "amp_a_path, amp_b_path, and di_file are all required"}), 400

    instrument_type = data.get("instrument_type", "guitar")
    input_profile_id = data.get("input_profile_id", "vintage_humbucker")
    custom_input_gain_db = data.get("custom_input_gain_db")
    calibration_mode = data.get("calibration_mode", "auto")
    reference_input_level_dbu = float(data.get("reference_input_level_dbu", DEFAULT_REFERENCE_INPUT_LEVEL_DBU))
    try:
        test_gain_db = float(data.get("test_gain_db", 0.0) or 0.0)
    except (TypeError, ValueError):
        return jsonify({"error": "test_gain_db must be a number"}), 400

    try:
        get_profile(instrument_type, input_profile_id)
        input_profile_gain_db = resolve_profile_gain_db(instrument_type, input_profile_id, custom_input_gain_db)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if calibration_mode not in ("auto", "raw"):
        return jsonify({"error": f"unknown calibration_mode: {calibration_mode!r} (expected 'auto' or 'raw')"}), 400

    try:
        amp_a = load_nam(amp_a_path)
        amp_b = load_nam(amp_b_path)
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"failed to load .nam file: {exc}"}), 400

    try:
        dry, sample_rate = _load_di(di_file)
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    for label, amp in (("Amp A", amp_a), ("Amp B", amp_b)):
        if amp.sample_rate is not None and float(amp.sample_rate) != float(sample_rate):
            return jsonify({
                "error": (
                    f"{label} expects {amp.sample_rate} Hz but {di_file} is "
                    f"{sample_rate} Hz. Resampling isn't implemented -- pick a "
                    "DI clip at the model's sample rate."
                )
            }), 400

    try:
        pair = render_pair(
            amp_a, amp_b, dry, sample_rate,
            instrument_type=instrument_type,
            input_profile_id=input_profile_id,
            input_profile_gain_db=input_profile_gain_db,
            test_gain_db=test_gain_db,
            calibration_mode=calibration_mode,
            reference_input_level_dbu=reference_input_level_dbu,
        )
    except NamRenderError as exc:
        return jsonify({"error": str(exc)}), 500

    _rendered_pair_cache["pair"] = pair
    _rendered_pair_cache["amp_a_summary"] = amp_a.summary()
    _rendered_pair_cache["amp_b_summary"] = amp_b.summary()
    _rendered_pair_cache["di_file"] = di_file
    _rendered_pair_cache["amp_a_path"] = amp_a_path
    _rendered_pair_cache["amp_b_path"] = amp_b_path

    warnings = []
    if pair.calibration_warning:
        warnings.append(pair.calibration_warning)
    if pair.input_peak_dbfs >= PEAK_WARNING_THRESHOLD_DBFS:
        warnings.append(
            "This simulated input exceeds 0 dBFS relative to the reference DI. "
            "The floating-point renderer can process it, but a real ADC at "
            "this combined profile + test gain would have clipped. Treat "
            "this as a stress test."
        )

    suggested_crossover = suggest_crossover_dbfs(pair.source_envelope_db)

    return jsonify({
        "amp_a": amp_a.summary(),
        "amp_b": amp_b.summary(),
        "di_file": di_file,
        "sample_rate": sample_rate,
        "duration_s": len(dry) / sample_rate,

        "instrument_type": instrument_type,
        "input_profile": input_profile_id,
        "input_profile_gain_db": input_profile_gain_db,
        "test_gain_db": test_gain_db,

        "input_peak_dbfs": pair.input_peak_dbfs,

        "calibration_mode": pair.calibration_mode,
        "calibration_applied": pair.calibration_applied,
        "reference_input_level_dbu": pair.reference_input_level_dbu,

        "amp_a_input_level_dbu": pair.amp_a_model_input_level_dbu,
        "amp_b_input_level_dbu": pair.amp_b_model_input_level_dbu,

        "amp_a_calibration_gain_db": pair.amp_a_calibration_gain_db,
        "amp_b_calibration_gain_db": pair.amp_b_calibration_gain_db,

        "suggested_crossover_dbfs": suggested_crossover,
        "envelope_percentiles": envelope_percentiles(pair.source_envelope_db[pair.source_envelope_db > -50.0]),

        "warnings": warnings,
    })


@app.route("/api/profile_coverage", methods=["POST"])
def api_profile_coverage():
    """For the currently-rendered DI, report what fraction of active playing
    time each instrument profile would land in Amp A / transition / Amp B
    at the given crossover settings. Cheap -- no NAM inference, reuses the
    cached pair's source (un-profiled) envelope, so it updates instantly as
    crossover/transition/custom-gain change and does NOT require a rerender.
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render the amp pair first (POST /api/render_pair)."}), 400

    data = request.get_json(force=True)
    try:
        crossover_dbfs = float(data.get("crossover_dbfs", -22.0))
        transition_width_db = float(data.get("transition_width_db", DEFAULT_TRANSITION_WIDTH_DB))
    except (TypeError, ValueError):
        return jsonify({"error": "crossover_dbfs/transition_width_db must be numbers"}), 400

    instrument_type = data.get("instrument_type", pair.instrument_type)
    custom_gain_db = data.get("custom_input_gain_db")
    profiles = PROFILES_BY_INSTRUMENT.get(instrument_type)
    order = PROFILE_ORDER_BY_INSTRUMENT.get(instrument_type)
    if profiles is None:
        return jsonify({"error": f"unknown instrument_type: {instrument_type!r}"}), 400

    profile_gains = []
    for pid in order:
        profile = profiles[pid]
        if profile.requires_custom_gain:
            if custom_gain_db is None:
                continue
            profile_gains.append((pid, max(-12.0, min(12.0, float(custom_gain_db)))))
        else:
            profile_gains.append((pid, profile.gain_db))

    results = analyse_profile_coverage(pair.source_envelope_db, profile_gains, crossover_dbfs, transition_width_db)

    coverage = [
        {
            "profile_id": r.profile_id,
            "label": profiles[r.profile_id].label,
            "gain_db": r.gain_db,
            "amp_a_fraction": r.amp_a_fraction,
            "transition_fraction": r.transition_fraction,
            "amp_b_fraction": r.amp_b_fraction,
        }
        for r in results
    ]

    all_amp_a = all(c["amp_a_fraction"] > 0.95 for c in coverage) if coverage else False
    all_amp_b = all(c["amp_b_fraction"] > 0.95 for c in coverage) if coverage else False
    reachability_warning = None
    if all_amp_a:
        reachability_warning = "Crossover is probably too high: normal guitar/bass output rarely reaches Amp B with this DI."
    elif all_amp_b:
        reachability_warning = "Crossover is probably too low: the hybrid spends almost no time in Amp A."

    return jsonify({"coverage": coverage, "reachability_warning": reachability_warning})


def _parse_blend_params(data: dict):
    """Fixed Blend equivalent of _parse_hybrid_params -- mix_b/manual trim/
    auto_level only, no crossover/transition (Blend has no envelope, see
    hybrid/fixed_blend.py)."""
    mix_b = max(0.0, min(1.0, float(data.get("mix_b", 0.5))))
    manual_b_trim_db = float(data.get("manual_b_trim_db", 0.0))
    auto_level = bool(data.get("auto_level", True))
    return mix_b, manual_b_trim_db, auto_level


def _parse_character_params(data: dict):
    """Character controls are independent percentages; no raw-output trim is
    used because Character Blend has one donor path and derived correction."""
    def mix(name, default=0.5):
        return max(0.0, min(1.0, float(data.get(name, default))))
    optional = lambda name: None if data.get(name) is None else mix(name)
    return {
        "tone_mix_b": mix("tone_mix_b"), "feel_mix_b": mix("feel_mix_b"), "drive_mix_b": mix("drive_mix_b"),
        "drive_low_mix_b": optional("drive_low_mix_b"), "drive_mid_mix_b": optional("drive_mid_mix_b"), "drive_high_mix_b": optional("drive_high_mix_b"),
    }


def _build_character_result(pair: RenderedPair, data: dict):
    params = _parse_character_params(data)
    # Preview analysis is intentionally derived from this already-auditioned
    # pair.  The frozen analysis is then reused by official target generation.
    design = CharacterBlendDesign(amp_a_path="", amp_b_path="", **params)
    config = CharacterAnalysisConfig()
    cache_dir = WORK_DIR / "character_analysis"
    a_path, b_path = _rendered_pair_cache.get("amp_a_path"), _rendered_pair_cache.get("amp_b_path")
    a_hash, b_hash = (sha256_file(a_path) if a_path else ""), (sha256_file(b_path) if b_path else "")
    analysis_a = load_cached_analysis(cache_dir, a_hash, config) if a_hash else None
    analysis_b = load_cached_analysis(cache_dir, b_hash, config) if b_hash else None
    if analysis_a is None:
        analysis_a = analyse_rendered_audio(pair.dry, pair.amp_a, pair.sample_rate, config, a_hash)
        if a_hash: store_cached_analysis(cache_dir, analysis_a)
    if analysis_b is None:
        analysis_b = analyse_rendered_audio(pair.dry, pair.amp_b, pair.sample_rate, config, b_hash)
        if b_hash: store_cached_analysis(cache_dir, analysis_b)
    return build_character_blend(pair, design, analysis_a=analysis_a, analysis_b=analysis_b), params


@app.route("/api/character/low_level_check", methods=["POST"])
def api_character_low_level_check():
    """Pre-flight low-level response sweep for the CURRENT Character Blend
    controls (docs/blend-mode-fixes.md, Phase 6) -- lets the UI show whether
    this design is healthy across a soft-playing gain sweep before the user
    spends time generating a bundle/training on Kaggle. Uses the exact same
    build_character_blend() as preview and the training-bundle gate (Phase 7)."""
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    a_path, b_path = _rendered_pair_cache.get("amp_a_path"), _rendered_pair_cache.get("amp_b_path")
    if pair is None or not a_path or not b_path:
        return jsonify({"error": "Render and audition an amp pair first (POST /api/render_pair)."}), 400
    data = request.get_json(force=True)
    try:
        result, params = _build_character_result(pair, data)
    except (TypeError, ValueError):
        return jsonify({"error": "character tone/feel/drive controls must be numbers"}), 400
    design = CharacterBlendDesign(
        amp_a_path="", amp_b_path="",
        analysis_a=json.loads(json.dumps(result.analysis_a.to_dict())),
        analysis_b=json.loads(json.dumps(result.analysis_b.to_dict())),
        **params,
    )
    reference = pair.profiled_dry[: int(pair.sample_rate * LOW_LEVEL_CHECK_REFERENCE_SECONDS)]
    if len(reference) == 0:
        reference = pair.profiled_dry
    amp_a, amp_b = load_nam(a_path), load_nam(b_path)

    def build_pair_at_gain(gain_db: float):
        scaled = (reference * db_to_amplitude(gain_db)).astype(np.float32)
        a = render(amp_a, (scaled * db_to_amplitude(pair.amp_a_calibration_gain_db)).astype(np.float32), pair.sample_rate)
        b = render(amp_b, (scaled * db_to_amplitude(pair.amp_b_calibration_gain_db)).astype(np.float32), pair.sample_rate)
        return SimpleNamespace(dry=scaled, amp_a=a, amp_b=b, sample_rate=pair.sample_rate)

    check = evaluate_low_level_response(build_pair_at_gain, design)
    return jsonify({"low_level_response": check.to_dict()})


def _parse_hybrid_params(data: dict):
    """Shared crossover/transition/trim parsing for /api/preview,
    /api/blend_info, and /api/blend_curve -- all drive the same
    build_hybrid() call. Input level is NOT one of these params -- it's a
    render-stage concern handled by /api/render_pair's input-profile
    settings, not a blend-stage one (build_hybrid's `dry_gain_db` is a
    deprecated test-only escape hatch, deliberately not exposed here)."""
    crossover_dbfs = float(data.get("crossover_dbfs", -22.0))
    transition_width_db = float(data.get("transition_width_db", DEFAULT_TRANSITION_WIDTH_DB))
    manual_b_trim_db = float(data.get("manual_b_trim_db", 0.0))
    auto_level = bool(data.get("auto_level", True))
    return crossover_dbfs, transition_width_db, manual_b_trim_db, auto_level


@app.route("/api/blend_info", methods=["POST"])
def api_blend_info():
    """Report the auto/manual/effective trim breakdown for the current
    crossover/transition/trim settings, without generating any audio.

    This is what lets the UI show a live "Auto match / Manual tweak /
    Effective trim" readout as sliders move, rather than only after a
    Preview Hybrid click -- build_hybrid() is cheap (no NAM inference), so
    calling it on every slider `input` event is fine.
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render the amp pair first (POST /api/render_pair)."}), 400

    data = request.get_json(force=True)
    try:
        crossover_dbfs, transition_width_db, manual_b_trim_db, auto_level = _parse_hybrid_params(data)
    except (TypeError, ValueError):
        return jsonify({"error": "crossover_dbfs/transition_width_db/manual_b_trim_db must be numbers"}), 400

    result = build_hybrid(
        pair,
        crossover_dbfs=crossover_dbfs,
        transition_width_db=transition_width_db,
        auto_level=auto_level,
        manual_b_trim_db=manual_b_trim_db,
    )
    return jsonify({
        "auto_trim_db": result.auto_trim_db,
        "manual_trim_db": result.manual_trim_db,
        "effective_b_trim_db": result.effective_b_trim_db,
        "alignment_offset_samples": result.alignment_offset_samples,
    })


@app.route("/api/mix_info", methods=["POST"])
def api_mix_info():
    """Mode-aware trim/mix readout, for both design modes -- the generic
    replacement for /api/blend_info now that "Blend" is a real design mode
    (see docs/blend-mode.md "API": /api/blend_info is kept as a
    backwards-compatible Hybrid-only alias, never removed). Cheap for both
    modes -- no NAM inference, safe on every slider move.
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render the amp pair first (POST /api/render_pair)."}), 400

    data = request.get_json(force=True)
    mode = data.get("mode", "hybrid")
    if mode == "blend":
        try:
            mix_b, manual_b_trim_db, auto_level = _parse_blend_params(data)
        except (TypeError, ValueError):
            return jsonify({"error": "mix_b/manual_b_trim_db must be numbers"}), 400
        result = build_fixed_blend(pair, mix_b=mix_b, auto_level=auto_level, manual_b_trim_db=manual_b_trim_db)
        return jsonify({
            "mode": "blend",
            "mix_b": result.mix_b,
            "mix_a": 1.0 - result.mix_b,
            "auto_trim_db": result.auto_trim_db,
            "manual_trim_db": result.manual_trim_db,
            "effective_b_trim_db": result.effective_b_trim_db,
            "alignment_offset_samples": result.alignment_offset_samples,
        })
    elif mode == "character":
        try:
            result, params = _build_character_result(pair, data)
        except (TypeError, ValueError):
            return jsonify({"error": "character tone/feel/drive controls must be numbers"}), 400
        return jsonify({"mode": "character", "tone_mix_b": params["tone_mix_b"], "feel_mix_b": params["feel_mix_b"], "drive_mix_b": params["drive_mix_b"], "drive_weight_b_min": float(result.drive_weight_b.min()), "drive_weight_b_max": float(result.drive_weight_b.max())})
    elif mode == "hybrid":
        try:
            crossover_dbfs, transition_width_db, manual_b_trim_db, auto_level = _parse_hybrid_params(data)
        except (TypeError, ValueError):
            return jsonify({"error": "crossover_dbfs/transition_width_db/manual_b_trim_db must be numbers"}), 400
        result = build_hybrid(
            pair, crossover_dbfs=crossover_dbfs, transition_width_db=transition_width_db,
            auto_level=auto_level, manual_b_trim_db=manual_b_trim_db,
        )
        return jsonify({
            "mode": "hybrid",
            "auto_trim_db": result.auto_trim_db,
            "manual_trim_db": result.manual_trim_db,
            "effective_b_trim_db": result.effective_b_trim_db,
            "alignment_offset_samples": result.alignment_offset_samples,
        })
    else:
        return jsonify({"error": f"unknown mode: {mode!r} (expected 'hybrid', 'blend', or 'character')"}), 400


@app.route("/api/blend_curve", methods=["POST"])
def api_blend_curve():
    """Downsampled time series for the "journey between amps" visualization:
    the profile-adjusted dry envelope and the resulting Amp B blend weight,
    both against a shared time axis.

    Downsampled with a simple stride (not min/max decimation) to a fixed
    number of points -- this is a debug/visualization aid, not the audio
    path, so losing brief spikes between sampled points is an acceptable
    tradeoff for a small, fast response.
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render the amp pair first (POST /api/render_pair)."}), 400

    data = request.get_json(force=True)
    try:
        crossover_dbfs, transition_width_db, manual_b_trim_db, auto_level = _parse_hybrid_params(data)
    except (TypeError, ValueError):
        return jsonify({"error": "crossover_dbfs/transition_width_db/manual_b_trim_db must be numbers"}), 400
    max_points = int(data.get("max_points", 600))

    result = build_hybrid(
        pair,
        crossover_dbfs=crossover_dbfs,
        transition_width_db=transition_width_db,
        auto_level=auto_level,
        manual_b_trim_db=manual_b_trim_db,
    )

    n = len(result.blend_curve)
    stride = max(1, n // max_points)
    idx = range(0, n, stride)
    times = [i / pair.sample_rate for i in idx]
    envelope_db = [float(result.envelope_db[i]) for i in idx]
    blend_weight = [float(result.blend_curve[i]) for i in idx]

    return jsonify({
        "times": times,
        "envelope_db": envelope_db,
        "blend_weight": blend_weight,
        "crossover_dbfs": crossover_dbfs,
        "transition_width_db": transition_width_db,
    })


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Return audio (WAV bytes) for Amp A, Amp B, the Hybrid crossfade, or
    the Fixed Blend mix.

    Cheap for `source="hybrid"`/`source="blend"`: reuses the RenderedPair
    cached by /api/render_pair and only recombines already-rendered audio --
    no NAM inference here, so this is safe to call on every crossover/
    transition/trim/mix slider move. If `cab_preview_enabled` and `cab_path`
    are given, the SAME shared cab is applied to Amp A, Amp B, AND the
    Hybrid/Blend result (see docs/blend-mode.md "CAB PREVIEW SEMANTICS") so
    A/Result/B comparisons stay fair -- applied AFTER the amp combination,
    BEFORE preview_safety_limiter (playback safety net only -- never used on
    a training target, see hybrid/safety.py).
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render the amp pair first (POST /api/render_pair)."}), 400

    data = request.get_json(force=True)
    source = data.get("source", "hybrid")

    headers = {}
    if source == "a":
        audio = pair.amp_a
    elif source == "b":
        audio = pair.amp_b
    elif source == "hybrid":
        try:
            crossover_dbfs, transition_width_db, manual_b_trim_db, auto_level = _parse_hybrid_params(data)
        except (TypeError, ValueError):
            return jsonify({"error": "crossover_dbfs/transition_width_db/manual_b_trim_db must be numbers"}), 400

        result = build_hybrid(
            pair,
            crossover_dbfs=crossover_dbfs,
            transition_width_db=transition_width_db,
            auto_level=auto_level,
            manual_b_trim_db=manual_b_trim_db,
        )
        audio = result.hybrid
        headers = {
            "X-Auto-Trim-Db": f"{result.auto_trim_db:.3f}",
            "X-Manual-Trim-Db": f"{result.manual_trim_db:.3f}",
            "X-Effective-Trim-Db": f"{result.effective_b_trim_db:.3f}",
            "X-Alignment-Offset-Samples": str(result.alignment_offset_samples),
        }
    elif source == "blend":
        try:
            mix_b, manual_b_trim_db, auto_level = _parse_blend_params(data)
        except (TypeError, ValueError):
            return jsonify({"error": "mix_b/manual_b_trim_db must be numbers"}), 400

        result = build_fixed_blend(pair, mix_b=mix_b, auto_level=auto_level, manual_b_trim_db=manual_b_trim_db)
        audio = result.blend
        headers = {
            "X-Mix-B": f"{result.mix_b:.4f}",
            "X-Auto-Trim-Db": f"{result.auto_trim_db:.3f}",
            "X-Manual-Trim-Db": f"{result.manual_trim_db:.3f}",
            "X-Effective-Trim-Db": f"{result.effective_b_trim_db:.3f}",
            "X-Alignment-Offset-Samples": str(result.alignment_offset_samples),
        }
    elif source == "character":
        try:
            result, params = _build_character_result(pair, data)
        except (TypeError, ValueError):
            return jsonify({"error": "character tone/feel/drive controls must be numbers"}), 400
        audio = result.blend
        headers = {"X-Tone-Mix-B": f"{params['tone_mix_b']:.4f}", "X-Feel-Mix-B": f"{params['feel_mix_b']:.4f}", "X-Drive-Mix-B": f"{params['drive_mix_b']:.4f}"}
    else:
        return jsonify({"error": f"unknown source: {source!r} (expected a, b, hybrid, blend, or character)"}), 400

    try:
        cab = _parse_cab_params(data, pair.sample_rate)
    except CabIrError as exc:
        return jsonify({"error": f"cab preview error: {exc}"}), 400
    if cab is not None:
        from hybrid.cab_ir import apply_cab_ir
        audio = apply_cab_ir(audio.astype("float32"), cab)

    audio = preview_safety_limiter(audio.astype("float32"))
    buf = io.BytesIO()
    sf.write(buf, audio, pair.sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(buf.read(), mimetype="audio/wav", headers=headers)


@app.post("/api/live_blend_stems")
def api_live_blend_stems():
    """Return the cached A/B renders as an aligned stereo WAV for browser-side
    live *fixed* blending.

    The left channel is Amp A; the right is Amp B after the exact same
    level-match and manual trim used by :func:`build_fixed_blend`.  The browser
    can therefore change only the linear mix gain without another HTTP request
    or NAM inference.  This deliberately does not apply the preview limiter:
    it is applied after the live mix by Web Audio's safety compressor, whereas
    limiting either stem first would change the blend.
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render the amp pair first (POST /api/render_pair)."}), 400

    data = request.get_json(force=True)
    try:
        mix_b, manual_b_trim_db, auto_level = _parse_blend_params(data)
    except (TypeError, ValueError):
        return jsonify({"error": "mix_b/manual_b_trim_db must be numbers"}), 400

    # `mix_b` is irrelevant to the stems themselves, but calling the canonical
    # builder means its trim/length policy cannot drift from the exported A2.
    result = build_fixed_blend(
        pair, mix_b=mix_b, auto_level=auto_level, manual_b_trim_db=manual_b_trim_db,
        align_enabled=False,
    )
    n = min(len(pair.amp_a), len(pair.amp_b), len(pair.envelope_db))
    amp_a = pair.amp_a[:n].astype(np.float32)
    amp_b = pair.amp_b[:n].astype(np.float32) * (10.0 ** (result.effective_b_trim_db / 20.0))

    try:
        cab = _parse_cab_params(data, pair.sample_rate)
    except CabIrError as exc:
        return jsonify({"error": f"cab live-preview error: {exc}"}), 400
    if cab is not None:
        # Convolution is linear, so applying the shared cabinet to both stems
        # before the browser's linear blend is exactly equivalent to applying
        # it to their blend afterwards.
        from hybrid.cab_ir import apply_cab_ir
        amp_a = apply_cab_ir(amp_a, cab)
        amp_b = apply_cab_ir(amp_b, cab)

    buf = io.BytesIO()
    sf.write(buf, np.column_stack((amp_a, amp_b)), pair.sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return Response(buf.read(), mimetype="audio/wav", headers={
        "X-Auto-Trim-Db": f"{result.auto_trim_db:.3f}",
        "X-Effective-Trim-Db": f"{result.effective_b_trim_db:.3f}",
        "X-Live-Audition": "fixed-blend-stems",
    })


@app.route("/api/training_input/status", methods=["GET"])
def api_training_input_status():
    """Whether an official NAM training input has been uploaded/is usable --
    see docs/phase3.md section 7. Never falls back to a genre DI clip."""
    if not TRAINING_INPUT_PATH.is_file():
        return jsonify({"ready": False, "path": str(TRAINING_INPUT_PATH), "error": "no official training input uploaded yet"})
    try:
        _, info = validate_training_input(TRAINING_INPUT_PATH)
    except TrainingInputError as exc:
        return jsonify({"ready": False, "path": str(TRAINING_INPUT_PATH), "error": str(exc)})
    return jsonify({
        "ready": True, "path": str(TRAINING_INPUT_PATH),
        "sample_rate": info.sample_rate, "frame_count": info.frame_count, "sha256": info.sha256,
    })


@app.get("/api/local_training/status")
def api_local_training_status():
    return jsonify(_local_training_manager.status())


@app.post("/api/local_training/setup")
def api_local_training_setup():
    try:
        _local_training_manager.setup()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(_local_training_manager.status()), 202


@app.post("/api/local_training/start")
def api_local_training_start():
    data = request.get_json(force=True)
    design_id = str(data.get("design_id") or "")
    preset = str(data.get("epoch_preset") or DEFAULT_EPOCH_PRESET)
    if preset not in A2_EPOCH_PRESETS:
        return jsonify({"error": f"unknown epoch preset: {preset}"}), 400
    manifest = A2_OUTPUT_DIR / secure_filename(design_id) / "training_manifest.json"
    try:
        _local_training_manager.train(manifest, preset)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_local_training_manager.status()), 202


@app.route("/api/training_input/upload", methods=["POST"])
def api_training_input_upload():
    """Accept the official NAM training input WAV picked in the browser.
    Validated immediately (mono, 48 kHz, finite) -- an invalid file is
    rejected and not saved, per docs/phase3.md section 7's "ABORT, do not
    bypass the check"."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "no file uploaded"}), 400
    TRAINING_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    upload.save(TRAINING_INPUT_PATH)
    try:
        _, info = validate_training_input(TRAINING_INPUT_PATH)
    except TrainingInputError as exc:
        TRAINING_INPUT_PATH.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "ready": True, "path": str(TRAINING_INPUT_PATH),
        "sample_rate": info.sample_rate, "frame_count": info.frame_count, "sha256": info.sha256,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Freeze the currently-auditioned design (Hybrid or Blend, `mode` in the
    request body, defaulting to "hybrid" for backward compatibility) and
    generate a real, reproducible A2 training bundle from it -- see
    hybrid/design.py, hybrid/fixed_blend.py, hybrid/training_target.py, and
    hybrid/blend_training_target.py. Requires a rendered/auditioned amp pair
    (POST /api/render_pair) and an uploaded official NAM training input
    (POST /api/training_input/upload) -- never trains on the preview/genre DI.
    """
    pair: RenderedPair | None = _rendered_pair_cache["pair"]
    if pair is None:
        return jsonify({"error": "Render and audition an amp pair first (POST /api/render_pair)."}), 400
    amp_a_path = _rendered_pair_cache["amp_a_path"]
    amp_b_path = _rendered_pair_cache["amp_b_path"]
    di_file = _rendered_pair_cache["di_file"]

    if not TRAINING_INPUT_PATH.is_file():
        return jsonify({
            "error": "Official NAM training input is missing. Upload one first (POST /api/training_input/upload).",
            "training_input_ready": False,
        }), 400

    data = request.get_json(force=True)
    mode = data.get("mode", "hybrid")
    if mode not in ("hybrid", "blend", "character"):
        return jsonify({"error": f"unknown mode: {mode!r} (expected 'hybrid', 'blend', or 'character')"}), 400

    try:
        cab = _resolve_cab_design(data, pair.sample_rate)
    except CabIrError as exc:
        return jsonify({"error": f"cab error: {exc}"}), 400

    # A timestamp is useful for a log, but a poor identity for a model: it
    # leaks all the way through the bundle, Kaggle resources, and the browser
    # download name.  Keep a human-readable display name in the manifest and
    # derive one safe, stable filesystem stem from it.
    requested_name = str(data.get("model_name") or data.get("design_id") or "").strip()
    if not requested_name:
        amp_a_name = Path(amp_a_path).stem
        amp_b_name = Path(amp_b_path).stem
        joiner = " + " if mode in ("blend", "character") else " to "
        requested_name = f"{amp_a_name}{joiner}{amp_b_name}"
    if len(requested_name) > 100:
        return jsonify({"error": "model_name must be 100 characters or fewer"}), 400
    design_id = secure_filename(requested_name)
    if not design_id:
        return jsonify({"error": "model_name must contain letters or numbers"}), 400
    bundle_dir = A2_OUTPUT_DIR / design_id
    if (bundle_dir / "training_manifest.json").is_file():
        return jsonify({
            "error": f"a training bundle named {requested_name!r} already exists; choose a distinct model name",
            "model_name": requested_name,
        }), 409

    if mode == "character":
        try:
            result, params = _build_character_result(pair, data)
        except (TypeError, ValueError):
            return jsonify({"error": "character tone/feel/drive controls must be numbers"}), 400
        design = freeze_character_design(pair, result, amp_a_path=amp_a_path, amp_b_path=amp_b_path,
            amp_a_sha256=sha256_file(amp_a_path), amp_b_sha256=sha256_file(amp_b_path), design_di_file=di_file, cab=cab, **params)
        try:
            design.write_json(bundle_dir / "character_design.json")
            bundle = generate_character_training_bundle(design, TRAINING_INPUT_PATH, bundle_dir)
        except (TrainingInputError, CabIrError, OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
    elif mode == "blend":
        try:
            mix_b, manual_b_trim_db, auto_level = _parse_blend_params(data)
        except (TypeError, ValueError):
            return jsonify({"error": "mix_b/manual_b_trim_db must be numbers"}), 400

        # Alignment is never exposed as a UI control, matching Hybrid mode.
        result = build_fixed_blend(pair, mix_b=mix_b, auto_level=auto_level, manual_b_trim_db=manual_b_trim_db, align_enabled=False)
        design = freeze_blend_design(
            pair, result,
            amp_a_path=amp_a_path, amp_b_path=amp_b_path,
            alignment_enabled=False, design_di_file=di_file, cab=cab,
        )
        try:
            design.write_json(bundle_dir / "blend_design.json")
            bundle = generate_blend_training_bundle(design, TRAINING_INPUT_PATH, bundle_dir)
        except (TrainingInputError, CabIrError, OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        try:
            crossover_dbfs, transition_width_db, manual_b_trim_db, auto_level = _parse_hybrid_params(data)
        except (TypeError, ValueError):
            return jsonify({"error": "crossover_dbfs/transition_width_db/manual_b_trim_db must be numbers"}), 400

        # Alignment is never exposed as a UI control (see freeze_design/build_hybrid
        # call sites) -- it stays off, matching every other build_hybrid() call in
        # this app.
        result = build_hybrid(
            pair,
            crossover_dbfs=crossover_dbfs,
            transition_width_db=transition_width_db,
            auto_level=auto_level,
            manual_b_trim_db=manual_b_trim_db,
            align_enabled=False,
        )

        design = freeze_design(
            pair, result,
            amp_a_path=amp_a_path, amp_b_path=amp_b_path,
            crossover_dbfs=crossover_dbfs, transition_width_db=transition_width_db,
            alignment_enabled=False, design_di_file=di_file, cab=cab,
        )

        try:
            design.write_json(bundle_dir / "hybrid_design.json")
            bundle = generate_training_bundle(design, TRAINING_INPUT_PATH, bundle_dir)
        except (TrainingInputError, CabIrError, OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    # These fields are deliberately part of the training manifest rather than
    # merely response/UI state: both local and Kaggle exporters consume the
    # manifest to set the final NAM's embedded UserMetadata and file basename.
    bundle.manifest["model_name"] = requested_name
    bundle.manifest["artifact_stem"] = design_id
    bundle.manifest["artifact_filename"] = f"{design_id}.nam"
    with open(bundle.training_manifest_path, "w", encoding="utf-8") as f:
        json.dump(bundle.manifest, f, indent=2)

    return jsonify({
        "design_id": design_id,
        "model_name": requested_name,
        "download_filename": f"{design_id}.nam",
        "mode": mode,
        "bundle_dir": str(bundle.bundle_dir),
        "input_path": str(bundle.input_path),
        "target_path": str(bundle.hybrid_target_path),
        "manifest_path": str(bundle.training_manifest_path),
        "safety_report": {
            "raw_peak_dbfs": bundle.safety.raw_peak_dbfs,
            "final_peak_dbfs": bundle.safety.final_peak_dbfs,
            "gain_reduction_db": bundle.safety.gain_reduction_db,
        },
        "calibration_summary": {
            "requested_mode": design.calibration_mode,
            "effective_mode": design.calibration_effective_mode,
            "applied": design.calibration_applied,
            "amp_a_gain_db": design.amp_a_calibration_gain_db,
            "amp_b_gain_db": design.amp_b_calibration_gain_db,
        },
        "cab_summary": design.cab.to_dict() if design.cab else None,
        "low_level_response": bundle.manifest.get("low_level_response"),
        "training_command": f"python scripts/train_a2.py {bundle.training_manifest_path}",
        "epoch_presets": A2_EPOCH_PRESETS,
        "default_epoch_preset": "high_def" if mode == "character" else DEFAULT_EPOCH_PRESET,
        "warnings": bundle.warnings,
        "implemented": True,
    })


def _suggested_nam_filename(design_id: str) -> str:
    """Download filename for a trained A2 model -- reads the bundle's own
    training_manifest.json (already-recorded amp filenames/mode/mix) and
    defers to hybrid.metadata.suggested_nam_filename so the naming logic
    lives in one place shared with scripts/train_a2.py, rather than
    duplicated per caller. Falls back to `<design_id>.nam` if the manifest
    is missing/unreadable (e.g. a bundle generated before this existed)."""
    manifest_path = A2_OUTPUT_DIR / secure_filename(str(design_id)) / "training_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    return suggested_nam_filename(manifest, fallback=str(design_id))


def _job_dict_for_client(job) -> dict:
    """job.to_dict() plus `download_filename` -- the EXACT filename
    GET /api/kaggle/jobs/<id>/download will actually save the file as,
    computed the same way here so the UI's download button/link can never
    show a different name (e.g. the internal `hybrid_a2.nam` export
    basename baked inside the Kaggle kernel, job.output_nam_path's own
    basename) than what the browser actually saves."""
    data = job.to_dict()
    data["download_filename"] = _suggested_nam_filename(job.design_id)
    return data


@app.route("/api/kaggle/status", methods=["GET"])
def api_kaggle_status():
    """CLI/auth/quota status plus the most recent job for a design, if any --
    see hybrid/kaggle_training.py. Never raises for a missing/unauthenticated
    CLI -- reports that plainly instead."""
    info = _kaggle_manager.status()
    design_id = request.args.get("design_id")
    if design_id:
        job = find_active_job(A2_OUTPUT_DIR, design_id)
        info["job"] = _job_dict_for_client(job) if job else None
    return jsonify(info)


@app.route("/api/kaggle/auth/start", methods=["POST"])
def api_kaggle_auth_start():
    """Starts `kaggle auth login` as a non-blocking background process --
    never a custom OAuth implementation, never reads/stores the resulting
    credential (docs/kaggle_training.md)."""
    if not _kaggle_manager.cli.is_installed():
        return jsonify({"error": "Kaggle CLI is not installed. Run: pip install kaggle", "command": "pip install kaggle"}), 400
    started = _kaggle_manager.cli.launch_auth_login()
    if not started:
        logger.warning("could not launch kaggle auth login")
    return jsonify({
        "started": started,
        "command": "kaggle auth login",
        "note": "If a browser window didn't open, run the command above yourself, then refresh /api/kaggle/status.",
    })


@app.route("/api/kaggle/train", methods=["POST"])
def api_kaggle_train():
    """Start a Kaggle GPU job for an already-generated training bundle.
    Never recomputes the design/manifest -- reuses the bundle written by
    POST /api/generate.

    Returns as soon as the (fast) CLI/auth pre-checks pass -- the actual
    stage/upload/verify/kernel pipeline runs on a background thread
    (KaggleJobManager.submit_async), NOT inline in this request. A real
    production upload was observed taking several minutes under real
    network conditions (see docs/kaggle_training.md); blocking this request
    for that long left the Flask dev server unresponsive with no way for
    the UI to show progress, and any interruption lost the job's state
    entirely. Poll GET /api/kaggle/jobs/<job_id> for progress.
    """
    data = request.get_json(force=True, silent=True) or {}
    design_id = data.get("design_id")
    if not design_id:
        return jsonify({"error": "design_id is required (from a prior POST /api/generate response)"}), 400
    bundle_dir = A2_OUTPUT_DIR / secure_filename(str(design_id))
    manifest_path = bundle_dir / "training_manifest.json"
    if not manifest_path.is_file():
        return jsonify({"error": f"no generated training bundle found for design_id {design_id!r} -- call POST /api/generate first"}), 400

    # Character teachers are deliberately more complex than a source NAM;
    # honor their 120-epoch default for direct API users as well as the UI.
    manifest_mode = json.loads(manifest_path.read_text(encoding="utf-8")).get("mode")
    epoch_preset = data.get("epoch_preset", "high_def" if manifest_mode == "character" else DEFAULT_EPOCH_PRESET)
    if epoch_preset not in A2_EPOCH_PRESETS:
        return jsonify({"error": f"epoch_preset must be one of {sorted(A2_EPOCH_PRESETS)}, got {epoch_preset!r}"}), 400

    try:
        job = _kaggle_manager.submit_async(design_id, bundle_dir, epoch_preset=epoch_preset)
    except KaggleTrainingError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"job_id": job.job_id, "design_id": job.design_id, "state": job.state})


@app.route("/api/kaggle/jobs/<job_id>", methods=["GET"])
def api_kaggle_job_status(job_id: str):
    design_id = request.args.get("design_id")
    if not design_id:
        return jsonify({"error": "design_id query parameter is required"}), 400
    job = load_job(A2_OUTPUT_DIR, design_id, job_id)
    if job is None:
        return jsonify({"error": f"unknown job {job_id!r} for design {design_id!r}"}), 404
    job = _kaggle_manager.refresh(job)
    # Pull fresh logs from Kaggle on every poll while the job is still
    # active -- reading only the local log FILE here (as this used to do)
    # meant it never actually changed unless something else happened to hit
    # /jobs/<id>/logs first, so the UI looked stuck even while training was
    # progressing normally.
    tail = _kaggle_manager.fetch_logs(job) if job.state not in ("complete", "failed") else _kaggle_manager.read_log_tail(job)
    response = _job_dict_for_client(job)
    response["progress"] = _kaggle_manager.parse_progress(tail)
    response["log_tail"] = "\n".join(tail.splitlines()[-20:])
    return jsonify(response)


@app.route("/api/kaggle/jobs/<job_id>/logs", methods=["GET"])
def api_kaggle_job_logs(job_id: str):
    design_id = request.args.get("design_id")
    if not design_id:
        return jsonify({"error": "design_id query parameter is required"}), 400
    job = load_job(A2_OUTPUT_DIR, design_id, job_id)
    if job is None:
        return jsonify({"error": f"unknown job {job_id!r} for design {design_id!r}"}), 404
    tail = _kaggle_manager.fetch_logs(job) if job.state not in ("complete", "failed") else _kaggle_manager.read_log_tail(job)
    return jsonify({"log_tail": tail, "progress": _kaggle_manager.parse_progress(tail)})


@app.route("/api/kaggle/jobs/<job_id>/download", methods=["GET"])
def api_kaggle_job_download(job_id: str):
    """Serves the trained .nam directly as a browser download, so the user
    doesn't have to locate it on the server's filesystem themselves (it
    normally lands several directories deep under
    work/a2/<design_id>/kaggle/<job_id>/output/...). Only ever serves the
    exact path this app itself already recorded on job.output_nam_path for
    a completed job -- never a caller-supplied path."""
    design_id = request.args.get("design_id")
    if not design_id:
        return jsonify({"error": "design_id query parameter is required"}), 400
    job = load_job(A2_OUTPUT_DIR, design_id, job_id)
    if job is None:
        return jsonify({"error": f"unknown job {job_id!r} for design {design_id!r}"}), 404
    if job.state != "complete" or not job.output_nam_path:
        return jsonify({"error": f"job {job_id!r} has no downloadable model yet (state={job.state})"}), 400

    nam_path = Path(job.output_nam_path)
    if not nam_path.is_file():
        return jsonify({"error": f"recorded model file no longer exists on disk: {nam_path}"}), 404

    download_name = _suggested_nam_filename(design_id)
    return send_file(nam_path, as_attachment=True, download_name=download_name)


@app.route("/api/kaggle/jobs/<job_id>/recover", methods=["POST"])
def api_kaggle_job_recover(job_id: str):
    """Recovers a job whose Kaggle training genuinely completed but whose
    local output download/validation failed for an unrelated reason (e.g.
    the `--file-pattern` regex bug) -- never re-uploads the dataset, never
    re-pushes the kernel, never re-runs training. See
    KaggleJobManager.retry_download."""
    design_id = request.args.get("design_id") or (request.get_json(force=True, silent=True) or {}).get("design_id")
    if not design_id:
        return jsonify({"error": "design_id is required"}), 400
    job = load_job(A2_OUTPUT_DIR, design_id, job_id)
    if job is None:
        return jsonify({"error": f"unknown job {job_id!r} for design {design_id!r}"}), 404
    try:
        job = _kaggle_manager.retry_download(job)
    except KaggleTrainingError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_job_dict_for_client(job))


@app.route("/api/kaggle/jobs/<job_id>/cleanup", methods=["POST"])
def api_kaggle_job_cleanup(job_id: str):
    design_id = request.args.get("design_id") or (request.get_json(force=True, silent=True) or {}).get("design_id")
    if not design_id:
        return jsonify({"error": "design_id is required"}), 400
    job = load_job(A2_OUTPUT_DIR, design_id, job_id)
    if job is None:
        return jsonify({"error": f"unknown job {job_id!r} for design {design_id!r}"}), 404
    try:
        job = _kaggle_manager.cleanup(job)
    except KaggleTrainingError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_job_dict_for_client(job))


if __name__ == "__main__":
    # Keep the local tool safe and single-process by default.  Opt into the
    # Flask debugger/reloader explicitly while developing.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=os.environ.get("FLASK_DEBUG") == "1")

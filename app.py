"""Hybrid NAM Builder -- Flask app entry point.

Experimental proof-of-concept. See README.md for the overall concept and current
limitations. Run with:

    python app.py

then open http://127.0.0.1:5000/ in a browser.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import soundfile as sf
from flask import Flask, Response, jsonify, render_template, request

from hybrid.blend import DEFAULT_TRANSITION_WIDTH_DB, TRANSITION_WIDTH_PRESETS_DB
from hybrid.nam_loader import load_nam
from hybrid.pipeline import RenderedPair, build_hybrid, render_pair
from hybrid.render import NamRenderError
from hybrid.safety import preview_safety_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DI_DIR = BASE_DIR / "assets" / "di"
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# Single-process, single-user local tool (see README) -- a module-level cache
# for the last-rendered amp pair is the whole point of splitting
# render_pair()/build_hybrid() apart: sliders should only ever hit
# build_hybrid() against this, never re-invoke NAM inference.
_rendered_pair_cache: dict = {"pair": None, "amp_a_summary": None, "amp_b_summary": None, "di_file": None}


@app.route("/")
def index():
    di_files = sorted(p.name for p in DI_DIR.glob("*.wav"))
    return render_template(
        "index.html",
        di_files=di_files,
        transition_presets=TRANSITION_WIDTH_PRESETS_DB,
        default_transition_width_db=DEFAULT_TRANSITION_WIDTH_DB,
    )


@app.route("/api/nam/inspect", methods=["POST"])
def api_nam_inspect():
    """Given a server-side path to a .nam file, return its parsed metadata.

    NOTE: for this proof-of-concept, the UI passes a filesystem path (this app
    is meant to run locally next to the user's own .nam files) rather than
    uploading the file, to keep the skeleton small. A real upload flow can
    replace this later without changing hybrid/nam_loader.py.
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
    call this only when Amp A, Amp B, or the DI clip changes, never on a
    crossover/transition/trim slider move (that's /api/preview, against the
    cached RenderedPair below).
    """
    data = request.get_json(force=True)
    amp_a_path = data.get("amp_a_path", "")
    amp_b_path = data.get("amp_b_path", "")
    di_file = data.get("di_file", "")
    if not amp_a_path or not amp_b_path or not di_file:
        return jsonify({"error": "amp_a_path, amp_b_path, and di_file are all required"}), 400

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
        pair = render_pair(amp_a, amp_b, dry, sample_rate)
    except NamRenderError as exc:
        return jsonify({"error": str(exc)}), 500

    _rendered_pair_cache["pair"] = pair
    _rendered_pair_cache["amp_a_summary"] = amp_a.summary()
    _rendered_pair_cache["amp_b_summary"] = amp_b.summary()
    _rendered_pair_cache["di_file"] = di_file

    return jsonify({
        "amp_a": amp_a.summary(),
        "amp_b": amp_b.summary(),
        "di_file": di_file,
        "sample_rate": sample_rate,
        "duration_s": len(dry) / sample_rate,
    })


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Return audio (WAV bytes) for Amp A, Amp B, or the hybrid blend.

    Cheap for `source="hybrid"`: reuses the RenderedPair cached by
    /api/render_pair and calls build_hybrid() only -- no NAM inference here,
    so this is safe to call on every crossover/transition/trim slider move.
    Applies preview_safety_limiter (playback safety net only -- never used on
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
            crossover_dbfs = float(data.get("crossover_dbfs", -22.0))
            transition_width_db = float(data.get("transition_width_db", DEFAULT_TRANSITION_WIDTH_DB))
            manual_b_trim_db = float(data.get("manual_b_trim_db", 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "crossover_dbfs/transition_width_db/manual_b_trim_db must be numbers"}), 400
        auto_level = bool(data.get("auto_level", True))

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
    else:
        return jsonify({"error": f"unknown source: {source!r} (expected a, b, or hybrid)"}), 400

    audio = preview_safety_limiter(audio.astype("float32"))
    buf = io.BytesIO()
    sf.write(buf, audio, pair.sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(buf.read(), mimetype="audio/wav", headers=headers)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate a synthetic hybrid training target. Not functional yet -- depends
    on hybrid.render.render() being implemented first (see that module)."""
    return jsonify({
        "error": "Hybrid target generation depends on NAM rendering, which is not implemented yet. See hybrid/render.py.",
        "implemented": False,
    }), 501


if __name__ == "__main__":
    app.run(debug=True)

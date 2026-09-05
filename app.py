"""Hybrid NAM Builder -- Flask app entry point.

Experimental proof-of-concept. See README.md for the overall concept and current
limitations. Run with:

    python app.py

then open http://127.0.0.1:5000/ in a browser.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from hybrid.blend import DEFAULT_TRANSITION_WIDTH_DB, TRANSITION_WIDTH_PRESETS_DB
from hybrid.nam_loader import load_nam
from hybrid.render import RenderNotImplementedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DI_DIR = BASE_DIR / "assets" / "di"
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


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


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Render Amp A, Amp B, or the hybrid blend against a chosen DI clip.

    Not functional yet: this calls into hybrid.render.render(), which raises
    RenderNotImplementedError until real NAM inference is wired in (see that
    module's docstring). The endpoint exists now so the UI and request/response
    shape are settled ahead of that work.
    """
    try:
        _ = request.get_json(force=True)
        raise RenderNotImplementedError(
            "NAM inference is not wired in yet. See hybrid/render.py."
        )
    except RenderNotImplementedError as exc:
        return jsonify({"error": str(exc), "implemented": False}), 501


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

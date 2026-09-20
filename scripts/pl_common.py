"""Fixed-virtual-gain playability diagnostic (no training). Shared code: teacher (envelope-driven target) rendering for ANALYSIS ONLY, using
the same functions and conventions as scripts/p4e_build.py (which cannot be imported: it runs main() at import).
Import after setting SINGLE_NAM_AMP."""
import json
import numpy as np
from p4e_common import *          # SR, REPO, P4E, MAN, HELD, OFFS, clip, feats2, fixed_T, intended_T, model_path, capture, db, render, tone, harmonics
from hybrid.envelope import bounded_causal_envelope_db
from hybrid.multi_blend import GainChain, chain_weights, multi_blend

REF = -30.0
PL = P4E / "playability"

def audit(amp): return json.loads((REPO / "work" / "p4" / amp / "audit.json").read_text())["captures"]

def render_aligned(amp, g, x):
    """Real capture render with the verified Phase 4A alignment (identical to p4e_build.render_aligned / p4e_real.aligned)."""
    y = render(capture(g), x, SR); c = audit(amp)[f"{g:g}"]["correction"]; sh = -int(c["samples"]) if c and "samples" in c else 0
    return np.concatenate([y[sh:], np.zeros(sh, y.dtype)]) if sh > 0 else (np.concatenate([np.zeros(-sh, y.dtype), y[:sh]]) if sh < 0 else y)

def teacher(amp, x_in, gains, levels_db):
    """Exactly p4e_build.segment: env = causal envelope of the signal that reaches the NAM (x_in); each selected capture k is rendered on
    x_in rescaled by (REF - L_k); weights from the envelope. Returns (teacher waveform in raw target units, envelope dB, weights (N,T))."""
    chain = GainChain(tuple(gains), tuple(levels_db), REF)
    env = bounded_causal_envelope_db(x_in, SR)
    renders = [render_aligned(amp, g, (x_in * db(chain.input_scale_db(k))).astype(np.float32)) for k, g in enumerate(chain.labels)]
    return multi_blend(env, renders, chain), env, chain_weights(env, chain.levels_db)

def config(amp, key):
    """(gains, levels_db, output scale c) of the training target behind a model key (A_s0 / B_s0 / v3_C3)."""
    if key.startswith("v3_"):
        d = REPO / "work" / "cg" / amp / f"cg_{key.split('C')[1]}"
    else:
        d = P4E / amp / f"{key.split('_')[0]}_bundle"
    m = json.loads((d / "manifest.json").read_text()); return m["gains"], m["levels_db"], m["output_scale_c"]

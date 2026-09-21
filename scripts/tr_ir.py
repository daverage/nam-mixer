"""Native NAM vs the same signals through the repo test IR (NOT the user's IR), evaluated exactly in the frequency domain (an IR is LTI, so P_out = P_in * |IR(f)|^2)."""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = "jcm800"
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from tr_common import *
from hybrid.cab_ir import get_prepared_cab_ir
IRP = REPO / "assets" / "nam_models" / "V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav"; prep = get_prepared_cab_ir(str(IRP), SR); ir = np.asarray(prep.samples if hasattr(prep, "samples") else prep.ir, float)
Hp = np.abs(np.fft.rfft(ir, 2 * (NP - 1))[: NP // 2 + 1]) ** 2 if False else None
from scipy.signal import freqz
_, h = freqz(ir, worN=FREQ * 2 * np.pi / SR); Hp = np.abs(h) ** 2
out = {"ir": IRP.name, "note": "repo test IR, not the user's IR"}
np.set_printoptions(linewidth=250, precision=1, suppress=True)
for amp in ("jcm800", "vibrolux"):
    S = Spec(amp, "held"); out[amp] = {}
    for g in S.gains:
        f = lambda H: np.mean([shape_db(S.P(g, d, o, m) * (H if H is not None else 1), S.P(g, d, o, "real") * (H if H is not None else 1))[0][(FC_BANDS >= 80)] for d in S.dis for o in S.offs for m in ("FC_s0", "FC_s1")], axis=0)
        bd = lambda H, k: float(np.mean([broad_db(S.P(g, d, o, m) * (H if H is not None else 1), S.P(g, d, o, "real") * (H if H is not None else 1))[k] for d in S.dis for o in S.offs for m in ("FC_s0", "FC_s1")]))
        out[amp][f"{g:g}"] = {"native_hf_4_10k": bd(None, "high 4-10k"), "IR_hf_4_10k": bd(Hp, "high 4-10k"), "native_mean_abs": float(np.abs(f(None)).mean()), "IR_mean_abs": float(np.abs(f(Hp)).mean()), "native_HFshare": bd(None, "HF>3k share"), "IR_HFshare": bd(Hp, "HF>3k share"), "native_low": bd(None, "low 80-250"), "IR_low": bd(Hp, "low 80-250")}
        if g in (1, 5, 10): print(amp, g, {k: round(v, 2) for k, v in out[amp][f"{g:g}"].items()})
(REPO / "work" / "tr" / "ir_analysis.json").write_text(json.dumps(out, indent=1))

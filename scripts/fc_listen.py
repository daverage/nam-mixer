"""Listening set for the NEW candidate (reuses the ten shortlist clips already heard): REAL reference, and blind candidates v3 C3, B seed 0, FC seed 0, FC seed 1, plus a hidden real copy.
Usage: python scripts/fc_listen.py stage <amp>   (renders; per amp)   |   python scripts/fc_listen.py finalize   (blind page + key)   |   python scripts/fc_listen.py export   (descriptive unblinded folder; use AFTER listening)"""
import csv, hashlib, json, os, random, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
mode_ = sys.argv[1]
if mode_ == "stage": os.environ["SINGLE_NAM_AMP"] = sys.argv[2]
else: os.environ["SINGLE_NAM_AMP"] = "jcm800"
import numpy as np, soundfile as sf
from fc_common import *
import pl_listen_short as S
LGN = FCDIR / "listening"; LGO = P4E / "listening_fixed_gain"; STAGE = LGN / "_stage"; PRE = 10 ** (4.5 / 20); GAP = np.zeros(SR // 2, np.float32)
CANDS = ["v3_C3", "B_s0", "FC_s0", "FC_s1"]
if mode_ == "stage":
    amp = sys.argv[2]; from single_nam_common import render_file_nam
    old = json.loads((LGO / "_stage" / f"stage_{amp}.json").read_text()); di_name = old["di"]; gains, Tanch = FC_CFG[amp]; levels = [t + REF for t in Tanch]; full = clip(di_name)
    models = {k: fc_model_path(amp, k) for k in CANDS}
    def T_for(k, g): return T_of_position(amp, gains, Tanch, g) if k.startswith("FC_") else (intended_T(amp, "B", [g])[0] if k.startswith("B_") else fixed_T(g))
    def render_all(x0, g):
        out = {"REFERENCE": render_aligned(amp, g, x0.astype(np.float32))}
        for k, (nam, c) in models.items(): out[k] = render_file_nam(nam, (x0 * db(T_for(k, g))).astype(np.float32)) / c
        tw, env, w = teacher(amp, (x0 * db(T_for("FC_s0", g))).astype(np.float32), gains, levels); out["TEACHER_FC"] = tw.astype(np.float32); return out, w[:, env > env.max() - 30].mean(axis=1).tolist()
    def write(sid, waves, dry, musical):
        d = STAGE / amp / sid; d.mkdir(parents=True, exist_ok=True); waves = {k: v * PRE for k, v in waves.items()}; ref = waves["REFERENCE"]; info = {}
        for mode in ("native", "levelmatched"):
            w = {k: (v if mode == "native" or k == "REFERENCE" else v * np.sqrt(np.mean(ref[SR // 2:] ** 2) / max(np.mean(v[SR // 2:] ** 2), 1e-20))) for k, v in waves.items()}
            gain = min(1.0, 0.95 / max(float(np.max(np.abs(v))) for v in w.values())); info[mode] = float(gain)
            for k, v in w.items(): sf.write(d / f"{k}__{mode}.wav", np.clip(v * gain, -1, 1).astype(np.float32), SR, subtype="PCM_16")
            if mode == "native":
                for nm, y in (("DI_dry_original", dry), ("DI_musical_input", musical)): sf.write(d / f"{nm}__native.wav", np.clip(y * PRE * gain, -1, 1).astype(np.float32), SR, subtype="PCM_16")
        return info
    info_all = {}
    for (a, g, pk) in S.SHORT:
        if a != amp: continue
        st = next(s for s in old["stimuli"] if s["virtual_gain"] == g and (s["pick"] == pk or (pk.startswith("soft then") and s["kind"] == "sequence"))); s0 = st["window_start_s"]
        if st["kind"] == "sequence":
            segs = [full[int(s0 * SR): int((s0 + 4.0) * SR)] * db(o) for o in (-12.0, 0.0, 6.0)]; parts = [render_all(x.astype(np.float32), g) for x in segs]
            waves = {k: np.concatenate([parts[0][0][k], GAP, parts[1][0][k], GAP, parts[2][0][k]]) for k in parts[0][0]}; wm = np.mean([p[1] for p in parts], axis=0).tolist()
            musical = np.concatenate([segs[0], GAP, segs[1], GAP, segs[2]]).astype(np.float32); dry = full[int(s0 * SR): int((s0 + 4.0) * SR)]
        else:
            x0 = (full[int(s0 * SR): int((s0 + 5.0) * SR)] * db(st["pick_offset_db"])).astype(np.float32); waves, wm = render_all(x0, g); musical = x0; dry = full[int(s0 * SR): int((s0 + 5.0) * SR)]
        info_all[st["sid"]] = {"common_gain": write(st["sid"], waves, dry, musical), "input_gain_db": {k: T_for(k, g) for k in CANDS}, "teacher_weights": dict(zip([f"G{x:g}" for x in gains], wm)), "kind": st["kind"], "duration_s": st["duration_s"], "window_start_s": s0, "pick": st["pick"], "pick_offset_db": st["pick_offset_db"], "di": di_name}
        print(amp, st["sid"], "done", flush=True)
    (STAGE / f"stage_{amp}.json").write_text(json.dumps(info_all, indent=1, default=float))
elif mode_ == "finalize":
    stage = {}
    for amp in ("jcm800", "vibrolux"):
        for sid, v in json.loads((STAGE / f"stage_{amp}.json").read_text()).items(): stage[(amp, sid)] = v
    items = []; rng = random.Random("fc-listening-20260921"); order = list(S.SHORT)  # same order as the shortlist
    key = {}; pub = []; blocks = []
    for b in range(0, len(order), 5):
        base = CANDS + ["HIDDEN_REFERENCE"]; random.Random(f"fc-block-{b}").shuffle(base); rows = [base[r:] + base[:r] for r in range(5)]; random.Random(f"fc-rows-{b}").shuffle(rows); blocks += rows
    LABELS = ["A", "B", "C", "D", "E"]; audio = LGN / "audio"
    for i, (amp, g, pk) in enumerate(order, 1):
        sid_stage = next(sid for (a, sid), v in stage.items() if a == amp and sid.startswith(f"{amp}_G{g:g}_") and (sid.endswith(pk) or (pk.startswith("soft then") and sid.endswith("sequence")))); src = STAGE / amp / sid_stage; sid = f"F{i:02d}"; dst = audio / sid; dst.mkdir(parents=True, exist_ok=True)
        lab = dict(zip(LABELS, blocks[i - 1])); files = {"REFERENCE": {}}
        for mode in ("native", "levelmatched"):
            shutil.copyfile(src / f"REFERENCE__{mode}.wav", dst / f"REFERENCE__{mode}.wav"); files["REFERENCE"][mode] = f"audio/{sid}/REFERENCE__{mode}.wav"
            for L, m in lab.items(): shutil.copyfile(src / f"{'REFERENCE' if m == 'HIDDEN_REFERENCE' else m}__{mode}.wav", dst / f"{L}__{mode}.wav"); files.setdefault(L, {})[mode] = f"audio/{sid}/{L}__{mode}.wav"
        kind = stage[(amp, sid_stage)]["kind"]; title = f"{'Marshall JCM800' if amp == 'jcm800' else 'Fender Vibrolux'}: virtual gain {g:g} (Input gain fixed): " + ("soft, then normal, then hard picking in one clip" if kind == "sequence" else f"{pk} picking")
        instr = ("Listen to the reference (soft, then normal, then hard) and rate how well each candidate keeps that same behaviour. One of A-E is the real amp again." if kind == "sequence" else "Listen to the reference, then A-E, and rate how close each is to the reference. One of A-E is the real amp again.")
        pub.append({"id": sid, "kind": kind, "title": title, "instruction": instr, "labels": LABELS, "files": files}); key[sid] = {"amp": amp, "virtual_gain": g, "pick": pk if kind == "single" else "sequence", "kind": kind, "labels": lab, "stage_id": sid_stage}
    (LGN / "listening_KEY_do_not_open_before_listening.json").write_text(json.dumps(key, indent=1)); (LGN / "stimuli_public.json").write_text(json.dumps(pub, indent=1))
    html = (ROOT / "scripts" / "pl_listening_tool_template.html").read_text().replace("/*STIMULI_JSON*/[]", json.dumps(pub)).replace("/*STORE_KEY*/", "fc_").replace("/*SUBTITLE*/", "New candidate (FC) versus v3 C3 and B: the same ten clips you heard before, new blind labels. Five sounds per item (A-E), one is the real amp.")
    (LGN / "listening_tool_fc.html").write_text(html); print("blind page written:", LGN / "listening_tool_fc.html", len(pub), "items")

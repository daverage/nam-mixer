"""Fixed-virtual-gain listening set, stage 2: shuffle, blind labels + hidden reference, key, worksheet, listening page, protocol and live-guitar guide.
Deterministic (seeded). Usage: pl_listen_finalize.py -> work/p4e/listening_fixed_gain/ (audio, listening_tool.html, KEY) and docs/phase4e/listening_fixed_gain/ (protocol, worksheet, live guide)."""
import csv, hashlib, json, os, random, shutil
from pathlib import Path
os.environ["SINGLE_NAM_AMP"] = "jcm800"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import p4e_common as C
from single_nam_common import capture_path
import importlib

ROOT = Path(__file__).resolve().parent.parent; LG = C.P4E / "listening_fixed_gain"; DOCD = ROOT / "docs" / "phase4e" / "listening_fixed_gain"; DOCD.mkdir(parents=True, exist_ok=True)
NAMES = {"jcm800": "Marshall JCM800", "vibrolux": "Fender Super-Sonic Vibrolux"}
MODELS = ["v3_C3", "B_s0", "B_s1"]; LABELS = ["A", "B", "C", "D"]
rng = random.Random("fixedgain-listening-playlist-20260921")
stim = []
for amp in ("jcm800", "vibrolux"): stim += json.loads((LG / "_stage" / f"stage_{amp}.json").read_text())["stimuli"]
rng.shuffle(stim)
pub, key = [], {}
# balanced (Latin-square) blind labelling: in every block of four items each model/hidden reference appears exactly once at each letter
blocks = []
for b in range(0, len(stim), 4):
    base = MODELS + ["HIDDEN_REFERENCE"]; random.Random(f"fixedgain-block-{b}").shuffle(base)
    rows = [base[r:] + base[:r] for r in range(4)]; random.Random(f"fixedgain-rows-{b}").shuffle(rows); blocks += rows
for i, s in enumerate(stim, 1):
    sid = f"S{i:02d}"; src = LG / "_stage" / s["amp"] / s["sid"]; dst = LG / "audio" / sid; dst.mkdir(parents=True, exist_ok=True)
    lab = dict(zip(LABELS, blocks[i - 1]))
    files = {"REFERENCE": {}}
    for mode in ("native", "levelmatched"):
        shutil.copyfile(src / f"REFERENCE__{mode}.wav", dst / f"REFERENCE__{mode}.wav"); files["REFERENCE"][mode] = f"audio/{sid}/REFERENCE__{mode}.wav"
        for L, m in lab.items():
            srcname = "REFERENCE" if m == "HIDDEN_REFERENCE" else m
            shutil.copyfile(src / f"{srcname}__{mode}.wav", dst / f"{L}__{mode}.wav"); files.setdefault(L, {})[mode] = f"audio/{sid}/{L}__{mode}.wav"
    hidden = [L for L, m in lab.items() if m == "HIDDEN_REFERENCE"][0]
    assert hashlib.sha256((dst / f"{hidden}__native.wav").read_bytes()).hexdigest() == hashlib.sha256((dst / "REFERENCE__native.wav").read_bytes()).hexdigest()
    g = s["virtual_gain"]; seq = s["kind"] == "sequence"
    title = f"{NAMES[s['amp']]}: virtual gain {g:g}" + (" (Input gain fixed): soft, then normal, then hard picking in one clip" if seq else f" (Input gain fixed): {s['pick']} picking")
    instr = ("Listen to the reference: it plays the same phrase soft, then normal, then hard. Notice how its character changes as the player digs in. Then rate each candidate on how well it keeps that same behaviour across the three parts. One of A-D is the real amp again."
             if seq else "Listen to the reference, then to A-D. Rate how close each is to the reference at this playing intensity. One of A-D is the real amp again.")
    pub.append({"id": sid, "kind": s["kind"], "title": title, "instruction": instr, "labels": LABELS, "files": files})
    key[sid] = {"amp": s["amp"], "virtual_gain": g, "pick": s["pick"], "kind": s["kind"], "di": s["di"], "labels": lab, "input_gain_db": s["input_gain_db"], "stage_id": s["sid"]}
(LG / "listening_KEY_do_not_open_before_listening.json").write_text(json.dumps(key, indent=1))
html = (ROOT / "scripts" / "pl_listening_tool_template.html").read_text().replace("/*STIMULI_JSON*/[]", json.dumps(pub)).replace("/*STORE_KEY*/", "").replace("/*SUBTITLE*/", "Full set (32 items).")
(LG / "listening_tool.html").write_text(html)
(LG / "stimuli_public.json").write_text(json.dumps(pub, indent=1))
rows = [{"stimulus": p["id"], "kind": p["kind"], "title": p["title"], **{f"{L} closeness (0-100)": "" for L in LABELS}, "closest (letter)": "", "which sound like a different amp when played harder/softer": "", "notes": ""} for p in pub]
for d in (LG, DOCD):
    with open(d / "listening_worksheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# ---- protocol
(DOCD / "README.md").write_text("""# Fixed-virtual-gain listening: does it still sound like that amp when you play softer or harder?

**Status: PERCEPTUAL EVALUATION PENDING. No human has listened to this set.** Built from the existing Phase 4E models; nothing was trained for it.

## What you are judging
A guitarist selects a virtual amp-gain setting (the player's Input gain), leaves it there, and then plays softly or hard. Does the NAM still sound and feel like THAT amp? Each item fixes the Input gain and compares, against the real amp capture at the same setting, three candidate models (old v3 C3, the new Phase 4E B seed 0, B seed 1) plus a hidden copy of the real amp (a check on attention). Everything is blind: the candidates are lettered A-D at random, differently for every item, and the key is in a separate file that you should not open until you have finished.

- **Single items (24):** the same 5-second phrase played soft (-12 dB), normal, or hard (+6 dB) at the same Input gain. Rate how close each candidate is to the reference.
- **Sequence items (8):** one clip with soft, then normal, then hard playing. Listen to how the REFERENCE changes as the player digs in, then rate each candidate on whether it keeps that same behaviour (a real amp also gets more saturated and compressed when played harder; the failure is a DIFFERENCE from the real amp).
- **Amps and settings:** Vibrolux (clean guitar) at virtual gains 3, 5, 7, 10; JCM800 (driven guitar) at 2, 5, 8, 10. The measured problems were largest on the Vibrolux mid range and small on the JCM800 from G4 upward, so both are included.
- **Two versions of every clip:** *native level* keeps the real loudness differences (judge loudness progression); *level-matched* removes them (judge tone, saturation and dynamics). Use the toggle in the page; ideally hear each item both ways.

## How to run it
1. Open `work/p4e/listening_fixed_gain/listening_tool.html` in a browser (double-click). Audio files are in the same folder. Use good headphones at a fixed volume.
2. Go through the items in order (the order is randomised). Rate each of A-D 0-100, tick anything that sounds wrong, add notes.
3. Click **Export results (CSV)**. Ratings are also kept in your browser between sessions.
4. Analyse with `python scripts/pl_listening_analyze.py <your_csv>`; it opens the key and prints results per candidate, playing intensity and amp, and whether the hidden reference was recognised.
Alternatively fill in `listening_worksheet.csv` by hand.

## Where the audio came from (reproducible)
`scripts/pl_listen_stage.py <amp>` then `scripts/pl_listen_finalize.py` (seeded). Held-out DIs only: `clean_mayer` for the Vibrolux, `moderate_brit` for the JCM800; 5 s single clips and 13 s sequences chosen deterministically as the windows with the widest natural dynamics. Each model is played at its OWN intended mapping: B at the response-distance Input gain, v3 C3 at the fixed rule (see `LIVE_GUITAR_TEST.md` for the numbers). One common gain per item keeps relative levels; PCM 16-bit, 48 kHz, mono.

## Limits
One DI per amp; three playing intensities (-12, 0, +6 dB) are global level changes of a recorded performance, not a real player changing touch; clips show tone, saturation and dynamics but **cannot establish playing feel** (see the live-guitar test). Raw model output for the JCM800 is scaled by a training constant (see the live guide); the clips already undo it.
""")
# ---- live guide
os.environ["SINGLE_NAM_AMP"] = "vibrolux"
def caps(amp):
    os.environ["SINGLE_NAM_AMP"] = amp
    import single_nam_common as sc; importlib.reload(sc)
    return [sc.capture_path(float(g)).name for g in range(1, 11)]
capn = {a: caps(a) for a in NAMES}
def Ts(amp, rule):
    gs = list(range(1, 11)); return [C.intended_T(amp, "B", [float(g)])[0] for g in gs] if rule == "B" else [C.fixed_T(float(g)) for g in gs]
live = ["# Live guitar test: does the NAM stay that amp when you change how you play?\n",
"The clips show tone and dynamics; they cannot show **playing feel**. The closest test of the experience we want to deliver is: guitar into an ordinary NAM player, ONE standard `.nam`, the Input knob fixed, and only your picking, guitar volume knob or pickup changing. Do this after (or alongside) the blind clips. Status of any result: pending until you do it.\n",
"## Setup\n",
"1. Load the model in your NAM player: `docs/phase4e/models/vibrolux_P4E_B_s0.nam` (and, for comparison, `docs/phase4e/models/v3/vibrolux_ContinuousGain_3Captures.nam`; the JCM800 equivalents are `jcm800_P4E_B_s0.nam` and `v3/jcm800_ContinuousGain_3Captures.nam`).",
"2. **Reference amp:** load the original capture for the setting you are testing in a second player instance/slot, with Input gain 0 dB (it IS that physical amp setting). File names are below.",
"3. Match output levels once, and do not touch them again. The Vibrolux models need no output compensation (training constant 1.0). The JCM800 models output about 4.7 dB quieter than the real amp: raise the player's Output gain by +4.7 dB for them.",
"4. Set the model's **Input gain** to the value in the table for the virtual setting, then LEAVE IT. Change only how you play.",
"5. If your player applies its own input/output calibration from the .nam metadata (auto level compensation), either turn it off or make sure it is applied identically to every model and to the reference, so that only the trained behaviour differs.\n",
"## Input gain per virtual setting (dB)\n",
"The two models use different mappings; each is the model's own intended mapping. The official plugin's nominal Input range may stop at about -20 dB, so virtual gain 1 (-22 dB) may not be reachable; use -20 and note it. **Physical reference capture files** are listed for each setting.\n"]
for amp, nm in NAMES.items():
    live += [f"### {nm}\n", "| Virtual gain | Input gain: Phase 4E B (dB) | Input gain: v3 C3 (dB) | Reference capture file |", "|---:|---:|---:|---|"]
    tb, tc = Ts(amp, "B"), Ts(amp, "C3")
    for g in range(1, 11): live.append(f"| {g} | {tb[g-1]:+.1f} | {tc[g-1]:+.1f} | `{capn[amp][g-1]}` |")
    live.append("")
live += ["## What to do, per setting (start with Vibrolux 3, 5, 7 and JCM800 2, 8)\n",
"1. Play the same phrase or riff **softly**, at your **normal** touch, then **hard**. First on the reference capture, then on the B model, then on v3 C3, without touching the Input knob.",
"2. Repeat with your guitar's **volume knob** rolled back a third and then a half (the same idea: quieter musical input at the same Input gain), and with a hotter and a weaker pickup if you have them.",
"3. Notes to make for each: Does it still sound like the SAME amp as you dig in, or does it drift toward a cleaner/dirtier one? Does the **attack** and **bloom** feel like the reference? Does hard picking compress or fizz differently from the reference? Does softer playing clean up the way the reference does? Any clicks or sudden character jumps while dynamics change (crossfade artefacts)?",
"4. Then turn the Input knob deliberately and check the progression from clean to saturated still feels continuous and useful.",
"5. Record a few takes if you can (same phrase, soft/normal/hard) so they can be compared later or added to the blind set.\n",
"## Known limits to keep in mind\n",
"The models are 48 kHz A2 NAMs trained on the level-driven target described in `docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md`; the measured drift is largest on the Vibrolux mid range (G3-G8) and small on the JCM800 from G4 up. Real playing feel also depends on latency, your guitar and the player's Input-gain implementation, none of which the clips capture.\n"]
(DOCD / "LIVE_GUITAR_TEST.md").write_text("\n".join(live))
# v3 C3 models for the live test (copies; originals untouched)
(DOCD.parent / "models" / "v3").mkdir(parents=True, exist_ok=True)
for amp in NAMES:
    nam, _ = C.model_path(amp, "v3_C3"); shutil.copyfile(nam, DOCD.parent / "models" / "v3" / f"{amp}_ContinuousGain_3Captures.nam")
print("stimuli", len(pub), "audio dirs", len(list((LG / "audio").glob("S*"))), "files", len(list((LG / "audio").rglob("*.wav"))))

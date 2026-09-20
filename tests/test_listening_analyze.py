import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
import pl_listening_analyze as L

def _fixture(tmp_path):
    key = {}; rows = []; rng = np.random.default_rng(0)
    truth = {"HIDDEN_REFERENCE": 92, "B_s0": 72, "B_s1": 70, "v3_C3": 45}
    for i in range(1, 9):
        sid = f"S{i:02d}"; order = ["v3_C3", "B_s0", "B_s1", "HIDDEN_REFERENCE"]; rng.shuffle(order); labels = dict(zip("ABCD", order))
        key[sid] = {"amp": "vibrolux" if i % 2 else "jcm800", "virtual_gain": 3.0, "pick": ["soft", "normal", "hard", "soft then normal then hard"][i % 4], "kind": "sequence" if i % 4 == 3 else "single", "labels": labels}
        for lab, m in labels.items():
            rows.append({"listener": "t", "gear": "", "date": "", "stimulus_id": sid, "label": lab, "closeness": truth[m] + int(rng.integers(-3, 4)), "issues": "too dark / dull" if m == "v3_C3" else "", "notes": "", "mode_at_rating": "native"})
    kp = tmp_path / "key.json"; kp.write_text(json.dumps(key)); cp = tmp_path / "r.csv"
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return kp, cp

def test_resolves_labels_and_ranks_models(tmp_path):
    kp, cp = _fixture(tmp_path); a = L.analyse(L.load([cp], json.loads(kp.read_text())))
    m = a["mean_by_model"]; assert m["HIDDEN_REFERENCE"] > m["B_s0"] > m["v3_C3"] and abs(m["B_s0"] - 72) < 3
    assert a["hidden_reference_rated_top_fraction"] == 1.0
    assert a["wins"]["v3_C3"] == 0 and a["wins"]["B_s0"] + a["wins"]["B_s1"] == a["n_items_with_candidates"] == 8
    assert a["paired_B_minus_C3"]["B_better"] == 8 and a["paired_B_minus_C3"]["mean"] > 20
    assert a["issue_tags"]["v3_C3"]["too dark / dull"] == 8 and "B_s0" not in a["issue_tags"]
    assert "Attention check" in L.markdown(a)

def test_ignores_unknown_stimuli_and_labels(tmp_path):
    kp, cp = _fixture(tmp_path); key = json.loads(kp.read_text()); key.pop("S01")
    assert L.analyse(L.load([cp], key))["n_items"] == 7

"""Build the SHORTLIST version of the blind listening page (10 of the 32 items) from the existing audio, key and template. Same blind labels, same ratings CSV format
(scripts/pl_listening_analyze.py works on either page's export). Usage: python scripts/pl_listen_short.py -> work/p4e/listening_fixed_gain/listening_tool_short.html"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; LG = ROOT / "work" / "p4e" / "listening_fixed_gain"
key = json.loads((LG / "listening_KEY_do_not_open_before_listening.json").read_text()); pub = {p["id"]: p for p in json.loads((LG / "stimuli_public.json").read_text())}
# mirrors docs/phase4e/listening_fixed_gain/LIVE_GUITAR_TEST_QUICK.md: (amp, virtual gain, picking) in listening order
SHORT = [("vibrolux", 10.0, "soft"), ("vibrolux", 10.0, "normal"), ("vibrolux", 7.0, "soft"), ("vibrolux", 7.0, "hard"), ("vibrolux", 7.0, "soft then normal then hard"),
         ("jcm800", 8.0, "soft"), ("jcm800", 8.0, "normal"), ("vibrolux", 5.0, "normal"), ("jcm800", 5.0, "normal"), ("jcm800", 5.0, "hard")]
by = {(k["amp"], k["virtual_gain"], k["pick"]): sid for sid, k in key.items()}
items = [pub[by[t]] for t in SHORT]
assert len(items) == 10
html = (ROOT / "scripts" / "pl_listening_tool_template.html").read_text().replace("/*STIMULI_JSON*/[]", json.dumps(items)).replace("/*STORE_KEY*/", "short_").replace("/*SUBTITLE*/", "Shortlist version (10 items, about 15 minutes). Same blind labels as the full set.")
(LG / "listening_tool_short.html").write_text(html); print("wrote", LG / "listening_tool_short.html", len(items), "items")

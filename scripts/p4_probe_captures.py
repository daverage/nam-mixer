"""4A/4B data: probe every real capture (RAW render, no lag correction) -> work/p4/<amp>/captures.json.
Usage: p4_probe_captures.py <amp>"""
import json
import os
import sys
import time

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402

from p4_common import ALL_GAINS, SR, capture, capture_path, out_dir, probe, render  # noqa: E402

amp = sys.argv[1]
res = {"amp": amp, "captures": {}}
t0 = time.time()
for g in ALL_GAINS:
    p = capture_path(g)
    raw = json.load(open(p))
    md = raw.get("metadata", {})
    m = capture(g)
    fn = lambda x, m=m: render(m, x, SR)
    res["captures"][f"{g:g}"] = {"file": p.name, "sha_size": p.stat().st_size,
        "metadata": {k: md.get(k) for k in ("loudness", "gain", "gear_make", "gear_model", "tone_type", "modeled_by", "name", "date")},
        "input_level_dbu": m.input_level_dbu, "output_level_dbu": m.output_level_dbu, "sample_rate": raw.get("sample_rate"),
        "probe": probe(fn)}
    print(f"{amp} G{g:g} done ({time.time()-t0:.0f}s)", flush=True)
(out_dir(amp) / "captures.json").write_text(json.dumps(res))

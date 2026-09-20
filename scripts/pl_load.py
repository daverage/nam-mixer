"""Load the playability case files (no side effects)."""
import glob, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; PL = ROOT / "work" / "p4e" / "playability"
def load_cases(amp):
    cs = []
    for f in sorted(glob.glob(str(PL / f"cases_{amp}_*.json"))): cs += json.loads(Path(f).read_text())["cases"]
    return cs
OFFS = [-12.0, -6.0, 0.0, 6.0]

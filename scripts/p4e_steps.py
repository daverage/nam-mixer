"""Run with .venv-a2: record the final-checkpoint optimiser step count and epoch for every Phase 4E model -> work/p4e/train_steps.json"""
import json, glob
from pathlib import Path
import torch
R = Path(__file__).resolve().parent.parent / "work" / "p4e"; out = {}
for d in sorted(R.glob("*/*_bundle/*_P4E_*_s*")):
    if not d.is_dir(): continue
    cks = sorted(d.glob("lightning_logs/version_*/checkpoints/*.ckpt"), key=lambda p: int(p.name.split("epoch=")[1][:4]))
    if cks:
        c = torch.load(cks[-1], map_location="cpu", weights_only=False); out[d.name] = {"final_epoch": c["epoch"], "global_step": c["global_step"], "ckpt": cks[-1].name}
(R / "train_steps.json").write_text(json.dumps(out, indent=1)); print(out)

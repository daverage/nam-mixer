#!/usr/bin/env python3
"""Measure the embedded-cab package check's error on the real Sequential renderer:
max |Sequential(dry) - IR*(Full head(dry)) * scalar| after warm-up, vs the 3e-6 threshold."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, ".")
from hybrid.core.cab_ir import CabDesign, apply_cab_ir, load_and_prepare_cab_ir
from hybrid.core.nam_loader import load_nam
from hybrid.core.render import SLIM_FULL, find_sequential_nam_render_exe, render
from hybrid.training.embedded_completion import _sequential_warmup_samples
from hybrid.training.sequential_nam import package_embedded_artifacts

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "work/embedded_tolerance"); OUT.mkdir(parents=True, exist_ok=True); SR = 48000
HEAD = Path("docs/history/Continuous Gain/phase4e/models/jcm800_P4E_B_s0.nam")
rng = np.random.default_rng(0)
t = np.arange(SR) / SR
irs = {
    "identity (1 tap)": np.array([1.0]),
    "short (2 taps)": np.array([1.0, 0.25]),
    "real V30 (24001 taps)": sf.read("assets/nam_models/V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav", dtype="float32")[0],
    "synthetic 1 s decay (48000 taps)": rng.standard_normal(SR) * np.exp(-t / 0.12) * 0.05,
    "hot 0.25 s (sum|h| large)": np.abs(rng.standard_normal(SR // 4)) * np.exp(-np.arange(SR // 4) / 2000) * 0.02,
}
dry0, _ = sf.read("assets/di/moderate_brit.wav", dtype="float32"); dry0 = dry0[: 8 * SR]
inputs = {"DI 0 dB": dry0, "DI +12 dB": (dry0 * 10 ** (12 / 20)).astype(np.float32)}
head = render(load_nam(HEAD), dry0, SR, slim=SLIM_FULL)  # warm check
seq_exe = find_sequential_nam_render_exe()
print(f"{'IR':34s} {'input':10s} {'scalar':>7s} {'peak|exp|':>9s} {'max|err|':>10s} {'p99.9|err|':>10s} {'max/peak':>9s}  3e-6")
for name, ir in irs.items():
    ir_path = OUT / (name.split()[0] + ".wav"); sf.write(ir_path, np.asarray(ir, np.float32), SR, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(ir_path, SR)
    cab = CabDesign(selected=True, ir_working_path=str(ir_path), sha256=prepared.sha256, export_mode="embedded", original_filename=ir_path.name)
    for scalar in (1.0, 10 ** (-4 / 20)):
        d = OUT / f"{name.split()[0]}-{scalar:.3f}"
        art = package_embedded_artifacts(HEAD, d, cab, sample_rate=SR, final_scalar=scalar)
        seq = load_nam(art["sequential_nam_path"]); warm = _sequential_warmup_samples(art["sequential_nam_path"])
        for label, dry in inputs.items():
            h = render(load_nam(HEAD), dry, SR, slim=SLIM_FULL)
            expected = apply_cab_ir(h, prepared) * scalar
            actual = render(seq, dry, SR, executable=seq_exe)
            err = np.abs(actual[warm:].astype(np.float64) - expected[warm:])
            peak = float(np.max(np.abs(expected[warm:])))
            flag = "FAIL" if err.max() > 3e-6 else "ok"
            print(f"{name:34s} {label:10s} {scalar:7.3f} {peak:9.3f} {err.max():10.2e} {np.percentile(err, 99.9):10.2e} {err.max()/peak:9.1e}  {flag}")
        if name.startswith("real"):
            h = render(load_nam(HEAD), dry0, SR, slim=SLIM_FULL); actual = render(seq, dry0, SR, executable=seq_exe)
            faults = {
                "fault: IR 1 sample late": apply_cab_ir(h, type(prepared)(**{**prepared.__dict__, "samples": np.concatenate([[0.0], prepared.samples[:-1]]).astype(np.float32)})) * scalar,
                "fault: scalar +0.01 dB": apply_cab_ir(h, prepared) * scalar * 10 ** (0.01 / 20),
                "fault: IR cut at 99.9% energy": apply_cab_ir(h, type(prepared)(**{**prepared.__dict__, "samples": np.where(np.arange(len(prepared.samples)) < prepared.energy_999_samples, prepared.samples, 0).astype(np.float32)})) * scalar,
            }
            for fname, exp in faults.items():
                e = np.abs(actual[warm:].astype(np.float64) - exp[warm:])
                print(f"    {fname:30s} scalar {scalar:.3f}: max|err| {e.max():.2e}  p99.9 {np.percentile(e, 99.9):.2e}")

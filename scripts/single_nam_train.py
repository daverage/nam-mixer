"""Train ONE standard A2 (PackedWaveNet) .nam from a single_nam dataset via the official nam.train.core.train().

Only the data split / input-version detection is patched (custom, non-official input file); architecture,
optimiser, checkpointing and export are the stock official path. Run with .venv-a2.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nam.train.core as core
from nam.train import metadata as md
from nam.train.metadata import TRAINING_KEY

from single_nam_common import OUT


def patch(manifest, ny):
    ts = manifest["train_stop"]

    def data_config(input_version, input_path, output_path, ny, latency):
        return {
            "train": {"ny": ny, "stop_samples": ts},
            "validation": {"ny": None, "start_samples": ts},
            "common": {"x_path": input_path, "y_path": output_path, "delay": latency, "allow_unequal_lengths": True},
            "joint": [],
        }

    core._detect_input_version = lambda p: (core._Version(4, 0, 0), False)
    core._get_data_config = data_config
    core._check_data = lambda *a, **k: md.DataChecks(version=1, passed=True)
    core._analyze_latency = lambda user_latency, *a, **k: md.Latency(
        manual=user_latency,
        calibration=md.LatencyCalibration(
            algorithm_version=0, delays=[], safety_factor=0, recommended=None,
            warnings=md.LatencyCalibrationWarnings(matches_lookahead=False, disagreement_too_high=False, not_detected=True)))
    core._get_final_latency = lambda la: la.manual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    d = OUT / a.dataset
    man = json.loads((d / "manifest.json").read_text())
    patch(man, 8192)
    name = a.name or f"{a.dataset}_e{a.epochs}_s{a.seed}"
    outdir = d / name
    outdir.mkdir(exist_ok=True)
    t0 = time.time()
    res = core.train(str(d / "input.wav"), str(d / "target.wav"), str(outdir), epochs=a.epochs, latency=0,
                     batch_size=16, ny=8192, seed=a.seed, silent=True, modelname="model", ignore_checks=True,
                     fast_dev_run=a.fast)
    assert res is not None and res.model is not None
    res.model.net.export(outdir, basename=name, other_metadata={TRAINING_KEY: res.metadata.model_dump()})
    info = {"name": name, "epochs": a.epochs, "seed": a.seed, "train_seconds": time.time() - t0,
            "validation_esr": res.metadata.validation_esr, "output_scale_c": man["output_scale_c"],
            "n_params": sum(p.numel() for p in res.model.net.parameters())}
    (outdir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    main()

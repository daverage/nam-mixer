"""Feasibility experiment: can ONE small conditional neural network learn

    (input audio, knob position) -> amplifier output

directly from several real gain-sweep captures, generalizing to genuinely
withheld gain positions -- see
docs/CONTINUOUS_GAIN_CONDITIONAL_MODEL.md.

This is deliberately NOT the official `neural-amp-modeler` architecture and
NOT run through `scripts/train_a2.py`. It is a small, self-contained
PyTorch model built only to answer the feasibility question; it must be run
in the SEPARATE training environment (`.venv-a2`, see
requirements-training.txt / CLAUDE.md) because it needs torch, which the
normal Flask runtime environment deliberately does not have.

Architecture (kept intentionally simple -- this is a feasibility check, not
an architecture search):

  - A small causal dilated-convolution stack (WaveNet-family, gated
    activation), FiLM-conditioned on a single scalar: the knob position,
    normalized to [0, 1] over the TRAINING gain range. No hand-designed
    response coordinate, pot-law model, or level/spectral trajectory is
    fed in -- only the raw knob value, so the network has to discover
    whatever nonlinear mapping exists on its own.
  - `channels` residual channels, `layers` dilated conv layers (dilation
    doubling: 1, 2, 4, ..., 2^(layers-1)), kernel size 3.
  - Receptive field = 1 + (kernel_size - 1) * sum(dilations).

Training data: real captures at the TRAINING gains only, rendered through
several DI clips for content diversity (never the WITHHELD gains' captures
-- those are loaded only to produce ground truth for scoring). Evaluation
uses a temporally held-out segment of the "standard" DI clip
(assets/di/moderate_brit.wav) that was NEVER included in any training
crop, on top of the gain positions themselves being held out -- so the
withheld-gain score is not confounded by the model having simply memorized
that exact audio segment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.audio_metrics import rms_dbfs, spectral_magnitude_correlation
from hybrid.continuous_gain import GainCapture, GainCaptureSet, interpolate_output, run_ground_truth_harness
from hybrid.nam_loader import load_nam
from hybrid.render import render
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DI_FILES = [
    _REPO_ROOT / "assets/di/moderate_brit.wav",   # first N seconds only -- see EVAL_HOLDOUT_SECONDS
    _REPO_ROOT / "assets/di/clean_smooth.wav",
    _REPO_ROOT / "assets/di/high_metalcore.wav",
]
EVAL_DI_FILE = _REPO_ROOT / "assets/di/moderate_brit.wav"
TRAIN_SECONDS_PER_CLIP = 12.0
EVAL_HOLDOUT_SECONDS = 6.0  # taken from the END of moderate_brit.wav, never used in any training crop


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class FiLMGatedBlock(nn.Module):
    """One causal dilated-conv residual block with FiLM conditioning from
    the (already-embedded) gain vector, gated (tanh * sigmoid) activation
    -- the standard WaveNet building block, minus anything not needed for
    this feasibility check (no skip-channel/residual-channel split, no
    mixed-precision tricks, no bias-free convs)."""

    def __init__(self, channels: int, cond_dim: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation)
        self.film = nn.Linear(cond_dim, 2 * channels)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, channels, time); cond: (batch, cond_dim)
        padded = nn.functional.pad(x, (self.pad, 0))  # causal: pad only on the left
        y = self.conv(padded)  # (batch, 2*channels, time)
        scale_shift = self.film(cond).unsqueeze(-1)  # (batch, 2*channels, 1)
        y = y + scale_shift
        filt, gate = y.chunk(2, dim=1)
        activated = torch.tanh(filt) * torch.sigmoid(gate)
        residual = x + self.residual_proj(activated)
        skip = self.skip_proj(activated)
        return residual, skip


class ConditionalGainNAM(nn.Module):
    """audio (batch, 1, time), gain (batch, 1) in [0, 1] -> audio (batch, 1, time).

    Deliberately NOT the official NAM WaveNet -- a small, self-contained
    model built only to test whether gain-conditioning is feasible at all.
    """

    def __init__(self, channels: int = 24, cond_dim: int = 24, layers: int = 10, kernel_size: int = 3):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, 1)
        self.cond_mlp = nn.Sequential(nn.Linear(1, cond_dim), nn.ReLU(), nn.Linear(cond_dim, cond_dim))
        self.blocks = nn.ModuleList([
            FiLMGatedBlock(channels, cond_dim, dilation=2 ** i, kernel_size=kernel_size)
            for i in range(layers)
        ])
        self.output_conv1 = nn.Conv1d(channels, channels, 1)
        self.output_conv2 = nn.Conv1d(channels, 1, 1)
        self.dilations = [2 ** i for i in range(layers)]
        self.kernel_size = kernel_size

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, audio: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(audio)
        cond = self.cond_mlp(gain)
        skip_sum = 0.0
        for block in self.blocks:
            x, skip = block(x, cond)
            skip_sum = skip_sum + skip
        out = torch.relu(skip_sum)
        out = torch.relu(self.output_conv1(out))
        return self.output_conv2(out)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def esr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Same ESR definition as hybrid.validation.compute_esr_metrics's raw
    ESR, used as the training loss so training progress is directly
    comparable to the evaluation metric."""
    error_energy = torch.sum((pred - target) ** 2)
    target_energy = torch.sum(target ** 2)
    return error_energy / (target_energy + eps)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _load_di_seconds(path: Path, seconds: float, offset_seconds: float = 0.0) -> tuple[np.ndarray, int]:
    dry, sr = sf.read(path)
    dry = dry.astype(np.float32)
    start = int(offset_seconds * sr)
    end = start + int(seconds * sr)
    return dry[start:end], sr


def build_training_pairs(gains: list[float], load_model, sample_rate_check: int) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Returns [(normalized_gain, dry_chunk, target_chunk), ...] -- one
    entry per (gain, DI clip) combination, real NAM inference throughout."""
    lo, hi = min(gains), max(gains)
    pairs = []
    for clip_path in TRAIN_DI_FILES:
        offset = EVAL_HOLDOUT_SECONDS if clip_path == EVAL_DI_FILE else 0.0
        dry, sr = _load_di_seconds(clip_path, TRAIN_SECONDS_PER_CLIP, offset_seconds=offset)
        assert sr == sample_rate_check, f"{clip_path} sample rate {sr} != {sample_rate_check}"
        for gain in gains:
            target = render(load_model(gain), dry, sr)
            normalized = (gain - lo) / (hi - lo) if hi > lo else 0.0
            pairs.append((normalized, dry, target))
    return pairs


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@dataclass
class TrainingRecord:
    step: int
    loss: float


def train_model(
    model: ConditionalGainNAM, pairs: list[tuple[float, np.ndarray, np.ndarray]],
    *, steps: int, crop_length: int, batch_size: int, lr: float, device: str, seed: int = 0,
) -> list[TrainingRecord]:
    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    min_len = min(len(dry) for _, dry, _ in pairs)
    if crop_length > min_len:
        raise ValueError(f"crop_length={crop_length} exceeds shortest training clip length={min_len}")

    for step in range(steps):
        batch_gain, batch_audio, batch_target = [], [], []
        for _ in range(batch_size):
            gain, dry, target = pairs[rng.integers(0, len(pairs))]
            start = rng.integers(0, len(dry) - crop_length + 1)
            batch_gain.append(gain)
            batch_audio.append(dry[start:start + crop_length])
            batch_target.append(target[start:start + crop_length])

        audio_t = torch.tensor(np.stack(batch_audio), dtype=torch.float32, device=device).unsqueeze(1)
        target_t = torch.tensor(np.stack(batch_target), dtype=torch.float32, device=device).unsqueeze(1)
        gain_t = torch.tensor(batch_gain, dtype=torch.float32, device=device).unsqueeze(1)

        pred = model(audio_t, gain_t)
        loss = esr_loss(pred, target_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            history.append(TrainingRecord(step=step, loss=float(loss.item())))
    return history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def model_infer(model: ConditionalGainNAM, dry: np.ndarray, normalized_gain: float, device: str) -> np.ndarray:
    model.eval()
    audio_t = torch.tensor(dry, dtype=torch.float32, device=device).view(1, 1, -1)
    gain_t = torch.tensor([[normalized_gain]], dtype=torch.float32, device=device)
    out = model(audio_t, gain_t)
    return out.squeeze().cpu().numpy()


def score(reconstructed: np.ndarray, real: np.ndarray) -> dict:
    n = min(len(reconstructed), len(real))
    r, x = reconstructed[:n], real[:n]
    esr = compute_esr_metrics(r, x)
    level_delta_db = abs(rms_dbfs(r) - rms_dbfs(x))
    spectral_corr = spectral_magnitude_correlation(r, x)
    return {
        "raw_esr": esr["raw_esr"],
        "gain_normalized_esr": esr["gain_normalized_esr"],
        "level_delta_db": level_delta_db,
        "spectral_correlation": spectral_corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("amp_dir", type=Path)
    parser.add_argument("--pattern", required=True, help='e.g. "jcm800-high-g{gain}-11.4dBu.nam"')
    parser.add_argument("--train-gains", required=True, help="Comma-separated, e.g. 1,3,5,7,9")
    parser.add_argument("--test-gains", required=True, help="Comma-separated, e.g. 2,4,6,8")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--crop-length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path")
    parser.add_argument("--audio-out-dir", type=Path, default=None, help="If set, write real/model/baseline WAVs for each withheld gain here")
    parser.add_argument("--seed", type=int, default=0, help="Seeds both model weight initialization and batch sampling for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # model weight init is otherwise unseeded and run-to-run results vary noticeably at this model scale
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_gains = [float(g) for g in args.train_gains.split(",")]
    test_gains = [float(g) for g in args.test_gains.split(",")]

    def gain_path(gain: float) -> Path:
        for candidate in (f"{gain:g}", f"{gain:.1f}", f"a{gain:.0f}", f"a{gain:.1f}"):
            path = args.amp_dir / args.pattern.format(gain=candidate)
            if path.exists():
                return path
        raise FileNotFoundError(f"No capture for gain={gain} in {args.amp_dir}")

    model_cache: dict[float, object] = {}

    def load_model(gain: float):
        if gain not in model_cache:
            model_cache[gain] = load_nam(gain_path(gain))
        return model_cache[gain]

    print(f"Rendering training data for gains {train_gains} ...")
    dry_probe, sample_rate = _load_di_seconds(TRAIN_DI_FILES[0], 1.0)
    pairs = build_training_pairs(train_gains, load_model, sample_rate)
    print(f"  {len(pairs)} (gain, DI clip) training pairs, {sum(len(d) for _, d, _ in pairs) / sample_rate:.1f}s total audio")

    model = ConditionalGainNAM(channels=args.channels, layers=args.layers)
    print(f"Model: {args.channels} channels, {args.layers} layers, receptive field {model.receptive_field} samples "
          f"({model.receptive_field / sample_rate * 1000:.1f} ms @ {sample_rate} Hz), {model.n_parameters()} parameters")

    start_time = time.time()
    history = train_model(
        model, pairs, steps=args.steps, crop_length=args.crop_length,
        batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
    )
    train_seconds = time.time() - start_time
    print(f"Training took {train_seconds:.1f}s. Final loss: {history[-1].loss:.4f} (started at {history[0].loss:.4f})")

    # Evaluation audio: the held-out tail of moderate_brit.wav, never seen in any training crop.
    eval_dry, eval_sr = _load_di_seconds(EVAL_DI_FILE, EVAL_HOLDOUT_SECONDS, offset_seconds=0.0)
    assert eval_sr == sample_rate

    lo, hi = min(train_gains), max(train_gains)

    def normalized(gain: float) -> float:
        return (gain - lo) / (hi - lo) if hi > lo else 0.0

    def real_output(gain: float) -> np.ndarray:
        return render(load_model(gain), eval_dry, eval_sr)

    print("\n=== Reconstructing TRAINING gains (sanity check) ===")
    training_results = []
    for gain in train_gains:
        real = real_output(gain)
        model_recon = model_infer(model, eval_dry, normalized(gain), device)
        metrics = score(model_recon, real)
        training_results.append({"gain": gain, **metrics})
        print(f"  Gain {gain:g}: raw_esr={metrics['raw_esr']:.4f}  spectral_corr={metrics['spectral_correlation']:.4f}  level_Δ={metrics['level_delta_db']:.2f}dB")

    print("\n=== Reconstructing WITHHELD gains: conditional model vs knob-linear interpolation ===")
    # Build a GainCaptureSet over the TRAINING gains only, for the baseline.
    training_captures = [GainCapture(model=load_model(g), control_position=g, label=f"G{g:g}") for g in train_gains]
    capture_set = GainCaptureSet(training_captures)
    harness = run_ground_truth_harness(capture_set, eval_dry, eval_sr, calibration_mode="raw")

    withheld_results = []
    for gain in test_gains:
        real = real_output(gain)
        model_recon = model_infer(model, eval_dry, normalized(gain), device)
        model_metrics = score(model_recon, real)

        if lo <= gain <= hi:
            frac = (gain - lo) / (hi - lo)
            baseline_recon = interpolate_output(harness, capture_set, frac)
            baseline_metrics = score(baseline_recon, real)
        else:
            baseline_metrics = {k: float("nan") for k in model_metrics}

        withheld_results.append({"gain": gain, "model": model_metrics, "baseline": baseline_metrics})
        print(f"  Gain {gain:g}:")
        print(f"    conditional model:   raw_esr={model_metrics['raw_esr']:.4f}  spectral_corr={model_metrics['spectral_correlation']:.4f}  level_Δ={model_metrics['level_delta_db']:.2f}dB")
        print(f"    knob-linear interp:  raw_esr={baseline_metrics['raw_esr']:.4f}  spectral_corr={baseline_metrics['spectral_correlation']:.4f}  level_Δ={baseline_metrics['level_delta_db']:.2f}dB")

        if args.audio_out_dir is not None:
            args.audio_out_dir.mkdir(parents=True, exist_ok=True)
            n = min(len(real), len(model_recon))
            sf.write(args.audio_out_dir / f"gain_{gain:g}_real.wav", real[:n], eval_sr)
            sf.write(args.audio_out_dir / f"gain_{gain:g}_model.wav", model_recon[:n], eval_sr)
            if lo <= gain <= hi:
                sf.write(args.audio_out_dir / f"gain_{gain:g}_baseline.wav", baseline_recon[:n], eval_sr)

    report = {
        "amp_dir": str(args.amp_dir), "pattern": args.pattern, "seed": args.seed,
        "train_gains": train_gains, "test_gains": test_gains,
        "model_channels": args.channels, "model_layers": args.layers,
        "model_parameters": model.n_parameters(), "receptive_field_samples": model.receptive_field,
        "sample_rate": sample_rate, "training_seconds_wall_clock": train_seconds,
        "training_steps": args.steps, "training_history": [asdict(h) for h in history],
        "training_gain_reconstruction": training_results,
        "withheld_gain_results": withheld_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()

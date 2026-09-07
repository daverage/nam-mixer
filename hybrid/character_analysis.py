"""Deterministic, cacheable measurements used by Character Blend.

The analysis deliberately measures broad output character rather than trying
to reverse engineer a NAM/circuit.  It is also independent of Flask and NAM
loading, which makes synthetic tests and future dedicated probe renderers
straightforward.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .envelope import bounded_causal_envelope_db

ANALYSIS_VERSION = 1
DEFAULT_LEVELS_DB = (-24.0, -18.0, -12.0, -6.0, 0.0, 6.0)
DEFAULT_FREQUENCIES_HZ = tuple(np.geomspace(80.0, 10_000.0, 24))
_EPS = 1e-10


@dataclass(frozen=True)
class CharacterAnalysisConfig:
    levels_db: tuple[float, ...] = DEFAULT_LEVELS_DB
    frequencies_hz: tuple[float, ...] = DEFAULT_FREQUENCIES_HZ
    level_window_db: float = 3.0
    version: int = ANALYSIS_VERSION

    def cache_key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class AmpLevelAnalysis:
    input_gain_db: float
    input_rms_dbfs: float
    output_rms_dbfs: float
    output_peak_dbfs: float
    compression_gain_db: float
    spectrum_db: tuple[float, ...]


@dataclass(frozen=True)
class AmpCharacterAnalysis:
    sample_rate: int
    frequencies_hz: tuple[float, ...]
    levels: tuple[AmpLevelAnalysis, ...]
    config_hash: str
    source_hash: str = ""
    version: int = ANALYSIS_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AmpCharacterAnalysis":
        return AmpCharacterAnalysis(
            sample_rate=int(data["sample_rate"]), frequencies_hz=tuple(data["frequencies_hz"]),
            levels=tuple(AmpLevelAnalysis(**{**item, "spectrum_db": tuple(item["spectrum_db"])}) for item in data["levels"]),
            config_hash=data["config_hash"], source_hash=data.get("source_hash", ""), version=int(data.get("version", 1)),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db_rms(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), _EPS))) if len(audio) else -120.0


def _spectrum(audio: np.ndarray, sample_rate: int, frequencies: tuple[float, ...]) -> tuple[float, ...]:
    # A deterministic Hann-windowed measurement.  Interpolation occurs in
    # log-frequency/dB space, intentionally discarding narrow phase detail.
    if len(audio) < 8:
        return tuple([-120.0] * len(frequencies))
    n = min(len(audio), max(256, int(sample_rate * 0.5)))
    x = np.asarray(audio[:n], dtype=np.float64) * np.hanning(n)
    mags = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(x)), _EPS))
    bins = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return tuple(float(v) for v in np.interp(np.log(frequencies), np.log(np.maximum(bins, 1.0)), mags))


def analyse_rendered_audio(
    dry: np.ndarray, rendered: np.ndarray, sample_rate: int,
    config: CharacterAnalysisConfig = CharacterAnalysisConfig(), source_hash: str = "",
) -> AmpCharacterAnalysis:
    """Measure a rendered source against the same dry material at each level.

    When a preview DI does not contain enough samples near a requested level,
    the nearest samples are used.  This keeps the design deterministic and
    makes the limitation visible in provenance rather than inventing signal.
    """
    n = min(len(dry), len(rendered))
    dry, rendered = np.asarray(dry[:n], dtype=np.float64), np.asarray(rendered[:n], dtype=np.float64)
    envelope = bounded_causal_envelope_db(dry, sample_rate)
    levels: list[AmpLevelAnalysis] = []
    for level in config.levels_db:
        mask = np.abs(envelope - level) <= config.level_window_db
        if mask.sum() < 64:
            # Stable nearest-level fallback, bounded so a pathological input
            # still has a well-defined analysis result.
            count = min(max(64, n // 32), n)
            idx = np.argpartition(np.abs(envelope - level), count - 1)[:count] if n else np.array([], dtype=int)
            mask = np.zeros(n, dtype=bool); mask[idx] = True
        x, y = dry[mask], rendered[mask]
        input_rms, output_rms = _db_rms(x), _db_rms(y)
        levels.append(AmpLevelAnalysis(
            input_gain_db=float(level), input_rms_dbfs=input_rms, output_rms_dbfs=output_rms,
            output_peak_dbfs=float(20.0 * np.log10(max(float(np.max(np.abs(y))) if len(y) else 0.0, _EPS))),
            compression_gain_db=output_rms - input_rms, spectrum_db=_spectrum(y, sample_rate, config.frequencies_hz),
        ))
    return AmpCharacterAnalysis(sample_rate, config.frequencies_hz, tuple(levels), config.cache_key(), source_hash, config.version)


def analysis_cache_path(cache_dir: str | Path, source_hash: str, config: CharacterAnalysisConfig) -> Path:
    return Path(cache_dir) / f"character-analysis-{source_hash}-{config.cache_key()}.json"


def load_cached_analysis(cache_dir: str | Path, source_hash: str, config: CharacterAnalysisConfig) -> AmpCharacterAnalysis | None:
    path = analysis_cache_path(cache_dir, source_hash, config)
    if not path.is_file(): return None
    return AmpCharacterAnalysis.from_dict(json.loads(path.read_text(encoding="utf-8")))


def store_cached_analysis(cache_dir: str | Path, analysis: AmpCharacterAnalysis) -> Path:
    path = analysis_cache_path(cache_dir, analysis.source_hash, CharacterAnalysisConfig(
        levels_db=tuple(level.input_gain_db for level in analysis.levels), frequencies_hz=analysis.frequencies_hz,
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    return path

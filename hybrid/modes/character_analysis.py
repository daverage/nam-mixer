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

from ..core.envelope import bounded_causal_envelope_db

# Analysis method versions (recorded in every analysis and frozen config):
#   1 -- per-level spectrum from the first 0.5 s of the (non-contiguous)
#        samples near that level, stitched together. The seams between
#        stitched runs add broadband/HF energy that is not in the audio.
#   2 -- per-level spectrum averaged over real contiguous frames whose level
#        is near that level (see _contiguous_level_spectrum). No seams.
# Version 2 is the default since 2026-09-24 (a listening comparison found no
# audible preference and it removes a known measurement error). Frozen designs
# always keep the version they were analysed with; a stored analysis without
# a recorded version is version 1 (AmpCharacterAnalysis.from_dict).
ANALYSIS_VERSION = 2
SUPPORTED_ANALYSIS_VERSIONS = (1, 2)
SPECTRUM_FRAME_SAMPLES = 4096
SPECTRUM_MIN_FRAMES = 4
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
    # The config's level window, so a frozen design can record the config
    # that produced this analysis. Analyses stored before this field existed
    # were all measured with the default window.
    level_window_db: float = CharacterAnalysisConfig.level_window_db

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AmpCharacterAnalysis":
        return AmpCharacterAnalysis(
            sample_rate=int(data["sample_rate"]), frequencies_hz=tuple(data["frequencies_hz"]),
            levels=tuple(AmpLevelAnalysis(**{**item, "spectrum_db": tuple(item["spectrum_db"])}) for item in data["levels"]),
            config_hash=data["config_hash"], source_hash=data.get("source_hash", ""), version=int(data.get("version", 1)),
            level_window_db=float(data.get("level_window_db", CharacterAnalysisConfig.level_window_db)),
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


def _contiguous_level_spectrum(rendered: np.ndarray, envelope: np.ndarray, level_db: float, window_db: float,
                               sample_rate: int, frequencies: tuple[float, ...]) -> tuple[float, ...]:
    """Version-2 spectrum: average the Hann-windowed power spectra of real,
    contiguous frames whose median dry-envelope level is within `window_db`
    of `level_db` (or, if fewer than SPECTRUM_MIN_FRAMES qualify, the frames
    nearest that level). Each frame is an unbroken stretch of audio, so no
    splice discontinuities enter the measurement."""
    n = len(rendered)
    if n < 8:
        return tuple([-120.0] * len(frequencies))
    frame = min(SPECTRUM_FRAME_SAMPLES, n)
    hop = max(1, frame // 2)
    starts = np.arange(0, n - frame + 1, hop)
    frame_levels = np.median(np.lib.stride_tricks.sliding_window_view(envelope[:n], frame)[::hop], axis=1)
    distance = np.abs(frame_levels - level_db)
    chosen = np.flatnonzero(distance <= window_db)
    if len(chosen) < SPECTRUM_MIN_FRAMES:
        chosen = np.argsort(distance, kind="stable")[: min(SPECTRUM_MIN_FRAMES, len(starts))]
    window = np.hanning(frame)
    power = np.zeros(frame // 2 + 1)
    for start in starts[np.sort(chosen)]:
        power += np.abs(np.fft.rfft(np.asarray(rendered[start:start + frame], dtype=np.float64) * window)) ** 2
    mags = 10.0 * np.log10(np.maximum(power / len(chosen), _EPS ** 2))
    bins = np.fft.rfftfreq(frame, 1.0 / sample_rate)
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
    if config.version not in SUPPORTED_ANALYSIS_VERSIONS:
        raise ValueError(f"unsupported Character analysis version {config.version}; supported: {SUPPORTED_ANALYSIS_VERSIONS}")
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
        if config.version == 1:
            spectrum = _spectrum(y, sample_rate, config.frequencies_hz)
        else:
            spectrum = _contiguous_level_spectrum(rendered, envelope, level, config.level_window_db,
                                                  sample_rate, config.frequencies_hz)
        levels.append(AmpLevelAnalysis(
            input_gain_db=float(level), input_rms_dbfs=input_rms, output_rms_dbfs=output_rms,
            output_peak_dbfs=float(20.0 * np.log10(max(float(np.max(np.abs(y))) if len(y) else 0.0, _EPS))),
            compression_gain_db=output_rms - input_rms, spectrum_db=spectrum,
        ))
    return AmpCharacterAnalysis(sample_rate, config.frequencies_hz, tuple(levels), config.cache_key(), source_hash, config.version,
                                config.level_window_db)


def analysis_cache_key(source_hash: str, dry: np.ndarray, rendered: np.ndarray) -> str:
    """Key for everything an analysis depends on: the source .nam plus the
    exact dry and rendered audio measured. The DI, profile/test gain,
    calibration and per-amp input gain all show up in those two arrays, so
    changing any of them can never serve a stale analysis."""
    digest = hashlib.sha256(source_hash.encode("utf-8"))
    for audio in (dry, rendered):
        digest.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
    return digest.hexdigest()


def analysis_cache_path(cache_dir: str | Path, cache_key: str, config: CharacterAnalysisConfig) -> Path:
    return Path(cache_dir) / f"character-analysis-{cache_key}-{config.cache_key()}.json"


def load_cached_analysis(cache_dir: str | Path, cache_key: str, config: CharacterAnalysisConfig) -> AmpCharacterAnalysis | None:
    path = analysis_cache_path(cache_dir, cache_key, config)
    if not path.is_file(): return None
    return AmpCharacterAnalysis.from_dict(json.loads(path.read_text(encoding="utf-8")))


def store_cached_analysis(cache_dir: str | Path, analysis: AmpCharacterAnalysis, config: CharacterAnalysisConfig, cache_key: str) -> Path:
    path = analysis_cache_path(cache_dir, cache_key, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    return path

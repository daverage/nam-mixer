"""Phase-safety measurements for the actual weighted Parallel Blend sum.

This deliberately does not try to decide whether two different amp waveforms
are "the same".  It measures the useful product question instead: when the two
already-weighted components are added, do bands where both are materially
present lose persistent energy through destructive interaction?
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


FRAME_SIZE = 2048
MAX_ANALYSIS_FRAMES = 96
ACTIVE_FRAME_RANGE_DB = 40.0
MATERIAL_BAND_RANGE_DB = 45.0
MATERIAL_COMPONENT_BALANCE_DB = -18.0
PRESENCE_LIMIT_DB = 18.0
CAUTION_LOSS_DB = -2.0
PROBLEM_LOSS_DB = -4.0
POLARITY_IMPROVEMENT_DB = 3.0

# Broad enough to be stable across notes, narrow enough to name a useful area.
_BANDS = (
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 12000.0),
)


@dataclass(frozen=True)
class BandInteraction:
    low_hz: int
    high_hz: int
    interaction_db: float
    component_balance_db: float
    material: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParallelCompatibility:
    status: str
    summary: str
    presence_status: str
    presence_summary: str
    amp_a_weighted_dbfs: float | None
    amp_b_weighted_dbfs: float | None
    worst_band: BandInteraction | None
    bands: tuple[BandInteraction, ...]
    frames_analysed: int
    polarity_inverted: bool
    polarity_recommended: bool = False
    recommended_polarity: str | None = None
    alternative_status: str | None = None
    alternative_worst_interaction_db: float | None = None
    method: str = "parallel-band-interaction-v1"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bands"] = [band.to_dict() for band in self.bands]
        data["worst_band"] = self.worst_band.to_dict() if self.worst_band else None
        return data


def _db(value: float) -> float:
    return 10.0 * np.log10(max(float(value), 1e-20))


def _rms_dbfs(audio: np.ndarray) -> float | None:
    if len(audio) == 0:
        return None
    power = float(np.mean(np.square(np.asarray(audio, dtype=np.float64))))
    return round(_db(power), 2) if power > 1e-20 else None


def _frames(audio: np.ndarray, starts: np.ndarray, window: np.ndarray) -> np.ndarray:
    return np.stack([audio[start:start + len(window)] * window for start in starts])


def _format_band(band: BandInteraction) -> str:
    if band.high_hz >= 1000:
        low = f"{band.low_hz / 1000:g} kHz" if band.low_hz >= 1000 else f"{band.low_hz} Hz"
        high = f"{band.high_hz / 1000:g} kHz"
        return f"{low}–{high}"
    return f"{band.low_hz}–{band.high_hz} Hz"


def _presence(a: np.ndarray, b: np.ndarray) -> tuple[str, str, float | None, float | None]:
    a_db = _rms_dbfs(a)
    b_db = _rms_dbfs(b)
    if a_db is None and b_db is None:
        return "insufficient_signal", "There is not enough signal to assess whether both amps are present.", a_db, b_db
    if a_db is None:
        return "amp_a_absent", "Amp A is absent at this mix setting.", a_db, b_db
    if b_db is None:
        return "amp_b_absent", "Amp B is absent at this mix setting.", a_db, b_db
    delta = a_db - b_db
    if delta > PRESENCE_LIMIT_DB:
        return "amp_b_obscured", f"Amp B is {abs(delta):.1f} dB below Amp A and may be hard to hear.", a_db, b_db
    if delta < -PRESENCE_LIMIT_DB:
        return "amp_a_obscured", f"Amp A is {abs(delta):.1f} dB below Amp B and may be hard to hear.", a_db, b_db
    return "both_present", f"Both amps have material level in this mix ({abs(delta):.1f} dB apart).", a_db, b_db


def analyse_parallel_compatibility(
    amp_a_component: np.ndarray,
    amp_b_component: np.ndarray,
    sample_rate: int,
    *,
    polarity_inverted: bool = False,
) -> ParallelCompatibility:
    """Assess cancellation and presence in two already-weighted components."""
    n = min(len(amp_a_component), len(amp_b_component))
    a = np.asarray(amp_a_component[:n], dtype=np.float64)
    b = np.asarray(amp_b_component[:n], dtype=np.float64)
    presence_status, presence_summary, a_db, b_db = _presence(a, b)

    frame_size = min(FRAME_SIZE, n)
    if frame_size < 128 or sample_rate <= 0:
        return ParallelCompatibility(
            status="insufficient_signal", summary="Not enough audio to assess parallel cancellation.",
            presence_status=presence_status, presence_summary=presence_summary,
            amp_a_weighted_dbfs=a_db, amp_b_weighted_dbfs=b_db, worst_band=None,
            bands=(), frames_analysed=0, polarity_inverted=polarity_inverted,
        )

    hop = max(1, frame_size // 2)
    starts = np.arange(0, n - frame_size + 1, hop, dtype=int)
    window = np.hanning(frame_size)
    frame_power = np.array([
        np.mean(np.square(a[start:start + frame_size]) + np.square(b[start:start + frame_size]))
        for start in starts
    ])
    if not len(frame_power) or float(frame_power.max(initial=0.0)) <= 1e-20:
        return ParallelCompatibility(
            status="insufficient_signal", summary="Not enough active playing to assess parallel cancellation.",
            presence_status=presence_status, presence_summary=presence_summary,
            amp_a_weighted_dbfs=a_db, amp_b_weighted_dbfs=b_db, worst_band=None,
            bands=(), frames_analysed=0, polarity_inverted=polarity_inverted,
        )
    active = starts[frame_power >= frame_power.max() * (10.0 ** (-ACTIVE_FRAME_RANGE_DB / 10.0))]
    if len(active) > MAX_ANALYSIS_FRAMES:
        active = active[np.linspace(0, len(active) - 1, MAX_ANALYSIS_FRAMES, dtype=int)]

    spectrum_a = np.fft.rfft(_frames(a, active, window), axis=1)
    spectrum_b = np.fft.rfft(_frames(b, active, window), axis=1)
    power_a = np.sum(np.abs(spectrum_a) ** 2, axis=0)
    power_b = np.sum(np.abs(spectrum_b) ** 2, axis=0)
    power_mix = np.sum(np.abs(spectrum_a + spectrum_b) ** 2, axis=0)
    reference = power_a + power_b
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    max_reference = float(reference.max(initial=0.0))

    bands: list[BandInteraction] = []
    nyquist = sample_rate / 2.0
    for low, high in _BANDS:
        if low >= nyquist:
            continue
        high = min(high, nyquist)
        indices = (frequencies >= low) & (frequencies < high)
        if not np.any(indices):
            continue
        pa = float(power_a[indices].sum())
        pb = float(power_b[indices].sum())
        pmix = float(power_mix[indices].sum())
        pref = pa + pb
        balance = _db(min(pa, pb) / max(pa, pb)) if min(pa, pb) > 1e-20 else -200.0
        energetic = pref >= max_reference * (10.0 ** (-MATERIAL_BAND_RANGE_DB / 10.0))
        material = bool(energetic and balance >= MATERIAL_COMPONENT_BALANCE_DB)
        bands.append(BandInteraction(
            low_hz=int(round(low)), high_hz=int(round(high)),
            interaction_db=round(_db(pmix / pref), 2) if pref > 1e-20 else 0.0,
            component_balance_db=round(balance, 2), material=material,
        ))

    scored = [band for band in bands if band.material]
    worst = min(scored, key=lambda band: band.interaction_db) if scored else None
    if not scored:
        status = "insufficient_overlap"
        summary = "The amps do not share enough energy in the same bands to assess cancellation; no phase fault was identified."
    elif worst.interaction_db <= PROBLEM_LOSS_DB:
        status = "problem"
        summary = f"Strong repeatable cancellation is reducing {_format_band(worst)} by {abs(worst.interaction_db):.1f} dB on this performance."
    elif worst.interaction_db <= CAUTION_LOSS_DB:
        status = "colouration"
        summary = f"Some phase coloration is present around {_format_band(worst)} ({abs(worst.interaction_db):.1f} dB reduction), but no severe cancellation was found."
    else:
        status = "safe"
        summary = "No strong repeatable cancellation was detected in the actual parallel sum on this performance."

    return ParallelCompatibility(
        status=status, summary=summary, presence_status=presence_status,
        presence_summary=presence_summary, amp_a_weighted_dbfs=a_db,
        amp_b_weighted_dbfs=b_db, worst_band=worst, bands=tuple(bands),
        frames_analysed=len(active), polarity_inverted=polarity_inverted,
    )


def analyse_parallel_with_polarity_choice(
    amp_a_component: np.ndarray,
    amp_b_component_original: np.ndarray,
    sample_rate: int,
    *,
    polarity_inverted: bool = False,
) -> ParallelCompatibility:
    """Analyse the selected polarity and decide whether the other is a clear fix."""
    original = analyse_parallel_compatibility(amp_a_component, amp_b_component_original, sample_rate)
    inverted = analyse_parallel_compatibility(amp_a_component, -amp_b_component_original, sample_rate, polarity_inverted=True)
    selected, alternative = (inverted, original) if polarity_inverted else (original, inverted)

    original_worst = original.worst_band.interaction_db if original.worst_band else None
    inverted_worst = inverted.worst_band.interaction_db if inverted.worst_band else None
    recommend_inverted = bool(
        original.status == "problem" and original_worst is not None and inverted_worst is not None
        and inverted_worst - original_worst >= POLARITY_IMPROVEMENT_DB
        and inverted.status != "problem"
    )
    recommend_original = bool(
        inverted.status == "problem" and original_worst is not None and inverted_worst is not None
        and original_worst - inverted_worst >= POLARITY_IMPROVEMENT_DB
        and original.status != "problem"
    )
    data = selected.to_dict()
    data.update(
        polarity_recommended=recommend_inverted,
        recommended_polarity="inverted" if recommend_inverted else "original" if recommend_original else None,
        alternative_status=alternative.status,
        alternative_worst_interaction_db=(alternative.worst_band.interaction_db if alternative.worst_band else None),
    )
    data["bands"] = tuple(BandInteraction(**band) for band in data["bands"])
    data["worst_band"] = BandInteraction(**data["worst_band"]) if data["worst_band"] else None
    return ParallelCompatibility(**data)

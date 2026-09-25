"""Multi-window A/B timing diagnostic -- analysis only, never alters audio.

`hybrid.core.align.estimate_offset` answers "which shift maximises the
similarity of these two signals?". For two tonally different amp renders that
is NOT the same question as "is there a fixed time offset between them?":
different filtering, distortion and compression move the correlation peak
without any latency being involved, and one whole-render peak cannot tell the
two apart.

`analyse_alignment` asks the second question instead. It measures the offset
independently in several separate, transient-rich regions of the shared DI
render, and only reports a fixed offset when those independent measurements
agree. A genuine fixed latency shifts every region by the same amount; a
tonal/phase difference produces region-dependent (material-dependent)
"offsets", which are reported as `ambiguous` and never as a correction.

Offset sign follows `estimate_offset(amp_a, amp_b)`: positive means Amp B
lags (arrives later than) Amp A.

This module does not correct anything. `align_to_reference` stays opt-in and
the app keeps passing `enabled=False`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .align import estimate_offset

# The whole-render estimate_offset default (2000 samples) is a search range,
# not a plausible latency. A NAM model's own latency difference is at most a
# few milliseconds, so anything outside this range is not a fixed offset
# this diagnostic should recommend. 256 samples = 5.3 ms at 48 kHz.
DIAGNOSTIC_MAX_LAG_SAMPLES = 256

WINDOW_SECONDS = 0.25         # length of each measured region
ONSET_PREROLL_SECONDS = 0.01  # each region starts just before its transient
FRAME_SECONDS = 0.01          # hop of the dry-signal energy frames used to find transients
ONSET_LOOKBACK_FRAMES = 3     # a frame's onset strength = its dB minus the loudest of these previous frames
MIN_ONSET_RISE_DB = 6.0       # only real transients start a region, not sustained tails
MAX_WINDOWS = 12
MIN_WINDOWS = 4               # fewer usable regions than this -> not enough evidence

MIN_DRY_RMS_DBFS = -50.0      # region must carry real playing in the DI...
MIN_AMP_RMS_DBFS = -60.0      # ...and in both amp renders

# A region's measurement only counts when the two renders actually resemble
# each other at the chosen lag, and the peak is not pinned to the edge of the
# search range (then the real maximum is outside it, or there is none).
#
# Deliberately high. Measured on real captures (docs/alignment_diagnostic.md):
# pairs whose regions correlate at only ~0.5-0.8 (Deluxe Reverb vs Twin Reverb,
# an amp vs the same amp with its cab) can still agree region-to-region on
# one lag, but that lag moves with the DI material (-3, -2, -7 samples on three
# clips), which a real latency cannot do: it is the phase response of two
# different tones, not a delay. Pairs close enough in tone for a sample offset
# to mean anything correlate at >= 0.95 on every clip tried.
MIN_WINDOW_CORRELATION = 0.9

AGREEMENT_TOLERANCE_SAMPLES = 1  # |offset - median| within this -> the region agrees
MIN_AGREEMENT_FRACTION = 0.8     # agreeing regions / signal-bearing regions
ZERO_OFFSET_TOLERANCE_SAMPLES = 1  # |median| within this -> "aligned" (21 us at 48 kHz)

STATUS_ALIGNED = "aligned"
STATUS_FIXED_OFFSET = "fixed_offset"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_INSUFFICIENT_SIGNAL = "insufficient_signal"


@dataclass(frozen=True)
class WindowMeasurement:
    start_sample: int
    length_samples: int
    offset_samples: int
    correlation: float    # normalized A/B correlation at offset_samples
    at_search_edge: bool  # |offset| == max lag: the true peak may lie outside the range
    reliable: bool        # counts towards agreement


@dataclass(frozen=True)
class AlignmentDiagnostic:
    status: str
    recommended_offset_samples: int  # 0 unless status == "fixed_offset"
    agreement_fraction: float        # agreeing regions / signal-bearing regions (0 when none)
    median_correlation: float | None # over reliable regions
    windows_analysed: int
    windows_reliable: int
    max_lag_samples: int
    sample_rate: int
    reason: str
    windows: list[WindowMeasurement] = field(default_factory=list)
    # The old single whole-render estimate (estimate_offset's own default
    # max lag) -- reported for comparison only, never used for a decision.
    whole_render_offset_samples: int | None = None

    @property
    def per_window_offsets(self) -> list[int]:
        return [w.offset_samples for w in self.windows]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["per_window_offsets"] = self.per_window_offsets
        return data


def _rms_dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def _correlation_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Normalized correlation of b against a at `lag`, scored exactly the way
    estimate_offset scores a lag (mean-removed, overlap only)."""
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    x, y = (a[: n - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: n + lag])
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom >= 1e-12 else 0.0


def _round_half_toward_zero(x: float) -> int:
    """A median exactly between two integer lags (e.g. six regions at -1, six
    at -2) recommends the SMALLER shift, never the larger one."""
    return int(np.sign(x) * np.ceil(abs(x) - 0.5))


def select_windows(dry: np.ndarray, sample_rate: int, *, max_windows: int = MAX_WINDOWS) -> list[tuple[int, int]]:
    """Deterministically pick up to `max_windows` non-overlapping regions,
    each starting just before one of the DI's strongest note onsets.

    Onset strength is the rise of a 10 ms frame's level over the loudest of
    the preceding frames; ties break on frame level, then position. Returns
    sorted (start, length) pairs."""
    dry = np.asarray(dry, dtype=np.float64)
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    length = int(round(WINDOW_SECONDS * sample_rate))
    preroll = int(round(ONSET_PREROLL_SECONDS * sample_rate))
    n_frames = len(dry) // frame
    if length <= 0 or len(dry) < length or n_frames == 0:
        return []
    power = np.mean(np.square(dry[: n_frames * frame]).reshape(n_frames, frame), axis=1)
    level_db = 10.0 * np.log10(np.maximum(power, 1e-20))
    previous = np.full(n_frames, -200.0)
    for k in range(1, ONSET_LOOKBACK_FRAMES + 1):
        previous[k:] = np.maximum(previous[k:], level_db[:-k])
    rise_db = level_db - previous

    candidates = [i for i in range(n_frames) if level_db[i] >= MIN_DRY_RMS_DBFS and rise_db[i] >= MIN_ONSET_RISE_DB]
    candidates.sort(key=lambda i: (-rise_db[i], -level_db[i], i))
    chosen: list[int] = []
    for i in candidates:
        start = max(0, i * frame - preroll)
        if start + length > len(dry):
            continue
        if all(abs(start - other) >= length for other in chosen):
            chosen.append(start)
            if len(chosen) == max_windows:
                break
    return [(start, length) for start in sorted(chosen)]


def analyse_alignment(
    amp_a: np.ndarray,
    amp_b: np.ndarray,
    dry: np.ndarray,
    sample_rate: int,
    *,
    max_lag: int = DIAGNOSTIC_MAX_LAG_SAMPLES,
    include_whole_render: bool = True,
) -> AlignmentDiagnostic:
    """Is there strong evidence of ONE stable time offset between Amp A and
    Amp B? Read-only: the arrays passed in are never modified.

    `dry` is the signal both amps received (the pair's profiled DI); it only
    chooses where to measure.
    """
    amp_a = np.asarray(amp_a)
    amp_b = np.asarray(amp_b)
    n = min(len(amp_a), len(amp_b), len(dry))
    whole = int(estimate_offset(amp_a[:n], amp_b[:n])) if include_whole_render and n > 1 else None

    windows: list[WindowMeasurement] = []
    for start, length in select_windows(np.asarray(dry)[:n], sample_rate):
        a = amp_a[start:start + length]
        b = amp_b[start:start + length]
        if _rms_dbfs(a) < MIN_AMP_RMS_DBFS or _rms_dbfs(b) < MIN_AMP_RMS_DBFS:
            continue
        lag = min(max_lag, length - 2)
        offset = int(estimate_offset(a, b, max_lag=lag))
        correlation = _correlation_at(a, b, offset)
        at_edge = lag > 0 and abs(offset) >= lag
        windows.append(WindowMeasurement(
            start_sample=start, length_samples=length, offset_samples=offset,
            correlation=round(correlation, 4), at_search_edge=at_edge,
            reliable=(not at_edge) and correlation >= MIN_WINDOW_CORRELATION,
        ))

    def result(status: str, reason: str, *, recommended: int = 0, agreement: float = 0.0) -> AlignmentDiagnostic:
        reliable = [w for w in windows if w.reliable]
        return AlignmentDiagnostic(
            status=status, recommended_offset_samples=recommended,
            agreement_fraction=round(agreement, 4),
            median_correlation=round(float(np.median([w.correlation for w in reliable])), 4) if reliable else None,
            windows_analysed=len(windows), windows_reliable=len(reliable),
            max_lag_samples=max_lag, sample_rate=sample_rate, reason=reason,
            windows=windows, whole_render_offset_samples=whole,
        )

    if len(windows) < MIN_WINDOWS:
        return result(STATUS_INSUFFICIENT_SIGNAL, (
            f"Only {len(windows)} region(s) with enough playing in the DI and both renders "
            f"(need {MIN_WINDOWS}); timing cannot be assessed on this material."))

    reliable = [w for w in windows if w.reliable]
    if len(reliable) < MIN_WINDOWS:
        return result(STATUS_AMBIGUOUS, (
            f"Only {len(reliable)} of {len(windows)} regions gave a clear timing match "
            f"(correlation >= {MIN_WINDOW_CORRELATION} inside +/-{max_lag} samples). The amps differ "
            "too much in tone/phase for a sample shift to be meaningful; no correction recommended."))

    median = _round_half_toward_zero(float(np.median([w.offset_samples for w in reliable])))
    agreeing = [w for w in reliable if abs(w.offset_samples - median) <= AGREEMENT_TOLERANCE_SAMPLES]
    agreement = len(agreeing) / len(windows)
    if agreement < MIN_AGREEMENT_FRACTION:
        offsets = sorted({w.offset_samples for w in windows})
        return result(STATUS_AMBIGUOUS, (
            f"Regions disagree: {len(agreeing)} of {len(windows)} measure {median:+d} samples "
            f"(+/-{AGREEMENT_TOLERANCE_SAMPLES}); measured offsets range {offsets[0]:+d}..{offsets[-1]:+d}. "
            "That points to a tone/phase difference, not a fixed delay; no correction recommended."),
            agreement=agreement)

    recommended = _round_half_toward_zero(float(np.median([w.offset_samples for w in agreeing])))
    if abs(recommended) <= ZERO_OFFSET_TOLERANCE_SAMPLES:
        return result(STATUS_ALIGNED, (
            f"{len(agreeing)} of {len(windows)} regions agree on {recommended:+d} samples: "
            "no fixed offset detected."), agreement=agreement)
    return result(STATUS_FIXED_OFFSET, (
        f"{len(agreeing)} of {len(windows)} independent regions agree that Amp B "
        f"{'lags' if recommended > 0 else 'leads'} Amp A by {abs(recommended)} samples "
        f"({abs(recommended) / sample_rate * 1000:.2f} ms)."),
        recommended=recommended, agreement=agreement)

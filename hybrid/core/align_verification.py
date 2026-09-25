"""Cross-DI verification of a fixed A/B timing offset -- measurement only.

`hybrid.core.align_diagnostic.analyse_alignment` checks that several regions
of ONE DI agree on one lag. Real captures showed that is not enough: two
differently-voiced amps can agree region-to-region within a clip while the
agreed lag moves with the material (Deluxe Reverb vs Twin Reverb measured -3,
-3, -6 and -7 samples on four DIs). That is phase response, not latency. A
genuine fixed latency gives the same lag on every piece of material.

`verify_fixed_offset_across_dis` renders both amps on a small fixed set of the
bundled DIs and runs the same multi-region diagnostic on each. It verifies an
offset only when every DI that has enough signal reports `fixed_offset`, at
least `MIN_AGREEING_DIS` of them do, and all of those offsets (plus the
preview DI's own) lie within `CROSS_DI_TOLERANCE_SAMPLES` of one consensus
integer. Anything else is `rejected`. The old whole-render estimate is never
used, not even as a fallback.

It only runs when the preview DI's diagnostic already reports `fixed_offset`:
otherwise no correction could be offered, so the extra renders are skipped.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .align_diagnostic import (
    STATUS_FIXED_OFFSET,
    STATUS_INSUFFICIENT_SIGNAL,
    AlignmentDiagnostic,
    _round_half_toward_zero,
    analyse_alignment,
)

VERIFICATION_METHOD = "multi-region-cross-di-v1"

# Fixed, versioned material from assets/di/. The first VERIFICATION_DI_COUNT
# that are not the preview DI are used, so the verification clips are always
# independent of the clip the preview diagnostic already measured.
VERIFICATION_DI_FILES = ("clean_mayer.wav", "moderate_brit.wav", "high_thrash.wav", "moderate_hotrod.wav")
VERIFICATION_DI_COUNT = 3
VERIFICATION_MAX_SECONDS = 20.0   # bound render cost; 20 s still yields the 12 regions
MIN_AGREEING_DIS = 2
CROSS_DI_TOLERANCE_SAMPLES = 1

STATUS_VERIFIED = "verified"
STATUS_REJECTED = "rejected"
STATUS_NOT_RUN = "not_run"

# (dry, sample_rate) -> (amp_a_render, amp_b_render, dry_both_amps_received)
PairRenderer = Callable[[np.ndarray, int], tuple]


@dataclass(frozen=True)
class DiVerification:
    di_file: str
    status: str
    recommended_offset_samples: int
    per_window_offsets: list[int]
    windows_reliable: int
    windows_analysed: int
    median_correlation: Optional[float]


@dataclass(frozen=True)
class CrossDiVerification:
    status: str
    offset_samples: Optional[int]   # the verified consensus, only when status == "verified"
    reason: str
    method: str = VERIFICATION_METHOD
    primary_status: Optional[str] = None
    primary_offset_samples: Optional[int] = None
    per_di: list[DiVerification] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    def to_dict(self) -> dict:
        return asdict(self)


def verification_di_files(preview_di_file: Optional[str]) -> list[str]:
    return [f for f in VERIFICATION_DI_FILES if f != preview_di_file][:VERIFICATION_DI_COUNT]


def _load_di(path: Path, sample_rate: int) -> Optional[np.ndarray]:
    dry, sr = sf.read(path, dtype="float32")
    if sr != sample_rate:
        return None  # no resampling here; that DI simply cannot be used
    if dry.ndim > 1:
        dry = dry[:, 0]
    return dry[: int(VERIFICATION_MAX_SECONDS * sample_rate)]


def verify_fixed_offset_across_dis(
    render_pair: PairRenderer,
    di_dir: Path,
    sample_rate: int,
    *,
    primary: Optional[AlignmentDiagnostic],
    preview_di_file: Optional[str] = None,
    measurement_cache: Optional[dict] = None,
    cache_key: Optional[tuple] = None,
) -> CrossDiVerification:
    """Render both amps on each verification DI (same input to both, via
    `render_pair`) and decide whether one fixed offset holds across them.

    `measurement_cache`/`cache_key` let a caller reuse per-DI measurements for
    the same source models and render settings (`cache_key` must identify
    both); the decision itself is always recomputed from them."""
    start = time.monotonic()
    primary_status = primary.status if primary else None
    primary_offset = primary.recommended_offset_samples if primary else None
    base = dict(primary_status=primary_status, primary_offset_samples=primary_offset)
    if primary is None or primary.status != STATUS_FIXED_OFFSET:
        return CrossDiVerification(
            STATUS_NOT_RUN, None,
            "The preview DI did not show a fixed offset, so there is nothing to verify.", **base)

    per_di: list[DiVerification] = []
    for di_file in verification_di_files(preview_di_file):
        key = (cache_key, VERIFICATION_METHOD, di_file, sample_rate) if cache_key is not None else None
        if measurement_cache is not None and key in measurement_cache:
            per_di.append(measurement_cache[key])
            continue
        dry = _load_di(Path(di_dir) / di_file, sample_rate)
        if dry is None:
            continue
        amp_a, amp_b, received = render_pair(dry, sample_rate)
        d = analyse_alignment(amp_a, amp_b, received, sample_rate, include_whole_render=False)
        measured = DiVerification(
            di_file=di_file, status=d.status, recommended_offset_samples=d.recommended_offset_samples,
            per_window_offsets=d.per_window_offsets, windows_reliable=d.windows_reliable,
            windows_analysed=d.windows_analysed, median_correlation=d.median_correlation)
        if measurement_cache is not None and key is not None:
            measurement_cache[key] = measured
        per_di.append(measured)
    elapsed = round(time.monotonic() - start, 3)

    def done(status: str, reason: str, offset: Optional[int] = None) -> CrossDiVerification:
        return CrossDiVerification(status, offset, reason, per_di=per_di, elapsed_s=elapsed, **base)

    counted = [v for v in per_di if v.status != STATUS_INSUFFICIENT_SIGNAL]
    others = [v for v in counted if v.status != STATUS_FIXED_OFFSET]
    if others:
        listed = ", ".join(f"{v.di_file}: {v.status}" for v in others)
        return done(STATUS_REJECTED, f"Not every DI shows a fixed offset ({listed}); no correction recommended.")
    if len(counted) < MIN_AGREEING_DIS:
        return done(STATUS_REJECTED,
                    f"Only {len(counted)} verification DI(s) had enough signal (need {MIN_AGREEING_DIS}); "
                    "no correction recommended.")

    # The consensus is the complete stable A/B relationship, including any
    # natural +/-1 the diagnostic alone would call "aligned" (e.g. natural +1
    # plus a genuine +7 latency verifies as +8, and +8 is what is corrected).
    offsets = [primary.recommended_offset_samples] + [v.recommended_offset_samples for v in counted]
    consensus = _round_half_toward_zero(float(np.median(offsets)))
    if any(abs(o - consensus) > CROSS_DI_TOLERANCE_SAMPLES for o in offsets):
        shown = ", ".join(f"{o:+d}" for o in offsets)
        return done(STATUS_REJECTED,
                    f"The apparent offset changes with the material ({shown} samples across DIs). A fixed "
                    "latency cannot do that; this is the amps' differing phase response, so no correction "
                    "is recommended.")
    if consensus == 0:
        return done(STATUS_REJECTED, "The DIs agree on no offset; no correction needed.")
    return done(STATUS_VERIFIED,
                f"{len(counted) + 1} independent DIs agree that Amp B "
                f"{'lags' if consensus > 0 else 'leads'} Amp A by {abs(consensus)} samples.",
                offset=consensus)

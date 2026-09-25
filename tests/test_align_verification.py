"""Cross-DI verification: one offset must hold across independent material."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import hybrid.core.align_verification as av
from hybrid.core.align_diagnostic import analyse_alignment
from hybrid.core.align_verification import verification_di_files, verify_fixed_offset_across_dis

DI_DIR = Path(__file__).resolve().parents[1] / "assets" / "di"
PREVIEW_DI = "moderate_brit.wav"
SR = 48000


def _delay(x, n):
    x = np.asarray(x, dtype=np.float32)
    if n >= 0:
        return np.concatenate([np.zeros(n, dtype=np.float32), x])[: len(x)]
    return np.concatenate([x[-n:], np.zeros(-n, dtype=np.float32)])


def _preview_dry():
    dry, _ = sf.read(DI_DIR / PREVIEW_DI, dtype="float32")
    return dry[: 20 * SR]


def _primary(delay):
    dry = _preview_dry()
    return analyse_alignment(np.tanh(2 * dry), _delay(np.tanh(2 * dry), delay), dry, SR)


class Renderer:
    """Amp B = Amp A delayed; the delay can differ per call (i.e. per DI)."""

    def __init__(self, delays):
        self.delays = list(delays)
        self.calls = 0

    def __call__(self, dry, sample_rate):
        delay = self.delays[min(self.calls, len(self.delays) - 1)]
        self.calls += 1
        a = np.tanh(2 * dry).astype(np.float32)
        b = a if delay is None else _delay(a, delay)
        if delay is None:  # a render that no longer resembles Amp A
            b = (np.random.default_rng(self.calls).standard_normal(len(a)) * np.abs(a)).astype(np.float32)
        return a, b, dry


def _verify(renderer, primary, **kwargs):
    return verify_fixed_offset_across_dis(renderer, DI_DIR, SR, primary=primary, preview_di_file=PREVIEW_DI, **kwargs)


def test_verification_uses_fixed_independent_dis():
    files = verification_di_files(PREVIEW_DI)
    assert len(files) == av.VERIFICATION_DI_COUNT == 3
    assert PREVIEW_DI not in files
    assert verification_di_files(PREVIEW_DI) == files  # deterministic


def test_same_offset_across_dis_is_verified():
    primary = _primary(7)
    assert primary.status == "fixed_offset"
    result = _verify(Renderer([7]), primary)
    assert result.status == "verified" and result.verified
    assert result.offset_samples == 7
    assert [d.status for d in result.per_di] == ["fixed_offset"] * 3
    assert "lags" in result.reason


def test_negative_offset_across_dis_is_verified():
    result = _verify(Renderer([-12]), _primary(-12))
    assert (result.status, result.offset_samples) == ("verified", -12)
    assert "leads" in result.reason


def test_offsets_within_one_sample_agree():
    result = _verify(Renderer([7, 8, 7]), _primary(7))
    assert (result.status, result.offset_samples) == ("verified", 7)


def test_offset_that_moves_with_the_material_is_rejected():
    """The Deluxe-vs-Twin failure mode: each DI alone looks fixed."""
    result = _verify(Renderer([3, 6, 9]), _primary(3))
    assert all(d.status == "fixed_offset" for d in result.per_di)
    assert result.status == "rejected"
    assert result.offset_samples is None
    assert "changes with the material" in result.reason


def test_preview_offset_must_also_agree():
    result = _verify(Renderer([12]), _primary(7))
    assert result.status == "rejected"


def test_an_ambiguous_di_blocks_verification():
    result = _verify(Renderer([7, None, 7]), _primary(7))
    assert [d.status for d in result.per_di].count("ambiguous") == 1
    assert result.status == "rejected"
    assert result.offset_samples is None


def test_not_run_unless_the_preview_di_shows_a_fixed_offset():
    renderer = Renderer([7])
    for delay in (0, None):
        dry = _preview_dry()
        a = np.tanh(2 * dry)
        b = _delay(a, 0) if delay == 0 else np.random.default_rng(0).standard_normal(len(a)) * np.abs(a)
        primary = analyse_alignment(a, b, dry, SR)
        assert primary.status in ("aligned", "ambiguous")
        result = _verify(renderer, primary)
        assert result.status == "not_run" and result.offset_samples is None
    assert _verify(renderer, None).status == "not_run"
    assert renderer.calls == 0  # no extra renders


def test_too_few_dis_with_signal_is_rejected(tmp_path):
    for name in av.VERIFICATION_DI_FILES:
        sf.write(tmp_path / name, np.zeros(SR * 2, dtype=np.float32), SR)
    result = verify_fixed_offset_across_dis(Renderer([7]), tmp_path, SR, primary=_primary(7), preview_di_file=PREVIEW_DI)
    assert all(d.status == "insufficient_signal" for d in result.per_di)
    assert result.status == "rejected"


def test_measurements_are_cached_per_source_identity():
    cache: dict = {}
    renderer = Renderer([7])
    first = _verify(renderer, _primary(7), measurement_cache=cache, cache_key=("a", "b"))
    second = _verify(renderer, _primary(7), measurement_cache=cache, cache_key=("a", "b"))
    assert renderer.calls == 3
    assert first.per_di == second.per_di and second.status == "verified"
    _verify(renderer, _primary(7), measurement_cache=cache, cache_key=("a", "other-b"))
    assert renderer.calls == 6


def test_whole_render_estimate_is_never_consulted(monkeypatch):
    seen_lengths = []
    real = av.analyse_alignment

    def spy(*args, **kwargs):
        assert kwargs.get("include_whole_render") is False
        seen_lengths.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(av, "analyse_alignment", spy)
    result = _verify(Renderer([7]), _primary(7))
    assert result.status == "verified" and len(seen_lengths) == 3
    assert "whole_render_offset_samples" not in result.to_dict()["per_di"][0]

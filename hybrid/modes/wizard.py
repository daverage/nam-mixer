"""Plain-language observations for the guided Tone Wizard.

This deliberately reports what was measured from the current preview render;
it does not try to identify an amplifier or promise how a different guitar
will behave.  The wizard uses it to explain its defaults in musician-facing
language while the existing render/level-match pipeline remains authoritative.
"""
from __future__ import annotations

from .character_analysis import AmpCharacterAnalysis


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _description(analysis: AmpCharacterAnalysis) -> dict[str, float | str]:
    levels = analysis.levels
    gains = [level.compression_gain_db for level in levels]
    # A falling gain from soft to loud input is a useful, conservative signal
    # of compression. It is a description of this render, not a circuit model.
    compression_change = gains[-1] - gains[0] if len(gains) >= 2 else 0.0
    output_rms = _mean([level.output_rms_dbfs for level in levels])
    high_band = _mean([
        value
        for level in levels
        for frequency, value in zip(analysis.frequencies_hz, level.spectrum_db)
        if frequency >= 2000.0
    ])
    return {
        "output_rms_dbfs": output_rms,
        "compression_change_db": compression_change,
        "high_band_db": high_band,
        "compression_label": "more compressed" if compression_change < -1.5 else "more open",
    }


def summarise_amp_pair(analysis_a: AmpCharacterAnalysis, analysis_b: AmpCharacterAnalysis) -> dict:
    """Return bounded, explainable observations for two same-DI renders."""
    a, b = _description(analysis_a), _description(analysis_b)
    level_difference = float(a["output_rms_dbfs"]) - float(b["output_rms_dbfs"])
    brightness_difference = float(a["high_band_db"]) - float(b["high_band_db"])
    compression_difference = float(a["compression_change_db"]) - float(b["compression_change_db"])

    if abs(level_difference) < 1.0:
        level_text = "The two renders are already close in average level."
    else:
        louder = "Amp A" if level_difference > 0 else "Amp B"
        level_text = f"{louder} is about {abs(level_difference):.1f} dB louder in this preview; auto level match will compensate near the hand-off."

    if abs(brightness_difference) < 1.5:
        tone_text = "Their broad high-frequency balance is similar in this preview."
    else:
        brighter = "Amp A" if brightness_difference > 0 else "Amp B"
        tone_text = f"{brighter} has the brighter broad high-frequency balance in this preview."

    if abs(compression_difference) < 1.0:
        feel_text = "They react similarly to changing input level in this preview."
    else:
        more_compressed = "Amp A" if compression_difference < 0 else "Amp B"
        feel_text = f"{more_compressed} shows more compression as the input gets louder."

    return {
        "amp_a": a,
        "amp_b": b,
        "level_text": level_text,
        "tone_text": tone_text,
        "feel_text": feel_text,
    }

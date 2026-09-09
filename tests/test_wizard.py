from hybrid.character_analysis import AmpCharacterAnalysis, AmpLevelAnalysis
from hybrid.wizard import summarise_amp_pair


def _analysis(output_rms: float, compression: tuple[float, float], high: float) -> AmpCharacterAnalysis:
    return AmpCharacterAnalysis(
        sample_rate=48_000,
        frequencies_hz=(100.0, 3_000.0),
        levels=(
            AmpLevelAnalysis(-18.0, -18.0, output_rms, output_rms + 3, compression[0], (-30.0, high)),
            AmpLevelAnalysis(-6.0, -6.0, output_rms, output_rms + 3, compression[1], (-30.0, high)),
        ),
        config_hash="test",
    )


def test_wizard_summary_describes_measured_level_tone_and_compression():
    result = summarise_amp_pair(
        _analysis(-12.0, (8.0, 7.5), -20.0),
        _analysis(-16.0, (9.0, 5.0), -14.0),
    )
    assert "Amp A" in result["level_text"]
    assert "Amp B" in result["tone_text"]
    assert "Amp B" in result["feel_text"]

import math

import pytest

from hybrid.input_profiles import (
    BASS_PROFILES,
    GUITAR_PROFILES,
    db_to_amplitude,
    get_profile,
    resolve_profile_gain_db,
)


def test_guitar_profile_preset_values():
    expected = {
        "vintage_single": -7.0,
        "standard_single": -3.0,
        "vintage_humbucker": 0.0,
        "p90": 1.0,
        "modern_humbucker": 2.5,
        "hot_humbucker": 4.5,
        "extreme_passive": 6.0,
    }
    for profile_id, gain_db in expected.items():
        assert GUITAR_PROFILES[profile_id].gain_db == gain_db


def test_bass_profile_preset_values():
    expected = {
        "standard_jp": 0.0,
        "modern_passive_bass": 1.5,
        "hot_passive_bass": 3.5,
    }
    for profile_id, gain_db in expected.items():
        assert BASS_PROFILES[profile_id].gain_db == gain_db


def test_active_profiles_require_custom_gain_and_have_no_fake_universal_offset():
    for profile in (GUITAR_PROFILES["active_buffered"], BASS_PROFILES["active_preamped_bass"]):
        assert profile.requires_custom_gain is True
        assert profile.gain_db is None


def test_db_to_amplitude_round_trip():
    assert db_to_amplitude(0.0) == pytest.approx(1.0)
    assert 20.0 * math.log10(db_to_amplitude(6.0)) == pytest.approx(6.0)
    assert 20.0 * math.log10(125.0 / 250.0) == pytest.approx(-6.02, abs=0.01)
    assert 20.0 * math.log10(425.0 / 250.0) == pytest.approx(4.61, abs=0.01)


def test_resolve_profile_gain_db_fixed_profile():
    assert resolve_profile_gain_db("guitar", "hot_humbucker", custom_gain_db=None) == 4.5


def test_resolve_profile_gain_db_custom_required():
    with pytest.raises(ValueError):
        resolve_profile_gain_db("guitar", "active_buffered", custom_gain_db=None)
    assert resolve_profile_gain_db("guitar", "active_buffered", custom_gain_db=8.0) == 8.0


def test_resolve_profile_gain_db_custom_is_clamped_to_engineering_range():
    assert resolve_profile_gain_db("guitar", "active_buffered", custom_gain_db=99.0) == 12.0
    assert resolve_profile_gain_db("bass", "active_preamped_bass", custom_gain_db=-99.0) == -12.0


def test_get_profile_rejects_unknown_instrument_or_id():
    with pytest.raises(ValueError):
        get_profile("banjo", "vintage_humbucker")
    with pytest.raises(ValueError):
        get_profile("guitar", "not_a_real_profile")

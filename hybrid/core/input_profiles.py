"""Research-grounded relative-output profiles for common pickup types.

The goal here is NOT to claim we can measure a real guitar's actual output
level -- see docs/INPUT_PROFILE_RESEARCH.md for the full research writeup and
its caveats. It's to give the crossover-reachability question ("does a
realistic player actually drive this hybrid into Amp B?") a defensible,
documented set of relative-gain simulations to test against, instead of an
arbitrary test-only slider.

Guitar profiles are relative to a vintage/PAF-style passive humbucker (0 dB).
Bass profiles are a SEPARATE reference family relative to a standard passive
Jazz/Precision bass (0 dB) -- bass pickups are not meaningfully comparable to
guitar pickups on the same 0 dB point, so they must never be mixed into one
list or compared numerically against each other.

Active/buffered pickups deliberately have no fixed preset: manufacturer data
shows output varying by ~9-10 dB within a single product line depending on
measurement method, so a universal "active = +X dB" number would be a made-up
precision this project can't back up. Those profiles set `gain_db=None` and
`requires_custom_gain=True`; callers must supply an explicit gain (clamped to
the -12..+12 dB engineering range) when one of these is selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CUSTOM_GAIN_MIN_DB = -12.0
CUSTOM_GAIN_MAX_DB = 12.0


def db_to_amplitude(db: float) -> float:
    """Convert a dB gain to a linear amplitude multiplier."""
    return 10.0 ** (db / 20.0)


@dataclass(frozen=True)
class InputProfile:
    id: str
    instrument: str  # "guitar" or "bass"
    label: str
    gain_db: Optional[float]  # None when requires_custom_gain is True
    reference: str
    research_range_db: Optional[str]
    description: str
    confidence: str
    requires_custom_gain: bool = False
    source_notes: str = ""


_GUITAR_REFERENCE = "Vintage/PAF-style passive humbucker = 0 dB"

GUITAR_PROFILES: dict[str, InputProfile] = {
    "vintage_single": InputProfile(
        id="vintage_single", instrument="guitar",
        label="Vintage / low-output single coil",
        gain_db=-7.0, reference=_GUITAR_REFERENCE,
        research_range_db="~90-125 mV vs 250 mV ref (-8.9 to -6.0 dB)",
        description="Low-output vintage-style single coil (e.g. early Strat/Tele pickups).",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "standard_single": InputProfile(
        id="standard_single", instrument="guitar",
        label="Standard / hotter single coil",
        gain_db=-3.0, reference=_GUITAR_REFERENCE,
        research_range_db="~160-200 mV vs 250 mV ref (-3.9 to -1.9 dB)",
        description="Mid-output single coil, hotter-wound than vintage-spec.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "vintage_humbucker": InputProfile(
        id="vintage_humbucker", instrument="guitar",
        label="Vintage / PAF humbucker",
        gain_db=0.0, reference=_GUITAR_REFERENCE,
        research_range_db="~220-250 mV vs 250 mV ref (-1.1 to 0 dB)",
        description="The guitar reference point -- a vintage/PAF-style passive humbucker.",
        confidence="research-based category simulation",
        source_notes="Defined as the 0 dB reference point for this profile family.",
    ),
    "p90": InputProfile(
        id="p90", instrument="guitar",
        label="P90",
        gain_db=1.0, reference=_GUITAR_REFERENCE,
        research_range_db="~270-287 mV vs 250 mV ref (+0.7 to +1.2 dB)",
        description="P90-style single coil, slightly hotter than a PAF humbucker.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "modern_humbucker": InputProfile(
        id="modern_humbucker", instrument="guitar",
        label="Medium / modern humbucker",
        gain_db=2.5, reference=_GUITAR_REFERENCE,
        research_range_db="~300-375 mV vs 250 mV ref (+1.6 to +3.5 dB)",
        description="Modern medium-output passive humbucker.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "hot_humbucker": InputProfile(
        id="hot_humbucker", instrument="guitar",
        label="Hot humbucker",
        gain_db=4.5, reference=_GUITAR_REFERENCE,
        research_range_db="~400-435 mV vs 250 mV ref (+4.1 to +4.8 dB)",
        description="Hot-wound passive humbucker.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "extreme_passive": InputProfile(
        id="extreme_passive", instrument="guitar",
        label="Extreme passive",
        gain_db=6.0, reference=_GUITAR_REFERENCE,
        research_range_db="~510 mV vs 250 mV ref (+6.2 dB)",
        description="The hottest commonly-available passive pickups.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "active_buffered": InputProfile(
        id="active_buffered", instrument="guitar",
        label="Active / buffered",
        gain_db=None, reference=_GUITAR_REFERENCE, research_range_db=None,
        description="Active instruments vary too much for one trustworthy fixed output preset. Set the relative level manually.",
        confidence="no defensible universal preset -- custom gain required",
        requires_custom_gain=True,
        source_notes="Manufacturer active-pickup data shows ~9.5 dB variation within one product line alone.",
    ),
}

GUITAR_PROFILE_ORDER = [
    "vintage_single", "standard_single", "vintage_humbucker", "p90",
    "modern_humbucker", "hot_humbucker", "extreme_passive", "active_buffered",
]

_BASS_REFERENCE = "Standard passive Jazz/Precision bass = 0 dB"

BASS_PROFILES: dict[str, InputProfile] = {
    "standard_jp": InputProfile(
        id="standard_jp", instrument="bass",
        label="Standard Jazz / Precision bass",
        gain_db=0.0, reference=_BASS_REFERENCE,
        research_range_db="~150-163 mV",
        description="The bass reference point -- a standard passive Jazz/Precision-style bass.",
        confidence="research-based category simulation",
        source_notes="Defined as the 0 dB reference point for this profile family. NOT comparable to guitar profile dB values.",
    ),
    "modern_passive_bass": InputProfile(
        id="modern_passive_bass", instrument="bass",
        label="Modern / hotter passive bass",
        gain_db=1.5, reference=_BASS_REFERENCE,
        research_range_db="~170-200 mV",
        description="Moderately hotter passive bass pickup.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "hot_passive_bass": InputProfile(
        id="hot_passive_bass", instrument="bass",
        label="Very high-output passive bass",
        gain_db=3.5, reference=_BASS_REFERENCE,
        research_range_db="~230-250 mV",
        description="High-output passive bass pickup.",
        confidence="research-based category simulation",
        source_notes="Rounded from manufacturer relative-output data -- see docs/INPUT_PROFILE_RESEARCH.md",
    ),
    "active_preamped_bass": InputProfile(
        id="active_preamped_bass", instrument="bass",
        label="Active / preamped bass",
        gain_db=None, reference=_BASS_REFERENCE, research_range_db=None,
        description="Active/preamped basses vary heavily by pickup, preamp, EQ, and supply voltage -- some active pickup elements measure BELOW stock passive output before the onboard preamp. Set the relative level manually.",
        confidence="no defensible universal preset -- custom gain required",
        requires_custom_gain=True,
        source_notes="Manufacturer data shows some active bass pickup elements below stock passive output pre-preamp.",
    ),
}

BASS_PROFILE_ORDER = [
    "standard_jp", "modern_passive_bass", "hot_passive_bass", "active_preamped_bass",
]

PROFILES_BY_INSTRUMENT = {"guitar": GUITAR_PROFILES, "bass": BASS_PROFILES}
PROFILE_ORDER_BY_INSTRUMENT = {"guitar": GUITAR_PROFILE_ORDER, "bass": BASS_PROFILE_ORDER}


def get_profile(instrument_type: str, profile_id: str) -> InputProfile:
    profiles = PROFILES_BY_INSTRUMENT.get(instrument_type)
    if profiles is None:
        raise ValueError(f"unknown instrument_type: {instrument_type!r} (expected 'guitar' or 'bass')")
    profile = profiles.get(profile_id)
    if profile is None:
        raise ValueError(f"unknown input_profile_id {profile_id!r} for instrument {instrument_type!r}")
    return profile


def resolve_profile_gain_db(instrument_type: str, profile_id: str, custom_gain_db: Optional[float]) -> float:
    """Return the gain to actually apply for `profile_id`, validating the
    custom-gain requirement and clamping any custom value to the documented
    engineering range."""
    profile = get_profile(instrument_type, profile_id)
    if profile.requires_custom_gain:
        if custom_gain_db is None:
            raise ValueError(
                f"input profile {profile_id!r} requires a custom relative gain (none was provided)"
            )
        return max(CUSTOM_GAIN_MIN_DB, min(CUSTOM_GAIN_MAX_DB, float(custom_gain_db)))
    return profile.gain_db

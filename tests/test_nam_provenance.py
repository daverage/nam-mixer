"""Tests for hybrid/nam_provenance.py -- shared source-model provenance and
export-name suffix logic used by all three design modes' manifest builders
and by hybrid/a2_training_settings.py's user_metadata_kwargs."""
from types import SimpleNamespace

import pytest

from hybrid import nam_provenance as prov


def _model(**kwargs):
    defaults = dict(name=None, modeled_by=None, gear_type=None, gear_make=None,
                     gear_model=None, tone_type=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_source_metadata_fields_extracts_expected_keys():
    model = _model(name="Amp X", modeled_by="Someone", gear_type="amp",
                    gear_make="Fender", gear_model="Deluxe", tone_type="clean")
    fields = prov.source_metadata_fields(model)
    assert fields == {
        "name": "Amp X", "modeled_by": "Someone", "gear_type": "amp",
        "gear_make": "Fender", "gear_model": "Deluxe", "tone_type": "clean",
    }


def test_agreed_tone_type_requires_exact_match_on_official_value():
    a = _model(tone_type="hi_gain")
    b = _model(tone_type="hi_gain")
    assert prov.agreed_tone_type(a, b) == "hi_gain"

    b2 = _model(tone_type="clean")
    assert prov.agreed_tone_type(a, b2) is None

    assert prov.agreed_tone_type(_model(tone_type=None), _model(tone_type=None)) is None


def test_agreed_tone_type_rejects_unofficial_value():
    a = _model(tone_type="djent")
    b = _model(tone_type="djent")
    assert prov.agreed_tone_type(a, b) is None


@pytest.mark.parametrize("export_mode,expected", [
    ("none", prov.SUFFIX_AMP_ONLY),
    (None, prov.SUFFIX_AMP_ONLY),
    ("learned", prov.SUFFIX_LEARNED_CAB),
    ("embedded", ""),
])
def test_export_name_suffix(export_mode, expected):
    cab = {"export_mode": export_mode} if export_mode is not None else None
    assert prov.export_name_suffix(cab) == expected


def test_build_export_name_no_cab():
    assert prov.build_export_name("British American High Gain", None) == "British American High Gain [Amp Only]"


def test_build_export_name_learned_cab_uses_display_name():
    cab = {"export_mode": "learned", "display_name": "Modern Boutique 4x12"}
    assert prov.build_export_name("British American High Gain", cab) == \
        "British American High Gain + Modern Boutique 4x12 [Learned Cab]"


def test_build_export_name_learned_cab_falls_back_to_filename():
    cab = {"export_mode": "learned", "original_filename": "cab_ir_01.wav"}
    assert prov.build_export_name("Base", cab) == "Base + cab_ir_01.wav [Learned Cab]"


def test_build_export_name_embedded_leaves_base_unsuffixed():
    cab = {"export_mode": "embedded", "display_name": "Modern Boutique 4x12"}
    assert prov.build_export_name("British American High Gain", cab) == "British American High Gain"


def test_cabinet_display_name_prefers_display_name_over_filename():
    cab = {"display_name": "Modern Boutique 4x12", "original_filename": "cab_ir_01.wav"}
    assert prov.cabinet_display_name(cab) == "Modern Boutique 4x12"
    assert prov.cabinet_display_name({"original_filename": "cab_ir_01.wav"}) == "cab_ir_01.wav"
    assert prov.cabinet_display_name(None) == "Cabinet"

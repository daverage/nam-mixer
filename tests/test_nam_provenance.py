"""Tests for hybrid/training/nam_provenance.py -- shared source-model provenance and
export-name suffix logic used by all three design modes' manifest builders
and by hybrid/training/a2_training_settings.py's user_metadata_kwargs."""
from types import SimpleNamespace

import pytest

from hybrid.training import nam_provenance as prov


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


@pytest.mark.parametrize("cab, expected", [
    (None, "none"),
    ({}, "none"),
    ({"export_mode": "none", "selected": True}, "none"),          # preview-only cab: external IR needed
    ({"export_mode": "learned", "baked": True}, "learned"),
    ({"baked": True}, "learned"),                                 # historical record: cab was baked in
    ({"baked": False}, "none"),
    ({"export_mode": "embedded"}, "embedded"),
])
def test_cab_export_mode_includes_historical_baked_records(cab, expected):
    assert prov.cab_export_mode(cab) == expected


def test_cabinet_display_name_prefers_display_name_over_filename():
    assert prov.cabinet_display_name({"display_name": "Modern Boutique 4x12", "original_filename": "cab_ir_01.wav"}) == "Modern Boutique 4x12"
    assert prov.cabinet_display_name({"original_filename": "cab_ir_01.wav"}) == "cab_ir_01.wav"
    assert prov.cabinet_display_name(None) == "Cabinet"


def test_app_export_name_is_the_shared_rule_for_every_export_mode():
    """The app (a2_training_settings.user_metadata_kwargs) must use exactly
    export_model_name -- no second, test-only naming routine."""
    from hybrid.training.a2_training_settings import user_metadata_kwargs
    from tests.test_a2_training_settings import EXPORT_NAME_CASES

    for _label, manifest, expected in EXPORT_NAME_CASES:
        assert prov.export_model_name(manifest) == expected == user_metadata_kwargs(manifest)["name"]


def test_embedded_package_name():
    assert prov.embedded_package_name("Studio", "Boutique 4x12") == "Studio + Boutique 4x12 [Embedded Cab · Full]"
    assert prov.embedded_package_name(None, "  ") == "NAM Head + Cabinet [Embedded Cab · Full]"

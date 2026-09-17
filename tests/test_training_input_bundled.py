"""Tests for the bundled official NAM v3.0.0 training input
(assets/training/official_nam_v3_input.wav) and app.py's one-time local
seeding of it into a fresh work directory -- see assets/training/README.md
and the comment next to app.py's TRAINING_INPUT_PATH.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_ASSET = REPO_ROOT / "assets" / "training" / "official_nam_v3_input.wav"


def test_bundled_asset_matches_official_v3_md5():
    from hybrid.training_target import OFFICIAL_V3_INPUT_MD5

    assert BUNDLED_ASSET.is_file()
    digest = hashlib.md5(BUNDLED_ASSET.read_bytes()).hexdigest()
    assert digest == OFFICIAL_V3_INPUT_MD5


def _load_app_module_with_fresh_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("NAM_MIXER_DATA_DIR", str(tmp_path / "data"))
    spec = importlib.util.spec_from_file_location("app_fresh_data_dir", REPO_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fresh_work_dir_is_seeded_from_bundled_asset_without_any_upload(monkeypatch, tmp_path):
    app_fresh = _load_app_module_with_fresh_data_dir(monkeypatch, tmp_path)
    try:
        assert app_fresh.TRAINING_INPUT_PATH.is_file()
        digest = hashlib.md5(app_fresh.TRAINING_INPUT_PATH.read_bytes()).hexdigest()
        assert digest == hashlib.md5(BUNDLED_ASSET.read_bytes()).hexdigest()

        with app_fresh.app.test_client() as client:
            resp = client.get("/api/training_input/status")
            assert resp.status_code == 200
            assert resp.get_json()["ready"] is True
    finally:
        sys.modules.pop("app_fresh_data_dir", None)


def test_manual_upload_still_replaces_the_seeded_default(monkeypatch, tmp_path):
    """The bundled default is only a fallback, never a lock: a real upload
    (rejected here since it's not valid audio) still attempts to replace it,
    proving nothing special-cases the seeded file to protect it."""
    app_fresh = _load_app_module_with_fresh_data_dir(monkeypatch, tmp_path)
    try:
        with app_fresh.app.test_client() as client:
            import io
            resp = client.post(
                "/api/training_input/upload",
                data={"file": (io.BytesIO(b"not a real official input"), "input.wav")},
            )
            assert resp.status_code == 400
            assert not app_fresh.TRAINING_INPUT_PATH.is_file()  # invalid upload unlinks it
    finally:
        sys.modules.pop("app_fresh_data_dir", None)

import numpy as np

from hybrid.audio_metrics import rms_dbfs


def test_rms_dbfs_matches_existing_blend_semantics():
    assert np.isneginf(rms_dbfs(np.array([])))
    assert np.isclose(rms_dbfs(np.array([0.1, -0.1])), -20.0)
    assert rms_dbfs(np.array([0.0]), floor_dbfs=-90.0) == -90.0

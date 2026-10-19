import numpy as np
import pytest

from waterdet import geo, indices
from waterdet.config import load_config, require_gee_project, validate_config

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_ndwi_mndwi_values():
    g, nir, swir = np.array([0.2, 0.1]), np.array([0.1, 0.3]), np.array([0.05, 0.2])
    np.testing.assert_allclose(indices.ndwi(g, nir), [(0.2 - 0.1) / 0.3, (0.1 - 0.3) / 0.4], rtol=1e-5)
    np.testing.assert_allclose(indices.mndwi(g, swir), [(0.2 - 0.05) / 0.25, (0.1 - 0.2) / 0.3], rtol=1e-5)


def test_normalized_difference_zero_denominator_and_nan():
    out = indices.normalized_difference(np.array([0.0, np.nan, 0.1]), np.array([0.0, 0.1, 0.1]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.0


def test_feature_stack_order_and_missing_band():
    b = {k: np.full((2, 3), v, np.float32) for k, v in dict(B2=.05, B3=.1, B4=.07, B8=.3, B11=.2).items()}
    f = indices.feature_stack(b)
    assert f.shape == (7, 2, 3) and indices.FEATURE_NAMES[5] == "NDWI"
    np.testing.assert_allclose(f[5], (0.1 - 0.3) / 0.4, rtol=1e-5)
    np.testing.assert_allclose(f[6], (0.1 - 0.2) / 0.3, rtol=1e-5)
    b.pop("B11")
    with pytest.raises(KeyError):
        indices.feature_stack(b)


def test_utm_and_bbox():
    assert geo.utm_epsg(3.4, 6.5) == 32631
    assert geo.utm_epsg(3.4, -6.5) == 32731
    assert geo.utm_epsg(-0.1, 51.5) == 32630
    with pytest.raises(ValueError):
        geo.validate_bbox([3.5, 6.4, 3.3, 6.6])
    with pytest.raises(ValueError):
        geo.validate_bbox([3.3, 6.4, 3.5])
    n = geo.estimate_pixels([3.30, 6.40, 3.55, 6.60], 10)
    assert 5_500_000 < n < 6_800_000


def test_shipped_config_loads_and_placeholder_project_is_rejected():
    cfg = load_config(ROOT / "config.yaml")
    with pytest.raises(ValueError, match="placeholder"):
        require_gee_project(cfg)
    cfg["gee"]["project"] = "my-project-123"
    assert require_gee_project(cfg) == "my-project-123"


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c["periods"]["dry"].update(end="2025-01-01"), "after start"),
    (lambda c: c.update(training_period="nope"), "training_period"),
    (lambda c: c["labels"].update(primary_mode="gsw_dw", modes=["gsw"]), "primary_mode"),
    (lambda c: c["imagery"].update(cloud_score_threshold=1.5), "cloud_score_threshold"),
    (lambda c: c["model"].update(max_depth=[0]), "max_depth"),
    (lambda c: c["aoi"].update(bbox=[1, 2, 3]), "bbox"),
])
def test_config_validation_errors(mutate, msg):
    cfg = load_config(ROOT / "config.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=msg):
        validate_config(cfg)

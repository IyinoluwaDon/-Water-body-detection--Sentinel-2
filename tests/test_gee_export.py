"""Earth Engine calls cannot run offline. This only checks our own logic around them with a mock `ee`."""

from unittest.mock import MagicMock

import pytest

from waterdet import gee_export
from waterdet.io import EXPORT_BAND_ORDER


@pytest.fixture()
def fake_ee(monkeypatch):
    ee = MagicMock(name="ee")
    monkeypatch.setattr(gee_export, "_ee", lambda: ee)
    return ee


def test_start_exports_parameters(fake_ee, small_cfg):
    tasks = gee_export.start_exports(small_cfg)
    assert len(tasks) == 2
    calls = fake_ee.batch.Export.image.toDrive.call_args_list
    assert len(calls) == 2
    kw = calls[0].kwargs
    assert kw["scale"] == 10 and kw["crs"] == "EPSG:32631" and kw["fileFormat"] == "GeoTIFF"
    assert kw["folder"] == "waterdet" and kw["fileNamePrefix"] == "s2_dry" and kw["maxPixels"] >= 1e9
    assert calls[1].kwargs["fileNamePrefix"] == "s2_wet"
    assert all(t.start.called for t in tasks)
    # planar (non-geodesic) rectangle, as intended
    assert fake_ee.Geometry.Rectangle.call_args.kwargs["geodesic"] is False


def test_oversized_aoi_is_refused(fake_ee, small_cfg):
    small_cfg["aoi"]["bbox"] = [3.0, 6.0, 5.0, 8.0]
    with pytest.raises(ValueError, match="Shrink"):
        gee_export.start_exports(small_cfg)
    assert not fake_ee.batch.Export.image.toDrive.called


def test_export_band_order_is_documented_order(fake_ee, small_cfg):
    gee_export.export_image(object(), "2025-12-01", "2026-03-01", small_cfg)
    fake_ee.Image.cat.return_value.toFloat.return_value.select.assert_called_with(EXPORT_BAND_ORDER)

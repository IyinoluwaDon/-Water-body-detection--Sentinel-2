"""Build the REAL Earth Engine computation graph offline and check it is well-formed.

The earthengine-api package ships the list of valid server functions. Initialising the client with that list
(no sign-in, no network) means a wrong function name, a wrong argument name or a wrong band name in our export
code fails here instead of during your first run. It does not run the computation on Google's servers.
"""

import json

import pytest

ee = pytest.importorskip("ee")
apitestcase = pytest.importorskip("ee.apitestcase")

from waterdet import gee_export as G  # noqa: E402
from waterdet.config import load_config  # noqa: E402
from waterdet.io import EXPORT_BAND_ORDER  # noqa: E402


class _Offline(apitestcase.ApiTestCase):
    def runTest(self):  # pragma: no cover - required by unittest.TestCase
        pass


@pytest.fixture()
def offline_ee():
    case = _Offline()
    case.setUp()
    yield
    case.tearDown()


@pytest.fixture()
def cfg():
    return load_config("config.yaml")


def test_export_image_graph_builds_and_serialises(offline_ee, cfg):
    aoi = G.aoi_geometry(cfg["aoi"]["bbox"])
    p = cfg["periods"]["dry"]
    img = G.export_image(aoi, p["start"], p["end"], cfg)
    graph = json.loads(img.serialize())
    text = json.dumps(graph)
    for needed in ("COPERNICUS/S2_SR_HARMONIZED", "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED",
                   "JRC/GSW1_4/GlobalSurfaceWater", "GOOGLE/DYNAMICWORLD/V1", "cs_cdf"):
        assert needed in text, f"{needed} missing from the Earth Engine graph"
    for band in EXPORT_BAND_ORDER:
        assert band in text, f"band {band} missing from the Earth Engine graph"


def test_export_task_parameters(offline_ee, cfg):
    aoi = G.aoi_geometry(cfg["aoi"]["bbox"])
    p = cfg["periods"]["dry"]
    task = ee.batch.Export.image.toDrive(
        image=G.export_image(aoi, p["start"], p["end"], cfg), description="waterdet_test", folder="waterdet",
        fileNamePrefix="s2_dry", region=aoi, scale=10, crs="EPSG:32631", maxPixels=int(1e10), fileFormat="GeoTIFF")
    opts = task.config["fileExportOptions"]
    assert opts["fileFormat"] == "GEO_TIFF"
    assert opts["driveDestination"] == {"folder": "waterdet", "filenamePrefix": "s2_dry"}


def test_dynamic_world_band_survives_an_empty_collection(offline_ee, cfg):
    """Regression: an empty Dynamic World window once dropped the dw_water band and failed the export."""
    aoi = G.aoi_geometry(cfg["aoi"]["bbox"])
    text = G.label_bands(aoi, "2026-06-01", "2026-10-01").serialize()
    assert "\"functionName\": \"If\"" in text and "Collection.size" in text and "dw_water" in text

"""Google Earth Engine side: build a cloud-masked Sentinel-2 composite plus label bands and export to Drive.

Everything heavy (cloud masking, compositing) runs on Earth Engine's servers; everything modelling-related
runs locally on the exported GeoTIFF. Run this on Colab or any machine where you can sign in to Earth Engine.

CLI:
    python -m waterdet.gee_export preflight --config config.yaml   # counts scenes, prints AOI preview URL
    python -m waterdet.gee_export export    --config config.yaml   # starts one Drive export task per period
    python -m waterdet.gee_export status    --config config.yaml   # shows task states

Exported GeoTIFF band order (float32; masked pixels = -9999):
    B2, B3, B4, B8, B11     surface reflectance 0-1 (median of clear observations)
    n_obs                    number of clear observations behind each pixel
    gsw_occurrence           JRC GSW v1.4 occurrence % (0 where water was never seen)
    gsw_max_extent           JRC GSW v1.4: 1 if water was ever detected, else 0
    dw_water                 Dynamic World 'water' probability, median over the period
"""

from __future__ import annotations

import argparse

from .config import load_config, require_gee_project
from .geo import estimate_pixels, utm_epsg, validate_bbox
from .io import EXPORT_BAND_ORDER, NODATA, SPECTRAL_BANDS

S2_ID = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_ID = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
GSW_ID = "JRC/GSW1_4/GlobalSurfaceWater"
DW_ID = "GOOGLE/DYNAMICWORLD/V1"
MAX_PIXELS_WARN = 60_000_000  # ~6,000 km2 at 10 m. The pipeline holds 9 float32 bands in RAM (~2.2 GB at this size)


def _ee():
    try:
        import ee
    except ImportError as exc:  # pragma: no cover
        raise ImportError("earthengine-api is not installed. Run: pip install earthengine-api") from exc
    return ee


def init_ee(project: str, authenticate_if_needed: bool = True) -> None:
    """Initialise Earth Engine; opens the sign-in flow once if there is no cached credential."""
    ee = _ee()
    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001  (ee raises several different types when credentials are missing)
        if not authenticate_if_needed:
            raise
        ee.Authenticate()
        ee.Initialize(project=project)


def aoi_geometry(bbox):
    ee = _ee()
    lon_min, lat_min, lon_max, lat_max = validate_bbox(bbox)
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj="EPSG:4326", geodesic=False)


def clear_collection(aoi, start: str, end: str, *, cloud_score_threshold: float, max_scene_cloud_pct: float):
    """Sentinel-2 L2A scenes with Cloud Score+ masking applied; bands scaled to reflectance 0-1.

    `end` is exclusive (Earth Engine filterDate semantics). Cloud Score+ is used instead of the
    QA60 band because QA60 is a coarse polygon mask and was not populated 2022-01-25 to 2024-02-28.
    """
    ee = _ee()
    scenes = (
        ee.ImageCollection(S2_ID)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud_pct))
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_ID), ["cs_cdf"])
    )

    def _mask(img):
        clear = img.select("cs_cdf").gte(cloud_score_threshold)
        return img.select(SPECTRAL_BANDS).divide(10000).updateMask(clear)

    return scenes.map(_mask)


def label_bands(aoi, start: str, end: str):
    """GSW and Dynamic World bands used to build training labels (see waterdet.labels)."""
    ee = _ee()
    gsw = ee.Image(GSW_ID)
    occurrence = gsw.select("occurrence").unmask(0).rename("gsw_occurrence")
    max_extent = gsw.select("max_extent").unmask(0).rename("gsw_max_extent")
    dw_col = ee.ImageCollection(DW_ID).filterBounds(aoi).filterDate(start, end).select("water")
    # Dynamic World can have no images in a cloudy window. A median of nothing has no bands and would make the
    # whole export fail, so fall back to a fully masked band (it becomes NaN and is simply not used).
    empty = ee.Image.constant(0).rename("dw_water").updateMask(ee.Image.constant(0))
    dw = ee.Image(ee.Algorithms.If(dw_col.size().gt(0), dw_col.median().rename("dw_water"), empty))
    return ee.Image.cat([occurrence, max_extent, dw])


def export_image(aoi, start: str, end: str, cfg: dict):
    """The multi-band image that is exported for one period."""
    ee = _ee()
    im = cfg["imagery"]
    col = clear_collection(aoi, start, end, cloud_score_threshold=im["cloud_score_threshold"],
                           max_scene_cloud_pct=im["max_scene_cloud_pct"])
    composite = col.median().select(SPECTRAL_BANDS)
    n_obs = col.select("B3").count().rename("n_obs")
    img = ee.Image.cat([composite, n_obs, label_bands(aoi, start, end)]).toFloat()
    img = img.select(EXPORT_BAND_ORDER)  # guarantees the documented band order
    return img.unmask(NODATA)


def preflight(cfg: dict) -> dict:
    """Cheap server-side checks to run BEFORE exporting. Returns counts per period."""
    ee = _ee()
    aoi = aoi_geometry(cfg["aoi"]["bbox"])
    im = cfg["imagery"]
    out = {}
    for name, p in cfg["periods"].items():
        base = ee.ImageCollection(S2_ID).filterBounds(aoi).filterDate(p["start"], p["end"])
        n_all = base.size().getInfo()
        n_cloud_ok = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", im["max_scene_cloud_pct"])).size().getInfo()
        n_cs = ee.ImageCollection(CLOUD_SCORE_ID).filterBounds(aoi).filterDate(p["start"], p["end"]).size().getInfo()
        out[name] = {"s2_scenes": n_all, "after_scene_cloud_filter": n_cloud_ok, "cloud_score_plus_images": n_cs}
    return out


def preview_url(cfg: dict, period: str, dimensions: int = 900) -> str:
    """Thumbnail URL of the true-colour composite, to eyeball the AOI before exporting."""
    aoi = aoi_geometry(cfg["aoi"]["bbox"])
    p = cfg["periods"][period]
    im = cfg["imagery"]
    col = clear_collection(aoi, p["start"], p["end"], cloud_score_threshold=im["cloud_score_threshold"],
                           max_scene_cloud_pct=im["max_scene_cloud_pct"])
    return col.median().getThumbURL({"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.25, "region": aoi,
                                     "dimensions": dimensions, "format": "png"})


def start_exports(cfg: dict) -> list:
    """Start one Export.image.toDrive task per period. Returns the task objects."""
    ee = _ee()
    bbox = validate_bbox(cfg["aoi"]["bbox"])
    scale = cfg["imagery"]["scale_m"]
    n_px = estimate_pixels(bbox, scale)
    if n_px > MAX_PIXELS_WARN:
        raise ValueError(f"AOI at {scale} m is about {n_px:,} pixels (> {MAX_PIXELS_WARN:,}). Shrink the AOI or raise scale_m.")
    aoi = aoi_geometry(bbox)
    crs = f"EPSG:{utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)}"
    tasks = []
    for name, p in cfg["periods"].items():
        task = ee.batch.Export.image.toDrive(
            image=export_image(aoi, p["start"], p["end"], cfg),
            description=f"waterdet_{cfg['aoi'].get('name', 'aoi')}_{name}",
            folder=cfg["gee"]["drive_folder"],
            fileNamePrefix=f"s2_{name}",
            region=aoi,
            scale=scale,
            crs=crs,
            maxPixels=int(1e10),
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append(task)
    return tasks


def task_states() -> list[dict]:
    """State of the most recent waterdet export tasks."""
    ee = _ee()
    return [{"description": t["description"], "state": t["state"], "error": t.get("error_message")}
            for t in ee.data.getTaskList() if str(t.get("description", "")).startswith("waterdet_")][:10]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["preflight", "export", "status"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    init_ee(require_gee_project(cfg))
    if args.action == "preflight":
        for k, v in preflight(cfg).items():
            print(k, v)
    elif args.action == "export":
        for t in start_exports(cfg):
            print("started:", t.status().get("description"), t.id)
        print("Check progress in the Earth Engine Tasks tab or run the 'status' action.")
    else:
        for row in task_states():
            print(row)


if __name__ == "__main__":
    main()

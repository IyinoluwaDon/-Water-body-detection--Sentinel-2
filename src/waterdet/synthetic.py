"""Synthetic study area with KNOWN truth, used to test the whole pipeline without Earth Engine.

Features of the fake world (400 x 500 px at 10 m, UTM 31N):
  * a sinuous river (its eastern half is turbid: brighter NIR) and an elliptical lake  -> permanent water
  * a small pond that is water only in the wet period                                  -> seasonal water
  * a dark shadow patch on land: NDWI correctly says land, MNDWI wrongly says water   -> index trap
  * a 'reclaimed' patch: land today, but GSW (old) says water, Dynamic World says land -> label trap
  * GSW is made coarse (3x3 blocks) to mimic 30 m labels against 10 m imagery
  * a cloud hole in the wet period (no valid observations)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

H, W = 400, 500
BANDS = ["B2", "B3", "B4", "B8", "B11", "n_obs", "gsw_occurrence", "gsw_max_extent", "dw_water"]


def _coarse(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Majority-vote the mask on a k x k grid and expand back (mimics a 30 m product)."""
    h, w = mask.shape
    hh, ww = -(-h // k) * k, -(-w // k) * k
    pad = np.zeros((hh, ww), float)
    pad[:h, :w] = mask
    blocks = pad.reshape(hh // k, k, ww // k, k).mean(axis=(1, 3)) >= 0.5
    return np.kron(blocks, np.ones((k, k), bool))[:h, :w]


def make_world(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    river = np.abs(yy - (200 + 40 * np.sin(xx / 60.0))) < 10
    lake = ((yy - 100) / 50.0) ** 2 + ((xx - 350) / 70.0) ** 2 < 1
    # keep the lake away from the pond and reclaimed patch positions
    pond = (yy - 330) ** 2 + (xx - 120) ** 2 < 14**2
    reclaimed = (yy >= 300) & (yy < 370) & (xx >= 360) & (xx < 460)
    shadow = (yy >= 30) & (yy < 70) & (xx >= 30) & (xx < 100)
    permanent = river | lake
    turbid = river & (xx > 250)
    return dict(yy=yy, xx=xx, river=river, lake=lake, pond=pond, reclaimed=reclaimed, shadow=shadow,
                permanent=permanent, turbid=turbid, rng=rng)


def make_bands(world: dict, period: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + (0 if period == "dry" else 1))
    truth = world["permanent"] | (world["pond"] if period == "wet" else False)
    land_mean = dict(B2=0.05, B3=0.08, B4=0.07, B8=0.30, B11=0.22)
    water_mean = dict(B2=0.07, B3=0.08, B4=0.06, B8=0.03, B11=0.015)
    turbid_mean = dict(B2=0.09, B3=0.10, B4=0.09, B8=0.07, B11=0.04)
    shadow_mean = dict(B2=0.03, B3=0.03, B4=0.03, B8=0.04, B11=0.02)
    bands = {}
    for b in ["B2", "B3", "B4", "B8", "B11"]:
        arr = np.where(truth, water_mean[b], land_mean[b]).astype(np.float32)
        arr = np.where(world["turbid"], turbid_mean[b], arr)
        arr = np.where(world["shadow"], shadow_mean[b], arr)
        scale = 0.012 if b in ("B8", "B11") else 0.008
        arr = arr + rng.normal(0, scale * np.where(truth, 1.0, 2.5), arr.shape).astype(np.float32)
        bands[b] = arr.astype(np.float32)
    n_obs = np.full((H, W), 10.0 if period == "dry" else 6.0, np.float32)
    if period == "wet":
        n_obs[0:50, 450:500] = 0.0                      # cloud hole
        for b in ["B2", "B3", "B4", "B8", "B11"]:
            bands[b][0:50, 450:500] = np.nan
    bands["n_obs"] = n_obs
    # Labels: GSW is old (reclaimed patch still water) and coarse; seasonal pond has partial occurrence.
    gsw_water = _coarse(world["permanent"] | world["reclaimed"])
    pond_c = _coarse(world["pond"])
    occ = np.zeros((H, W), np.float32)
    occ[pond_c] = 40.0
    occ[gsw_water] = 95.0
    bands["gsw_occurrence"] = occ
    bands["gsw_max_extent"] = (occ > 0).astype(np.float32)
    dw = np.where(truth, 0.9, 0.05) + rng.normal(0, 0.03, (H, W))
    bands["dw_water"] = np.clip(dw, 0, 1).astype(np.float32)
    bands["_truth"] = truth
    return bands


def write_stack(path: Path, bands: dict) -> Path:
    """Write the exported-GeoTIFF layout (band descriptions, -9999 for masked values)."""
    data = np.stack([np.nan_to_num(bands[b], nan=-9999.0).astype(np.float32) for b in BANDS])
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=len(BANDS), dtype="float32",
                       crs="EPSG:32631", transform=from_origin(500000, 710000, 10, 10)) as dst:
        dst.write(data)
        dst.descriptions = tuple(BANDS)
    return path




def demo_config():
    """Small, deterministic configuration for the artificial study area."""
    return {
        "gee": {"project": "demo", "drive_folder": "waterdet"},
        "aoi": {"name": "synthetic", "bbox": [3.30, 6.40, 3.55, 6.60]},
        "imagery": {"scale_m": 10, "cloud_score_threshold": 0.6, "max_scene_cloud_pct": 60, "min_valid_obs": 3},
        "periods": {"dry": {"start": "2025-12-01", "end": "2026-03-01"},
                    "wet": {"start": "2026-06-01", "end": "2026-10-01"}},
        "training_period": "dry",
        "labels": {"min_gsw_occurrence": 75, "edge_buffer_px": 3, "dw_water_threshold": 0.5,
                   "modes": ["gsw", "gsw_dw"], "primary_mode": "gsw_dw"},
        "sampling": {"n_per_class": 2000, "block_size_m": 500, "test_size": 0.3},
        "model": {"max_depth": [1, 2, 3, 4, 6], "min_samples_leaf": [1, 20], "cv_folds": 3, "scoring": "f1"},
        "reference": {"csv": "unused", "shore_px": 2, "frame_n": {"water_core": 60, "land_core": 60, "shoreline": 40}},
        "seed": 42,
    }
